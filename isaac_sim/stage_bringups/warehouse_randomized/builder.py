from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import math
from pathlib import Path
import re
import traceback
from typing import Any

import numpy as np
import yaml

from isaac_sim.goal_generator.object_goal_sampler_utils import ObjectBBox3D, OccupancyMap
from isaac_sim.stage_bringups.runtime_utils import (
    add_reference,
    compute_world_bbox,
    define_xform,
    get_stage,
    get_world_pose_xyzw,
    maybe_start_sim_app,
    new_stage,
    quaternion_xyzw_to_yaw,
    set_stage_units,
    set_xform_pose,
)
from isaac_sim.stage_bringups.warehouse_randomized.maps import (
    MapExportResult,
    export_bbox_occupancy_map,
    export_occupancy_map,
    sample_multi_robot_rollouts,
)
from isaac_sim.stage_bringups.warehouse_randomized.robots import (
    RuntimeRobotController,
    build_robot_adapter,
)
from isaac_sim.stage_bringups.warehouse_randomized.ros_bridge import InternalIsaacRosBridge
from isaac_sim.stage_bringups.warehouse_randomized.templates import (
    KeepoutZone,
    ObjectGroupSpec,
    ObjectRandomizationSpec,
    PlacementZone,
    SelectorSpec,
    TemplateMapSpec,
    WarehouseTemplate,
)


@dataclass
class RandomizedWarehouseBuildResult:
    scene_id: str
    seed: int
    template_id: str
    variant_id: str
    bundle_dir: Path
    scene_usd_path: Path
    team_config_path: Path
    nav2_map_path: Path
    mapf_map_path: Path
    manifest_path: Path
    robot_prim_paths: list[str]
    rollouts_dir: Path
    ros_bridge: InternalIsaacRosBridge | None
    robot_controllers: list[RuntimeRobotController]
    validation_summary: dict[str, Any]


class RandomizedWarehouseBuilder:
    def __init__(
        self,
        *,
        sim_app=None,
        repo_root: str | Path,
        scene_root_dir: str | Path,
        scene_id: str,
        template: WarehouseTemplate,
        seed: int,
        robot_models: list[str],
        robot_count: int,
        rollout_count: int,
        language_instruction: str,
        enable_ros2_runtime: bool,
        rollout_control_topic: str,
        rollout_reset_done_topic: str,
        map_export_mode: str = "bbox",
        overwrite: bool = False,
    ) -> None:
        self._repo_root = Path(repo_root).expanduser().resolve()
        self._scene_root_dir = Path(scene_root_dir).expanduser().resolve()
        self._scene_id = str(scene_id).strip()
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._robot_count = int(robot_count)
        self._rollout_count = int(rollout_count)
        self._language_instruction = str(language_instruction).strip()
        self._enable_ros2_runtime = bool(enable_ros2_runtime)
        self._rollout_control_topic = str(rollout_control_topic).strip()
        self._rollout_reset_done_topic = str(rollout_reset_done_topic).strip()
        self._map_export_mode = str(map_export_mode).strip().lower() or "bbox"
        self._overwrite = bool(overwrite)

        if self._robot_count < 2 or self._robot_count > 5:
            raise ValueError(f"robot_count must be between 2 and 5, got {self._robot_count}.")
        if self._rollout_count <= 0:
            raise ValueError("rollout_count must be positive.")
        if self._map_export_mode not in {"bbox", "omap"}:
            raise ValueError("map_export_mode must be either 'bbox' or 'omap'.")

        if not isinstance(template, WarehouseTemplate):
            raise TypeError("template must be a resolved WarehouseTemplate instance.")
        self._template = template

        expanded_models = [str(model).strip() for model in robot_models if str(model).strip()]
        if not expanded_models:
            expanded_models = ["nova_carter"]
        if len(expanded_models) == 1:
            expanded_models = expanded_models * self._robot_count
        if len(expanded_models) != self._robot_count:
            raise ValueError(
                "robot_models must contain either one model or exactly robot_count models."
            )

        self._robot_names = [f"robot{index}" for index in range(1, self._robot_count + 1)]
        self._robot_adapters = [build_robot_adapter(model_id) for model_id in expanded_models]

        self._sim_app = sim_app
        self._bundle_dir = self._scene_root_dir / self._scene_id
        self._scene_usd_path = self._bundle_dir / "scene.usd"
        self._manifest_path = self._bundle_dir / "scene_manifest.yaml"
        self._team_config_path = self._bundle_dir / "team_config.yaml"
        self._failure_snapshot_path = self._bundle_dir / "build_failure_snapshot.yaml"
        self._rollouts_dir = self._bundle_dir / "rollouts"
        self._nav2_map_path = self._bundle_dir / "nav2_map.yaml"
        self._mapf_map_path = self._bundle_dir / "mapf_map.yaml"

        self._env_prim_path = "/World/Env/Warehouse"
        self._randomized_props_root = "/World/Env/RandomizedProps"
        self._robots_root = "/World/Robots"

        self._assets_root_path = ""
        self._resolved_base_environment_usd = ""
        self._randomization_records: list[dict[str, Any]] = []
        self._resolved_group_paths: dict[str, list[str]] = {}
        self._resolved_group_details: dict[str, dict[str, Any]] = {}
        self._resolved_light_paths: dict[str, list[str]] = {}
        self._rejection_summary: dict[str, dict[str, int]] = {}
        self._accepted_layout_bboxes: list[dict[str, Any]] = []
        self._nav2_map_result: MapExportResult | None = None
        self._mapf_map_result: MapExportResult | None = None
        self._focus_object: ObjectBBox3D | None = None
        self._rollouts_payload: list[dict[str, Any]] = []
        self._sampling_validation: dict[str, Any] = {}
        self._robot_prim_paths: list[str] = []
        self._robot_controllers: list[RuntimeRobotController] = []
        self._ros_bridge: InternalIsaacRosBridge | None = None
        self._environment_bbox: tuple[np.ndarray, np.ndarray] | None = None
        self._resolved_nav2_map_spec: TemplateMapSpec | None = None
        self._resolved_mapf_map_spec: TemplateMapSpec | None = None
        self._map_export_debug: dict[str, Any] = {}

    @property
    def sim_app(self):
        return self._sim_app

    def ensure_sim_app(self, *, headless: bool) -> None:
        if self._sim_app is not None:
            return
        self._sim_app = maybe_start_sim_app(
            headless=headless,
            enable_ros2_bridge=self._enable_ros2_runtime,
            extra_extensions=(
                ["isaacsim.asset.gen.omap"] if self._map_export_mode == "omap" else []
            ),
        )

    def build(self) -> RandomizedWarehouseBuildResult:
        current_step = "prepare_output_bundle"
        try:
            self._prepare_output_bundle()
            current_step = "resolve_assets_root"
            self._resolve_assets_root()
            current_step = "build_stage_base"
            self._build_stage_base()
            current_step = "apply_randomization_pipeline"
            self._apply_randomization_pipeline()
            current_step = "resolve_focus_object"
            self._resolve_focus_object()
            current_step = "export_maps"
            self._export_maps()
            current_step = "sample_rollouts"
            self._sample_rollouts()
            current_step = "spawn_robots"
            self._spawn_robots()
            current_step = "save_scene_usd"
            self._save_scene_usd()
            current_step = "write_team_config"
            self._write_team_config()
            current_step = "write_manifest"
            self._write_manifest()
            if self._enable_ros2_runtime:
                current_step = "initialize_ros_runtime"
                self._initialize_ros_runtime()
            if self._failure_snapshot_path.exists():
                try:
                    self._failure_snapshot_path.unlink()
                except Exception:
                    pass
        except Exception as exc:
            self._write_failure_snapshot(current_step=current_step, exception=exc)
            raise RuntimeError(
                f"Randomized warehouse build failed during step '{current_step}' "
                f"for scene '{self._scene_id}'."
            ) from exc

        return RandomizedWarehouseBuildResult(
            scene_id=self._scene_id,
            seed=self._seed,
            template_id=self._template.template_id,
            variant_id=self._template.variant_id,
            bundle_dir=self._bundle_dir,
            scene_usd_path=self._scene_usd_path,
            team_config_path=self._team_config_path,
            nav2_map_path=self._nav2_map_path,
            mapf_map_path=self._mapf_map_path,
            manifest_path=self._manifest_path,
            robot_prim_paths=list(self._robot_prim_paths),
            rollouts_dir=self._rollouts_dir,
            ros_bridge=self._ros_bridge,
            robot_controllers=list(self._robot_controllers),
            validation_summary=self._build_validation_summary(),
        )

    @property
    def failure_snapshot_path(self) -> Path:
        return self._failure_snapshot_path

    def _prepare_output_bundle(self) -> None:
        if self._bundle_dir.exists() and not self._overwrite:
            existing_files = [path.name for path in self._bundle_dir.iterdir()]
            if existing_files:
                raise FileExistsError(
                    f"Scene bundle already exists and is not empty: {self._bundle_dir}. "
                    "Pass overwrite=True or choose a different scene id."
                )
        self._bundle_dir.mkdir(parents=True, exist_ok=True)
        self._rollouts_dir.mkdir(parents=True, exist_ok=True)
        if self._failure_snapshot_path.exists():
            try:
                self._failure_snapshot_path.unlink()
            except Exception:
                pass

    def _resolve_assets_root(self) -> None:
        from isaacsim.storage.native import get_assets_root_path

        self._assets_root_path = str(get_assets_root_path()).rstrip("/")
        if not self._assets_root_path:
            raise RuntimeError("Isaac Sim assets root path is unavailable.")

    def _resolve_template_usd_path(self, usd_path: str) -> str:
        clean_path = str(usd_path).strip()
        if not clean_path:
            raise ValueError("Template base_environment_usd is empty.")
        if "://" in clean_path:
            return clean_path

        local_path = Path(clean_path).expanduser()
        if local_path.is_absolute() and local_path.exists():
            return str(local_path.resolve())

        if clean_path.startswith("/"):
            return f"{self._assets_root_path}{clean_path}"

        candidate_path = (self._template.template_config_path.parent / clean_path).resolve()
        if candidate_path.exists():
            return str(candidate_path)

        repo_relative_path = (self._repo_root / clean_path).resolve()
        if repo_relative_path.exists():
            return str(repo_relative_path)

        return str(candidate_path)

    def _build_stage_base(self) -> None:
        new_stage()
        set_stage_units(1.0)
        define_xform("/World")
        define_xform("/World/Env")
        define_xform(self._randomized_props_root)
        define_xform(self._robots_root)
        self._resolved_base_environment_usd = self._resolve_template_usd_path(
            self._template.base_environment_usd
        )
        add_reference(self._resolved_base_environment_usd, self._env_prim_path)
        self._ensure_physics_scene()
        self._update_sim(12)
        self._environment_bbox = self._safe_environment_bbox()

    def _apply_randomization_pipeline(self) -> None:
        self._apply_light_randomization()
        for randomizer in self._template.object_randomizers:
            if randomizer.policy == "appearance_only":
                self._apply_appearance_randomization(randomizer)
            elif randomizer.policy == "jittered_existing":
                self._apply_jitter_randomization(randomizer)
            elif randomizer.policy in {"copied_from_group", "spawned_floor_prop"}:
                self._apply_copied_props(randomizer)
            else:
                raise ValueError(
                    f"Unsupported randomization policy '{randomizer.policy}' in template "
                    f"'{self._template.template_id}'."
                )

        self._validate_final_layout()

    def _resolve_light_paths(
        self,
        selectors: tuple[SelectorSpec, ...],
        *,
        cache_key: str,
    ) -> list[str]:
        if cache_key in self._resolved_light_paths:
            return list(self._resolved_light_paths[cache_key])
        resolved = self._resolve_selector_paths(selectors, root_mode="match")
        self._resolved_light_paths[cache_key] = resolved
        return list(resolved)

    def _resolve_group_paths(self, group_name: str, *, required: bool | None = None) -> list[str]:
        if group_name in self._resolved_group_paths:
            cached_paths = list(self._resolved_group_paths[group_name])
            group = self._template.group_map.get(group_name)
            must_exist = (
                bool(group.required) if group is not None and required is None else bool(required)
            )
            if must_exist and not cached_paths:
                raise RuntimeError(
                    f"Required object group '{group_name}' resolved to no valid prims in template "
                    f"'{self._template.template_id}'."
                )
            return cached_paths

        group = self._template.group_map.get(group_name)
        if group is None:
            raise KeyError(
                f"Template '{self._template.template_id}' does not define object group '{group_name}'."
            )

        matched_paths = self._resolve_selector_paths(group.selectors, root_mode=group.root_mode)
        resolved_paths: list[str] = []
        rejected_paths: list[dict[str, str]] = []
        for path in matched_paths:
            rejection_reason = self._validate_group_root(path, group=group)
            if rejection_reason is not None:
                rejected_paths.append({"prim_path": path, "reason": rejection_reason})
                continue
            resolved_paths.append(path)

        self._resolved_group_paths[group_name] = resolved_paths
        self._resolved_group_details[group_name] = {
            "required": bool(group.required),
            "root_mode": group.root_mode,
            "selectors": [
                {"mode": selector.mode, "value": selector.value} for selector in group.selectors
            ],
            "matched_paths": matched_paths,
            "resolved_paths": list(resolved_paths),
            "rejected_paths": rejected_paths,
        }

        must_exist = group.required if required is None else bool(required)
        if must_exist and not resolved_paths:
            rejected_summary = ", ".join(
                f"{entry['prim_path']} ({entry['reason']})" for entry in rejected_paths[:8]
            )
            raise RuntimeError(
                f"Required object group '{group_name}' resolved to no valid prims in template "
                f"'{self._template.template_id}'. Rejections: {rejected_summary or 'none'}"
            )

        return list(resolved_paths)

    def _resolve_randomizer_target_paths(self, spec: ObjectRandomizationSpec) -> list[str]:
        if spec.target_group_name:
            return self._resolve_group_paths(spec.target_group_name, required=spec.required)
        if spec.selectors:
            return self._resolve_selector_paths(spec.selectors, root_mode="nearest_xform")
        if spec.required:
            raise RuntimeError(f"Randomizer '{spec.name}' has no target_group_name or selectors.")
        return []

    def _resolve_randomizer_source_paths(self, spec: ObjectRandomizationSpec) -> list[str]:
        if spec.source_group_name:
            return self._resolve_group_paths(spec.source_group_name, required=spec.required)
        if spec.selectors:
            return self._resolve_selector_paths(spec.selectors, root_mode="nearest_xform")
        if spec.required:
            raise RuntimeError(f"Randomizer '{spec.name}' has no source_group_name or selectors.")
        return []

    def _resolve_zones(
        self,
        zone_ids: tuple[str, ...],
        *,
        allowed_types: set[str] | None = None,
    ) -> list[PlacementZone]:
        zone_map = self._template.zone_map
        if zone_ids:
            resolved: list[PlacementZone] = []
            for zone_id in zone_ids:
                zone = zone_map.get(zone_id)
                if zone is None:
                    raise KeyError(
                        f"Template '{self._template.template_id}' does not define placement zone "
                        f"'{zone_id}'."
                    )
                if allowed_types and zone.zone_type not in allowed_types:
                    allowed = ", ".join(sorted(allowed_types))
                    raise ValueError(
                        f"Zone '{zone_id}' has type '{zone.zone_type}', expected one of: {allowed}"
                    )
                resolved.append(zone)
            return resolved

        if allowed_types is None:
            return list(self._template.placement_zones)
        return [zone for zone in self._template.placement_zones if zone.zone_type in allowed_types]

    def _resolve_selector_paths(
        self,
        selectors: tuple[SelectorSpec, ...],
        *,
        root_mode: str,
    ) -> list[str]:
        if not selectors:
            return []
        stage = get_stage()
        resolved: list[str] = []
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if not (path == self._env_prim_path or path.startswith(f"{self._env_prim_path}/")):
                continue
            if any(self._selector_matches(prim, selector) for selector in selectors):
                resolved.append(self._canonicalize_prim_path(path, root_mode=root_mode))
        return self._dedupe_root_paths(resolved)

    def _selector_matches(self, prim, selector: SelectorSpec) -> bool:
        path = prim.GetPath().pathString
        name = prim.GetName()
        value = str(selector.value)
        if selector.mode == "exact_path":
            return self._exact_path_matches(path, value)
        if selector.mode == "glob":
            return fnmatch(path, value) or fnmatch(name, value)
        if selector.mode == "regex":
            regex = re.compile(value)
            return bool(regex.search(path) or regex.search(name))
        if selector.mode == "semantic":
            semantic_value = value.lower()
            for attribute in prim.GetAttributes():
                attr_name = attribute.GetName().lower()
                if "semantic" not in attr_name:
                    continue
                attr_value = attribute.Get()
                if attr_value and semantic_value in str(attr_value).lower():
                    return True
            custom_data = prim.GetCustomData() or {}
            for custom_value in custom_data.values():
                if semantic_value in str(custom_value).lower():
                    return True
            return False
        return False

    def _exact_path_matches(self, path: str, value: str) -> bool:
        if path == value:
            return True
        env_prefix = self._env_prim_path.rstrip("/")
        if not (value.startswith(f"{env_prefix}/") and path.startswith(f"{env_prefix}/")):
            return False

        expected_relative = value[len(env_prefix) :].strip("/")
        candidate_relative = path[len(env_prefix) :].strip("/")
        if not expected_relative or not candidate_relative:
            return False
        return candidate_relative == expected_relative or candidate_relative.endswith(
            f"/{expected_relative}"
        )

    def _canonicalize_prim_path(self, prim_path: str, *, root_mode: str) -> str:
        from pxr import UsdGeom

        stage = get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return prim_path

        if root_mode in {"match", "exact_path"}:
            return prim_path

        if prim.IsA(UsdGeom.Xform) or prim.GetTypeName() in {"Xform", "Scope"}:
            return prim_path

        current = prim.GetParent()
        while current and current.IsValid():
            current_path = current.GetPath().pathString
            if current_path in {"/", self._env_prim_path}:
                break
            if current.IsA(UsdGeom.Xform) or current.GetTypeName() in {"Xform", "Scope"}:
                return current_path
            current = current.GetParent()
        return prim_path

    def _dedupe_root_paths(self, prim_paths: list[str]) -> list[str]:
        deduped: list[str] = []
        for path in sorted(set(prim_paths), key=len):
            if any(path == other or path.startswith(f"{other}/") for other in deduped):
                continue
            deduped.append(path)
        return deduped

    def _validate_group_root(self, prim_path: str, *, group: ObjectGroupSpec) -> str | None:
        from pxr import UsdGeom, UsdShade

        stage = get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return "missing_prim"
        if prim_path == self._env_prim_path:
            return "environment_root"

        lower_path = prim_path.lower()
        prim_name = prim.GetName().lower()
        prim_type = str(prim.GetTypeName()).lower()
        if any(token in lower_path for token in ("/looks", "/materials", "/shaders")):
            return "material_subtree"
        if prim_name in {"looks", "materials", "shaders"}:
            return "material_container"
        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader):
            return "material_prim"
        if prim_type in {"material", "shader", "geomsubset"}:
            return f"invalid_type:{prim_type}"

        try:
            bbox_min, bbox_max = compute_world_bbox(prim_path)
        except Exception as exc:
            return f"bbox_error:{type(exc).__name__}"

        extent_xyz = bbox_max - bbox_min
        if not np.all(np.isfinite(extent_xyz)):
            return "non_finite_bbox"
        if np.any(extent_xyz < 0.01):
            return "degenerate_bbox"

        planar_extent_x = float(extent_xyz[0])
        planar_extent_y = float(extent_xyz[1])
        if (
            planar_extent_x > 12.0
            or planar_extent_y > 12.0
            or planar_extent_x * planar_extent_y > 80.0
        ):
            return "suspiciously_large_bbox"

        if group.root_mode != "match" and not (
            prim.IsA(UsdGeom.Xform) or prim.GetTypeName() in {"Xform", "Scope"}
        ):
            return "non_transform_root"
        return None

    def _sample_candidate_yaw_deg(self, base_yaw_deg: float, spec: ObjectRandomizationSpec) -> float:
        if spec.snapped_yaw_deg:
            base_yaw_deg = float(self._rng.choice(np.array(spec.snapped_yaw_deg, dtype=float)))
        return float(base_yaw_deg + self._rng.uniform(spec.yaw_jitter_deg[0], spec.yaw_jitter_deg[1]))

    def _sample_zone_xy(self, zone: PlacementZone) -> tuple[float, float]:
        return (
            float(self._rng.uniform(zone.min_xyz[0], zone.max_xyz[0])),
            float(self._rng.uniform(zone.min_xyz[1], zone.max_xyz[1])),
        )

    def _bbox_within_zone(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        zone: PlacementZone,
        *,
        margin_m: float = 0.02,
    ) -> bool:
        return (
            bbox_min[0] >= float(zone.min_xyz[0]) - margin_m
            and bbox_max[0] <= float(zone.max_xyz[0]) + margin_m
            and bbox_min[1] >= float(zone.min_xyz[1]) - margin_m
            and bbox_max[1] <= float(zone.max_xyz[1]) + margin_m
            and bbox_min[2] >= float(zone.min_xyz[2]) - margin_m
            and bbox_max[2] <= float(zone.max_xyz[2]) + margin_m
        )

    def _place_object_in_zone(
        self,
        prim_path: str,
        *,
        zone: PlacementZone,
        base_z: float,
        yaw_deg: float,
        uniform_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        bbox_size_limit = np.array(zone.max_xyz, dtype=float) - np.array(zone.min_xyz, dtype=float)
        candidate_xy = self._sample_zone_xy(zone)
        candidate_position = np.array([candidate_xy[0], candidate_xy[1], base_z], dtype=float)

        set_xform_pose(
            prim_path,
            tuple(candidate_position.tolist()),
            yaw_deg=yaw_deg,
            scale_xyz=(uniform_scale, uniform_scale, uniform_scale),
        )
        self._update_sim(2)
        bbox_min, bbox_max = compute_world_bbox(prim_path)
        bbox_size = bbox_max - bbox_min
        if np.any(bbox_size > (bbox_size_limit + 1e-3)):
            return None

        shift_xyz = np.zeros(3, dtype=float)
        if zone.zone_type == "support":
            shift_xyz[2] += float(zone.min_xyz[2]) - float(bbox_min[2])
        else:
            if bbox_min[2] < float(zone.min_xyz[2]):
                shift_xyz[2] += float(zone.min_xyz[2]) - float(bbox_min[2])
        if bbox_max[2] > float(zone.max_xyz[2]):
            shift_xyz[2] += float(zone.max_xyz[2]) - float(bbox_max[2])
        if bbox_min[0] < float(zone.min_xyz[0]):
            shift_xyz[0] += float(zone.min_xyz[0]) - float(bbox_min[0])
        if bbox_max[0] > float(zone.max_xyz[0]):
            shift_xyz[0] += float(zone.max_xyz[0]) - float(bbox_max[0])
        if bbox_min[1] < float(zone.min_xyz[1]):
            shift_xyz[1] += float(zone.min_xyz[1]) - float(bbox_min[1])
        if bbox_max[1] > float(zone.max_xyz[1]):
            shift_xyz[1] += float(zone.max_xyz[1]) - float(bbox_max[1])

        if np.linalg.norm(shift_xyz) > 1e-6:
            candidate_position += shift_xyz
            set_xform_pose(
                prim_path,
                tuple(candidate_position.tolist()),
                yaw_deg=yaw_deg,
                scale_xyz=(uniform_scale, uniform_scale, uniform_scale),
            )
            self._update_sim(2)
            bbox_min, bbox_max = compute_world_bbox(prim_path)

        if not self._bbox_within_zone(bbox_min, bbox_max, zone):
            return None

        return candidate_position, bbox_min, bbox_max

    def _record_rejection(self, category: str, reason: str) -> None:
        category_summary = self._rejection_summary.setdefault(category, {})
        category_summary[reason] = int(category_summary.get(reason, 0)) + 1

    def _append_layout_bbox(
        self,
        *,
        prim_path: str,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        margin_m: float,
        category: str,
        source_group_name: str = "",
    ) -> None:
        self._accepted_layout_bboxes.append(
            {
                "prim_path": prim_path,
                "bbox_min": np.array(bbox_min, dtype=float),
                "bbox_max": np.array(bbox_max, dtype=float),
                "margin_m": float(margin_m),
                "category": category,
                "source_group_name": source_group_name,
            }
        )

    def _remove_prim(self, prim_path: str) -> None:
        stage = get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            stage.RemovePrim(prim_path)

    def _duplicate_prim(self, source_prim_path: str, dest_prim_path: str) -> None:
        import omni.kit.commands
        import omni.usd

        self._remove_prim(dest_prim_path)
        try:
            omni.kit.commands.execute(
                "CopyPrim",
                path_from=source_prim_path,
                path_to=dest_prim_path,
            )
        except Exception:
            duplicate_prim = getattr(omni.usd, "duplicate_prim", None)
            if callable(duplicate_prim):
                duplicate_prim(get_stage(), source_prim_path, dest_prim_path)
        copied_prim = get_stage().GetPrimAtPath(dest_prim_path)
        if not copied_prim or not copied_prim.IsValid():
            raise RuntimeError(
                f"Failed to copy source prim '{source_prim_path}' to '{dest_prim_path}'."
            )

    def _apply_light_randomization(self) -> None:
        for light_spec in self._template.light_randomizers:
            light_paths = self._resolve_light_paths(
                light_spec.selectors,
                cache_key=light_spec.name,
            )
            for light_path in light_paths:
                stage = get_stage()
                prim = stage.GetPrimAtPath(light_path)
                if not prim or not prim.IsValid():
                    continue

                intensity = float(
                    self._rng.uniform(light_spec.intensity_range[0], light_spec.intensity_range[1])
                )
                color = self._sample_rgb(light_spec.color_value_range)
                temperature = float(
                    self._rng.uniform(light_spec.temperature_range[0], light_spec.temperature_range[1])
                )
                self._set_attr_if_present(prim, ("intensity", "inputs:intensity"), intensity)
                self._set_attr_if_present(prim, ("color", "inputs:color"), color)
                self._set_attr_if_present(
                    prim,
                    ("enableColorTemperature", "inputs:enableColorTemperature"),
                    True,
                )
                self._set_attr_if_present(
                    prim,
                    ("colorTemperature", "inputs:colorTemperature"),
                    temperature,
                )

                self._randomization_records.append(
                    {
                        "category": "light",
                        "name": light_spec.name,
                        "prim_path": light_path,
                        "intensity": intensity,
                        "color_rgb": color,
                        "temperature_k": temperature,
                    }
                )

    def _apply_appearance_randomization(self, spec: ObjectRandomizationSpec) -> None:
        from pxr import Gf, UsdGeom, Vt

        object_paths = self._resolve_randomizer_target_paths(spec)
        if spec.required and not object_paths:
            raise RuntimeError(f"Required randomizer '{spec.name}' resolved to no prims.")
        for object_path in object_paths:
            color = self._sample_rgb(spec.color_value_range)
            for prim in get_stage().Traverse():
                path = prim.GetPath().pathString
                if not (path == object_path or path.startswith(f"{object_path}/")):
                    continue
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                gprim = UsdGeom.Gprim(prim)
                gprim.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))

            self._randomization_records.append(
                {
                    "category": "appearance",
                    "name": spec.name,
                    "prim_path": object_path,
                    "color_rgb": color,
                }
            )

    def _apply_jitter_randomization(self, spec: ObjectRandomizationSpec) -> None:
        object_paths = self._resolve_randomizer_target_paths(spec)
        if spec.required and not object_paths:
            raise RuntimeError(f"Required randomization selector '{spec.name}' resolved to no prims.")

        zone_ids = tuple(spec.anchor_zone_ids or spec.support_zone_ids)
        candidate_zones = (
            self._resolve_zones(zone_ids, allowed_types={"anchor", "support"})
            if zone_ids
            else []
        )
        for object_path in object_paths:
            original_position, original_orientation = get_world_pose_xyzw(object_path)
            original_yaw_deg = quaternion_xyzw_to_yaw(original_orientation) * 180.0 / math.pi
            accepted_bbox: tuple[np.ndarray, np.ndarray] | None = None
            accepted_position = original_position.copy()
            accepted_yaw_deg = original_yaw_deg
            accepted_scale = 1.0
            accepted_zone_id = ""
            accepted_from_original = True

            for attempt_index in range(spec.max_attempts):
                candidate_yaw_deg = self._sample_candidate_yaw_deg(original_yaw_deg, spec)
                candidate_scale = float(
                    self._rng.uniform(spec.uniform_scale_range[0], spec.uniform_scale_range[1])
                )
                zone_id = ""

                if candidate_zones:
                    zone = candidate_zones[int(self._rng.integers(0, len(candidate_zones)))]
                    placement = self._place_object_in_zone(
                        object_path,
                        zone=zone,
                        base_z=float(original_position[2]),
                        yaw_deg=candidate_yaw_deg,
                        uniform_scale=candidate_scale,
                    )
                    if placement is None:
                        self._record_rejection(spec.name, "outside_zone")
                        continue
                    candidate_position, bbox_min, bbox_max = placement
                    zone_id = zone.zone_id
                else:
                    candidate_position = original_position.copy()
                    candidate_position[0] += float(
                        self._rng.uniform(-spec.xy_jitter_m[0], spec.xy_jitter_m[0])
                    )
                    candidate_position[1] += float(
                        self._rng.uniform(-spec.xy_jitter_m[1], spec.xy_jitter_m[1])
                    )
                    set_xform_pose(
                        object_path,
                        tuple(candidate_position.tolist()),
                        yaw_deg=candidate_yaw_deg,
                        scale_xyz=(candidate_scale, candidate_scale, candidate_scale),
                    )
                    self._update_sim(2)
                    bbox_min, bbox_max = compute_world_bbox(object_path)

                if self._intersects_keepout(
                    bbox_min,
                    bbox_max,
                    keepout_zone_ids=spec.keepout_zone_ids,
                    margin_m=spec.collision_margin_m,
                ):
                    self._record_rejection(spec.name, "keepout")
                    continue
                if self._bbox_intersects_accepted_layout(
                    bbox_min,
                    bbox_max,
                    margin_m=spec.collision_margin_m,
                ):
                    self._record_rejection(spec.name, "layout_overlap")
                    continue
                if not self._physx_overlap_clear(object_path, bbox_min, bbox_max):
                    self._record_rejection(spec.name, "physics_overlap")
                    continue

                accepted_bbox = (bbox_min, bbox_max)
                accepted_position = candidate_position
                accepted_yaw_deg = candidate_yaw_deg
                accepted_scale = candidate_scale
                accepted_zone_id = zone_id
                accepted_from_original = False
                break

            set_xform_pose(
                object_path,
                tuple(accepted_position.tolist()),
                yaw_deg=accepted_yaw_deg,
                scale_xyz=(accepted_scale, accepted_scale, accepted_scale),
            )
            self._update_sim(2)

            bbox_min, bbox_max = compute_world_bbox(object_path)
            if self._intersects_keepout(
                bbox_min,
                bbox_max,
                keepout_zone_ids=spec.keepout_zone_ids,
                margin_m=spec.collision_margin_m,
            ) or self._bbox_intersects_accepted_layout(
                bbox_min,
                bbox_max,
                margin_m=spec.collision_margin_m,
            ) or not self._physx_overlap_clear(object_path, bbox_min, bbox_max):
                raise RuntimeError(
                    f"Object '{object_path}' from randomizer '{spec.name}' could not be placed safely."
                )
            self._append_layout_bbox(
                prim_path=object_path,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                margin_m=spec.collision_margin_m,
                category="layout",
                source_group_name=spec.target_group_name,
            )
            self._randomization_records.append(
                {
                    "category": "layout",
                    "name": spec.name,
                    "prim_path": object_path,
                    "policy": spec.policy,
                    "target_group_name": spec.target_group_name,
                    "zone_id": accepted_zone_id,
                    "accepted_from_original": bool(accepted_from_original),
                    "accepted_bbox": None
                    if accepted_bbox is None
                    else {
                        "min_xyz": accepted_bbox[0].tolist(),
                        "max_xyz": accepted_bbox[1].tolist(),
                    },
                    "position_xyz": accepted_position.tolist(),
                    "yaw_deg": accepted_yaw_deg,
                    "uniform_scale": accepted_scale,
                }
            )

    def _apply_copied_props(self, spec: ObjectRandomizationSpec) -> None:
        source_paths = self._resolve_randomizer_source_paths(spec)
        if spec.required and not source_paths:
            raise RuntimeError(f"Required copy randomizer '{spec.name}' resolved to no source prims.")

        support_zones = self._resolve_zones(
            spec.support_zone_ids,
            allowed_types={"support", "anchor"},
        )
        if spec.required and not support_zones:
            raise RuntimeError(
                f"Required copy randomizer '{spec.name}' does not have any support zones."
            )

        prop_count = int(self._rng.integers(spec.spawn_count_range[0], spec.spawn_count_range[1] + 1))
        if prop_count <= 0:
            return

        accepted_count = 0
        for prop_index in range(prop_count):
            prop_path = f"{self._randomized_props_root}/Copy_{spec.name}_{prop_index + 1}"
            accepted_payload: dict[str, Any] | None = None

            for _ in range(spec.max_attempts):
                if not source_paths or not support_zones:
                    break

                source_path = source_paths[int(self._rng.integers(0, len(source_paths)))]
                source_position, source_orientation = get_world_pose_xyzw(source_path)
                source_yaw_deg = quaternion_xyzw_to_yaw(source_orientation) * 180.0 / math.pi
                zone = support_zones[int(self._rng.integers(0, len(support_zones)))]
                uniform_scale = float(
                    self._rng.uniform(spec.uniform_scale_range[0], spec.uniform_scale_range[1])
                )
                yaw_deg = self._sample_candidate_yaw_deg(source_yaw_deg, spec)

                self._remove_prim(prop_path)
                self._duplicate_prim(source_path, prop_path)
                self._update_sim(2)

                placement = self._place_object_in_zone(
                    prop_path,
                    zone=zone,
                    base_z=float(source_position[2]),
                    yaw_deg=yaw_deg,
                    uniform_scale=uniform_scale,
                )
                if placement is None:
                    self._record_rejection(spec.name, "outside_zone")
                    self._remove_prim(prop_path)
                    continue

                position_xyz, bbox_min, bbox_max = placement
                if self._intersects_keepout(
                    bbox_min,
                    bbox_max,
                    keepout_zone_ids=spec.keepout_zone_ids,
                    margin_m=spec.collision_margin_m,
                ):
                    self._record_rejection(spec.name, "keepout")
                    self._remove_prim(prop_path)
                    continue
                if self._bbox_intersects_accepted_layout(
                    bbox_min,
                    bbox_max,
                    margin_m=spec.collision_margin_m,
                ):
                    self._record_rejection(spec.name, "layout_overlap")
                    self._remove_prim(prop_path)
                    continue
                if not self._physx_overlap_clear(prop_path, bbox_min, bbox_max):
                    self._record_rejection(spec.name, "physics_overlap")
                    self._remove_prim(prop_path)
                    continue

                self._append_layout_bbox(
                    prim_path=prop_path,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    margin_m=spec.collision_margin_m,
                    category="copied_prop",
                    source_group_name=spec.source_group_name,
                )
                accepted_payload = {
                    "category": "copied_prop",
                    "name": spec.name,
                    "prim_path": prop_path,
                    "policy": spec.policy,
                    "source_prim_path": source_path,
                    "source_group_name": spec.source_group_name,
                    "zone_id": zone.zone_id,
                    "position_xyz": position_xyz.tolist(),
                    "yaw_deg": yaw_deg,
                    "uniform_scale": uniform_scale,
                    "accepted_bbox": {
                        "min_xyz": bbox_min.tolist(),
                        "max_xyz": bbox_max.tolist(),
                    },
                }
                break

            if accepted_payload is None:
                self._remove_prim(prop_path)
                continue

            accepted_count += 1
            self._randomization_records.append(accepted_payload)

        if spec.required and accepted_count <= 0:
            raise RuntimeError(
                f"Required randomizer '{spec.name}' did not produce any accepted copied props."
            )

        self._randomization_records.append(
            {
                "category": "spawn_summary",
                "name": spec.name,
                "policy": spec.policy,
                "source_group_name": spec.source_group_name,
                "requested_count": prop_count,
                "accepted_count": accepted_count,
            }
        )

    def _resolve_focus_object(self) -> None:
        focus_paths: list[str] = []
        for group_name in self._template.focus_group_names:
            focus_paths.extend(self._resolve_group_paths(group_name, required=False))
        focus_paths = self._dedupe_root_paths(focus_paths)
        largest_bbox: ObjectBBox3D | None = None
        largest_area = -1.0
        for focus_path in focus_paths:
            bbox_min, bbox_max = compute_world_bbox(focus_path)
            area = float((bbox_max[0] - bbox_min[0]) * (bbox_max[1] - bbox_min[1]))
            if area > largest_area:
                largest_area = area
                largest_bbox = ObjectBBox3D(
                    prim_path=focus_path,
                    min_xyz=bbox_min,
                    max_xyz=bbox_max,
                )
        self._focus_object = largest_bbox

    def _export_maps(self) -> None:
        import omni.timeline

        def _export_with_selected_start(
            *,
            yaml_path,
            resolution_m,
            origin_hint_xyz,
            min_bound_xyz,
            max_bound_xyz,
        ):
            map_spec = TemplateMapSpec(
                resolution_m=float(resolution_m),
                origin_hint_xyz=tuple(float(value) for value in origin_hint_xyz),
                min_bound_xyz=tuple(float(value) for value in min_bound_xyz),
                max_bound_xyz=tuple(float(value) for value in max_bound_xyz),
            )
            candidate_start_locations = self._candidate_map_start_locations(map_spec)
            start_location_xyz = (
                candidate_start_locations[0]
                if candidate_start_locations
                else self._choose_map_start_location(map_spec)
            )
            self._map_export_debug[str(yaml_path)] = {
                "mode": "omap",
                "candidate_start_locations_xyz": [
                    [float(value) for value in candidate_xyz]
                    for candidate_xyz in candidate_start_locations[:10]
                ],
                "chosen_start_location_xyz": [float(value) for value in start_location_xyz],
            }
            self._update_sim(16)
            result = export_occupancy_map(
                yaml_path=yaml_path,
                resolution_m=resolution_m,
                origin_hint_xyz=origin_hint_xyz,
                min_bound_xyz=min_bound_xyz,
                max_bound_xyz=max_bound_xyz,
                start_location_xyz=start_location_xyz,
            )
            self._map_export_debug[str(yaml_path)]["quality_score"] = list(
                self._map_quality_score(result)
            )
            return result

        def _export_with_bbox_rasterization(
            *,
            yaml_path,
            resolution_m,
            origin_hint_xyz,
            min_bound_xyz,
            max_bound_xyz,
        ):
            map_spec = TemplateMapSpec(
                resolution_m=float(resolution_m),
                origin_hint_xyz=tuple(float(value) for value in origin_hint_xyz),
                min_bound_xyz=tuple(float(value) for value in min_bound_xyz),
                max_bound_xyz=tuple(float(value) for value in max_bound_xyz),
            )
            obstacle_bboxes, debug_payload = self._collect_bbox_map_obstacle_bboxes(map_spec)
            self._map_export_debug[str(yaml_path)] = dict(debug_payload)
            result = export_bbox_occupancy_map(
                yaml_path=yaml_path,
                resolution_m=resolution_m,
                origin_hint_xyz=origin_hint_xyz,
                min_bound_xyz=min_bound_xyz,
                max_bound_xyz=max_bound_xyz,
                obstacle_bboxes=obstacle_bboxes,
            )
            self._map_export_debug[str(yaml_path)]["quality_score"] = list(
                self._map_quality_score(result)
            )
            return result

        timeline = omni.timeline.get_timeline_interface()
        was_playing = timeline.is_playing()
        if not was_playing:
            timeline.play()

        try:
            self._ensure_physics_scene()
            self._resolved_nav2_map_spec = self._resolve_map_spec(self._template.nav2_map)
            self._resolved_mapf_map_spec = self._resolve_map_spec(self._template.mapf_map)
            export_map = (
                _export_with_selected_start
                if self._map_export_mode == "omap"
                else _export_with_bbox_rasterization
            )
            self._nav2_map_result = export_map(
                yaml_path=self._nav2_map_path,
                resolution_m=self._resolved_nav2_map_spec.resolution_m,
                origin_hint_xyz=self._resolved_nav2_map_spec.origin_hint_xyz,
                min_bound_xyz=self._resolved_nav2_map_spec.min_bound_xyz,
                max_bound_xyz=self._resolved_nav2_map_spec.max_bound_xyz,
            )
            self._mapf_map_result = export_map(
                yaml_path=self._mapf_map_path,
                resolution_m=self._resolved_mapf_map_spec.resolution_m,
                origin_hint_xyz=self._resolved_mapf_map_spec.origin_hint_xyz,
                min_bound_xyz=self._resolved_mapf_map_spec.min_bound_xyz,
                max_bound_xyz=self._resolved_mapf_map_spec.max_bound_xyz,
            )
        finally:
            if not was_playing:
                pause_timeline = getattr(timeline, "pause", None)
                if callable(pause_timeline):
                    pause_timeline()
                else:
                    timeline.stop()
                self._update_sim(2)

    def _collect_bbox_map_obstacle_bboxes(
        self,
        map_spec: TemplateMapSpec,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
        from pxr import Usd, UsdGeom

        try:
            from pxr import UsdPhysics
        except Exception:
            UsdPhysics = None

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ],
        )
        search_roots = [self._env_prim_path, self._randomized_props_root]
        ignored_tokens = (
            "/looks",
            "/materials",
            "/shaders",
            "light",
            "camera",
            "sensor",
            "helper",
            "visualization",
            "debug",
            "viz",
            "floor",
            "ground",
            "ceiling",
            "sky",
        )
        map_span_x = float(map_spec.max_bound_xyz[0]) - float(map_spec.min_bound_xyz[0])
        map_span_y = float(map_spec.max_bound_xyz[1]) - float(map_spec.min_bound_xyz[1])
        min_obstacle_top_z = float(map_spec.min_bound_xyz[2]) + 0.08
        seen: set[tuple[float, float, float, float, float, float]] = set()
        obstacle_bboxes: list[tuple[np.ndarray, np.ndarray]] = []
        debug_payload: dict[str, Any] = {
            "mode": "bbox",
            "collision_gprim_count": 0,
            "collision_root_count": 0,
            "fallback_render_gprim_count": 0,
            "fallback_render_root_count": 0,
            "accepted_layout_bbox_count": 0,
            "skipped_shell_like_count": 0,
            "skipped_invalid_root_count": 0,
            "obstacle_bbox_count": 0,
        }
        processed_root_paths: set[str] = set()

        def _append_bbox(
            bbox_min: np.ndarray,
            bbox_max: np.ndarray,
            *,
            source_key: str,
        ) -> bool:
            extent_xyz = bbox_max - bbox_min
            if not (np.all(np.isfinite(bbox_min)) and np.all(np.isfinite(bbox_max))):
                return False
            if np.any(extent_xyz <= 0.0):
                return False
            if float(bbox_max[2]) <= min_obstacle_top_z:
                return False
            if float(extent_xyz[2]) < 0.02:
                return False
            if float(extent_xyz[0]) < 0.02 and float(extent_xyz[1]) < 0.02:
                return False
            if (
                float(extent_xyz[0]) > map_span_x * 0.85
                and float(extent_xyz[1]) > map_span_y * 0.85
            ) or (float(extent_xyz[0] * extent_xyz[1]) > (map_span_x * map_span_y * 0.60)):
                debug_payload["skipped_shell_like_count"] += 1
                return False

            dedupe_key = tuple(
                round(float(value), 3)
                for value in (
                    bbox_min[0],
                    bbox_min[1],
                    bbox_min[2],
                    bbox_max[0],
                    bbox_max[1],
                    bbox_max[2],
                )
            )
            if dedupe_key in seen:
                return False
            seen.add(dedupe_key)
            obstacle_bboxes.append((np.array(bbox_min, dtype=float), np.array(bbox_max, dtype=float)))
            debug_payload[source_key] = int(debug_payload.get(source_key, 0)) + 1
            return True

        def _root_bbox_for_map(root_path: str) -> tuple[np.ndarray, np.ndarray] | None:
            if not root_path or root_path in processed_root_paths or root_path == self._env_prim_path:
                return None
            processed_root_paths.add(root_path)

            root_path_lower = root_path.lower()
            if any(token in root_path_lower for token in ignored_tokens):
                debug_payload["skipped_invalid_root_count"] += 1
                return None
            try:
                bbox_min, bbox_max = compute_world_bbox(root_path)
            except Exception:
                debug_payload["skipped_invalid_root_count"] += 1
                return None
            extent_xyz = bbox_max - bbox_min
            if not (np.all(np.isfinite(bbox_min)) and np.all(np.isfinite(bbox_max))):
                debug_payload["skipped_invalid_root_count"] += 1
                return None
            if np.any(extent_xyz <= 0.0):
                debug_payload["skipped_invalid_root_count"] += 1
                return None
            if float(extent_xyz[2]) < 0.02:
                debug_payload["skipped_invalid_root_count"] += 1
                return None
            if float(extent_xyz[0]) < 0.02 and float(extent_xyz[1]) < 0.02:
                debug_payload["skipped_invalid_root_count"] += 1
                return None
            return bbox_min, bbox_max

        def _collect_stage_bboxes(
            *,
            require_collision_api: bool,
            gprim_key: str,
            root_key: str,
        ) -> int:
            accepted = 0
            for prim in get_stage().Traverse():
                prim_path = prim.GetPath().pathString
                if not any(
                    prim_path == search_root or prim_path.startswith(f"{search_root}/")
                    for search_root in search_roots
                ):
                    continue
                if not prim or not prim.IsValid() or not prim.IsActive():
                    continue
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                if require_collision_api and (UsdPhysics is None or not prim.HasAPI(UsdPhysics.CollisionAPI)):
                    continue

                prim_path_lower = prim_path.lower()
                if any(token in prim_path_lower for token in ignored_tokens):
                    continue
                debug_payload[gprim_key] = int(debug_payload.get(gprim_key, 0)) + 1

                root_path = self._canonicalize_prim_path(prim_path, root_mode="nearest_xform")
                root_bbox = _root_bbox_for_map(root_path)
                if root_bbox is None:
                    continue
                bbox_min, bbox_max = root_bbox
                if _append_bbox(bbox_min, bbox_max, source_key=root_key):
                    accepted += 1
            return accepted

        collision_count = _collect_stage_bboxes(
            require_collision_api=True,
            gprim_key="collision_gprim_count",
            root_key="collision_root_count",
        )
        if collision_count <= 0:
            _collect_stage_bboxes(
                require_collision_api=False,
                gprim_key="fallback_render_gprim_count",
                root_key="fallback_render_root_count",
            )

        for accepted in self._accepted_layout_bboxes:
            _append_bbox(
                np.array(accepted["bbox_min"], dtype=float),
                np.array(accepted["bbox_max"], dtype=float),
                source_key="accepted_layout_bbox_count",
            )

        debug_payload["obstacle_bbox_count"] = len(obstacle_bboxes)
        debug_payload["environment_bbox"] = self._environment_bbox_dict()
        return obstacle_bboxes, debug_payload

    def _sample_rollouts(self) -> None:
        occupancy_map = OccupancyMap.load(str(self._nav2_map_path))
        focus_xy = None if self._focus_object is None else self._focus_object.center_xyz[:2]
        max_planar_radius_m = max(
            float(adapter.default_planar_radius_m) for adapter in self._robot_adapters
        )
        self._rollouts_payload, self._sampling_validation = sample_multi_robot_rollouts(
            occupancy_map=occupancy_map,
            robot_names=self._robot_names,
            rollout_count=self._rollout_count,
            rng=self._rng,
            inflation_radius_m=max_planar_radius_m + 0.10,
            min_pairwise_distance_m=max(1.5, max_planar_radius_m * 2.5),
            min_goal_distance_m=3.0,
            focus_xy=focus_xy,
            focus_distance_range_m=self._template.focus_distance_range_m,
        )

    def _spawn_robots(self) -> None:
        if not self._rollouts_payload:
            raise RuntimeError("No rollouts were generated before robot spawning.")

        first_rollout = self._rollouts_payload[0]
        for index, (robot_name, adapter) in enumerate(zip(self._robot_names, self._robot_adapters), start=1):
            robot_prim_path = f"{self._robots_root}/{adapter.model_id}_{index}"
            robot_config = next(
                robot for robot in first_rollout["robots"] if robot["name"] == robot_name
            )
            initial_pose = robot_config["initial_pose"]
            adapter.spawn_robot(
                assets_root_path=self._assets_root_path,
                prim_path=robot_prim_path,
                robot_index=index,
                position_xyz=(
                    float(initial_pose["x"]),
                    float(initial_pose["y"]),
                    float(initial_pose["z"]),
                ),
                yaw_deg=float(initial_pose["yaw"]) * 180.0 / math.pi,
            )
            self._robot_prim_paths.append(robot_prim_path)

        self._update_sim(10)

    def _save_scene_usd(self) -> None:
        import omni.usd

        omni.usd.get_context().save_as_stage(str(self._scene_usd_path))

    def _initialize_ros_runtime(self) -> None:
        if self._sim_app is None:
            raise RuntimeError("A SimulationApp is required before the ROS runtime can be initialized.")

        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        self._update_sim(8)

        self._robot_controllers = []
        for robot_name, adapter, robot_prim_path in zip(
            self._robot_names,
            self._robot_adapters,
            self._robot_prim_paths,
        ):
            controller = adapter.initialize_runtime_controller(
                sim_app=self._sim_app,
                namespace=robot_name,
                root_prim_path=robot_prim_path,
            )
            self._robot_controllers.append(controller)

        self._ros_bridge = InternalIsaacRosBridge(
            sim_app=self._sim_app,
            robot_controllers=self._robot_controllers,
            rollout_control_topic=self._rollout_control_topic,
            rollout_reset_done_topic=self._rollout_reset_done_topic,
        )

    def _write_team_config(self) -> None:
        payload = self._build_team_config_payload()
        with self._team_config_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)

    def _write_manifest(self) -> None:
        manifest = self._build_manifest_payload()
        with self._manifest_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(manifest, stream, sort_keys=False)

    def _write_failure_snapshot(self, *, current_step: str, exception: Exception) -> None:
        try:
            self._bundle_dir.mkdir(parents=True, exist_ok=True)
            map_export_mode = getattr(self, "_map_export_mode", "bbox")
            payload = {
                "scene_id": self._scene_id,
                "seed": int(self._seed),
                "current_step": current_step,
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback": traceback.format_exc(),
                "template": {
                    "template_id": self._template.template_id,
                    "variant_id": self._template.variant_id,
                    "map_export_mode": map_export_mode,
                    "description": self._template.description,
                    "template_definition_path": str(self._template.template_config_path),
                    "shared_defaults_config_path": str(self._template.shared_defaults_config_path),
                    "preset_config_path": (
                        None
                        if self._template.preset_config_path is None
                        else str(self._template.preset_config_path)
                    ),
                    "source_template_usd_path": str(self._template.source_template_usd_path),
                    "base_environment_usd": self._template.base_environment_usd,
                    "resolved_base_environment_usd": self._resolved_base_environment_usd,
                    "environment_bbox": self._environment_bbox_dict(),
                    "metadata": dict(self._template.metadata),
                },
                "resolved_object_groups": dict(self._resolved_group_details),
                "resolved_light_selectors": dict(self._resolved_light_paths),
                "rejection_summary": dict(self._rejection_summary),
                "map_export_debug": self._map_export_debug_payload(),
                "randomization_records": list(self._randomization_records),
                "accepted_layout_bboxes": [
                    {
                        "prim_path": entry["prim_path"],
                        "bbox_min": entry["bbox_min"].tolist(),
                        "bbox_max": entry["bbox_max"].tolist(),
                        "margin_m": float(entry["margin_m"]),
                        "category": entry.get("category", ""),
                        "source_group_name": entry.get("source_group_name", ""),
                    }
                    for entry in self._accepted_layout_bboxes
                ],
                "validation_summary": self._build_validation_summary(),
            }
            with self._failure_snapshot_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(payload, stream, sort_keys=False)
        except Exception:
            pass

    def _map_result_dict(self, result: MapExportResult) -> dict[str, Any]:
        return {
            "yaml_path": str(result.yaml_path),
            "png_path": str(result.png_path),
            "resolution_m": float(result.resolution_m),
            "origin_xyz": list(result.origin_xyz),
            "min_bound_xyz": list(result.min_bound_xyz),
            "max_bound_xyz": list(result.max_bound_xyz),
            "width_px": int(result.width_px),
            "height_px": int(result.height_px),
            "occupied_cells": int(result.occupied_cells),
            "free_cells": int(result.free_cells),
            "unknown_cells": int(result.unknown_cells),
        }

    def _build_validation_summary(self) -> dict[str, Any]:
        map_export_mode = getattr(self, "_map_export_mode", "bbox")
        return {
            "scene_id": self._scene_id,
            "seed": int(self._seed),
            "template_id": self._template.template_id,
            "variant_id": self._template.variant_id,
            "map_export_mode": map_export_mode,
            "randomized_object_count": len(self._randomization_records),
            "accepted_layout_boxes": len(self._accepted_layout_bboxes),
            "resolved_object_groups": dict(self._resolved_group_details),
            "rejection_summary": dict(self._rejection_summary),
            "environment_bbox": self._environment_bbox_dict(),
            "nav2_map": None if self._nav2_map_result is None else self._map_result_dict(self._nav2_map_result),
            "mapf_map": None if self._mapf_map_result is None else self._map_result_dict(self._mapf_map_result),
            "rollout_sampling": dict(self._sampling_validation),
            "map_export_debug": self._map_export_debug_payload(),
            "robot_count": int(self._robot_count),
            "rollout_count": int(self._rollout_count),
            "ros_runtime_enabled": bool(self._enable_ros2_runtime),
        }

    def _build_team_config_payload(self) -> dict[str, Any]:
        first_robot_model = self._robot_adapters[0].model_id if self._robot_adapters else ""
        map_export_mode = getattr(self, "_map_export_mode", "bbox")
        return {
            "scene_id": self._scene_id,
            "template_id": self._template.template_id,
            "variant_id": self._template.variant_id,
            "map_export_mode": map_export_mode,
            "template_definition_path": str(self._template.template_config_path),
            "shared_defaults_config_path": str(self._template.shared_defaults_config_path),
            "preset_config_path": (
                None if self._template.preset_config_path is None else str(self._template.preset_config_path)
            ),
            "source_template_usd_path": str(self._template.source_template_usd_path),
            "seed": int(self._seed),
            "usd_path": str(self._scene_usd_path),
            "robot_model": first_robot_model,
            "language_instruction": self._language_instruction,
            "environment": {
                "nav2_map": str(self._nav2_map_path),
                "mapf_map": str(self._mapf_map_path),
            },
            "sampling_contract": {
                "goal_sampling_mode": "occupancy_map_plus_focus_object",
                "language_grounded": False,
            },
            "rollouts": self._rollouts_payload,
            "validation": self._build_validation_summary(),
        }

    def _build_manifest_payload(self) -> dict[str, Any]:
        map_export_mode = getattr(self, "_map_export_mode", "bbox")
        return {
            "scene_id": self._scene_id,
            "seed": int(self._seed),
            "template": {
                "template_id": self._template.template_id,
                "variant_id": self._template.variant_id,
                "map_export_mode": map_export_mode,
                "description": self._template.description,
                "template_definition_path": str(self._template.template_config_path),
                "shared_defaults_config_path": str(self._template.shared_defaults_config_path),
                "preset_config_path": (
                    None
                    if self._template.preset_config_path is None
                    else str(self._template.preset_config_path)
                ),
                "source_template_usd_path": str(self._template.source_template_usd_path),
                "base_environment_usd": self._template.base_environment_usd,
                "resolved_base_environment_usd": self._resolved_base_environment_usd,
                "environment_bbox": self._environment_bbox_dict(),
                "metadata": dict(self._template.metadata),
            },
            "scene_usd_path": str(self._scene_usd_path),
            "language_instruction": self._language_instruction,
            "robot_models": [adapter.model_id for adapter in self._robot_adapters],
            "robot_namespaces": list(self._robot_names),
            "focus_object": None if self._focus_object is None else self._focus_object.as_dict(),
            "resolved_object_groups": dict(self._resolved_group_details),
            "resolved_light_selectors": dict(self._resolved_light_paths),
            "rejection_summary": dict(self._rejection_summary),
            "sampling_contract": {
                "goal_sampling_mode": "occupancy_map_plus_focus_object",
                "language_grounded": False,
            },
            "randomization_records": self._randomization_records,
            "map_export_debug": self._map_export_debug_payload(),
            "maps": {
                "nav2": None if self._nav2_map_result is None else self._map_result_dict(self._nav2_map_result),
                "mapf": None if self._mapf_map_result is None else self._map_result_dict(self._mapf_map_result),
            },
            "validation_summary": self._build_validation_summary(),
            "team_config_path": str(self._team_config_path),
            "rollouts_dir": str(self._rollouts_dir),
            "ros_runtime": {
                "enabled": bool(self._enable_ros2_runtime),
                "rollout_control_topic": self._rollout_control_topic,
                "rollout_reset_done_topic": self._rollout_reset_done_topic,
            },
        }

    def _sample_rgb(
        self,
        value_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ) -> tuple[float, float, float]:
        return (
            float(self._rng.uniform(value_range[0][0], value_range[0][1])),
            float(self._rng.uniform(value_range[1][0], value_range[1][1])),
            float(self._rng.uniform(value_range[2][0], value_range[2][1])),
        )

    def _set_attr_if_present(self, prim, attribute_names: tuple[str, ...], value: Any) -> None:
        for attribute_name in attribute_names:
            attribute = prim.GetAttribute(attribute_name)
            if attribute and attribute.IsValid():
                attribute.Set(value)
                return

    def _intersects_keepout(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        *,
        keepout_zone_ids: tuple[str, ...],
        margin_m: float,
    ) -> bool:
        keepout_zone_ids = tuple(keepout_zone_ids)
        known_zone_ids = {zone.zone_id for zone in self._template.keepout_zones}
        unknown_zone_ids = [zone_id for zone_id in keepout_zone_ids if zone_id not in known_zone_ids]
        if unknown_zone_ids:
            raise KeyError(
                f"Template '{self._template.template_id}' references unknown keepout zones: "
                f"{unknown_zone_ids}"
            )
        if not keepout_zone_ids:
            keepout_zones = self._template.keepout_zones
        else:
            keepout_zones = tuple(
                zone for zone in self._template.keepout_zones if zone.zone_id in keepout_zone_ids
            )
        for zone in keepout_zones:
            if self._bbox_intersects_keepout_zone(bbox_min, bbox_max, zone, margin_m):
                return True
        return False

    def _bbox_intersects_keepout_zone(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        zone: KeepoutZone,
        margin_m: float,
    ) -> bool:
        expanded_min_x = float(zone.min_xy[0]) - float(margin_m)
        expanded_max_x = float(zone.max_xy[0]) + float(margin_m)
        expanded_min_y = float(zone.min_xy[1]) - float(margin_m)
        expanded_max_y = float(zone.max_xy[1]) + float(margin_m)
        return not (
            bbox_max[0] < expanded_min_x
            or bbox_min[0] > expanded_max_x
            or bbox_max[1] < expanded_min_y
            or bbox_min[1] > expanded_max_y
        )

    def _bbox_intersects_accepted_layout(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        *,
        margin_m: float,
    ) -> bool:
        for existing in self._accepted_layout_bboxes:
            existing_min = existing["bbox_min"]
            existing_max = existing["bbox_max"]
            expanded_margin = float(margin_m) + float(existing["margin_m"])
            if not (
                bbox_max[0] < existing_min[0] - expanded_margin
                or bbox_min[0] > existing_max[0] + expanded_margin
                or bbox_max[1] < existing_min[1] - expanded_margin
                or bbox_min[1] > existing_max[1] + expanded_margin
                or bbox_max[2] < existing_min[2] - expanded_margin
                or bbox_min[2] > existing_max[2] + expanded_margin
            ):
                return True
        return False

    def _physx_overlap_clear(
        self,
        prim_path: str,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
    ) -> bool:
        try:
            import carb
            from omni.physx import get_physx_scene_query_interface
        except Exception:
            return True

        extent_xyz = np.maximum(0.05, 0.5 * (bbox_max - bbox_min))
        origin_xyz = 0.5 * (bbox_min + bbox_max)
        hits: list[str] = []

        def _report_hit(hit) -> bool:
            hit_path = getattr(hit, "rigid_body", None) or getattr(hit, "rigidBody", None)
            if hit_path:
                hits.append(str(hit_path))
            return True

        get_physx_scene_query_interface().overlap_box(
            carb.Float3(float(extent_xyz[0]), float(extent_xyz[1]), float(extent_xyz[2])),
            carb.Float3(float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2])),
            carb.Float4(0.0, 0.0, 0.0, 1.0),
            _report_hit,
            False,
        )

        for hit_path in hits:
            hit_path_lower = hit_path.lower()
            if hit_path == prim_path or hit_path.startswith(f"{prim_path}/"):
                continue
            if any(token in hit_path_lower for token in ("ground", "collisionplane", "floor")):
                continue
            return False
        return True

    def _validate_final_layout(self) -> None:
        for accepted in self._accepted_layout_bboxes:
            bbox_min, bbox_max = compute_world_bbox(accepted["prim_path"])
            if not self._physx_overlap_clear(
                accepted["prim_path"],
                bbox_min,
                bbox_max,
            ):
                raise RuntimeError(
                    f"Final overlap validation failed for {accepted['prim_path']} after randomization."
                )

    def _ensure_physics_scene(self) -> None:
        from pxr import PhysxSchema, Sdf, UsdPhysics

        stage = get_stage()
        scene_path = Sdf.Path("/World/physicsScene")
        scene_prim = stage.GetPrimAtPath(scene_path)
        if not scene_prim or not scene_prim.IsValid():
            scene_prim = UsdPhysics.Scene.Define(stage, scene_path).GetPrim()

        physx_scene = PhysxSchema.PhysxSceneAPI.Get(stage, scene_path)
        if not physx_scene or not physx_scene.GetPrim().IsValid():
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)

        if physx_scene:
            physx_scene.CreateEnableCCDAttr(True)
            physx_scene.CreateEnableStabilizationAttr(True)
            physx_scene.CreateEnableGPUDynamicsAttr(False)
            physx_scene.CreateBroadphaseTypeAttr("MBP")
            physx_scene.CreateSolverTypeAttr("TGS")

    def _choose_map_start_location(self, map_spec) -> tuple[float, float, float]:
        candidates = self._candidate_map_start_locations(map_spec)
        return candidates[0] if candidates else (
            0.5 * (float(map_spec.min_bound_xyz[0]) + float(map_spec.max_bound_xyz[0])),
            0.5 * (float(map_spec.min_bound_xyz[1]) + float(map_spec.max_bound_xyz[1])),
            max(0.2, float(map_spec.min_bound_xyz[2]) + 0.05),
        )

    def _candidate_map_start_locations(self, map_spec: TemplateMapSpec) -> list[tuple[float, float, float]]:
        fallback = (
            0.5 * (float(map_spec.min_bound_xyz[0]) + float(map_spec.max_bound_xyz[0])),
            0.5 * (float(map_spec.min_bound_xyz[1]) + float(map_spec.max_bound_xyz[1])),
            max(0.2, float(map_spec.min_bound_xyz[2]) + 0.05),
        )
        candidate_points: list[tuple[float, float, float]] = []
        preferred_zones = [
            zone for zone in self._template.placement_zones if zone.zone_type == "support"
        ] + [
            zone for zone in self._template.placement_zones if zone.zone_type == "anchor"
        ]
        for zone in preferred_zones:
            zone_min = np.array(zone.min_xyz, dtype=float)
            zone_max = np.array(zone.max_xyz, dtype=float)
            zone_center = 0.5 * (zone_min + zone_max)
            zone_z = max(float(fallback[2]), float(zone.min_xyz[2]) + 0.05)
            candidate_points.append((float(zone_center[0]), float(zone_center[1]), zone_z))
            for scale_x, scale_y in ((0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)):
                candidate_points.append(
                    (
                        float(zone_min[0] + scale_x * (zone_max[0] - zone_min[0])),
                        float(zone_min[1] + scale_y * (zone_max[1] - zone_min[1])),
                        zone_z,
                    )
                )

        env_bbox = self._environment_bbox if self._environment_bbox_is_usable(map_spec) else None
        if env_bbox is not None:
            bbox_min, bbox_max = env_bbox
            bbox_z = max(float(fallback[2]), float(bbox_min[2]) + 0.05)
            for scale_x, scale_y in ((0.5, 0.5), (0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8)):
                candidate_points.append(
                    (
                        float(bbox_min[0] + scale_x * (bbox_max[0] - bbox_min[0])),
                        float(bbox_min[1] + scale_y * (bbox_max[1] - bbox_min[1])),
                        bbox_z,
                    )
                )

        for scale_x, scale_y in ((0.5, 0.5), (0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8)):
            candidate_points.append(
                (
                    float(map_spec.min_bound_xyz[0] + scale_x * (map_spec.max_bound_xyz[0] - map_spec.min_bound_xyz[0])),
                    float(map_spec.min_bound_xyz[1] + scale_y * (map_spec.max_bound_xyz[1] - map_spec.min_bound_xyz[1])),
                    float(fallback[2]),
                )
            )
        candidate_points.append(fallback)

        deduped_points: list[tuple[float, float, float]] = []
        seen: set[tuple[float, float, float]] = set()
        for candidate_xyz_tuple in candidate_points:
            candidate_xyz = np.array(candidate_xyz_tuple, dtype=float)
            rounded = tuple(round(float(value), 3) for value in candidate_xyz.tolist())
            if rounded in seen:
                continue
            seen.add(rounded)
            if self._point_within_map_bounds(candidate_xyz, map_spec) and self._point_is_clear_for_map_start(
                candidate_xyz,
                clearance_m=0.35,
            ):
                deduped_points.append(tuple(candidate_xyz.tolist()))
        if deduped_points:
            return deduped_points
        return [fallback]

    def _safe_environment_bbox(self) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            bbox_min, bbox_max = compute_world_bbox(self._env_prim_path)
        except Exception:
            return None
        if not (np.all(np.isfinite(bbox_min)) and np.all(np.isfinite(bbox_max))):
            return None
        if np.any((bbox_max - bbox_min) <= 0.0):
            return None
        return (bbox_min, bbox_max)

    def _resolve_map_spec(self, map_spec: TemplateMapSpec) -> TemplateMapSpec:
        if not self._environment_bbox_is_usable(map_spec):
            return map_spec
        env_bbox = self._environment_bbox
        if env_bbox is None:
            return map_spec

        bbox_min, bbox_max = env_bbox
        pad_xy = max(0.5, float(map_spec.resolution_m) * 8.0)
        min_bound_xyz = (
            min(float(map_spec.min_bound_xyz[0]), float(bbox_min[0]) - pad_xy),
            min(float(map_spec.min_bound_xyz[1]), float(bbox_min[1]) - pad_xy),
            float(map_spec.min_bound_xyz[2]),
        )
        max_bound_xyz = (
            max(float(map_spec.max_bound_xyz[0]), float(bbox_max[0]) + pad_xy),
            max(float(map_spec.max_bound_xyz[1]), float(bbox_max[1]) + pad_xy),
            max(float(map_spec.max_bound_xyz[2]), float(bbox_max[2]) + 0.1),
        )
        origin_hint_xyz = (
            float(min_bound_xyz[0]),
            float(min_bound_xyz[1]),
            float(map_spec.origin_hint_xyz[2]),
        )
        return TemplateMapSpec(
            resolution_m=float(map_spec.resolution_m),
            origin_hint_xyz=origin_hint_xyz,
            min_bound_xyz=min_bound_xyz,
            max_bound_xyz=max_bound_xyz,
        )

    def _environment_bbox_is_usable(self, map_spec: TemplateMapSpec) -> bool:
        env_bbox = self._environment_bbox
        if env_bbox is None:
            return False

        bbox_min, bbox_max = env_bbox
        bbox_span = bbox_max - bbox_min
        if not (np.all(np.isfinite(bbox_span)) and np.all(bbox_span > 0.0)):
            return False

        template_span_x = float(map_spec.max_bound_xyz[0]) - float(map_spec.min_bound_xyz[0])
        template_span_y = float(map_spec.max_bound_xyz[1]) - float(map_spec.min_bound_xyz[1])
        if bbox_span[0] > template_span_x * 1.75 + 4.0 or bbox_span[1] > template_span_y * 1.75 + 4.0:
            return False

        template_center_x = 0.5 * (float(map_spec.min_bound_xyz[0]) + float(map_spec.max_bound_xyz[0]))
        template_center_y = 0.5 * (float(map_spec.min_bound_xyz[1]) + float(map_spec.max_bound_xyz[1]))
        bbox_center_x = 0.5 * (float(bbox_min[0]) + float(bbox_max[0]))
        bbox_center_y = 0.5 * (float(bbox_min[1]) + float(bbox_max[1]))
        if abs(bbox_center_x - template_center_x) > max(2.0, template_span_x * 0.35):
            return False
        if abs(bbox_center_y - template_center_y) > max(2.0, template_span_y * 0.35):
            return False

        resolved_min_x = min(float(map_spec.min_bound_xyz[0]), float(bbox_min[0]) - 0.5)
        resolved_min_y = min(float(map_spec.min_bound_xyz[1]), float(bbox_min[1]) - 0.5)
        resolved_max_x = max(float(map_spec.max_bound_xyz[0]), float(bbox_max[0]) + 0.5)
        resolved_max_y = max(float(map_spec.max_bound_xyz[1]), float(bbox_max[1]) + 0.5)
        width_px = int(round((resolved_max_x - resolved_min_x) / float(map_spec.resolution_m)))
        height_px = int(round((resolved_max_y - resolved_min_y) / float(map_spec.resolution_m)))
        return width_px <= 2048 and height_px <= 2048

    def _point_within_map_bounds(self, point_xyz: np.ndarray, map_spec: TemplateMapSpec) -> bool:
        return (
            float(map_spec.min_bound_xyz[0]) <= float(point_xyz[0]) <= float(map_spec.max_bound_xyz[0])
            and float(map_spec.min_bound_xyz[1]) <= float(point_xyz[1]) <= float(map_spec.max_bound_xyz[1])
            and float(map_spec.min_bound_xyz[2]) <= float(point_xyz[2]) <= float(map_spec.max_bound_xyz[2]) + 1.0
        )

    def _environment_bbox_dict(self) -> dict[str, list[float]] | None:
        environment_bbox = getattr(self, "_environment_bbox", None)
        if environment_bbox is None:
            return None
        bbox_min, bbox_max = environment_bbox
        return {
            "min_xyz": bbox_min.tolist(),
            "max_xyz": bbox_max.tolist(),
        }

    def _map_export_debug_payload(self) -> dict[str, Any]:
        return dict(getattr(self, "_map_export_debug", {}) or {})

    def _map_quality_score(self, result: MapExportResult) -> tuple[float, float, int]:
        total_cells = max(1, int(result.width_px) * int(result.height_px))
        free_ratio = float(result.free_cells) / float(total_cells)
        unknown_ratio = float(result.unknown_cells) / float(total_cells)
        return (free_ratio, -unknown_ratio, int(result.free_cells))

    def _point_is_clear_for_map_start(self, point_xyz: np.ndarray, *, clearance_m: float) -> bool:
        x = float(point_xyz[0])
        y = float(point_xyz[1])
        z = float(point_xyz[2])
        margin = float(clearance_m)

        for accepted in self._accepted_layout_bboxes:
            bbox_min = accepted["bbox_min"]
            bbox_max = accepted["bbox_max"]
            if (
                bbox_min[0] - margin <= x <= bbox_max[0] + margin
                and bbox_min[1] - margin <= y <= bbox_max[1] + margin
                and bbox_min[2] - margin <= z <= bbox_max[2] + margin
            ):
                return False
        return True

    def _update_sim(self, steps: int) -> None:
        if self._sim_app is None:
            return
        for _ in range(max(1, int(steps))):
            self._sim_app.update()
