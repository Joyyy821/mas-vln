from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import itertools
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
    _export_occupancy_map,
    _export_resampled_occupancy_map,
    sample_multi_robot_rollouts,
)
from isaac_sim.stage_bringups.warehouse_randomized.instructions.forklift_near_shelf import (
    FOCUS_SELECTOR_ID,
    select_focus_object,
)
from isaac_sim.stage_bringups.warehouse_randomized.robots import (
    RuntimeRobotController,
    build_robot_adapter,
)
from isaac_sim.stage_bringups.warehouse_randomized.robot_teams import (
    DEFAULT_ROBOT_TEAM_MODE,
    RobotTeamPolicy,
    build_fixed_robot_team,
    robot_team_policy_payload,
    sample_priority_robot_team,
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

OMAP_EXTENSION_NAME = "isaacsim.asset.gen.omap"
OMAP_DEFAULT_Z_MIN = 0.1
OMAP_DEFAULT_Z_MAX = 0.62
OMAP_PHYSICS_WARMUP_UPDATES = 120
ROS_COSTMAP_PARAMS_RELS = (
    Path("ros2_ws/src/carters_goal/config/mapf_costmap_params_isaac.yaml"),
    Path("ros2_ws/src/carters_nav2/config/warehouse/multi_robot_carter_rpp_controller_only_params.yaml"),
    Path("ros2_ws/src/carters_nav2/config/warehouse/multi_robot_carter_navigation_params_1.yaml"),
)
DEFAULT_ROS_COSTMAP_INFLATION_RADIUS_M = 1.0


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
        overwrite: bool = False,
        scene_only: bool = False,
        spawn_robots_only: bool = False,
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
        self._overwrite = bool(overwrite)
        self._scene_only = bool(scene_only)
        self._spawn_robots_only = bool(spawn_robots_only)

        if self._scene_only and self._spawn_robots_only:
            raise ValueError("scene_only and spawn_robots_only cannot both be true.")

        if self._rollout_count <= 0:
            raise ValueError("rollout_count must be positive.")

        if not isinstance(template, WarehouseTemplate):
            raise TypeError("template must be a resolved WarehouseTemplate instance.")
        self._template = template

        expanded_models = [str(model).strip() for model in robot_models if str(model).strip()]
        self._robot_team_policy: RobotTeamPolicy = template.robot_team
        self._fixed_robot_team: list[dict[str, str]] | None = None
        if expanded_models or self._robot_count > 0:
            if self._robot_count <= 0:
                self._robot_count = len(expanded_models)
            if self._robot_count < 2 or self._robot_count > 5:
                raise ValueError(f"robot_count must be between 2 and 5, got {self._robot_count}.")
            self._fixed_robot_team = build_fixed_robot_team(
                model_ids=expanded_models,
                robot_count=self._robot_count,
            )
            self._robot_names = [member["name"] for member in self._fixed_robot_team]
            self._robot_adapters = [
                build_robot_adapter(member["model"]) for member in self._fixed_robot_team
            ]
        else:
            self._robot_count = 0
            self._robot_names = []
            self._robot_adapters = [
                build_robot_adapter(model_id) for model_id in self._robot_team_policy.model_priority
            ]

        self._sim_app = sim_app
        self._bundle_dir = self._scene_root_dir / self._scene_id
        self._scene_usd_path = self._bundle_dir / "scene.usd"
        self._manifest_path = self._bundle_dir / "scene_manifest.yaml"
        self._team_config_path = self._bundle_dir / "team_config.yaml"
        self._failure_snapshot_path = self._bundle_dir / "build_failure_snapshot.yaml"
        self._rollouts_dir = self._bundle_dir / "rollouts"
        self._nav2_map_path = self._bundle_dir / "nav2_map.yaml"
        self._mapf_map_path = self._bundle_dir / "mapf_map.yaml"
        self._collection_metadata_path = self._scene_root_dir / "collection_metadata.yaml"

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
        self._focus_selection_debug: dict[str, Any] = {}
        self._rollouts_payload: list[dict[str, Any]] = []
        self._rollout_robot_teams: list[list[dict[str, str]]] = []
        self._sampling_validation: dict[str, Any] = {}
        self._robot_prim_paths: list[str] = []
        self._robot_controllers: list[RuntimeRobotController] = []
        self._ros_bridge: InternalIsaacRosBridge | None = None
        self._environment_bbox: tuple[np.ndarray, np.ndarray] | None = None
        self._resolved_nav2_map_spec: TemplateMapSpec | None = None
        self._resolved_mapf_map_spec: TemplateMapSpec | None = None
        self._map_export_debug: dict[str, Any] = {}
        self._omap_extension_enabled = False
        self._omap_physics_warmup_updates = 0

    @property
    def sim_app(self):
        return self._sim_app

    def ensure_sim_app(self, *, headless: bool) -> None:
        if self._sim_app is not None:
            return
        self._sim_app = maybe_start_sim_app(
            headless=headless,
            enable_ros2_bridge=self._enable_ros2_runtime,
            extra_extensions=(OMAP_EXTENSION_NAME,),
        )

    def _robot_team_policy_payload(self) -> dict[str, Any]:
        return robot_team_policy_payload(getattr(self, "_robot_team_policy", self._template.robot_team))

    def _active_robot_team_mode(self) -> str:
        if getattr(self, "_fixed_robot_team", None) is not None:
            return "fixed_robot_team_debug"
        return getattr(self, "_robot_team_policy", self._template.robot_team).mode

    def _activate_robot_team(self, robot_team: list[dict[str, str]]) -> None:
        self._robot_names = [str(member["name"]) for member in robot_team]
        self._robot_adapters = [build_robot_adapter(member["model"]) for member in robot_team]
        self._robot_count = len(robot_team)

    def _rollout_team_from_robots(
        self,
        robots: list[dict[str, Any]],
        *,
        fallback_models: list[str] | None = None,
        fallback_model: str = "nova_carter",
    ) -> list[dict[str, str]]:
        fallback_models = list(fallback_models or [])
        team: list[dict[str, str]] = []
        for index, robot in enumerate(robots):
            model = str(robot.get("model", "") or "").strip()
            if not model and index < len(fallback_models):
                model = str(fallback_models[index]).strip()
            if not model:
                model = fallback_model
            name = str(robot.get("name", "") or "").strip() or model
            team.append({"name": name, "model": model})
            robot["name"] = name
            robot["model"] = model
        return team

    def _sample_rollout_robot_teams(self) -> list[list[dict[str, str]]]:
        if self._fixed_robot_team is not None:
            return [
                [dict(member) for member in self._fixed_robot_team]
                for _ in range(self._rollout_count)
            ]
        return [
            sample_priority_robot_team(policy=self._robot_team_policy, rng=self._rng)
            for _ in range(self._rollout_count)
        ]

    def _robot_model_ids_for_manifest(self) -> list[str]:
        fixed_team = getattr(self, "_fixed_robot_team", None)
        if fixed_team is not None:
            return [member["model"] for member in fixed_team]
        policy = getattr(self, "_robot_team_policy", self._template.robot_team)
        return list(policy.model_priority)

    def _prim_name_for_robot(self, *, robot_name: str, model_id: str, index: int) -> str:
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(robot_name).strip())
        clean = clean.strip("_")
        if clean:
            return clean
        return f"{model_id}_{int(index)}"

    def build(self) -> RandomizedWarehouseBuildResult:
        if self._spawn_robots_only:
            return self._build_spawn_robots_only()

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
            current_step = "save_scene_usd"
            self._save_scene_usd()
            if not self._scene_only:
                current_step = "sample_rollouts"
                self._sample_rollouts()
                if self._enable_ros2_runtime:
                    current_step = "spawn_robots"
                    self._spawn_robots()
            current_step = "write_team_config"
            if not self._scene_only:
                self._write_team_config()
            current_step = "write_manifest"
            self._write_manifest()
            if self._enable_ros2_runtime and not self._scene_only:
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

    def _build_spawn_robots_only(self) -> RandomizedWarehouseBuildResult:
        current_step = "prepare_spawn_robots_only_bundle"
        try:
            self._prepare_spawn_robots_only_bundle()
            current_step = "resolve_assets_root"
            self._resolve_assets_root()
            current_step = "open_existing_scene_usd"
            self._open_existing_scene_usd()
            current_step = "resolve_focus_object"
            self._resolve_focus_object()
            using_existing_team_config = self._team_config_path.exists() and not self._overwrite
            if using_existing_team_config:
                current_step = "load_existing_team_config"
                self._load_existing_team_config()
            else:
                current_step = "sample_rollouts"
                self._sample_rollouts()
            current_step = "spawn_robots"
            self._spawn_robots()
            if not using_existing_team_config:
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
                f"Randomized warehouse spawn-robots-only failed during step '{current_step}' "
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
        self._write_collection_metadata()
        if self._failure_snapshot_path.exists():
            try:
                self._failure_snapshot_path.unlink()
            except Exception:
                pass

    def _prepare_spawn_robots_only_bundle(self) -> None:
        if not self._bundle_dir.exists():
            raise FileNotFoundError(
                f"Scene bundle does not exist for spawn-robots-only mode: {self._bundle_dir}"
            )
        if not self._scene_usd_path.exists():
            raise FileNotFoundError(f"Scene USD does not exist: {self._scene_usd_path}")
        if not self._nav2_map_path.exists():
            raise FileNotFoundError(f"Nav2 map does not exist: {self._nav2_map_path}")
        self._rollouts_dir.mkdir(parents=True, exist_ok=True)
        self._write_collection_metadata()
        if self._failure_snapshot_path.exists():
            try:
                self._failure_snapshot_path.unlink()
            except Exception:
                pass

    def _build_collection_metadata_payload(self) -> dict[str, Any]:
        return {
            "language_instruction": self._language_instruction,
            "focus_selector": FOCUS_SELECTOR_ID,
            "robot_team_mode": DEFAULT_ROBOT_TEAM_MODE,
        }

    def _write_collection_metadata(self) -> None:
        self._collection_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self._collection_metadata_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self._build_collection_metadata_payload(), stream, sort_keys=False)

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

    def _assign_unique_anchor_zones(
        self,
        *,
        object_positions_xy: dict[str, np.ndarray],
        candidate_zones: list[PlacementZone],
    ) -> dict[str, PlacementZone]:
        if not object_positions_xy or len(candidate_zones) < len(object_positions_xy):
            return {}

        object_paths = list(object_positions_xy)
        best_assignment: dict[str, PlacementZone] = {}
        best_cost = float("inf")
        for zone_tuple in itertools.permutations(candidate_zones, len(object_paths)):
            total_cost = 0.0
            for object_path, zone in zip(object_paths, zone_tuple):
                delta_xy = self._zone_center_xy(zone) - np.array(object_positions_xy[object_path][:2], dtype=float)
                total_cost += float(np.sum(delta_xy ** 2))
            if total_cost < best_cost:
                best_cost = total_cost
                best_assignment = {
                    object_path: zone
                    for object_path, zone in zip(object_paths, zone_tuple)
                }
        return best_assignment

    def _zone_center_xy(self, zone: PlacementZone) -> np.ndarray:
        return np.array(
            [
                0.5 * (float(zone.min_xyz[0]) + float(zone.max_xyz[0])),
                0.5 * (float(zone.min_xyz[1]) + float(zone.max_xyz[1])),
            ],
            dtype=float,
        )

    def _ordered_candidate_zones_for_object(
        self,
        *,
        original_position: np.ndarray,
        candidate_zones: list[PlacementZone],
        reserved_zone_ids: set[str] | None = None,
        prefer_unused: bool = False,
    ) -> list[PlacementZone]:
        reserved_zone_ids = set(reserved_zone_ids or set())
        original_xy = np.array(original_position[:2], dtype=float)

        def _sort_key(zone: PlacementZone) -> tuple[int, float, str]:
            center_xy = self._zone_center_xy(zone)
            distance_sq = float(np.sum((center_xy - original_xy) ** 2))
            is_reserved = int(bool(prefer_unused and zone.zone_id in reserved_zone_ids))
            return (is_reserved, distance_sq, zone.zone_id)

        return sorted(candidate_zones, key=_sort_key)

    def _nearest_zone_id_for_position(
        self,
        *,
        position_xyz: np.ndarray,
        candidate_zones: list[PlacementZone],
    ) -> str:
        if not candidate_zones:
            return ""
        original_xy = np.array(position_xyz[:2], dtype=float)
        nearest_zone = min(
            candidate_zones,
            key=lambda zone: float(np.sum((self._zone_center_xy(zone) - original_xy) ** 2)),
        )
        return nearest_zone.zone_id

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
        stage = get_stage()
        duplicate_prim = getattr(omni.usd, "duplicate_prim", None)
        if callable(duplicate_prim):
            try:
                duplicate_prim(stage, source_prim_path, dest_prim_path)
            except Exception:
                pass
            else:
                copied_prim = stage.GetPrimAtPath(dest_prim_path)
                if copied_prim and copied_prim.IsValid():
                    return

        try:
            try:
                omni.kit.commands.execute(
                    "CopyPrim",
                    path_from=source_prim_path,
                    path_to=dest_prim_path,
                    select_new_prim=False,
                )
            except Exception:
                omni.kit.commands.execute(
                    "CopyPrim",
                    path_from=source_prim_path,
                    path_to=dest_prim_path,
                )
        except Exception:
            if callable(duplicate_prim):
                duplicate_prim(stage, source_prim_path, dest_prim_path)

        copied_prim = stage.GetPrimAtPath(dest_prim_path)
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
        original_positions_by_path: dict[str, np.ndarray] = {}
        original_orientations_by_path: dict[str, np.ndarray] = {}
        original_yaws_by_path: dict[str, float] = {}
        for object_path in object_paths:
            original_position, original_orientation = get_world_pose_xyzw(object_path)
            original_positions_by_path[object_path] = original_position
            original_orientations_by_path[object_path] = original_orientation
            original_yaws_by_path[object_path] = (
                quaternion_xyzw_to_yaw(original_orientation) * 180.0 / math.pi
            )

        assigned_anchor_zones: dict[str, PlacementZone] = {}
        if spec.anchor_zone_ids:
            assigned_anchor_zones = self._assign_unique_anchor_zones(
                object_positions_xy=original_positions_by_path,
                candidate_zones=candidate_zones,
            )
        reserve_unique_zones = bool(spec.anchor_zone_ids) and len(candidate_zones) >= len(object_paths) > 0
        reserved_zone_ids: set[str] = set()
        for object_path in object_paths:
            original_position = original_positions_by_path[object_path]
            original_orientation = original_orientations_by_path[object_path]
            original_yaw_deg = original_yaws_by_path[object_path]
            accepted_bbox: tuple[np.ndarray, np.ndarray] | None = None
            accepted_position = original_position.copy()
            accepted_yaw_deg = original_yaw_deg
            accepted_scale = 1.0
            accepted_zone_id = ""
            accepted_from_original = True
            if assigned_anchor_zones:
                ordered_candidate_zones = [assigned_anchor_zones[object_path]]
            else:
                ordered_candidate_zones = self._ordered_candidate_zones_for_object(
                    original_position=original_position,
                    candidate_zones=candidate_zones,
                    reserved_zone_ids=reserved_zone_ids,
                    prefer_unused=reserve_unique_zones,
                )

            for attempt_index in range(spec.max_attempts):
                candidate_yaw_deg = self._sample_candidate_yaw_deg(original_yaw_deg, spec)
                candidate_scale = float(
                    self._rng.uniform(spec.uniform_scale_range[0], spec.uniform_scale_range[1])
                )
                zone_id = ""

                if ordered_candidate_zones:
                    zone = ordered_candidate_zones[attempt_index % len(ordered_candidate_zones)]
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
                if self._bbox_intersects_environment_obstacles(
                    object_path,
                    bbox_min,
                    bbox_max,
                    margin_m=spec.collision_margin_m,
                ):
                    self._record_rejection(spec.name, "environment_overlap")
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

            if accepted_bbox is None and ordered_candidate_zones:
                self._record_rejection(spec.name, "unplaced_in_zone")
                if spec.required:
                    set_xform_pose(
                        object_path,
                        tuple(original_position.tolist()),
                        yaw_deg=original_yaw_deg,
                    )
                    raise RuntimeError(
                        f"Required object '{object_path}' from randomizer '{spec.name}' "
                        "could not be placed inside its configured warehouse zones."
                    )
                self._remove_prim(object_path)
                self._randomization_records.append(
                    {
                        "category": "layout_skip",
                        "name": spec.name,
                        "prim_path": object_path,
                        "policy": spec.policy,
                        "target_group_name": spec.target_group_name,
                        "reason": "unplaced_in_zone",
                        "candidate_zone_ids": [
                            zone.zone_id for zone in ordered_candidate_zones
                        ],
                    }
                )
                continue

            if not accepted_zone_id and ordered_candidate_zones and reserve_unique_zones:
                accepted_zone_id = self._nearest_zone_id_for_position(
                    position_xyz=accepted_position,
                    candidate_zones=ordered_candidate_zones,
                )
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
            ) or self._bbox_intersects_environment_obstacles(
                object_path,
                bbox_min,
                bbox_max,
                margin_m=spec.collision_margin_m,
            ) or not self._physx_overlap_clear(object_path, bbox_min, bbox_max):
                raise RuntimeError(
                    f"Object '{object_path}' from randomizer '{spec.name}' could not be placed safely."
                )
            if reserve_unique_zones and accepted_zone_id:
                reserved_zone_ids.add(accepted_zone_id)
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
                if self._bbox_intersects_environment_obstacles(
                    prop_path,
                    bbox_min,
                    bbox_max,
                    margin_m=spec.collision_margin_m,
                ):
                    self._record_rejection(spec.name, "environment_overlap")
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
        candidates: list[ObjectBBox3D] = []
        for focus_path in focus_paths:
            bbox_min, bbox_max = compute_world_bbox(focus_path)
            candidates.append(
                ObjectBBox3D(
                    prim_path=focus_path,
                    min_xyz=bbox_min,
                    max_xyz=bbox_max,
                )
            )
        self._focus_object, self._focus_selection_debug = select_focus_object(candidates)

    def _export_maps(self) -> None:
        self._enable_omap_extension()
        self._ensure_physics_scene()
        self._resolved_nav2_map_spec = self._template.nav2_map
        self._resolved_mapf_map_spec = self._template.mapf_map
        self._initialize_physics_for_omap()
        self._nav2_map_result = self._export_omap_map_from_spec(
            yaml_path=self._nav2_map_path,
            map_spec=self._resolved_nav2_map_spec,
        )
        if float(self._resolved_mapf_map_spec.resolution_m) >= float(self._nav2_map_result.resolution_m):
            self._mapf_map_result = self._export_resampled_omap_map_from_source(
                yaml_path=self._mapf_map_path,
                map_spec=self._resolved_mapf_map_spec,
                source_result=self._nav2_map_result,
            )
        else:
            self._mapf_map_result = self._export_omap_map_from_spec(
                yaml_path=self._mapf_map_path,
                map_spec=self._resolved_mapf_map_spec,
            )

    def _enable_omap_extension(self) -> None:
        try:
            from isaacsim.core.utils.extensions import enable_extension
        except Exception:
            from omni.isaac.core.utils.extensions import enable_extension

        enable_extension(OMAP_EXTENSION_NAME)
        self._omap_extension_enabled = True
        if self._sim_app is not None:
            self._sim_app.update()

    def _initialize_physics_for_omap(self, num_updates: int = OMAP_PHYSICS_WARMUP_UPDATES) -> None:
        import omni.kit.app
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        app = omni.kit.app.get_app()
        was_playing = timeline.is_playing()
        timeline.play()
        for _ in range(max(1, int(num_updates))):
            if self._sim_app is not None:
                self._sim_app.update()
            else:
                app.update()
        if not was_playing:
            pause_timeline = getattr(timeline, "pause", None)
            if callable(pause_timeline):
                pause_timeline()
            else:
                timeline.stop()
        self._omap_physics_warmup_updates = max(1, int(num_updates))

    def _omap_z_range(self, map_spec: TemplateMapSpec) -> tuple[float, float]:
        if map_spec.occupancy_z_range_m is None:
            return (OMAP_DEFAULT_Z_MIN, OMAP_DEFAULT_Z_MAX)
        return (
            float(map_spec.occupancy_z_range_m[0]),
            float(map_spec.occupancy_z_range_m[1]),
        )

    def _export_omap_map_from_spec(
        self,
        *,
        yaml_path: str | Path,
        map_spec: TemplateMapSpec,
    ) -> MapExportResult:
        z_min, z_max = self._omap_z_range(map_spec)
        base_debug_payload: dict[str, Any] = {
            "rasterization_mode": "omap_generator",
            "health_status": "running",
            "omap_extension_name": OMAP_EXTENSION_NAME,
            "omap_extension_enabled": bool(getattr(self, "_omap_extension_enabled", False)),
            "physics_warmup_updates": int(getattr(self, "_omap_physics_warmup_updates", 0)),
            "env_prim_path": self._env_prim_path,
            "resolution_m": float(map_spec.resolution_m),
            "z_min": float(z_min),
            "z_max": float(z_max),
            "omap_origin_xyz": [0.0, 0.0, 0.0],
            "bounds_source": "environment_bbox",
        }
        try:
            result = _export_occupancy_map(
                env_prim_path=self._env_prim_path,
                yaml_path=yaml_path,
                resolution_m=float(map_spec.resolution_m),
                origin_xyz=(0.0, 0.0, 0.0),
                z_min=z_min,
                z_max=z_max,
                rotate_180=True,
            )
        except Exception as exc:
            self._map_export_debug[str(yaml_path)] = {
                **base_debug_payload,
                "health_status": "failed",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            }
            raise

        self._map_export_debug[str(yaml_path)] = {
            **base_debug_payload,
            **dict(result.debug),
            "health_status": "ok",
            "quality_score": list(self._map_quality_score(result)),
        }
        return result

    def _export_resampled_omap_map_from_source(
        self,
        *,
        yaml_path: str | Path,
        map_spec: TemplateMapSpec,
        source_result: MapExportResult,
    ) -> MapExportResult:
        z_min, z_max = self._omap_z_range(map_spec)
        base_debug_payload: dict[str, Any] = {
            "rasterization_mode": "omap_generator_resampled",
            "source_rasterization_mode": "omap_generator",
            "health_status": "running",
            "omap_extension_name": OMAP_EXTENSION_NAME,
            "omap_extension_enabled": bool(getattr(self, "_omap_extension_enabled", False)),
            "physics_warmup_updates": int(getattr(self, "_omap_physics_warmup_updates", 0)),
            "env_prim_path": self._env_prim_path,
            "resolution_m": float(map_spec.resolution_m),
            "z_min": float(z_min),
            "z_max": float(z_max),
            "bounds_source": "environment_bbox",
        }
        try:
            result = _export_resampled_occupancy_map(
                yaml_path=yaml_path,
                source_result=source_result,
                resolution_m=float(map_spec.resolution_m),
            )
        except Exception as exc:
            self._map_export_debug[str(yaml_path)] = {
                **base_debug_payload,
                "health_status": "failed",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            }
            raise

        self._map_export_debug[str(yaml_path)] = {
            **base_debug_payload,
            **dict(result.debug),
            "health_status": "ok",
            "quality_score": list(self._map_quality_score(result)),
        }
        return result

    def _map_planar_area_m2(self) -> float:
        map_spec = self._resolved_nav2_map_spec or self._template.nav2_map
        span_x = float(map_spec.max_bound_xyz[0]) - float(map_spec.min_bound_xyz[0])
        span_y = float(map_spec.max_bound_xyz[1]) - float(map_spec.min_bound_xyz[1])
        return max(1.0, span_x * span_y)

    def _obstacle_root_area_limit_m2(self) -> float:
        return max(24.0, self._map_planar_area_m2() * 0.20)

    def _bbox_intersects_occupancy_z_range(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        *,
        occupancy_z_range_m: tuple[float, float] | None,
    ) -> bool:
        if occupancy_z_range_m is None:
            return True
        return not (
            float(bbox_max[2]) <= float(occupancy_z_range_m[0])
            or float(bbox_min[2]) >= float(occupancy_z_range_m[1])
        )

    def _is_ignored_obstacle_path(self, prim_path: str) -> bool:
        path_lower = str(prim_path).lower()
        return any(
            token in path_lower
            for token in (
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
        )

    def _path_matches_any_subtree(
        self,
        prim_path: str,
        root_paths: set[str] | None,
    ) -> bool:
        if not root_paths:
            return False
        clean_path = str(prim_path).strip()
        for root_path in root_paths:
            clean_root = str(root_path).strip()
            if not clean_root:
                continue
            if clean_path == clean_root or clean_path.startswith(f"{clean_root}/"):
                return True
        return False

    def _bbox_is_valid_obstacle_root(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        *,
        max_planar_area_m2: float,
    ) -> bool:
        extent_xyz = bbox_max - bbox_min
        if not (np.all(np.isfinite(bbox_min)) and np.all(np.isfinite(bbox_max))):
            return False
        if np.any(extent_xyz <= 0.0):
            return False
        if float(extent_xyz[2]) < 0.02:
            return False
        if float(extent_xyz[0]) < 0.02 and float(extent_xyz[1]) < 0.02:
            return False
        return float(extent_xyz[0] * extent_xyz[1]) <= float(max_planar_area_m2)

    def _canonicalize_obstacle_root_path(
        self,
        prim_path: str,
        *,
        max_planar_area_m2: float,
    ) -> str:
        from pxr import UsdGeom

        stage = get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return prim_path

        candidate_path = prim_path
        current = prim
        if not (current.IsA(UsdGeom.Xform) or current.GetTypeName() in {"Xform", "Scope"}):
            current = prim.GetParent()

        while current and current.IsValid():
            current_path = current.GetPath().pathString
            if current_path in {"/", self._env_prim_path}:
                break
            if self._is_ignored_obstacle_path(current_path):
                break
            if not (current.IsA(UsdGeom.Xform) or current.GetTypeName() in {"Xform", "Scope"}):
                current = current.GetParent()
                continue
            try:
                bbox_min, bbox_max = compute_world_bbox(current_path)
            except Exception:
                break
            if not self._bbox_is_valid_obstacle_root(
                bbox_min,
                bbox_max,
                max_planar_area_m2=max_planar_area_m2,
            ):
                break
            candidate_path = current_path
            current = current.GetParent()
        return candidate_path

    def _collect_environment_obstacle_root_bboxes(
        self,
        *,
        occupancy_z_range_m: tuple[float, float] | None = None,
        exclude_root_paths: set[str] | None = None,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
        from pxr import Usd, UsdGeom

        try:
            from pxr import UsdPhysics
        except Exception:
            UsdPhysics = None

        exclude_paths = {str(path).strip() for path in (exclude_root_paths or set()) if str(path).strip()}
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ],
        )
        max_planar_area_m2 = self._obstacle_root_area_limit_m2()
        search_roots = [self._env_prim_path, self._randomized_props_root]
        stats: dict[str, Any] = {
            "obstacle_source": "environment_scan",
            "collision_gprim_considered_count": 0,
            "collision_root_count": 0,
            "fallback_render_gprim_considered_count": 0,
            "fallback_render_root_count": 0,
            "skipped_invalid_bbox_count": 0,
            "skipped_elevated_gprim_count": 0,
            "skipped_large_root_count": 0,
            "skipped_ignored_path_count": 0,
            "skipped_excluded_subtree_count": 0,
        }

        def _collect(require_collision_api: bool, *, gprim_key: str, root_key: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            roots: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for prim in get_stage().Traverse():
                prim_path = prim.GetPath().pathString
                if not any(
                    prim_path == search_root or prim_path.startswith(f"{search_root}/")
                    for search_root in search_roots
                ):
                    continue
                if self._path_matches_any_subtree(prim_path, exclude_paths):
                    stats["skipped_excluded_subtree_count"] += 1
                    continue
                if not prim or not prim.IsValid() or not prim.IsActive():
                    continue
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                if require_collision_api and (UsdPhysics is None or not prim.HasAPI(UsdPhysics.CollisionAPI)):
                    continue
                if self._is_ignored_obstacle_path(prim_path):
                    stats["skipped_ignored_path_count"] += 1
                    continue

                stats[gprim_key] = int(stats.get(gprim_key, 0)) + 1
                world_bound = bbox_cache.ComputeWorldBound(prim)
                aligned = world_bound.ComputeAlignedBox()
                bbox_min = np.array([float(value) for value in aligned.GetMin()], dtype=float)
                bbox_max = np.array([float(value) for value in aligned.GetMax()], dtype=float)
                if not (np.all(np.isfinite(bbox_min)) and np.all(np.isfinite(bbox_max))):
                    stats["skipped_invalid_bbox_count"] += 1
                    continue
                if not self._bbox_intersects_occupancy_z_range(
                    bbox_min,
                    bbox_max,
                    occupancy_z_range_m=occupancy_z_range_m,
                ):
                    stats["skipped_elevated_gprim_count"] += 1
                    continue

                root_path = self._canonicalize_obstacle_root_path(
                    prim_path,
                    max_planar_area_m2=max_planar_area_m2,
                )
                if self._path_matches_any_subtree(root_path, exclude_paths):
                    stats["skipped_excluded_subtree_count"] += 1
                    continue
                if self._is_ignored_obstacle_path(root_path):
                    continue
                if root_path in roots:
                    continue
                try:
                    root_bbox_min, root_bbox_max = compute_world_bbox(root_path)
                except Exception:
                    stats["skipped_invalid_bbox_count"] += 1
                    continue
                if not self._bbox_is_valid_obstacle_root(
                    root_bbox_min,
                    root_bbox_max,
                    max_planar_area_m2=max_planar_area_m2,
                ):
                    stats["skipped_large_root_count"] += 1
                    continue
                if not self._bbox_intersects_occupancy_z_range(
                    np.asarray(root_bbox_min, dtype=float),
                    np.asarray(root_bbox_max, dtype=float),
                    occupancy_z_range_m=occupancy_z_range_m,
                ):
                    continue
                roots[root_path] = (
                    np.array(root_bbox_min, dtype=float),
                    np.array(root_bbox_max, dtype=float),
                )
                stats[root_key] = int(stats.get(root_key, 0)) + 1
            return roots

        roots = _collect(
            require_collision_api=True,
            gprim_key="collision_gprim_considered_count",
            root_key="collision_root_count",
        )
        if not roots:
            roots = _collect(
                require_collision_api=False,
                gprim_key="fallback_render_gprim_considered_count",
                root_key="fallback_render_root_count",
            )
        return roots, stats

    def _sample_rollouts(self) -> None:
        occupancy_map = OccupancyMap.load(str(self._nav2_map_path))
        focus_xy = None if self._focus_object is None else self._focus_object.center_xyz[:2]
        self._rollout_robot_teams = self._sample_rollout_robot_teams()
        team_model_ids = sorted(
            {
                member["model"]
                for team in self._rollout_robot_teams
                for member in team
                if str(member.get("model", "")).strip()
            }
        )
        team_adapters = [build_robot_adapter(model_id) for model_id in team_model_ids]
        max_planar_radius_m = max(
            float(adapter.default_planar_radius_m) for adapter in team_adapters
        )
        robot_padding_inflation_radius_m = max_planar_radius_m + 0.10
        ros_costmap_inflation_radius_m = self._ros_costmap_inflation_radius_m()
        sampling_inflation_radius_m = max(
            robot_padding_inflation_radius_m,
            ros_costmap_inflation_radius_m,
        )
        self._rollouts_payload, self._sampling_validation = sample_multi_robot_rollouts(
            occupancy_map=occupancy_map,
            rollout_count=self._rollout_count,
            rng=self._rng,
            inflation_radius_m=sampling_inflation_radius_m,
            min_pairwise_distance_m=max(1.5, max_planar_radius_m * 2.5),
            min_goal_distance_m=3.0,
            robot_teams=self._rollout_robot_teams,
            focus_xy=focus_xy,
            focus_distance_range_m=self._template.focus_distance_range_m,
        )
        if self._rollout_robot_teams:
            self._activate_robot_team(self._rollout_robot_teams[0])
        self._sampling_validation.update(
            {
                "inflation_source": "max_robot_padding_and_ros_costmaps",
                "robot_padding_inflation_radius_m": float(robot_padding_inflation_radius_m),
                "ros_costmap_inflation_radius_m": float(ros_costmap_inflation_radius_m),
                "ros_costmap_params_paths": [
                    str(self._repo_root / relative_path)
                    for relative_path in ROS_COSTMAP_PARAMS_RELS
                ],
                "robot_team_mode": self._active_robot_team_mode(),
                "robot_team_policy": self._robot_team_policy_payload(),
            }
        )

    def _load_existing_team_config(self) -> None:
        with self._team_config_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        rollouts = list(payload.get("rollouts", []) or [])
        if not rollouts:
            raise RuntimeError(f"Existing team config has no rollouts: {self._team_config_path}")

        first_robots = list(rollouts[0].get("robots", []) or [])
        if not first_robots:
            raise RuntimeError(
                f"Existing team config first rollout has no robots: {self._team_config_path}"
            )

        robot_models = [
            str(model).strip()
            for model in (payload.get("robot_models", []) or [])
            if str(model).strip()
        ]
        if not robot_models:
            robot_model = str(payload.get("robot_model", "nova_carter") or "nova_carter").strip()
            robot_models = [robot_model]
        if len(robot_models) == 1:
            max_robots = max(len(list(rollout.get("robots", []) or [])) for rollout in rollouts)
            robot_models = robot_models * max_robots

        self._rollout_robot_teams = []
        for rollout in rollouts:
            robots = list(rollout.get("robots", []) or [])
            if not robots:
                raise RuntimeError(
                    f"Existing team config rollout {rollout.get('id', '<unknown>')} has no robots: "
                    f"{self._team_config_path}"
                )
            self._rollout_robot_teams.append(
                self._rollout_team_from_robots(
                    robots,
                    fallback_models=robot_models,
                    fallback_model=str(payload.get("robot_model", "nova_carter") or "nova_carter"),
                )
            )

        self._activate_robot_team(self._rollout_robot_teams[0])
        self._rollout_count = len(rollouts)
        self._rollouts_payload = rollouts
        validation = payload.get("validation", {}) or {}
        self._sampling_validation = dict(validation.get("rollout_sampling", {}) or {})
        language_instruction = str(payload.get("language_instruction", "") or "").strip()
        if language_instruction:
            self._language_instruction = language_instruction

    def _spawn_robots(self) -> None:
        if not self._rollouts_payload:
            raise RuntimeError("No rollouts were generated before robot spawning.")
        stage = get_stage()
        robots_root = stage.GetPrimAtPath(self._robots_root)
        if not robots_root or not robots_root.IsValid():
            define_xform(self._robots_root)
            robots_root = stage.GetPrimAtPath(self._robots_root)
        for child in list(robots_root.GetChildren()):
            stage.RemovePrim(child.GetPath())
        self._robot_prim_paths = []

        first_rollout = self._rollouts_payload[0]
        first_robots = list(first_rollout.get("robots", []) or [])
        first_team = self._rollout_team_from_robots(first_robots)
        self._activate_robot_team(first_team)
        for index, robot_config in enumerate(first_robots, start=1):
            adapter = build_robot_adapter(robot_config["model"])
            robot_prim_path = (
                f"{self._robots_root}/"
                f"{self._prim_name_for_robot(robot_name=robot_config['name'], model_id=adapter.model_id, index=index)}"
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

    def _open_existing_scene_usd(self) -> None:
        import omni.usd

        context = omni.usd.get_context()
        opened = context.open_stage(str(self._scene_usd_path))
        if opened is False:
            raise RuntimeError(f"Failed to open scene USD: {self._scene_usd_path}")
        self._update_sim(12)
        self._environment_bbox = self._safe_environment_bbox()

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
                "collection_metadata_path": self._collection_metadata_path_string(),
                "resolved_object_groups": dict(self._resolved_group_details),
                "resolved_light_selectors": dict(self._resolved_light_paths),
                "focus_selection": self._focus_selection_debug_payload(),
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
        return {
            "scene_id": self._scene_id,
            "seed": int(self._seed),
            "template_id": self._template.template_id,
            "variant_id": self._template.variant_id,
            "randomized_object_count": len(self._randomization_records),
            "accepted_layout_boxes": len(self._accepted_layout_bboxes),
            "resolved_object_groups": dict(self._resolved_group_details),
            "focus_selection": self._focus_selection_debug_payload(),
            "rejection_summary": dict(self._rejection_summary),
            "environment_bbox": self._environment_bbox_dict(),
            "nav2_map": None if self._nav2_map_result is None else self._map_result_dict(self._nav2_map_result),
            "mapf_map": None if self._mapf_map_result is None else self._map_result_dict(self._mapf_map_result),
            "rollout_sampling": dict(self._sampling_validation),
            "map_export_debug": self._map_export_debug_payload(),
            "robot_count": int(self._robot_count),
            "rollout_robot_counts": [
                len(list(rollout.get("robots", []) or []))
                for rollout in self._rollouts_payload
            ],
            "robot_team_mode": self._active_robot_team_mode(),
            "robot_team_policy": self._robot_team_policy_payload(),
            "rollout_count": int(self._rollout_count),
            "ros_runtime_enabled": bool(self._enable_ros2_runtime),
        }

    def _build_team_config_payload(self) -> dict[str, Any]:
        first_robot_model = self._robot_adapters[0].model_id if self._robot_adapters else ""
        return {
            "scene_id": self._scene_id,
            "template_id": self._template.template_id,
            "variant_id": self._template.variant_id,
            "template_definition_path": str(self._template.template_config_path),
            "shared_defaults_config_path": str(self._template.shared_defaults_config_path),
            "collection_metadata_path": self._collection_metadata_path_string(),
            "preset_config_path": (
                None if self._template.preset_config_path is None else str(self._template.preset_config_path)
            ),
            "source_template_usd_path": str(self._template.source_template_usd_path),
            "seed": int(self._seed),
            "usd_path": str(self._scene_usd_path),
            "robot_model": first_robot_model,
            "robot_models": self._robot_model_ids_for_manifest(),
            "robot_team_mode": self._active_robot_team_mode(),
            "robot_team_policy": self._robot_team_policy_payload(),
            "language_instruction": self._language_instruction,
            "environment": {
                "nav2_map": self._bundle_relative_path(self._nav2_map_path),
                "mapf_map": self._bundle_relative_path(self._mapf_map_path),
            },
            "sampling_contract": {
                "goal_sampling_mode": "occupancy_map_plus_focus_object",
                "focus_selector": FOCUS_SELECTOR_ID,
                "language_grounded": True,
            },
            "rollouts": self._rollouts_payload,
            "validation": self._build_validation_summary(),
        }

    def _bundle_relative_path(self, path: Path) -> str:
        try:
            return path.relative_to(self._bundle_dir).as_posix()
        except ValueError:
            return str(path)

    def _build_manifest_payload(self) -> dict[str, Any]:
        return {
            "scene_id": self._scene_id,
            "seed": int(self._seed),
            "template": {
                "template_id": self._template.template_id,
                "variant_id": self._template.variant_id,
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
            "collection_metadata_path": self._collection_metadata_path_string(),
            "language_instruction": self._language_instruction,
            "robot_models": self._robot_model_ids_for_manifest(),
            "robot_namespaces": list(self._robot_names),
            "rollout_robot_teams": list(getattr(self, "_rollout_robot_teams", [])),
            "robot_team_mode": self._active_robot_team_mode(),
            "robot_team_policy": self._robot_team_policy_payload(),
            "focus_object": None if self._focus_object is None else self._focus_object.as_dict(),
            "focus_selection": self._focus_selection_debug_payload(),
            "resolved_object_groups": dict(self._resolved_group_details),
            "resolved_light_selectors": dict(self._resolved_light_paths),
            "rejection_summary": dict(self._rejection_summary),
            "sampling_contract": {
                "goal_sampling_mode": "occupancy_map_plus_focus_object",
                "focus_selector": FOCUS_SELECTOR_ID,
                "language_grounded": True,
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

    def _bbox_intersects_environment_obstacles(
        self,
        prim_path: str,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        *,
        margin_m: float = 0.0,
    ) -> bool:
        candidate_root_path = self._canonicalize_obstacle_root_path(
            prim_path,
            max_planar_area_m2=self._obstacle_root_area_limit_m2(),
        )
        obstacle_roots, _ = self._collect_environment_obstacle_root_bboxes(
            occupancy_z_range_m=None,
            exclude_root_paths={candidate_root_path, prim_path},
        )
        for obstacle_root_path, (obstacle_min, obstacle_max) in obstacle_roots.items():
            if obstacle_root_path == candidate_root_path:
                continue
            expanded_margin = float(margin_m)
            if not (
                bbox_max[0] < obstacle_min[0] - expanded_margin
                or bbox_min[0] > obstacle_max[0] + expanded_margin
                or bbox_max[1] < obstacle_min[1] - expanded_margin
                or bbox_min[1] > obstacle_max[1] + expanded_margin
                or bbox_max[2] < obstacle_min[2] - expanded_margin
                or bbox_min[2] > obstacle_max[2] + expanded_margin
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
            if self._bbox_intersects_environment_obstacles(
                accepted["prim_path"],
                bbox_min,
                bbox_max,
                margin_m=float(accepted.get("margin_m", 0.0)),
            ):
                raise RuntimeError(
                    f"Final environment-overlap validation failed for {accepted['prim_path']} after randomization."
                )
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
            occupancy_z_range_m=map_spec.occupancy_z_range_m,
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

    def _ros_costmap_inflation_radius_m(self) -> float:
        inflation_radii: list[float] = []

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                if "inflation_radius" in node:
                    try:
                        inflation_radii.append(float(node["inflation_radius"]))
                    except (TypeError, ValueError):
                        pass
                for value in node.values():
                    _walk(value)
            elif isinstance(node, list):
                for value in node:
                    _walk(value)

        for relative_path in ROS_COSTMAP_PARAMS_RELS:
            params_path = self._repo_root / relative_path
            if not params_path.exists():
                continue
            with params_path.open("r", encoding="utf-8") as stream:
                _walk(yaml.safe_load(stream) or {})

        return max(inflation_radii) if inflation_radii else DEFAULT_ROS_COSTMAP_INFLATION_RADIUS_M

    def _focus_selection_debug_payload(self) -> dict[str, Any]:
        return dict(getattr(self, "_focus_selection_debug", {}) or {})

    def _collection_metadata_path_string(self) -> str:
        collection_metadata_path = getattr(self, "_collection_metadata_path", None)
        if collection_metadata_path is not None:
            return str(collection_metadata_path)
        team_config_path = getattr(self, "_team_config_path", None)
        if team_config_path is not None:
            return str(Path(team_config_path).parent.parent / "collection_metadata.yaml")
        return ""

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
