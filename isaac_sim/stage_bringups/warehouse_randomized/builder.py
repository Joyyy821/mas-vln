from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import math
from pathlib import Path
import re
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
    ObjectRandomizationSpec,
    SelectorSpec,
    WarehouseTemplate,
    build_template_catalog,
)


@dataclass
class RandomizedWarehouseBuildResult:
    scene_id: str
    seed: int
    template_id: str
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
        template_id: str,
        seed: int,
        robot_models: list[str],
        robot_count: int,
        rollout_count: int,
        language_instruction: str,
        enable_ros2_runtime: bool,
        rollout_control_topic: str,
        rollout_reset_done_topic: str,
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
        self._overwrite = bool(overwrite)

        if self._robot_count < 2 or self._robot_count > 5:
            raise ValueError(f"robot_count must be between 2 and 5, got {self._robot_count}.")
        if self._rollout_count <= 0:
            raise ValueError("rollout_count must be positive.")

        template_catalog = build_template_catalog(self._repo_root)
        if template_id not in template_catalog:
            raise ValueError(
                f"Unknown template_id '{template_id}'. Available templates: {sorted(template_catalog)}"
            )
        self._template: WarehouseTemplate = template_catalog[template_id]

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
        self._rollouts_dir = self._bundle_dir / "rollouts"
        self._nav2_map_path = self._bundle_dir / "nav2_map.yaml"
        self._mapf_map_path = self._bundle_dir / "mapf_map.yaml"

        self._env_prim_path = "/World/Env/Warehouse"
        self._randomized_props_root = "/World/Env/RandomizedProps"
        self._robots_root = "/World/Robots"

        self._assets_root_path = ""
        self._randomization_records: list[dict[str, Any]] = []
        self._resolved_selectors: dict[str, list[str]] = {}
        self._accepted_layout_bboxes: list[dict[str, Any]] = []
        self._nav2_map_result: MapExportResult | None = None
        self._mapf_map_result: MapExportResult | None = None
        self._focus_object: ObjectBBox3D | None = None
        self._rollouts_payload: list[dict[str, Any]] = []
        self._sampling_validation: dict[str, Any] = {}
        self._robot_prim_paths: list[str] = []
        self._robot_controllers: list[RuntimeRobotController] = []
        self._ros_bridge: InternalIsaacRosBridge | None = None

    @property
    def sim_app(self):
        return self._sim_app

    def ensure_sim_app(self, *, headless: bool) -> None:
        if self._sim_app is not None:
            return
        self._sim_app = maybe_start_sim_app(
            headless=headless,
            enable_ros2_bridge=self._enable_ros2_runtime,
            extra_extensions=["isaacsim.asset.gen.omap"],
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
        except Exception as exc:
            raise RuntimeError(
                f"Randomized warehouse build failed during step '{current_step}' "
                f"for scene '{self._scene_id}'."
            ) from exc

        return RandomizedWarehouseBuildResult(
            scene_id=self._scene_id,
            seed=self._seed,
            template_id=self._template.template_id,
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

    def _resolve_assets_root(self) -> None:
        from isaacsim.storage.native import get_assets_root_path

        self._assets_root_path = str(get_assets_root_path()).rstrip("/")
        if not self._assets_root_path:
            raise RuntimeError("Isaac Sim assets root path is unavailable.")

    def _build_stage_base(self) -> None:
        new_stage()
        set_stage_units(1.0)
        define_xform("/World")
        define_xform("/World/Env")
        define_xform(self._randomized_props_root)
        define_xform(self._robots_root)
        add_reference(
            f"{self._assets_root_path}{self._template.base_environment_usd_rel}",
            self._env_prim_path,
        )
        self._ensure_physics_scene()
        self._update_sim(12)

    def _apply_randomization_pipeline(self) -> None:
        self._apply_light_randomization()
        for randomizer in self._template.object_randomizers:
            if randomizer.policy == "appearance_only":
                self._apply_appearance_randomization(randomizer)
            elif randomizer.policy == "jittered_existing":
                self._apply_jitter_randomization(randomizer)
            elif randomizer.policy == "spawned_floor_prop":
                self._apply_spawned_floor_props(randomizer)

        self._validate_final_layout()

    def _resolve_selector_paths(self, selectors: tuple[SelectorSpec, ...], *, policy: str) -> list[str]:
        cache_key = f"{policy}::" + "|".join(f"{selector.mode}:{selector.value}" for selector in selectors)
        if cache_key in self._resolved_selectors:
            return list(self._resolved_selectors[cache_key])

        stage = get_stage()
        resolved: list[str] = []
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if not (path == self._env_prim_path or path.startswith(f"{self._env_prim_path}/")):
                continue
            if any(self._selector_matches(prim, selector) for selector in selectors):
                resolved.append(self._canonicalize_prim_path(path, policy=policy))

        deduped: list[str] = []
        for path in sorted(set(resolved), key=len):
            if any(path == other or path.startswith(f"{other}/") for other in deduped):
                continue
            deduped.append(path)

        self._resolved_selectors[cache_key] = deduped
        return list(deduped)

    def _selector_matches(self, prim, selector: SelectorSpec) -> bool:
        path = prim.GetPath().pathString
        name = prim.GetName()
        value = str(selector.value)
        if selector.mode == "exact_path":
            return path == value
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

    def _canonicalize_prim_path(self, prim_path: str, *, policy: str) -> str:
        from pxr import UsdGeom

        stage = get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return prim_path

        if policy == "appearance_only":
            return prim_path

        current = prim
        best = prim
        while True:
            parent = current.GetParent()
            if not parent or not parent.IsValid():
                break
            parent_path = parent.GetPath().pathString
            if parent_path in {"/", self._env_prim_path}:
                break
            if parent.IsA(UsdGeom.Xform) or parent.GetTypeName() in {"Xform", "Scope"}:
                best = parent
                current = parent
                continue
            break
        return best.GetPath().pathString

    def _apply_light_randomization(self) -> None:
        for light_spec in self._template.light_randomizers:
            light_paths = self._resolve_selector_paths(light_spec.selectors, policy="appearance_only")
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

        object_paths = self._resolve_selector_paths(spec.selectors, policy=spec.policy)
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
        object_paths = self._resolve_selector_paths(spec.selectors, policy=spec.policy)
        if spec.required and not object_paths:
            raise RuntimeError(f"Required randomization selector '{spec.name}' resolved to no prims.")

        for object_path in object_paths:
            original_position, original_orientation = get_world_pose_xyzw(object_path)
            original_yaw_deg = quaternion_xyzw_to_yaw(original_orientation) * 180.0 / math.pi
            accepted_bbox = None
            accepted_position = original_position.copy()
            accepted_yaw_deg = original_yaw_deg
            accepted_scale = 1.0

            for attempt_index in range(spec.max_attempts):
                candidate_position = original_position.copy()
                candidate_position[0] += float(self._rng.uniform(-spec.xy_jitter_m[0], spec.xy_jitter_m[0]))
                candidate_position[1] += float(self._rng.uniform(-spec.xy_jitter_m[1], spec.xy_jitter_m[1]))
                candidate_yaw_deg = float(
                    original_yaw_deg + self._rng.uniform(spec.yaw_jitter_deg[0], spec.yaw_jitter_deg[1])
                )
                candidate_scale = float(
                    self._rng.uniform(spec.uniform_scale_range[0], spec.uniform_scale_range[1])
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
                    continue
                if self._bbox_intersects_accepted_layout(
                    bbox_min,
                    bbox_max,
                    margin_m=spec.collision_margin_m,
                ):
                    continue
                if not self._physx_overlap_clear(object_path, bbox_min, bbox_max):
                    continue

                accepted_bbox = (bbox_min, bbox_max)
                accepted_position = candidate_position
                accepted_yaw_deg = candidate_yaw_deg
                accepted_scale = candidate_scale
                break

            set_xform_pose(
                object_path,
                tuple(accepted_position.tolist()),
                yaw_deg=accepted_yaw_deg,
                scale_xyz=(accepted_scale, accepted_scale, accepted_scale),
            )
            self._update_sim(2)

            bbox_min, bbox_max = compute_world_bbox(object_path)
            self._accepted_layout_bboxes.append(
                {
                    "prim_path": object_path,
                    "bbox_min": bbox_min,
                    "bbox_max": bbox_max,
                    "margin_m": float(spec.collision_margin_m),
                }
            )
            self._randomization_records.append(
                {
                    "category": "layout",
                    "name": spec.name,
                    "prim_path": object_path,
                    "policy": spec.policy,
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

    def _apply_spawned_floor_props(self, spec: ObjectRandomizationSpec) -> None:
        prop_count = int(self._rng.integers(spec.spawn_count_range[0], spec.spawn_count_range[1] + 1))
        if prop_count <= 0:
            return

        try:
            from isaacsim.core.api.objects import FixedCuboid
        except Exception:
            from omni.isaac.core.objects import FixedCuboid

        spawn_min_x = self._template.nav2_map.min_bound_xyz[0] + 1.0
        spawn_max_x = self._template.nav2_map.max_bound_xyz[0] - 1.0
        spawn_min_y = self._template.nav2_map.min_bound_xyz[1] + 1.0
        spawn_max_y = self._template.nav2_map.max_bound_xyz[1] - 1.0

        accepted_count = 0
        for prop_index in range(prop_count):
            prop_path = f"{self._randomized_props_root}/Prop_{spec.name}_{prop_index + 1}"
            accepted = False
            accepted_payload: dict[str, Any] | None = None

            for _ in range(spec.max_attempts):
                size_x = float(
                    self._rng.uniform(spec.spawn_size_range_m[0][0], spec.spawn_size_range_m[0][1])
                )
                size_y = float(
                    self._rng.uniform(spec.spawn_size_range_m[1][0], spec.spawn_size_range_m[1][1])
                )
                size_z = float(
                    self._rng.uniform(spec.spawn_size_range_m[2][0], spec.spawn_size_range_m[2][1])
                )
                position_xyz = np.array(
                    [
                        float(self._rng.uniform(spawn_min_x, spawn_max_x)),
                        float(self._rng.uniform(spawn_min_y, spawn_max_y)),
                        0.5 * size_z,
                    ],
                    dtype=float,
                )
                yaw_deg = float(self._rng.uniform(-180.0, 180.0))
                color = np.array(self._sample_rgb(spec.color_value_range), dtype=float)

                FixedCuboid(
                    prim_path=prop_path,
                    name=Path(prop_path).name,
                    position=position_xyz,
                    scale=np.array([size_x, size_y, size_z], dtype=float),
                    color=color,
                    size=1.0,
                )
                set_xform_pose(prop_path, tuple(position_xyz.tolist()), yaw_deg=yaw_deg, scale_xyz=(size_x, size_y, size_z))
                self._update_sim(2)

                bbox_min, bbox_max = compute_world_bbox(prop_path)
                if self._intersects_keepout(
                    bbox_min,
                    bbox_max,
                    keepout_zone_ids=spec.keepout_zone_ids,
                    margin_m=spec.collision_margin_m,
                ):
                    get_stage().RemovePrim(prop_path)
                    continue
                if self._bbox_intersects_accepted_layout(
                    bbox_min,
                    bbox_max,
                    margin_m=spec.collision_margin_m,
                ):
                    get_stage().RemovePrim(prop_path)
                    continue
                if not self._physx_overlap_clear(prop_path, bbox_min, bbox_max):
                    get_stage().RemovePrim(prop_path)
                    continue

                self._accepted_layout_bboxes.append(
                    {
                        "prim_path": prop_path,
                        "bbox_min": bbox_min,
                        "bbox_max": bbox_max,
                        "margin_m": float(spec.collision_margin_m),
                    }
                )
                accepted_payload = {
                    "category": "spawned_floor_prop",
                    "name": spec.name,
                    "prim_path": prop_path,
                    "policy": spec.policy,
                    "position_xyz": position_xyz.tolist(),
                    "yaw_deg": yaw_deg,
                    "size_xyz": [size_x, size_y, size_z],
                    "color_rgb": color.tolist(),
                }
                accepted = True
                break

            if not accepted:
                get_stage().RemovePrim(prop_path)
                continue

            accepted_count += 1
            if accepted_payload is not None:
                self._randomization_records.append(accepted_payload)

        self._randomization_records.append(
            {
                "category": "spawn_summary",
                "name": spec.name,
                "requested_count": prop_count,
                "accepted_count": accepted_count,
            }
        )

    def _resolve_focus_object(self) -> None:
        focus_paths = self._resolve_selector_paths(
            self._template.focus_object_selectors,
            policy="jittered_existing",
        )
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

        def _export_with_retries(
            *,
            yaml_path,
            resolution_m,
            origin_hint_xyz,
            min_bound_xyz,
            max_bound_xyz,
            start_location_xyz,
        ):
            last_exception: Exception | None = None
            for warmup_steps in (8, 16, 32):
                self._update_sim(warmup_steps)
                try:
                    return export_occupancy_map(
                        yaml_path=yaml_path,
                        resolution_m=resolution_m,
                        origin_hint_xyz=origin_hint_xyz,
                        min_bound_xyz=min_bound_xyz,
                        max_bound_xyz=max_bound_xyz,
                        start_location_xyz=start_location_xyz,
                    )
                except RuntimeError as exc:
                    if "empty buffer" not in str(exc).lower():
                        raise
                    last_exception = exc
            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"Failed to export occupancy map for {yaml_path}.")

        timeline = omni.timeline.get_timeline_interface()
        was_playing = timeline.is_playing()
        if not was_playing:
            timeline.play()

        try:
            self._ensure_physics_scene()
            self._nav2_map_result = _export_with_retries(
                yaml_path=self._nav2_map_path,
                resolution_m=self._template.nav2_map.resolution_m,
                origin_hint_xyz=self._template.nav2_map.origin_hint_xyz,
                min_bound_xyz=self._template.nav2_map.min_bound_xyz,
                max_bound_xyz=self._template.nav2_map.max_bound_xyz,
                start_location_xyz=self._choose_map_start_location(self._template.nav2_map),
            )
            self._mapf_map_result = _export_with_retries(
                yaml_path=self._mapf_map_path,
                resolution_m=self._template.mapf_map.resolution_m,
                origin_hint_xyz=self._template.mapf_map.origin_hint_xyz,
                min_bound_xyz=self._template.mapf_map.min_bound_xyz,
                max_bound_xyz=self._template.mapf_map.max_bound_xyz,
                start_location_xyz=self._choose_map_start_location(self._template.mapf_map),
            )
        finally:
            if not was_playing:
                pause_timeline = getattr(timeline, "pause", None)
                if callable(pause_timeline):
                    pause_timeline()
                else:
                    timeline.stop()
                self._update_sim(2)

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
        first_robot_model = self._robot_adapters[0].model_id if self._robot_adapters else ""
        payload = {
            "scene_id": self._scene_id,
            "template_id": self._template.template_id,
            "seed": int(self._seed),
            "usd_path": str(self._scene_usd_path),
            "robot_model": first_robot_model,
            "language_instruction": self._language_instruction,
            "environment": {
                "nav2_map": str(self._nav2_map_path),
                "mapf_map": str(self._mapf_map_path),
            },
            "rollouts": self._rollouts_payload,
            "validation": self._build_validation_summary(),
        }
        with self._team_config_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)

    def _write_manifest(self) -> None:
        manifest = {
            "scene_id": self._scene_id,
            "seed": int(self._seed),
            "template": {
                "template_id": self._template.template_id,
                "description": self._template.description,
                "metadata": dict(self._template.metadata),
            },
            "scene_usd_path": str(self._scene_usd_path),
            "language_instruction": self._language_instruction,
            "robot_models": [adapter.model_id for adapter in self._robot_adapters],
            "robot_namespaces": list(self._robot_names),
            "focus_object": None if self._focus_object is None else self._focus_object.as_dict(),
            "resolved_selectors": dict(self._resolved_selectors),
            "randomization_records": self._randomization_records,
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
        with self._manifest_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(manifest, stream, sort_keys=False)

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
            "randomized_object_count": len(self._randomization_records),
            "accepted_layout_boxes": len(self._accepted_layout_bboxes),
            "nav2_map": None if self._nav2_map_result is None else self._map_result_dict(self._nav2_map_result),
            "mapf_map": None if self._mapf_map_result is None else self._map_result_dict(self._mapf_map_result),
            "rollout_sampling": dict(self._sampling_validation),
            "robot_count": int(self._robot_count),
            "rollout_count": int(self._rollout_count),
            "ros_runtime_enabled": bool(self._enable_ros2_runtime),
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

        extent_xyz = 0.5 * (bbox_max - bbox_min)
        extent_xyz[2] = max(0.05, min(extent_xyz[2], 0.35))
        origin_xyz = 0.5 * (bbox_min + bbox_max)
        origin_xyz[2] = bbox_min[2] + extent_xyz[2] + 0.05
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
            if not self._physx_overlap_clear(
                accepted["prim_path"],
                accepted["bbox_min"],
                accepted["bbox_max"],
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
        occupancy_map = OccupancyMap.load(str(map_spec.reference_yaml_path))
        free_rows, free_cols = np.nonzero(occupancy_map.free_mask)
        fallback = (
            0.5 * (float(map_spec.min_bound_xyz[0]) + float(map_spec.max_bound_xyz[0])),
            0.5 * (float(map_spec.min_bound_xyz[1]) + float(map_spec.max_bound_xyz[1])),
            max(0.2, float(map_spec.min_bound_xyz[2])),
        )
        if free_rows.size == 0:
            return fallback

        free_xy = occupancy_map.grid_to_world_xy(free_rows, free_cols)
        center_xy = np.array(fallback[:2], dtype=float)
        distance_order = np.argsort(np.sum((free_xy - center_xy) ** 2, axis=1))
        candidate_z = float(fallback[2])

        for candidate_index in distance_order.tolist():
            candidate_xyz = np.array(
                [float(free_xy[candidate_index][0]), float(free_xy[candidate_index][1]), candidate_z],
                dtype=float,
            )
            if self._point_is_clear_for_map_start(candidate_xyz, clearance_m=0.35):
                return tuple(candidate_xyz.tolist())

        return fallback

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
