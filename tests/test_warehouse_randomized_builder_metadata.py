from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from isaac_sim.goal_generator.object_goal_sampler_utils import ObjectBBox3D
from isaac_sim.stage_bringups.warehouse_randomized import builder as warehouse_builder_module
from isaac_sim.stage_bringups.warehouse_randomized.builder import RandomizedWarehouseBuilder
from isaac_sim.stage_bringups.warehouse_randomized.instructions.forklift_near_shelf import (
    select_focus_object,
)
from isaac_sim.stage_bringups.warehouse_randomized.maps import MapExportResult
from isaac_sim.stage_bringups.warehouse_randomized.templates import (
    ObjectGroupSpec,
    ObjectRandomizationSpec,
    PlacementZone,
    SelectorSpec,
    TemplateMapSpec,
    WarehouseTemplate,
)


def _make_template(tmpdir: str) -> WarehouseTemplate:
    tmp_path = Path(tmpdir)
    return WarehouseTemplate(
        template_id="2",
        variant_id="open",
        description="Test open template.",
        shared_defaults_config_path=tmp_path / "warehouse_shared.yaml",
        preset_config_path=tmp_path / "warehouse_open.yaml",
        source_template_usd_path=tmp_path / "warehouse_template_2.usd",
        base_environment_usd=str(tmp_path / "warehouse_template_2.usd"),
        nav2_map=TemplateMapSpec(
            0.05,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 2.0),
            (0.1, 0.62),
            None,
        ),
        mapf_map=TemplateMapSpec(
            0.2,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 2.0),
            (0.1, 0.62),
            None,
        ),
        light_randomizers=(),
        object_groups=(
            ObjectGroupSpec(
                group_id="focus_objects",
                selectors=(
                    SelectorSpec(mode="exact_path", value="/World/Forklift"),
                    SelectorSpec(mode="exact_path", value="/World/Forklift_01"),
                ),
            ),
        ),
        placement_zones=(),
        keepout_zones=(),
        object_randomizers=(),
        focus_group_names=("focus_objects",),
        focus_distance_range_m=(2.5, 5.0),
        metadata={"layout_class": "open"},
    )


class WarehouseRandomizedBuilderMetadataTests(unittest.TestCase):
    def _minimal_builder_for_build_order(self, tmp_path: Path) -> RandomizedWarehouseBuilder:
        builder = object.__new__(RandomizedWarehouseBuilder)
        builder._scene_id = "template_2_open"
        builder._seed = 77
        builder._template = _make_template(str(tmp_path))
        builder._bundle_dir = tmp_path
        builder._scene_usd_path = tmp_path / "scene.usd"
        builder._team_config_path = tmp_path / "team_config.yaml"
        builder._nav2_map_path = tmp_path / "nav2_map.yaml"
        builder._mapf_map_path = tmp_path / "mapf_map.yaml"
        builder._manifest_path = tmp_path / "scene_manifest.yaml"
        builder._rollouts_dir = tmp_path / "rollouts"
        builder._failure_snapshot_path = tmp_path / "build_failure_snapshot.yaml"
        builder._ros_bridge = None
        builder._robot_controllers = []
        builder._robot_prim_paths = []
        builder._enable_ros2_runtime = False
        builder._scene_only = False
        builder._spawn_robots_only = False
        builder._build_validation_summary = lambda: {}
        return builder

    def test_default_build_saves_robot_free_scene_before_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            builder = self._minimal_builder_for_build_order(tmp_path)
            calls: list[str] = []

            for name in (
                "_prepare_output_bundle",
                "_resolve_assets_root",
                "_build_stage_base",
                "_apply_randomization_pipeline",
                "_resolve_focus_object",
                "_export_maps",
                "_save_scene_usd",
                "_sample_rollouts",
                "_write_team_config",
                "_write_manifest",
            ):
                setattr(builder, name, lambda name=name: calls.append(name))
            builder._spawn_robots = lambda: calls.append("_spawn_robots")

            result = builder.build()

            self.assertEqual(
                calls,
                [
                    "_prepare_output_bundle",
                    "_resolve_assets_root",
                    "_build_stage_base",
                    "_apply_randomization_pipeline",
                    "_resolve_focus_object",
                    "_export_maps",
                    "_save_scene_usd",
                    "_sample_rollouts",
                    "_write_team_config",
                    "_write_manifest",
                ],
            )
            self.assertEqual(result.scene_usd_path, tmp_path / "scene.usd")

    def test_scene_only_build_skips_sampling_team_config_and_robot_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            builder = self._minimal_builder_for_build_order(tmp_path)
            builder._scene_only = True
            calls: list[str] = []

            for name in (
                "_prepare_output_bundle",
                "_resolve_assets_root",
                "_build_stage_base",
                "_apply_randomization_pipeline",
                "_resolve_focus_object",
                "_export_maps",
                "_save_scene_usd",
                "_write_manifest",
            ):
                setattr(builder, name, lambda name=name: calls.append(name))
            builder._sample_rollouts = lambda: calls.append("_sample_rollouts")
            builder._spawn_robots = lambda: calls.append("_spawn_robots")
            builder._write_team_config = lambda: calls.append("_write_team_config")

            builder.build()

            self.assertNotIn("_sample_rollouts", calls)
            self.assertNotIn("_spawn_robots", calls)
            self.assertNotIn("_write_team_config", calls)
            self.assertEqual(calls[-2:], ["_save_scene_usd", "_write_manifest"])

    def test_spawn_robots_only_loads_existing_scene_without_resaving_scene_usd(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            builder = self._minimal_builder_for_build_order(tmp_path)
            builder._spawn_robots_only = True
            calls: list[str] = []

            for name in (
                "_prepare_spawn_robots_only_bundle",
                "_resolve_assets_root",
                "_open_existing_scene_usd",
                "_resolve_focus_object",
                "_sample_rollouts",
                "_spawn_robots",
                "_write_team_config",
                "_write_manifest",
            ):
                setattr(builder, name, lambda name=name: calls.append(name))
            builder._save_scene_usd = lambda: calls.append("_save_scene_usd")

            builder.build()

            self.assertEqual(
                calls,
                [
                    "_prepare_spawn_robots_only_bundle",
                    "_resolve_assets_root",
                    "_open_existing_scene_usd",
                    "_resolve_focus_object",
                    "_sample_rollouts",
                    "_spawn_robots",
                    "_write_team_config",
                    "_write_manifest",
                ],
            )
            self.assertNotIn("_save_scene_usd", calls)

    def test_spawn_robots_only_uses_existing_team_config_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            builder = self._minimal_builder_for_build_order(tmp_path)
            builder._spawn_robots_only = True
            builder._overwrite = False
            builder._team_config_path.write_text(
                yaml.safe_dump(
                    {
                        "robot_model": "nova_carter",
                        "language_instruction": "go to the forklift near the shelf",
                        "rollouts": [
                            {
                                "id": 1,
                                "robots": [
                                    {
                                        "name": "robot1",
                                        "initial_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                                        "goal_pose": {"x": 1.0, "y": 1.0, "z": 0.0, "yaw": 0.0},
                                    },
                                    {
                                        "name": "robot2",
                                        "initial_pose": {"x": 0.0, "y": 1.0, "z": 0.0, "yaw": 0.0},
                                        "goal_pose": {"x": 1.0, "y": 2.0, "z": 0.0, "yaw": 0.0},
                                    },
                                ],
                            }
                        ],
                        "validation": {"rollout_sampling": {"inflation_radius_m": 1.0}},
                    }
                ),
                encoding="utf-8",
            )
            calls: list[str] = []

            for name in (
                "_prepare_spawn_robots_only_bundle",
                "_resolve_assets_root",
                "_open_existing_scene_usd",
                "_resolve_focus_object",
                "_load_existing_team_config",
                "_spawn_robots",
                "_write_manifest",
            ):
                setattr(builder, name, lambda name=name: calls.append(name))
            builder._sample_rollouts = lambda: calls.append("_sample_rollouts")
            builder._write_team_config = lambda: calls.append("_write_team_config")

            builder.build()

            self.assertIn("_load_existing_team_config", calls)
            self.assertNotIn("_sample_rollouts", calls)
            self.assertNotIn("_write_team_config", calls)

    def test_team_config_and_manifest_include_template_and_variant_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            builder = object.__new__(RandomizedWarehouseBuilder)
            builder._scene_id = "template_2_open"
            builder._seed = 77
            builder._template = _make_template(tmpdir)
            builder._bundle_dir = tmp_path
            builder._scene_usd_path = tmp_path / "scene.usd"
            builder._team_config_path = tmp_path / "team_config.yaml"
            builder._manifest_path = tmp_path / "scene_manifest.yaml"
            builder._rollouts_dir = tmp_path / "rollouts"
            builder._nav2_map_path = tmp_path / "nav2_map.yaml"
            builder._mapf_map_path = tmp_path / "mapf_map.yaml"
            builder._collection_metadata_path = tmp_path / "collection_metadata.yaml"
            builder._robot_adapters = [SimpleNamespace(model_id="nova_carter")]
            builder._robot_names = ["robot1", "robot2", "robot3"]
            builder._randomization_records = []
            builder._accepted_layout_bboxes = []
            builder._resolved_group_details = {}
            builder._resolved_light_paths = {}
            builder._rejection_summary = {}
            builder._nav2_map_result = None
            builder._mapf_map_result = None
            builder._focus_object = None
            builder._focus_selection_debug = {
                "selector_id": warehouse_builder_module.FOCUS_SELECTOR_ID,
                "selected_reason": "smallest_world_center_x",
            }
            builder._rollouts_payload = []
            builder._sampling_validation = {}
            builder._robot_count = 3
            builder._rollout_count = 5
            builder._enable_ros2_runtime = False
            builder._language_instruction = "Look at the forklift."
            builder._resolved_base_environment_usd = str(tmp_path / "warehouse_template_2.usd")
            builder._rollout_control_topic = "/control"
            builder._rollout_reset_done_topic = "/done"

            team_config_payload = builder._build_team_config_payload()
            manifest_payload = builder._build_manifest_payload()

            self.assertEqual(team_config_payload["template_id"], "2")
            self.assertEqual(team_config_payload["variant_id"], "open")
            self.assertEqual(
                team_config_payload["shared_defaults_config_path"],
                str(tmp_path / "warehouse_shared.yaml"),
            )
            self.assertEqual(
                team_config_payload["preset_config_path"],
                str(tmp_path / "warehouse_open.yaml"),
            )
            self.assertEqual(
                team_config_payload["source_template_usd_path"],
                str(tmp_path / "warehouse_template_2.usd"),
            )
            self.assertEqual(team_config_payload["validation"]["variant_id"], "open")
            self.assertEqual(
                team_config_payload["environment"],
                {
                    "nav2_map": "nav2_map.yaml",
                    "mapf_map": "mapf_map.yaml",
                },
            )
            self.assertEqual(
                team_config_payload["collection_metadata_path"],
                str(tmp_path / "collection_metadata.yaml"),
            )
            self.assertEqual(
                team_config_payload["robot_team_mode"],
                warehouse_builder_module.DEFAULT_ROBOT_TEAM_MODE,
            )
            self.assertEqual(
                team_config_payload["sampling_contract"]["focus_selector"],
                warehouse_builder_module.FOCUS_SELECTOR_ID,
            )
            self.assertTrue(team_config_payload["sampling_contract"]["language_grounded"])

            self.assertEqual(manifest_payload["template"]["template_id"], "2")
            self.assertEqual(manifest_payload["template"]["variant_id"], "open")
            self.assertEqual(
                manifest_payload["template"]["source_template_usd_path"],
                str(tmp_path / "warehouse_template_2.usd"),
            )
            self.assertEqual(manifest_payload["validation_summary"]["template_id"], "2")
            self.assertEqual(manifest_payload["validation_summary"]["variant_id"], "open")
            self.assertEqual(
                manifest_payload["collection_metadata_path"],
                str(tmp_path / "collection_metadata.yaml"),
            )
            self.assertEqual(
                manifest_payload["focus_selection"]["selector_id"],
                warehouse_builder_module.FOCUS_SELECTOR_ID,
            )
            self.assertEqual(team_config_payload["validation"]["nav2_map"], None)

    def test_collection_metadata_payload_is_written_at_collection_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            builder = object.__new__(RandomizedWarehouseBuilder)
            builder._language_instruction = "go to the forklift near the shelf"
            builder._collection_metadata_path = tmp_path / "collection_metadata.yaml"

            builder._write_collection_metadata()

            payload = yaml.safe_load(
                builder._collection_metadata_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload,
                {
                    "language_instruction": "go to the forklift near the shelf",
                    "focus_selector": warehouse_builder_module.FOCUS_SELECTOR_ID,
                    "robot_team_mode": warehouse_builder_module.DEFAULT_ROBOT_TEAM_MODE,
                },
            )

    def test_instruction_selector_chooses_smaller_world_x_candidate(self) -> None:
        forklift = ObjectBBox3D(
            prim_path="/World/Env/Warehouse/Forklift",
            min_xyz=np.array([2.0, 0.0, 0.0], dtype=float),
            max_xyz=np.array([4.0, 2.0, 2.0], dtype=float),
        )
        forklift_01 = ObjectBBox3D(
            prim_path="/World/Env/Warehouse/Forklift_01",
            min_xyz=np.array([-5.0, 0.0, 0.0], dtype=float),
            max_xyz=np.array([-3.0, 2.0, 2.0], dtype=float),
        )

        selected, debug = select_focus_object([forklift, forklift_01])

        self.assertIsNotNone(selected)
        self.assertEqual(selected.prim_path, forklift_01.prim_path)
        self.assertEqual(debug["selector_id"], warehouse_builder_module.FOCUS_SELECTOR_ID)
        self.assertTrue(debug["selector_path"].endswith("forklift_near_shelf.py"))
        self.assertEqual(debug["selected_reason"], "smallest_world_center_x")
        self.assertEqual(debug["candidate_count"], 2)

    def test_resolve_focus_object_records_selector_debug(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = object.__new__(RandomizedWarehouseBuilder)
            builder._template = _make_template(tmpdir)
            focus_paths = [
                "/World/Env/Warehouse/Forklift",
                "/World/Env/Warehouse/Forklift_01",
            ]
            builder._resolve_group_paths = lambda group_name, required=None: focus_paths
            builder._dedupe_root_paths = lambda paths: list(dict.fromkeys(paths))

            boxes = {
                "/World/Env/Warehouse/Forklift": (
                    np.array([2.0, -1.0, 0.0], dtype=float),
                    np.array([4.0, 1.0, 2.0], dtype=float),
                ),
                "/World/Env/Warehouse/Forklift_01": (
                    np.array([-7.0, -1.0, 0.0], dtype=float),
                    np.array([-5.0, 1.0, 2.0], dtype=float),
                ),
            }
            original_compute_world_bbox = warehouse_builder_module.compute_world_bbox
            warehouse_builder_module.compute_world_bbox = lambda prim_path: boxes[prim_path]
            try:
                builder._resolve_focus_object()
            finally:
                warehouse_builder_module.compute_world_bbox = original_compute_world_bbox

            self.assertEqual(
                builder._focus_object.prim_path,
                "/World/Env/Warehouse/Forklift_01",
            )
            self.assertEqual(
                builder._focus_selection_debug["selector_id"],
                warehouse_builder_module.FOCUS_SELECTOR_ID,
            )
            self.assertEqual(builder._focus_selection_debug["candidate_count"], 2)

    def test_optional_zoned_jitter_removes_object_when_no_zone_candidate_fits(self) -> None:
        builder = object.__new__(RandomizedWarehouseBuilder)
        zone = PlacementZone(
            zone_id="support_a",
            zone_type="support",
            min_xyz=(0.0, 0.0, 0.0),
            max_xyz=(1.0, 1.0, 1.0),
        )
        builder._rng = np.random.default_rng(1)
        builder._template = SimpleNamespace(template_id="2")
        builder._randomization_records = []
        builder._rejection_summary = {}
        builder._resolve_randomizer_target_paths = lambda spec: ["/World/Obj"]
        builder._resolve_zones = lambda zone_ids, allowed_types=None: [zone]
        builder._ordered_candidate_zones_for_object = lambda **kwargs: [zone]
        builder._place_object_in_zone = lambda *args, **kwargs: None
        builder._remove_prim_calls = []
        builder._remove_prim = lambda prim_path: builder._remove_prim_calls.append(prim_path)

        original_get_world_pose = warehouse_builder_module.get_world_pose_xyzw
        original_quaternion_to_yaw = warehouse_builder_module.quaternion_xyzw_to_yaw
        warehouse_builder_module.get_world_pose_xyzw = lambda prim_path: (
            np.array([5.0, 5.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        )
        warehouse_builder_module.quaternion_xyzw_to_yaw = lambda quat: 0.0
        try:
            builder._apply_jitter_randomization(
                ObjectRandomizationSpec(
                    name="floor_prop_layout",
                    policy="jittered_existing",
                    target_group_name="floor_props",
                    support_zone_ids=("support_a",),
                    max_attempts=2,
                    required=False,
                )
            )
        finally:
            warehouse_builder_module.get_world_pose_xyzw = original_get_world_pose
            warehouse_builder_module.quaternion_xyzw_to_yaw = original_quaternion_to_yaw

        self.assertEqual(builder._remove_prim_calls, ["/World/Obj"])
        self.assertEqual(
            builder._rejection_summary["floor_prop_layout"]["unplaced_in_zone"],
            1,
        )
        self.assertEqual(builder._randomization_records[0]["category"], "layout_skip")
        self.assertEqual(builder._randomization_records[0]["reason"], "unplaced_in_zone")

    def test_required_zoned_jitter_raises_when_no_zone_candidate_fits(self) -> None:
        builder = object.__new__(RandomizedWarehouseBuilder)
        zone = PlacementZone(
            zone_id="anchor_a",
            zone_type="anchor",
            min_xyz=(0.0, 0.0, 0.0),
            max_xyz=(1.0, 1.0, 1.0),
        )
        builder._rng = np.random.default_rng(1)
        builder._template = SimpleNamespace(template_id="2")
        builder._randomization_records = []
        builder._rejection_summary = {}
        builder._resolve_randomizer_target_paths = lambda spec: ["/World/Forklift"]
        builder._resolve_zones = lambda zone_ids, allowed_types=None: [zone]
        builder._assign_unique_anchor_zones = lambda **kwargs: {}
        builder._ordered_candidate_zones_for_object = lambda **kwargs: [zone]
        builder._place_object_in_zone = lambda *args, **kwargs: None

        original_get_world_pose = warehouse_builder_module.get_world_pose_xyzw
        original_quaternion_to_yaw = warehouse_builder_module.quaternion_xyzw_to_yaw
        original_set_xform_pose = warehouse_builder_module.set_xform_pose
        warehouse_builder_module.get_world_pose_xyzw = lambda prim_path: (
            np.array([5.0, 5.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        )
        warehouse_builder_module.quaternion_xyzw_to_yaw = lambda quat: 0.0
        warehouse_builder_module.set_xform_pose = lambda *args, **kwargs: None
        try:
            with self.assertRaisesRegex(RuntimeError, "could not be placed inside"):
                builder._apply_jitter_randomization(
                    ObjectRandomizationSpec(
                        name="forklift_layout",
                        policy="jittered_existing",
                        target_group_name="forklifts",
                        anchor_zone_ids=("anchor_a",),
                        max_attempts=2,
                        required=True,
                    )
                )
        finally:
            warehouse_builder_module.get_world_pose_xyzw = original_get_world_pose
            warehouse_builder_module.quaternion_xyzw_to_yaw = original_quaternion_to_yaw
            warehouse_builder_module.set_xform_pose = original_set_xform_pose

        self.assertEqual(
            builder._rejection_summary["forklift_layout"]["unplaced_in_zone"],
            1,
        )

    def test_choose_map_start_location_prefers_support_zone_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = object.__new__(RandomizedWarehouseBuilder)
            builder._template = _make_template(tmpdir)
            builder._template = WarehouseTemplate(
                template_id=builder._template.template_id,
                variant_id=builder._template.variant_id,
                description=builder._template.description,
                shared_defaults_config_path=builder._template.shared_defaults_config_path,
                preset_config_path=builder._template.preset_config_path,
                source_template_usd_path=builder._template.source_template_usd_path,
                base_environment_usd=builder._template.base_environment_usd,
                nav2_map=builder._template.nav2_map,
                mapf_map=builder._template.mapf_map,
                light_randomizers=builder._template.light_randomizers,
                object_groups=builder._template.object_groups,
                placement_zones=(
                    PlacementZone(
                        zone_id="floor_a",
                        zone_type="support",
                        min_xyz=(1.0, 2.0, 0.0),
                        max_xyz=(5.0, 8.0, 2.0),
                    ),
                ),
                keepout_zones=builder._template.keepout_zones,
                object_randomizers=builder._template.object_randomizers,
                focus_group_names=builder._template.focus_group_names,
                focus_distance_range_m=builder._template.focus_distance_range_m,
                metadata=builder._template.metadata,
            )
            builder._accepted_layout_bboxes = []
            builder._environment_bbox = None
            builder._point_is_clear_for_map_start = lambda point_xyz, clearance_m: True

            start_xyz = builder._choose_map_start_location(builder._template.nav2_map)

            self.assertEqual(start_xyz, (3.0, 5.0, 0.2))

    def test_ensure_sim_app_requests_omap_extension(self) -> None:
        builder = object.__new__(RandomizedWarehouseBuilder)
        builder._sim_app = None
        builder._enable_ros2_runtime = False
        calls = []

        original_maybe_start_sim_app = warehouse_builder_module.maybe_start_sim_app

        def _fake_maybe_start_sim_app(**kwargs):
            calls.append(kwargs)
            return object()

        warehouse_builder_module.maybe_start_sim_app = _fake_maybe_start_sim_app
        try:
            builder.ensure_sim_app(headless=True)
        finally:
            warehouse_builder_module.maybe_start_sim_app = original_maybe_start_sim_app

        self.assertEqual(calls[0]["extra_extensions"], (warehouse_builder_module.OMAP_EXTENSION_NAME,))
        self.assertIsNotNone(builder._sim_app)

    def test_ros_costmap_inflation_radius_reads_max_ros_config_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            for inflation_radius, relative_path in zip(
                (0.6, 0.8, 1.0),
                warehouse_builder_module.ROS_COSTMAP_PARAMS_RELS,
            ):
                params_path = repo_root / relative_path
                params_path.parent.mkdir(parents=True, exist_ok=True)
                params_path.write_text(
                    yaml.safe_dump(
                        {
                            "some_costmap": {
                                "ros__parameters": {
                                    "inflation_layer": {
                                        "plugin": "nav2_costmap_2d::InflationLayer",
                                        "inflation_radius": inflation_radius,
                                    }
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            builder = object.__new__(RandomizedWarehouseBuilder)
            builder._repo_root = repo_root

            self.assertEqual(builder._ros_costmap_inflation_radius_m(), 1.0)

    def test_ros_costmap_inflation_radius_falls_back_when_config_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = object.__new__(RandomizedWarehouseBuilder)
            builder._repo_root = Path(tmpdir)

            self.assertEqual(
                builder._ros_costmap_inflation_radius_m(),
                warehouse_builder_module.DEFAULT_ROS_COSTMAP_INFLATION_RADIUS_M,
            )

    def test_export_maps_enables_omap_and_resamples_coarser_mapf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            builder = object.__new__(RandomizedWarehouseBuilder)
            builder._template = _make_template(tmpdir)
            builder._nav2_map_path = tmp_path / "nav2_map.yaml"
            builder._mapf_map_path = tmp_path / "mapf_map.yaml"
            calls: list[str] = []

            def _fake_export_omap(*, yaml_path, map_spec):
                calls.append(f"export:{Path(yaml_path).name}")
                return MapExportResult(
                    yaml_path=Path(yaml_path),
                    png_path=Path(yaml_path).with_suffix(".png"),
                    resolution_m=float(map_spec.resolution_m),
                    origin_xyz=(0.0, 0.0, 0.0),
                    min_bound_xyz=(0.0, 0.0, 0.1),
                    max_bound_xyz=(1.0, 1.0, 0.62),
                    width_px=10,
                    height_px=10,
                    occupied_cells=1,
                    free_cells=99,
                    unknown_cells=0,
                )

            def _fake_export_resampled(*, yaml_path, map_spec, source_result):
                calls.append(f"resample:{Path(yaml_path).name}:{Path(source_result.yaml_path).name}")
                return MapExportResult(
                    yaml_path=Path(yaml_path),
                    png_path=Path(yaml_path).with_suffix(".png"),
                    resolution_m=float(map_spec.resolution_m),
                    origin_xyz=source_result.origin_xyz,
                    min_bound_xyz=source_result.min_bound_xyz,
                    max_bound_xyz=source_result.max_bound_xyz,
                    width_px=3,
                    height_px=3,
                    occupied_cells=1,
                    free_cells=8,
                    unknown_cells=0,
                )

            builder._enable_omap_extension = lambda: calls.append("enable_omap")
            builder._ensure_physics_scene = lambda: calls.append("ensure_physics_scene")
            builder._initialize_physics_for_omap = lambda: calls.append("warmup")
            builder._export_omap_map_from_spec = _fake_export_omap
            builder._export_resampled_omap_map_from_source = _fake_export_resampled

            builder._export_maps()

        self.assertEqual(
            calls,
            [
                "enable_omap",
                "ensure_physics_scene",
                "warmup",
                "export:nav2_map.yaml",
                "resample:mapf_map.yaml:nav2_map.yaml",
            ],
        )

    def test_export_omap_map_from_spec_records_debug(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            builder = object.__new__(RandomizedWarehouseBuilder)
            builder._env_prim_path = "/World/Env/Warehouse"
            builder._map_export_debug = {}
            builder._omap_extension_enabled = True
            builder._omap_physics_warmup_updates = 120
            builder._map_quality_score = lambda result: (0.5, -0.1, result.free_cells)

            map_spec = TemplateMapSpec(
                0.05,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (5.0, 5.0, 2.0),
                (0.1, 0.62),
            )

            original_export_occupancy_map = warehouse_builder_module._export_occupancy_map

            def _fake_export_occupancy_map(**kwargs):
                self.assertEqual(kwargs["env_prim_path"], "/World/Env/Warehouse")
                self.assertEqual(kwargs["resolution_m"], 0.05)
                self.assertEqual(kwargs["origin_xyz"], (0.0, 0.0, 0.0))
                self.assertEqual(kwargs["z_min"], 0.1)
                self.assertEqual(kwargs["z_max"], 0.62)
                yaml_path = Path(kwargs["yaml_path"]).resolve()
                return MapExportResult(
                    yaml_path=yaml_path,
                    png_path=yaml_path.with_suffix(".png"),
                    resolution_m=float(kwargs["resolution_m"]),
                    origin_xyz=(-1.0, -2.0, 0.0),
                    min_bound_xyz=(-1.0, -2.0, 0.1),
                    max_bound_xyz=(1.0, 2.0, 0.62),
                    width_px=40,
                    height_px=80,
                    occupied_cells=12,
                    free_cells=3000,
                    unknown_cells=188,
                    debug={
                        "generator_dimensions": [40, 80],
                        "generator_buffer_size": 3200,
                        "image_pixel_histogram": {0: 12, 205: 188, 255: 3000},
                    },
                )

            warehouse_builder_module._export_occupancy_map = _fake_export_occupancy_map
            try:
                builder._export_omap_map_from_spec(
                    yaml_path=tmp_path / "nav2_map.yaml",
                    map_spec=map_spec,
                )
            finally:
                warehouse_builder_module._export_occupancy_map = original_export_occupancy_map

            debug = builder._map_export_debug[str(tmp_path / "nav2_map.yaml")]
            self.assertEqual(debug["rasterization_mode"], "omap_generator")
            self.assertEqual(debug["health_status"], "ok")
            self.assertTrue(debug["omap_extension_enabled"])
            self.assertEqual(debug["physics_warmup_updates"], 120)
            self.assertEqual(debug["generator_dimensions"], [40, 80])
            self.assertEqual(debug["generator_buffer_size"], 3200)
            self.assertNotIn("omap_fallback_reason", debug)

    def test_export_omap_map_from_spec_fails_fast_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            builder = object.__new__(RandomizedWarehouseBuilder)
            builder._env_prim_path = "/World/Env/Warehouse"
            builder._map_export_debug = {}
            builder._omap_extension_enabled = True
            builder._omap_physics_warmup_updates = 120

            map_spec = TemplateMapSpec(
                0.05,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (5.0, 5.0, 2.0),
                (0.1, 0.62),
            )

            original_export_occupancy_map = warehouse_builder_module._export_occupancy_map
            warehouse_builder_module._export_occupancy_map = lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("omap boom")
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "omap boom"):
                    builder._export_omap_map_from_spec(
                        yaml_path=tmp_path / "nav2_map.yaml",
                        map_spec=map_spec,
                    )
            finally:
                warehouse_builder_module._export_occupancy_map = original_export_occupancy_map

            debug = builder._map_export_debug[str(tmp_path / "nav2_map.yaml")]
            self.assertEqual(debug["rasterization_mode"], "omap_generator")
            self.assertEqual(debug["health_status"], "failed")
            self.assertEqual(debug["failure_type"], "RuntimeError")
            self.assertIn("omap boom", debug["failure_message"])

    def test_export_resampled_omap_map_from_source_records_debug(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            builder = object.__new__(RandomizedWarehouseBuilder)
            builder._env_prim_path = "/World/Env/Warehouse"
            builder._map_export_debug = {}
            builder._omap_extension_enabled = True
            builder._omap_physics_warmup_updates = 120
            builder._map_quality_score = lambda result: (0.5, -0.1, result.free_cells)
            source_result = MapExportResult(
                yaml_path=tmp_path / "nav2_map.yaml",
                png_path=tmp_path / "nav2_map.png",
                resolution_m=0.05,
                origin_xyz=(-1.0, -2.0, 0.0),
                min_bound_xyz=(-1.0, -2.0, 0.1),
                max_bound_xyz=(1.0, 2.0, 0.62),
                width_px=40,
                height_px=80,
                occupied_cells=12,
                free_cells=3000,
                unknown_cells=188,
            )
            map_spec = TemplateMapSpec(
                0.2,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (5.0, 5.0, 2.0),
                (0.1, 0.62),
            )

            original_export_resampled = warehouse_builder_module._export_resampled_occupancy_map

            def _fake_export_resampled(**kwargs):
                self.assertEqual(kwargs["source_result"], source_result)
                self.assertEqual(kwargs["resolution_m"], 0.2)
                yaml_path = Path(kwargs["yaml_path"]).resolve()
                return MapExportResult(
                    yaml_path=yaml_path,
                    png_path=yaml_path.with_suffix(".png"),
                    resolution_m=float(kwargs["resolution_m"]),
                    origin_xyz=source_result.origin_xyz,
                    min_bound_xyz=source_result.min_bound_xyz,
                    max_bound_xyz=source_result.max_bound_xyz,
                    width_px=10,
                    height_px=20,
                    occupied_cells=8,
                    free_cells=180,
                    unknown_cells=12,
                    debug={
                        "target_dimensions": [10, 20],
                        "resample_policy": "occupied_over_unknown_over_free",
                    },
                )

            warehouse_builder_module._export_resampled_occupancy_map = _fake_export_resampled
            try:
                builder._export_resampled_omap_map_from_source(
                    yaml_path=tmp_path / "mapf_map.yaml",
                    map_spec=map_spec,
                    source_result=source_result,
                )
            finally:
                warehouse_builder_module._export_resampled_occupancy_map = original_export_resampled

            debug = builder._map_export_debug[str(tmp_path / "mapf_map.yaml")]
            self.assertEqual(debug["rasterization_mode"], "omap_generator_resampled")
            self.assertEqual(debug["source_rasterization_mode"], "omap_generator")
            self.assertEqual(debug["health_status"], "ok")
            self.assertEqual(debug["target_dimensions"], [10, 20])
            self.assertEqual(debug["resample_policy"], "occupied_over_unknown_over_free")

    def test_path_matches_any_subtree_treats_descendants_as_excluded(self) -> None:
        builder = object.__new__(RandomizedWarehouseBuilder)

        self.assertTrue(
            builder._path_matches_any_subtree(
                "/World/Env/Warehouse/Forklift_01/Collision/Mesh",
                {"/World/Env/Warehouse/Forklift_01"},
            )
        )
        self.assertTrue(
            builder._path_matches_any_subtree(
                "/World/Env/Warehouse/Forklift_01",
                {"/World/Env/Warehouse/Forklift_01"},
            )
        )
        self.assertFalse(
            builder._path_matches_any_subtree(
                "/World/Env/Warehouse/Forklift_02/Collision/Mesh",
                {"/World/Env/Warehouse/Forklift_01"},
            )
        )

    def test_ordered_candidate_zones_prefers_unused_nearest_zone(self) -> None:
        builder = object.__new__(RandomizedWarehouseBuilder)
        zones = [
            PlacementZone("zone_a", "anchor", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
            PlacementZone("zone_b", "anchor", (8.0, 0.0, 0.0), (10.0, 2.0, 2.0)),
            PlacementZone("zone_c", "anchor", (4.0, 0.0, 0.0), (6.0, 2.0, 2.0)),
        ]

        ordered = builder._ordered_candidate_zones_for_object(
            original_position=np.array([5.1, 1.1, 0.0], dtype=float),
            candidate_zones=zones,
            reserved_zone_ids={"zone_c"},
            prefer_unused=True,
        )

        self.assertEqual([zone.zone_id for zone in ordered], ["zone_b", "zone_a", "zone_c"])

    def test_assign_unique_anchor_zones_minimizes_total_distance(self) -> None:
        builder = object.__new__(RandomizedWarehouseBuilder)
        zones = [
            PlacementZone("south_east", "anchor", (2.0, -7.0, 0.0), (6.0, -1.0, 2.0)),
            PlacementZone("south_west", "anchor", (-10.0, -7.0, 0.0), (-6.0, -1.0, 2.0)),
        ]

        assigned = builder._assign_unique_anchor_zones(
            object_positions_xy={
                "/World/Forklift": np.array([4.0, -2.0], dtype=float),
                "/World/Forklift_01": np.array([-8.0, -2.0], dtype=float),
            },
            candidate_zones=zones,
        )

        self.assertEqual(assigned["/World/Forklift"].zone_id, "south_east")
        self.assertEqual(assigned["/World/Forklift_01"].zone_id, "south_west")

if __name__ == "__main__":
    unittest.main()
