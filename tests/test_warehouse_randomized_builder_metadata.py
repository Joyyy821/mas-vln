from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from isaac_sim.stage_bringups.warehouse_randomized import builder as warehouse_builder_module
from isaac_sim.stage_bringups.warehouse_randomized.builder import RandomizedWarehouseBuilder
from isaac_sim.stage_bringups.warehouse_randomized.maps import MapExportResult
from isaac_sim.stage_bringups.warehouse_randomized.templates import (
    ObjectGroupSpec,
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
                selectors=(SelectorSpec(mode="exact_path", value="/World/Forklift"),),
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
    def test_team_config_and_manifest_include_template_and_variant_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            builder = object.__new__(RandomizedWarehouseBuilder)
            builder._scene_id = "template_2_open"
            builder._seed = 77
            builder._template = _make_template(tmpdir)
            builder._scene_usd_path = tmp_path / "scene.usd"
            builder._team_config_path = tmp_path / "team_config.yaml"
            builder._manifest_path = tmp_path / "scene_manifest.yaml"
            builder._rollouts_dir = tmp_path / "rollouts"
            builder._nav2_map_path = tmp_path / "nav2_map.yaml"
            builder._mapf_map_path = tmp_path / "mapf_map.yaml"
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

            self.assertEqual(manifest_payload["template"]["template_id"], "2")
            self.assertEqual(manifest_payload["template"]["variant_id"], "open")
            self.assertEqual(
                manifest_payload["template"]["source_template_usd_path"],
                str(tmp_path / "warehouse_template_2.usd"),
            )
            self.assertEqual(manifest_payload["validation_summary"]["template_id"], "2")
            self.assertEqual(manifest_payload["validation_summary"]["variant_id"], "open")
            self.assertEqual(team_config_payload["validation"]["nav2_map"], None)

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
