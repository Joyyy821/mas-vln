from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from isaac_sim.stage_bringups.warehouse_randomized.builder import RandomizedWarehouseBuilder
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
        nav2_map=TemplateMapSpec(0.05, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (10.0, 10.0, 2.0)),
        mapf_map=TemplateMapSpec(0.2, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (10.0, 10.0, 2.0)),
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
            builder._map_export_mode = "bbox"

            team_config_payload = builder._build_team_config_payload()
            manifest_payload = builder._build_manifest_payload()

            self.assertEqual(team_config_payload["template_id"], "2")
            self.assertEqual(team_config_payload["variant_id"], "open")
            self.assertEqual(team_config_payload["map_export_mode"], "bbox")
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
            self.assertEqual(team_config_payload["validation"]["map_export_mode"], "bbox")

            self.assertEqual(manifest_payload["template"]["template_id"], "2")
            self.assertEqual(manifest_payload["template"]["variant_id"], "open")
            self.assertEqual(manifest_payload["template"]["map_export_mode"], "bbox")
            self.assertEqual(
                manifest_payload["template"]["source_template_usd_path"],
                str(tmp_path / "warehouse_template_2.usd"),
            )
            self.assertEqual(manifest_payload["validation_summary"]["template_id"], "2")
            self.assertEqual(manifest_payload["validation_summary"]["variant_id"], "open")
            self.assertEqual(manifest_payload["validation_summary"]["map_export_mode"], "bbox")

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


if __name__ == "__main__":
    unittest.main()
