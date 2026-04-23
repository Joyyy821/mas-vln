from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import yaml

from isaac_sim.stage_bringups.warehouse_randomized import templates


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)


def _shared_defaults_payload() -> dict:
    return {
        "template_assets": {
            "usd_root_dir": "isaac_sim/usd",
            "usd_filename_glob": "warehouse_template_*.usd",
            "template_id_regex": r"^warehouse_template_(\d+)\.usd$",
        },
        "metadata": {
            "template_family": "warehouse_randomized_test",
        },
        "nav2_map": {
            "resolution_m": 0.05,
            "origin_hint_xyz": [0.0, 0.0, 0.0],
            "min_bound_xyz": [0.0, 0.0, 0.0],
            "max_bound_xyz": [10.0, 10.0, 2.0],
        },
        "mapf_map": {
            "resolution_m": 0.2,
            "origin_hint_xyz": [0.0, 0.0, 0.0],
            "min_bound_xyz": [0.0, 0.0, 0.0],
            "max_bound_xyz": [10.0, 10.0, 2.0],
        },
        "focus_group_names": ["focus_objects"],
        "focus_distance_range_m": [2.0, 5.0],
        "keepout_zones": [
            {"zone_id": "main_keepout", "min_xy": [0.0, 0.0], "max_xy": [1.0, 1.0]},
            {"zone_id": "pad_keepout", "min_xy": [2.0, 2.0], "max_xy": [3.0, 3.0]},
        ],
        "placement_zones": [
            {"zone_id": "anchor_a", "zone_type": "anchor", "min_xyz": [0.0, 0.0, 0.0], "max_xyz": [1.0, 1.0, 1.0]},
            {"zone_id": "anchor_b", "zone_type": "anchor", "min_xyz": [1.0, 1.0, 0.0], "max_xyz": [2.0, 2.0, 1.0]},
            {"zone_id": "anchor_c", "zone_type": "anchor", "min_xyz": [2.0, 2.0, 0.0], "max_xyz": [3.0, 3.0, 1.0]},
            {"zone_id": "support_main", "zone_type": "support", "min_xyz": [0.0, 0.0, 0.0], "max_xyz": [5.0, 5.0, 1.0]},
        ],
        "object_groups": [
            {
                "group_id": "focus_objects",
                "selectors": [{"mode": "exact_path", "value": "/World/Forklift"}],
            },
            {
                "group_id": "appearance_props",
                "selectors": [{"mode": "regex", "value": "(?i)box"}],
                "required": False,
            },
            {
                "group_id": "forklifts",
                "selectors": [{"mode": "exact_path", "value": "/World/Forklift"}],
            },
            {
                "group_id": "small_loose_props",
                "selectors": [{"mode": "regex", "value": "(?i)box"}],
                "required": False,
            },
        ],
        "light_randomizers": [
            {
                "name": "warehouse_lights",
                "selectors": [{"mode": "regex", "value": "(?i)light"}],
                "intensity_range": [100.0, 200.0],
                "temperature_range": [3500.0, 4500.0],
            }
        ],
        "object_randomizers": [
            {
                "name": "appearance_props_colors",
                "policy": "appearance_only",
                "target_group_name": "appearance_props",
                "color_value_range": [[0.4, 0.8], [0.4, 0.8], [0.4, 0.8]],
            },
            {
                "name": "forklift_layout",
                "policy": "jittered_existing",
                "target_group_name": "forklifts",
                "anchor_zone_ids": ["anchor_a", "anchor_b", "anchor_c"],
                "keepout_zone_ids": ["main_keepout"],
                "snapped_yaw_deg": [0, 90],
                "xy_jitter_m": [0.5, 0.5],
                "yaw_jitter_deg": [-10.0, 10.0],
                "collision_margin_m": 0.25,
            },
            {
                "name": "copied_floor_clutter",
                "policy": "copied_from_group",
                "source_group_name": "small_loose_props",
                "support_zone_ids": ["support_main"],
                "keepout_zone_ids": ["main_keepout"],
                "spawn_count_range": [1, 3],
            },
        ],
    }


def _open_preset_payload() -> dict:
    return {
        "variant_id": "open",
        "description": "Open preset for tests.",
        "metadata": {
            "layout_class": "open",
        },
        "focus_distance_range_m": [3.0, 6.0],
        "light_randomizer_overrides": {
            "warehouse_lights": {
                "intensity_range": [250.0, 300.0],
            }
        },
        "object_randomizer_overrides": {
            "forklift_layout": {
                "anchor_zone_ids": ["anchor_a", "anchor_b"],
                "keepout_zone_ids": ["main_keepout", "pad_keepout"],
                "xy_jitter_m": [0.2, 0.2],
            },
            "copied_floor_clutter": {
                "enabled": False,
            },
        },
    }


class WarehouseRandomizedTemplateTests(unittest.TestCase):
    def test_discover_template_assets_parses_numeric_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config_dir = repo_root / "configs"
            usd_dir = repo_root / "isaac_sim" / "usd"
            usd_dir.mkdir(parents=True, exist_ok=True)
            (usd_dir / "warehouse_template_1.usd").write_text("#usda", encoding="utf-8")
            (usd_dir / "warehouse_template_20.usd").write_text("#usda", encoding="utf-8")
            (usd_dir / "ignore_me.usd").write_text("#usda", encoding="utf-8")
            _write_yaml(config_dir / "warehouse_shared.yaml", _shared_defaults_payload())

            shared_defaults = templates.load_shared_warehouse_defaults(
                repo_root,
                template_registry_dirs=[config_dir],
            )
            assets = templates.discover_template_assets(
                repo_root,
                shared_defaults=shared_defaults,
            )

            self.assertEqual(list(sorted(assets, key=int)), ["1", "20"])
            self.assertEqual(assets["1"].usd_path, (usd_dir / "warehouse_template_1.usd").resolve())
            self.assertEqual(assets["20"].usd_path, (usd_dir / "warehouse_template_20.usd").resolve())

    def test_compose_template_applies_overrides_and_supports_base_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config_dir = repo_root / "configs"
            usd_dir = repo_root / "isaac_sim" / "usd"
            usd_dir.mkdir(parents=True, exist_ok=True)
            (usd_dir / "warehouse_template_1.usd").write_text("#usda", encoding="utf-8")
            _write_yaml(config_dir / "warehouse_shared.yaml", _shared_defaults_payload())
            _write_yaml(config_dir / "warehouse_open.yaml", _open_preset_payload())

            shared_defaults = templates.load_shared_warehouse_defaults(
                repo_root,
                template_registry_dirs=[config_dir],
            )
            assets = templates.discover_template_assets(
                repo_root,
                shared_defaults=shared_defaults,
            )
            presets = templates.load_randomization_presets(
                repo_root,
                template_registry_dirs=[config_dir],
            )

            open_template = templates.compose_warehouse_template(
                assets["1"],
                shared_defaults=shared_defaults,
                preset=presets["open"],
            )
            self.assertEqual(open_template.template_id, "1")
            self.assertEqual(open_template.variant_id, "open")
            self.assertEqual(open_template.source_template_usd_path, (usd_dir / "warehouse_template_1.usd").resolve())
            self.assertEqual(open_template.shared_defaults_config_path, (config_dir / "warehouse_shared.yaml").resolve())
            self.assertEqual(open_template.preset_config_path, (config_dir / "warehouse_open.yaml").resolve())
            self.assertEqual(open_template.light_randomizers[0].intensity_range, (250.0, 300.0))
            self.assertEqual(open_template.focus_distance_range_m, (3.0, 6.0))
            self.assertEqual(open_template.metadata["layout_class"], "open")

            randomizer_names = [randomizer.name for randomizer in open_template.object_randomizers]
            self.assertEqual(randomizer_names, ["appearance_props_colors", "forklift_layout"])
            forklift_layout = next(
                randomizer for randomizer in open_template.object_randomizers if randomizer.name == "forklift_layout"
            )
            self.assertEqual(forklift_layout.anchor_zone_ids, ("anchor_a", "anchor_b"))
            self.assertEqual(forklift_layout.keepout_zone_ids, ("main_keepout", "pad_keepout"))
            self.assertEqual(forklift_layout.xy_jitter_m, (0.2, 0.2))

            base_template = templates.compose_warehouse_template(
                assets["1"],
                shared_defaults=shared_defaults,
                preset=None,
            )
            self.assertEqual(base_template.variant_id, "base")
            self.assertEqual(base_template.preset_config_path, None)
            self.assertEqual(base_template.light_randomizers, ())
            self.assertEqual(base_template.object_randomizers, ())


if __name__ == "__main__":
    unittest.main()
