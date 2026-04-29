from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from isaac_sim.stage_bringups import build_stage_warehouse_randomized as warehouse_cli


class BuildStageWarehouseRandomizedTests(unittest.TestCase):
    def test_plan_bundle_specs_defaults_to_first_template(self) -> None:
        specs = warehouse_cli.plan_bundle_specs(
            available_template_ids=["10", "2", "1"],
            base_seed=123,
        )

        self.assertEqual(
            [spec.variant_id for spec in specs],
            [warehouse_cli.DEFAULT_RANDOMIZATION_STRENGTH] * 5,
        )
        self.assertTrue(all(spec.template_id == "1" for spec in specs))
        self.assertEqual(
            [spec.scene_id for spec in specs],
            ["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
        )
        self.assertEqual([spec.template_scene_index for spec in specs], [1, 2, 3, 4, 5])

    def test_plan_bundle_specs_rejects_scene_prefix_with_all_templates(self) -> None:
        with self.assertRaisesRegex(ValueError, "--scene-id"):
            warehouse_cli.plan_bundle_specs(
                available_template_ids=["1", "2"],
                all_template=True,
                scene_id_prefix="batch",
                base_seed=1,
            )

    def test_plan_bundle_specs_expands_all_templates_as_flat_siblings(self) -> None:
        specs = warehouse_cli.plan_bundle_specs(
            available_template_ids=["2", "1"],
            all_template=True,
            base_seed=99,
        )

        self.assertEqual(len(specs), 10)
        self.assertEqual(
            [spec.scene_id for spec in specs[:5]],
            ["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
        )
        self.assertEqual(
            [spec.scene_id for spec in specs[5:]],
            ["scene_6", "scene_7", "scene_8", "scene_9", "scene_10"],
        )
        self.assertTrue(all(spec.variant_id == "balanced" for spec in specs))

    def test_plan_bundle_specs_all_templates_can_produce_scene_1_to_25(self) -> None:
        specs = warehouse_cli.plan_bundle_specs(
            available_template_ids=["5", "3", "1", "4", "2"],
            all_template=True,
            base_seed=99,
        )

        self.assertEqual(len(specs), 25)
        self.assertEqual(specs[0].scene_id, "scene_1")
        self.assertEqual(specs[-1].scene_id, "scene_25")
        self.assertEqual(
            [spec.scene_id for spec in specs[5:10]],
            ["scene_6", "scene_7", "scene_8", "scene_9", "scene_10"],
        )

    def test_plan_bundle_specs_uses_scene_id_prefix_for_single_template(self) -> None:
        specs = warehouse_cli.plan_bundle_specs(
            available_template_ids=["1", "2", "3"],
            requested_template_id="3",
            scene_id_prefix="trial",
            base_seed=5,
        )

        self.assertEqual(specs[0].scene_id, "trial_scene_11")
        self.assertEqual(specs[-1].scene_id, "trial_scene_15")

    def test_plan_bundle_specs_supports_strength_and_scene_count(self) -> None:
        specs = warehouse_cli.plan_bundle_specs(
            available_template_ids=["1"],
            requested_template_id="1",
            base_seed=9,
            randomization_strength="messy",
            scenes_per_template=2,
        )

        self.assertEqual([spec.variant_id for spec in specs], ["messy", "messy"])
        self.assertEqual(
            [spec.scene_id for spec in specs],
            ["scene_1", "scene_2"],
        )

    def test_parse_selected_variant_ids_supports_base_only_shortcut(self) -> None:
        variant_ids = warehouse_cli._parse_selected_variant_ids(
            variants=[],
            base_only=True,
        )

        self.assertEqual(variant_ids, ("base",))

    def test_parse_selected_variant_ids_rejects_conflicting_args(self) -> None:
        with self.assertRaisesRegex(ValueError, "--base-only"):
            warehouse_cli._parse_selected_variant_ids(
                variants=["balanced"],
                base_only=True,
            )

    def test_parse_randomization_strength_defaults_to_balanced(self) -> None:
        self.assertEqual(
            warehouse_cli._parse_randomization_strength(
                randomization_strength="",
                variants=[],
                base_only=False,
            ),
            "balanced",
        )

    def test_parse_selected_variant_ids_rejects_multiple_variants(self) -> None:
        with self.assertRaisesRegex(ValueError, "one randomization strength"):
            warehouse_cli._parse_selected_variant_ids(
                variants=["balanced,open"],
                base_only=False,
            )

    def test_derive_bundle_seed_is_stable_and_variant_specific(self) -> None:
        first_seed = warehouse_cli._derive_bundle_seed(42, "1", "balanced")
        second_seed = warehouse_cli._derive_bundle_seed(42, "1", "balanced")
        open_seed = warehouse_cli._derive_bundle_seed(42, "1", "open")
        second_scene_seed = warehouse_cli._derive_bundle_seed(42, "1", "balanced", 2)

        self.assertEqual(first_seed, second_seed)
        self.assertNotEqual(first_seed, open_seed)
        self.assertNotEqual(first_seed, second_scene_seed)

    def test_collection_metadata_payload_uses_instruction_selector_contract(self) -> None:
        payload = warehouse_cli.build_collection_metadata_payload(
            "go to the forklift near the shelf"
        )

        self.assertEqual(payload["language_instruction"], "go to the forklift near the shelf")
        self.assertEqual(payload["focus_selector"], warehouse_cli.DEFAULT_FOCUS_SELECTOR_ID)
        self.assertEqual(payload["robot_team_mode"], warehouse_cli.DEFAULT_ROBOT_TEAM_MODE)

    def test_write_and_load_collection_metadata_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_root_dir = Path(tmpdir)

            path = warehouse_cli._write_collection_metadata(
                scene_root_dir,
                "go to the forklift near the shelf",
            )
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))

            self.assertEqual(path.name, warehouse_cli.DEFAULT_COLLECTION_METADATA_FILENAME)
            self.assertEqual(payload["language_instruction"], "go to the forklift near the shelf")
            self.assertEqual(
                warehouse_cli._load_collection_metadata_language(scene_root_dir),
                "go to the forklift near the shelf",
            )


if __name__ == "__main__":
    unittest.main()
