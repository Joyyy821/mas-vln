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

    def test_select_bundle_specs_fills_missing_slot_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_root_dir = Path(tmpdir)
            for scene_number in range(1, 10):
                bundle_dir = scene_root_dir / f"scene_{scene_number}"
                bundle_dir.mkdir(parents=True)
                (bundle_dir / "manifest.yaml").write_text("ok: true\n", encoding="utf-8")

            specs = warehouse_cli.select_bundle_specs_for_run(
                scene_root_dir=scene_root_dir,
                available_template_ids=["1", "2", "3"],
                requested_template_id="2",
                base_seed=99,
                scenes_per_template=1,
                overwrite=False,
            )

            self.assertEqual([spec.scene_id for spec in specs], ["scene_10"])

    def test_select_bundle_specs_appends_after_highest_existing_scene_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_root_dir = Path(tmpdir)
            for scene_number in range(1, 11):
                bundle_dir = scene_root_dir / f"scene_{scene_number}"
                bundle_dir.mkdir(parents=True)
                (bundle_dir / "manifest.yaml").write_text("ok: true\n", encoding="utf-8")

            specs = warehouse_cli.select_bundle_specs_for_run(
                scene_root_dir=scene_root_dir,
                available_template_ids=["1", "2", "3"],
                requested_template_id="2",
                base_seed=99,
                scenes_per_template=1,
                overwrite=False,
            )

            self.assertEqual([spec.scene_id for spec in specs], ["scene_11"])

    def test_select_bundle_specs_all_templates_appends_new_scene_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_root_dir = Path(tmpdir)
            for scene_number in range(1, 11):
                bundle_dir = scene_root_dir / f"scene_{scene_number}"
                bundle_dir.mkdir(parents=True)
                (bundle_dir / "manifest.yaml").write_text("ok: true\n", encoding="utf-8")

            specs = warehouse_cli.select_bundle_specs_for_run(
                scene_root_dir=scene_root_dir,
                available_template_ids=["1", "2", "3"],
                all_template=True,
                base_seed=99,
                scenes_per_template=1,
                overwrite=False,
            )

            self.assertEqual(
                [spec.scene_id for spec in specs],
                ["scene_11", "scene_12", "scene_13"],
            )
            self.assertEqual([spec.template_id for spec in specs], ["1", "2", "3"])

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
        retry_seed = warehouse_cli._derive_bundle_seed(42, "1", "balanced", 1, 2)

        self.assertEqual(first_seed, second_seed)
        self.assertNotEqual(first_seed, open_seed)
        self.assertNotEqual(first_seed, second_scene_seed)
        self.assertNotEqual(first_seed, retry_seed)

    def test_build_scene_bundle_with_retries_resamples_seed(self) -> None:
        calls = []
        original_build_scene_bundle = warehouse_cli._build_scene_bundle

        def fake_build_scene_bundle(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("placement failed")
            return object(), "sim_app"

        warehouse_cli._build_scene_bundle = fake_build_scene_bundle
        try:
            spec = warehouse_cli.BundleBuildSpec(
                template_id="2",
                variant_id="balanced",
                scene_id="scene_10",
                seed=1,
                template_scene_index=5,
                scene_number=10,
            )

            result, sim_app, failure = warehouse_cli._build_scene_bundle_with_retries(
                bundle_spec=spec,
                base_seed=99,
                max_scene_attempts=2,
                template=object(),
                scene_root_dir=Path("/tmp/randomized_warehouse_test"),
                robot_models=["nova_carter"],
                robot_count=3,
                rollout_count=5,
                language_instruction="go to the forklift near the shelf",
                enable_ros2_runtime=False,
                rollout_control_topic="/control",
                rollout_reset_done_topic="/done",
                overwrite=False,
                headless=True,
                keep_sim_running=False,
                scene_only=True,
            )
        finally:
            warehouse_cli._build_scene_bundle = original_build_scene_bundle

        self.assertIsNotNone(result)
        self.assertEqual(sim_app, "sim_app")
        self.assertIsNone(failure)
        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[0]["overwrite"])
        self.assertTrue(calls[1]["overwrite"])
        self.assertNotEqual(calls[0]["seed"], calls[1]["seed"])

    def test_build_scene_bundle_with_retries_returns_failure_after_budget(self) -> None:
        calls = []
        original_build_scene_bundle = warehouse_cli._build_scene_bundle

        def fake_build_scene_bundle(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("still blocked")

        warehouse_cli._build_scene_bundle = fake_build_scene_bundle
        try:
            spec = warehouse_cli.BundleBuildSpec(
                template_id="2",
                variant_id="balanced",
                scene_id="scene_10",
                seed=1,
                template_scene_index=5,
                scene_number=10,
            )

            result, sim_app, failure = warehouse_cli._build_scene_bundle_with_retries(
                bundle_spec=spec,
                base_seed=99,
                max_scene_attempts=2,
                template=object(),
                scene_root_dir=Path("/tmp/randomized_warehouse_test"),
                robot_models=["nova_carter"],
                robot_count=3,
                rollout_count=5,
                language_instruction="go to the forklift near the shelf",
                enable_ros2_runtime=False,
                rollout_control_topic="/control",
                rollout_reset_done_topic="/done",
                overwrite=False,
                headless=True,
                keep_sim_running=False,
                scene_only=True,
            )
        finally:
            warehouse_cli._build_scene_bundle = original_build_scene_bundle

        self.assertIsNone(result)
        self.assertIsNone(sim_app)
        self.assertIsNotNone(failure)
        self.assertEqual(len(failure.attempts), 2)
        self.assertEqual(len(calls), 2)

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
