from __future__ import annotations

import unittest

from isaac_sim.stage_bringups import build_stage_warehouse_randomized as warehouse_cli


class BuildStageWarehouseRandomizedTests(unittest.TestCase):
    def test_plan_bundle_specs_defaults_to_first_template(self) -> None:
        specs = warehouse_cli.plan_bundle_specs(
            available_template_ids=["10", "2", "1"],
            base_seed=123,
        )

        self.assertEqual(
            [spec.variant_id for spec in specs],
            list(warehouse_cli.DEFAULT_VARIANT_IDS),
        )
        self.assertTrue(all(spec.template_id == "1" for spec in specs))
        self.assertEqual(
            [spec.scene_id for spec in specs],
            [
                "template_1_base",
                "template_1_balanced",
                "template_1_messy",
                "template_1_open",
            ],
        )

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

        self.assertEqual(len(specs), 8)
        self.assertEqual(
            [spec.scene_id for spec in specs[:4]],
            [
                "template_1_base",
                "template_1_balanced",
                "template_1_messy",
                "template_1_open",
            ],
        )
        self.assertEqual(
            [spec.scene_id for spec in specs[4:]],
            [
                "template_2_base",
                "template_2_balanced",
                "template_2_messy",
                "template_2_open",
            ],
        )

    def test_plan_bundle_specs_uses_scene_id_prefix_for_single_template(self) -> None:
        specs = warehouse_cli.plan_bundle_specs(
            available_template_ids=["3"],
            requested_template_id="3",
            scene_id_prefix="trial",
            base_seed=5,
        )

        self.assertEqual(specs[0].scene_id, "trial_template_3_base")
        self.assertEqual(specs[-1].scene_id, "trial_template_3_open")

    def test_derive_bundle_seed_is_stable_and_variant_specific(self) -> None:
        first_seed = warehouse_cli._derive_bundle_seed(42, "1", "balanced")
        second_seed = warehouse_cli._derive_bundle_seed(42, "1", "balanced")
        open_seed = warehouse_cli._derive_bundle_seed(42, "1", "open")

        self.assertEqual(first_seed, second_seed)
        self.assertNotEqual(first_seed, open_seed)


if __name__ == "__main__":
    unittest.main()
