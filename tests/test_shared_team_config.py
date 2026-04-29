from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_TEAM_CONFIG_PATH = (
    REPO_ROOT / "ros2_ws" / "src" / "carters_goal" / "carters_goal" / "shared_team_config.py"
)


def _load_shared_team_config():
    packages_module = types.ModuleType("ament_index_python.packages")
    packages_module.get_package_share_directory = lambda package_name: str(
        REPO_ROOT / "ros2_ws" / "src" / package_name
    )

    root_module = types.ModuleType("ament_index_python")
    root_module.packages = packages_module

    previous_root = sys.modules.get("ament_index_python")
    previous_packages = sys.modules.get("ament_index_python.packages")
    sys.modules["ament_index_python"] = root_module
    sys.modules["ament_index_python.packages"] = packages_module
    try:
        spec = importlib.util.spec_from_file_location(
            "shared_team_config_for_tests",
            SHARED_TEAM_CONFIG_PATH,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {SHARED_TEAM_CONFIG_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_root is None:
            sys.modules.pop("ament_index_python", None)
        else:
            sys.modules["ament_index_python"] = previous_root
        if previous_packages is None:
            sys.modules.pop("ament_index_python.packages", None)
        else:
            sys.modules["ament_index_python.packages"] = previous_packages


shared_team_config = _load_shared_team_config()


class SharedTeamConfigTests(unittest.TestCase):
    def test_randomized_bundle_defaults_to_scene_rollouts_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "experiments" / "randomized_warehouse" / "template_1_base"
            bundle_dir.mkdir(parents=True)
            team_config_path = bundle_dir / "team_config.yaml"
            team_config_path.write_text("rollouts: []\n", encoding="utf-8")

            self.assertEqual(
                shared_team_config.rollout_run_dir("", team_config_path, 3),
                bundle_dir / "rollouts" / "3",
            )
            self.assertTrue(
                shared_team_config.use_rollout_id_run_directory("", team_config_path)
            )

    def test_randomized_bundle_marker_defaults_to_scene_rollouts_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "custom_scene"
            bundle_dir.mkdir()
            (bundle_dir / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
            team_config_path = bundle_dir / "team_config.yaml"
            team_config_path.write_text("rollouts: []\n", encoding="utf-8")

            self.assertEqual(
                shared_team_config.rollout_run_dir("", team_config_path, 7),
                bundle_dir / "rollouts" / "7",
            )

    def test_explicit_rollouts_dir_uses_rollout_id_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            team_config_path = tmp_path / "team_config.yaml"
            team_config_path.write_text("rollouts: []\n", encoding="utf-8")
            rollouts_dir = tmp_path / "rollouts"

            self.assertEqual(
                shared_team_config.rollout_run_dir(str(rollouts_dir), team_config_path, 5),
                rollouts_dir / "5",
            )
            self.assertTrue(
                shared_team_config.use_rollout_id_run_directory(
                    str(rollouts_dir),
                    team_config_path,
                )
            )

    def test_explicit_bundle_dir_appends_rollouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "experiments" / "randomized_warehouse" / "template_1_base"
            bundle_dir.mkdir(parents=True)
            team_config_path = bundle_dir / "team_config.yaml"
            team_config_path.write_text("rollouts: []\n", encoding="utf-8")

            self.assertEqual(
                shared_team_config.rollout_run_dir(str(bundle_dir), team_config_path, 2),
                bundle_dir / "rollouts" / "2",
            )

    def test_non_bundle_config_preserves_yaml_stem_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            experiments_dir = tmp_path / "experiments"
            config_dir = tmp_path / "config"
            config_dir.mkdir()
            team_config_path = config_dir / "warehouse_team_config.yaml"
            team_config_path.write_text("robots: []\n", encoding="utf-8")

            self.assertEqual(
                shared_team_config.rollout_run_dir(str(experiments_dir), team_config_path, 4),
                experiments_dir / "warehouse_team_config" / "4",
            )
            self.assertFalse(
                shared_team_config.use_rollout_id_run_directory(
                    str(experiments_dir),
                    team_config_path,
                )
            )


if __name__ == "__main__":
    unittest.main()
