from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TEAM_CONFIG_UTILS_PATH = (
    REPO_ROOT / "ros2_ws" / "src" / "carters_nav2" / "launch" / "team_config_utils.py"
)


def _load_team_config_utils():
    spec = importlib.util.spec_from_file_location("team_config_utils", TEAM_CONFIG_UTILS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {TEAM_CONFIG_UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


team_config_utils = _load_team_config_utils()


def _write_team_config(path: Path, environment: dict[str, str]) -> None:
    payload = {
        "environment": environment,
        "robots": [
            {
                "name": "robot1",
                "initial_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                "goal_pose": {"x": 1.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class TeamConfigUtilsTests(unittest.TestCase):
    def test_relative_bundle_map_paths_resolve_next_to_team_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            bundle_dir = tmp_path / "template_1_base"
            maps_dir = tmp_path / "package_maps"
            bundle_dir.mkdir()
            maps_dir.mkdir()
            (bundle_dir / "nav2_map.yaml").write_text("image: nav2_map.png\n", encoding="utf-8")
            (bundle_dir / "mapf_map.yaml").write_text("image: mapf_map.png\n", encoding="utf-8")
            (maps_dir / "nav2_map.yaml").write_text("image: wrong.png\n", encoding="utf-8")

            team_config_path = bundle_dir / "team_config.yaml"
            _write_team_config(
                team_config_path,
                {
                    "nav2_map": "nav2_map.yaml",
                    "mapf_map": "mapf_map.yaml",
                },
            )

            config = team_config_utils.load_team_config(
                str(team_config_path),
                maps_dir=str(maps_dir),
            )

            self.assertEqual(config["nav2_map"], str(bundle_dir / "nav2_map.yaml"))
            self.assertEqual(config["mapf_map"], str(bundle_dir / "mapf_map.yaml"))

    def test_missing_host_absolute_map_path_falls_back_to_bundle_basename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            (bundle_dir / "mapf_map.yaml").write_text("image: mapf_map.png\n", encoding="utf-8")

            team_config_path = bundle_dir / "team_config.yaml"
            _write_team_config(
                team_config_path,
                {
                    "mapf_map": "/home/yjiao/missing/randomized_warehouse/template_1_base/mapf_map.yaml",
                },
            )

            config = team_config_utils.load_team_config(str(team_config_path))

            self.assertEqual(config["mapf_map"], str(bundle_dir / "mapf_map.yaml"))

    def test_package_maps_dir_remains_fallback_for_shared_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_dir = tmp_path / "config"
            maps_dir = tmp_path / "maps"
            config_dir.mkdir()
            maps_dir.mkdir()
            (maps_dir / "carter_warehouse_navigation.yaml").write_text(
                "image: carter_warehouse_navigation.png\n",
                encoding="utf-8",
            )

            team_config_path = config_dir / "warehouse_team_config.yaml"
            _write_team_config(
                team_config_path,
                {
                    "nav2_map": "carter_warehouse_navigation.yaml",
                },
            )

            config = team_config_utils.load_team_config(
                str(team_config_path),
                maps_dir=str(maps_dir),
            )

            self.assertEqual(
                config["nav2_map"],
                str(maps_dir / "carter_warehouse_navigation.yaml"),
            )


if __name__ == "__main__":
    unittest.main()
