from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CARTERS_GOAL_SRC = REPO_ROOT / "ros2_ws" / "src" / "carters_goal"


def _load_batch_runner_module():
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
    sys.path.insert(0, str(CARTERS_GOAL_SRC))
    try:
        from carters_goal import randomized_warehouse_batch_runner

        return randomized_warehouse_batch_runner
    finally:
        try:
            sys.path.remove(str(CARTERS_GOAL_SRC))
        except ValueError:
            pass
        if previous_root is None:
            sys.modules.pop("ament_index_python", None)
        else:
            sys.modules["ament_index_python"] = previous_root
        if previous_packages is None:
            sys.modules.pop("ament_index_python.packages", None)
        else:
            sys.modules["ament_index_python.packages"] = previous_packages


batch_runner = _load_batch_runner_module()


def _write_team_config(path: Path, rollout_ids: list[int]) -> None:
    rollouts = []
    for rollout_id in rollout_ids:
        rollouts.append(
            {
                "id": rollout_id,
                "robots": [
                    {
                        "name": "nova_carter",
                        "model": "nova_carter",
                        "initial_pose": {"x": 0, "y": 0, "z": 0, "yaw": 0},
                        "goal_pose": {"x": 1, "y": 0, "z": 0, "yaw": 0},
                    }
                ],
            }
        )
    path.write_text(yaml.safe_dump({"environment": {}, "rollouts": rollouts}), encoding="utf-8")


def _make_scene(root: Path, scene_id: str, rollout_ids: list[int]) -> Path:
    scene_dir = root / scene_id
    scene_dir.mkdir(parents=True)
    (scene_dir / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
    (scene_dir / "mapf_map.yaml").write_text("image: mapf_map.png\n", encoding="utf-8")
    _write_team_config(scene_dir / "team_config.yaml", rollout_ids)
    return scene_dir


class RandomizedWarehouseBatchRunnerTests(unittest.TestCase):
    def test_discovers_scene_bundles_in_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_scene(root, "scene_10", [1])
            _make_scene(root, "scene_2", [1, 3])

            scenes = batch_runner.discover_scene_bundles(root)

            self.assertEqual([scene.scene_id for scene in scenes], ["scene_2", "scene_10"])
            self.assertEqual(scenes[0].rollout_ids, (1, 3))

    def test_preflight_reports_missing_required_scene_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scene_4").mkdir()

            with self.assertRaisesRegex(RuntimeError, "scene_4: missing"):
                batch_runner.discover_scene_bundles(root)

    def test_continue_starts_inside_selected_scene_then_larger_scene_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_scene(root, "scene_1", [1, 2])
            _make_scene(root, "scene_2", [1, 2, 3])
            _make_scene(root, "scene_3", [1])
            scenes = batch_runner.discover_scene_bundles(root)

            items = batch_runner.build_batch_items(
                scenes,
                continue_enabled=True,
                continue_scene_id="scene_2",
                continue_rollout_id=2,
            )

            self.assertEqual(
                [(item.scene.scene_id, item.rollout_id) for item in items],
                [("scene_2", 2), ("scene_2", 3), ("scene_3", 1)],
            )

    def test_child_launch_command_sets_scene_rollout_map_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_dir = _make_scene(root, "scene_1", [5])
            scene = batch_runner.discover_scene_bundles(root)[0]

            command = batch_runner.build_single_rollout_launch_command(
                scene=scene,
                rollout_id=5,
                launch_options={
                    "execution_backend": "timed_tracker",
                    "record_cmd_vel_topic_suffix": "cmd_vel",
                    "save_traj_plot": "true",
                    "experiments_dir": "",
                },
            )

            self.assertIn(f"team_config_file:={scene_dir / 'team_config.yaml'}", command)
            self.assertIn("rollout_id:=5", command)
            self.assertIn(f"map:={scene_dir / 'mapf_map.yaml'}", command)
            self.assertIn("overwrite_existing_rollout:=true", command)
            self.assertIn("execution_backend:=timed_tracker", command)
            self.assertIn("record_cmd_vel_topic_suffix:=cmd_vel", command)
            self.assertNotIn("save_traj_plot:=true", command)

    def test_success_cleanup_removes_tracker_diagnostics_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rollout_dir = Path(tmpdir) / "rollout"
            rollout_dir.mkdir()
            tracker_csv = rollout_dir / "mapf_timed_tracker_pid123_exec001_nova_carter.csv"
            tracker_csv.write_text("elapsed\n", encoding="utf-8")
            other_tracker_csv = rollout_dir / "mapf_timed_tracker_pid456_exec001_jackal.csv"
            other_tracker_csv.write_text("elapsed\n", encoding="utf-8")
            velocity_csv = rollout_dir / "nova_carter_velocity.csv"
            velocity_csv.write_text("timestamp_ns,vx,vy,wz\n", encoding="utf-8")
            run_config = rollout_dir / "run_config.yaml"
            run_config.write_text("run_id: 1\n", encoding="utf-8")
            plots_dir = rollout_dir / "tracking_plots"
            plots_dir.mkdir()
            (plots_dir / "summary.png").write_text("plot\n", encoding="utf-8")

            removed_files, removed_directories = batch_runner.cleanup_successful_rollout_artifacts(
                rollout_dir
            )

            self.assertEqual(removed_files, 2)
            self.assertEqual(removed_directories, 1)
            self.assertFalse(tracker_csv.exists())
            self.assertFalse(other_tracker_csv.exists())
            self.assertFalse(plots_dir.exists())
            self.assertTrue(velocity_csv.is_file())
            self.assertTrue(run_config.is_file())

    def test_success_cleanup_can_preserve_combined_xy_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rollout_dir = Path(tmpdir) / "rollout"
            rollout_dir.mkdir()
            tracker_csv = rollout_dir / "mapf_timed_tracker_pid123_exec001_nova_carter.csv"
            tracker_csv.write_text("elapsed\n", encoding="utf-8")
            plots_dir = rollout_dir / "tracking_plots"
            plots_dir.mkdir()
            combined_plot = plots_dir / "combined_xy_overlay.png"
            combined_plot.write_text("combined\n", encoding="utf-8")
            stale_agent_plot = plots_dir / "mapf_timed_tracker_pid123_exec001_nova_carter.png"
            stale_agent_plot.write_text("agent\n", encoding="utf-8")

            removed_files, removed_directories = batch_runner.cleanup_successful_rollout_artifacts(
                rollout_dir,
                preserve_combined_xy_overlay=True,
            )

            self.assertEqual(removed_files, 2)
            self.assertEqual(removed_directories, 0)
            self.assertFalse(tracker_csv.exists())
            self.assertFalse(stale_agent_plot.exists())
            self.assertTrue(combined_plot.is_file())


if __name__ == "__main__":
    unittest.main()
