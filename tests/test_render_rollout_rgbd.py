from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from isaac_sim.rendering.render_rollout_rgbd import (
    DEFAULT_CAMERA_CONFIG_PATH,
    _build_robot_render_active_masks,
    _depth_frame_m_to_uint16_mm,
    _discover_randomized_warehouse_render_targets,
    _load_camera_settings,
    _prepare_camera_output_dirs,
    _recorded_pose_samples_from_velocity,
    _remove_render_generated_artifacts,
    _render_frame_name,
    _resize_depth_frame_mm,
    _resize_rgb_frame,
    _rollout_has_render_outputs,
    _validate_unique_render_frame_names,
)
from isaac_sim.rendering.rollout_io import (
    RobotPose,
    RolloutData,
    RolloutRobotData,
    VelocitySample,
    load_rollout,
    resolve_rollout_scene_usd_path,
    temporary_team_config_file,
)


class RenderRolloutRGBDTest(unittest.TestCase):
    def test_load_rollout_and_temp_team_config_preserve_robot_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "scene_1" / "rollouts" / "1"
            rollout_dir.mkdir(parents=True)
            (rollout_dir / "jackal_velocity.csv").write_text(
                "timestamp_ns,vx,vy,wz,x,y,yaw\n0,0,0,0,1,2,0\n",
                encoding="utf-8",
            )
            run_config = {
                "run_id": 1,
                "language_instruction": "go to the forklift near the shelf",
                "record_settings": {"source": "simulator_tf_and_odometry"},
                "team_config": {
                    "environment": {"mapf_map": "mapf_map.yaml"},
                    "robots": [
                        {
                            "name": "jackal",
                            "model": "jackal",
                            "initial_pose": {"x": 1.0, "y": 2.0, "z": 0.0, "yaw": 0.0},
                            "goal_pose": {"x": 3.0, "y": 4.0, "z": 0.0, "yaw": 1.0},
                        }
                    ],
                },
            }
            (rollout_dir / "run_config.yaml").write_text(
                yaml.safe_dump(run_config, sort_keys=False),
                encoding="utf-8",
            )

            rollout = load_rollout(rollout_dir)
            self.assertEqual(rollout.robots[0].model, "jackal")

            with temporary_team_config_file(rollout) as temp_config:
                payload = yaml.safe_load(temp_config.read_text(encoding="utf-8"))
            self.assertEqual(payload["robots"][0]["model"], "jackal")

    def test_load_rollout_defaults_model_to_robot_name_for_legacy_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "scene_1" / "rollouts" / "1"
            rollout_dir.mkdir(parents=True)
            (rollout_dir / "limo_velocity.csv").write_text(
                "timestamp_ns,vx,vy,wz,x,y,yaw\n0,0,0,0,1,2,0\n",
                encoding="utf-8",
            )
            run_config = {
                "run_id": 1,
                "team_config": {
                    "robots": [
                        {
                            "name": "limo",
                            "initial_pose": {"x": 1.0, "y": 2.0, "z": 0.0, "yaw": 0.0},
                            "goal_pose": {"x": 3.0, "y": 4.0, "z": 0.0, "yaw": 1.0},
                        }
                    ],
                },
            }
            (rollout_dir / "run_config.yaml").write_text(
                yaml.safe_dump(run_config, sort_keys=False),
                encoding="utf-8",
            )

            rollout = load_rollout(rollout_dir)
            self.assertEqual(rollout.robots[0].model, "limo")

    def test_load_rollout_reads_pose_columns_when_cmd_vel_columns_are_appended(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "scene_1" / "rollouts" / "1"
            rollout_dir.mkdir(parents=True)
            (rollout_dir / "nova_carter_velocity.csv").write_text(
                "timestamp_ns,vx,vy,wz,x,y,yaw,cmd_vel_timestamp_ns,cmd_vx,cmd_vy,cmd_wz\n"
                "100,0.1,0.2,0.3,1.0,2.0,0.4,90,0.5,0.0,0.6\n",
                encoding="utf-8",
            )
            (rollout_dir / "run_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "run_id": 1,
                        "record_settings": {"source": "simulator_tf_and_odometry"},
                        "team_config": {
                            "robots": [
                                {
                                    "name": "nova_carter",
                                    "model": "nova_carter",
                                    "initial_pose": {"x": 0, "y": 0, "z": 0, "yaw": 0},
                                    "goal_pose": {"x": 1, "y": 0, "z": 0, "yaw": 0},
                                }
                            ]
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            rollout = load_rollout(rollout_dir)
            sample = rollout.robots[0].velocity_samples[0]
            self.assertTrue(sample.has_pose)
            self.assertEqual((sample.x, sample.y, sample.yaw), (1.0, 2.0, 0.4))
            self.assertEqual(
                (sample.cmd_vel_timestamp_ns, sample.cmd_vx, sample.cmd_vy, sample.cmd_wz),
                (90, 0.5, 0.0, 0.6),
            )

    def test_load_rollout_recovers_model_from_bundle_team_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir) / "scene_1"
            rollout_dir = bundle_dir / "rollouts" / "2"
            rollout_dir.mkdir(parents=True)
            (rollout_dir / "agent_alpha_velocity.csv").write_text(
                "timestamp_ns,vx,vy,wz,x,y,yaw\n0,0,0,0,1,2,0\n",
                encoding="utf-8",
            )
            (bundle_dir / "team_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "rollouts": [
                            {
                                "id": 2,
                                "robots": [
                                    {
                                        "name": "agent_alpha",
                                        "model": "jackal",
                                        "initial_pose": {"x": 0, "y": 0, "z": 0, "yaw": 0},
                                        "goal_pose": {"x": 1, "y": 0, "z": 0, "yaw": 0},
                                    }
                                ],
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (rollout_dir / "run_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "run_id": 2,
                        "team_config": {
                            "robots": [
                                {
                                    "name": "agent_alpha",
                                    "initial_pose": {"x": 0, "y": 0, "z": 0, "yaw": 0},
                                    "goal_pose": {"x": 1, "y": 0, "z": 0, "yaw": 0},
                                }
                            ]
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            rollout = load_rollout(rollout_dir)
            self.assertEqual(rollout.robots[0].model, "jackal")

    def test_scene_usd_resolves_from_rollout_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir) / "scene_3"
            rollout_dir = bundle_dir / "rollouts" / "7"
            rollout_dir.mkdir(parents=True)
            scene_usd = bundle_dir / "scene.usd"
            scene_usd.write_text("#usda 1.0\n", encoding="utf-8")

            rollout = load_rollout_fixture(rollout_dir)
            self.assertEqual(resolve_rollout_scene_usd_path(rollout), scene_usd.resolve())

    def test_camera_config_parses_two_bev_cameras_and_output_resolution(self) -> None:
        settings = _load_camera_settings(DEFAULT_CAMERA_CONFIG_PATH)
        self.assertEqual(settings.output_resolution, (224, 224))
        self.assertEqual(len(settings.bev_cameras), 2)
        self.assertEqual(settings.robot_camera.mode, "asset_if_available")
        self.assertIn("third_person", settings.robot_camera.asset_camera_reject_tokens)
        self.assertEqual(settings.robot_camera.model_overrides["limo"]["mode"], "virtual_only")

    def test_prepare_camera_output_dirs_includes_robot_and_bev_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dirs = _prepare_camera_output_dirs(
                root / "rgb",
                root / "depth",
                ("jackal", "bev_half_south"),
            )
            self.assertTrue(output_dirs["jackal"][0].is_dir())
            self.assertTrue(output_dirs["jackal"][1].is_dir())
            self.assertTrue(output_dirs["bev_half_south"][0].is_dir())
            self.assertTrue(output_dirs["bev_half_south"][1].is_dir())

    def test_robot_camera_active_mask_stops_at_each_robot_last_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "scene_1" / "rollouts" / "1"
            rollout_dir.mkdir(parents=True)
            rollout = load_rollout_fixture(rollout_dir)
            nova = rollout.robots[0]
            short_robot = nova.__class__(
                name="nova_carter",
                model="nova_carter",
                initial_pose=nova.initial_pose,
                goal_pose=nova.goal_pose,
                velocity_path=nova.velocity_path,
                velocity_samples=(
                    VelocitySample(0, 0.0, 0.0, 0.0),
                    VelocitySample(1_000_000_000, 0.0, 0.0, 0.0),
                    VelocitySample(2_000_000_000, 0.0, 0.0, 0.0),
                ),
                tracker_path=None,
                tracker_samples=(),
            )
            long_robot = nova.__class__(
                name="jackal",
                model="jackal",
                initial_pose=nova.initial_pose,
                goal_pose=nova.goal_pose,
                velocity_path=rollout_dir / "jackal_velocity.csv",
                velocity_samples=(
                    VelocitySample(0, 0.0, 0.0, 0.0),
                    VelocitySample(1_000_000_000, 0.0, 0.0, 0.0),
                    VelocitySample(2_000_000_000, 0.0, 0.0, 0.0),
                    VelocitySample(3_000_000_000, 0.0, 0.0, 0.0),
                    VelocitySample(4_000_000_000, 0.0, 0.0, 0.0),
                ),
                tracker_path=None,
                tracker_samples=(),
            )
            rollout = rollout.__class__(
                rollout_id=rollout.rollout_id,
                rollout_dir=rollout.rollout_dir,
                run_config_path=rollout.run_config_path,
                created_at=rollout.created_at,
                language_instruction=rollout.language_instruction,
                record_settings=rollout.record_settings,
                team_config_snapshot=rollout.team_config_snapshot,
                robots=(short_robot, long_robot),
                replay_timestamps_ns=(
                    0,
                    1_000_000_000,
                    2_000_000_000,
                    3_000_000_000,
                    4_000_000_000,
                ),
            )

            masks = _build_robot_render_active_masks(rollout, rollout.replay_timestamps_ns)

            self.assertEqual(masks["nova_carter"], [True, True, True, False, False])
            self.assertEqual(masks["jackal"], [True, True, True, True, True])

    def test_recorded_pose_samples_deduplicate_repeated_timestamps(self) -> None:
        robot = RolloutRobotData(
            name="nova_carter",
            model="nova_carter",
            initial_pose=RobotPose(0.0, 0.0, 0.0, 0.0),
            goal_pose=RobotPose(1.0, 0.0, 0.0, 0.0),
            velocity_path=Path("nova_carter_velocity.csv"),
            velocity_samples=(
                VelocitySample(10, 0.0, 0.0, 0.0, x=1.0, y=1.0, yaw=0.1),
                VelocitySample(10, 0.0, 0.0, 0.0, x=2.0, y=2.0, yaw=0.2),
                VelocitySample(20, 0.0, 0.0, 0.0, x=3.0, y=3.0, yaw=0.3),
            ),
            tracker_path=None,
            tracker_samples=(),
        )
        rollout = RolloutData(
            rollout_id=1,
            rollout_dir=Path("."),
            run_config_path=Path("run_config.yaml"),
            created_at="",
            language_instruction="",
            record_settings={"source": "simulator_tf_and_odometry"},
            team_config_snapshot={},
            robots=(robot,),
            replay_timestamps_ns=(10, 20),
        )

        poses = _recorded_pose_samples_from_velocity(rollout, robot)

        self.assertEqual([pose.timestamp_ns for pose in poses], [10, 20])
        self.assertEqual((poses[0].x, poses[0].y, poses[0].yaw), (2.0, 2.0, 0.2))

    def test_render_frame_name_uses_timestamp_and_rejects_duplicates(self) -> None:
        rollout = RolloutData(
            rollout_id=3,
            rollout_dir=Path("."),
            run_config_path=Path("run_config.yaml"),
            created_at="",
            language_instruction="",
            record_settings={},
            team_config_snapshot={},
            robots=(),
            replay_timestamps_ns=(),
        )

        self.assertEqual(_render_frame_name(123456789), "frame_123456789.png")
        _validate_unique_render_frame_names(rollout, (123456789, 223456789))
        with self.assertRaisesRegex(RuntimeError, "duplicate render frame timestamp names"):
            _validate_unique_render_frame_names(rollout, (123456789, 123456789))

    def test_image_resizing_preserves_rgb_and_depth_formats(self) -> None:
        rgb = np.zeros((10, 20, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        resized_rgb = _resize_rgb_frame(rgb, (224, 224))
        self.assertEqual(resized_rgb.shape, (224, 224, 3))
        self.assertEqual(resized_rgb.dtype, np.uint8)

        depth_m = np.ones((10, 20), dtype=np.float32) * 1.25
        depth_mm = _depth_frame_m_to_uint16_mm(depth_m)
        resized_depth = _resize_depth_frame_mm(depth_mm, (224, 224))
        self.assertEqual(resized_depth.shape, (224, 224))
        self.assertEqual(resized_depth.dtype, np.uint16)
        self.assertEqual(int(resized_depth[0, 0]), 1250)

    def test_batch_discovery_skips_failed_missing_and_existing_rollouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_dir = root / "scene_1"
            rollouts_dir = scene_dir / "rollouts"
            rollouts_dir.mkdir(parents=True)
            _write_scene_team_config(scene_dir, [1, 2, 3, 4])
            _write_minimal_run_config(rollouts_dir / "1", 1)
            _write_minimal_run_config(rollouts_dir / "2", 2)
            _write_minimal_run_config(rollouts_dir / "3_failed", 3)
            _write_minimal_run_config(rollouts_dir / "1_old", 1)
            (rollouts_dir / "2" / "rgb" / "nova_carter").mkdir(parents=True)
            (rollouts_dir / "2" / "rgb" / "nova_carter" / "frame_000000.png").write_bytes(
                b"rendered"
            )

            discovery = _discover_randomized_warehouse_render_targets(root)

            self.assertEqual(
                [(target.scene_id, target.rollout_id) for target in discovery.targets],
                [("scene_1", 1)],
            )
            self.assertEqual(
                [(entry["scene_id"], entry["rollout_id"]) for entry in discovery.skipped_failed_rollouts],
                [("scene_1", 3)],
            )
            self.assertEqual(
                [(entry["scene_id"], entry["rollout_id"]) for entry in discovery.skipped_missing_rollouts],
                [("scene_1", 4)],
            )
            self.assertEqual(
                [(entry["scene_id"], entry["rollout_id"]) for entry in discovery.skipped_existing_outputs],
                [("scene_1", 2)],
            )

    def test_batch_discovery_overwrite_selects_existing_render_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_dir = root / "scene_2"
            rollouts_dir = scene_dir / "rollouts"
            rollouts_dir.mkdir(parents=True)
            _write_scene_team_config(scene_dir, [1])
            _write_minimal_run_config(rollouts_dir / "1", 1)
            (rollouts_dir / "1" / "depth" / "bev_half_south").mkdir(parents=True)
            (rollouts_dir / "1" / "depth" / "bev_half_south" / "frame_000000.png").write_bytes(
                b"rendered"
            )

            discovery = _discover_randomized_warehouse_render_targets(root, overwrite=True)

            self.assertEqual(
                [(target.scene_id, target.rollout_id) for target in discovery.targets],
                [("scene_2", 1)],
            )
            self.assertFalse(discovery.skipped_existing_outputs)

    def test_batch_discovery_reports_skipped_scene_without_rollouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "scene_1").mkdir()

            discovery = _discover_randomized_warehouse_render_targets(root)

            self.assertFalse(discovery.targets)
            self.assertEqual(discovery.skipped_scenes[0]["scene_id"], "scene_1")
            self.assertEqual(discovery.skipped_scenes[0]["reason"], "missing rollouts directory")

    def test_remove_render_generated_artifacts_keeps_navigation_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "scene_1" / "rollouts" / "1"
            rollout_dir.mkdir(parents=True)
            (rollout_dir / "rgb" / "nova_carter").mkdir(parents=True)
            (rollout_dir / "depth" / "nova_carter").mkdir(parents=True)
            (rollout_dir / "rgb" / "nova_carter" / "frame_000000.png").write_bytes(b"rgb")
            (rollout_dir / "depth" / "nova_carter" / "frame_000000.png").write_bytes(b"depth")
            (rollout_dir / "render_manifest.csv").write_text("frame_index\n", encoding="utf-8")
            (rollout_dir / "replay_pose_nova_carter.csv").write_text("x\n", encoding="utf-8")
            (rollout_dir / "run_config.yaml").write_text("run_id: 1\n", encoding="utf-8")
            (rollout_dir / "nova_carter_velocity.csv").write_text(
                "timestamp_ns,vx,vy,wz\n0,0,0,0\n",
                encoding="utf-8",
            )

            self.assertTrue(_rollout_has_render_outputs(rollout_dir))
            _remove_render_generated_artifacts(rollout_dir)

            self.assertFalse((rollout_dir / "rgb").exists())
            self.assertFalse((rollout_dir / "depth").exists())
            self.assertFalse((rollout_dir / "render_manifest.csv").exists())
            self.assertFalse((rollout_dir / "replay_pose_nova_carter.csv").exists())
            self.assertTrue((rollout_dir / "run_config.yaml").is_file())
            self.assertTrue((rollout_dir / "nova_carter_velocity.csv").is_file())
            self.assertFalse(_rollout_has_render_outputs(rollout_dir))


def load_rollout_fixture(rollout_dir: Path):
    from isaac_sim.rendering.rollout_io import RobotPose, RolloutData, RolloutRobotData

    return RolloutData(
        rollout_id=7,
        rollout_dir=rollout_dir,
        run_config_path=rollout_dir / "run_config.yaml",
        created_at="",
        language_instruction="",
        record_settings={},
        team_config_snapshot={},
        robots=(
            RolloutRobotData(
                name="nova_carter",
                model="nova_carter",
                initial_pose=RobotPose(0.0, 0.0, 0.0, 0.0),
                goal_pose=RobotPose(1.0, 0.0, 0.0, 0.0),
                velocity_path=rollout_dir / "nova_carter_velocity.csv",
                velocity_samples=(),
                tracker_path=None,
                tracker_samples=(),
            ),
        ),
        replay_timestamps_ns=(),
    )


def _write_scene_team_config(scene_dir: Path, rollout_ids: list[int]) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rollouts": [
            {
                "id": rollout_id,
                "robots": [],
            }
            for rollout_id in rollout_ids
        ]
    }
    (scene_dir / "team_config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _write_minimal_run_config(rollout_dir: Path, rollout_id: int) -> None:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "run_config.yaml").write_text(
        yaml.safe_dump({"run_id": rollout_id, "team_config": {"robots": []}}, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
