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
    _load_camera_settings,
    _prepare_camera_output_dirs,
    _resize_depth_frame_mm,
    _resize_rgb_frame,
)
from isaac_sim.rendering.rollout_io import (
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


if __name__ == "__main__":
    unittest.main()
