from __future__ import annotations

import math
import unittest
import warnings

from isaac_sim.rendering.rollout_io import RobotPose, VelocitySample
from isaac_sim.rendering.trajectory_integration import (
    TimedPose,
    build_pose_trajectory,
    integrate_velocity_samples,
)


def _sample(timestamp_ns: int, vx: float, vy: float, wz: float) -> VelocitySample:
    return VelocitySample(timestamp_ns=timestamp_ns, vx=vx, vy=vy, wz=wz)


class TrajectoryIntegrationTest(unittest.TestCase):
    def test_straight_motion_integration(self) -> None:
        trajectory = integrate_velocity_samples(
            RobotPose(x=0.0, y=0.0, z=0.0, yaw=0.0),
            [_sample(0, 1.0, 0.0, 0.0), _sample(1_000_000_000, 0.0, 0.0, 0.0)],
            source_label="straight_motion",
        )

        pose = trajectory.pose_at(1_000_000_000)
        self.assertAlmostEqual(pose.x, 1.0, places=6)
        self.assertAlmostEqual(pose.y, 0.0, places=6)
        self.assertAlmostEqual(pose.yaw, 0.0, places=6)

    def test_pure_rotation_integration(self) -> None:
        angular_velocity = math.pi
        trajectory = integrate_velocity_samples(
            RobotPose(x=1.0, y=2.0, z=0.0, yaw=0.0),
            [_sample(0, 0.0, 0.0, angular_velocity), _sample(1_000_000_000, 0.0, 0.0, 0.0)],
            source_label="pure_rotation",
        )

        pose = trajectory.pose_at(1_000_000_000)
        self.assertAlmostEqual(pose.x, 1.0, places=6)
        self.assertAlmostEqual(pose.y, 2.0, places=6)
        self.assertAlmostEqual(pose.yaw, math.pi, places=6)

    def test_arc_motion_matches_exact_se2_update(self) -> None:
        angular_velocity = 1.0
        trajectory = integrate_velocity_samples(
            RobotPose(x=0.0, y=0.0, z=0.0, yaw=0.0),
            [_sample(0, 1.0, 0.0, angular_velocity), _sample(1_000_000_000, 0.0, 0.0, 0.0)],
            source_label="arc_motion",
        )

        pose = trajectory.pose_at(1_000_000_000)
        self.assertAlmostEqual(pose.x, math.sin(1.0), places=6)
        self.assertAlmostEqual(pose.y, 1.0 - math.cos(1.0), places=6)
        self.assertAlmostEqual(pose.yaw, 1.0, places=6)

    def test_lateral_body_motion_is_respected(self) -> None:
        trajectory = integrate_velocity_samples(
            RobotPose(x=0.0, y=0.0, z=0.0, yaw=0.0),
            [_sample(0, 0.0, 1.0, 0.0), _sample(1_000_000_000, 0.0, 0.0, 0.0)],
            source_label="lateral_motion",
        )

        pose = trajectory.pose_at(1_000_000_000)
        self.assertAlmostEqual(pose.x, 0.0, places=6)
        self.assertAlmostEqual(pose.y, 1.0, places=6)
        self.assertAlmostEqual(pose.yaw, 0.0, places=6)

    def test_tiny_angular_velocity_uses_straight_line_limit(self) -> None:
        trajectory = integrate_velocity_samples(
            RobotPose(x=0.0, y=0.0, z=0.0, yaw=0.0),
            [_sample(0, 1.0, 0.5, 1e-9), _sample(1_000_000_000, 0.0, 0.0, 0.0)],
            source_label="tiny_angular_velocity",
        )

        pose = trajectory.pose_at(1_000_000_000)
        self.assertAlmostEqual(pose.x, 1.0, places=6)
        self.assertAlmostEqual(pose.y, 0.5, places=6)
        self.assertAlmostEqual(pose.yaw, 1e-9, places=6)

    def test_unsorted_and_duplicate_timestamps_warn_and_keep_last_duplicate(self) -> None:
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            trajectory = integrate_velocity_samples(
                RobotPose(x=0.0, y=0.0, z=0.0, yaw=0.0),
                [
                    _sample(2_000_000_000, 1.0, 0.0, 0.0),
                    _sample(1_000_000_000, 5.0, 0.0, 0.0),
                    _sample(1_000_000_000, 2.0, 0.0, 0.0),
                    _sample(3_000_000_000, 0.0, 0.0, 0.0),
                ],
                source_label="unsorted_duplicate",
            )

        warning_messages = [str(item.message) for item in captured_warnings]
        self.assertTrue(any("non-monotonic" in message for message in warning_messages))
        self.assertTrue(any("duplicate timestamps" in message for message in warning_messages))

        pose = trajectory.pose_at(3_000_000_000)
        self.assertAlmostEqual(pose.x, 3.0, places=6)
        self.assertAlmostEqual(pose.y, 0.0, places=6)

    def test_pose_trajectory_interpolates_recorded_pose_samples(self) -> None:
        trajectory = build_pose_trajectory(
            [
                TimedPose(timestamp_ns=0, x=0.0, y=0.0, z=0.0, yaw=0.0),
                TimedPose(timestamp_ns=1_000_000_000, x=1.0, y=2.0, z=0.0, yaw=math.pi / 2.0),
            ],
            source_label="recorded_pose",
        )

        pose = trajectory.pose_at(500_000_000)
        self.assertAlmostEqual(pose.x, 0.5, places=6)
        self.assertAlmostEqual(pose.y, 1.0, places=6)
        self.assertAlmostEqual(pose.yaw, math.pi / 4.0, places=6)

    def test_pose_trajectory_interpolates_yaw_across_pi_with_shortest_turn(self) -> None:
        trajectory = build_pose_trajectory(
            [
                TimedPose(
                    timestamp_ns=0,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    yaw=math.radians(170.0),
                ),
                TimedPose(
                    timestamp_ns=1_000_000_000,
                    x=1.0,
                    y=0.0,
                    z=0.0,
                    yaw=math.radians(-170.0),
                ),
            ],
            source_label="wrapped_pose",
        )

        pose = trajectory.pose_at(500_000_000)
        self.assertAlmostEqual(pose.x, 0.5, places=6)
        self.assertAlmostEqual(abs(pose.yaw), math.pi, places=6)


if __name__ == "__main__":
    unittest.main()
