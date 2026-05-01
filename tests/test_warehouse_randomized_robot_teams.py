from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from isaac_sim.stage_bringups.warehouse_randomized.robot_teams import (
    RobotTeamPolicy,
    build_fixed_robot_team,
    sample_priority_robot_team,
)
from isaac_sim.stage_bringups.warehouse_randomized.robots import (
    RobotAdapter,
    build_robot_adapter,
)
from isaac_sim.stage_bringups.warehouse_randomized.ros_bridge import _wheel_action_arrays


class WarehouseRandomizedRobotTeamTests(unittest.TestCase):
    def test_priority_team_uses_model_namespaces_without_duplicates(self) -> None:
        policy = RobotTeamPolicy(count_distribution=((4, 1.0),))
        team = sample_priority_robot_team(policy=policy, rng=np.random.default_rng(3))

        self.assertEqual(
            team,
            [
                {"name": "nova_carter", "model": "nova_carter"},
                {"name": "carter_v1", "model": "carter_v1"},
                {"name": "jackal", "model": "jackal"},
                {"name": "limo", "model": "limo"},
            ],
        )
        self.assertEqual(len({member["name"] for member in team}), 4)
        self.assertNotIn("robot1", {member["name"] for member in team})

    def test_fixed_duplicate_legacy_team_keeps_robot_number_namespaces(self) -> None:
        team = build_fixed_robot_team(model_ids=["nova_carter"], robot_count=3)

        self.assertEqual([member["name"] for member in team], ["robot1", "robot2", "robot3"])
        self.assertEqual([member["model"] for member in team], ["nova_carter"] * 3)

    def test_requested_robot_adapters_resolve_builtin_asset_paths(self) -> None:
        expected_paths = {
            "nova_carter": "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
            "carter_v1": "/Isaac/Robots/NVIDIA/Carter/carter_v1.usd",
            "jackal": "/Isaac/Robots/Clearpath/Jackal/jackal.usd",
            "limo": "/Isaac/Robots/AgilexRobotics/limo/limo.usd",
        }

        for model_id, usd_rel in expected_paths.items():
            adapter = build_robot_adapter(model_id)
            self.assertEqual(adapter.model_id, model_id)
            self.assertEqual(adapter.sensorless_usd_rel, usd_rel)

    def test_robot_adapter_no_longer_has_id_tag_builder(self) -> None:
        self.assertFalse(hasattr(RobotAdapter, "attach_numeric_id_tag"))

    def test_wheel_action_arrays_support_multiple_wheels_per_side(self) -> None:
        controller = SimpleNamespace(
            left_wheel_joint_indices=np.array([0, 2], dtype=np.int32),
            right_wheel_joint_indices=np.array([1, 3], dtype=np.int32),
            wheel_distance_m=0.5,
            wheel_radius_m=0.1,
        )

        indices, velocities = _wheel_action_arrays(
            controller,
            linear_x=1.0,
            angular_z=0.4,
        )

        self.assertEqual(indices.tolist(), [0, 2, 1, 3])
        self.assertTrue(np.allclose(velocities[:2], [9.0, 9.0]))
        self.assertTrue(np.allclose(velocities[2:], [11.0, 11.0]))


if __name__ == "__main__":
    unittest.main()
