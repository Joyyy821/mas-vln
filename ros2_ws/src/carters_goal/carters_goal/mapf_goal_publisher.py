# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Bool
from mapf_msgs.msg import GlobalPlan


class MapfGoalPublisher(Node):
    def __init__(self, default_goals: List[List[float]]) -> None:
        super().__init__("mapf_goal_publisher")

        self.declare_parameter("goal_topic", "/mapf_base/goal_for_each")
        self.declare_parameter("goal_init_topic", "/mapf_base/goal_init_flag")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_period_sec", 0.5)
        self.declare_parameter("publish_count", 5)
        self.declare_parameter("agent_num", len(default_goals))
        self.declare_parameter("wait_for_subscribers", True)
        self.declare_parameter("stop_on_plan", True)
        self.declare_parameter("global_plan_topic", "/mapf_base/global_plan")

        flat_default = [value for goal in default_goals for value in goal]
        self.declare_parameter("goal_array", flat_default)

        self._goal_topic = self.get_parameter("goal_topic").value
        self._goal_init_topic = self.get_parameter("goal_init_topic").value
        self._frame_id = self.get_parameter("frame_id").value
        self._publish_count = int(self.get_parameter("publish_count").value)
        self._agent_num = int(self.get_parameter("agent_num").value)
        self._wait_for_subscribers = bool(self.get_parameter("wait_for_subscribers").value)
        self._stop_on_plan = bool(self.get_parameter("stop_on_plan").value)
        self._global_plan_topic = self.get_parameter("global_plan_topic").value

        period = float(self.get_parameter("publish_period_sec").value)
        flat_goal_array = list(self.get_parameter("goal_array").value)

        self._pose_array = self._build_pose_array(flat_goal_array)
        self._goal_init_msg = Bool()
        self._goal_init_msg.data = True

        self._goal_pub = self.create_publisher(PoseArray, self._goal_topic, 1)
        self._goal_init_pub = self.create_publisher(Bool, self._goal_init_topic, 1)
        self._plan_sub = None
        self._plan_received = False

        if self._stop_on_plan:
            self._plan_sub = self.create_subscription(
                GlobalPlan,
                self._global_plan_topic,
                self._global_plan_callback,
                1,
            )

        self._publish_idx = 0
        self._waiting_logged = False
        self._timer = self.create_timer(period, self._timer_callback)

    def _build_pose_array(self, flat_goal_array: List[float]) -> PoseArray:
        expected_size = self._agent_num * 7
        if len(flat_goal_array) < expected_size:
            raise ValueError(
                f"goal_array expects at least {expected_size} values, "
                f"but got {len(flat_goal_array)}"
            )

        pose_array = PoseArray()
        pose_array.header.frame_id = self._frame_id
        pose_array.poses = []

        for i in range(self._agent_num):
            base = i * 7
            pose = Pose()
            pose.position.x = float(flat_goal_array[base])
            pose.position.y = float(flat_goal_array[base + 1])
            pose.position.z = float(flat_goal_array[base + 2])
            pose.orientation.x = float(flat_goal_array[base + 3])
            pose.orientation.y = float(flat_goal_array[base + 4])
            pose.orientation.z = float(flat_goal_array[base + 5])
            pose.orientation.w = float(flat_goal_array[base + 6])
            pose_array.poses.append(pose)

        return pose_array

    def _global_plan_callback(self, plan_msg: GlobalPlan) -> None:
        if plan_msg.global_plan:
            self._plan_received = True
            self.get_logger().info("Received /global_plan. Stopping goal initialization publisher.")
            self._timer.cancel()

    def _timer_callback(self) -> None:
        if self._wait_for_subscribers:
            goal_sub_count = self._goal_pub.get_subscription_count()
            init_sub_count = self._goal_init_pub.get_subscription_count()
            if goal_sub_count < 1 or init_sub_count < 1:
                if not self._waiting_logged:
                    self.get_logger().info(
                        "Waiting for subscribers on goal topics before publishing."
                    )
                    self._waiting_logged = True
                return
            self._waiting_logged = False

        if self._plan_received:
            return

        self._pose_array.header.stamp = self.get_clock().now().to_msg()
        self._goal_pub.publish(self._pose_array)
        self._goal_init_pub.publish(self._goal_init_msg)

        self._publish_idx += 1
        self.get_logger().info(
            f"Published MAPF goals and init flag ({self._publish_idx}/{self._publish_count})."
        )

        if self._publish_count > 0 and self._publish_idx >= self._publish_count:
            self._timer.cancel()
            self.get_logger().info("Finished MAPF goal initialization publishing.")


def main() -> None:
    example_goals = [
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ]

    rclpy.init()
    node = MapfGoalPublisher(example_goals)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
