# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseArray, Pose
from lifecycle_msgs.msg import TransitionEvent
from std_msgs.msg import Bool
from mapf_msgs.msg import GlobalPlan


class MapfGoalPublisher(Node):
    def __init__(self) -> None:
        super().__init__("mapf_goal_publisher")

        self.declare_parameter("goal_topic", "/mapf_base/goal_for_each")
        self.declare_parameter("goal_init_topic", "/mapf_base/goal_init_flag")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_period_sec", 0.5)
        self.declare_parameter("publish_count", 5)
        self.declare_parameter("agent_num", 0)
        self.declare_parameter("wait_for_subscribers", True)
        self.declare_parameter("wait_for_mapf_active", True)
        self.declare_parameter("stop_on_plan", True)
        self.declare_parameter("global_plan_topic", "/mapf_base/global_plan")
        self.declare_parameter(
            "mapf_transition_topic", "/mapf_base/mapf_base_node/transition_event"
        )
        self.declare_parameter("goal_array", Parameter.Type.DOUBLE_ARRAY)

        self._goal_topic = self.get_parameter("goal_topic").value
        self._goal_init_topic = self.get_parameter("goal_init_topic").value
        self._frame_id = self.get_parameter("frame_id").value
        self._publish_count = int(self.get_parameter("publish_count").value)
        self._agent_num = int(self.get_parameter("agent_num").value)
        self._wait_for_subscribers = bool(self.get_parameter("wait_for_subscribers").value)
        self._wait_for_mapf_active = bool(self.get_parameter("wait_for_mapf_active").value)
        self._stop_on_plan = bool(self.get_parameter("stop_on_plan").value)
        self._global_plan_topic = self.get_parameter("global_plan_topic").value
        self._mapf_transition_topic = self.get_parameter("mapf_transition_topic").value

        period = float(self.get_parameter("publish_period_sec").value)
        flat_goal_array = [float(value) for value in self.get_parameter("goal_array").value]

        if self._agent_num <= 0:
            if not flat_goal_array or len(flat_goal_array) % 7 != 0:
                raise ValueError(
                    "agent_num must be set or goal_array must contain 7 values per robot."
                )
            self._agent_num = len(flat_goal_array) // 7

        self._pose_array = self._build_pose_array(flat_goal_array)
        self._goal_init_msg = Bool()
        self._goal_init_msg.data = True

        self._goal_pub = self.create_publisher(PoseArray, self._goal_topic, 1)
        self._goal_init_pub = self.create_publisher(Bool, self._goal_init_topic, 1)
        self._plan_sub = None
        self._transition_sub = None
        self._plan_received = False
        self._mapf_is_active = not self._wait_for_mapf_active

        if self._stop_on_plan:
            self._plan_sub = self.create_subscription(
                GlobalPlan,
                self._global_plan_topic,
                self._global_plan_callback,
                1,
            )
        if self._wait_for_mapf_active:
            self._transition_sub = self.create_subscription(
                TransitionEvent,
                self._mapf_transition_topic,
                self._transition_callback,
                1,
            )

        self._publish_idx = 0
        self._waiting_logged = False
        self._waiting_for_active_logged = False
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
            quat = self._normalize_quaternion(
                float(flat_goal_array[base + 3]),
                float(flat_goal_array[base + 4]),
                float(flat_goal_array[base + 5]),
                float(flat_goal_array[base + 6]),
                i,
            )
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            pose_array.poses.append(pose)

        return pose_array

    def _normalize_quaternion(
        self, x: float, y: float, z: float, w: float, agent_idx: int
    ) -> List[float]:
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-8:
            self.get_logger().warn(
                f"Agent {agent_idx} goal quaternion has near-zero norm. "
                "Using identity orientation."
            )
            return [0.0, 0.0, 0.0, 1.0]

        if abs(norm - 1.0) > 1e-3:
            self.get_logger().warn(
                f"Agent {agent_idx} goal quaternion is not normalized "
                f"(norm={norm:.6f}). Normalizing it."
            )

        return [x / norm, y / norm, z / norm, w / norm]

    def _global_plan_callback(self, plan_msg: GlobalPlan) -> None:
        if plan_msg.global_plan:
            self._plan_received = True
            self.get_logger().info("Received /global_plan. Stopping goal initialization publisher.")
            self._timer.cancel()

    def _transition_callback(self, event: TransitionEvent) -> None:
        goal_label = event.goal_state.label.lower()
        if goal_label == "active":
            self._mapf_is_active = True
            return

        if goal_label in {"inactive", "unconfigured", "finalized"}:
            self._mapf_is_active = False

    def _timer_callback(self) -> None:
        if self._wait_for_mapf_active and not self._mapf_is_active:
            if not self._waiting_for_active_logged:
                self.get_logger().info(
                    "Waiting for /mapf_base/mapf_base_node to become active before publishing goals."
                )
                self._waiting_for_active_logged = True
            return
        self._waiting_for_active_logged = False

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
    rclpy.init()
    node = MapfGoalPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
