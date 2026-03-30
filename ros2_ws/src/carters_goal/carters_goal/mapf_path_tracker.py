# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import math
from typing import List

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from mapf_msgs.msg import GlobalPlan
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class MapfPathTracker(Node):
    def __init__(self) -> None:
        super().__init__("path_tracker")

        self.declare_parameter("agent_num", 1)
        self.declare_parameter("global_frame_id", "map")
        self.declare_parameter("global_plan_topic", "global_plan")
        self.declare_parameter("control_frequency", 10.0)
        self.declare_parameter("interpolate_spacing", 0.1)
        self.declare_parameter("lookahead_distance", 0.35)
        self.declare_parameter("waypoint_tolerance", 0.15)
        self.declare_parameter("goal_position_tolerance", 0.1)
        self.declare_parameter("goal_yaw_tolerance", 0.1)
        self.declare_parameter("rotate_in_place_yaw_threshold", 0.4)
        self.declare_parameter("slow_down_radius", 0.5)
        self.declare_parameter("linear_kp", 1.0)
        self.declare_parameter("angular_kp", 2.0)
        self.declare_parameter("max_linear_speed", 0.5)
        self.declare_parameter("max_angular_speed", 1.0)

        self._agent_num = int(self.get_parameter("agent_num").value)
        self._global_frame_id = str(self.get_parameter("global_frame_id").value)
        self._global_plan_topic = str(self.get_parameter("global_plan_topic").value)
        self._control_frequency = float(self.get_parameter("control_frequency").value)
        self._interpolate_spacing = float(self.get_parameter("interpolate_spacing").value)
        self._lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self._waypoint_tolerance = float(self.get_parameter("waypoint_tolerance").value)
        self._goal_position_tolerance = float(
            self.get_parameter("goal_position_tolerance").value
        )
        self._goal_yaw_tolerance = float(self.get_parameter("goal_yaw_tolerance").value)
        self._rotate_in_place_yaw_threshold = float(
            self.get_parameter("rotate_in_place_yaw_threshold").value
        )
        self._slow_down_radius = float(self.get_parameter("slow_down_radius").value)
        self._linear_kp = float(self.get_parameter("linear_kp").value)
        self._angular_kp = float(self.get_parameter("angular_kp").value)
        self._max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self._max_angular_speed = float(self.get_parameter("max_angular_speed").value)

        self._base_frame_ids: List[str] = []
        self._cmd_vel_topics: List[str] = []
        self._cmd_vel_publishers = []
        for idx in range(self._agent_num):
            base_frame_param = f"base_frame_id.agent_{idx}"
            cmd_vel_param = f"cmd_vel_topic.agent_{idx}"
            self.declare_parameter(base_frame_param, f"robot{idx + 1}/base_link")
            self.declare_parameter(cmd_vel_param, f"/robot{idx + 1}/cmd_vel")

            base_frame_id = str(self.get_parameter(base_frame_param).value)
            cmd_vel_topic = str(self.get_parameter(cmd_vel_param).value)
            self._base_frame_ids.append(base_frame_id)
            self._cmd_vel_topics.append(cmd_vel_topic)
            self._cmd_vel_publishers.append(self.create_publisher(Twist, cmd_vel_topic, 1))

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._current_plan: GlobalPlan | None = None
        self._paths: List[List[PoseStamped]] = [[] for _ in range(self._agent_num)]
        self._path_indices: List[int] = [0 for _ in range(self._agent_num)]
        self._goal_reached: List[bool] = [False for _ in range(self._agent_num)]
        self._has_active_plan = False
        self._last_tf_warn_ns = [0 for _ in range(self._agent_num)]

        self._plan_sub = self.create_subscription(
            GlobalPlan,
            self._global_plan_topic,
            self._plan_callback,
            1,
        )
        self._control_timer = self.create_timer(1.0 / self._control_frequency, self._control_loop)
        self.get_logger().info(
            "Initialized custom MAPF path tracker on "
            f"'{self.resolve_topic_name(self._global_plan_topic)}' "
            f"for {self._agent_num} agents. cmd_vel topics: {self._cmd_vel_topics}"
        )

    def _plan_callback(self, plan_msg: GlobalPlan) -> None:
        if len(plan_msg.global_plan) < self._agent_num:
            self.get_logger().warn(
                f"Received plan for {len(plan_msg.global_plan)} agents, "
                f"but configured for {self._agent_num}."
            )
            return

        if self._current_plan is not None and self._plans_equal(self._current_plan, plan_msg):
            return

        self._current_plan = copy.deepcopy(plan_msg)
        self._paths = [
            self._densify_path(plan_msg.global_plan[idx].plan) for idx in range(self._agent_num)
        ]
        self._path_indices = [0 for _ in range(self._agent_num)]
        self._goal_reached = [len(path) == 0 for path in self._paths]
        self._has_active_plan = any(not reached for reached in self._goal_reached)

        path_sizes = ", ".join(str(len(path)) for path in self._paths)
        self.get_logger().info(
            f"Loaded MAPF tracking plan with densified path sizes [{path_sizes}]."
        )

    def _control_loop(self) -> None:
        if not self._has_active_plan:
            self._publish_zero_to_all()
            return

        for agent_idx in range(self._agent_num):
            if self._goal_reached[agent_idx]:
                self._publish_zero(agent_idx)
                continue

            pose = self._lookup_robot_pose(agent_idx)
            if pose is None:
                self._publish_zero(agent_idx)
                continue

            twist = self._compute_twist(agent_idx, pose[0], pose[1], pose[2])
            self._cmd_vel_publishers[agent_idx].publish(twist)

        if all(self._goal_reached):
            self.get_logger().info("Finished executing MAPF plan with custom path tracker.")
            self._has_active_plan = False
            self._publish_zero_to_all()

    def _lookup_robot_pose(self, agent_idx: int) -> tuple[float, float, float] | None:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._global_frame_id,
                self._base_frame_ids[agent_idx],
                Time(),
            )
        except TransformException as exc:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_tf_warn_ns[agent_idx] > int(2e9):
                self.get_logger().warn(
                    f"Unable to lookup {self._global_frame_id} -> "
                    f"{self._base_frame_ids[agent_idx]}: {exc}"
                )
                self._last_tf_warn_ns[agent_idx] = now_ns
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = self._quaternion_to_yaw(rotation)
        return translation.x, translation.y, yaw

    def _compute_twist(self, agent_idx: int, x: float, y: float, yaw: float) -> Twist:
        path = self._paths[agent_idx]
        idx = self._path_indices[agent_idx]
        final_pose = path[-1]
        final_x = final_pose.pose.position.x
        final_y = final_pose.pose.position.y
        final_yaw = self._quaternion_to_yaw(final_pose.pose.orientation)

        while idx < len(path) - 1:
            waypoint = path[idx].pose.position
            if self._distance(x, y, waypoint.x, waypoint.y) > self._waypoint_tolerance:
                break
            idx += 1
        self._path_indices[agent_idx] = idx

        dist_to_goal = self._distance(x, y, final_x, final_y)
        if idx >= len(path) - 1 and dist_to_goal < self._goal_position_tolerance:
            yaw_error = self._normalize_angle(final_yaw - yaw)
            if abs(yaw_error) < self._goal_yaw_tolerance:
                self._goal_reached[agent_idx] = True
                return Twist()

            twist = Twist()
            twist.angular.z = self._clamp(
                self._angular_kp * yaw_error,
                -self._max_angular_speed,
                self._max_angular_speed,
            )
            return twist

        target_idx = self._select_lookahead_index(path, idx, x, y)
        target = path[target_idx].pose.position
        heading_to_target = math.atan2(target.y - y, target.x - x)
        heading_error = self._normalize_angle(heading_to_target - yaw)

        twist = Twist()
        if abs(heading_error) > self._rotate_in_place_yaw_threshold:
            twist.angular.z = self._clamp(
                self._angular_kp * heading_error,
                -self._max_angular_speed,
                self._max_angular_speed,
            )
            return twist

        distance_to_target = self._distance(x, y, target.x, target.y)
        linear_speed = self._linear_kp * distance_to_target * max(0.0, math.cos(heading_error))
        if dist_to_goal < self._slow_down_radius:
            linear_speed *= max(dist_to_goal / max(self._slow_down_radius, 1e-6), 0.2)

        twist.linear.x = self._clamp(linear_speed, 0.0, self._max_linear_speed)
        twist.angular.z = self._clamp(
            self._angular_kp * heading_error,
            -self._max_angular_speed,
            self._max_angular_speed,
        )
        return twist

    def _select_lookahead_index(
        self, path: List[PoseStamped], start_idx: int, x: float, y: float
    ) -> int:
        for idx in range(start_idx, len(path)):
            pose = path[idx].pose.position
            if self._distance(x, y, pose.x, pose.y) >= self._lookahead_distance:
                return idx
        return len(path) - 1

    def _densify_path(self, raw_path: Path) -> List[PoseStamped]:
        compact_path = self._remove_duplicate_poses(raw_path)
        if not compact_path:
            return []

        dense_path: List[PoseStamped] = [copy.deepcopy(compact_path[0])]
        for idx in range(len(compact_path) - 1):
            start_pose = compact_path[idx]
            end_pose = compact_path[idx + 1]
            segment_length = self._distance(
                start_pose.pose.position.x,
                start_pose.pose.position.y,
                end_pose.pose.position.x,
                end_pose.pose.position.y,
            )
            if segment_length < 1e-8:
                continue

            num_steps = max(1, int(math.ceil(segment_length / self._interpolate_spacing)))
            segment_yaw = math.atan2(
                end_pose.pose.position.y - start_pose.pose.position.y,
                end_pose.pose.position.x - start_pose.pose.position.x,
            )
            for step in range(1, num_steps + 1):
                ratio = step / num_steps
                pose = PoseStamped()
                pose.header.frame_id = self._global_frame_id
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = (
                    start_pose.pose.position.x
                    + ratio * (end_pose.pose.position.x - start_pose.pose.position.x)
                )
                pose.pose.position.y = (
                    start_pose.pose.position.y
                    + ratio * (end_pose.pose.position.y - start_pose.pose.position.y)
                )
                pose.pose.position.z = (
                    start_pose.pose.position.z
                    + ratio * (end_pose.pose.position.z - start_pose.pose.position.z)
                )
                if idx == len(compact_path) - 2 and step == num_steps:
                    pose.pose.orientation = self._normalize_quaternion(end_pose.pose.orientation)
                else:
                    pose.pose.orientation = self._yaw_to_quaternion(segment_yaw)
                dense_path.append(pose)

        return dense_path

    def _remove_duplicate_poses(self, raw_path: Path) -> List[PoseStamped]:
        compact_poses: List[PoseStamped] = []
        for pose_stamped in raw_path.poses:
            if not compact_poses:
                compact_poses.append(copy.deepcopy(pose_stamped))
                continue

            last_pose = compact_poses[-1].pose.position
            curr_pose = pose_stamped.pose.position
            if self._distance(last_pose.x, last_pose.y, curr_pose.x, curr_pose.y) > 1e-8:
                compact_poses.append(copy.deepcopy(pose_stamped))

        return compact_poses

    def _publish_zero(self, agent_idx: int) -> None:
        self._cmd_vel_publishers[agent_idx].publish(Twist())

    def _publish_zero_to_all(self) -> None:
        for agent_idx in range(self._agent_num):
            self._publish_zero(agent_idx)

    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        quat = Quaternion()
        quat.z = math.sin(yaw * 0.5)
        quat.w = math.cos(yaw * 0.5)
        return quat

    def _normalize_quaternion(self, quat_in: Quaternion) -> Quaternion:
        norm = math.sqrt(
            quat_in.x * quat_in.x
            + quat_in.y * quat_in.y
            + quat_in.z * quat_in.z
            + quat_in.w * quat_in.w
        )
        quat_out = Quaternion()
        if norm < 1e-8:
            quat_out.w = 1.0
            return quat_out

        quat_out.x = quat_in.x / norm
        quat_out.y = quat_in.y / norm
        quat_out.z = quat_in.z / norm
        quat_out.w = quat_in.w / norm
        return quat_out

    def _quaternion_to_yaw(self, quat: Quaternion) -> float:
        return math.atan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z),
        )

    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _distance(self, x0: float, y0: float, x1: float, y1: float) -> float:
        return math.hypot(x1 - x0, y1 - y0)

    def _clamp(self, value: float, min_value: float, max_value: float) -> float:
        return max(min(value, max_value), min_value)

    def _plans_equal(self, lhs: GlobalPlan, rhs: GlobalPlan) -> bool:
        if lhs.makespan != rhs.makespan or len(lhs.global_plan) != len(rhs.global_plan):
            return False

        for lhs_plan, rhs_plan in zip(lhs.global_plan, rhs.global_plan):
            if lhs_plan.time_step != rhs_plan.time_step:
                return False
            if len(lhs_plan.plan.poses) != len(rhs_plan.plan.poses):
                return False
            for lhs_pose, rhs_pose in zip(lhs_plan.plan.poses, rhs_plan.plan.poses):
                if lhs_pose.pose.position.x != rhs_pose.pose.position.x:
                    return False
                if lhs_pose.pose.position.y != rhs_pose.pose.position.y:
                    return False
        return True

    def destroy_node(self) -> bool:
        self._publish_zero_to_all()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = MapfPathTracker()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
