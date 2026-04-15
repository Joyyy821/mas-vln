# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import math
from typing import List

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Quaternion, Twist
from mapf_msgs.msg import GlobalPlan
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


class MapfNav2Executor(Node):
    def __init__(self) -> None:
        super().__init__("plan_executor")

        self.declare_parameter("agent_num", 1)
        self.declare_parameter("global_frame_id", "map")
        self.declare_parameter("global_plan_topic", "global_plan")
        self.declare_parameter("control_frequency", 10.0)
        self.declare_parameter("goal_yaw_tolerance", 0.08)
        self.declare_parameter("pre_rotate_yaw_tolerance", 0.12)
        self.declare_parameter("angular_kp", 2.0)
        self.declare_parameter("max_angular_speed", 1.0)
        self.declare_parameter("controller_id", "")
        self.declare_parameter("goal_checker_id", "")
        self.declare_parameter("progress_checker_id", "")
        self.declare_parameter("execution_status_topic", "/mapf_base/plan_execution_status")
        self.declare_parameter("rollout_control_topic", "")
        self.declare_parameter("team_config_file", "")
        self.declare_parameter("experiments_dir", "")

        self._agent_num = int(self.get_parameter("agent_num").value)
        self._global_frame_id = str(self.get_parameter("global_frame_id").value)
        self._global_plan_topic = str(self.get_parameter("global_plan_topic").value)
        self._control_frequency = float(self.get_parameter("control_frequency").value)
        self._goal_yaw_tolerance = float(self.get_parameter("goal_yaw_tolerance").value)
        self._pre_rotate_yaw_tolerance = float(
            self.get_parameter("pre_rotate_yaw_tolerance").value
        )
        self._angular_kp = float(self.get_parameter("angular_kp").value)
        self._max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self._controller_id = str(self.get_parameter("controller_id").value)
        self._goal_checker_id = str(self.get_parameter("goal_checker_id").value)
        self._progress_checker_id = str(self.get_parameter("progress_checker_id").value)
        self._execution_status_topic = str(self.get_parameter("execution_status_topic").value)

        self._agent_names: List[str] = []
        self._base_frame_ids: List[str] = []
        self._cmd_vel_topics: List[str] = []
        self._cmd_vel_publishers = []
        self._action_clients: List[ActionClient] = []
        for idx in range(self._agent_num):
            agent_param = f"agent_name.agent_{idx}"
            base_frame_param = f"base_frame_id.agent_{idx}"
            cmd_vel_param = f"cmd_vel_topic.agent_{idx}"
            default_agent_name = f"robot{idx + 1}"

            self.declare_parameter(agent_param, default_agent_name)
            self.declare_parameter(base_frame_param, f"{default_agent_name}/base_link")
            self.declare_parameter(cmd_vel_param, f"/{default_agent_name}/cmd_vel")

            agent_name = str(self.get_parameter(agent_param).value)
            base_frame_id = str(self.get_parameter(base_frame_param).value)
            cmd_vel_topic = str(self.get_parameter(cmd_vel_param).value)

            self._agent_names.append(agent_name)
            self._base_frame_ids.append(base_frame_id)
            self._cmd_vel_topics.append(cmd_vel_topic)
            self._cmd_vel_publishers.append(self.create_publisher(Twist, cmd_vel_topic, 1))
            self._action_clients.append(
                ActionClient(self, FollowPath, f"/{agent_name}/follow_path")
            )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._current_plan: GlobalPlan | None = None
        self._pending_plan: GlobalPlan | None = None
        self._active = False
        self._execution_id = 0
        self._completed = [False] * self._agent_num
        self._phases = ["idle"] * self._agent_num
        self._controller_paths: List[Path | None] = [None] * self._agent_num
        self._pre_rotate_yaws: List[float | None] = [None] * self._agent_num
        self._final_goal_yaws: List[float | None] = [None] * self._agent_num
        self._last_tf_warn_ns = [0] * self._agent_num

        self._plan_sub = self.create_subscription(
            GlobalPlan,
            self._global_plan_topic,
            self._plan_callback,
            1,
        )
        self._execution_status_pub = self.create_publisher(
            String, self._execution_status_topic, 10
        )
        self._startup_timer = self.create_timer(0.5, self._start_pending_plan)
        self._execution_timer = self.create_timer(
            1.0 / self._control_frequency,
            self._execution_timer_callback,
        )
        self.get_logger().info(
            "Initialized MAPF Nav2 executor on "
            f"'{self.resolve_topic_name(self._global_plan_topic)}' "
            f"for {self._agent_num} agents. FollowPath targets: {self._agent_names}"
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
        if self._pending_plan is not None and self._plans_equal(self._pending_plan, plan_msg):
            return

        if self._active:
            self.get_logger().warn(
                "Received a new plan while execution is in progress. Ignoring it."
            )
            return

        self._pending_plan = plan_msg
        self._start_pending_plan()

    def _start_pending_plan(self) -> None:
        if self._pending_plan is None or self._active:
            return

        for idx, action_client in enumerate(self._action_clients):
            if not action_client.wait_for_server(timeout_sec=0.5):
                self.get_logger().warn(
                    f"FollowPath server for {self._agent_names[idx]} is not ready yet."
                )
                return

        self._current_plan = self._pending_plan
        self._pending_plan = None
        self._active = True
        self._execution_id += 1
        self._completed = [False] * self._agent_num
        self._phases = ["idle"] * self._agent_num
        self._controller_paths = [None] * self._agent_num
        self._pre_rotate_yaws = [None] * self._agent_num
        self._final_goal_yaws = [None] * self._agent_num

        self._publish_execution_status("active")
        self.get_logger().info(
            f"Starting FollowPath execution of MAPF plan with makespan "
            f"{self._current_plan.makespan}."
        )
        self._dispatch_current_plan()

    def _dispatch_current_plan(self) -> None:
        if self._current_plan is None:
            self._active = False
            return

        for agent_idx in range(self._agent_num):
            raw_path = self._current_plan.global_plan[agent_idx].plan
            controller_path = self._build_controller_path(raw_path)
            if not controller_path.poses:
                self.get_logger().warn(
                    f"Agent {agent_idx} has an empty MAPF path. Marking as complete."
                )
                self._completed[agent_idx] = True
                self._phases[agent_idx] = "done"
                continue

            first_pose = controller_path.poses[0].pose.position
            last_pose = controller_path.poses[-1].pose.position
            self.get_logger().info(
                f"Agent {agent_idx} prepared FollowPath with {len(controller_path.poses)} poses "
                f"from ({first_pose.x:.3f}, {first_pose.y:.3f}) "
                f"to ({last_pose.x:.3f}, {last_pose.y:.3f})"
            )

            self._controller_paths[agent_idx] = controller_path
            self._pre_rotate_yaws[agent_idx] = self._quaternion_to_yaw(
                controller_path.poses[0].pose.orientation
            )
            self._final_goal_yaws[agent_idx] = self._extract_final_goal_yaw(raw_path)
            self._phases[agent_idx] = "pre_rotate"

        self._check_if_finished()

    def _execution_timer_callback(self) -> None:
        if not self._active:
            return

        execution_id = self._execution_id
        for agent_idx in range(self._agent_num):
            phase = self._phases[agent_idx]

            if phase == "pre_rotate":
                if not self._rotate_agent_towards(
                    agent_idx,
                    self._pre_rotate_yaws[agent_idx],
                    self._pre_rotate_yaw_tolerance,
                ):
                    continue

                self._publish_zero(agent_idx)
                self._phases[agent_idx] = "follow_path_pending"
                self._send_follow_path(agent_idx, execution_id)
                continue

            if phase == "post_rotate":
                if not self._rotate_agent_towards(
                    agent_idx,
                    self._final_goal_yaws[agent_idx],
                    self._goal_yaw_tolerance,
                ):
                    continue

                self._publish_zero(agent_idx)
                self._completed[agent_idx] = True
                self._phases[agent_idx] = "done"
                self.get_logger().info(
                    f"Agent {agent_idx} finished its final in-place rotation."
                )
                self._check_if_finished()
                continue

            if phase == "done":
                self._publish_zero(agent_idx)

    def _send_follow_path(self, agent_idx: int, execution_id: int) -> None:
        controller_path = self._controller_paths[agent_idx]
        if controller_path is None or not controller_path.poses:
            self._completed[agent_idx] = True
            self._phases[agent_idx] = "done"
            self._check_if_finished()
            return

        goal_msg = FollowPath.Goal()
        goal_msg.path = controller_path
        goal_fields = goal_msg.get_fields_and_field_types()
        if "controller_id" in goal_fields and self._controller_id:
            goal_msg.controller_id = self._controller_id
        if "goal_checker_id" in goal_fields and self._goal_checker_id:
            goal_msg.goal_checker_id = self._goal_checker_id
        if "progress_checker_id" in goal_fields and self._progress_checker_id:
            goal_msg.progress_checker_id = self._progress_checker_id

        first_pose = controller_path.poses[0].pose.position
        last_pose = controller_path.poses[-1].pose.position
        self.get_logger().info(
            f"Agent {agent_idx} send FollowPath with {len(controller_path.poses)} poses "
            f"from ({first_pose.x:.3f}, {first_pose.y:.3f}) "
            f"to ({last_pose.x:.3f}, {last_pose.y:.3f})"
        )

        send_goal_future = self._action_clients[agent_idx].send_goal_async(goal_msg)
        send_goal_future.add_done_callback(
            lambda future, idx=agent_idx, exec_id=execution_id: self._goal_response_callback(
                future, idx, exec_id
            )
        )

    def _goal_response_callback(self, future, agent_idx: int, execution_id: int) -> None:
        if execution_id != self._execution_id or not self._active:
            return

        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Agent {agent_idx} FollowPath request failed: {exc}")
            self._abort_execution()
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"Agent {agent_idx} FollowPath goal was rejected.")
            self._abort_execution()
            return

        self._phases[agent_idx] = "follow_path"
        self.get_logger().info(f"Agent {agent_idx} FollowPath accepted, waiting for result")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result, idx=agent_idx, exec_id=execution_id: self._result_callback(
                result, idx, exec_id
            )
        )

    def _result_callback(self, future, agent_idx: int, execution_id: int) -> None:
        if execution_id != self._execution_id or not self._active:
            return

        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"Agent {agent_idx} FollowPath result failed: {exc}")
            self._abort_execution()
            return

        if result is None:
            self.get_logger().error(f"Agent {agent_idx} returned an empty FollowPath result.")
            self._abort_execution()
            return

        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                f"Agent {agent_idx} FollowPath failed with status {result.status}."
            )
            self._abort_execution()
            return

        self.get_logger().info(
            f"Agent {agent_idx} finished FollowPath translation, "
            "starting final in-place rotation."
        )
        self._phases[agent_idx] = "post_rotate"

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

    def _rotate_agent_towards(
        self,
        agent_idx: int,
        target_yaw: float | None,
        tolerance: float,
    ) -> bool:
        if target_yaw is None:
            return True

        pose = self._lookup_robot_pose(agent_idx)
        if pose is None:
            self._publish_zero(agent_idx)
            return False

        yaw_error = self._normalize_angle(target_yaw - pose[2])
        if abs(yaw_error) <= tolerance:
            return True

        twist = Twist()
        twist.angular.z = self._clamp(
            self._angular_kp * yaw_error,
            -self._max_angular_speed,
            self._max_angular_speed,
        )
        self._cmd_vel_publishers[agent_idx].publish(twist)
        return False

    def _check_if_finished(self) -> None:
        if self._active and all(self._completed):
            self.get_logger().info("Finished executing MAPF plan.")
            self._active = False
            self._current_plan = None
            self._publish_zero_to_all()
            self._publish_execution_status("succeeded")
            self._start_pending_plan()

    def _abort_execution(self) -> None:
        self._active = False
        self._current_plan = None
        self._phases = ["idle"] * self._agent_num
        self._controller_paths = [None] * self._agent_num
        self._pre_rotate_yaws = [None] * self._agent_num
        self._final_goal_yaws = [None] * self._agent_num
        self._publish_zero_to_all()
        self._publish_execution_status("failed")

    def _build_controller_path(self, raw_path: Path) -> Path:
        path = Path()
        path.header.frame_id = self._global_frame_id
        path.header.stamp = self.get_clock().now().to_msg()

        compact_poses = self._remove_duplicate_poses(raw_path)
        if not compact_poses:
            return path

        for idx, pose_stamped in enumerate(compact_poses):
            pose_copy = copy.deepcopy(pose_stamped)
            pose_copy.header.frame_id = self._global_frame_id
            pose_copy.header.stamp = path.header.stamp

            if len(compact_poses) == 1:
                pose_copy.pose.orientation = self._normalize_quaternion(
                    compact_poses[idx].pose.orientation
                )
            else:
                yaw = self._compute_heading(compact_poses, idx)
                pose_copy.pose.orientation = self._yaw_to_quaternion(yaw)
            path.poses.append(pose_copy)

        return path

    def _extract_final_goal_yaw(self, raw_path: Path) -> float | None:
        if not raw_path.poses:
            return None

        return self._quaternion_to_yaw(
            self._normalize_quaternion(raw_path.poses[-1].pose.orientation)
        )

    def _remove_duplicate_poses(self, raw_path: Path) -> List:
        compact_poses = []
        for pose_stamped in raw_path.poses:
            if not compact_poses:
                compact_poses.append(copy.deepcopy(pose_stamped))
                continue

            last_pose = compact_poses[-1].pose.position
            curr_pose = pose_stamped.pose.position
            if self._squared_distance(last_pose.x, last_pose.y, curr_pose.x, curr_pose.y) > 1e-8:
                compact_poses.append(copy.deepcopy(pose_stamped))

        return compact_poses

    def _compute_heading(self, poses: List, idx: int) -> float:
        if len(poses) == 1:
            return 0.0

        current = poses[idx].pose.position
        next_idx = idx + 1
        while next_idx < len(poses):
            nxt = poses[next_idx].pose.position
            if self._squared_distance(current.x, current.y, nxt.x, nxt.y) > 1e-8:
                return math.atan2(nxt.y - current.y, nxt.x - current.x)
            next_idx += 1

        prev_idx = idx - 1
        while prev_idx >= 0:
            prv = poses[prev_idx].pose.position
            if self._squared_distance(current.x, current.y, prv.x, prv.y) > 1e-8:
                return math.atan2(current.y - prv.y, current.x - prv.x)
            prev_idx -= 1

        return 0.0

    def _publish_zero(self, agent_idx: int) -> None:
        self._cmd_vel_publishers[agent_idx].publish(Twist())

    def _publish_zero_to_all(self) -> None:
        for agent_idx in range(self._agent_num):
            self._publish_zero(agent_idx)

    def _publish_execution_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self._execution_status_pub.publish(msg)

    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        quat = Quaternion()
        quat.z = math.sin(yaw / 2.0)
        quat.w = math.cos(yaw / 2.0)
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

    def _clamp(self, value: float, min_value: float, max_value: float) -> float:
        return max(min(value, max_value), min_value)

    def _squared_distance(self, x0: float, y0: float, x1: float, y1: float) -> float:
        return (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)

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
    node = MapfNav2Executor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
