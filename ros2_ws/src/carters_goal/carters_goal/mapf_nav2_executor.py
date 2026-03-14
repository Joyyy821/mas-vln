# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import math
from typing import List

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Quaternion
from mapf_msgs.msg import GlobalPlan
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node


class MapfNav2Executor(Node):
    def __init__(self) -> None:
        super().__init__("plan_executor")

        self.declare_parameter("agent_num", 1)
        self.declare_parameter("global_frame_id", "map")
        self.declare_parameter("global_plan_topic", "global_plan")
        self.declare_parameter("controller_id", "")
        self.declare_parameter("goal_checker_id", "")
        self.declare_parameter("progress_checker_id", "")

        self._agent_num = int(self.get_parameter("agent_num").value)
        self._global_frame_id = str(self.get_parameter("global_frame_id").value)
        self._global_plan_topic = str(self.get_parameter("global_plan_topic").value)
        self._controller_id = str(self.get_parameter("controller_id").value)
        self._goal_checker_id = str(self.get_parameter("goal_checker_id").value)
        self._progress_checker_id = str(self.get_parameter("progress_checker_id").value)

        self._agent_names: List[str] = []
        self._action_clients: List[ActionClient] = []
        for idx in range(self._agent_num):
            param_name = f"agent_name.agent_{idx}"
            default_agent_name = f"robot{idx + 1}"
            self.declare_parameter(param_name, default_agent_name)
            agent_name = str(self.get_parameter(param_name).value)
            self._agent_names.append(agent_name)
            self._action_clients.append(
                ActionClient(self, FollowPath, f"/{agent_name}/follow_path")
            )

        self._current_plan: GlobalPlan | None = None
        self._pending_plan: GlobalPlan | None = None
        self._active = False
        self._execution_id = 0
        self._completed = [False] * self._agent_num

        self._plan_sub = self.create_subscription(
            GlobalPlan,
            self._global_plan_topic,
            self._plan_callback,
            1,
        )
        self._startup_timer = self.create_timer(0.5, self._start_pending_plan)

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
        self.get_logger().info(
            f"Starting FollowPath execution of MAPF plan with makespan "
            f"{self._current_plan.makespan}."
        )
        self._dispatch_current_plan()

    def _dispatch_current_plan(self) -> None:
        if self._current_plan is None:
            self._active = False
            return

        execution_id = self._execution_id

        for agent_idx in range(self._agent_num):
            raw_path = self._current_plan.global_plan[agent_idx].plan
            controller_path = self._build_controller_path(raw_path)
            if not controller_path.poses:
                self.get_logger().warn(
                    f"Agent {agent_idx} has an empty MAPF path. Marking as complete."
                )
                self._completed[agent_idx] = True
                continue

            goal_msg = FollowPath.Goal()
            goal_msg.path = controller_path
            goal_fields = goal_msg.get_fields_and_field_types()
            if "controller_id" in goal_fields:
                goal_msg.controller_id = self._controller_id
            if "goal_checker_id" in goal_fields:
                goal_msg.goal_checker_id = self._goal_checker_id
            if "progress_checker_id" in goal_fields:
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

        self._check_if_finished()

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

        self.get_logger().info(f"Agent {agent_idx} finished its FollowPath.")
        self._completed[agent_idx] = True
        self._check_if_finished()

    def _check_if_finished(self) -> None:
        if self._active and all(self._completed):
            self.get_logger().info("Finished executing MAPF plan.")
            self._active = False
            self._current_plan = None
            self._start_pending_plan()

    def _abort_execution(self) -> None:
        self._active = False
        self._current_plan = None

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

            yaw = self._compute_heading(compact_poses, idx)
            pose_copy.pose.orientation = self._yaw_to_quaternion(yaw)
            path.poses.append(pose_copy)

        return path

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

    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        quat = Quaternion()
        quat.z = math.sin(yaw / 2.0)
        quat.w = math.cos(yaw / 2.0)
        return quat

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
