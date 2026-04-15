#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from lifecycle_msgs.msg import TransitionEvent
from mapf_msgs.msg import GlobalPlan
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, String

from carters_goal.rollout_control import build_rollout_control_message
from carters_goal.shared_team_config import rollout_run_dir, team_config_utils


def _now_sec(node: Node) -> float:
    return node.get_clock().now().nanoseconds / 1e9


class RolloutManager(Node):
    def __init__(self) -> None:
        super().__init__("rollout_manager")

        self.declare_parameter("team_config_file", "")
        self.declare_parameter("experiments_dir", "")
        self.declare_parameter("rollout_control_topic", "/carters_goal/rollout_control")
        self.declare_parameter("rollout_reset_done_topic", "/carters_goal/rollout_reset_done")
        self.declare_parameter("execution_status_topic", "/mapf_base/plan_execution_status")
        self.declare_parameter("goal_topic", "/mapf_base/goal_for_each")
        self.declare_parameter("goal_init_topic", "/mapf_base/goal_init_flag")
        self.declare_parameter("global_plan_topic", "/mapf_base/global_plan")
        self.declare_parameter("mapf_transition_topic", "/mapf_base/mapf_base_node/transition_event")
        self.declare_parameter("goal_frame_id", "map")
        self.declare_parameter("wait_for_mapf_active", True)
        self.declare_parameter("skip_existed_rollout", False)
        self.declare_parameter("rollout_control_publish_period_sec", 0.5)
        self.declare_parameter("goal_publish_period_sec", 0.5)
        self.declare_parameter("reset_timeout_sec", 15.0)
        self.declare_parameter("execution_timeout_sec", 300.0)
        self.declare_parameter("post_rollout_delay_sec", 1.0)

        team_config_file = str(self.get_parameter("team_config_file").value).strip()
        if not team_config_file:
            raise ValueError("team_config_file must be provided to RolloutManager.")

        self._team_config_path = team_config_file
        self._experiments_dir = str(self.get_parameter("experiments_dir").value).strip()
        self._goal_frame_id = str(self.get_parameter("goal_frame_id").value).strip() or "map"
        self._wait_for_mapf_active = bool(self.get_parameter("wait_for_mapf_active").value)
        self._skip_existed_rollout = bool(self.get_parameter("skip_existed_rollout").value)
        self._rollout_control_publish_period_sec = float(
            self.get_parameter("rollout_control_publish_period_sec").value
        )
        self._goal_publish_period_sec = float(
            self.get_parameter("goal_publish_period_sec").value
        )
        self._reset_timeout_sec = float(self.get_parameter("reset_timeout_sec").value)
        self._execution_timeout_sec = float(self.get_parameter("execution_timeout_sec").value)
        self._post_rollout_delay_sec = float(self.get_parameter("post_rollout_delay_sec").value)

        self._team_config = team_config_utils.load_multi_rollout_config(self._team_config_path)
        self._rollouts: list[dict[str, Any]] = list(self._team_config["rollouts"])
        if not self._rollouts:
            raise ValueError(f"No rollouts were configured in {self._team_config_path}.")

        self._language_instruction = str(self._team_config.get("language_instruction", "")).strip()
        self._goal_init_msg = Bool()
        self._goal_init_msg.data = True

        rollout_control_topic = str(self.get_parameter("rollout_control_topic").value)
        rollout_reset_done_topic = str(self.get_parameter("rollout_reset_done_topic").value)
        execution_status_topic = str(self.get_parameter("execution_status_topic").value)
        goal_topic = str(self.get_parameter("goal_topic").value)
        goal_init_topic = str(self.get_parameter("goal_init_topic").value)
        global_plan_topic = str(self.get_parameter("global_plan_topic").value)
        mapf_transition_topic = str(self.get_parameter("mapf_transition_topic").value)

        self._rollout_control_pub = self.create_publisher(PoseArray, rollout_control_topic, 10)
        self._goal_pub = self.create_publisher(PoseArray, goal_topic, 10)
        self._goal_init_pub = self.create_publisher(Bool, goal_init_topic, 10)

        self._reset_done_sub = self.create_subscription(
            Int32,
            rollout_reset_done_topic,
            self._reset_done_callback,
            10,
        )
        self._execution_status_sub = self.create_subscription(
            String,
            execution_status_topic,
            self._execution_status_callback,
            10,
        )
        self._global_plan_sub = self.create_subscription(
            GlobalPlan,
            global_plan_topic,
            self._global_plan_callback,
            10,
        )
        self._transition_sub = None
        if self._wait_for_mapf_active:
            self._transition_sub = self.create_subscription(
                TransitionEvent,
                mapf_transition_topic,
                self._transition_callback,
                10,
            )

        self._mapf_is_active = not self._wait_for_mapf_active
        self._rollout_cursor = 0
        self._current_rollout: dict[str, Any] | None = None
        self._current_control_msg: PoseArray | None = None
        self._current_goal_msg: PoseArray | None = None
        self._last_reset_done_rollout_id: int | None = None
        self._latest_execution_status = ""
        self._plan_received = False
        self._execution_started = False
        self._state = "wait_for_mapf_active" if self._wait_for_mapf_active else "start_next"
        self._state_started_at_sec = _now_sec(self)
        self._last_rollout_control_publish_sec = float("-inf")
        self._last_goal_publish_sec = float("-inf")
        self._idle_logged = False
        self._stop_signal_sent = False

        self._tick_timer = self.create_timer(0.1, self._tick)
        self.get_logger().info(
            "Loaded rollout manager with "
            f"{len(self._rollouts)} rollouts from {self._team_config_path}. "
            f"Language instruction: {self._language_instruction or '<empty>'}"
        )

    def _set_state(self, state: str) -> None:
        self._state = state
        self._state_started_at_sec = _now_sec(self)
        self._idle_logged = False

    def _transition_callback(self, event: TransitionEvent) -> None:
        goal_label = event.goal_state.label.lower()
        if goal_label == "active":
            self._mapf_is_active = True
            return

        if goal_label in {"inactive", "unconfigured", "finalized"}:
            self._mapf_is_active = False

    def _reset_done_callback(self, msg: Int32) -> None:
        self._last_reset_done_rollout_id = int(msg.data)

    def _execution_status_callback(self, msg: String) -> None:
        self._latest_execution_status = str(msg.data).strip().lower()
        if self._latest_execution_status == "active":
            self._execution_started = True

    def _global_plan_callback(self, msg: GlobalPlan) -> None:
        if self._current_rollout is None:
            return
        if len(msg.global_plan) < self._team_config["agent_num"]:
            return
        self._plan_received = True
        self._execution_started = True

    def _tick(self) -> None:
        if self._state == "wait_for_mapf_active":
            self._handle_wait_for_mapf_active()
            return

        if self._state == "start_next":
            self._start_next_rollout()
            return

        if self._state == "wait_for_reset":
            self._handle_wait_for_reset()
            return

        if self._state == "wait_for_execution":
            self._handle_wait_for_execution()
            return

        if self._state == "post_rollout":
            self._handle_post_rollout()
            return

        if self._state == "done":
            self._handle_done()
            return

        self.get_logger().error(f"Unknown rollout-manager state '{self._state}'.")
        self._set_state("done")

    def _handle_wait_for_mapf_active(self) -> None:
        if self._mapf_is_active:
            self.get_logger().info("MAPF lifecycle node is active. Starting rollout sequence.")
            self._set_state("start_next")
            return

        if not self._idle_logged:
            self.get_logger().info(
                "Waiting for /mapf_base/mapf_base_node to become active before starting rollouts."
            )
            self._idle_logged = True

    def _start_next_rollout(self) -> None:
        self._current_rollout = None
        self._current_control_msg = None
        self._current_goal_msg = None
        self._last_reset_done_rollout_id = None
        self._latest_execution_status = ""
        self._plan_received = False
        self._execution_started = False

        while self._rollout_cursor < len(self._rollouts):
            rollout = self._rollouts[self._rollout_cursor]
            rollout_id = int(rollout["id"])
            output_dir = rollout_run_dir(
                self._experiments_dir,
                team_config_path=self._team_config_path,
                rollout_id=rollout_id,
            )
            if self._skip_existed_rollout and output_dir.exists():
                self.get_logger().info(
                    f"Skipping rollout {rollout_id} because {output_dir} already exists."
                )
                self._rollout_cursor += 1
                continue

            self._current_rollout = rollout
            self._current_control_msg = build_rollout_control_message(
                rollout_id,
                rollout["initial_pose_array"],
            )
            self._current_goal_msg = self._build_goal_message(rollout["goal_pose_array"])
            self._publish_rollout_control(force=True)
            self.get_logger().info(
                f"Starting rollout {rollout_id}/{len(self._rollouts)}. "
                f"Output directory: {output_dir}"
            )
            self._set_state("wait_for_reset")
            return

        self.get_logger().info("Finished all configured rollouts.")
        self._set_state("done")

    def _handle_wait_for_reset(self) -> None:
        assert self._current_rollout is not None

        rollout_id = int(self._current_rollout["id"])
        if self._last_reset_done_rollout_id == rollout_id:
            self.get_logger().info(
                f"Simulator reset completed for rollout {rollout_id}. Publishing goals."
            )
            self._publish_goal(force=True)
            self._set_state("wait_for_execution")
            return

        self._publish_rollout_control()
        if _now_sec(self) - self._state_started_at_sec > self._reset_timeout_sec:
            raise RuntimeError(
                f"Timed out waiting for simulator reset acknowledgement for rollout {rollout_id}."
            )

    def _handle_wait_for_execution(self) -> None:
        assert self._current_rollout is not None
        rollout_id = int(self._current_rollout["id"])

        if self._latest_execution_status == "active":
            self._execution_started = True
        if self._plan_received:
            self._execution_started = True

        if self._execution_started and self._latest_execution_status == "failed":
            raise RuntimeError(f"Plan execution failed during rollout {rollout_id}.")

        if self._execution_started and self._latest_execution_status == "succeeded":
            self.get_logger().info(f"Rollout {rollout_id} completed successfully.")
            self._rollout_cursor += 1
            self._set_state("post_rollout")
            return

        if not self._plan_received and self._latest_execution_status != "active":
            self._publish_goal()

        if _now_sec(self) - self._state_started_at_sec > self._execution_timeout_sec:
            raise RuntimeError(f"Timed out waiting for rollout {rollout_id} to finish execution.")

    def _handle_post_rollout(self) -> None:
        if _now_sec(self) - self._state_started_at_sec < self._post_rollout_delay_sec:
            return
        self._set_state("start_next")

    def _handle_done(self) -> None:
        if self._stop_signal_sent:
            return

        stop_msg = build_rollout_control_message(0, [])
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        self._rollout_control_pub.publish(stop_msg)
        self._stop_signal_sent = True
        self.get_logger().info("Published rollout-control stop signal.")

    def _publish_rollout_control(self, *, force: bool = False) -> None:
        if self._current_control_msg is None:
            return

        now_sec = _now_sec(self)
        if (
            not force
            and now_sec - self._last_rollout_control_publish_sec
            < self._rollout_control_publish_period_sec
        ):
            return

        self._current_control_msg.header.stamp = self.get_clock().now().to_msg()
        self._rollout_control_pub.publish(self._current_control_msg)
        self._last_rollout_control_publish_sec = now_sec

    def _publish_goal(self, *, force: bool = False) -> None:
        if self._current_goal_msg is None:
            return

        now_sec = _now_sec(self)
        if not force and now_sec - self._last_goal_publish_sec < self._goal_publish_period_sec:
            return

        self._current_goal_msg.header.stamp = self.get_clock().now().to_msg()
        self._goal_pub.publish(self._current_goal_msg)
        self._goal_init_pub.publish(self._goal_init_msg)
        self._last_goal_publish_sec = now_sec

    def _build_goal_message(self, flat_goal_array: list[float]) -> PoseArray:
        msg = PoseArray()
        msg.header.frame_id = self._goal_frame_id
        msg.poses = []
        for offset in range(0, len(flat_goal_array), 7):
            pose = Pose()
            pose.position.x = float(flat_goal_array[offset + 0])
            pose.position.y = float(flat_goal_array[offset + 1])
            pose.position.z = float(flat_goal_array[offset + 2])
            pose.orientation.x = float(flat_goal_array[offset + 3])
            pose.orientation.y = float(flat_goal_array[offset + 4])
            pose.orientation.z = float(flat_goal_array[offset + 5])
            pose.orientation.w = float(flat_goal_array[offset + 6])
            msg.poses.append(pose)
        return msg


def main() -> None:
    rclpy.init()
    node: RolloutManager | None = None
    try:
        node = RolloutManager()
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
