# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion, Twist
from mapf_msgs.msg import GlobalPlan, SinglePlan
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from carters_goal.rollout_control import parse_rollout_id
from carters_goal.shared_team_config import rollout_run_dir


@dataclass
class TrajectorySample:
    time_from_start: float
    x: float
    y: float
    yaw: float


@dataclass
class ReferenceState:
    x: float
    y: float
    yaw: float
    linear_velocity: float
    angular_velocity: float


@dataclass
class AgentTrajectory:
    samples: List[TrajectorySample]
    pre_rotate_yaw: float | None
    final_goal_yaw: float | None

    @property
    def translation_duration(self) -> float:
        if not self.samples:
            return 0.0
        return self.samples[-1].time_from_start

    @property
    def final_sample(self) -> TrajectorySample | None:
        if not self.samples:
            return None
        return self.samples[-1]


@dataclass
class TrackingErrorStats:
    sample_count: int = 0
    sum_sq_position_error: float = 0.0
    sum_abs_position_error: float = 0.0
    max_position_error: float = 0.0
    sum_sq_yaw_error: float = 0.0
    sum_abs_yaw_error: float = 0.0
    max_yaw_error: float = 0.0
    linear_saturation_count: int = 0
    angular_saturation_count: int = 0
    max_ref_linear_velocity: float = 0.0
    max_cmd_linear_velocity: float = 0.0
    max_cmd_angular_velocity: float = 0.0

    def update(
        self,
        position_error: float,
        yaw_error: float,
        ref_linear_velocity: float,
        cmd_linear_velocity: float,
        cmd_angular_velocity: float,
        linear_saturated: bool,
        angular_saturated: bool,
    ) -> None:
        abs_yaw_error = abs(yaw_error)
        self.sample_count += 1
        self.sum_sq_position_error += position_error * position_error
        self.sum_abs_position_error += position_error
        self.max_position_error = max(self.max_position_error, position_error)
        self.sum_sq_yaw_error += yaw_error * yaw_error
        self.sum_abs_yaw_error += abs_yaw_error
        self.max_yaw_error = max(self.max_yaw_error, abs_yaw_error)
        self.max_ref_linear_velocity = max(
            self.max_ref_linear_velocity, abs(ref_linear_velocity)
        )
        self.max_cmd_linear_velocity = max(
            self.max_cmd_linear_velocity, abs(cmd_linear_velocity)
        )
        self.max_cmd_angular_velocity = max(
            self.max_cmd_angular_velocity, abs(cmd_angular_velocity)
        )
        if linear_saturated:
            self.linear_saturation_count += 1
        if angular_saturated:
            self.angular_saturation_count += 1


class MapfTimedTracker(Node):
    def __init__(self) -> None:
        super().__init__("plan_executor")

        self.declare_parameter("agent_num", 1)
        self.declare_parameter("global_frame_id", "map")
        self.declare_parameter("global_plan_topic", "global_plan")
        self.declare_parameter("control_frequency", 10.0)
        self.declare_parameter("mapf_step_duration", 0.5)
        self.declare_parameter("goal_position_tolerance", 0.08)
        self.declare_parameter("goal_yaw_tolerance", 0.08)
        self.declare_parameter("pre_rotate_yaw_tolerance", 0.12)
        self.declare_parameter("slow_down_radius", 0.4)
        self.declare_parameter("tracking_longitudinal_kp", 1.8)
        self.declare_parameter("tracking_lateral_kp", 4.0)
        self.declare_parameter("tracking_yaw_kp", 2.5)
        self.declare_parameter("tracking_longitudinal_ki", 0.0)
        self.declare_parameter("tracking_lateral_ki", 0.05)
        self.declare_parameter("tracking_yaw_ki", 0.05)
        self.declare_parameter("integral_limit", 0.4)
        self.declare_parameter("max_linear_speed", 0.5)
        self.declare_parameter("max_angular_speed", 1.0)
        self.declare_parameter("save_tracking_log", True)
        self.declare_parameter("tracking_log_dir", "/tmp/mapf_timed_tracker")
        self.declare_parameter("execution_status_topic", "/mapf_base/plan_execution_status")
        self.declare_parameter("rollout_control_topic", "")
        self.declare_parameter("team_config_file", "")
        self.declare_parameter("experiments_dir", "")

        self._agent_num = int(self.get_parameter("agent_num").value)
        self._global_frame_id = str(self.get_parameter("global_frame_id").value)
        self._global_plan_topic = str(self.get_parameter("global_plan_topic").value)
        self._control_frequency = float(self.get_parameter("control_frequency").value)
        self._mapf_step_duration = float(self.get_parameter("mapf_step_duration").value)
        self._goal_position_tolerance = float(
            self.get_parameter("goal_position_tolerance").value
        )
        self._goal_yaw_tolerance = float(self.get_parameter("goal_yaw_tolerance").value)
        self._pre_rotate_yaw_tolerance = float(
            self.get_parameter("pre_rotate_yaw_tolerance").value
        )
        self._slow_down_radius = float(self.get_parameter("slow_down_radius").value)
        self._tracking_longitudinal_kp = float(
            self.get_parameter("tracking_longitudinal_kp").value
        )
        self._tracking_lateral_kp = float(self.get_parameter("tracking_lateral_kp").value)
        self._tracking_yaw_kp = float(self.get_parameter("tracking_yaw_kp").value)
        self._tracking_longitudinal_ki = float(
            self.get_parameter("tracking_longitudinal_ki").value
        )
        self._tracking_lateral_ki = float(self.get_parameter("tracking_lateral_ki").value)
        self._tracking_yaw_ki = float(self.get_parameter("tracking_yaw_ki").value)
        self._integral_limit = float(self.get_parameter("integral_limit").value)
        self._max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self._max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self._save_tracking_log = bool(self.get_parameter("save_tracking_log").value)
        self._tracking_log_dir = str(self.get_parameter("tracking_log_dir").value)
        self._default_tracking_log_dir = self._tracking_log_dir
        self._execution_status_topic = str(self.get_parameter("execution_status_topic").value)
        self._rollout_control_topic = str(self.get_parameter("rollout_control_topic").value).strip()
        team_config_file = str(self.get_parameter("team_config_file").value).strip()
        self._team_config_path = Path(team_config_file).expanduser().resolve() if team_config_file else None
        self._experiments_dir = str(self.get_parameter("experiments_dir").value).strip()
        self._active_rollout_id: int | None = None

        self._agent_names: List[str] = []
        self._base_frame_ids: List[str] = []
        self._cmd_vel_topics: List[str] = []
        self._cmd_vel_publishers = []
        for idx in range(self._agent_num):
            agent_param = f"agent_name.agent_{idx}"
            base_frame_param = f"base_frame_id.agent_{idx}"
            cmd_vel_param = f"cmd_vel_topic.agent_{idx}"
            self.declare_parameter(agent_param, f"robot{idx + 1}")
            self.declare_parameter(base_frame_param, f"robot{idx + 1}/base_link")
            self.declare_parameter(cmd_vel_param, f"/robot{idx + 1}/cmd_vel")

            agent_name = str(self.get_parameter(agent_param).value)
            base_frame_id = str(self.get_parameter(base_frame_param).value)
            cmd_vel_topic = str(self.get_parameter(cmd_vel_param).value)
            self._agent_names.append(agent_name)
            self._base_frame_ids.append(base_frame_id)
            self._cmd_vel_topics.append(cmd_vel_topic)
            self._cmd_vel_publishers.append(self.create_publisher(Twist, cmd_vel_topic, 1))

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._current_plan: GlobalPlan | None = None
        self._pending_plan: GlobalPlan | None = None
        self._trajectories: List[AgentTrajectory | None] = [None for _ in range(self._agent_num)]
        self._phases = ["idle" for _ in range(self._agent_num)]
        self._pre_rotate_complete = [False for _ in range(self._agent_num)]
        self._completed = [False for _ in range(self._agent_num)]
        self._active = False
        self._translation_started = False
        self._trajectory_start_time: Time | None = None
        self._max_translation_duration = 0.0
        self._last_tf_warn_ns = [0 for _ in range(self._agent_num)]
        self._tracking_error_stats = [TrackingErrorStats() for _ in range(self._agent_num)]
        self._tracking_summary_logged = [False for _ in range(self._agent_num)]
        self._tracking_log_rows = [[] for _ in range(self._agent_num)]
        self._integral_x = [0.0 for _ in range(self._agent_num)]
        self._integral_y = [0.0 for _ in range(self._agent_num)]
        self._integral_yaw = [0.0 for _ in range(self._agent_num)]
        self._last_control_time: Time | None = None
        self._execution_index = 0

        self._plan_sub = self.create_subscription(
            GlobalPlan,
            self._global_plan_topic,
            self._plan_callback,
            1,
        )
        self._execution_status_pub = self.create_publisher(
            String, self._execution_status_topic, 10
        )
        self._rollout_control_sub = None
        if self._rollout_control_topic:
            self._rollout_control_sub = self.create_subscription(
                PoseArray,
                self._rollout_control_topic,
                self._rollout_control_callback,
                10,
            )
        self._startup_timer = self.create_timer(0.5, self._start_pending_plan)
        self._control_timer = self.create_timer(1.0 / self._control_frequency, self._control_loop)
        self.get_logger().info(
            "Initialized MAPF timed tracker on "
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
        if self._pending_plan is not None and self._plans_equal(self._pending_plan, plan_msg):
            return

        if self._active:
            self.get_logger().warn(
                "Received a new plan while timed execution is in progress. Ignoring it."
            )
            return

        self._pending_plan = copy.deepcopy(plan_msg)
        self._start_pending_plan()

    def _start_pending_plan(self) -> None:
        if self._pending_plan is None or self._active:
            return

        self._current_plan = self._pending_plan
        self._pending_plan = None
        self._trajectories = []
        self._phases = ["idle" for _ in range(self._agent_num)]
        self._pre_rotate_complete = [False for _ in range(self._agent_num)]
        self._completed = [False for _ in range(self._agent_num)]
        self._active = True
        self._translation_started = False
        self._trajectory_start_time = None
        self._last_control_time = None
        self._execution_index += 1
        self._tracking_error_stats = [TrackingErrorStats() for _ in range(self._agent_num)]
        self._tracking_summary_logged = [False for _ in range(self._agent_num)]
        self._tracking_log_rows = [[] for _ in range(self._agent_num)]
        self._reset_all_integrators()
        self._publish_execution_status("active")

        durations = []
        for agent_idx in range(self._agent_num):
            trajectory = self._build_trajectory(self._current_plan.global_plan[agent_idx])
            self._trajectories.append(trajectory)
            if trajectory is None:
                self._completed[agent_idx] = True
                self._pre_rotate_complete[agent_idx] = True
                self._phases[agent_idx] = "done"
                durations.append(0.0)
                continue

            durations.append(trajectory.translation_duration)
            self._pre_rotate_complete[agent_idx] = trajectory.pre_rotate_yaw is None
            self._phases[agent_idx] = "track" if trajectory.pre_rotate_yaw is None else "pre_rotate"

        self._max_translation_duration = max(durations, default=0.0)
        duration_text = ", ".join(f"{duration:.2f}s" for duration in durations)
        self.get_logger().info(
            f"Loaded timed MAPF plan with per-agent translation durations [{duration_text}]. "
            "Waiting for all agents to finish pre-rotation before starting the shared clock."
        )
        self._log_reference_separation()

        if all(self._pre_rotate_complete):
            self._start_translation_phase(self.get_clock().now())

    def _control_loop(self) -> None:
        if not self._active:
            self._publish_zero_to_all()
            return

        now = self.get_clock().now()
        dt = 1.0 / self._control_frequency
        if self._last_control_time is not None:
            dt = max((now - self._last_control_time).nanoseconds / 1e9, 1e-3)
        self._last_control_time = now

        if not self._translation_started:
            for agent_idx, trajectory in enumerate(self._trajectories):
                if trajectory is None or self._completed[agent_idx]:
                    self._publish_zero(agent_idx)
                    continue

                if self._pre_rotate_complete[agent_idx]:
                    self._publish_zero(agent_idx)
                    continue

                if self._rotate_agent_towards(
                    agent_idx,
                    trajectory.pre_rotate_yaw,
                    self._pre_rotate_yaw_tolerance,
                ):
                    self._pre_rotate_complete[agent_idx] = True
                    self._publish_zero(agent_idx)
                    self.get_logger().info(
                        f"Agent {agent_idx} finished pre-rotation and is waiting at the shared start barrier."
                    )

            if all(self._pre_rotate_complete):
                self._start_translation_phase(now)
            return

        assert self._trajectory_start_time is not None
        elapsed = max((now - self._trajectory_start_time).nanoseconds / 1e9, 0.0)
        translation_phase_complete = elapsed >= self._max_translation_duration

        for agent_idx, trajectory in enumerate(self._trajectories):
            if trajectory is None or self._completed[agent_idx]:
                self._publish_zero(agent_idx)
                continue

            phase = self._phases[agent_idx]
            if phase == "track":
                if self._track_agent(agent_idx, trajectory, elapsed, dt, translation_phase_complete):
                    self._reset_integrator(agent_idx)
                    if trajectory.final_goal_yaw is None:
                        self._completed[agent_idx] = True
                        self._phases[agent_idx] = "done"
                        self._publish_zero(agent_idx)
                        self._log_tracking_error_summary(agent_idx, trajectory)
                    else:
                        self._phases[agent_idx] = "post_rotate"
                        self._publish_zero(agent_idx)
                        self.get_logger().info(
                            f"Agent {agent_idx} reached timed translation target and is starting post-rotation."
                        )
                continue

            if phase == "post_rotate":
                if self._rotate_agent_towards(
                    agent_idx,
                    trajectory.final_goal_yaw,
                    self._goal_yaw_tolerance,
                ):
                    self._publish_zero(agent_idx)
                    self._completed[agent_idx] = True
                    self._phases[agent_idx] = "done"
                    self.get_logger().info(
                        f"Agent {agent_idx} finished its final in-place rotation."
                    )
                    self._log_tracking_error_summary(agent_idx, trajectory)
                continue

            self._publish_zero(agent_idx)

        if all(self._completed):
            self.get_logger().info("Finished executing MAPF plan with timed trajectory tracking.")
            self._active = False
            self._translation_started = False
            self._current_plan = None
            self._publish_zero_to_all()
            self._publish_execution_status("succeeded")
            self._start_pending_plan()

    def _reset_execution_state(self, reason: str | None = None) -> None:
        had_execution_state = (
            self._active
            or self._current_plan is not None
            or self._pending_plan is not None
            or any(phase != "idle" for phase in self._phases)
        )

        self._active = False
        self._translation_started = False
        self._current_plan = None
        self._pending_plan = None
        self._trajectories = [None for _ in range(self._agent_num)]
        self._phases = ["idle" for _ in range(self._agent_num)]
        self._pre_rotate_complete = [False for _ in range(self._agent_num)]
        self._completed = [False for _ in range(self._agent_num)]
        self._trajectory_start_time = None
        self._last_control_time = None
        self._max_translation_duration = 0.0
        self._tracking_error_stats = [TrackingErrorStats() for _ in range(self._agent_num)]
        self._tracking_summary_logged = [False for _ in range(self._agent_num)]
        self._tracking_log_rows = [[] for _ in range(self._agent_num)]
        self._reset_all_integrators()
        self._publish_zero_to_all()

        if had_execution_state and reason:
            self.get_logger().info(reason)

    def _start_translation_phase(self, now: Time) -> None:
        self._translation_started = True
        self._trajectory_start_time = now
        self._last_control_time = now
        self._reset_all_integrators()
        for agent_idx, trajectory in enumerate(self._trajectories):
            if trajectory is None or self._completed[agent_idx]:
                continue
            if self._phases[agent_idx] == "pre_rotate":
                self._phases[agent_idx] = "track"
        self.get_logger().info(
            "All agents finished pre-rotation. Starting the shared timed trajectory clock "
            f"at t=0.0s with a global translation horizon of {self._max_translation_duration:.2f}s."
        )

    def _log_reference_separation(self) -> None:
        active_agents = [
            (agent_idx, trajectory)
            for agent_idx, trajectory in enumerate(self._trajectories)
            if trajectory is not None
        ]
        if len(active_agents) < 2:
            return

        sample_dt = max(min(self._mapf_step_duration / 5.0, 0.1), 0.02)
        global_min_distance = float("inf")
        global_min_pair = None
        global_min_time = 0.0

        for left_index in range(len(active_agents) - 1):
            left_agent_idx, left_trajectory = active_agents[left_index]
            for right_index in range(left_index + 1, len(active_agents)):
                right_agent_idx, right_trajectory = active_agents[right_index]
                elapsed = 0.0
                pair_min_distance = float("inf")
                pair_min_time = 0.0
                while elapsed <= self._max_translation_duration + 1e-6:
                    left_ref = self._sample_trajectory(left_trajectory, elapsed)
                    right_ref = self._sample_trajectory(right_trajectory, elapsed)
                    distance = self._distance(
                        left_ref.x,
                        left_ref.y,
                        right_ref.x,
                        right_ref.y,
                    )
                    if distance < pair_min_distance:
                        pair_min_distance = distance
                        pair_min_time = elapsed
                    elapsed += sample_dt

                self.get_logger().info(
                    f"Reference separation agent {left_agent_idx} vs {right_agent_idx}: "
                    f"min={pair_min_distance:.3f} m at t={pair_min_time:.2f}s."
                )
                if pair_min_distance < global_min_distance:
                    global_min_distance = pair_min_distance
                    global_min_pair = (left_agent_idx, right_agent_idx)
                    global_min_time = pair_min_time

        if global_min_pair is not None:
            self.get_logger().info(
                f"Minimum planned inter-agent separation: {global_min_distance:.3f} m "
                f"between agent {global_min_pair[0]} and agent {global_min_pair[1]} "
                f"at t={global_min_time:.2f}s."
            )

    def _track_agent(
        self,
        agent_idx: int,
        trajectory: AgentTrajectory,
        elapsed: float,
        dt: float,
        translation_phase_complete: bool,
    ) -> bool:
        pose = self._lookup_robot_pose(agent_idx)
        if pose is None:
            self._publish_zero(agent_idx)
            return False

        final_sample = trajectory.final_sample
        if final_sample is None:
            self._publish_zero(agent_idx)
            return True

        final_distance = self._distance(pose[0], pose[1], final_sample.x, final_sample.y)
        if translation_phase_complete and final_distance <= self._goal_position_tolerance:
            self._publish_zero(agent_idx)
            return True

        ref = self._sample_trajectory(trajectory, elapsed)
        if (
            elapsed >= trajectory.translation_duration
            and final_distance <= self._slow_down_radius
        ):
            ref = ReferenceState(
                x=ref.x,
                y=ref.y,
                yaw=pose[2],
                linear_velocity=0.0,
                angular_velocity=0.0,
            )
        dx = ref.x - pose[0]
        dy = ref.y - pose[1]
        position_error = math.hypot(dx, dy)
        cos_yaw = math.cos(pose[2])
        sin_yaw = math.sin(pose[2])
        error_x_body = cos_yaw * dx + sin_yaw * dy
        error_y_body = -sin_yaw * dx + cos_yaw * dy
        yaw_error = self._normalize_angle(ref.yaw - pose[2])

        self._integral_x[agent_idx] = self._clamp(
            self._integral_x[agent_idx] + error_x_body * dt,
            -self._integral_limit,
            self._integral_limit,
        )
        self._integral_y[agent_idx] = self._clamp(
            self._integral_y[agent_idx] + error_y_body * dt,
            -self._integral_limit,
            self._integral_limit,
        )
        self._integral_yaw[agent_idx] = self._clamp(
            self._integral_yaw[agent_idx] + yaw_error * dt,
            -self._integral_limit,
            self._integral_limit,
        )

        linear_cmd = (
            ref.linear_velocity
            + self._tracking_longitudinal_kp * error_x_body
            + self._tracking_longitudinal_ki * self._integral_x[agent_idx]
        )
        linear_cmd *= max(0.0, math.cos(yaw_error))

        if final_distance < self._slow_down_radius:
            linear_cmd *= max(final_distance / max(self._slow_down_radius, 1e-6), 0.25)

        angular_cmd = (
            ref.angular_velocity
            + self._tracking_lateral_kp * error_y_body
            + self._tracking_lateral_ki * self._integral_y[agent_idx]
            + self._tracking_yaw_kp * yaw_error
            + self._tracking_yaw_ki * self._integral_yaw[agent_idx]
        )

        clamped_linear_cmd = self._clamp(linear_cmd, 0.0, self._max_linear_speed)
        clamped_angular_cmd = self._clamp(
            angular_cmd,
            -self._max_angular_speed,
            self._max_angular_speed,
        )
        linear_saturated = abs(clamped_linear_cmd - linear_cmd) > 1e-6
        angular_saturated = abs(clamped_angular_cmd - angular_cmd) > 1e-6
        self._tracking_error_stats[agent_idx].update(
            position_error,
            yaw_error,
            ref.linear_velocity,
            clamped_linear_cmd,
            clamped_angular_cmd,
            linear_saturated,
            angular_saturated,
        )
        self._tracking_log_rows[agent_idx].append(
            {
                "elapsed": elapsed,
                "phase": "track",
                "ref_x": ref.x,
                "ref_y": ref.y,
                "ref_yaw": ref.yaw,
                "ref_linear_velocity": ref.linear_velocity,
                "ref_angular_velocity": ref.angular_velocity,
                "actual_x": pose[0],
                "actual_y": pose[1],
                "actual_yaw": pose[2],
                "cmd_linear_x": clamped_linear_cmd,
                "cmd_angular_z": clamped_angular_cmd,
                "position_error": position_error,
                "yaw_error": yaw_error,
                "linear_saturated": int(linear_saturated),
                "angular_saturated": int(angular_saturated),
            }
        )

        twist = Twist()
        twist.linear.x = clamped_linear_cmd
        twist.angular.z = clamped_angular_cmd
        self._cmd_vel_publishers[agent_idx].publish(twist)
        return False

    def _build_trajectory(self, single_plan: SinglePlan) -> AgentTrajectory | None:
        if not single_plan.plan.poses:
            return None

        time_steps = list(single_plan.time_step)
        if len(time_steps) != len(single_plan.plan.poses):
            time_steps = list(range(len(single_plan.plan.poses)))

        base_time_step = time_steps[0] if time_steps else 0
        times = [max((step - base_time_step) * self._mapf_step_duration, 0.0) for step in time_steps]
        poses = [copy.deepcopy(pose) for pose in single_plan.plan.poses]

        final_goal_yaw = self._quaternion_to_yaw(
            self._normalize_quaternion(poses[-1].pose.orientation)
        )
        segment_headings = self._compute_segment_headings(poses)
        first_motion_heading = next((heading for heading in segment_headings if heading is not None), None)
        last_motion_heading = next(
            (heading for heading in reversed(segment_headings) if heading is not None),
            final_goal_yaw,
        )

        sample_yaws: List[float] = []
        for idx in range(len(poses)):
            if idx == len(poses) - 1:
                sample_yaws.append(last_motion_heading)
                continue

            heading = segment_headings[idx]
            if heading is not None:
                sample_yaws.append(heading)
                continue

            next_heading = next(
                (candidate for candidate in segment_headings[idx + 1 :] if candidate is not None),
                last_motion_heading,
            )
            sample_yaws.append(next_heading)

        samples = [
            TrajectorySample(
                time_from_start=times[idx],
                x=poses[idx].pose.position.x,
                y=poses[idx].pose.position.y,
                yaw=sample_yaws[idx],
            )
            for idx in range(len(poses))
        ]

        return AgentTrajectory(
            samples=samples,
            pre_rotate_yaw=first_motion_heading,
            final_goal_yaw=final_goal_yaw,
        )

    def _compute_segment_headings(self, poses: List[PoseStamped]) -> List[float | None]:
        headings: List[float | None] = []
        for idx in range(len(poses) - 1):
            current = poses[idx].pose.position
            nxt = poses[idx + 1].pose.position
            if self._distance(current.x, current.y, nxt.x, nxt.y) <= 1e-8:
                headings.append(None)
                continue
            headings.append(math.atan2(nxt.y - current.y, nxt.x - current.x))
        return headings

    def _sample_trajectory(self, trajectory: AgentTrajectory, elapsed: float) -> ReferenceState:
        samples = trajectory.samples
        if not samples:
            return ReferenceState(0.0, 0.0, 0.0, 0.0, 0.0)

        if len(samples) == 1 or elapsed <= samples[0].time_from_start:
            sample = samples[0]
            return ReferenceState(sample.x, sample.y, sample.yaw, 0.0, 0.0)

        if elapsed >= trajectory.translation_duration:
            sample = samples[-1]
            return ReferenceState(sample.x, sample.y, sample.yaw, 0.0, 0.0)

        for idx in range(len(samples) - 1):
            start = samples[idx]
            end = samples[idx + 1]
            if elapsed > end.time_from_start:
                continue

            dt = max(end.time_from_start - start.time_from_start, 1e-6)
            ratio = self._clamp((elapsed - start.time_from_start) / dt, 0.0, 1.0)
            dx = end.x - start.x
            dy = end.y - start.y
            distance = math.hypot(dx, dy)
            yaw = start.yaw if distance <= 1e-8 else math.atan2(dy, dx)
            linear_velocity = distance / dt if distance > 1e-8 else 0.0
            return ReferenceState(
                x=start.x + ratio * dx,
                y=start.y + ratio * dy,
                yaw=yaw,
                linear_velocity=linear_velocity,
                angular_velocity=0.0,
            )

        sample = samples[-1]
        return ReferenceState(sample.x, sample.y, sample.yaw, 0.0, 0.0)

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
            self._tracking_yaw_kp * yaw_error,
            -self._max_angular_speed,
            self._max_angular_speed,
        )
        self._cmd_vel_publishers[agent_idx].publish(twist)
        return False

    def _reset_integrator(self, agent_idx: int) -> None:
        self._integral_x[agent_idx] = 0.0
        self._integral_y[agent_idx] = 0.0
        self._integral_yaw[agent_idx] = 0.0

    def _reset_all_integrators(self) -> None:
        for agent_idx in range(self._agent_num):
            self._reset_integrator(agent_idx)

    def _log_tracking_error_summary(
        self,
        agent_idx: int,
        trajectory: AgentTrajectory,
    ) -> None:
        if self._tracking_summary_logged[agent_idx]:
            return

        self._tracking_summary_logged[agent_idx] = True
        stats = self._tracking_error_stats[agent_idx]
        pose = self._lookup_robot_pose(agent_idx)
        final_sample = trajectory.final_sample
        log_path = self._write_tracking_log(agent_idx)

        final_position_error_text = "n/a"
        final_yaw_error_text = "n/a"
        if pose is not None and final_sample is not None:
            final_position_error = self._distance(
                pose[0],
                pose[1],
                final_sample.x,
                final_sample.y,
            )
            final_target_yaw = (
                trajectory.final_goal_yaw
                if trajectory.final_goal_yaw is not None
                else final_sample.yaw
            )
            final_yaw_error = abs(self._normalize_angle(final_target_yaw - pose[2]))
            final_position_error_text = f"{final_position_error:.3f} m"
            final_yaw_error_text = f"{final_yaw_error:.3f} rad"

        if stats.sample_count == 0:
            self.get_logger().info(
                f"Agent {agent_idx} tracking error summary: no timed tracking samples were recorded. "
                f"Final pose error: position={final_position_error_text}, yaw={final_yaw_error_text}."
            )
            return

        position_rmse = math.sqrt(stats.sum_sq_position_error / stats.sample_count)
        position_mae = stats.sum_abs_position_error / stats.sample_count
        yaw_rmse = math.sqrt(stats.sum_sq_yaw_error / stats.sample_count)
        yaw_mae = stats.sum_abs_yaw_error / stats.sample_count
        linear_saturation_ratio = stats.linear_saturation_count / stats.sample_count
        angular_saturation_ratio = stats.angular_saturation_count / stats.sample_count
        self.get_logger().info(
            f"Agent {agent_idx} tracking error summary: "
            f"position_rmse={position_rmse:.3f} m, "
            f"position_mae={position_mae:.3f} m, "
            f"position_max={stats.max_position_error:.3f} m, "
            f"yaw_rmse={yaw_rmse:.3f} rad, "
            f"yaw_mae={yaw_mae:.3f} rad, "
            f"yaw_max={stats.max_yaw_error:.3f} rad, "
            f"max_ref_linear_velocity={stats.max_ref_linear_velocity:.3f} m/s, "
            f"max_cmd_linear_velocity={stats.max_cmd_linear_velocity:.3f} m/s, "
            f"max_cmd_angular_velocity={stats.max_cmd_angular_velocity:.3f} rad/s, "
            f"linear_saturation_ratio={linear_saturation_ratio:.2%}, "
            f"angular_saturation_ratio={angular_saturation_ratio:.2%}, "
            f"final_position_error={final_position_error_text}, "
            f"final_yaw_error={final_yaw_error_text}, "
            f"samples={stats.sample_count}, "
            f"log_file={log_path if log_path is not None else 'disabled'}."
        )

    def _write_tracking_log(self, agent_idx: int) -> str | None:
        if not self._save_tracking_log:
            return None

        rows = self._tracking_log_rows[agent_idx]
        if not rows:
            return None

        os.makedirs(self._tracking_log_dir, exist_ok=True)
        file_name = (
            f"mapf_timed_tracker_pid{os.getpid()}_exec{self._execution_index:03d}_"
            f"{self._agent_names[agent_idx]}.csv"
        )
        path = os.path.join(self._tracking_log_dir, file_name)
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _publish_zero(self, agent_idx: int) -> None:
        self._cmd_vel_publishers[agent_idx].publish(Twist())

    def _publish_zero_to_all(self) -> None:
        for agent_idx in range(self._agent_num):
            self._publish_zero(agent_idx)

    def _publish_execution_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self._execution_status_pub.publish(msg)

    def _rollout_control_callback(self, msg: PoseArray) -> None:
        rollout_id = parse_rollout_id(msg.header.frame_id)
        if rollout_id is None:
            return

        if rollout_id <= 0:
            self._reset_execution_state("Received rollout stop signal. Clearing timed tracker state.")
            self._active_rollout_id = None
            self._tracking_log_dir = self._default_tracking_log_dir
            return

        if self._active_rollout_id == rollout_id:
            return

        self._reset_execution_state(
            f"Preparing timed tracker for rollout {rollout_id}. Clearing any stale execution state."
        )
        self._active_rollout_id = rollout_id
        if self._team_config_path is None:
            return

        self._tracking_log_dir = str(
            rollout_run_dir(
                self._experiments_dir,
                team_config_path=self._team_config_path,
                rollout_id=rollout_id,
            )
        )
        self.get_logger().info(
            f"Tracking logs for rollout {rollout_id} will be written under {self._tracking_log_dir}."
        )

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
    node = MapfTimedTracker()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
