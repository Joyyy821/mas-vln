#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any

import rclpy
import yaml
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from carters_goal.rollout_control import parse_rollout_id
from carters_goal.shared_team_config import (
    resolve_experiments_root,
    rollout_run_dir,
    team_config_utils,
)


def _next_run_id(run_config_dir: Path) -> int:
    if not run_config_dir.exists():
        return 1

    existing_ids: list[int] = []
    for candidate in run_config_dir.iterdir():
        if not candidate.is_dir():
            continue
        if not candidate.name.isdigit():
            continue
        existing_ids.append(int(candidate.name))

    if not existing_ids:
        return 1
    return max(existing_ids) + 1


def _join_topic(namespace: str, suffix: str) -> str:
    clean_suffix = suffix.strip("/")
    if not clean_suffix:
        return f"/{namespace}"
    return f"/{namespace}/{clean_suffix}"


def _stamp_to_nanoseconds(msg: Odometry) -> int:
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _pose_array_to_legacy_pose_dict(pose_array: list[float]) -> dict[str, float]:
    pose_dict = team_config_utils.pose_array_to_pose_dict(pose_array)
    return {
        "x": float(pose_dict["x"]),
        "y": float(pose_dict["y"]),
        "z": float(pose_dict["z"]),
        "yaw": float(pose_dict["yaw"]),
    }


def _rollout_to_legacy_team_config(
    team_config: dict[str, Any],
    rollout: dict[str, Any],
) -> dict[str, Any]:
    return {
        "environment": dict(team_config.get("environment", {})),
        "robots": [
            {
                "name": str(robot["name"]),
                "initial_pose": _pose_array_to_legacy_pose_dict(robot["initial_pose"]),
                "goal_pose": _pose_array_to_legacy_pose_dict(robot["goal_pose"]),
            }
            for robot in rollout.get("robots", [])
        ],
    }


class RobotVelocityRecorder(Node):
    def __init__(self) -> None:
        super().__init__("robot_velocity_recorder")

        self.declare_parameter("team_config_file", "")
        self.declare_parameter("robot_namespaces", [""])
        self.declare_parameter("odom_topic_suffix", "chassis/odom")
        self.declare_parameter("record_frequency_hz", 20.0)
        self.declare_parameter("experiments_dir", "")
        self.declare_parameter("rollout_control_topic", "")

        team_config_value = str(self.get_parameter("team_config_file").value).strip()
        if not team_config_value:
            raise ValueError("team_config_file must be provided to RobotVelocityRecorder.")

        self._team_config_path = Path(team_config_value).expanduser().resolve()
        if not self._team_config_path.exists():
            raise FileNotFoundError(f"Team config file not found: {self._team_config_path}")

        self._team_config = team_config_utils.load_multi_rollout_config(str(self._team_config_path))
        self._rollout_by_id = {
            int(rollout["id"]): rollout for rollout in self._team_config["rollouts"]
        }
        requested_namespaces = list(self.get_parameter("robot_namespaces").value)
        self._robot_namespaces = [str(name) for name in requested_namespaces if str(name).strip()]
        if not self._robot_namespaces:
            self._robot_namespaces = list(self._team_config["robot_namespaces"])
        if not self._robot_namespaces:
            raise ValueError(
                "RobotVelocityRecorder could not determine any robot namespaces from parameters "
                f"or {self._team_config_path}."
            )

        self._odom_topic_suffix = str(self.get_parameter("odom_topic_suffix").value).strip()
        self._record_frequency_hz = max(float(self.get_parameter("record_frequency_hz").value), 0.0)
        self._min_period_ns = (
            int(1_000_000_000.0 / self._record_frequency_hz)
            if self._record_frequency_hz > 0.0
            else 0
        )
        self._experiments_dir = str(self.get_parameter("experiments_dir").value).strip()
        self._experiments_root = resolve_experiments_root(
            self._experiments_dir, self._team_config_path
        )
        self._rollout_control_topic = str(self.get_parameter("rollout_control_topic").value).strip()

        self._active_rollout_id: int | None = None
        self._active_rollout: dict[str, Any] | None = None
        self._run_id: int | None = None
        self._run_dir: Path | None = None
        self._metadata_path: Path | None = None
        self._created_at = ""

        self._sample_counts = {name: 0 for name in self._robot_namespaces}
        self._last_recorded_stamp_ns = {name: None for name in self._robot_namespaces}
        self._file_handles: dict[str, Any] = {}
        self._csv_writers: dict[str, csv.writer] = {}
        self._file_paths: dict[str, Path] = {}
        self._odom_subscriptions = []

        for namespace in self._robot_namespaces:
            topic_name = _join_topic(namespace, self._odom_topic_suffix)
            self._odom_subscriptions.append(
                self.create_subscription(
                    Odometry,
                    topic_name,
                    lambda msg, robot_name=namespace: self._odom_callback(robot_name, msg),
                    50,
                )
            )

        self._control_sub = None
        if self._rollout_control_topic:
            self._control_sub = self.create_subscription(
                PoseArray,
                self._rollout_control_topic,
                self._rollout_control_callback,
                10,
            )
            self.get_logger().info(
                f"Waiting for rollout-control messages on {self._rollout_control_topic}."
            )
        else:
            default_rollout_id = int(self._team_config["first_rollout"]["id"])
            self._activate_rollout(default_rollout_id, use_legacy_directory=True)

    def _rollout_control_callback(self, msg: PoseArray) -> None:
        rollout_id = parse_rollout_id(msg.header.frame_id)
        if rollout_id is None:
            return

        if rollout_id <= 0:
            self._close_active_rollout()
            return

        self._activate_rollout(rollout_id, use_legacy_directory=False)

    def _activate_rollout(self, rollout_id: int, *, use_legacy_directory: bool) -> None:
        if self._active_rollout_id == rollout_id and self._run_dir is not None:
            return

        rollout = self._rollout_by_id.get(rollout_id)
        if rollout is None:
            self.get_logger().error(
                f"Ignoring rollout {rollout_id} because it is not defined in {self._team_config_path}."
            )
            return

        self._close_active_rollout()

        if use_legacy_directory:
            run_config_dir = self._experiments_root / self._team_config_path.stem.strip()
            run_id = _next_run_id(run_config_dir)
            run_dir = run_config_dir / str(run_id)
        else:
            run_id = rollout_id
            run_dir = rollout_run_dir(
                self._experiments_dir,
                team_config_path=self._team_config_path,
                rollout_id=rollout_id,
            )
            if run_dir.exists():
                raise RuntimeError(
                    f"Refusing to overwrite existing rollout directory {run_dir}. "
                    "Use skip_existed_rollout on the rollout manager to skip it."
                )

        run_dir.mkdir(parents=True, exist_ok=False)

        self._active_rollout_id = rollout_id
        self._active_rollout = rollout
        self._run_id = run_id
        self._run_dir = run_dir
        self._metadata_path = run_dir / "run_config.yaml"
        self._created_at = dt.datetime.now().astimezone().isoformat()
        self._sample_counts = {name: 0 for name in self._robot_namespaces}
        self._last_recorded_stamp_ns = {name: None for name in self._robot_namespaces}
        self._file_handles = {}
        self._csv_writers = {}
        self._file_paths = {}

        for namespace in self._robot_namespaces:
            file_path = run_dir / f"{namespace}_velocity.csv"
            file_handle = file_path.open("w", encoding="utf-8", newline="")
            writer = csv.writer(file_handle)
            writer.writerow(["timestamp_ns", "vx", "vy", "wz"])
            file_handle.flush()
            self._file_handles[namespace] = file_handle
            self._csv_writers[namespace] = writer
            self._file_paths[namespace] = file_path

        self._write_metadata()
        self.get_logger().info(
            "Recording simulator odometry velocities for rollout "
            f"{rollout_id} into {run_dir}."
        )

    def _close_active_rollout(self) -> None:
        if self._run_dir is None:
            self._active_rollout_id = None
            self._active_rollout = None
            return

        self._write_metadata()
        for file_handle in self._file_handles.values():
            file_handle.close()

        self._active_rollout_id = None
        self._active_rollout = None
        self._run_id = None
        self._run_dir = None
        self._metadata_path = None
        self._created_at = ""
        self._file_handles = {}
        self._csv_writers = {}
        self._file_paths = {}

    def _odom_callback(self, robot_name: str, msg: Odometry) -> None:
        if self._active_rollout_id is None:
            return

        stamp_ns = _stamp_to_nanoseconds(msg)
        if stamp_ns <= 0:
            stamp_ns = self.get_clock().now().nanoseconds

        last_recorded_stamp_ns = self._last_recorded_stamp_ns[robot_name]
        if last_recorded_stamp_ns is not None and stamp_ns <= last_recorded_stamp_ns:
            last_recorded_stamp_ns = None
            self._last_recorded_stamp_ns[robot_name] = None
        if (
            self._min_period_ns > 0
            and last_recorded_stamp_ns is not None
            and stamp_ns - last_recorded_stamp_ns < self._min_period_ns
        ):
            return

        self._csv_writers[robot_name].writerow(
            [
                stamp_ns,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.angular.z,
            ]
        )
        self._file_handles[robot_name].flush()
        self._last_recorded_stamp_ns[robot_name] = stamp_ns
        self._sample_counts[robot_name] += 1

    def _write_metadata(self) -> None:
        if self._run_dir is None or self._metadata_path is None:
            return
        if self._active_rollout is None:
            return

        team_config_snapshot = _rollout_to_legacy_team_config(
            self._team_config,
            self._active_rollout,
        )

        metadata = {
            "run_id": self._run_id,
            "created_at": self._created_at,
            "run_directory": str(self._run_dir),
            "run_config_name": self._team_config_path.name,
            "run_config_path": str(self._team_config_path),
            "language_instruction": str(self._team_config.get("language_instruction", "") or ""),
            "record_settings": {
                "source": "simulator_odometry",
                "odom_topic_suffix": self._odom_topic_suffix,
                "record_frequency_hz": self._record_frequency_hz,
                "timestamp_field": "header.stamp",
                "velocity_fields": {
                    "vx": "twist.twist.linear.x",
                    "vy": "twist.twist.linear.y",
                    "wz": "twist.twist.angular.z",
                },
                "timestamp_units": "nanoseconds",
                "message_type": "nav_msgs/msg/Odometry",
            },
            "team_config": team_config_snapshot,
            "artifacts": {
                namespace: self._file_paths[namespace].name for namespace in self._robot_namespaces
            },
            "robots": [
                {
                    "name": namespace,
                    "odom_topic": _join_topic(namespace, self._odom_topic_suffix),
                    "recording_file": self._file_paths[namespace].name,
                    "sample_count": self._sample_counts[namespace],
                }
                for namespace in self._robot_namespaces
            ],
        }
        with self._metadata_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False)

    def destroy_node(self) -> bool:
        self._close_active_rollout()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node: RobotVelocityRecorder | None = None
    try:
        node = RobotVelocityRecorder()
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
