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
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def _load_yaml_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _extract_robot_namespaces(config: Any) -> list[str]:
    robots = config.get("robots", []) if isinstance(config, dict) else []
    namespaces: list[str] = []
    for index, robot in enumerate(robots):
        if not isinstance(robot, dict):
            continue
        namespaces.append(str(robot.get("name", f"robot{index + 1}")))
    return namespaces


def _find_repo_root(search_paths: list[Path]) -> Path:
    for root in search_paths:
        resolved = root.resolve()
        candidates = [resolved] + list(resolved.parents)
        for candidate in candidates:
            if (candidate / ".gitignore").exists() and (candidate / "ros2_ws").exists():
                return candidate
    return Path.cwd().resolve()


def _resolve_experiments_root(experiments_dir: str, team_config_path: Path) -> Path:
    if experiments_dir:
        return Path(experiments_dir).expanduser().resolve()

    repo_root = _find_repo_root([Path.cwd(), team_config_path, Path(__file__)])
    return repo_root / "experiments"


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


class RobotVelocityRecorder(Node):
    def __init__(self) -> None:
        super().__init__("robot_velocity_recorder")

        self.declare_parameter("team_config_file", "")
        # An empty Python list is inferred as BYTE_ARRAY by rclpy, which breaks
        # launch-time string-array overrides. Use a placeholder string entry and
        # drop blank values after reading the parameter.
        self.declare_parameter("robot_namespaces", [""])
        self.declare_parameter("odom_topic_suffix", "chassis/odom")
        self.declare_parameter("record_frequency_hz", 20.0)
        self.declare_parameter("experiments_dir", "")

        team_config_value = str(self.get_parameter("team_config_file").value).strip()
        if not team_config_value:
            raise ValueError("team_config_file must be provided to RobotVelocityRecorder.")

        self._team_config_path = Path(team_config_value).expanduser().resolve()
        if not self._team_config_path.exists():
            raise FileNotFoundError(f"Team config file not found: {self._team_config_path}")

        self._team_config = _load_yaml_file(self._team_config_path) or {}
        requested_namespaces = list(self.get_parameter("robot_namespaces").value)
        self._robot_namespaces = [str(name) for name in requested_namespaces if str(name).strip()]
        if not self._robot_namespaces:
            self._robot_namespaces = _extract_robot_namespaces(self._team_config)
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
        experiments_dir = str(self.get_parameter("experiments_dir").value).strip()
        self._experiments_root = _resolve_experiments_root(experiments_dir, self._team_config_path)

        run_config_folder = self._team_config_path.stem.strip()
        run_config_dir = self._experiments_root / run_config_folder
        run_id = _next_run_id(run_config_dir)
        self._run_dir = run_config_dir / str(run_id)
        self._run_dir.mkdir(parents=True, exist_ok=False)

        self._run_id = run_id
        self._metadata_path = self._run_dir / "run_config.yaml"
        self._created_at = dt.datetime.now().astimezone().isoformat()

        self._sample_counts = {name: 0 for name in self._robot_namespaces}
        self._last_recorded_stamp_ns = {name: None for name in self._robot_namespaces}
        self._file_handles: dict[str, Any] = {}
        self._csv_writers: dict[str, csv.writer] = {}
        self._file_paths: dict[str, Path] = {}
        self._odom_subscriptions = []

        for namespace in self._robot_namespaces:
            file_path = self._run_dir / f"{namespace}_velocity.csv"
            file_handle = file_path.open("w", encoding="utf-8", newline="")
            writer = csv.writer(file_handle)
            writer.writerow(["timestamp_ns", "vx", "vy", "wz"])
            file_handle.flush()
            self._file_handles[namespace] = file_handle
            self._csv_writers[namespace] = writer
            self._file_paths[namespace] = file_path

            topic_name = _join_topic(namespace, self._odom_topic_suffix)
            self._odom_subscriptions.append(
                self.create_subscription(
                    Odometry,
                    topic_name,
                    lambda msg, robot_name=namespace: self._odom_callback(robot_name, msg),
                    50,
                )
            )

        self._write_metadata()
        self.get_logger().info(
            "Recording simulator odometry velocities for "
            f"{len(self._robot_namespaces)} robots into {self._run_dir}."
        )

    def _odom_callback(self, robot_name: str, msg: Odometry) -> None:
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
        metadata = {
            "run_id": self._run_id,
            "created_at": self._created_at,
            "run_directory": str(self._run_dir),
            "run_config_name": self._team_config_path.name,
            "run_config_path": str(self._team_config_path),
            "language_instruction": "",
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
            "team_config": self._team_config,
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
        self._write_metadata()
        for file_handle in self._file_handles.values():
            file_handle.close()
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
