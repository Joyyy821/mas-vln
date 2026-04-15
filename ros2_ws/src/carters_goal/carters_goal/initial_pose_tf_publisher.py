#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import rclpy
from geometry_msgs.msg import PoseArray, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32
from tf2_msgs.msg import TFMessage

from carters_goal.rollout_control import parse_rollout_id, pose_array_to_flat_list


def _stamp_to_nanoseconds(sec: int, nanosec: int) -> int:
    return int(sec) * 1_000_000_000 + int(nanosec)


def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * float(yaw)
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _pose_array_to_pose_dict(pose_array: list[float]) -> dict[str, float]:
    return {
        "x": float(pose_array[0]),
        "y": float(pose_array[1]),
        "z": float(pose_array[2]),
        "qx": float(pose_array[3]),
        "qy": float(pose_array[4]),
        "qz": float(pose_array[5]),
        "qw": float(pose_array[6]),
        "yaw": _quaternion_to_yaw(
            float(pose_array[3]),
            float(pose_array[4]),
            float(pose_array[5]),
            float(pose_array[6]),
        ),
    }


def _join_topic(namespace: str, suffix: str) -> str:
    clean_suffix = suffix.strip("/")
    if not clean_suffix:
        return f"/{namespace}"
    return f"/{namespace}/{clean_suffix}"


class InitialPoseTfPublisher(Node):
    """Publish piecewise-constant map-to-odom transforms for rollout resets."""

    def __init__(self) -> None:
        super().__init__("initial_pose_tf_publisher")

        self.declare_parameter("global_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("robot_namespaces", ["robot1", "robot2"])
        self.declare_parameter(
            "initial_poses",
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                2.5,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        )
        self.declare_parameter("rollout_control_topic", "")
        self.declare_parameter("rollout_reset_done_topic", "")
        self.declare_parameter("odom_topic_suffix", "chassis/odom")
        self.declare_parameter("reset_alignment_timeout_sec", 0.5)
        self.declare_parameter("publish_frequency_hz", 10.0)
        self.declare_parameter("publish_global_tf", False)

        self._global_frame = str(self.get_parameter("global_frame").value)
        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._robot_namespaces = list(self.get_parameter("robot_namespaces").value)
        self._flat_poses = [float(value) for value in self.get_parameter("initial_poses").value]
        self._publish_global_tf = bool(self.get_parameter("publish_global_tf").value)
        self._odom_topic_suffix = str(self.get_parameter("odom_topic_suffix").value).strip()
        rollout_control_topic = str(self.get_parameter("rollout_control_topic").value).strip()
        rollout_reset_done_topic = str(
            self.get_parameter("rollout_reset_done_topic").value
        ).strip()
        publish_frequency_hz = max(float(self.get_parameter("publish_frequency_hz").value), 1.0)
        self._reset_alignment_timeout_ns = int(
            max(float(self.get_parameter("reset_alignment_timeout_sec").value), 0.0) * 1e9
        )

        expected_values = len(self._robot_namespaces) * 7
        if len(self._flat_poses) != expected_values:
            raise ValueError(
                f"Expected {expected_values} initial pose values for "
                f"{len(self._robot_namespaces)} robots, got {len(self._flat_poses)}."
            )

        self._pending_rollout_id: int | None = None
        self._pending_flat_poses: list[float] | None = None
        self._alignment_rollout_id: int | None = None
        self._alignment_flat_poses: list[float] | None = None
        self._alignment_started_ns: int | None = None
        self._last_applied_rollout_id: int | None = None
        self._last_reset_done_rollout_id: int | None = None
        self._latest_odom_by_namespace: dict[str, dict[str, Any] | None] = {
            namespace: None for namespace in self._robot_namespaces
        }

        tf_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )
        self._local_tf_pubs = {
            namespace: self.create_publisher(TFMessage, f"/{namespace}/tf", tf_qos)
            for namespace in self._robot_namespaces
        }
        self._global_tf_pub = (
            self.create_publisher(TFMessage, "/tf", tf_qos) if self._publish_global_tf else None
        )

        self._odom_subs = []
        for namespace in self._robot_namespaces:
            self._odom_subs.append(
                self.create_subscription(
                    Odometry,
                    _join_topic(namespace, self._odom_topic_suffix),
                    lambda msg, robot_namespace=namespace: self._odom_callback(
                        robot_namespace, msg
                    ),
                    20,
                )
            )

        self._control_sub = None
        self._reset_done_sub = None
        if rollout_control_topic:
            self._control_sub = self.create_subscription(
                PoseArray,
                rollout_control_topic,
                self._rollout_control_callback,
                10,
            )
        if rollout_reset_done_topic:
            self._reset_done_sub = self.create_subscription(
                Int32,
                rollout_reset_done_topic,
                self._rollout_reset_done_callback,
                10,
            )

        self._timer = self.create_timer(1.0 / publish_frequency_hz, self._publish_transforms)
        self._publish_transforms()

    def _rollout_control_callback(self, msg: PoseArray) -> None:
        rollout_id = parse_rollout_id(msg.header.frame_id)
        if rollout_id is None:
            return

        if rollout_id <= 0:
            self._clear_pending_rollout_state()
            return

        flat_poses = pose_array_to_flat_list(msg)
        expected_values = len(self._robot_namespaces) * 7
        if len(flat_poses) != expected_values:
            self.get_logger().warn(
                f"Ignoring rollout {rollout_id} pose update because it contains "
                f"{len(flat_poses)} values and {expected_values} were expected."
            )
            return

        if self._reset_done_sub is not None:
            self._pending_rollout_id = rollout_id
            self._pending_flat_poses = flat_poses
            if (
                self._last_reset_done_rollout_id == rollout_id
                and rollout_id != self._last_applied_rollout_id
            ):
                self._start_alignment(rollout_id, flat_poses)
            return

        self._start_alignment(rollout_id, flat_poses)

    def _rollout_reset_done_callback(self, msg: Int32) -> None:
        rollout_id = int(msg.data)
        if rollout_id <= 0:
            self._clear_pending_rollout_state()
            return

        self._last_reset_done_rollout_id = rollout_id
        if rollout_id == self._last_applied_rollout_id:
            return

        if rollout_id != self._pending_rollout_id or self._pending_flat_poses is None:
            return

        self._start_alignment(rollout_id, self._pending_flat_poses)

    def _odom_callback(self, namespace: str, msg: Odometry) -> None:
        pose = msg.pose.pose
        self._latest_odom_by_namespace[namespace] = {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "yaw": _quaternion_to_yaw(
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
            "stamp_ns": _stamp_to_nanoseconds(msg.header.stamp.sec, msg.header.stamp.nanosec),
        }
        self._maybe_finalize_pending_alignment(force=False)

    def _start_alignment(self, rollout_id: int, desired_base_flat_poses: list[float]) -> None:
        self._alignment_rollout_id = rollout_id
        self._alignment_flat_poses = list(desired_base_flat_poses)
        self._alignment_started_ns = self.get_clock().now().nanoseconds
        self._maybe_finalize_pending_alignment(force=False)

    def _maybe_finalize_pending_alignment(self, *, force: bool) -> None:
        if self._alignment_rollout_id is None or self._alignment_flat_poses is None:
            return

        rollout_id = self._alignment_rollout_id
        desired_base_flat_poses = list(self._alignment_flat_poses)
        alignment_started_ns = self._alignment_started_ns or 0

        latest_odom_entries = [
            self._latest_odom_by_namespace[namespace] for namespace in self._robot_namespaces
        ]
        have_all_odom = all(entry is not None for entry in latest_odom_entries)
        have_fresh_odom = have_all_odom and all(
            int(entry["stamp_ns"]) >= alignment_started_ns
            for entry in latest_odom_entries
            if entry is not None
        )

        if not force and not have_fresh_odom:
            return

        if have_all_odom:
            map_to_odom_flat_poses = self._build_map_to_odom_pose_array(desired_base_flat_poses)
            aligned_with_odom = True
        else:
            map_to_odom_flat_poses = desired_base_flat_poses
            aligned_with_odom = False
            self.get_logger().warn(
                f"Rollout {rollout_id} map->odom alignment is missing odometry for at least one robot. "
                "Falling back to direct desired poses."
            )

        self._alignment_rollout_id = None
        self._alignment_flat_poses = None
        self._alignment_started_ns = None
        self._pending_rollout_id = None
        self._pending_flat_poses = None
        self._apply_map_to_odom_poses(
            rollout_id,
            map_to_odom_flat_poses,
            aligned_with_odom=aligned_with_odom,
        )

    def _build_map_to_odom_pose_array(self, desired_base_flat_poses: list[float]) -> list[float]:
        map_to_odom_flat_poses: list[float] = []

        for idx, namespace in enumerate(self._robot_namespaces):
            desired_pose = _pose_array_to_pose_dict(
                desired_base_flat_poses[idx * 7 : (idx + 1) * 7]
            )
            odom_pose = self._latest_odom_by_namespace[namespace]
            if odom_pose is None:
                raise RuntimeError(
                    f"Missing odometry pose for {namespace} while aligning map->odom."
                )

            map_to_odom_yaw = float(desired_pose["yaw"]) - float(odom_pose["yaw"])
            cos_yaw = math.cos(map_to_odom_yaw)
            sin_yaw = math.sin(map_to_odom_yaw)
            rotated_odom_x = cos_yaw * float(odom_pose["x"]) - sin_yaw * float(odom_pose["y"])
            rotated_odom_y = sin_yaw * float(odom_pose["x"]) + cos_yaw * float(odom_pose["y"])
            qx, qy, qz, qw = _yaw_to_quaternion(map_to_odom_yaw)

            map_to_odom_flat_poses.extend(
                [
                    float(desired_pose["x"]) - rotated_odom_x,
                    float(desired_pose["y"]) - rotated_odom_y,
                    float(desired_pose["z"]) - float(odom_pose["z"]),
                    qx,
                    qy,
                    qz,
                    qw,
                ]
            )

        return map_to_odom_flat_poses

    def _apply_map_to_odom_poses(
        self,
        rollout_id: int,
        flat_poses: list[float],
        *,
        aligned_with_odom: bool,
    ) -> None:
        self._flat_poses = list(flat_poses)
        self._last_applied_rollout_id = rollout_id
        alignment_label = "odometry-aligned" if aligned_with_odom else "direct"
        self.get_logger().info(
            f"Updated map->odom transforms for rollout {rollout_id} ({alignment_label})."
        )
        self._publish_transforms()

    def _clear_pending_rollout_state(self) -> None:
        self._pending_rollout_id = None
        self._pending_flat_poses = None
        self._alignment_rollout_id = None
        self._alignment_flat_poses = None
        self._alignment_started_ns = None
        self._last_reset_done_rollout_id = None

    def _publish_transforms(self) -> None:
        if (
            self._alignment_rollout_id is not None
            and self._alignment_started_ns is not None
            and self.get_clock().now().nanoseconds - self._alignment_started_ns
            >= self._reset_alignment_timeout_ns
        ):
            self._maybe_finalize_pending_alignment(force=True)

        stamp = self.get_clock().now().to_msg()
        global_transforms: list[TransformStamped] = []

        for idx, namespace in enumerate(self._robot_namespaces):
            offset = idx * 7

            local_transform = TransformStamped()
            local_transform.header.stamp = stamp
            local_transform.header.frame_id = self._global_frame
            local_transform.child_frame_id = self._odom_frame
            local_transform.transform.translation.x = float(self._flat_poses[offset + 0])
            local_transform.transform.translation.y = float(self._flat_poses[offset + 1])
            local_transform.transform.translation.z = float(self._flat_poses[offset + 2])
            local_transform.transform.rotation.x = float(self._flat_poses[offset + 3])
            local_transform.transform.rotation.y = float(self._flat_poses[offset + 4])
            local_transform.transform.rotation.z = float(self._flat_poses[offset + 5])
            local_transform.transform.rotation.w = float(self._flat_poses[offset + 6])

            self._local_tf_pubs[namespace].publish(TFMessage(transforms=[local_transform]))

            if self._global_tf_pub is not None:
                global_transform = TransformStamped()
                global_transform.header.stamp = stamp
                global_transform.header.frame_id = self._global_frame
                global_transform.child_frame_id = f"{namespace}/{self._odom_frame}"
                global_transform.transform = local_transform.transform
                global_transforms.append(global_transform)

        if self._global_tf_pub is not None and global_transforms:
            self._global_tf_pub.publish(TFMessage(transforms=global_transforms))


def main() -> None:
    rclpy.init()
    node = InitialPoseTfPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
