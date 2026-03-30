# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class InitialPoseTfPublisher(Node):
    """Publish static map-to-odom transforms from configured robot spawn poses."""

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

        self._global_frame = str(self.get_parameter("global_frame").value)
        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._robot_namespaces = list(self.get_parameter("robot_namespaces").value)
        flat_poses = list(self.get_parameter("initial_poses").value)

        expected_values = len(self._robot_namespaces) * 7
        if len(flat_poses) != expected_values:
            raise ValueError(
                f"Expected {expected_values} initial pose values for "
                f"{len(self._robot_namespaces)} robots, got {len(flat_poses)}."
            )

        self._static_broadcaster = StaticTransformBroadcaster(self)
        tf_static_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._local_tf_static_pubs = {
            namespace: self.create_publisher(
                TFMessage,
                f"/{namespace}/tf_static",
                tf_static_qos,
            )
            for namespace in self._robot_namespaces
        }

        global_transforms = self._build_global_transforms(flat_poses)
        self._static_broadcaster.sendTransform(global_transforms)
        self._publish_local_static_transforms(flat_poses)

        for transform in global_transforms:
            self.get_logger().info(
                "Published static TF "
                f"{transform.header.frame_id} -> {transform.child_frame_id}"
            )

    def _build_global_transforms(self, flat_poses: List[float]) -> List[TransformStamped]:
        transforms: List[TransformStamped] = []
        stamp = self.get_clock().now().to_msg()

        for idx, namespace in enumerate(self._robot_namespaces):
            offset = idx * 7
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self._global_frame
            transform.child_frame_id = f"{namespace}/{self._odom_frame}"
            transform.transform.translation.x = float(flat_poses[offset + 0])
            transform.transform.translation.y = float(flat_poses[offset + 1])
            transform.transform.translation.z = float(flat_poses[offset + 2])
            transform.transform.rotation.x = float(flat_poses[offset + 3])
            transform.transform.rotation.y = float(flat_poses[offset + 4])
            transform.transform.rotation.z = float(flat_poses[offset + 5])
            transform.transform.rotation.w = float(flat_poses[offset + 6])
            transforms.append(transform)

        return transforms

    def _publish_local_static_transforms(self, flat_poses: List[float]) -> None:
        stamp = self.get_clock().now().to_msg()

        for idx, namespace in enumerate(self._robot_namespaces):
            offset = idx * 7
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self._global_frame
            transform.child_frame_id = self._odom_frame
            transform.transform.translation.x = float(flat_poses[offset + 0])
            transform.transform.translation.y = float(flat_poses[offset + 1])
            transform.transform.translation.z = float(flat_poses[offset + 2])
            transform.transform.rotation.x = float(flat_poses[offset + 3])
            transform.transform.rotation.y = float(flat_poses[offset + 4])
            transform.transform.rotation.z = float(flat_poses[offset + 5])
            transform.transform.rotation.w = float(flat_poses[offset + 6])
            self._local_tf_static_pubs[namespace].publish(TFMessage(transforms=[transform]))
            self.get_logger().info(
                f"Published local static TF on /{namespace}/tf_static: "
                f"{transform.header.frame_id} -> {transform.child_frame_id}"
            )


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
