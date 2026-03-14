# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from tf2_msgs.msg import TFMessage


class NamespacedTfBridge(Node):
    def __init__(self) -> None:
        super().__init__("namespaced_tf_bridge")

        self.declare_parameter("robot_namespaces", ["robot1", "robot2"])
        self.declare_parameter("output_tf_topic", "/tf")
        self.declare_parameter("output_tf_static_topic", "/tf_static")
        self.declare_parameter("keep_global_frames", ["map", "world"])

        self._robot_namespaces = list(self.get_parameter("robot_namespaces").value)
        self._keep_global_frames = set(self.get_parameter("keep_global_frames").value)

        tf_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )
        tf_static_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )

        output_tf_topic = self.get_parameter("output_tf_topic").value
        output_tf_static_topic = self.get_parameter("output_tf_static_topic").value
        self._tf_pub = self.create_publisher(TFMessage, output_tf_topic, tf_qos)
        self._tf_static_pub = self.create_publisher(TFMessage, output_tf_static_topic, tf_static_qos)

        self._tf_subs = []
        self._tf_static_subs = []
        for namespace in self._robot_namespaces:
            tf_topic = f"/{namespace}/tf"
            tf_static_topic = f"/{namespace}/tf_static"

            self._tf_subs.append(
                self.create_subscription(
                    TFMessage,
                    tf_topic,
                    self._make_callback(namespace, is_static=False),
                    tf_qos,
                )
            )
            self._tf_static_subs.append(
                self.create_subscription(
                    TFMessage,
                    tf_static_topic,
                    self._make_callback(namespace, is_static=True),
                    tf_static_qos,
                )
            )

        self.get_logger().info(
            "Bridging TF from namespaces "
            f"{self._robot_namespaces} to global /tf and /tf_static."
        )

    def _normalize_frame(self, frame_id: str) -> str:
        return frame_id[1:] if frame_id.startswith("/") else frame_id

    def _prefix_frame(self, frame_id: str, namespace: str) -> str:
        frame = self._normalize_frame(frame_id)
        if frame == "" or frame in self._keep_global_frames:
            return frame
        if frame.startswith(f"{namespace}/"):
            return frame
        return f"{namespace}/{frame}"

    def _make_callback(self, namespace: str, is_static: bool):
        def _callback(msg: TFMessage) -> None:
            bridged = TFMessage()
            for transform in msg.transforms:
                transform.header.frame_id = self._prefix_frame(
                    transform.header.frame_id, namespace
                )
                transform.child_frame_id = self._prefix_frame(
                    transform.child_frame_id, namespace
                )
                bridged.transforms.append(transform)

            if not bridged.transforms:
                return

            if is_static:
                self._tf_static_pub.publish(bridged)
            else:
                self._tf_pub.publish(bridged)

        return _callback


def main() -> None:
    rclpy.init()
    node = NamespacedTfBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
