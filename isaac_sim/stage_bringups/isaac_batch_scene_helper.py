#!/usr/bin/env python3
"""
ROS 2 Humble helper that bridges randomized-warehouse batch scene control to Isaac Sim.

This script runs with system Python outside Isaac Sim so it can import the ROS 2
Humble rclpy packages. It forwards JSON std_msgs/String requests over localhost
and republishes Isaac Sim JSON acknowledgements.
"""

from __future__ import annotations

import argparse
import json
import socket

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class IsaacBatchSceneHelper(Node):
    def __init__(self, host: str, port: int, scene_control_topic: str, scene_ready_topic: str):
        super().__init__("isaac_batch_scene_helper")

        self._socket = socket.create_connection((host, port), timeout=10.0)
        self._socket.setblocking(False)
        self._recv_buffer = ""
        self._ready_pub = self.create_publisher(String, scene_ready_topic, 10)
        self._control_sub = self.create_subscription(
            String,
            scene_control_topic,
            self._scene_control_callback,
            10,
        )
        self._poll_timer = self.create_timer(0.05, self._poll_socket)

        self.get_logger().info(
            f"Connected to Isaac Sim batch scene bridge at {host}:{port}. "
            f"Listening on {scene_control_topic} and publishing {scene_ready_topic}."
        )

    def _scene_control_callback(self, msg: String) -> None:
        try:
            payload = json.loads(str(msg.data))
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Ignoring invalid batch scene JSON: {exc}")
            return

        payload["type"] = "scene_request"
        self._send_message(payload)

    def _poll_socket(self) -> None:
        try:
            chunk = self._socket.recv(65536)
        except BlockingIOError:
            return

        if not chunk:
            raise RuntimeError("Isaac Sim batch scene bridge disconnected.")

        self._recv_buffer += chunk.decode("utf-8")
        while "\n" in self._recv_buffer:
            line, self._recv_buffer = self._recv_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self._handle_message(line)

    def _handle_message(self, line: str) -> None:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Failed to decode Isaac Sim batch scene message: {exc}")
            return

        if payload.get("type") != "scene_ready":
            return

        msg = String()
        msg.data = json.dumps(payload)
        self._ready_pub.publish(msg)

    def _send_message(self, message: dict) -> None:
        payload = (json.dumps(message) + "\n").encode("utf-8")
        self._socket.sendall(payload)

    def destroy_node(self) -> bool:
        try:
            self._socket.close()
        except Exception:
            pass
        return super().destroy_node()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--scene-control-topic", required=True)
    parser.add_argument("--scene-ready-topic", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    rclpy.init()
    node: IsaacBatchSceneHelper | None = None
    try:
        node = IsaacBatchSceneHelper(
            host=args.host,
            port=args.port,
            scene_control_topic=args.scene_control_topic,
            scene_ready_topic=args.scene_ready_topic,
        )
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
