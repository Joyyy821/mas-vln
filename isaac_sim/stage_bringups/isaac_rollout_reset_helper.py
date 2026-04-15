#!/usr/bin/env python3
"""
ROS 2 Humble helper that bridges rollout reset topics to a localhost socket.

This script is intentionally executed with the system Python outside Isaac Sim,
so it can import rclpy built for ROS 2 Humble's Python 3.10 runtime.
"""

from __future__ import annotations

import argparse
import json
import socket

import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Int32


def _parse_rollout_id(frame_id: str) -> int | None:
    prefix = "rollout:"
    if not frame_id.startswith(prefix):
        return None
    try:
        return int(frame_id[len(prefix) :])
    except ValueError:
        return None


def _pose_array_msg_to_flat_list(msg: PoseArray) -> list[float]:
    flat_pose_array: list[float] = []
    for pose in msg.poses:
        flat_pose_array.extend(
            [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ]
        )
    return flat_pose_array


class IsaacRolloutResetHelper(Node):
    def __init__(self, host: str, port: int, rollout_control_topic: str, rollout_reset_done_topic: str):
        super().__init__("isaac_rollout_reset_helper")

        self._socket = socket.create_connection((host, port), timeout=10.0)
        self._socket.setblocking(False)
        self._recv_buffer = ""
        self._reset_done_pub = self.create_publisher(Int32, rollout_reset_done_topic, 10)
        self._control_sub = self.create_subscription(
            PoseArray,
            rollout_control_topic,
            self._rollout_control_callback,
            10,
        )
        self._poll_timer = self.create_timer(0.05, self._poll_socket)

        self.get_logger().info(
            f"Connected to Isaac Sim rollout bridge at {host}:{port}. "
            f"Listening on {rollout_control_topic} and publishing {rollout_reset_done_topic}."
        )

    def _rollout_control_callback(self, msg: PoseArray) -> None:
        rollout_id = _parse_rollout_id(msg.header.frame_id)
        if rollout_id is None or rollout_id <= 0:
            return

        message = {
            "type": "reset",
            "rollout_id": rollout_id,
            "poses": _pose_array_msg_to_flat_list(msg),
        }
        self._send_message(message)

    def _poll_socket(self) -> None:
        try:
            chunk = self._socket.recv(65536)
        except BlockingIOError:
            return

        if not chunk:
            raise RuntimeError("Isaac Sim rollout bridge disconnected.")

        self._recv_buffer += chunk.decode("utf-8")
        while "\n" in self._recv_buffer:
            line, self._recv_buffer = self._recv_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self._handle_message(line)

    def _handle_message(self, line: str) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Failed to decode Isaac Sim rollout bridge message: {exc}")
            return

        if message.get("type") != "reset_done":
            return

        rollout_id = int(message.get("rollout_id", -1))
        if rollout_id <= 0:
            return

        ack_msg = Int32()
        ack_msg.data = rollout_id
        self._reset_done_pub.publish(ack_msg)

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
    parser.add_argument("--rollout-control-topic", required=True)
    parser.add_argument("--rollout-reset-done-topic", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    rclpy.init()
    node: IsaacRolloutResetHelper | None = None
    try:
        node = IsaacRolloutResetHelper(
            host=args.host,
            port=args.port,
            rollout_control_topic=args.rollout_control_topic,
            rollout_reset_done_topic=args.rollout_reset_done_topic,
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
