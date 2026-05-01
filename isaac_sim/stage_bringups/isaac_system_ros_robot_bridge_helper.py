#!/usr/bin/env python3
"""
ROS 2 Humble helper for heterogeneous Isaac Sim robot topics.

This script is intentionally executed with system Python outside Isaac Sim so
it can use the ROS 2 Humble rclpy packages from /opt/ros/humble.
"""

from __future__ import annotations

import argparse
import json
import math
import socket

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


def _sim_time_to_stamp(sim_time_sec: float):
    from builtin_interfaces.msg import Time

    stamp = Time()
    sec = int(math.floor(float(sim_time_sec)))
    nanosec = int(round((float(sim_time_sec) - sec) * 1_000_000_000.0))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    stamp.sec = sec
    stamp.nanosec = nanosec
    return stamp


class IsaacSystemRosRobotBridgeHelper(Node):
    def __init__(self, host: str, port: int, robot_namespaces: list[str]):
        super().__init__("isaac_system_ros_robot_bridge_helper")

        if not robot_namespaces:
            raise ValueError("At least one robot namespace is required.")

        self._socket = socket.create_connection((host, port), timeout=10.0)
        self._socket.setblocking(False)
        self._recv_buffer = ""

        tf_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )

        self._cmd_subs = []
        self._odom_pubs: dict[str, object] = {}
        self._tf_pubs: dict[str, object] = {}
        for namespace in robot_namespaces:
            clean_namespace = namespace.strip("/")
            self._cmd_subs.append(
                self.create_subscription(
                    Twist,
                    f"/{clean_namespace}/cmd_vel",
                    lambda msg, robot_namespace=clean_namespace: self._cmd_vel_callback(
                        robot_namespace,
                        msg,
                    ),
                    10,
                )
            )
            self._odom_pubs[clean_namespace] = self.create_publisher(
                Odometry,
                f"/{clean_namespace}/chassis/odom",
                20,
            )
            self._tf_pubs[clean_namespace] = self.create_publisher(
                TFMessage,
                f"/{clean_namespace}/tf",
                tf_qos,
            )

        self._poll_timer = self.create_timer(0.01, self._poll_socket)
        self.get_logger().info(
            f"Connected to Isaac Sim robot bridge at {host}:{port}. "
            f"Namespaces: {robot_namespaces}"
        )

    def _cmd_vel_callback(self, namespace: str, msg: Twist) -> None:
        message = {
            "type": "cmd_vel",
            "namespace": namespace,
            "linear_x": float(msg.linear.x),
            "angular_z": float(msg.angular.z),
        }
        self._send_message(message)

    def _poll_socket(self) -> None:
        try:
            chunk = self._socket.recv(65536)
        except BlockingIOError:
            return

        if not chunk:
            raise RuntimeError("Isaac Sim robot bridge disconnected.")

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
            self.get_logger().warn(f"Failed to decode Isaac Sim robot bridge message: {exc}")
            return

        if message.get("type") != "robot_state":
            return

        namespace = str(message.get("namespace", "")).strip("/")
        if namespace not in self._odom_pubs:
            return

        sim_time_sec = float(message.get("sim_time_sec", 0.0))
        stamp = _sim_time_to_stamp(sim_time_sec)
        position = [float(value) for value in message.get("position", [0.0, 0.0, 0.0])[:3]]
        orientation = [
            float(value)
            for value in message.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])[:4]
        ]
        linear = [float(value) for value in message.get("linear", [0.0, 0.0, 0.0])[:3]]
        angular_z = float(message.get("angular_z", 0.0))
        odom_frame_id = str(message.get("odom_frame_id") or "odom")
        base_frame_id = str(message.get("base_frame_id") or "base_link")

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = odom_frame_id
        odom_msg.child_frame_id = base_frame_id
        odom_msg.pose.pose.position.x = position[0]
        odom_msg.pose.pose.position.y = position[1]
        odom_msg.pose.pose.position.z = position[2]
        odom_msg.pose.pose.orientation.x = orientation[0]
        odom_msg.pose.pose.orientation.y = orientation[1]
        odom_msg.pose.pose.orientation.z = orientation[2]
        odom_msg.pose.pose.orientation.w = orientation[3]
        odom_msg.twist.twist.linear.x = linear[0]
        odom_msg.twist.twist.linear.y = linear[1]
        odom_msg.twist.twist.linear.z = linear[2]
        odom_msg.twist.twist.angular.z = angular_z
        self._odom_pubs[namespace].publish(odom_msg)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = odom_frame_id
        transform.child_frame_id = base_frame_id
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation.x = orientation[0]
        transform.transform.rotation.y = orientation[1]
        transform.transform.rotation.z = orientation[2]
        transform.transform.rotation.w = orientation[3]
        self._tf_pubs[namespace].publish(TFMessage(transforms=[transform]))

    def _send_message(self, message: dict) -> None:
        payload = (json.dumps(message) + "\n").encode("utf-8")
        try:
            self._socket.sendall(payload)
        except BlockingIOError:
            return

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
    parser.add_argument("--robot-namespace", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    rclpy.init()
    node: IsaacSystemRosRobotBridgeHelper | None = None
    try:
        node = IsaacSystemRosRobotBridgeHelper(
            host=args.host,
            port=args.port,
            robot_namespaces=args.robot_namespace,
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
