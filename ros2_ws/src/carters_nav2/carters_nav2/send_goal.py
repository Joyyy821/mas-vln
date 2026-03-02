#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler



class Nav2GoalSender(Node):
    def __init__(self):
        super().__init__("nav2_goal_sender")
        self.declare_parameter("robot_ns", "robot1")
        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("yaw", 0.0)  # radians
        self.declare_parameter("frame_id", "map")

        self.robot_ns = self.get_parameter("robot_ns").value
        self.action_name = f"/{self.robot_ns}/navigate_to_pose"
        self.client = ActionClient(self, NavigateToPose, self.action_name)

    def send(self):
        x = float(self.get_parameter("x").value)
        y = float(self.get_parameter("y").value)
        yaw = float(self.get_parameter("yaw").value)
        frame_id = str(self.get_parameter("frame_id").value)

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = frame_id
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        self.get_logger().info(f"Waiting for action server: {self.action_name} ...")
        if not self.client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Nav2 action server not available: {self.action_name}")

        self.get_logger().info(f"Sending goal to {self.action_name}: x={x}, y={y}, yaw={yaw}")
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return 1

        self.get_logger().info("Goal accepted. Waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        status = result_future.result().status
        self.get_logger().info(f"Result status: {status}")
        return 0


def main():
    rclpy.init()
    node = Nav2GoalSender()
    try:
        rc = node.send()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()