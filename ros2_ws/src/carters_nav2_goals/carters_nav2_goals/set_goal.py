# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
import sys
from geometry_msgs.msg import PoseWithCovarianceStamped
import time


class SetNavigationGoal(Node):
    def __init__(self):
        super().__init__("set_navigation_goal")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("action_server_name", "navigate_to_pose"),
                ("frame_id", "map"),
                ("map_yaml_path", rclpy.Parameter.Type.STRING),
                ("initial_pose", rclpy.Parameter.Type.DOUBLE_ARRAY),
            ],
        )

        action_server_name = self.get_parameter("action_server_name").value
        self._action_client = ActionClient(self, NavigateToPose, action_server_name)

        self.__initial_goal_publisher = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)

        self.__initial_pose = self.get_parameter("initial_pose").value
        self.__is_initial_pose_sent = True if self.__initial_pose is None else False

    def __send_initial_pose(self):
        """
        Publishes the initial pose.
        This function is only called once that too before sending any goal pose
        to the mission server.
        """
        goal = PoseWithCovarianceStamped()
        goal.header.frame_id = self.get_parameter("frame_id").value
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = self.__initial_pose[0]
        goal.pose.pose.position.y = self.__initial_pose[1]
        goal.pose.pose.position.z = self.__initial_pose[2]
        goal.pose.pose.orientation.x = self.__initial_pose[3]
        goal.pose.pose.orientation.y = self.__initial_pose[4]
        goal.pose.pose.orientation.z = self.__initial_pose[5]
        goal.pose.pose.orientation.w = self.__initial_pose[6]
        self.__initial_goal_publisher.publish(goal)

    def send_goal(self, goal_array=None):
        """
        Sends the goal to the action server.
        """

        if not self.__is_initial_pose_sent:
            self.get_logger().info("Sending initial pose")
            self.__send_initial_pose()
            self.__is_initial_pose_sent = True

            # Assumption is that initial pose is set after publishing first time in this duration.
            # Can be changed to more sophisticated way. e.g. /particlecloud topic has no msg until
            # the initial pose is set.
            time.sleep(10)
            self.get_logger().info("Sending first goal")

        self._action_client.wait_for_server()
        if goal_array is None:
            rclpy.shutdown()
            sys.exit(1)

        goal_msg = self.__format_goal_msg(goal_array)

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.__feedback_callback
        )
        self._send_goal_future.add_done_callback(self.__goal_response_callback)

    def __goal_response_callback(self, future):
        """
        Callback function to check the response(goal accpted/rejected) from the server.\n
        If the Goal is rejected it stops the execution for now.(We can change to resample the pose if rejected.)
        """

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected :(")
            rclpy.shutdown()
            return

        self.get_logger().info("Goal accepted :)")

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.__get_result_callback)

    def __format_goal_msg(self, pose):
        """
        Format the goal message from the provided goal array (pose).

        Parameters
        ----------
        pose : list
            List containing the goal coordinates and orientation.

        Returns
        -------
        NavigateToPose.Goal
            Formatted goal message.

        [NavigateToPose][goal] or None if the next goal couldn't be generated.

        """

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.get_parameter("frame_id").value
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        self.get_logger().info("Goal pose: {0}".format(pose))
        goal_msg.pose.pose.position.x = pose[0]
        goal_msg.pose.pose.position.y = pose[1]
        goal_msg.pose.pose.position.z = pose[2]
        goal_msg.pose.pose.orientation.x = pose[3]
        goal_msg.pose.pose.orientation.y = pose[4]
        goal_msg.pose.pose.orientation.z = pose[5]
        goal_msg.pose.pose.orientation.w = pose[6]
        return goal_msg
    
    def __get_result_callback(self, future):
        """
        Callback to check result.\n
        It calls the send_goal() function in case current goal sent count < required goals count.     
        """
        # Nav2 is sending empty message for success as well as for failure.
        result = future.result().result
        self.get_logger().info("Result: {0}".format(result))

        rclpy.shutdown()

    def __feedback_callback(self, feedback_msg):
        """
        This is feeback callback. We can compare/compute/log while the robot is on its way to goal.
        """
        # self.get_logger().info('FEEDBACK: {}\n'.format(feedback_msg))
        pass


def main():
    # Examples from goals.txt
    pose1 = [1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    pose2 = [2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    pose3 = [3.4, 4.5, 0.0, 0.5, 0.5, 0.5, 0.5]

    rclpy.init()
    set_goal = SetNavigationGoal()

    # simple test: send different goal to different robot (identify by namespace)
    ns = set_goal.get_namespace()
    set_goal.get_logger().info(f"Running in namespace: {ns}")
    if ns == "/robot1":
        goal_pose = pose1
    elif ns == "/robot2":
        goal_pose = pose2
    
    # send goal
    result = set_goal.send_goal(goal_pose)
    rclpy.spin(set_goal)
    # try:
    #     result = set_goal.send_goal(goal_pose)
    #     rclpy.spin(set_goal)
    # except KeyboardInterrupt:
    #     set_goal.get_logger().info('Interrupted by user.')
    # finally:
    #     set_goal.get_logger().info('Exiting...')
    #     set_goal.destroy_node()
    #     rclpy.shutdown()


if __name__ == "__main__":
    main()
