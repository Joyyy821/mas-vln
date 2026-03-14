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

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    map_yaml_file = LaunchConfiguration(
        "map_yaml_path",
        default=os.path.join(
            get_package_share_directory("carters_nav2"), "maps", "carter_warehouse_navigation.yaml"
        ),
    )

    # goal_text_file = LaunchConfiguration(
    #     "goal_text_file_path",
    #     default=os.path.join(get_package_share_directory("carters_nav2"), "assets", "goals.txt"),
    # )

    navigation_goal_node = Node(
        name="set_navigation_goal",
        package="carters_goal",
        executable="SetNavigationGoal",
        namespace="robot1",
        parameters=[
            {
                "map_yaml_path": map_yaml_file,
                "action_server_name": "navigate_to_pose",
                "initial_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.02],
            }
        ],
        output="screen",
    )

    navigation_goal_node_2 = Node(
        name="set_navigation_goal",
        package="carters_goal",
        executable="SetNavigationGoal",
        namespace="robot2",
        parameters=[
            {
                "map_yaml_path": map_yaml_file,
                "action_server_name": "navigate_to_pose",
                "initial_pose": [4.0, -1.0, 0.0, 0.0, 0.0, 0.99, 0.02],
            }
        ],
        output="screen",
    )

    # return LaunchDescription([navigation_goal_node])
    return LaunchDescription([
        navigation_goal_node,
        navigation_goal_node_2
    ])
