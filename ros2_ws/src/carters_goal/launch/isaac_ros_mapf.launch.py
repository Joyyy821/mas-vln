# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


############################################################################
#### Launch warehouse MAPF demo with nav2 path follower: ###################
# 1. Start Isaac Sim on host machine (source ros2 environment first to enable ros2 bridge)
# $ cd ~/isaac_sim
# $ ./python.sh /path/to/mas-vln/isaac_sim/scripts/build_stage_warehouse_carters.py
# 2. Launch caters nav2 example in Isaac ROS docker (source ros2 and ros2_ws)
# $ ros2 launch carters_nav2 warehouse_two_carters_nav2.launch.py
# 3. In another terminal, launch this MAPF demo (source ros2 and ros2_ws)
# $ ros2 launch carters_goal isaac_ros_mapf.launch.py run_plan_executor:=true
############################################################################


def generate_launch_description():
    carters_nav2_dir = get_package_share_directory("carters_nav2")
    carters_goal_dir = get_package_share_directory("carters_goal")

    default_map = os.path.join(
        carters_nav2_dir,
        "maps",
        "carter_warehouse_navigation_mapf.yaml",
    )
    mapf_params = os.path.join(carters_goal_dir, "config", "mapf_params_isaac.yaml")
    costmap_params = os.path.join(
        carters_goal_dir,
        "config",
        "mapf_costmap_params_isaac.yaml",
    )
    initial_pose_tf_params = os.path.join(
        carters_goal_dir,
        "config",
        "initial_pose_tf_params_isaac.yaml",
    )

    map_arg = DeclareLaunchArgument(
        "map",
        default_value=default_map,
        description="Full path to map yaml for mapf map_server",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock",
    )
    autostart_arg = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Autostart lifecycle nodes",
    )
    mapf_planner_arg = DeclareLaunchArgument(
        "mapf_planner",
        default_value="mapf_planner/CBSROS",
        description="MAPF planner plugin name",
    )
    run_goal_pub_arg = DeclareLaunchArgument(
        "run_goal_publisher",
        default_value="true",
        description="Run example MAPF goal publisher",
    )
    run_tf_bridge_arg = DeclareLaunchArgument(
        "run_tf_bridge",
        default_value="true",
        description="Bridge /robotX/tf into global /tf with prefixed frame ids",
    )
    run_initial_pose_tf_arg = DeclareLaunchArgument(
        "run_initial_pose_tf",
        default_value="true",
        description="Publish static map->robotX/odom transforms from Isaac spawn poses",
    )
    run_plan_executor_arg = DeclareLaunchArgument(
        "run_plan_executor",
        default_value="false",
        description="Execute sparse MAPF plans through Nav2 NavigateToPose servers",
    )

    map_file = LaunchConfiguration("map")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    mapf_planner = LaunchConfiguration("mapf_planner")
    run_goal_publisher = LaunchConfiguration("run_goal_publisher")
    run_tf_bridge = LaunchConfiguration("run_tf_bridge")
    run_initial_pose_tf = LaunchConfiguration("run_initial_pose_tf")
    run_plan_executor = LaunchConfiguration("run_plan_executor")

    lifecycle_nodes = ["map_server", "mapf_base_node"]

    mapf_group = GroupAction(
        [
            Node(
                namespace="mapf_base",
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[
                    mapf_params,
                    {
                        "yaml_filename": map_file,
                        "use_sim_time": use_sim_time,
                    },
                ],
            ),
            Node(
                namespace="mapf_base",
                package="mapf_base",
                executable="mapf_base_node",
                name="mapf_base_node",
                output="screen",
                parameters=[
                    costmap_params,
                    mapf_params,
                    {
                        "mapf_planner": mapf_planner,
                        "use_sim_time": use_sim_time,
                    },
                ],
            ),
            Node(
                namespace="mapf_base",
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_mapf",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "autostart": autostart,
                        "node_names": lifecycle_nodes,
                    }
                ],
            ),
            Node(
                namespace="mapf_base",
                package="mapf_base",
                executable="goal_transformer",
                name="goal_transformer",
                output="screen",
                parameters=[mapf_params, {"use_sim_time": use_sim_time}],
            ),
        ]
    )

    tf_bridge = Node(
        package="carters_goal",
        executable="NamespacedTfBridge",
        name="namespaced_tf_bridge",
        output="screen",
        condition=IfCondition(run_tf_bridge),
        parameters=[
            {
                "robot_namespaces": ["robot1", "robot2"],
            }
        ],
    )

    initial_pose_tf_publisher = Node(
        package="carters_goal",
        executable="InitialPoseTfPublisher",
        name="initial_pose_tf_publisher",
        output="screen",
        condition=IfCondition(run_initial_pose_tf),
        parameters=[initial_pose_tf_params, {"use_sim_time": use_sim_time}],
    )

    mapf_startup = TimerAction(
        period=1.0,
        actions=[mapf_group],
    )

    goal_publisher = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="carters_goal",
                executable="MapfGoalPublisher",
                name="mapf_goal_publisher",
                output="screen",
                condition=IfCondition(run_goal_publisher),
                parameters=[
                    {
                        "goal_topic": "/mapf_base/goal_for_each",
                        "goal_init_topic": "/mapf_base/goal_init_flag",
                        "global_plan_topic": "/mapf_base/global_plan",
                        "mapf_transition_topic": "/mapf_base/mapf_base_node/transition_event",
                        "frame_id": "map",
                        "publish_count": 0,
                        "wait_for_subscribers": True,
                        "wait_for_mapf_active": True,
                        "stop_on_plan": True,
                    }
                ],
            )
        ],
    )

    plan_executor = TimerAction(
        period=3.0,
        actions=[
            Node(
                namespace="mapf_base",
                package="carters_goal",
                executable="MapfNav2Executor",
                name="plan_executor",
                output="screen",
                condition=IfCondition(run_plan_executor),
                parameters=[mapf_params, {"use_sim_time": use_sim_time}],
            )
        ],
    )

    return LaunchDescription(
        [
            map_arg,
            use_sim_time_arg,
            autostart_arg,
            mapf_planner_arg,
            run_goal_pub_arg,
            run_tf_bridge_arg,
            run_initial_pose_tf_arg,
            run_plan_executor_arg,
            tf_bridge,
            initial_pose_tf_publisher,
            mapf_startup,
            goal_publisher,
            plan_executor,
        ]
    )
