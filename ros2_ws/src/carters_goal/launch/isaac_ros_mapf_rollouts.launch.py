# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    carters_goal_dir = get_package_share_directory("carters_goal")
    carters_nav2_dir = get_package_share_directory("carters_nav2")

    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(carters_goal_dir, "launch", "isaac_ros_mapf.launch.py")
        ),
        launch_arguments={
            "map": LaunchConfiguration("map"),
            "team_config_file": LaunchConfiguration("team_config_file"),
            "mapf_params_file": LaunchConfiguration("mapf_params_file"),
            "mapf_costmap_params_file": LaunchConfiguration("mapf_costmap_params_file"),
            "initial_pose_tf_params_file": LaunchConfiguration("initial_pose_tf_params_file"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "autostart": LaunchConfiguration("autostart"),
            "mapf_planner": LaunchConfiguration("mapf_planner"),
            "record_velocity": LaunchConfiguration("record_velocity"),
            "record_frequency_hz": LaunchConfiguration("record_frequency_hz"),
            "record_odom_topic_suffix": LaunchConfiguration("record_odom_topic_suffix"),
            "experiments_dir": LaunchConfiguration("experiments_dir"),
            "core_startup_delay": LaunchConfiguration("core_startup_delay"),
            "lifecycle_manager_delay": LaunchConfiguration("lifecycle_manager_delay"),
            "run_goal_publisher": "false",
            "goal_publisher_delay": LaunchConfiguration("goal_publisher_delay"),
            "run_tf_bridge": LaunchConfiguration("run_tf_bridge"),
            "run_initial_pose_tf": LaunchConfiguration("run_initial_pose_tf"),
            "run_plan_executor": LaunchConfiguration("run_plan_executor"),
            "execution_backend": LaunchConfiguration("execution_backend"),
            "plan_executor_delay": LaunchConfiguration("plan_executor_delay"),
            "rollout_control_topic": LaunchConfiguration("rollout_control_topic"),
            "rollout_reset_done_topic": LaunchConfiguration("rollout_reset_done_topic"),
            "execution_status_topic": LaunchConfiguration("execution_status_topic"),
        }.items(),
    )

    rollout_manager = Node(
        package="carters_goal",
        executable="RolloutManager",
        name="rollout_manager",
        output="screen",
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "team_config_file": LaunchConfiguration("team_config_file"),
                "experiments_dir": LaunchConfiguration("experiments_dir"),
                "rollout_control_topic": LaunchConfiguration("rollout_control_topic"),
                "rollout_reset_done_topic": LaunchConfiguration("rollout_reset_done_topic"),
                "execution_status_topic": LaunchConfiguration("execution_status_topic"),
                "skip_existed_rollout": LaunchConfiguration("skip_existed_rollout"),
                "reset_timeout_sec": LaunchConfiguration("reset_timeout_sec"),
                "execution_timeout_sec": LaunchConfiguration("execution_timeout_sec"),
                "post_rollout_delay_sec": LaunchConfiguration("post_rollout_delay_sec"),
                "goal_publish_period_sec": LaunchConfiguration("goal_publish_period_sec"),
                "wait_for_mapf_active": LaunchConfiguration("wait_for_mapf_active"),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value="",
                description="Optional full path to the map yaml for MAPF.",
            ),
            DeclareLaunchArgument(
                "team_config_file",
                default_value=os.path.join(
                    carters_nav2_dir,
                    "config",
                    "warehouse",
                    "warehouse_forklift.yaml",
                ),
                description="Full path to the rollout configuration YAML.",
            ),
            DeclareLaunchArgument(
                "mapf_params_file",
                default_value=os.path.join(carters_goal_dir, "config", "mapf_params_isaac.yaml"),
                description="Base MAPF params YAML.",
            ),
            DeclareLaunchArgument(
                "mapf_costmap_params_file",
                default_value=os.path.join(
                    carters_goal_dir, "config", "mapf_costmap_params_isaac.yaml"
                ),
                description="Base MAPF costmap params YAML.",
            ),
            DeclareLaunchArgument(
                "initial_pose_tf_params_file",
                default_value=os.path.join(
                    carters_goal_dir, "config", "initial_pose_tf_params_isaac.yaml"
                ),
                description="Base initial pose TF params YAML.",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("mapf_planner", default_value="mapf_planner/CBSROS"),
            DeclareLaunchArgument("record_velocity", default_value="true"),
            DeclareLaunchArgument("record_frequency_hz", default_value="20.0"),
            DeclareLaunchArgument("record_odom_topic_suffix", default_value="chassis/odom"),
            DeclareLaunchArgument("experiments_dir", default_value=""),
            DeclareLaunchArgument("core_startup_delay", default_value="1.0"),
            DeclareLaunchArgument("lifecycle_manager_delay", default_value="4.0"),
            DeclareLaunchArgument("goal_publisher_delay", default_value="2.0"),
            DeclareLaunchArgument("run_tf_bridge", default_value="true"),
            DeclareLaunchArgument("run_initial_pose_tf", default_value="true"),
            DeclareLaunchArgument("run_plan_executor", default_value="true"),
            DeclareLaunchArgument("execution_backend", default_value="tracker"),
            DeclareLaunchArgument("plan_executor_delay", default_value="1.5"),
            DeclareLaunchArgument(
                "rollout_control_topic",
                default_value="/carters_goal/rollout_control",
                description="PoseArray control topic for rollout resets and recorder/log switching.",
            ),
            DeclareLaunchArgument(
                "rollout_reset_done_topic",
                default_value="/carters_goal/rollout_reset_done",
                description="Int32 acknowledgement topic published by Isaac Sim after a reset.",
            ),
            DeclareLaunchArgument(
                "execution_status_topic",
                default_value="/mapf_base/plan_execution_status",
                description="Status topic published by the active MAPF executor.",
            ),
            DeclareLaunchArgument(
                "skip_existed_rollout",
                default_value="false",
                description="Skip rollouts whose output directory already exists.",
            ),
            DeclareLaunchArgument(
                "goal_publish_period_sec",
                default_value="0.5",
                description="How often to re-publish rollout goals until planning starts.",
            ),
            DeclareLaunchArgument(
                "reset_timeout_sec",
                default_value="15.0",
                description="How long to wait for Isaac Sim to acknowledge a rollout reset.",
            ),
            DeclareLaunchArgument(
                "execution_timeout_sec",
                default_value="300.0",
                description="How long to wait for one rollout to finish execution.",
            ),
            DeclareLaunchArgument(
                "post_rollout_delay_sec",
                default_value="1.0",
                description="Pause between finished rollout k and rollout k+1.",
            ),
            DeclareLaunchArgument(
                "wait_for_mapf_active",
                default_value="true",
                description="Wait for the lifecycle-managed MAPF node to become active before starting.",
            ),
            base_launch,
            rollout_manager,
        ]
    )
