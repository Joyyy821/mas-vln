# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    carters_goal_dir = get_package_share_directory("carters_goal")

    batch_runner = Node(
        package="carters_goal",
        executable="RandomizedWarehouseBatchRunner",
        name="randomized_warehouse_batch_runner",
        output="screen",
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "scene_root_dir": LaunchConfiguration("scene_root_dir"),
                "continue": LaunchConfiguration("continue"),
                "continue_scene_id": LaunchConfiguration("continue_scene_id"),
                "continue_rollout_id": LaunchConfiguration("continue_rollout_id"),
                "max_rerun": LaunchConfiguration("max_rerun"),
                "scene_control_topic": LaunchConfiguration("scene_control_topic"),
                "scene_ready_topic": LaunchConfiguration("scene_ready_topic"),
                "scene_ready_timeout_sec": LaunchConfiguration("scene_ready_timeout_sec"),
                "execution_timeout_sec": LaunchConfiguration("execution_timeout_sec"),
                "execution_start_timeout_sec": LaunchConfiguration("execution_start_timeout_sec"),
                "retry_cooldown_sec": LaunchConfiguration("retry_cooldown_sec"),
                "status_topic": LaunchConfiguration("execution_status_topic"),
                "mapf_params_file": LaunchConfiguration("mapf_params_file"),
                "mapf_costmap_params_file": LaunchConfiguration("mapf_costmap_params_file"),
                "initial_pose_tf_params_file": LaunchConfiguration("initial_pose_tf_params_file"),
                "autostart": LaunchConfiguration("autostart"),
                "mapf_planner": LaunchConfiguration("mapf_planner"),
                "record_velocity": LaunchConfiguration("record_velocity"),
                "record_frequency_hz": LaunchConfiguration("record_frequency_hz"),
                "record_odom_topic_suffix": LaunchConfiguration("record_odom_topic_suffix"),
                "record_cmd_vel_topic_suffix": LaunchConfiguration("record_cmd_vel_topic_suffix"),
                "experiments_dir": LaunchConfiguration("experiments_dir"),
                "core_startup_delay": LaunchConfiguration("core_startup_delay"),
                "lifecycle_manager_delay": LaunchConfiguration("lifecycle_manager_delay"),
                "goal_publisher_delay": LaunchConfiguration("goal_publisher_delay"),
                "run_tf_bridge": LaunchConfiguration("run_tf_bridge"),
                "run_initial_pose_tf": LaunchConfiguration("run_initial_pose_tf"),
                "run_plan_executor": LaunchConfiguration("run_plan_executor"),
                "execution_backend": LaunchConfiguration("execution_backend"),
                "plan_executor_delay": LaunchConfiguration("plan_executor_delay"),
            }
        ],
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4"),
            SetEnvironmentVariable("RMW_FASTRTPS_USE_SHM", "0"),
            DeclareLaunchArgument(
                "scene_root_dir",
                default_value="",
                description=(
                    "Directory containing scene_<n> randomized warehouse bundles. "
                    "Each bundle must contain scene.usd, mapf_map.yaml, and team_config.yaml."
                ),
            ),
            DeclareLaunchArgument(
                "continue",
                default_value="false",
                description="Continue from continue_scene_id / continue_rollout_id.",
            ),
            DeclareLaunchArgument(
                "continue_scene_id",
                default_value="",
                description="Scene id such as scene_11 when continue:=true.",
            ),
            DeclareLaunchArgument(
                "continue_rollout_id",
                default_value="0",
                description="Rollout id within continue_scene_id when continue:=true.",
            ),
            DeclareLaunchArgument(
                "max_rerun",
                default_value="2",
                description="Number of retries after the first failed attempt for each rollout.",
            ),
            DeclareLaunchArgument(
                "scene_control_topic",
                default_value="/carters_goal/batch_scene_control",
                description="JSON String topic used to ask Isaac Sim to load/spawn a rollout.",
            ),
            DeclareLaunchArgument(
                "scene_ready_topic",
                default_value="/carters_goal/batch_scene_ready",
                description="JSON String acknowledgement topic from the Isaac Sim batch server.",
            ),
            DeclareLaunchArgument(
                "scene_ready_timeout_sec",
                default_value="60.0",
                description="How long to wait for Isaac Sim to acknowledge scene/robot readiness.",
            ),
            DeclareLaunchArgument(
                "execution_timeout_sec",
                default_value="900.0",
                description="Hard wall-clock timeout for one child single-rollout launch attempt.",
            ),
            DeclareLaunchArgument(
                "execution_start_timeout_sec",
                default_value="300.0",
                description=(
                    "Hard wall-clock timeout for a child launch to publish any execution status. "
                    "This catches lifecycle bringup failures before the full execution timeout."
                ),
            ),
            DeclareLaunchArgument(
                "retry_cooldown_sec",
                default_value="2.0",
                description="Seconds to pause after terminating one child launch attempt.",
            ),
            DeclareLaunchArgument(
                "mapf_params_file",
                default_value=os.path.join(carters_goal_dir, "config", "mapf_params_isaac.yaml"),
                description="Base MAPF params YAML.",
            ),
            DeclareLaunchArgument(
                "mapf_costmap_params_file",
                default_value=os.path.join(
                    carters_goal_dir,
                    "config",
                    "mapf_costmap_params_isaac.yaml",
                ),
                description="Base MAPF costmap params YAML.",
            ),
            DeclareLaunchArgument(
                "initial_pose_tf_params_file",
                default_value=os.path.join(
                    carters_goal_dir,
                    "config",
                    "initial_pose_tf_params_isaac.yaml",
                ),
                description="Base initial pose TF params YAML.",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("mapf_planner", default_value="mapf_planner/CBSROS"),
            DeclareLaunchArgument("record_velocity", default_value="true"),
            DeclareLaunchArgument("record_frequency_hz", default_value="20.0"),
            DeclareLaunchArgument("record_odom_topic_suffix", default_value="chassis/odom"),
            DeclareLaunchArgument("record_cmd_vel_topic_suffix", default_value="cmd_vel"),
            DeclareLaunchArgument("experiments_dir", default_value=""),
            DeclareLaunchArgument("core_startup_delay", default_value="1.0"),
            DeclareLaunchArgument("lifecycle_manager_delay", default_value="4.0"),
            DeclareLaunchArgument("goal_publisher_delay", default_value="2.0"),
            DeclareLaunchArgument("run_tf_bridge", default_value="true"),
            DeclareLaunchArgument("run_initial_pose_tf", default_value="true"),
            DeclareLaunchArgument("run_plan_executor", default_value="true"),
            DeclareLaunchArgument("execution_backend", default_value="timed_tracker"),
            DeclareLaunchArgument("plan_executor_delay", default_value="1.5"),
            DeclareLaunchArgument(
                "execution_status_topic",
                default_value="/mapf_base/plan_execution_status",
                description="Status topic published by the active MAPF executor.",
            ),
            batch_runner,
        ]
    )
