# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_team_config_utils():
    helper_path = os.path.join(os.path.dirname(__file__), "team_config_utils.py")
    spec = importlib.util.spec_from_file_location("team_config_utils", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load team config utilities from {helper_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


team_config_utils = _load_team_config_utils()


def _launch_setup(context, *args, **kwargs):
    carter_nav2_bringup_dir = get_package_share_directory("carters_nav2")
    nav2_bringup_launch_dir = os.path.join(
        get_package_share_directory("nav2_bringup"),
        "launch",
    )

    maps_dir = os.path.join(carter_nav2_bringup_dir, "maps")
    team_config_path = LaunchConfiguration("team_config_file").perform(context)
    template_path = LaunchConfiguration("nav2_params_template_file").perform(context)
    map_override = LaunchConfiguration("map").perform(context)

    team_config = team_config_utils.load_team_config(team_config_path, maps_dir=maps_dir)
    map_yaml_file = map_override or team_config["nav2_map"]
    if not map_yaml_file:
        raise RuntimeError(
            "No Nav2 map was provided. Set environment.nav2_map in the team config "
            "or pass map:=/abs/path/to/map.yaml."
        )

    default_bt_xml_filename = LaunchConfiguration("default_bt_xml_filename")
    autostart = LaunchConfiguration("autostart")
    rviz_config_file = LaunchConfiguration("rviz_config")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_respawn = LaunchConfiguration("use_respawn")

    nav2_start_delay = float(LaunchConfiguration("nav2_start_delay").perform(context))
    rviz_start_delay = float(LaunchConfiguration("rviz_start_delay").perform(context))
    startup_spacing = float(LaunchConfiguration("startup_spacing").perform(context))

    nav_instances_cmds = []
    nav_instances_cmds.append(
        LogInfo(
            msg=(
                "Launching Nav2 for "
                f"{team_config['agent_num']} robots from {team_config_path}. "
                "Startup is staggered to reduce lifecycle and discovery races."
            )
        )
    )

    if team_config["agent_num"] != 2:
        nav_instances_cmds.append(
            LogInfo(
                msg=(
                    "[warehouse_two_carters_nav2] Team config contains "
                    f"{team_config['agent_num']} robots. The launch will start all of them."
                )
            )
        )

    for index, robot in enumerate(team_config["robots"]):
        params = team_config_utils.render_nav2_params(
            template_path=template_path,
            robot_namespace=robot["name"],
            initial_pose=robot["initial_pose"],
        )
        params_file = team_config_utils.write_temp_yaml(
            prefix=f"carters_nav2_{robot['name']}_",
            data=params,
        )

        robot_start_offset = startup_spacing * index

        nav_instances_cmds.append(
            GroupAction(
                [
                    TimerAction(
                        period=robot_start_offset + rviz_start_delay,
                        condition=IfCondition(use_rviz),
                        actions=[
                            LogInfo(
                                msg=(
                                    f"Launching RViz for {robot['name']} after "
                                    f"{robot_start_offset + rviz_start_delay:.1f}s."
                                )
                            ),
                            IncludeLaunchDescription(
                                PythonLaunchDescriptionSource(
                                    os.path.join(nav2_bringup_launch_dir, "rviz_launch.py")
                                ),
                                launch_arguments={
                                    "namespace": robot["name"],
                                    "use_namespace": "True",
                                    "rviz_config": rviz_config_file,
                                }.items(),
                            )
                        ],
                    ),
                    TimerAction(
                        period=robot_start_offset + nav2_start_delay,
                        actions=[
                            IncludeLaunchDescription(
                                PythonLaunchDescriptionSource(
                                    os.path.join(
                                        carter_nav2_bringup_dir,
                                        "launch",
                                        "carter_navigation_individual.launch.py",
                                    )
                                ),
                                launch_arguments={
                                    "namespace": robot["name"],
                                    "use_namespace": "True",
                                    "map": map_yaml_file,
                                    "use_sim_time": use_sim_time,
                                    "use_composition": "False",
                                    "use_respawn": use_respawn,
                                    "params_file": params_file,
                                    "default_bt_xml_filename": default_bt_xml_filename,
                                    "autostart": autostart,
                                    "use_rviz": "False",
                                    "use_simulator": "False",
                                    "headless": "False",
                                }.items(),
                            )
                        ],
                    ),
                    TimerAction(
                        period=robot_start_offset,
                        actions=[
                            LogInfo(
                                msg=(
                                    f"Starting {robot['name']} pointcloud->scan bridge now, "
                                    f"Nav2 after {nav2_start_delay:.1f}s, RViz after {rviz_start_delay:.1f}s."
                                )
                            ),
                            Node(
                                package="pointcloud_to_laserscan",
                                executable="pointcloud_to_laserscan_node",
                                remappings=[
                                    ("cloud_in", ["front_3d_lidar/lidar_points"]),
                                    ("scan", ["scan"]),
                                ],
                                parameters=[
                                    {
                                        "target_frame": "front_3d_lidar",
                                        "transform_tolerance": 0.01,
                                        "min_height": -0.4,
                                        "max_height": 1.5,
                                        "angle_min": -1.5708,
                                        "angle_max": 1.5708,
                                        "angle_increment": 0.0087,
                                        "scan_time": 0.3333,
                                        "range_min": 0.05,
                                        "range_max": 100.0,
                                        "use_inf": True,
                                        "inf_epsilon": 1.0,
                                    }
                                ],
                                name="pointcloud_to_laserscan",
                                namespace=robot["name"],
                                respawn=True,
                                respawn_delay=2.0,
                            ),
                        ],
                    ),
                ]
            )
        )

    return nav_instances_cmds


def generate_launch_description():
    carter_nav2_bringup_dir = get_package_share_directory("carters_nav2")
    rviz_config_dir = os.path.join(
        carter_nav2_bringup_dir,
        "rviz2",
        "carter_navigation_namespaced.rviz",
    )

    declare_team_config_file_cmd = DeclareLaunchArgument(
        "team_config_file",
        default_value=os.path.join(
            carter_nav2_bringup_dir,
            "config",
            "warehouse",
            "warehouse_team_config.yaml",
        ),
        description="Full path to the shared robot team configuration YAML.",
    )

    declare_params_template_file_cmd = DeclareLaunchArgument(
        "nav2_params_template_file",
        default_value=os.path.join(
            carter_nav2_bringup_dir,
            "config",
            "warehouse",
            "multi_robot_carter_navigation_params_1.yaml",
        ),
        description=(
            "Base Nav2 params YAML used as a per-robot template. "
            "The launch file fills in robot namespace and initial pose differences."
        ),
    )

    declare_map_yaml_cmd = DeclareLaunchArgument(
        "map",
        default_value="",
        description=(
            "Optional full path to the Nav2 map yaml. "
            "If empty, the value from the team config is used."
        ),
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="True",
        description="Use Isaac Sim clock if true.",
    )

    declare_bt_xml_cmd = DeclareLaunchArgument(
        "default_bt_xml_filename",
        default_value=os.path.join(
            get_package_share_directory("nav2_bt_navigator"),
            "behavior_trees",
            "navigate_w_replanning_and_recovery.xml",
        ),
        description="Full path to the behavior tree xml file to use.",
    )

    declare_autostart_cmd = DeclareLaunchArgument(
        "autostart",
        default_value="True",
        description="Automatically startup the stacks.",
    )

    declare_use_respawn_cmd = DeclareLaunchArgument(
        "use_respawn",
        default_value="True",
        description="Respawn Nav2 servers if one exits unexpectedly.",
    )

    declare_nav2_start_delay_cmd = DeclareLaunchArgument(
        "nav2_start_delay",
        default_value="2.0",
        description="Seconds to wait after each robot's scan bridge starts before launching its Nav2 stack.",
    )

    declare_rviz_start_delay_cmd = DeclareLaunchArgument(
        "rviz_start_delay",
        default_value="0.0",
        description="Seconds to wait after each robot's scan bridge starts before launching RViz.",
    )

    declare_startup_spacing_cmd = DeclareLaunchArgument(
        "startup_spacing",
        default_value="8.0",
        description="Additional seconds to wait between starting each robot's Nav2 stack.",
    )

    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        "rviz_config",
        default_value=rviz_config_dir,
        description="Full path to the RVIZ config file to use.",
    )

    declare_use_rviz_cmd = DeclareLaunchArgument(
        "use_rviz",
        default_value="True",
        description="Whether to start RVIZ.",
    )

    return LaunchDescription(
        [
            declare_team_config_file_cmd,
            declare_params_template_file_cmd,
            declare_map_yaml_cmd,
            declare_use_sim_time_cmd,
            declare_bt_xml_cmd,
            declare_use_rviz_cmd,
            declare_autostart_cmd,
            declare_use_respawn_cmd,
            declare_nav2_start_delay_cmd,
            declare_rviz_start_delay_cmd,
            declare_startup_spacing_cmd,
            declare_rviz_config_file_cmd,
            OpaqueFunction(function=_launch_setup),
        ]
    )
