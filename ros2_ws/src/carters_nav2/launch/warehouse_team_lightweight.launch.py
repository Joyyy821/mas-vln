# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
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


def _rviz_topic(
    value: str,
    *,
    depth: int = 5,
    durability: str = "Volatile",
    reliability: str = "Reliable",
) -> dict:
    return {
        "Depth": depth,
        "Durability Policy": durability,
        "History Policy": "Keep Last",
        "Reliability Policy": reliability,
        "Value": value,
    }


def _color(index: int) -> str:
    palette = [
        "255; 85; 85",
        "85; 170; 255",
        "255; 170; 0",
        "80; 220; 140",
        "220; 120; 255",
        "255; 230; 100",
    ]
    return palette[index % len(palette)]


def _robot_group(robot_name: str, index: int) -> dict:
    return {
        "Class": "rviz_common/Group",
        "Name": robot_name,
        "Enabled": True,
        "Displays": [
            {
                "Class": "rviz_default_plugins/Path",
                "Name": "MAPF Plan",
                "Enabled": True,
                "Color": _color(index),
                "Line Style": "Lines",
                "Line Width": 0.04,
                "Buffer Length": 1,
                "Head Diameter": 0.1,
                "Head Length": 0.1,
                "Length": 0.2,
                "Offset": {"X": 0, "Y": 0, "Z": 0},
                "Pose Color": "255; 255; 255",
                "Pose Style": "None",
                "Radius": 0.03,
                "Shaft Diameter": 0.02,
                "Shaft Length": 0.05,
                "Topic": _rviz_topic(f"/mapf_base/{robot_name}/plan"),
                "Value": True,
            },
            {
                "Class": "rviz_default_plugins/Path",
                "Name": "Nav2 Local Plan",
                "Enabled": False,
                "Color": "0; 150; 255",
                "Line Style": "Lines",
                "Line Width": 0.03,
                "Buffer Length": 1,
                "Head Diameter": 0.08,
                "Head Length": 0.08,
                "Length": 0.15,
                "Offset": {"X": 0, "Y": 0, "Z": 0},
                "Pose Color": "255; 255; 255",
                "Pose Style": "None",
                "Radius": 0.03,
                "Shaft Diameter": 0.02,
                "Shaft Length": 0.04,
                "Topic": _rviz_topic(f"/{robot_name}/local_plan"),
                "Value": False,
            },
            {
                "Class": "rviz_default_plugins/Map",
                "Name": "Nav2 Local Costmap",
                "Enabled": False,
                "Alpha": 0.5,
                "Color Scheme": "costmap",
                "Draw Behind": False,
                "Topic": _rviz_topic(
                    f"/{robot_name}/local_costmap/costmap",
                    depth=1,
                    durability="Transient Local",
                    reliability="Reliable",
                ),
                "Use Timestamp": False,
                "Value": False,
            },
            {
                "Class": "rviz_default_plugins/LaserScan",
                "Name": "Scan",
                "Enabled": True,
                "Color": "255; 255; 255",
                "Color Transformer": "Intensity",
                "Decay Time": 0,
                "Selectable": True,
                "Size (Pixels)": 3,
                "Size (m)": 0.01,
                "Style": "Flat Squares",
                "Topic": _rviz_topic(
                    f"/{robot_name}/scan",
                    reliability="Best Effort",
                ),
                "Use Fixed Frame": True,
                "Use rainbow": True,
                "Value": True,
            },
            {
                "Class": "rviz_default_plugins/PointCloud2",
                "Name": "3D Lidar",
                "Enabled": False,
                "Color Transformer": "AxisColor",
                "Axis": "Z",
                "Decay Time": 0,
                "Selectable": True,
                "Size (Pixels)": 3,
                "Size (m)": 0.03,
                "Style": "Flat Squares",
                "Topic": _rviz_topic(
                    f"/{robot_name}/front_3d_lidar/lidar_points",
                    reliability="Best Effort",
                ),
                "Use Fixed Frame": True,
                "Use rainbow": True,
                "Value": False,
            },
        ],
    }


def _build_rviz_config(robot_namespaces: list[str]) -> dict:
    displays = [
        {
            "Class": "rviz_default_plugins/Grid",
            "Name": "Grid",
            "Enabled": True,
            "Alpha": 0.5,
            "Cell Size": 1,
            "Color": "160; 160; 164",
            "Line Style": {"Line Width": 0.03, "Value": "Lines"},
            "Normal Cell Count": 0,
            "Offset": {"X": 0, "Y": 0, "Z": 0},
            "Plane": "XY",
            "Plane Cell Count": 10,
            "Reference Frame": "<Fixed Frame>",
            "Value": True,
        },
        {
            "Class": "rviz_default_plugins/TF",
            "Name": "TF",
            "Enabled": True,
            "Frame Timeout": 15,
            "Frames": {"All Enabled": False},
            "Marker Scale": 1,
            "Show Arrows": True,
            "Show Axes": True,
            "Show Names": False,
            "Tree": {},
            "Update Interval": 0,
            "Value": True,
        },
        {
            "Class": "rviz_default_plugins/Map",
            "Name": "MAPF Map",
            "Enabled": True,
            "Alpha": 1,
            "Color Scheme": "map",
            "Draw Behind": True,
            "Topic": _rviz_topic(
                "/mapf_base/map",
                depth=1,
                durability="Transient Local",
                reliability="Reliable",
            ),
            "Use Timestamp": False,
            "Value": True,
        },
        {
            "Class": "rviz_default_plugins/Map",
            "Name": "MAPF Costmap",
            "Enabled": True,
            "Alpha": 0.65,
            "Color Scheme": "costmap",
            "Draw Behind": False,
            "Topic": _rviz_topic(
                "/mapf_base/mapf_costmap/costmap",
                depth=1,
                durability="Transient Local",
                reliability="Reliable",
            ),
            "Use Timestamp": False,
            "Value": True,
        },
        {
            "Class": "rviz_default_plugins/PoseArray",
            "Name": "MAPF Goals",
            "Enabled": True,
            "Arrow Length": 0.25,
            "Axes Length": 0.3,
            "Axes Radius": 0.01,
            "Color": "255; 215; 0",
            "Head Length": 0.07,
            "Head Radius": 0.03,
            "Shaft Length": 0.18,
            "Shaft Radius": 0.01,
            "Shape": "Arrow (Flat)",
            "Topic": _rviz_topic("/mapf_base/goal_for_each", reliability="Reliable"),
            "Value": True,
        },
    ]
    displays.extend(_robot_group(robot_name, index) for index, robot_name in enumerate(robot_namespaces))

    return {
        "Panels": [
            {
                "Class": "rviz_common/Displays",
                "Name": "Displays",
                "Help Height": 195,
                "Property Tree Widget": {
                    "Expanded": ["/Global Options1", "/TF1/Frames1", "/TF1/Tree1"],
                    "Splitter Ratio": 0.58,
                },
                "Tree Height": 464,
            },
            {"Class": "rviz_common/Selection", "Name": "Selection"},
            {
                "Class": "rviz_common/Tool Properties",
                "Name": "Tool Properties",
                "Expanded": [],
                "Splitter Ratio": 0.59,
            },
            {
                "Class": "rviz_common/Views",
                "Name": "Views",
                "Expanded": ["/Current View1"],
                "Splitter Ratio": 0.5,
            },
        ],
        "Visualization Manager": {
            "Class": "",
            "Displays": displays,
            "Enabled": True,
            "Global Options": {
                "Background Color": "48; 48; 48",
                "Fixed Frame": "map",
                "Frame Rate": 30,
            },
            "Name": "root",
            "Tools": [
                {"Class": "rviz_default_plugins/MoveCamera"},
                {"Class": "rviz_default_plugins/Select"},
                {"Class": "rviz_default_plugins/FocusCamera"},
                {"Class": "rviz_default_plugins/Measure", "Line color": "128; 128; 0"},
            ],
            "Transformation": {"Current": {"Class": "rviz_default_plugins/TF"}},
            "Value": True,
            "Views": {
                "Current": {
                    "Angle": -1.57,
                    "Class": "rviz_default_plugins/TopDownOrtho",
                    "Enable Stereo Rendering": {
                        "Stereo Eye Separation": 0.06,
                        "Stereo Focal Distance": 1,
                        "Swap Stereo Eyes": False,
                        "Value": False,
                    },
                    "Invert Z Axis": False,
                    "Name": "Current View",
                    "Near Clip Distance": 0.01,
                    "Scale": 80.0,
                    "Target Frame": "<Fixed Frame>",
                    "Value": "TopDownOrtho (rviz_default_plugins)",
                    "X": 0.0,
                    "Y": 0.0,
                },
                "Saved": "~",
            },
        },
        "Window Geometry": {
            "Displays": {"collapsed": False},
            "Height": 900,
            "Hide Left Dock": False,
            "Hide Right Dock": True,
            "Selection": {"collapsed": False},
            "Tool Properties": {"collapsed": False},
            "Views": {"collapsed": True},
            "Width": 1500,
            "X": 180,
            "Y": 100,
        },
    }


def _launch_setup(context, *args, **kwargs):
    carters_nav2_dir = get_package_share_directory("carters_nav2")
    maps_dir = os.path.join(carters_nav2_dir, "maps")

    team_config_path = LaunchConfiguration("team_config_file").perform(context)
    rviz_config_override = LaunchConfiguration("rviz_config").perform(context)
    controller_params_template_file = LaunchConfiguration(
        "controller_params_template_file"
    ).perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")
    use_scan_bridge = LaunchConfiguration("use_scan_bridge")
    use_nav2_controller = LaunchConfiguration("use_nav2_controller")
    controller_autostart = LaunchConfiguration("controller_autostart")
    use_nav2_controller_enabled = (
        LaunchConfiguration("use_nav2_controller").perform(context).strip().lower() == "true"
    )

    initial_start_delay = float(LaunchConfiguration("initial_start_delay").perform(context))
    startup_spacing = float(LaunchConfiguration("startup_spacing").perform(context))
    rviz_start_delay = float(LaunchConfiguration("rviz_start_delay").perform(context))
    controller_start_delay = float(LaunchConfiguration("controller_start_delay").perform(context))

    team_config = team_config_utils.load_team_config(team_config_path, maps_dir=maps_dir)
    robot_namespaces = team_config["robot_namespaces"]

    actions = [
        LogInfo(
            msg=(
                "Launching lightweight warehouse debug bringup for "
                f"{team_config['agent_num']} robots from {team_config_path}. "
                "This launch does not start AMCL, planner_server, bt_navigator, or any map_server."
            )
        )
    ]
    if use_nav2_controller_enabled:
        actions.append(
            LogInfo(
                msg=(
                    "Controller-only Nav2 bringup is enabled. Each robot will start "
                    "controller_server with Regulated Pure Pursuit."
                )
            )
        )

    rviz_config_path = rviz_config_override
    if not rviz_config_path:
        rviz_config = _build_rviz_config(robot_namespaces)
        rviz_config_path = team_config_utils.write_temp_yaml("carters_nav2_rviz_", rviz_config)

    actions.append(
        TimerAction(
            period=initial_start_delay + rviz_start_delay,
            actions=[
                Node(
                    package="rviz2",
                    executable="rviz2",
                    name="warehouse_team_lightweight_rviz",
                    output="screen",
                    condition=IfCondition(use_rviz),
                    arguments=["-d", rviz_config_path],
                    parameters=[{"use_sim_time": use_sim_time}],
                )
            ],
        )
    )

    for index, robot_name in enumerate(team_config["robot_namespaces"]):
        robot_start_offset = initial_start_delay + (startup_spacing * index)
        robot_config = team_config["robots"][index]
        actions.append(
            TimerAction(
                period=robot_start_offset,
                actions=[
                    LogInfo(
                        msg=(
                            f"Starting lightweight pointcloud->scan bridge for {robot_name} "
                            f"after {robot_start_offset:.1f}s."
                        )
                    ),
                    Node(
                        package="pointcloud_to_laserscan",
                        executable="pointcloud_to_laserscan_node",
                        namespace=robot_name,
                        name="pointcloud_to_laserscan",
                        output="screen",
                        condition=IfCondition(use_scan_bridge),
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
                                "use_sim_time": use_sim_time,
                            }
                        ],
                        respawn=True,
                        respawn_delay=2.0,
                    ),
                ],
            )
        )

        if use_nav2_controller_enabled:
            controller_params = team_config_utils.render_nav2_params(
                template_path=controller_params_template_file,
                robot_namespace=robot_name,
                initial_pose=robot_config["initial_pose"],
            )
            controller_params["lifecycle_manager_controller"]["ros__parameters"][
                "autostart"
            ] = controller_autostart.perform(context).strip().lower() == "true"
            controller_params_file = team_config_utils.write_temp_yaml(
                prefix=f"carters_nav2_controller_{robot_name}_",
                data={robot_name: controller_params},
            )
            actions.append(
                TimerAction(
                    period=robot_start_offset + controller_start_delay,
                    actions=[
                        LogInfo(
                            msg=(
                                f"Starting controller-only Nav2 RPP server for {robot_name} "
                                f"after {robot_start_offset + controller_start_delay:.1f}s."
                            )
                        ),
                        Node(
                            package="nav2_controller",
                            executable="controller_server",
                            namespace=robot_name,
                            name="controller_server",
                            output="screen",
                            condition=IfCondition(use_nav2_controller),
                            remappings=[
                                ("/tf", "tf"),
                                ("/tf_static", "tf_static"),
                            ],
                            parameters=[controller_params_file, {"use_sim_time": use_sim_time}],
                            respawn=True,
                            respawn_delay=2.0,
                        ),
                        Node(
                            package="nav2_lifecycle_manager",
                            executable="lifecycle_manager",
                            namespace=robot_name,
                            name="lifecycle_manager_controller",
                            output="screen",
                            condition=IfCondition(use_nav2_controller),
                            parameters=[
                                {
                                    "use_sim_time": use_sim_time,
                                    "autostart": controller_autostart,
                                    "node_names": ["controller_server"],
                                }
                            ],
                        ),
                    ],
                )
            )

    return actions


def generate_launch_description():
    carters_nav2_dir = get_package_share_directory("carters_nav2")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "team_config_file",
                default_value=os.path.join(
                    carters_nav2_dir,
                    "config",
                    "warehouse",
                    "warehouse_team_config.yaml",
                ),
                description="Full path to the shared robot team configuration YAML.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="True",
                description="Use Isaac Sim clock if true.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="True",
                description="Whether to start a lightweight global RViz instance.",
            ),
            DeclareLaunchArgument(
                "use_scan_bridge",
                default_value="True",
                description="Whether to start pointcloud->scan bridges for each robot.",
            ),
            DeclareLaunchArgument(
                "use_nav2_controller",
                default_value="False",
                description=(
                    "Whether to start controller-only Nav2 FollowPath servers with "
                    "Regulated Pure Pursuit for each robot."
                ),
            ),
            DeclareLaunchArgument(
                "controller_autostart",
                default_value="True",
                description="Autostart the controller-only Nav2 lifecycle manager.",
            ),
            DeclareLaunchArgument(
                "controller_start_delay",
                default_value="1.0",
                description="Additional seconds to wait after each robot start offset before launching controller_server.",
            ),
            DeclareLaunchArgument(
                "controller_params_template_file",
                default_value=os.path.join(
                    carters_nav2_dir,
                    "config",
                    "warehouse",
                    "multi_robot_carter_rpp_controller_only_params.yaml",
                ),
                description=(
                    "Base Nav2 controller-only params YAML used as a per-robot template."
                ),
            ),
            DeclareLaunchArgument(
                "initial_start_delay",
                default_value="2.0",
                description="Seconds to wait before starting RViz and the first scan bridge.",
            ),
            DeclareLaunchArgument(
                "startup_spacing",
                default_value="1.0",
                description="Additional seconds to wait between starting each robot scan bridge.",
            ),
            DeclareLaunchArgument(
                "rviz_start_delay",
                default_value="0.0",
                description="Additional seconds to wait after initial_start_delay before launching RViz.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value="",
                description=(
                    "Optional full path to an RViz config file. If empty, a MAPF-oriented "
                    "config is generated from the team config."
                ),
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
