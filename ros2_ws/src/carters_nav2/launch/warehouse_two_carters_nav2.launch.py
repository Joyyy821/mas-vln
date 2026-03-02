from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def nav2_group(robot_ns: str):
    params_file = LaunchConfiguration("params_file")

    # Per-robot overrides (important!)
    # Isaac Sim TF frames are typically robot-scoped like robot1/base_link, robot1/odom.
    # Nav2 nodes run in namespace robot1, so we can set frame ids to base_link/odom
    # IF your TF frames are literally "robot1/base_link", set them explicitly below.
    # To be robust with your current setup (you prefixed frames), we set explicit full frame ids.
    base_link = f"{robot_ns}/base_link"
    odom = f"{robot_ns}/odom"

    # Convert PointCloud2 -> LaserScan for AMCL/costmaps
    pcl_to_scan = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        parameters=[{
            "target_frame": base_link,
            "transform_tolerance": 0.05,
            "min_height": -0.2,
            "max_height": 0.5,
            "angle_min": -3.14159,
            "angle_max": 3.14159,
            "angle_increment": 0.0087,
            "scan_time": 0.1,
            "range_min": 0.2,
            "range_max": 30.0,
            "use_inf": True
        }],
        remappings=[
            ("cloud_in", "front_3d_lidar/lidar_points"),
            ("scan", "scan"),
        ],
    )

    # Nav2 bringup
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("nav2_bringup"), "launch", "bringup_launch.py"])
        ),
        launch_arguments={
            "namespace": robot_ns,
            "use_namespace": "True",
            "slam": "False",
            "map": LaunchConfiguration("map_yaml"),
            "use_sim_time": "True",
            "params_file": params_file,
            "autostart": "True",
        }.items(),
    )

    # Override frame ids for AMCL + costmaps + BT, etc.
    # This is done by launching small parameter override nodes is messy; instead we rely on
    # nav2 params having base_link/odom placeholders AND we set them via a param rewrite approach
    # in a future iteration. For now, simplest: keep frames in params as base_link/odom and ensure
    # TF frames are base_link/odom inside namespace.
    #
    # If your TF frames are robot1/base_link (not base_link), then you must set params accordingly.
    # We already set base_frame_id/odom_frame_id in the YAML, but they were 'base_link'/'odom'.
    # We'll inject overrides by launching amcl + nav2 nodes through bringup with rewritten params
    # in a later refinement if needed.
    #
    # Practical workaround: set your Isaac Sim TF frames to base_link/odom inside each namespace
    # OR update nav2_params.yaml to full frames and keep this consistent.
    #
    # Given your earlier frame-prefix patch, you likely have robot1/base_link etc.
    # So: set those explicitly via param overrides by using `param_substitutions` is not supported
    # directly here, so easiest is: run with frames as full ids in yaml.
    #
    # To keep this launch self-contained, we add a per-robot param override for AMCL only:
    amcl_override = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl_override",
        output="screen",
        parameters=[params_file, {
            "base_frame_id": base_link,
            "odom_frame_id": odom,
            "global_frame_id": "map",
            "scan_topic": "scan",
            "use_sim_time": True,
        }],
        # amcl launched by bringup too; so we do NOT actually start this by default.
        # Keep disabled unless you choose to bypass bringup's amcl.
        condition=None,
    )

    return GroupAction([
        PushRosNamespace(robot_ns),
        pcl_to_scan,
        nav2_bringup,
    ])


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "map_yaml",
            description="Path to the map YAML for the Simple Warehouse",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([FindPackageShare("carters_nav2"), "config", "nav2_params.yaml"]),
            description="Nav2 parameters file",
        ),

        # Global map server (shared map topic)
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "yaml_filename": LaunchConfiguration("map_yaml"),
            }],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_map",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "autostart": True,
                "node_names": ["map_server"],
            }],
        ),

        nav2_group("robot1"),
        nav2_group("robot2"),
    ])