# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


############################################################################
#### Launch warehouse MAPF demo with direct path tracking: #################
# 1. Start Isaac Sim on the host machine.
#    $ ./python.sh /path/to/mas-vln/isaac_sim/scripts/build_stage_warehouse_carters.py
# 2. Optionally launch lightweight RViz/scan debug helpers with the same team config.
#    $ ros2 launch carters_nav2 warehouse_team_lightweight.launch.py
#    The full Nav2 bringup remains available if you still want to compare behavior.
#    $ ros2 launch carters_nav2 warehouse_two_carters_nav2.launch.py
# 3. Launch MAPF execution with the same team config.
#    Custom tracker:
#    $ ros2 launch carters_goal isaac_ros_mapf.launch.py run_plan_executor:=true execution_backend:=tracker
#    Time-aware custom tracker with shared pre-rotation barrier:
#    $ ros2 launch carters_goal isaac_ros_mapf.launch.py run_plan_executor:=true execution_backend:=timed_tracker
#    Nav2 controller-only FollowPath:
#    $ ros2 launch carters_goal isaac_ros_mapf.launch.py run_plan_executor:=true execution_backend:=nav2
############################################################################


def _load_team_config_utils():
    carters_nav2_dir = get_package_share_directory("carters_nav2")
    helper_path = os.path.join(carters_nav2_dir, "launch", "team_config_utils.py")
    spec = importlib.util.spec_from_file_location("team_config_utils", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load team config utilities from {helper_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


team_config_utils = _load_team_config_utils()


def _launch_setup(context, *args, **kwargs):
    carters_nav2_dir = get_package_share_directory("carters_nav2")

    maps_dir = os.path.join(carters_nav2_dir, "maps")
    team_config_path = LaunchConfiguration("team_config_file").perform(context)
    map_override = LaunchConfiguration("map").perform(context)
    mapf_params_path = LaunchConfiguration("mapf_params_file").perform(context)
    costmap_params_path = LaunchConfiguration("mapf_costmap_params_file").perform(context)
    initial_pose_tf_params_path = LaunchConfiguration("initial_pose_tf_params_file").perform(context)

    team_config = team_config_utils.load_team_config(team_config_path, maps_dir=maps_dir)
    robot_namespaces = team_config["robot_namespaces"]
    agent_num = team_config["agent_num"]
    map_file = map_override or team_config["mapf_map"]
    if not map_file:
        raise RuntimeError(
            "No MAPF map was provided. Set environment.mapf_map in the team config "
            "or pass map:=/abs/path/to/map.yaml."
        )

    base_frame_map = team_config_utils.build_agent_indexed_map(robot_namespaces, "base_link")
    plan_topic_map = team_config_utils.build_agent_indexed_map(robot_namespaces, "plan")
    goal_topic_map = team_config_utils.build_agent_indexed_map(robot_namespaces, "goal")
    cmd_vel_topic_map = team_config_utils.build_agent_indexed_map(
        robot_namespaces,
        "cmd_vel",
        leading_slash=True,
    )
    agent_name_map = {
        f"agent_{index}": namespace for index, namespace in enumerate(robot_namespaces)
    }

    mapf_params = team_config_utils.load_yaml_file(mapf_params_path)
    mapf_params["mapf_base"]["mapf_base_node"]["ros__parameters"]["agent_num"] = agent_num
    mapf_params["mapf_base"]["mapf_base_node"]["ros__parameters"]["base_frame_id"] = base_frame_map
    mapf_params["mapf_base"]["mapf_base_node"]["ros__parameters"]["plan_topic"] = plan_topic_map
    mapf_params["mapf_base"]["goal_transformer"]["ros__parameters"]["agent_num"] = agent_num
    mapf_params["mapf_base"]["goal_transformer"]["ros__parameters"]["goal_topic"] = goal_topic_map
    mapf_params["mapf_base"]["plan_executor"]["ros__parameters"]["agent_num"] = agent_num
    mapf_params["mapf_base"]["plan_executor"]["ros__parameters"]["base_frame_id"] = base_frame_map
    mapf_params["mapf_base"]["plan_executor"]["ros__parameters"]["cmd_vel_topic"] = (
        cmd_vel_topic_map
    )
    mapf_params["mapf_base"]["plan_executor"]["ros__parameters"]["agent_name"] = agent_name_map
    mapf_params["mapf_base"]["plan_executor"]["ros__parameters"]["controller_id"] = "FollowPath"
    mapf_params["mapf_base"]["plan_executor"]["ros__parameters"]["goal_checker_id"] = (
        "stopped_goal_checker"
    )
    mapf_params["mapf_base"]["plan_executor"]["ros__parameters"]["progress_checker_id"] = (
        "progress_checker"
    )
    generated_mapf_params = team_config_utils.write_temp_yaml("carters_goal_mapf_", mapf_params)

    costmap_params = team_config_utils.load_yaml_file(costmap_params_path)
    costmap_params["mapf_base"]["mapf_costmap"]["mapf_costmap"]["ros__parameters"][
        "robot_base_frame"
    ] = f"{robot_namespaces[0]}/base_link"
    generated_costmap_params = team_config_utils.write_temp_yaml(
        "carters_goal_costmap_",
        costmap_params,
    )

    initial_pose_tf_params = team_config_utils.load_yaml_file(initial_pose_tf_params_path)
    initial_pose_tf_params["initial_pose_tf_publisher"]["ros__parameters"][
        "robot_namespaces"
    ] = robot_namespaces
    initial_pose_tf_params["initial_pose_tf_publisher"]["ros__parameters"][
        "initial_poses"
    ] = team_config["initial_pose_array"]
    generated_initial_pose_tf_params = team_config_utils.write_temp_yaml(
        "carters_goal_initial_pose_tf_",
        initial_pose_tf_params,
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    mapf_planner = LaunchConfiguration("mapf_planner")
    record_velocity = LaunchConfiguration("record_velocity")
    record_frequency_hz = LaunchConfiguration("record_frequency_hz")
    record_odom_topic_suffix = LaunchConfiguration("record_odom_topic_suffix")
    experiments_dir = LaunchConfiguration("experiments_dir")
    core_startup_delay = LaunchConfiguration("core_startup_delay")
    lifecycle_manager_delay = LaunchConfiguration("lifecycle_manager_delay")
    run_goal_publisher = LaunchConfiguration("run_goal_publisher")
    goal_publisher_delay = LaunchConfiguration("goal_publisher_delay")
    run_tf_bridge = LaunchConfiguration("run_tf_bridge")
    run_initial_pose_tf = LaunchConfiguration("run_initial_pose_tf")
    run_plan_executor = LaunchConfiguration("run_plan_executor")
    plan_executor_delay = LaunchConfiguration("plan_executor_delay")
    rollout_control_topic = LaunchConfiguration("rollout_control_topic")
    rollout_reset_done_topic = LaunchConfiguration("rollout_reset_done_topic")
    execution_status_topic = LaunchConfiguration("execution_status_topic")
    execution_backend = LaunchConfiguration("execution_backend").perform(context).strip().lower()
    run_plan_executor_enabled = (
        LaunchConfiguration("run_plan_executor").perform(context).strip().lower() == "true"
    )
    run_tf_bridge_enabled = LaunchConfiguration("run_tf_bridge").perform(context).strip().lower() == "true"

    executor_executable = "MapfPathTracker"
    if execution_backend == "nav2":
        executor_executable = "MapfNav2Executor"
    elif execution_backend == "timed_tracker":
        executor_executable = "MapfTimedTracker"

    lifecycle_nodes = ["map_server", "mapf_base_node"]

    mapf_core_group = GroupAction(
        [
            Node(
                namespace="mapf_base",
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[
                    generated_mapf_params,
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
                    generated_costmap_params,
                    generated_mapf_params,
                    {
                        "mapf_planner": mapf_planner,
                        "use_sim_time": use_sim_time,
                    },
                ],
            ),
            Node(
                namespace="mapf_base",
                package="mapf_base",
                executable="goal_transformer",
                name="goal_transformer",
                output="screen",
                parameters=[generated_mapf_params, {"use_sim_time": use_sim_time}],
            ),
        ]
    )

    lifecycle_manager = Node(
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
    )

    tf_bridge = Node(
        package="carters_goal",
        executable="NamespacedTfBridge",
        name="namespaced_tf_bridge",
        output="screen",
        condition=IfCondition(run_tf_bridge),
        parameters=[{"robot_namespaces": robot_namespaces}],
    )

    initial_pose_tf_publisher = Node(
        package="carters_goal",
        executable="InitialPoseTfPublisher",
        name="initial_pose_tf_publisher",
        output="screen",
        condition=IfCondition(run_initial_pose_tf),
        parameters=[
            generated_initial_pose_tf_params,
            {
                "use_sim_time": use_sim_time,
                "rollout_control_topic": rollout_control_topic,
                "rollout_reset_done_topic": rollout_reset_done_topic,
                "odom_topic_suffix": record_odom_topic_suffix,
                "publish_global_tf": not run_tf_bridge_enabled,
            },
        ],
    )

    velocity_recorder = Node(
        package="carters_goal",
        executable="RobotVelocityRecorder",
        name="robot_velocity_recorder",
        output="screen",
        condition=IfCondition(record_velocity),
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "team_config_file": team_config_path,
                "robot_namespaces": robot_namespaces,
                "odom_topic_suffix": record_odom_topic_suffix,
                "record_frequency_hz": record_frequency_hz,
                "experiments_dir": experiments_dir,
                "rollout_control_topic": rollout_control_topic,
            }
        ],
    )

    mapf_core_startup = TimerAction(period=core_startup_delay, actions=[mapf_core_group])

    lifecycle_manager_startup = TimerAction(
        period=lifecycle_manager_delay,
        actions=[lifecycle_manager],
    )

    goal_publisher = TimerAction(
        period=goal_publisher_delay,
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
                        "min_global_plan_subscribers": 1 if run_plan_executor_enabled else 1,
                        "agent_num": agent_num,
                        "goal_array": team_config["goal_pose_array"],
                    }
                ],
            )
        ],
    )

    plan_executor = TimerAction(
        period=plan_executor_delay,
        actions=[
            Node(
                namespace="mapf_base",
                package="carters_goal",
                executable=executor_executable,
                name="plan_executor",
                output="screen",
                condition=IfCondition(run_plan_executor),
                parameters=[
                    generated_mapf_params,
                    {
                        "use_sim_time": use_sim_time,
                        "execution_status_topic": execution_status_topic,
                        "rollout_control_topic": rollout_control_topic,
                        "team_config_file": team_config_path,
                        "experiments_dir": experiments_dir,
                    },
                ],
            )
        ],
    )

    return [
        velocity_recorder,
        tf_bridge,
        initial_pose_tf_publisher,
        mapf_core_startup,
        lifecycle_manager_startup,
        goal_publisher,
        plan_executor,
    ]


def generate_launch_description():
    carters_nav2_dir = get_package_share_directory("carters_nav2")
    carters_goal_dir = get_package_share_directory("carters_goal")

    map_arg = DeclareLaunchArgument(
        "map",
        default_value="",
        description=(
            "Optional full path to the map yaml for MAPF. "
            "If empty, the value from the team config is used."
        ),
    )
    team_config_arg = DeclareLaunchArgument(
        "team_config_file",
        default_value=os.path.join(
            carters_nav2_dir,
            "config",
            "warehouse",
            "warehouse_forklift.yaml",
        ),
        description="Full path to the shared robot team configuration YAML.",
    )
    mapf_params_arg = DeclareLaunchArgument(
        "mapf_params_file",
        default_value=os.path.join(carters_goal_dir, "config", "mapf_params_isaac.yaml"),
        description="Base MAPF params YAML. Robot-specific sections are generated at launch time.",
    )
    costmap_params_arg = DeclareLaunchArgument(
        "mapf_costmap_params_file",
        default_value=os.path.join(carters_goal_dir, "config", "mapf_costmap_params_isaac.yaml"),
        description="Base MAPF costmap params YAML. Robot-specific sections are generated at launch time.",
    )
    initial_pose_tf_params_arg = DeclareLaunchArgument(
        "initial_pose_tf_params_file",
        default_value=os.path.join(carters_goal_dir, "config", "initial_pose_tf_params_isaac.yaml"),
        description="Base initial pose TF params YAML. Robot-specific sections are generated at launch time.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock.",
    )
    autostart_arg = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Autostart lifecycle nodes.",
    )
    mapf_planner_arg = DeclareLaunchArgument(
        "mapf_planner",
        default_value="mapf_planner/CBSROS",
        description="MAPF planner plugin name.",
    )
    record_velocity_arg = DeclareLaunchArgument(
        "record_velocity",
        default_value="false",
        description="Record time-stamped simulator odometry velocities for each robot.",
    )
    record_frequency_arg = DeclareLaunchArgument(
        "record_frequency_hz",
        default_value="20.0",
        description=(
            "Maximum write frequency for velocity recording. "
            "Set to 0 to record every odometry message."
        ),
    )
    record_odom_topic_suffix_arg = DeclareLaunchArgument(
        "record_odom_topic_suffix",
        default_value="chassis/odom",
        description="Robot-relative odometry topic suffix used as the simulator velocity source.",
    )
    experiments_dir_arg = DeclareLaunchArgument(
        "experiments_dir",
        default_value="",
        description=(
            "Optional override for the experiments directory. "
            "Defaults to <repo>/experiments."
        ),
    )
    core_startup_delay_arg = DeclareLaunchArgument(
        "core_startup_delay",
        default_value="1.0",
        description="Seconds to wait before launching map_server, mapf_base_node, and goal_transformer.",
    )
    lifecycle_manager_delay_arg = DeclareLaunchArgument(
        "lifecycle_manager_delay",
        default_value="4.0",
        description="Seconds to wait before starting the lifecycle manager bringup sequence.",
    )
    run_goal_pub_arg = DeclareLaunchArgument(
        "run_goal_publisher",
        default_value="true",
        description="Run MAPF goal publisher.",
    )
    goal_publisher_delay_arg = DeclareLaunchArgument(
        "goal_publisher_delay",
        default_value="2.0",
        description="Seconds to wait before starting the MAPF goal publisher.",
    )
    run_tf_bridge_arg = DeclareLaunchArgument(
        "run_tf_bridge",
        default_value="true",
        description="Bridge /robotX/tf into global /tf with prefixed frame ids.",
    )
    run_initial_pose_tf_arg = DeclareLaunchArgument(
        "run_initial_pose_tf",
        default_value="true",
        description="Publish static map->robotX/odom transforms from the team config spawn poses.",
    )
    run_plan_executor_arg = DeclareLaunchArgument(
        "run_plan_executor",
        default_value="false",
        description="Execute MAPF plans with the selected execution backend.",
    )
    execution_backend_arg = DeclareLaunchArgument(
        "execution_backend",
        default_value="tracker",
        description=(
            "Execution backend: 'tracker' for the legacy path tracker, "
            "'timed_tracker' for time-aware cmd_vel tracking with a shared pre-rotation barrier, "
            "or 'nav2' for FollowPath."
        ),
    )
    plan_executor_delay_arg = DeclareLaunchArgument(
        "plan_executor_delay",
        default_value="1.5",
        description="Seconds to wait before starting the selected MAPF executor.",
    )
    rollout_control_topic_arg = DeclareLaunchArgument(
        "rollout_control_topic",
        default_value="",
        description=(
            "Optional rollout-control PoseArray topic used for multi-rollout execution. "
            "Leave empty for the legacy single-rollout flow."
        ),
    )
    execution_status_topic_arg = DeclareLaunchArgument(
        "execution_status_topic",
        default_value="/mapf_base/plan_execution_status",
        description="Status topic published by the active MAPF executor.",
    )
    rollout_reset_done_topic_arg = DeclareLaunchArgument(
        "rollout_reset_done_topic",
        default_value="",
        description=(
            "Optional Int32 acknowledgement topic published after Isaac Sim finishes a rollout reset. "
            "When provided, the initial-pose TF publisher waits for this ack before switching poses."
        ),
    )

    return LaunchDescription(
        [
            map_arg,
            team_config_arg,
            mapf_params_arg,
            costmap_params_arg,
            initial_pose_tf_params_arg,
            use_sim_time_arg,
            autostart_arg,
            mapf_planner_arg,
            record_velocity_arg,
            record_frequency_arg,
            record_odom_topic_suffix_arg,
            experiments_dir_arg,
            core_startup_delay_arg,
            lifecycle_manager_delay_arg,
            run_goal_pub_arg,
            goal_publisher_delay_arg,
            run_tf_bridge_arg,
            run_initial_pose_tf_arg,
            run_plan_executor_arg,
            execution_backend_arg,
            plan_executor_delay_arg,
            rollout_control_topic_arg,
            rollout_reset_done_topic_arg,
            execution_status_topic_arg,
            OpaqueFunction(function=_launch_setup),
        ]
    )
