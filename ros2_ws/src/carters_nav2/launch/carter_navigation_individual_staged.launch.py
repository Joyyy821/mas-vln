import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import PushRosNamespace


def _launch_setup(context, *args, **kwargs):
    nav2_launch_dir = os.path.join(get_package_share_directory("nav2_bringup"), "launch")

    # Resolve robot-specific values now so the delayed navigation callback
    # cannot accidentally pick up another robot's launch context later.
    namespace = LaunchConfiguration("namespace").perform(context)
    map_yaml_file = LaunchConfiguration("map").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    params_file = LaunchConfiguration("params_file").perform(context)
    autostart = LaunchConfiguration("autostart").perform(context)
    use_composition = LaunchConfiguration("use_composition").perform(context)
    use_respawn = LaunchConfiguration("use_respawn").perform(context)
    localization_start_delay = float(
        LaunchConfiguration("localization_start_delay").perform(context)
    )
    navigation_activation_delay = float(
        LaunchConfiguration("navigation_activation_delay").perform(context)
    )
    log_level = LaunchConfiguration("log_level").perform(context)

    localization_cmd = TimerAction(
        period=localization_start_delay,
        actions=[
            GroupAction(
                actions=[
                    PushRosNamespace(namespace=namespace),
                    LogInfo(msg=["Launching localization stack for ", namespace]),
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, "localization_launch.py")),
                        launch_arguments={
                            "namespace": namespace,
                            "map": map_yaml_file,
                            "use_sim_time": use_sim_time,
                            "params_file": params_file,
                            "autostart": autostart,
                            "use_composition": use_composition,
                            "use_respawn": use_respawn,
                            "log_level": log_level,
                        }.items(),
                    ),
                ]
            ),
        ],
    )

    wait_for_localization_ready = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            (
                f'until ros2 lifecycle get /{namespace}/amcl 2>/dev/null | '
                f'grep -q "active"; do sleep 1; done; sleep {navigation_activation_delay}'
            ),
        ],
        output="screen",
    )

    wait_for_localization_cmd = TimerAction(
        period=localization_start_delay,
        actions=[
            LogInfo(msg=["Waiting for localization to become active for ", namespace]),
            wait_for_localization_ready,
        ],
    )

    navigation_group = GroupAction(
        actions=[
            PushRosNamespace(namespace=namespace),
            LogInfo(msg=["Launching navigation stack for ", namespace]),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, "navigation_launch.py")),
                launch_arguments={
                    "namespace": namespace,
                    "use_sim_time": use_sim_time,
                    "params_file": params_file,
                    "autostart": autostart,
                    "use_composition": use_composition,
                    "use_respawn": use_respawn,
                    "log_level": log_level,
                }.items(),
            ),
        ]
    )

    navigation_ready_event = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_localization_ready,
            on_exit=[navigation_group],
        )
    )

    return [localization_cmd, wait_for_localization_cmd, navigation_ready_event]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "namespace",
                default_value="",
                description="Top-level namespace",
            ),
            DeclareLaunchArgument(
                "map",
                description="Full path to map file to load",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="True",
                description="Use simulation (Isaac Sim) clock if true",
            ),
            DeclareLaunchArgument(
                "params_file",
                description="Full path to the ROS2 parameters file to use for all launched nodes",
            ),
            DeclareLaunchArgument(
                "autostart",
                default_value="true",
                description="Automatically startup the nav2 stack",
            ),
            DeclareLaunchArgument(
                "use_composition",
                default_value="False",
                description="Whether to use composed bringup",
            ),
            DeclareLaunchArgument(
                "use_respawn",
                default_value="True",
                description="Whether to respawn Nav2 servers if one exits unexpectedly",
            ),
            DeclareLaunchArgument(
                "localization_start_delay",
                default_value="0.0",
                description="Seconds to wait before launching map_server and AMCL",
            ),
            DeclareLaunchArgument(
                "navigation_activation_delay",
                default_value="8.0",
                description="Additional seconds to wait after AMCL becomes active before launching planner, controller, and costmaps",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="Nav2 log level",
            ),
            DeclareLaunchArgument(
                "default_bt_xml_filename",
                default_value="",
                description="Compatibility argument forwarded by parent launch",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
