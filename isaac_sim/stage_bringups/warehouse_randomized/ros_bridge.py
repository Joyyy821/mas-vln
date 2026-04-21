from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from isaac_sim.stage_bringups.runtime_utils import (
    get_world_pose_xyzw,
    quaternion_xyzw_to_yaw,
    set_xform_pose,
)
from isaac_sim.stage_bringups.warehouse_randomized.robots import RuntimeRobotController


def _parse_rollout_id(frame_id: str) -> int | None:
    prefix = "rollout:"
    if not str(frame_id).startswith(prefix):
        return None
    try:
        return int(str(frame_id)[len(prefix) :])
    except ValueError:
        return None


@dataclass
class _RobotStateCache:
    sim_time_sec: float | None = None
    position_xy: np.ndarray | None = None
    yaw_rad: float | None = None


class InternalIsaacRosBridge:
    """Minimal internal ROS bridge for sensorless differential-drive robots."""

    def __init__(
        self,
        *,
        sim_app,
        robot_controllers: list[RuntimeRobotController],
        rollout_control_topic: str,
        rollout_reset_done_topic: str,
    ) -> None:
        try:
            import rclpy
            from builtin_interfaces.msg import Time as TimeMsg
            from geometry_msgs.msg import PoseArray, TransformStamped, Twist
            from nav_msgs.msg import Odometry
            from rclpy.node import Node
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
            from rosgraph_msgs.msg import Clock
            from std_msgs.msg import Int32
            from tf2_msgs.msg import TFMessage
        except Exception as exc:
            raise RuntimeError(
                "ROS 2 Python packages are unavailable inside the Isaac Sim runtime. "
                "Launch this script from Isaac Sim 5.1 Python with the built-in ROS support enabled."
            ) from exc

        try:
            from isaacsim.core.utils.types import ArticulationAction
        except Exception:
            from omni.isaac.core.utils.types import ArticulationAction

        self._sim_app = sim_app
        self._robot_controllers = list(robot_controllers)
        self._rollout_control_topic = str(rollout_control_topic).strip()
        self._rollout_reset_done_topic = str(rollout_reset_done_topic).strip()
        self._rclpy = rclpy
        self._Clock = Clock
        self._TimeMsg = TimeMsg
        self._Int32 = Int32
        self._Odometry = Odometry
        self._TFMessage = TFMessage
        self._TransformStamped = TransformStamped
        self._ArticulationAction = ArticulationAction

        self._owns_rclpy = not self._rclpy.ok()
        if self._owns_rclpy:
            self._rclpy.init()

        self._node: Node = Node("isaac_randomized_warehouse_bridge")
        tf_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )
        self._clock_pub = self._node.create_publisher(Clock, "/clock", 20)
        self._reset_done_pub = self._node.create_publisher(Int32, self._rollout_reset_done_topic, 10)

        self._cmd_subs = []
        self._odom_pubs: dict[str, Any] = {}
        self._tf_pubs: dict[str, Any] = {}
        self._latest_cmd_by_namespace: dict[str, tuple[float, float, float]] = {}
        self._state_cache_by_namespace: dict[str, _RobotStateCache] = {}
        self._pending_reset: tuple[int, list[float]] | None = None
        self._last_applied_rollout_id: int | None = None

        for controller in self._robot_controllers:
            namespace = controller.namespace
            self._latest_cmd_by_namespace[namespace] = (0.0, 0.0, float("-inf"))
            self._state_cache_by_namespace[namespace] = _RobotStateCache()

            self._cmd_subs.append(
                self._node.create_subscription(
                    Twist,
                    f"/{namespace}/cmd_vel",
                    lambda msg, robot_namespace=namespace: self._cmd_vel_callback(robot_namespace, msg),
                    10,
                )
            )
            self._odom_pubs[namespace] = self._node.create_publisher(
                Odometry,
                f"/{namespace}/chassis/odom",
                20,
            )
            self._tf_pubs[namespace] = self._node.create_publisher(
                TFMessage,
                f"/{namespace}/tf",
                tf_qos,
            )

        self._rollout_control_sub = self._node.create_subscription(
            PoseArray,
            self._rollout_control_topic,
            self._rollout_control_callback,
            10,
        )

        self._node.get_logger().info(
            "Internal Isaac ROS bridge is active for namespaces "
            f"{[controller.namespace for controller in self._robot_controllers]}. "
            f"Reset topic: {self._rollout_control_topic}, ack topic: {self._rollout_reset_done_topic}"
        )

    def _cmd_vel_callback(self, namespace: str, msg) -> None:
        self._latest_cmd_by_namespace[namespace] = (
            float(msg.linear.x),
            float(msg.angular.z),
            time.monotonic(),
        )

    def _rollout_control_callback(self, msg) -> None:
        rollout_id = _parse_rollout_id(msg.header.frame_id)
        if rollout_id is None:
            return
        if rollout_id <= 0:
            self._pending_reset = None
            return

        flat_pose_array: list[float] = []
        for pose in msg.poses:
            flat_pose_array.extend(
                [
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ]
            )

        expected_values = len(self._robot_controllers) * 7
        if len(flat_pose_array) != expected_values:
            self._node.get_logger().warn(
                f"Ignoring rollout {rollout_id} reset because it contains {len(flat_pose_array)} "
                f"pose values and {expected_values} were expected."
            )
            return
        self._pending_reset = (rollout_id, flat_pose_array)

    def spin_once(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def step(self, sim_time_sec: float) -> None:
        self.publish_clock(sim_time_sec)
        self.apply_wheel_commands()
        self.process_pending_reset()
        self.publish_robot_state(sim_time_sec)

    def _sim_time_to_msg(self, sim_time_sec: float):
        stamp = self._TimeMsg()
        sec = int(math.floor(float(sim_time_sec)))
        nanosec = int(round((float(sim_time_sec) - sec) * 1_000_000_000.0))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        stamp.sec = sec
        stamp.nanosec = nanosec
        return stamp

    def publish_clock(self, sim_time_sec: float) -> None:
        clock_msg = self._Clock()
        stamp = self._sim_time_to_msg(sim_time_sec)
        clock_msg.clock.sec = stamp.sec
        clock_msg.clock.nanosec = stamp.nanosec
        self._clock_pub.publish(clock_msg)

    def apply_wheel_commands(self) -> None:
        now_monotonic = time.monotonic()
        for controller in self._robot_controllers:
            linear_x, angular_z, stamp_sec = self._latest_cmd_by_namespace[controller.namespace]
            if now_monotonic - stamp_sec > controller.cmd_timeout_sec:
                linear_x = 0.0
                angular_z = 0.0

            linear_x = float(
                np.clip(linear_x, -controller.max_linear_speed_mps, controller.max_linear_speed_mps)
            )
            angular_z = float(
                np.clip(
                    angular_z,
                    -controller.max_angular_speed_rps,
                    controller.max_angular_speed_rps,
                )
            )
            right_velocity = (
                (2.0 * linear_x) + (angular_z * controller.wheel_distance_m)
            ) / (2.0 * controller.wheel_radius_m)
            left_velocity = (
                (2.0 * linear_x) - (angular_z * controller.wheel_distance_m)
            ) / (2.0 * controller.wheel_radius_m)

            action = self._ArticulationAction(
                joint_velocities=np.array([left_velocity, right_velocity], dtype=np.float32),
                joint_indices=controller.wheel_joint_indices,
            )
            controller.articulation_controller.apply_action(action)

    def process_pending_reset(self) -> None:
        if self._pending_reset is None:
            return

        rollout_id, flat_pose_array = self._pending_reset
        self._pending_reset = None
        if rollout_id == self._last_applied_rollout_id:
            self._publish_reset_ack(rollout_id)
            return

        self._apply_reset(rollout_id, flat_pose_array)
        self._publish_reset_ack(rollout_id)

    def _apply_reset(self, rollout_id: int, flat_pose_array: list[float]) -> None:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        was_playing = timeline.is_playing()
        if was_playing:
            pause_fn = getattr(timeline, "pause", None)
            if callable(pause_fn):
                pause_fn()
            else:
                timeline.stop()
            for _ in range(5):
                self._sim_app.update()

        for namespace in self._latest_cmd_by_namespace:
            self._latest_cmd_by_namespace[namespace] = (0.0, 0.0, time.monotonic())
            self._state_cache_by_namespace[namespace] = _RobotStateCache()

        for idx, controller in enumerate(self._robot_controllers):
            offset = idx * 7
            x = float(flat_pose_array[offset + 0])
            y = float(flat_pose_array[offset + 1])
            z = float(flat_pose_array[offset + 2])
            qx = float(flat_pose_array[offset + 3])
            qy = float(flat_pose_array[offset + 4])
            qz = float(flat_pose_array[offset + 5])
            qw = float(flat_pose_array[offset + 6])
            yaw_rad = float(
                math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            )

            zero_action = self._ArticulationAction(
                joint_velocities=np.array([0.0, 0.0], dtype=np.float32),
                joint_indices=controller.wheel_joint_indices,
            )
            controller.articulation_controller.apply_action(zero_action)
            set_xform_pose(
                controller.root_prim_path,
                (x, y, z),
                yaw_deg=yaw_rad * 180.0 / math.pi,
            )

        for _ in range(8):
            self._sim_app.update()

        if was_playing:
            timeline.play()
            for _ in range(8):
                self._sim_app.update()

        self._last_applied_rollout_id = rollout_id

    def _publish_reset_ack(self, rollout_id: int) -> None:
        ack_msg = self._Int32()
        ack_msg.data = int(rollout_id)
        self._reset_done_pub.publish(ack_msg)

    def publish_robot_state(self, sim_time_sec: float) -> None:
        for controller in self._robot_controllers:
            position_xyz, orientation_xyzw = get_world_pose_xyzw(controller.root_prim_path)
            yaw_rad = quaternion_xyzw_to_yaw(orientation_xyzw)
            cache = self._state_cache_by_namespace[controller.namespace]

            linear_vx = 0.0
            linear_vy = 0.0
            angular_vz = 0.0
            if cache.sim_time_sec is not None and cache.position_xy is not None and cache.yaw_rad is not None:
                dt = float(sim_time_sec) - float(cache.sim_time_sec)
                if dt > 1e-6:
                    delta_xy = position_xyz[:2] - cache.position_xy
                    world_vx = float(delta_xy[0] / dt)
                    world_vy = float(delta_xy[1] / dt)
                    cos_yaw = math.cos(yaw_rad)
                    sin_yaw = math.sin(yaw_rad)
                    linear_vx = cos_yaw * world_vx + sin_yaw * world_vy
                    linear_vy = -sin_yaw * world_vx + cos_yaw * world_vy
                    angular_vz = math.atan2(
                        math.sin(yaw_rad - cache.yaw_rad),
                        math.cos(yaw_rad - cache.yaw_rad),
                    ) / dt

            cache.sim_time_sec = float(sim_time_sec)
            cache.position_xy = position_xyz[:2].copy()
            cache.yaw_rad = float(yaw_rad)

            stamp = self._sim_time_to_msg(sim_time_sec)

            odom_msg = self._Odometry()
            odom_msg.header.stamp = stamp
            odom_msg.header.frame_id = controller.odom_frame_id
            odom_msg.child_frame_id = controller.base_frame_id
            odom_msg.pose.pose.position.x = float(position_xyz[0])
            odom_msg.pose.pose.position.y = float(position_xyz[1])
            odom_msg.pose.pose.position.z = float(position_xyz[2])
            odom_msg.pose.pose.orientation.x = float(orientation_xyzw[0])
            odom_msg.pose.pose.orientation.y = float(orientation_xyzw[1])
            odom_msg.pose.pose.orientation.z = float(orientation_xyzw[2])
            odom_msg.pose.pose.orientation.w = float(orientation_xyzw[3])
            odom_msg.twist.twist.linear.x = float(linear_vx)
            odom_msg.twist.twist.linear.y = float(linear_vy)
            odom_msg.twist.twist.angular.z = float(angular_vz)
            self._odom_pubs[controller.namespace].publish(odom_msg)

            transform = self._TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = controller.odom_frame_id
            transform.child_frame_id = controller.base_frame_id
            transform.transform.translation.x = float(position_xyz[0])
            transform.transform.translation.y = float(position_xyz[1])
            transform.transform.translation.z = float(position_xyz[2])
            transform.transform.rotation.x = float(orientation_xyzw[0])
            transform.transform.rotation.y = float(orientation_xyzw[1])
            transform.transform.rotation.z = float(orientation_xyzw[2])
            transform.transform.rotation.w = float(orientation_xyzw[3])
            self._tf_pubs[controller.namespace].publish(self._TFMessage(transforms=[transform]))

    def shutdown(self) -> None:
        try:
            self._node.destroy_node()
        finally:
            if self._owns_rclpy and self._rclpy.ok():
                self._rclpy.shutdown()
