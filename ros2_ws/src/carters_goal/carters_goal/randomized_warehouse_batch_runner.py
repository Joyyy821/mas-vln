#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from carters_goal.shared_team_config import rollout_run_dir, team_config_utils

try:
    from rcl_interfaces.msg import ParameterDescriptor
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover - lets unit tests import pure helpers without ROS.
    ParameterDescriptor = None  # type: ignore[assignment]
    rclpy = None
    Node = object  # type: ignore[assignment]
    String = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SceneBundle:
    scene_id: str
    scene_num: int
    bundle_dir: Path
    scene_usd: Path
    team_config: Path
    mapf_map: Path
    rollout_ids: tuple[int, ...]


@dataclass(frozen=True)
class BatchItem:
    scene: SceneBundle
    rollout_id: int


COMBINED_XY_OVERLAY_NAME = "combined_xy_overlay.png"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _scene_number(scene_name: str) -> int | None:
    if not scene_name.startswith("scene_"):
        return None
    try:
        return int(scene_name.split("_", 1)[1])
    except ValueError:
        return None


def discover_scene_bundles(scene_root_dir: str | Path) -> list[SceneBundle]:
    root = Path(scene_root_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Scene root directory does not exist: {root}")

    discovered: list[tuple[int, Path]] = []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        scene_num = _scene_number(candidate.name)
        if scene_num is None:
            continue
        discovered.append((scene_num, candidate))
    discovered.sort(key=lambda item: item[0])

    missing: list[str] = []
    bundles: list[SceneBundle] = []
    for scene_num, bundle_dir in discovered:
        scene_usd = bundle_dir / "scene.usd"
        team_config = bundle_dir / "team_config.yaml"
        mapf_map = bundle_dir / "mapf_map.yaml"
        missing_files = [
            path.name for path in (scene_usd, team_config, mapf_map) if not path.is_file()
        ]
        if missing_files:
            missing.append(f"{bundle_dir.name}: missing {', '.join(missing_files)}")
            continue

        team_payload = team_config_utils.load_multi_rollout_config(str(team_config))
        rollout_ids = tuple(int(rollout_id) for rollout_id in team_payload["rollout_ids"])
        if not rollout_ids:
            missing.append(f"{bundle_dir.name}: team_config.yaml contains no rollouts")
            continue

        bundles.append(
            SceneBundle(
                scene_id=bundle_dir.name,
                scene_num=scene_num,
                bundle_dir=bundle_dir.resolve(),
                scene_usd=scene_usd.resolve(),
                team_config=team_config.resolve(),
                mapf_map=mapf_map.resolve(),
                rollout_ids=rollout_ids,
            )
        )

    if missing:
        joined = "\n  - " + "\n  - ".join(missing)
        raise RuntimeError(f"Randomized warehouse scene preflight failed:{joined}")
    if not bundles:
        raise RuntimeError(f"No scene_<n> bundles were found under {root}.")
    return bundles


def build_batch_items(
    scenes: list[SceneBundle],
    *,
    continue_enabled: bool,
    continue_scene_id: str,
    continue_rollout_id: int,
) -> list[BatchItem]:
    if not continue_enabled:
        return [
            BatchItem(scene=scene, rollout_id=rollout_id)
            for scene in scenes
            for rollout_id in scene.rollout_ids
        ]

    if not continue_scene_id:
        raise ValueError("continue_scene_id is required when continue is true.")
    if continue_rollout_id <= 0:
        raise ValueError("continue_rollout_id must be positive when continue is true.")

    scene_by_id = {scene.scene_id: scene for scene in scenes}
    if continue_scene_id not in scene_by_id:
        available = ", ".join(scene.scene_id for scene in scenes)
        raise ValueError(
            f"continue_scene_id={continue_scene_id!r} was not found. Available: {available}"
        )

    start_scene = scene_by_id[continue_scene_id]
    if continue_rollout_id not in start_scene.rollout_ids:
        available = ", ".join(str(rollout_id) for rollout_id in start_scene.rollout_ids)
        raise ValueError(
            f"continue_rollout_id={continue_rollout_id} was not found in "
            f"{continue_scene_id}. Available: {available}"
        )

    items: list[BatchItem] = []
    for scene in scenes:
        if scene.scene_num < start_scene.scene_num:
            continue
        for rollout_id in scene.rollout_ids:
            if scene.scene_num == start_scene.scene_num and rollout_id < continue_rollout_id:
                continue
            items.append(BatchItem(scene=scene, rollout_id=rollout_id))
    return items


def build_single_rollout_launch_command(
    *,
    scene: SceneBundle,
    rollout_id: int,
    launch_options: dict[str, str],
) -> list[str]:
    command = [
        "ros2",
        "launch",
        "carters_goal",
        "isaac_ros_mapf.launch.py",
        f"team_config_file:={scene.team_config}",
        f"rollout_id:={rollout_id}",
        f"map:={scene.mapf_map}",
        "run_goal_publisher:=true",
        "overwrite_existing_rollout:=true",
    ]
    for key, value in launch_options.items():
        if key == "save_traj_plot":
            continue
        if value != "":
            command.append(f"{key}:={value}")
    return command


def cleanup_successful_rollout_artifacts(
    rollout_dir: Path,
    *,
    preserve_combined_xy_overlay: bool = False,
) -> tuple[int, int]:
    removed_files = 0
    for csv_path in sorted(rollout_dir.glob("mapf_timed_tracker_*.csv")):
        if csv_path.is_file():
            csv_path.unlink()
            removed_files += 1

    removed_directories = 0
    tracking_plots_dir = rollout_dir / "tracking_plots"
    if tracking_plots_dir.is_dir():
        if preserve_combined_xy_overlay:
            for plot_path in sorted(tracking_plots_dir.iterdir()):
                if plot_path.name == COMBINED_XY_OVERLAY_NAME and plot_path.is_file():
                    continue
                if plot_path.is_dir():
                    shutil.rmtree(plot_path)
                    removed_directories += 1
                else:
                    plot_path.unlink()
                    removed_files += 1
            if not any(tracking_plots_dir.iterdir()):
                tracking_plots_dir.rmdir()
                removed_directories += 1
        else:
            shutil.rmtree(tracking_plots_dir)
            removed_directories += 1
    elif tracking_plots_dir.exists():
        tracking_plots_dir.unlink()
        removed_files += 1

    return removed_files, removed_directories


class RandomizedWarehouseBatchRunner(Node):
    def __init__(self) -> None:
        if rclpy is None or String is None:
            raise RuntimeError("rclpy is required to run RandomizedWarehouseBatchRunner.")
        super().__init__("randomized_warehouse_batch_runner")

        self.declare_parameter("scene_root_dir", "")
        self.declare_parameter("continue", False)
        self.declare_parameter("continue_scene_id", "")
        self.declare_parameter("continue_rollout_id", 0)
        self.declare_parameter("max_rerun", 0)
        self.declare_parameter("scene_control_topic", "/carters_goal/batch_scene_control")
        self.declare_parameter("scene_ready_topic", "/carters_goal/batch_scene_ready")
        self.declare_parameter("scene_ready_timeout_sec", 60.0)
        self.declare_parameter("execution_timeout_sec", 600.0)
        self.declare_parameter("execution_start_timeout_sec", 300.0)
        self.declare_parameter("retry_cooldown_sec", 2.0)
        self.declare_parameter("status_topic", "/mapf_base/plan_execution_status")
        self.declare_parameter("save_traj_plot", False)

        self._scene_root_dir = str(self.get_parameter("scene_root_dir").value).strip()
        if not self._scene_root_dir:
            raise ValueError("scene_root_dir must be provided.")
        self._continue_enabled = _as_bool(self.get_parameter("continue").value)
        self._continue_scene_id = str(self.get_parameter("continue_scene_id").value).strip()
        self._continue_rollout_id = int(self.get_parameter("continue_rollout_id").value)
        self._max_rerun = max(int(self.get_parameter("max_rerun").value), 0)
        self._scene_ready_timeout_sec = max(
            float(self.get_parameter("scene_ready_timeout_sec").value),
            1.0,
        )
        self._execution_timeout_sec = max(
            float(self.get_parameter("execution_timeout_sec").value),
            1.0,
        )
        self._execution_start_timeout_sec = max(
            float(self.get_parameter("execution_start_timeout_sec").value),
            1.0,
        )
        self._retry_cooldown_sec = max(
            float(self.get_parameter("retry_cooldown_sec").value),
            0.0,
        )
        self._save_traj_plot = _as_bool(self.get_parameter("save_traj_plot").value)

        self._launch_options = self._read_launch_options()
        scene_control_topic = str(self.get_parameter("scene_control_topic").value)
        scene_ready_topic = str(self.get_parameter("scene_ready_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        self._scene_control_pub = self.create_publisher(String, scene_control_topic, 10)
        self._scene_ready_sub = self.create_subscription(
            String,
            scene_ready_topic,
            self._scene_ready_callback,
            10,
        )
        self._status_sub = self.create_subscription(
            String,
            status_topic,
            self._status_callback,
            20,
        )
        self._latest_scene_ready: dict[str, Any] | None = None
        self._latest_execution_status = ""

    def _read_launch_options(self) -> dict[str, str]:
        option_names = [
            "mapf_params_file",
            "mapf_costmap_params_file",
            "initial_pose_tf_params_file",
            "use_sim_time",
            "autostart",
            "mapf_planner",
            "record_velocity",
            "record_frequency_hz",
            "record_odom_topic_suffix",
            "record_cmd_vel_topic_suffix",
            "experiments_dir",
            "core_startup_delay",
            "lifecycle_manager_delay",
            "goal_publisher_delay",
            "run_tf_bridge",
            "run_initial_pose_tf",
            "run_plan_executor",
            "execution_backend",
            "plan_executor_delay",
        ]
        options: dict[str, str] = {}
        descriptor = (
            ParameterDescriptor(dynamic_typing=True)
            if ParameterDescriptor is not None
            else None
        )
        for name in option_names:
            if not self.has_parameter(name):
                self.declare_parameter(name, descriptor=descriptor)
            value = self.get_parameter(name).value
            if isinstance(value, bool):
                options[name] = "true" if value else "false"
            elif value is None:
                options[name] = ""
            else:
                options[name] = str(value)
        return options

    def _scene_ready_callback(self, msg: String) -> None:
        try:
            payload = json.loads(str(msg.data))
        except json.JSONDecodeError:
            self.get_logger().warn(f"Ignoring invalid scene-ready JSON: {msg.data!r}")
            return
        self._latest_scene_ready = payload

    def _status_callback(self, msg: String) -> None:
        self._latest_execution_status = str(msg.data).strip().lower()

    def run(self) -> int:
        scenes = discover_scene_bundles(self._scene_root_dir)
        batch_items = build_batch_items(
            scenes,
            continue_enabled=self._continue_enabled,
            continue_scene_id=self._continue_scene_id,
            continue_rollout_id=self._continue_rollout_id,
        )
        self.get_logger().info(
            f"Starting randomized warehouse batch: {len(batch_items)} rollouts "
            f"across {len({item.scene.scene_id for item in batch_items})} scenes."
        )

        failures: list[str] = []
        for batch_index, item in enumerate(batch_items, start=1):
            success = self._run_item_with_retries(item, batch_index, len(batch_items))
            if not success:
                failures.append(f"{item.scene.scene_id}/rollout_{item.rollout_id}")

        if failures:
            self.get_logger().error(
                "Randomized warehouse batch finished with failed rollouts: "
                + ", ".join(failures)
            )
            return 1
        self.get_logger().info("Randomized warehouse batch finished successfully.")
        return 0

    def _run_item_with_retries(self, item: BatchItem, batch_index: int, total: int) -> bool:
        attempts = self._max_rerun + 1
        for attempt in range(1, attempts + 1):
            self.get_logger().info(
                f"[{batch_index}/{total}] Running {item.scene.scene_id} rollout "
                f"{item.rollout_id}, attempt {attempt}/{attempts}."
            )
            rollout_dir = rollout_run_dir(
                self._launch_options.get("experiments_dir", ""),
                team_config_path=item.scene.team_config,
                rollout_id=item.rollout_id,
            )
            if not self._request_host_scene(item, attempt):
                if attempt >= attempts:
                    self._plot_failed_rollout(rollout_dir)
                continue
            result = self._run_single_rollout_launch(item, attempt)
            if result == "succeeded":
                if self._save_traj_plot:
                    self._plot_successful_rollout(rollout_dir)
                self._cleanup_successful_rollout(rollout_dir)
                return True
            self.get_logger().warn(
                f"{item.scene.scene_id} rollout {item.rollout_id} attempt {attempt} "
                f"ended with status {result!r}."
            )
            if attempt >= attempts:
                self._plot_failed_rollout(rollout_dir)
        return False

    def _request_host_scene(self, item: BatchItem, attempt: int) -> bool:
        self._latest_scene_ready = None
        request = {
            "scene_id": item.scene.scene_id,
            "rollout_id": int(item.rollout_id),
            "attempt": int(attempt),
        }
        deadline = time.monotonic() + self._scene_ready_timeout_sec
        next_publish = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish:
                msg = String()
                msg.data = json.dumps(request)
                self._scene_control_pub.publish(msg)
                next_publish = now + 0.5
            rclpy.spin_once(self, timeout_sec=0.05)
            ready = self._latest_scene_ready
            if not ready:
                continue
            if (
                ready.get("scene_id") == request["scene_id"]
                and int(ready.get("rollout_id", -1)) == request["rollout_id"]
                and int(ready.get("attempt", -1)) == request["attempt"]
            ):
                if ready.get("status") == "ready":
                    return True
                self.get_logger().error(
                    f"Host scene server rejected {request}: {ready.get('message', ready)}"
                )
                return False
        self.get_logger().error(f"Timed out waiting for host scene-ready ack for {request}.")
        return False

    def _run_single_rollout_launch(self, item: BatchItem, attempt: int) -> str:
        command = build_single_rollout_launch_command(
            scene=item.scene,
            rollout_id=item.rollout_id,
            launch_options=self._launch_options,
        )
        self._latest_execution_status = ""
        self.get_logger().info("Starting child launch: " + " ".join(command))
        env = os.environ.copy()
        env["FASTDDS_BUILTIN_TRANSPORTS"] = "UDPv4"
        env["RMW_FASTRTPS_USE_SHM"] = "0"
        process = subprocess.Popen(
            command,
            preexec_fn=os.setsid,
            env=env,
        )
        deadline = time.monotonic() + self._execution_timeout_sec
        startup_deadline = time.monotonic() + self._execution_start_timeout_sec
        execution_started = False
        result = "process_exited"
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
                status = self._latest_execution_status
                if status:
                    execution_started = True
                if status in {"succeeded", "failed"}:
                    result = status
                    break
                return_code = process.poll()
                if return_code is not None:
                    if execution_started:
                        result = f"process_exited:{return_code}"
                    else:
                        result = f"startup_process_exited:{return_code}"
                    break
                if not execution_started and time.monotonic() >= startup_deadline:
                    result = "startup_timeout"
                    break
                if time.monotonic() >= deadline:
                    result = "timeout"
                    break
        finally:
            self._terminate_process_group(process)
            if self._retry_cooldown_sec > 0.0:
                time.sleep(self._retry_cooldown_sec)
        return result

    def _terminate_process_group(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)
        except ProcessLookupError:
            return

    def _plot_failed_rollout(self, rollout_dir: Path) -> None:
        self._plot_rollout_tracking_logs(
            rollout_dir,
            combined_only=False,
            missing_log_level="warn",
        )

    def _plot_successful_rollout(self, rollout_dir: Path) -> None:
        self._plot_rollout_tracking_logs(
            rollout_dir,
            combined_only=True,
            missing_log_level="info",
        )

    def _plot_rollout_tracking_logs(
        self,
        rollout_dir: Path,
        *,
        combined_only: bool,
        missing_log_level: str,
    ) -> None:
        if not rollout_dir.exists():
            self.get_logger().warn(f"No rollout directory exists to plot: {rollout_dir}")
            return
        csv_paths = sorted(rollout_dir.glob("mapf_timed_tracker_*.csv"))
        if not csv_paths:
            log_message = f"No timed-tracker CSV logs found under {rollout_dir}."
            if missing_log_level == "info":
                self.get_logger().info(log_message)
            else:
                self.get_logger().warn(log_message)
            return
        output_dir = rollout_dir / "tracking_plots"
        command = [
            "ros2",
            "run",
            "carters_goal",
            "PlotMapfTrackingLogs",
            str(rollout_dir),
            "--pattern",
            "mapf_timed_tracker_*.csv",
            "--output-dir",
            str(output_dir),
            "--no-show",
        ]
        if combined_only:
            command.append("--combined-only")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            self.get_logger().warn(
                f"Failed to plot timed-tracker logs for {rollout_dir}: exit {result.returncode}"
            )

    def _cleanup_successful_rollout(self, rollout_dir: Path) -> None:
        if not rollout_dir.exists():
            return
        removed_files, removed_directories = cleanup_successful_rollout_artifacts(
            rollout_dir,
            preserve_combined_xy_overlay=self._save_traj_plot,
        )
        if removed_files or removed_directories:
            self.get_logger().info(
                f"Cleaned successful rollout diagnostics under {rollout_dir}: "
                f"removed {removed_files} file(s) and {removed_directories} directory/directories."
            )


def main() -> None:
    if rclpy is None:
        raise RuntimeError("rclpy is required to run RandomizedWarehouseBatchRunner.")
    rclpy.init()
    node: RandomizedWarehouseBatchRunner | None = None
    exit_code = 1
    try:
        node = RandomizedWarehouseBatchRunner()
        exit_code = node.run()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
