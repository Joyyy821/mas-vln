#!/usr/bin/env python3
"""
Isaac Sim host-side batch scene server for randomized warehouse rollout collection.

Run this from Isaac Sim Python on the host before launching
carters_goal isaac_ros_mapf_rollouts.launch.py inside Docker. The ROS launch
publishes JSON scene requests; this script loads the requested scene USD, clears
and respawns the rollout-specific heterogeneous robot team, starts system-ROS
cmd_vel/odom/tf helpers, then acknowledges readiness.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaac_sim.stage_bringups.build_stage_warehouse_carters import (  # noqa: E402
    ENV_CMD_ANGULAR_SIGN,
    MAPS_DIR,
    IsaacSystemRosRobotBridge,
    _clear_prim_children,
    _disable_lidar_debug_visualization,
    _ensure_global_ros2_clock_graph,
    _ensure_xform_path,
    _maybe_start_sim_app,
    _open_stage,
    _set_isaac_namespace,
    team_config_utils,
)

BATCH_HELPER_PATH = SCRIPT_DIR / "isaac_batch_scene_helper.py"


class IsaacBatchSceneControlBridge:
    def __init__(
        self,
        *,
        scene_control_topic: str,
        scene_ready_topic: str,
    ) -> None:
        self._scene_control_topic = scene_control_topic
        self._scene_ready_topic = scene_ready_topic
        self._pending_request: dict[str, Any] | None = None
        self._last_ready_payload: dict[str, Any] | None = None
        self._helper_process: subprocess.Popen | None = None
        self._client_socket: socket.socket | None = None
        self._recv_buffer = ""
        self._helper_failure_logged = False

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(("127.0.0.1", 0))
        self._server_socket.listen(1)
        self._server_socket.setblocking(False)
        self._server_host, self._server_port = self._server_socket.getsockname()

        self._start_helper_process()
        print(
            "[OK] Isaac batch scene bridge listening on "
            f"{self._server_host}:{self._server_port}; ROS topics "
            f"{scene_control_topic} -> {scene_ready_topic}"
        )

    def _start_helper_process(self) -> None:
        if not BATCH_HELPER_PATH.exists():
            raise FileNotFoundError(f"Batch scene helper not found: {BATCH_HELPER_PATH}")

        helper_command = (
            "source /opt/ros/humble/setup.bash && "
            f"exec /usr/bin/python3 {shlex.quote(str(BATCH_HELPER_PATH))} "
            f"--host {shlex.quote(self._server_host)} "
            f"--port {self._server_port} "
            f"--scene-control-topic {shlex.quote(self._scene_control_topic)} "
            f"--scene-ready-topic {shlex.quote(self._scene_ready_topic)}"
        )
        self._helper_process = subprocess.Popen(
            ["/bin/bash", "-lc", helper_command],
            cwd=str(REPO_ROOT),
            env=self._build_helper_env(),
        )

    def _build_helper_env(self) -> dict[str, str]:
        helper_env: dict[str, str] = {}
        for key in (
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "TERM",
            "LANG",
            "LC_ALL",
            "DISPLAY",
            "XAUTHORITY",
            "ROS_DOMAIN_ID",
            "ROS_LOCALHOST_ONLY",
            "RMW_IMPLEMENTATION",
            "FASTRTPS_DEFAULT_PROFILES_FILE",
            "CYCLONEDDS_URI",
            "ROS_AUTOMATIC_DISCOVERY_RANGE",
        ):
            value = os.environ.get(key)
            if value:
                helper_env[key] = value
        helper_env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        helper_env["PYTHONNOUSERSITE"] = "1"
        return helper_env

    def spin_once(self) -> None:
        self._check_helper_process()
        self._accept_client_if_needed()
        self._read_client_messages()

    def pop_pending_request(self) -> dict[str, Any] | None:
        request = self._pending_request
        self._pending_request = None
        return request

    def publish_ready(self, payload: dict[str, Any]) -> None:
        ready_payload = {"type": "scene_ready", **payload}
        self._last_ready_payload = ready_payload
        self._send_message(ready_payload)

    def _request_key(self, payload: dict[str, Any]) -> tuple[str, int, int]:
        return (
            str(payload.get("scene_id", "")),
            int(payload.get("rollout_id", -1)),
            int(payload.get("attempt", -1)),
        )

    def _check_helper_process(self) -> None:
        if self._helper_process is None or self._helper_failure_logged:
            return
        return_code = self._helper_process.poll()
        if return_code is None:
            return
        self._helper_failure_logged = True
        print(
            "[WARN] Isaac batch scene helper exited early with code "
            f"{return_code}. Batch scene requests will not work until it reconnects."
        )

    def _accept_client_if_needed(self) -> None:
        if self._client_socket is not None:
            return
        try:
            client_socket, _ = self._server_socket.accept()
        except BlockingIOError:
            return
        client_socket.setblocking(False)
        self._client_socket = client_socket
        print("[OK] Batch scene helper connected to Isaac Sim.")

    def _read_client_messages(self) -> None:
        if self._client_socket is None:
            return
        try:
            chunk = self._client_socket.recv(65536)
        except BlockingIOError:
            return
        except OSError:
            self._close_client_socket()
            return
        if not chunk:
            self._close_client_socket()
            return

        self._recv_buffer += chunk.decode("utf-8")
        while "\n" in self._recv_buffer:
            line, self._recv_buffer = self._recv_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self._handle_client_message(line)

    def _handle_client_message(self, line: str) -> None:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Failed to decode batch scene request '{line}': {exc}")
            return
        if payload.get("type") != "scene_request":
            return

        if self._last_ready_payload and self._request_key(payload) == self._request_key(
            self._last_ready_payload
        ):
            self._send_message(self._last_ready_payload)
            return
        if self._pending_request and self._request_key(payload) == self._request_key(
            self._pending_request
        ):
            return
        self._pending_request = payload

    def _send_message(self, message: dict[str, Any]) -> None:
        if self._client_socket is None:
            return
        payload = (json.dumps(message) + "\n").encode("utf-8")
        try:
            self._client_socket.sendall(payload)
        except BlockingIOError:
            return
        except OSError:
            self._close_client_socket()

    def _close_client_socket(self) -> None:
        if self._client_socket is None:
            return
        try:
            self._client_socket.close()
        finally:
            self._client_socket = None
            self._recv_buffer = ""
            print("[WARN] Batch scene helper disconnected from Isaac Sim.")

    def shutdown(self) -> None:
        self._close_client_socket()
        try:
            self._server_socket.close()
        except Exception:
            pass
        if self._helper_process is not None and self._helper_process.poll() is None:
            self._helper_process.terminate()
            try:
                self._helper_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._helper_process.kill()


class RandomizedWarehouseBatchSceneServer:
    def __init__(
        self,
        *,
        sim_app,
        scene_root_dir: Path,
        warmup_updates: int,
    ) -> None:
        self._sim_app = sim_app
        self._scene_root_dir = scene_root_dir.expanduser().resolve()
        self._warmup_updates = max(int(warmup_updates), 1)
        self._robot_ros_bridge: IsaacSystemRosRobotBridge | None = None

    def close(self) -> None:
        self._shutdown_robot_bridge()

    def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        scene_id = str(request.get("scene_id", "")).strip()
        rollout_id = int(request.get("rollout_id", 0))
        attempt = int(request.get("attempt", 0))
        if not scene_id or rollout_id <= 0 or attempt <= 0:
            return {
                "scene_id": scene_id,
                "rollout_id": rollout_id,
                "attempt": attempt,
                "status": "error",
                "message": f"Invalid scene request: {request}",
            }

        try:
            robot_infos = self._load_scene_and_rollout(scene_id, rollout_id)
        except Exception as exc:
            print(
                f"[ERROR] Failed to prepare {scene_id} rollout {rollout_id}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            return {
                "scene_id": scene_id,
                "rollout_id": rollout_id,
                "attempt": attempt,
                "status": "error",
                "message": str(exc),
            }

        return {
            "scene_id": scene_id,
            "rollout_id": rollout_id,
            "attempt": attempt,
            "status": "ready",
            "robot_namespaces": [info["name"] for info in robot_infos],
            "robot_models": [info["model"] for info in robot_infos],
        }

    def spin_robot_bridge_once(self) -> None:
        if self._robot_ros_bridge is not None:
            self._robot_ros_bridge.spin_once()

    def step_robot_bridge(self, sim_time_sec: float) -> None:
        if self._robot_ros_bridge is not None:
            self._robot_ros_bridge.step(sim_time_sec)

    def _load_scene_and_rollout(self, scene_id: str, rollout_id: int) -> list[dict[str, Any]]:
        scene_dir = self._scene_root_dir / scene_id
        scene_usd = scene_dir / "scene.usd"
        team_config_file = scene_dir / "team_config.yaml"
        if not scene_usd.is_file():
            raise FileNotFoundError(f"Scene USD not found: {scene_usd}")
        if not team_config_file.is_file():
            raise FileNotFoundError(f"Team config not found: {team_config_file}")

        print(f"[INFO] Preparing {scene_id} rollout {rollout_id}")
        self._shutdown_robot_bridge()
        self._stop_timeline()
        self._settle_stage_updates()
        print(f"[INFO] Opening scene USD: {scene_usd}")
        _open_stage(str(scene_usd))
        self._settle_stage_updates()
        _ensure_xform_path("/World")
        _ensure_xform_path("/World/Robots")
        _clear_prim_children("/World/Robots")
        self._settle_stage_updates()

        robot_infos = self._spawn_rollout_robots(str(team_config_file), rollout_id)
        self._ensure_global_ros2_clock_graph_once("/clock")
        self._settle_stage_updates()
        self._start_timeline()
        self._settle_stage_updates()

        robot_controllers = []
        for robot_info in robot_infos:
            print(
                f"[INFO] Initializing articulation controller for "
                f"{robot_info['name']} at {robot_info['prim_path']}"
            )
            robot_controllers.append(
                robot_info["adapter"].initialize_runtime_controller(
                    sim_app=self._sim_app,
                    namespace=robot_info["name"],
                    root_prim_path=robot_info["prim_path"],
                )
            )
        self._robot_ros_bridge = IsaacSystemRosRobotBridge(
            sim_app=self._sim_app,
            robot_controllers=robot_controllers,
            cmd_angular_sign=ENV_CMD_ANGULAR_SIGN,
        )
        for _ in range(8):
            self._sim_app.update()
            self.spin_robot_bridge_once()
        print(
            f"[OK] Prepared {scene_id} rollout {rollout_id}: "
            + ", ".join(info["name"] for info in robot_infos)
        )
        return robot_infos

    def _spawn_rollout_robots(self, team_config_file: str, rollout_id: int) -> list[dict[str, Any]]:
        from isaacsim.storage.native import get_assets_root_path
        from isaac_sim.stage_bringups.warehouse_randomized.robots import build_robot_adapter

        assets_root_path = get_assets_root_path()
        team_config = team_config_utils.load_team_config(
            team_config_file,
            maps_dir=MAPS_DIR,
            rollout_id=rollout_id,
        )
        print(
            f"[INFO] Selected rollout {team_config['rollout_id']} with "
            f"{team_config['agent_num']} robots: {team_config['robot_namespaces']}"
        )

        robot_infos: list[dict[str, Any]] = []
        for index, robot in enumerate(team_config["robots"], start=1):
            robot_name = str(robot["name"]).strip("/") or f"robot_{index}"
            robot_model = str(robot.get("model") or "nova_carter").strip() or "nova_carter"
            adapter = build_robot_adapter(robot_model)
            robot_prim = f"/World/Robots/{_safe_robot_prim_name(robot_name, index)}"
            initial_pose = team_config_utils.pose_array_to_pose_dict(robot["initial_pose"])
            print(
                f"[INFO] Spawning robot {index}: name={robot_name}, "
                f"model={robot_model}, prim={robot_prim}"
            )
            adapter.spawn_robot(
                assets_root_path=assets_root_path,
                prim_path=robot_prim,
                robot_index=index,
                position_xyz=(
                    initial_pose["x"],
                    initial_pose["y"],
                    initial_pose["z"],
                ),
                yaw_deg=initial_pose["yaw"] * 180.0 / 3.141592653589793,
            )
            _set_isaac_namespace(robot_prim, robot_name)
            if robot_model.lower() == "jackal":
                _disable_lidar_debug_visualization(robot_prim)
            robot_infos.append(
                {
                    "name": robot_name,
                    "model": robot_model,
                    "prim_path": robot_prim,
                    "adapter": adapter,
                }
            )
        return robot_infos

    def _shutdown_robot_bridge(self) -> None:
        if self._robot_ros_bridge is None:
            return
        self._robot_ros_bridge.shutdown()
        self._robot_ros_bridge = None
        gc.collect()

    def _settle_stage_updates(self) -> None:
        for _ in range(self._warmup_updates):
            self._sim_app.update()

    def _start_timeline(self) -> None:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

    def _stop_timeline(self) -> None:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()

    def _ensure_global_ros2_clock_graph_once(self, clock_topic: str = "/clock") -> None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        graph_prim = stage.GetPrimAtPath("/World/ROS2ClockGraph")
        if graph_prim and graph_prim.IsValid():
            print("[OK] Reusing existing global ROS2 clock publisher graph at /World/ROS2ClockGraph")
            return
        _ensure_global_ros2_clock_graph(clock_topic)


def _safe_robot_prim_name(robot_name: str, index: int) -> str:
    safe_name = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in str(robot_name).strip("/")
    )
    return safe_name or f"robot_{index}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-root-dir",
        default=str(REPO_ROOT / "experiments" / "randomized_warehouse"),
        help="Directory containing scene_<n> randomized warehouse bundles.",
    )
    parser.add_argument(
        "--scene-control-topic",
        default="/carters_goal/batch_scene_control",
    )
    parser.add_argument(
        "--scene-ready-topic",
        default="/carters_goal/batch_scene_ready",
    )
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=10,
        help="Isaac app updates used after loading/spawning before acknowledgement.",
    )
    parser.add_argument("--gui", action="store_true", help="Run Isaac Sim with GUI.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sim_app = _maybe_start_sim_app(headless=not args.gui)
    scene_bridge: IsaacBatchSceneControlBridge | None = None
    scene_server: RandomizedWarehouseBatchSceneServer | None = None
    for _ in range(10):
        sim_app.update()

    try:
        scene_bridge = IsaacBatchSceneControlBridge(
            scene_control_topic=args.scene_control_topic,
            scene_ready_topic=args.scene_ready_topic,
        )
        scene_server = RandomizedWarehouseBatchSceneServer(
            sim_app=sim_app,
            scene_root_dir=Path(args.scene_root_dir),
            warmup_updates=args.warmup_updates,
        )

        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        print("[OK] Randomized warehouse batch scene server is ready.")
        while sim_app.is_running():
            scene_bridge.spin_once()
            request = scene_bridge.pop_pending_request()
            if request is not None:
                ready_payload = scene_server.process_request(request)
                scene_bridge.publish_ready(ready_payload)

            scene_server.spin_robot_bridge_once()
            sim_app.update()
            scene_server.step_robot_bridge(float(timeline.get_current_time()))
    except Exception:
        import traceback

        print("\n[ERROR] run_randomized_warehouse_batch_server.py failed:")
        traceback.print_exc()
        raise
    finally:
        if scene_server is not None:
            scene_server.close()
        if scene_bridge is not None:
            scene_bridge.shutdown()
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    main()
