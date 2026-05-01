#!/usr/bin/env python3
"""Open one randomized warehouse rollout with its heterogeneous robot team.

Run with Isaac Sim's Python, for example:

  ./python.sh /path/to/mas-vln/tests/validate_randomized_warehouse_hetero_rollout.py scene_1 --rollout-id 3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SCENE_ROOT_DIR = REPO_ROOT / "experiments" / "randomized_warehouse"
DEFAULT_ROBOT_ROOT = "/World/Robots"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load one randomized warehouse rollout, spawn its robots, and enable ROS cmd_vel topics."
    )
    parser.add_argument(
        "scene",
        nargs="?",
        default="",
        help="Scene id under experiments/randomized_warehouse, bundle directory, or team_config.yaml path.",
    )
    parser.add_argument("--scene-id", default="", help="Scene bundle id under --scene-root-dir.")
    parser.add_argument(
        "--scene-root-dir",
        default=str(DEFAULT_SCENE_ROOT_DIR),
        help="Directory containing randomized warehouse scene bundles.",
    )
    parser.add_argument("--team-config-file", default="", help="Direct path to team_config.yaml.")
    parser.add_argument("--rollout-id", type=int, default=1, help="Rollout id to load.")
    parser.add_argument(
        "--pose",
        choices=("initial", "goal"),
        default="initial",
        help="Whether to place robots at initial or goal poses before holding the GUI open.",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument(
        "--no-ros-bridge",
        action="store_true",
        help="Spawn robots for visual inspection only; do not create ROS subscriptions/publishers.",
    )
    parser.add_argument(
        "--no-hold-open",
        action="store_true",
        help="Close after the warmup instead of keeping the Isaac Sim GUI open.",
    )
    parser.add_argument("--warmup-updates", type=int, default=60)
    parser.add_argument("--robot-root", default=DEFAULT_ROBOT_ROOT)
    parser.add_argument(
        "--rollout-control-topic",
        default="/carters_goal/rollout_control",
        help="PoseArray topic used by the internal bridge for rollout resets.",
    )
    parser.add_argument(
        "--rollout-reset-done-topic",
        default="/carters_goal/rollout_reset_done",
        help="Int32 acknowledgement topic published after rollout reset.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _resolve_team_config_path(args: argparse.Namespace) -> Path:
    if str(args.team_config_file).strip():
        return Path(args.team_config_file).expanduser().resolve()

    selection = str(args.scene_id or args.scene).strip()
    if not selection:
        raise ValueError("Pass a scene id, bundle directory, or --team-config-file.")
    selection_path = Path(selection).expanduser()
    if selection_path.exists():
        selection_path = selection_path.resolve()
        return selection_path / "team_config.yaml" if selection_path.is_dir() else selection_path

    return Path(args.scene_root_dir).expanduser().resolve() / selection / "team_config.yaml"


def _scene_usd_path(team_config: dict[str, Any], bundle_dir: Path) -> Path:
    usd_path = str(team_config.get("usd_path", "") or "").strip()
    return Path(usd_path).expanduser().resolve() if usd_path else bundle_dir / "scene.usd"


def _select_rollout(team_config: dict[str, Any], rollout_id: int) -> dict[str, Any]:
    for rollout in team_config.get("rollouts", []) or []:
        if int(rollout.get("id", -1)) == int(rollout_id):
            return rollout
    available = [rollout.get("id") for rollout in team_config.get("rollouts", []) or []]
    raise ValueError(f"Rollout id {rollout_id} was not found. Available ids: {available}")


def _pose_xyz_yaw(pose: dict[str, Any]) -> tuple[tuple[float, float, float], float]:
    return (
        (
            float(pose.get("x", 0.0)),
            float(pose.get("y", 0.0)),
            float(pose.get("z", 0.0)),
        ),
        float(pose.get("yaw", 0.0)),
    )


def _prim_name(name: str, model_id: str) -> str:
    clean_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip()).strip("_")
    return clean_name or model_id


def _open_stage(scene_usd_path: Path, sim_app, *, warmup_updates: int) -> None:
    import omni.usd

    opened = omni.usd.get_context().open_stage(str(scene_usd_path))
    if opened is False:
        raise RuntimeError(f"Failed to open scene USD: {scene_usd_path}")
    for _ in range(max(1, int(warmup_updates))):
        sim_app.update()


def _spawn_rollout_robots(
    *,
    rollout: dict[str, Any],
    pose_key: str,
    robot_root_path: str,
    sim_app,
    warmup_updates: int,
) -> tuple[list[str], list[Any]]:
    from isaacsim.storage.native import get_assets_root_path
    from isaac_sim.stage_bringups.runtime_utils import ensure_xform_path, get_stage
    from isaac_sim.stage_bringups.warehouse_randomized.robots import build_robot_adapter

    stage = get_stage()
    ensure_xform_path(robot_root_path)
    root = stage.GetPrimAtPath(robot_root_path)
    if root and root.IsValid():
        for child in list(root.GetChildren()):
            stage.RemovePrim(child.GetPath())

    assets_root_path = str(get_assets_root_path()).rstrip("/")
    prim_paths: list[str] = []
    adapters: list[Any] = []
    for index, robot in enumerate(rollout.get("robots", []) or [], start=1):
        model_id = str(robot.get("model", "") or "").strip()
        if not model_id:
            raise ValueError(f"Rollout {rollout.get('id')} robot {index} is missing model.")
        name = str(robot.get("name", "") or model_id).strip()
        adapter = build_robot_adapter(model_id)
        pose_xyz, yaw_rad = _pose_xyz_yaw(robot.get(f"{pose_key}_pose", {}) or {})
        prim_path = f"{robot_root_path}/{_prim_name(name, adapter.model_id)}"
        adapter.spawn_robot(
            assets_root_path=assets_root_path,
            prim_path=prim_path,
            robot_index=index,
            position_xyz=pose_xyz,
            yaw_deg=math.degrees(yaw_rad),
        )
        prim_paths.append(prim_path)
        adapters.append(adapter)

    for _ in range(max(1, int(warmup_updates))):
        sim_app.update()
    return prim_paths, adapters


def _initialize_ros_bridge(
    *,
    sim_app,
    rollout: dict[str, Any],
    robot_prim_paths: list[str],
    adapters: list[Any],
    rollout_control_topic: str,
    rollout_reset_done_topic: str,
):
    import omni.timeline
    from isaac_sim.stage_bringups.warehouse_randomized.ros_bridge import InternalIsaacRosBridge

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(8):
        sim_app.update()

    controllers = []
    for robot, adapter, prim_path in zip(rollout.get("robots", []) or [], adapters, robot_prim_paths):
        controllers.append(
            adapter.initialize_runtime_controller(
                sim_app=sim_app,
                namespace=str(robot.get("name", adapter.model_id)),
                root_prim_path=prim_path,
            )
        )

    return InternalIsaacRosBridge(
        sim_app=sim_app,
        robot_controllers=controllers,
        rollout_control_topic=rollout_control_topic,
        rollout_reset_done_topic=rollout_reset_done_topic,
    )


def _hold_runtime(sim_app, ros_bridge, namespaces: list[str]) -> None:
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    print("\nHeterogeneous rollout is loaded. ROS cmd_vel topics are active for:")
    if ros_bridge is not None:
        print("  " + ", ".join(f"/{namespace}/cmd_vel" for namespace in namespaces))
    print("Close Isaac Sim or press Ctrl+C to exit.")
    try:
        while sim_app.is_running():
            if ros_bridge is not None:
                ros_bridge.spin_once()
            sim_app.update()
            if ros_bridge is not None:
                ros_bridge.step(float(timeline.get_current_time()))
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass


def main() -> int:
    args = _parse_args()
    ros_bridge = None
    sim_app = None
    try:
        team_config_path = _resolve_team_config_path(args)
        if not team_config_path.exists():
            raise FileNotFoundError(f"team_config.yaml does not exist: {team_config_path}")
        team_config = _load_yaml(team_config_path)
        scene_usd_path = _scene_usd_path(team_config, team_config_path.parent)
        if not scene_usd_path.exists():
            raise FileNotFoundError(f"scene.usd does not exist: {scene_usd_path}")
        rollout = _select_rollout(team_config, int(args.rollout_id))

        from isaac_sim.stage_bringups.runtime_utils import maybe_start_sim_app

        sim_app = maybe_start_sim_app(
            headless=bool(args.headless),
            enable_ros2_bridge=not bool(args.no_ros_bridge),
        )
        _open_stage(scene_usd_path, sim_app, warmup_updates=int(args.warmup_updates))
        robot_prim_paths, adapters = _spawn_rollout_robots(
            rollout=rollout,
            pose_key=str(args.pose),
            robot_root_path=str(args.robot_root),
            sim_app=sim_app,
            warmup_updates=int(args.warmup_updates),
        )
        if not args.no_ros_bridge:
            ros_bridge = _initialize_ros_bridge(
                sim_app=sim_app,
                rollout=rollout,
                robot_prim_paths=robot_prim_paths,
                adapters=adapters,
                rollout_control_topic=str(args.rollout_control_topic),
                rollout_reset_done_topic=str(args.rollout_reset_done_topic),
            )

        print(f"[OK] Loaded scene {team_config.get('scene_id', scene_usd_path.parent.name)} rollout {rollout.get('id')}.")
        if not args.headless and not args.no_hold_open:
            namespaces = [str(robot.get("name", "")) for robot in rollout.get("robots", []) or []]
            _hold_runtime(sim_app, ros_bridge, namespaces)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        if ros_bridge is not None:
            ros_bridge.shutdown()
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
