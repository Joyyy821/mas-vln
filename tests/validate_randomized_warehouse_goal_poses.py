#!/usr/bin/env python3
"""Open a randomized warehouse bundle and place robots at sampled goal poses.

Run with Isaac Sim's Python, for example:

  ./python.sh /path/to/mas-vln/tests/validate_randomized_warehouse_goal_poses.py scene_1
  ./python.sh /path/to/mas-vln/tests/validate_randomized_warehouse_goal_poses.py --list-scenes
  ./python.sh /path/to/mas-vln/tests/validate_randomized_warehouse_goal_poses.py --scene-id scene_1 --all-rollouts
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SCENE_ROOT_DIR = REPO_ROOT / "experiments" / "randomized_warehouse"
DEFAULT_ROBOT_ROOT = "/World/Robots"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one randomized warehouse scene bundle and move its robots to rollout goal poses."
        )
    )
    parser.add_argument(
        "scene",
        nargs="?",
        default="",
        help=(
            "Scene id under --scene-root-dir, bundle directory, or team_config.yaml path. "
            "Equivalent to --scene-id for simple scene ids."
        ),
    )
    parser.add_argument(
        "--scene-id",
        default="",
        help="Scene bundle directory name under --scene-root-dir.",
    )
    parser.add_argument(
        "--scene-root-dir",
        default=str(DEFAULT_SCENE_ROOT_DIR),
        help="Directory containing randomized warehouse scene bundle folders.",
    )
    parser.add_argument(
        "--team-config-file",
        default="",
        help="Direct path to a bundle team_config.yaml.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List scene ids with team_config.yaml under --scene-root-dir and exit.",
    )
    parser.add_argument(
        "--rollout-id",
        type=int,
        default=1,
        help="Rollout id to place. Ignored when --all-rollouts is set.",
    )
    parser.add_argument(
        "--all-rollouts",
        action="store_true",
        help="Cycle through every rollout in the selected team config.",
    )
    parser.add_argument(
        "--seconds-per-rollout",
        type=float,
        default=2.0,
        help="GUI dwell time after placing each rollout when --all-rollouts is set.",
    )
    parser.add_argument(
        "--pre-cycle-delay-sec",
        type=float,
        default=8.0,
        help=(
            "GUI delay before cycling --all-rollouts so the camera can be adjusted. "
            "Set to 0 to skip."
        ),
    )
    parser.add_argument(
        "--pre-cycle-updates",
        type=int,
        default=120,
        help="Extra SimulationApp updates before the --all-rollouts pre-cycle delay.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim headless and close after validation.",
    )
    parser.add_argument(
        "--no-hold-open",
        action="store_true",
        help="Close Isaac Sim after placement instead of keeping the GUI open.",
    )
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=60,
        help="SimulationApp update count after opening the scene and after each placement.",
    )
    parser.add_argument(
        "--robot-root",
        default=DEFAULT_ROBOT_ROOT,
        help="Stage prim containing robot root prims.",
    )
    parser.add_argument(
        "--no-spawn-missing-robots",
        action="store_true",
        help="Fail if robot prims are absent instead of spawning robots from the team config model.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _discover_scene_ids(scene_root_dir: Path) -> list[str]:
    if not scene_root_dir.exists():
        return []
    scene_ids = [
        path.name
        for path in scene_root_dir.iterdir()
        if path.is_dir() and (path / "team_config.yaml").exists()
    ]
    return sorted(scene_ids)


def _print_scene_ids(scene_root_dir: Path) -> None:
    scene_ids = _discover_scene_ids(scene_root_dir)
    print(f"Scene root: {scene_root_dir}")
    if not scene_ids:
        print("No randomized warehouse scene bundles with team_config.yaml were found.")
        return
    for scene_id in scene_ids:
        print(f"  {scene_id}")


def _resolve_team_config_path(args: argparse.Namespace) -> Path:
    scene_root_dir = Path(args.scene_root_dir).expanduser().resolve()
    if str(args.team_config_file).strip():
        return Path(args.team_config_file).expanduser().resolve()

    selection = str(args.scene_id or args.scene).strip()
    if not selection:
        raise ValueError("Select a scene with a positional scene id, --scene-id, or --team-config-file.")

    selection_path = Path(selection).expanduser()
    if selection_path.exists():
        selection_path = selection_path.resolve()
        if selection_path.is_dir():
            return selection_path / "team_config.yaml"
        return selection_path

    return scene_root_dir / selection / "team_config.yaml"


def _scene_usd_path(team_config: dict[str, Any], bundle_dir: Path) -> Path:
    usd_path = str(team_config.get("usd_path", "") or "").strip()
    if usd_path:
        return Path(usd_path).expanduser().resolve()
    return bundle_dir / "scene.usd"


def _selected_rollouts(team_config: dict[str, Any], *, rollout_id: int, all_rollouts: bool) -> list[dict[str, Any]]:
    rollouts = list(team_config.get("rollouts", []) or [])
    if not rollouts:
        raise ValueError("team_config.yaml does not contain any rollouts.")
    if all_rollouts:
        return rollouts

    for rollout in rollouts:
        if int(rollout.get("id", -1)) == int(rollout_id):
            return [rollout]
    available_ids = [rollout.get("id") for rollout in rollouts]
    raise ValueError(f"Rollout id {rollout_id} was not found. Available ids: {available_ids}")


def _robot_model_ids(team_config: dict[str, Any], robot_count: int) -> list[str]:
    robot_models = [str(model).strip() for model in team_config.get("robot_models", []) if str(model).strip()]
    if not robot_models:
        robot_model = str(team_config.get("robot_model", "nova_carter") or "nova_carter").strip()
        robot_models = [robot_model]
    if len(robot_models) == 1:
        robot_models = robot_models * robot_count
    if len(robot_models) < robot_count:
        robot_models.extend([robot_models[-1]] * (robot_count - len(robot_models)))
    return robot_models[:robot_count]


def _pose_xyz_yaw(pose: dict[str, Any]) -> tuple[tuple[float, float, float], float]:
    return (
        (
            float(pose.get("x", 0.0)),
            float(pose.get("y", 0.0)),
            float(pose.get("z", 0.0)),
        ),
        float(pose.get("yaw", 0.0)),
    )


def _team_config_sampling_inflation_radius_m(team_config: dict[str, Any]) -> float:
    validation = team_config.get("validation") or {}
    rollout_sampling = validation.get("rollout_sampling") or {}
    try:
        return max(0.0, float(rollout_sampling.get("inflation_radius_m", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _resolve_config_path(path_value: str, base_dir: Path) -> Path:
    path = Path(str(path_value or "").strip()).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _validate_goal_poses(
    team_config: dict[str, Any],
    rollouts: Sequence[dict[str, Any]],
    team_config_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    from isaac_sim.goal_generator.object_goal_sampler_utils import OccupancyMap

    nav2_map_value = str((team_config.get("environment") or {}).get("nav2_map", "") or "")
    nav2_map_path = (
        _resolve_config_path(nav2_map_value, team_config_dir)
        if nav2_map_value.strip()
        else None
    )
    occupancy_map = None
    inflated_free_mask = None
    sampling_inflation_radius_m = _team_config_sampling_inflation_radius_m(team_config)
    if nav2_map_path is not None and nav2_map_path.exists():
        occupancy_map = OccupancyMap.load(str(nav2_map_path), treat_unknown_as_occupied=True)
        inflated_free_mask = ~occupancy_map.inflate_occupied_mask(sampling_inflation_radius_m)

    reports: list[dict[str, Any]] = []
    pairwise_violations: list[dict[str, Any]] = []
    ok = True
    for rollout in rollouts:
        robots = list(rollout.get("robots", []) or [])
        goal_xy_values: list[tuple[str, np.ndarray]] = []
        for robot in robots:
            pose = robot.get("goal_pose", {}) or {}
            position_xyz, yaw_rad = _pose_xyz_yaw(pose)
            finite = all(math.isfinite(value) for value in (*position_xyz, yaw_rad))
            in_map = None
            is_free = None
            is_inflated_free = None
            grid_rc = None
            if occupancy_map is not None and finite:
                in_map = occupancy_map.contains_world_xy(position_xyz[0], position_xyz[1])
                if in_map:
                    row, col = occupancy_map.world_to_grid(position_xyz[0], position_xyz[1])
                    grid_rc = [int(row), int(col)]
                    is_free = bool(occupancy_map.free_mask[row, col])
                    is_inflated_free = bool(inflated_free_mask[row, col])
                else:
                    is_free = False
                    is_inflated_free = False
            goal_xy_values.append((str(robot.get("name", "")), np.array(position_xyz[:2], dtype=float)))
            robot_ok = bool(finite and (is_free is not False) and (is_inflated_free is not False))
            ok = ok and robot_ok
            reports.append(
                {
                    "rollout_id": rollout.get("id"),
                    "robot": robot.get("name", ""),
                    "goal_pose": {"x": position_xyz[0], "y": position_xyz[1], "z": position_xyz[2], "yaw": yaw_rad},
                    "finite": finite,
                    "in_nav2_map": in_map,
                    "nav2_grid_rc": grid_rc,
                    "nav2_free": is_free,
                    "sampling_inflation_radius_m": sampling_inflation_radius_m,
                    "nav2_inflated_free": is_inflated_free,
                    "ok": robot_ok,
                }
            )

        for index, (first_name, first_xy) in enumerate(goal_xy_values):
            for second_name, second_xy in goal_xy_values[index + 1 :]:
                distance_m = float(np.linalg.norm(first_xy - second_xy))
                if distance_m < 1.0:
                    ok = False
                    pairwise_violations.append(
                        {
                            "rollout_id": rollout.get("id"),
                            "first_robot": first_name,
                            "second_robot": second_name,
                            "distance_m": distance_m,
                            "min_distance_m": 1.0,
                        }
                    )
    return reports, pairwise_violations, ok


def _open_stage(scene_usd_path: Path, sim_app, *, warmup_updates: int) -> None:
    import omni.usd

    context = omni.usd.get_context()
    opened = context.open_stage(str(scene_usd_path))
    if opened is False:
        raise RuntimeError(f"Failed to open scene USD: {scene_usd_path}")
    for _ in range(max(1, int(warmup_updates))):
        sim_app.update()


def _robot_prim_paths(robot_root_path: str) -> list[str]:
    from isaac_sim.stage_bringups.runtime_utils import get_stage

    stage = get_stage()
    root = stage.GetPrimAtPath(robot_root_path)
    if not root or not root.IsValid():
        return []
    return sorted(
        [child.GetPath().pathString for child in root.GetChildren() if child.IsValid()],
        key=lambda path: path.lower(),
    )


def _ensure_robot_prims(
    *,
    robot_count: int,
    robot_models: Sequence[str],
    first_rollout: dict[str, Any],
    robot_root_path: str,
    spawn_missing_robots: bool,
) -> list[str]:
    existing_paths = _robot_prim_paths(robot_root_path)
    if len(existing_paths) >= robot_count:
        return existing_paths[:robot_count]
    if not spawn_missing_robots:
        raise RuntimeError(
            f"Found {len(existing_paths)} robot prims under {robot_root_path}, expected {robot_count}."
        )

    from isaacsim.storage.native import get_assets_root_path
    from isaac_sim.stage_bringups.runtime_utils import ensure_xform_path
    from isaac_sim.stage_bringups.warehouse_randomized.robots import build_robot_adapter

    assets_root_path = str(get_assets_root_path()).rstrip("/")
    ensure_xform_path(robot_root_path)
    robots = list(first_rollout.get("robots", []) or [])
    spawned_paths = list(existing_paths)
    for robot_index in range(len(existing_paths), robot_count):
        robot = robots[robot_index]
        position_xyz, yaw_rad = _pose_xyz_yaw(robot.get("goal_pose", {}) or {})
        model_id = robot_models[robot_index]
        adapter = build_robot_adapter(model_id)
        prim_path = f"{robot_root_path}/{adapter.model_id}_{robot_index + 1}"
        adapter.spawn_robot(
            assets_root_path=assets_root_path,
            prim_path=prim_path,
            robot_index=robot_index + 1,
            position_xyz=position_xyz,
            yaw_deg=math.degrees(yaw_rad),
        )
        spawned_paths.append(prim_path)
    return spawned_paths


def _place_rollout_goal_poses(
    *,
    rollout: dict[str, Any],
    robot_prim_paths: Sequence[str],
    sim_app,
    warmup_updates: int,
) -> None:
    from isaac_sim.stage_bringups.runtime_utils import set_xform_pose

    robots = list(rollout.get("robots", []) or [])
    if len(robots) > len(robot_prim_paths):
        raise RuntimeError(
            f"Rollout {rollout.get('id')} has {len(robots)} robots, "
            f"but only {len(robot_prim_paths)} robot prims are available."
        )

    for robot, prim_path in zip(robots, robot_prim_paths):
        position_xyz, yaw_rad = _pose_xyz_yaw(robot.get("goal_pose", {}) or {})
        set_xform_pose(prim_path, position_xyz, yaw_deg=math.degrees(yaw_rad))

    for _ in range(max(1, int(warmup_updates))):
        sim_app.update()


def _print_validation_report(
    reports: Sequence[dict[str, Any]],
    pairwise_violations: Sequence[dict[str, Any]],
    *,
    valid: bool,
) -> None:
    print("\nGoal pose validation:")
    for report in reports:
        map_text = "map=unchecked"
        if report["in_nav2_map"] is not None:
            map_text = (
                f"map={report['in_nav2_map']} raw_free={report['nav2_free']} "
                f"inflated_free={report['nav2_inflated_free']} "
                f"inflation={report['sampling_inflation_radius_m']:.3f}m "
                f"grid={report['nav2_grid_rc']}"
            )
        pose = report["goal_pose"]
        print(
            f"  rollout={report['rollout_id']} robot={report['robot']} "
            f"x={pose['x']:.3f} y={pose['y']:.3f} yaw={pose['yaw']:.3f} "
            f"finite={report['finite']} {map_text} ok={report['ok']}"
        )
    for violation in pairwise_violations:
        print(
            f"  rollout={violation['rollout_id']} pairwise_goal_distance "
            f"{violation['first_robot']}<->{violation['second_robot']} "
            f"distance={violation['distance_m']:.3f}m "
            f"min={violation['min_distance_m']:.3f}m ok=False"
        )
    print(f"Overall goal-pose validity: {'OK' if valid else 'FAILED'}")


def _hold_gui(sim_app) -> None:
    print("\nGUI is open with robots at the final selected rollout goal poses.")
    print("Close the Isaac Sim window or press Ctrl+C in this terminal to exit.")
    try:
        while sim_app.is_running():
            sim_app.update()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass


def _pre_cycle_gui_pause(sim_app, *, update_count: int, delay_sec: float) -> None:
    update_count = max(0, int(update_count))
    delay_sec = max(0.0, float(delay_sec))
    if update_count <= 0 and delay_sec <= 0.0:
        return

    print(
        "\nAdjust the Isaac Sim camera now; rollout goal-pose cycling will start "
        f"after {update_count} updates and {delay_sec:.1f}s."
    )
    try:
        for _ in range(update_count):
            if not sim_app.is_running():
                return
            sim_app.update()
            time.sleep(0.01)

        deadline = time.monotonic() + delay_sec
        while sim_app.is_running() and time.monotonic() < deadline:
            sim_app.update()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass


def main() -> int:
    args = _parse_args()
    scene_root_dir = Path(args.scene_root_dir).expanduser().resolve()
    if args.list_scenes:
        _print_scene_ids(scene_root_dir)
        return 0

    try:
        team_config_path = _resolve_team_config_path(args)
        if not team_config_path.exists():
            raise FileNotFoundError(f"team_config.yaml does not exist: {team_config_path}")
        bundle_dir = team_config_path.parent
        team_config = _load_yaml(team_config_path)
        scene_usd_path = _scene_usd_path(team_config, bundle_dir)
        if not scene_usd_path.exists():
            raise FileNotFoundError(f"scene.usd does not exist: {scene_usd_path}")

        rollouts = _selected_rollouts(
            team_config,
            rollout_id=int(args.rollout_id),
            all_rollouts=bool(args.all_rollouts),
        )
        reports, pairwise_violations, valid = _validate_goal_poses(
            team_config,
            rollouts,
            bundle_dir,
        )
        _print_validation_report(reports, pairwise_violations, valid=valid)

        from isaac_sim.stage_bringups.runtime_utils import maybe_start_sim_app

        sim_app = maybe_start_sim_app(headless=bool(args.headless))
        try:
            _open_stage(scene_usd_path, sim_app, warmup_updates=int(args.warmup_updates))
            first_rollout = rollouts[0]
            robot_count = len(first_rollout.get("robots", []) or [])
            robot_models = _robot_model_ids(team_config, robot_count)
            robot_prim_paths = _ensure_robot_prims(
                robot_count=robot_count,
                robot_models=robot_models,
                first_rollout=first_rollout,
                robot_root_path=str(args.robot_root),
                spawn_missing_robots=not bool(args.no_spawn_missing_robots),
            )

            if args.all_rollouts and not args.headless:
                _pre_cycle_gui_pause(
                    sim_app,
                    update_count=int(args.pre_cycle_updates),
                    delay_sec=float(args.pre_cycle_delay_sec),
                )

            for rollout in rollouts:
                _place_rollout_goal_poses(
                    rollout=rollout,
                    robot_prim_paths=robot_prim_paths,
                    sim_app=sim_app,
                    warmup_updates=int(args.warmup_updates),
                )
                print(f"[OK] Placed robots at rollout {rollout.get('id')} goal poses.")
                if args.all_rollouts and not args.headless:
                    time.sleep(max(0.0, float(args.seconds_per_rollout)))

            if not args.headless and not args.no_hold_open:
                _hold_gui(sim_app)
        finally:
            sim_app.close()

        return 0 if valid else 2
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
