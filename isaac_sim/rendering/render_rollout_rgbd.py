#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from .rollout_io import (
        REPO_ROOT,
        RolloutData,
        RolloutRobotData,
        load_rollouts,
        replay_elapsed_seconds,
        temporary_team_config_file,
    )
    from .trajectory_integration import (
        IntegratedTrajectory,
        TimedPose,
        integrate_velocity_samples,
        sample_trajectories_on_timestamps,
    )
except ImportError:
    from rollout_io import (
        REPO_ROOT,
        RolloutData,
        RolloutRobotData,
        load_rollouts,
        replay_elapsed_seconds,
        temporary_team_config_file,
    )
    from trajectory_integration import (
        IntegratedTrajectory,
        TimedPose,
        integrate_velocity_samples,
        sample_trajectories_on_timestamps,
    )


DEFAULT_CAMERA_WARMUP_STEPS = 8
DEFAULT_CAPTURE_UPDATES_PER_FRAME = 2
DEFAULT_RENDER_WIDTH_PX = 640


@dataclass(frozen=True)
class LiveRobotContext:
    robot_name: str
    root_prim_path: str
    camera_prim_path: str
    camera: Any


@dataclass(frozen=True)
class LiveRobotSpec:
    root_from_base_matrix: np.ndarray
    camera_rel_path_from_root: str
    resolution: tuple[int, int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay recorded Carter rollouts in Isaac Sim and render synchronized RGB and depth "
            "camera images for every trajectory frame."
        )
    )
    parser.add_argument(
        "--experiments-root",
        required=True,
        help="Directory containing rollout subdirectories with run_config.yaml files.",
    )
    parser.add_argument(
        "--rollout-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional rollout ids to render. Defaults to every rollout under the experiments root.",
    )
    return parser.parse_args()


def _load_module(module_name: str, module_path: Path, extra_sys_path: Path | None = None) -> Any:
    if extra_sys_path is not None:
        extra_path = str(extra_sys_path)
        if extra_path not in sys.path:
            sys.path.insert(0, extra_path)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_builder_module() -> Any:
    return _load_module(
        "build_stage_warehouse_carters",
        REPO_ROOT / "isaac_sim" / "stage_bringups" / "build_stage_warehouse_carters.py",
    )


def _load_goal_sampler_module() -> Any:
    goal_generator_dir = REPO_ROOT / "isaac_sim" / "goal_generator"
    return _load_module(
        "object_goal_sampler_core",
        goal_generator_dir / "object_goal_sampler_core.py",
        extra_sys_path=goal_generator_dir,
    )


def _enable_extension(extension_name: str) -> None:
    try:
        from isaacsim.core.utils.extensions import enable_extension
    except Exception:
        from omni.isaac.core.utils.extensions import enable_extension

    enable_extension(extension_name)


def _create_sim_app() -> Any:
    try:
        from isaacsim.simulation_app import SimulationApp
    except Exception:
        from omni.isaac.kit import SimulationApp

    return SimulationApp({"headless": True, "anti_aliasing": 0})


def _get_camera_class() -> Any:
    try:
        from isaacsim.sensors.camera import Camera

        return Camera
    except Exception:
        from omni.isaac.sensor import Camera

        return Camera


def _rotation_matrix_z(yaw_rad: float) -> np.ndarray:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0, 0.0],
            [sin_yaw, cos_yaw, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _translation_matrix(position_xyz: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = np.asarray(position_xyz, dtype=float)
    return matrix


def _yaw_from_matrix(matrix: np.ndarray) -> float:
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def _yaw_to_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


def _compute_root_pose_from_base_pose(base_pose: TimedPose, robot_spec: Any) -> tuple[tuple[float, float, float], float]:
    base_world = _translation_matrix((base_pose.x, base_pose.y, base_pose.z)) @ _rotation_matrix_z(
        base_pose.yaw
    )
    root_world = base_world @ np.linalg.inv(robot_spec.root_from_base_matrix)
    position_xyz = tuple(float(value) for value in root_world[:3, 3])
    yaw_rad = _yaw_from_matrix(root_world)
    return position_xyz, yaw_rad


def _build_live_robot_contexts(
    rollout: RolloutData,
    robot_prim_paths: list[str],
    robot_spec: Any,
) -> list[LiveRobotContext]:
    camera_class = _get_camera_class()
    live_contexts: list[LiveRobotContext] = []
    for robot, root_prim_path in zip(rollout.robots, robot_prim_paths):
        camera_prim_path = root_prim_path + robot_spec.camera_rel_path_from_root
        camera = camera_class(
            prim_path=camera_prim_path,
            resolution=robot_spec.resolution,
        )
        live_contexts.append(
            LiveRobotContext(
                robot_name=robot.name,
                root_prim_path=root_prim_path,
                camera_prim_path=camera_prim_path,
                camera=camera,
            )
        )
    return live_contexts


def _ensure_render_outputs_empty(rollout_dir: Path) -> tuple[Path, Path]:
    rgb_root = rollout_dir / "rgb"
    depth_root = rollout_dir / "depth"
    for output_root in (rgb_root, depth_root):
        if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
            raise RuntimeError(
                f"Refusing to overwrite existing rendered outputs in {output_root}. "
                "Remove the directory contents and rerun the renderer."
            )
        output_root.mkdir(parents=True, exist_ok=True)
    return rgb_root, depth_root


def _prepare_robot_output_dirs(
    rgb_root: Path,
    depth_root: Path,
    robots: tuple[RolloutRobotData, ...],
) -> dict[str, tuple[Path, Path]]:
    output_dirs: dict[str, tuple[Path, Path]] = {}
    for robot in robots:
        rgb_dir = rgb_root / robot.name
        depth_dir = depth_root / robot.name
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[robot.name] = (rgb_dir, depth_dir)
    return output_dirs


def _prepare_rollout_trajectories(
    rollout: RolloutData,
) -> tuple[dict[str, IntegratedTrajectory], dict[str, list[TimedPose]]]:
    trajectories = {
        robot.name: integrate_velocity_samples(
            robot.initial_pose,
            robot.velocity_samples,
            source_label=str(robot.velocity_path),
        )
        for robot in rollout.robots
    }
    sampled_poses = sample_trajectories_on_timestamps(trajectories, rollout.replay_timestamps_ns)
    return trajectories, sampled_poses


def _write_integrated_pose_csvs(
    rollout: RolloutData,
    sampled_poses_by_robot: dict[str, list[TimedPose]],
) -> None:
    elapsed_seconds = replay_elapsed_seconds(rollout.replay_timestamps_ns)
    for robot in rollout.robots:
        output_path = rollout.rollout_dir / f"integrated_pose_{robot.name}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp_ns", "elapsed_s", "x", "y", "z", "yaw", "qx", "qy", "qz", "qw"])
            for timed_pose, elapsed_s in zip(sampled_poses_by_robot[robot.name], elapsed_seconds):
                qx, qy, qz, qw = _yaw_to_quaternion(timed_pose.yaw)
                writer.writerow(
                    [
                        timed_pose.timestamp_ns,
                        f"{elapsed_s:.9f}",
                        f"{timed_pose.x:.9f}",
                        f"{timed_pose.y:.9f}",
                        f"{timed_pose.z:.9f}",
                        f"{timed_pose.yaw:.9f}",
                        f"{qx:.9f}",
                        f"{qy:.9f}",
                        f"{qz:.9f}",
                        f"{qw:.9f}",
                    ]
                )


def _warm_up_cameras(sim_app: Any, live_contexts: list[LiveRobotContext]) -> None:
    if not live_contexts:
        return
    for _ in range(DEFAULT_CAMERA_WARMUP_STEPS):
        sim_app.update()
    for context in live_contexts:
        if _get_rgb_frame(context.camera) is None:
            raise RuntimeError(
                f"Camera did not produce RGB data after warm-up: {context.camera_prim_path}"
            )
        if _get_depth_frame_m(context.camera) is None:
            raise RuntimeError(
                f"Camera did not produce depth data after warm-up: {context.camera_prim_path}"
            )


def _get_rgb_frame(camera: Any) -> np.ndarray | None:
    rgb_frame = camera.get_rgb()
    if rgb_frame is None:
        current_frame = getattr(camera, "get_current_frame", lambda: {})()
        rgb_frame = current_frame.get("rgb")
    if rgb_frame is None:
        return None
    rgb_array = np.asarray(rgb_frame)
    if rgb_array.ndim == 2:
        rgb_array = np.repeat(rgb_array[..., None], 3, axis=2)
    if rgb_array.shape[-1] > 3:
        rgb_array = rgb_array[..., :3]
    if rgb_array.dtype != np.uint8:
        if np.issubdtype(rgb_array.dtype, np.floating):
            if np.nanmax(rgb_array) <= 1.0:
                rgb_array = np.clip(rgb_array * 255.0, 0.0, 255.0)
            rgb_array = rgb_array.astype(np.uint8)
        else:
            rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
    return rgb_array


def _get_depth_frame_m(camera: Any) -> np.ndarray | None:
    depth_frame = None
    if hasattr(camera, "get_depth"):
        depth_frame = camera.get_depth()
    if depth_frame is None:
        current_frame = getattr(camera, "get_current_frame", lambda: {})()
        depth_frame = current_frame.get("distance_to_image_plane")
        if depth_frame is None:
            depth_frame = current_frame.get("depth")
    if depth_frame is None:
        return None
    depth_array = np.asarray(depth_frame, dtype=np.float32)
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    return depth_array


def _save_rgb_png(output_path: Path, rgb_frame: np.ndarray) -> None:
    Image.fromarray(rgb_frame).save(output_path)


def _save_depth_png_mm(output_path: Path, depth_frame_m: np.ndarray) -> None:
    depth_array = np.asarray(depth_frame_m, dtype=np.float32)
    valid_mask = np.isfinite(depth_array) & (depth_array > 0.0)
    depth_mm = np.zeros(depth_array.shape, dtype=np.uint16)
    if np.any(valid_mask):
        scaled_depth = np.rint(depth_array[valid_mask] * 1000.0)
        valid_scaled_mask = (scaled_depth > 0.0) & (scaled_depth <= np.iinfo(np.uint16).max)
        valid_indices = np.flatnonzero(valid_mask)
        depth_mm.flat[valid_indices[valid_scaled_mask]] = scaled_depth[valid_scaled_mask].astype(
            np.uint16
        )
    Image.fromarray(depth_mm).save(output_path)


def _cleanup_camera(context: LiveRobotContext) -> None:
    destroy = getattr(context.camera, "destroy", None)
    if callable(destroy):
        try:
            destroy()
        except Exception as exc:
            if "Invalid NodeObj object in Py_Node in getAttributes" in str(exc):
                return
            print(
                f"[WARN] Failed to destroy camera wrapper for {context.camera_prim_path}: {exc}",
                flush=True,
            )


def _ensure_camera_annotator(camera: Any, annotator_name: str, add_method_name: str) -> None:
    custom_annotators = getattr(camera, "_custom_annotators", None)
    if isinstance(custom_annotators, dict) and annotator_name in custom_annotators:
        return

    add_method = getattr(camera, add_method_name, None)
    if callable(add_method):
        add_method()


def _inspect_live_robot_spec(goal_sampler_module: Any, root_prim_path: str) -> LiveRobotSpec:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"Robot prim was not found on the live stage: {root_prim_path}")

    sampler_class = goal_sampler_module.ObjectReachingGoalSampler
    base_prim = sampler_class._resolve_robot_base_prim(root_prim, None)
    camera_prim = sampler_class._resolve_robot_camera_prim(root_prim, None)

    root_world = sampler_class._compute_world_transform_matrix(root_prim)
    base_world = sampler_class._compute_world_transform_matrix(base_prim)
    camera_world = sampler_class._compute_world_transform_matrix(camera_prim)
    root_from_base = np.linalg.inv(root_world) @ base_world
    camera_rel = np.linalg.inv(base_world) @ camera_world

    camera = UsdGeom.Camera(camera_prim)
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get() or 20.955)
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get() or 15.2908)
    aspect_ratio = horizontal_aperture / max(vertical_aperture, 1e-6)
    resolution = (
        DEFAULT_RENDER_WIDTH_PX,
        max(1, int(round(DEFAULT_RENDER_WIDTH_PX / max(aspect_ratio, 1e-6)))),
    )

    root_path = root_prim.GetPath().pathString
    camera_path = camera_prim.GetPath().pathString
    camera_rel_path = camera_path[len(root_path) :] if camera_path.startswith(root_path) else camera_path
    camera_forward_base = camera_rel[:3, :3] @ np.array([0.0, 0.0, -1.0], dtype=float)
    camera_forward_yaw = math.atan2(camera_forward_base[1], camera_forward_base[0])

    print(
        "[INFO] Live robot base/camera selection: "
        f"base_prim={base_prim.GetPath().pathString}, "
        f"camera_prim={camera_path}, "
        f"camera_rel_translation={camera_rel[:3, 3].tolist()}, "
        f"camera_forward_yaw_deg={math.degrees(camera_forward_yaw):.2f}, "
        f"resolution={resolution[0]}x{resolution[1]}",
        flush=True,
    )
    return LiveRobotSpec(
        root_from_base_matrix=root_from_base,
        camera_rel_path_from_root=camera_rel_path,
        resolution=resolution,
    )


def _render_rollout(
    sim_app: Any,
    builder_module: Any,
    goal_sampler_module: Any,
    rollout: RolloutData,
) -> None:
    rgb_root, depth_root = _ensure_render_outputs_empty(rollout.rollout_dir)
    robot_output_dirs = _prepare_robot_output_dirs(rgb_root, depth_root, rollout.robots)
    trajectories, sampled_poses_by_robot = _prepare_rollout_trajectories(rollout)
    del trajectories

    _write_integrated_pose_csvs(rollout, sampled_poses_by_robot)
    with temporary_team_config_file(rollout) as team_config_path:
        build_attempts = [
            ("render asset", {"enable_ros2": False}),
            (
                "ROS-authored asset",
                {
                    "enable_ros2": False,
                    "robot_usd_rel": getattr(builder_module, "NOVA_CARTER_ROS_USD_REL", None),
                },
            ),
        ]

        robot_prim_paths: list[str] | None = None
        robot_spec: LiveRobotSpec | None = None
        last_error: Exception | None = None
        for attempt_label, build_kwargs in build_attempts:
            try:
                filtered_kwargs = {
                    key: value for key, value in build_kwargs.items() if value is not None
                }
                robot_prim_paths = builder_module.build_stage(
                    str(team_config_path),
                    "",
                    **filtered_kwargs,
                )
                robot_spec = _inspect_live_robot_spec(goal_sampler_module, robot_prim_paths[0])
                if attempt_label != "render asset":
                    print(
                        f"[WARN] Falling back to {attempt_label} for rollout {rollout.rollout_id}.",
                        flush=True,
                    )
                break
            except Exception as exc:
                last_error = exc
                if attempt_label == "render asset":
                    print(
                        "[WARN] Failed to inspect the non-ROS Carter render asset for "
                        f"rollout {rollout.rollout_id}: {exc}. Retrying with the ROS-authored asset.",
                        flush=True,
                    )
                else:
                    raise

        if robot_prim_paths is None or robot_spec is None:
            if last_error is None:
                raise RuntimeError("Failed to build and inspect a live Carter stage for rendering.")
            raise last_error

    live_contexts = _build_live_robot_contexts(rollout, robot_prim_paths, robot_spec)

    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    try:
        timeline.play()
        for context in live_contexts:
            try:
                context.camera.initialize(attach_rgb_annotator=False)
            except TypeError:
                context.camera.initialize()
            _ensure_camera_annotator(context.camera, "rgb", "add_rgb_to_frame")
            _ensure_camera_annotator(
                context.camera,
                "distance_to_image_plane",
                "add_distance_to_image_plane_to_frame",
            )

        for _ in range(DEFAULT_CAMERA_WARMUP_STEPS):
            sim_app.update()
        _warm_up_cameras(sim_app, live_contexts)

        manifest_path = rollout.rollout_dir / "render_manifest.csv"
        elapsed_seconds = replay_elapsed_seconds(rollout.replay_timestamps_ns)
        with manifest_path.open("w", encoding="utf-8", newline="") as manifest_stream:
            manifest_writer = csv.writer(manifest_stream)
            manifest_writer.writerow(["frame_index", "timestamp_ns", "elapsed_s"])
            total_frames = len(rollout.replay_timestamps_ns)
            for frame_index, (timestamp_ns, elapsed_s) in enumerate(
                zip(rollout.replay_timestamps_ns, elapsed_seconds)
            ):
                for context in live_contexts:
                    base_pose = sampled_poses_by_robot[context.robot_name][frame_index]
                    position_xyz, yaw_rad = _compute_root_pose_from_base_pose(base_pose, robot_spec)
                    builder_module._set_xform_pose(
                        context.root_prim_path,
                        position_xyz,
                        yaw_deg=math.degrees(yaw_rad),
                    )

                for _ in range(DEFAULT_CAPTURE_UPDATES_PER_FRAME):
                    sim_app.update()

                for context in live_contexts:
                    rgb_frame = _get_rgb_frame(context.camera)
                    depth_frame_m = _get_depth_frame_m(context.camera)
                    if rgb_frame is None:
                        raise RuntimeError(f"RGB frame was unavailable for camera {context.camera_prim_path}")
                    if depth_frame_m is None:
                        raise RuntimeError(f"Depth frame was unavailable for camera {context.camera_prim_path}")

                    rgb_output_dir, depth_output_dir = robot_output_dirs[context.robot_name]
                    frame_name = f"frame_{frame_index:06d}.png"
                    _save_rgb_png(rgb_output_dir / frame_name, rgb_frame)
                    _save_depth_png_mm(depth_output_dir / frame_name, depth_frame_m)

                manifest_writer.writerow([frame_index, timestamp_ns, f"{elapsed_s:.9f}"])
                if frame_index == 0 or (frame_index + 1) == total_frames or (frame_index + 1) % 100 == 0:
                    print(
                        f"[INFO] Rollout {rollout.rollout_id}: rendered frame {frame_index + 1}/{total_frames}.",
                        flush=True,
                    )
    finally:
        for context in live_contexts:
            _cleanup_camera(context)
        live_contexts.clear()
        gc.collect()
        try:
            timeline.stop()
        except Exception as exc:
            print(f"[WARN] Failed to stop Isaac timeline cleanly after rollout {rollout.rollout_id}: {exc}", flush=True)
        for _ in range(2):
            sim_app.update()


def main() -> int:
    args = _parse_args()
    try:
        rollouts = load_rollouts(args.experiments_root, rollout_ids=args.rollout_ids)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"RenderRolloutRGBD: {exc}", file=sys.stderr)
        return 1

    builder_module = _load_builder_module()
    sim_app = _create_sim_app()
    if sim_app is None:
        raise RuntimeError("Failed to start Isaac Sim.")

    _enable_extension("isaacsim.sensors.camera")
    for _ in range(DEFAULT_CAMERA_WARMUP_STEPS):
        sim_app.update()

    sampler_module = _load_goal_sampler_module()

    try:
        for rollout in rollouts:
            print(
                f"[INFO] Rendering rollout {rollout.rollout_id} from {rollout.rollout_dir}...",
                flush=True,
            )
            _render_rollout(sim_app, builder_module, sampler_module, rollout)
            print(
                f"[INFO] Finished rollout {rollout.rollout_id}. RGB and depth images are stored under {rollout.rollout_dir}.",
                flush=True,
            )
    except Exception as exc:
        print(f"RenderRolloutRGBD: {exc}", file=sys.stderr)
        return 1
    finally:
        sim_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
