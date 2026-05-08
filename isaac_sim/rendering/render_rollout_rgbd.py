#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

try:
    from .rollout_io import (
        REPO_ROOT,
        RolloutData,
        RolloutRobotData,
        load_rollouts,
        replay_elapsed_seconds,
        resolve_rollout_scene_usd_path,
        temporary_team_config_file,
    )
    from .trajectory_integration import (
        TimedPose,
        build_pose_trajectory,
        integrate_velocity_samples,
    )
except ImportError:
    from rollout_io import (
        REPO_ROOT,
        RolloutData,
        RolloutRobotData,
        load_rollouts,
        replay_elapsed_seconds,
        resolve_rollout_scene_usd_path,
        temporary_team_config_file,
    )
    from trajectory_integration import (
        TimedPose,
        build_pose_trajectory,
        integrate_velocity_samples,
    )


DEFAULT_CAMERA_WARMUP_STEPS = 8
DEFAULT_CAPTURE_UPDATES_PER_FRAME = 2
DEFAULT_CAMERA_CONFIG_PATH = REPO_ROOT / "isaac_sim" / "rendering" / "camera_settings.yaml"


@dataclass(frozen=True)
class RobotCameraSettings:
    mode: str
    fallback_virtual_camera: dict[str, Any]
    model_overrides: dict[str, dict[str, Any]]
    asset_camera_reject_tokens: tuple[str, ...]


@dataclass(frozen=True)
class BeVCameraSettings:
    name: str
    position_xyz: tuple[float, float, float]
    target_xyz: tuple[float, float, float]
    up_axis: tuple[float, float, float]
    focal_length_mm: float | None
    horizontal_aperture_mm: float | None
    vertical_aperture_mm: float | None
    clipping_range_m: tuple[float, float] | None
    resolution: tuple[int, int] | None


@dataclass(frozen=True)
class CameraSettings:
    output_resolution: tuple[int, int]
    robot_camera: RobotCameraSettings
    bev_cameras: tuple[BeVCameraSettings, ...]


@dataclass(frozen=True)
class LiveCameraContext:
    camera_name: str
    output_name: str
    camera_type: str
    camera_prim_path: str
    camera: Any
    robot_name: str | None = None
    robot_model: str | None = None
    root_prim_path: str | None = None
    root_from_base_matrix: np.ndarray | None = None
    selection_mode: str = ""


@dataclass(frozen=True)
class LiveRobotMotionSpec:
    root_from_base_matrix: np.ndarray
    base_prim_path: str


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
    parser.add_argument(
        "--camera-config",
        default=str(DEFAULT_CAMERA_CONFIG_PATH),
        help=(
            "Shared RGBD camera settings YAML. Defaults to "
            f"{DEFAULT_CAMERA_CONFIG_PATH}."
        ),
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
        extra_sys_path=REPO_ROOT,
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


def _as_float_triplet(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a 3-value list.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _as_resolution(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a 2-value [width, height] list.")
    width = int(value[0])
    height = int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} must contain positive dimensions, got {value!r}.")
    return (width, height)


def _as_optional_float_pair(value: Any, label: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a 2-value list when provided.")
    return (float(value[0]), float(value[1]))


def _load_camera_settings(config_path: str | Path) -> CameraSettings:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Camera settings YAML does not exist: {path}")

    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}

    output_resolution = _as_resolution(
        payload.get("output_resolution", [224, 224]),
        f"{path}.output_resolution",
    )

    robot_camera_payload = dict(payload.get("robot_camera", {}) or {})
    fallback_camera = dict(robot_camera_payload.get("fallback_virtual_camera", {}) or {})
    if not fallback_camera:
        raise ValueError(
            f"{path}.robot_camera.fallback_virtual_camera is required. "
            "If you pasted a validation snippet, replace the existing robot_camera block "
            "with the full printed block or merge model_overrides into the existing block; "
            "do not add a second top-level robot_camera key."
        )
    fallback_camera.setdefault("prim_name", "RenderFallbackCamera")
    fallback_camera["position_xyz"] = _as_float_triplet(
        fallback_camera.get("position_xyz", [0.35, 0.0, 0.75]),
        f"{path}.robot_camera.fallback_virtual_camera.position_xyz",
    )
    fallback_camera["target_xyz"] = _as_float_triplet(
        fallback_camera.get("target_xyz", [2.0, 0.0, 0.45]),
        f"{path}.robot_camera.fallback_virtual_camera.target_xyz",
    )
    fallback_camera["up_axis"] = _as_float_triplet(
        fallback_camera.get("up_axis", [0.0, 0.0, 1.0]),
        f"{path}.robot_camera.fallback_virtual_camera.up_axis",
    )
    fallback_camera["clipping_range_m"] = _as_optional_float_pair(
        fallback_camera.get("clipping_range_m"),
        f"{path}.robot_camera.fallback_virtual_camera.clipping_range_m",
    )

    model_overrides = {
        str(model): dict(override or {})
        for model, override in dict(robot_camera_payload.get("model_overrides", {}) or {}).items()
    }
    asset_camera_reject_tokens = tuple(
        str(token).strip().lower()
        for token in list(
            robot_camera_payload.get(
                "asset_camera_reject_tokens",
                ["third_person", "third-person", "rear", "follow", "external", "debug"],
            )
            or []
        )
        if str(token).strip()
    )

    bev_payloads = list(payload.get("bev_cameras", []) or [])
    if len(bev_payloads) != 2:
        raise ValueError(f"{path}.bev_cameras must define exactly two BEV cameras.")

    bev_cameras: list[BeVCameraSettings] = []
    seen_names: set[str] = set()
    for index, camera_payload in enumerate(bev_payloads):
        if not isinstance(camera_payload, dict):
            raise ValueError(f"{path}.bev_cameras[{index}] must be a mapping.")
        name = str(camera_payload.get("name", "")).strip()
        if not name:
            raise ValueError(f"{path}.bev_cameras[{index}].name is required.")
        if name in seen_names:
            raise ValueError(f"{path}.bev_cameras contains duplicate name {name!r}.")
        seen_names.add(name)
        resolution = (
            _as_resolution(camera_payload["resolution"], f"{path}.bev_cameras[{index}].resolution")
            if "resolution" in camera_payload
            else None
        )
        bev_cameras.append(
            BeVCameraSettings(
                name=name,
                position_xyz=_as_float_triplet(
                    camera_payload.get("position_xyz"),
                    f"{path}.bev_cameras[{index}].position_xyz",
                ),
                target_xyz=_as_float_triplet(
                    camera_payload.get("target_xyz"),
                    f"{path}.bev_cameras[{index}].target_xyz",
                ),
                up_axis=_as_float_triplet(
                    camera_payload.get("up_axis", [0.0, 1.0, 0.0]),
                    f"{path}.bev_cameras[{index}].up_axis",
                ),
                focal_length_mm=(
                    float(camera_payload["focal_length_mm"])
                    if "focal_length_mm" in camera_payload
                    else None
                ),
                horizontal_aperture_mm=(
                    float(camera_payload["horizontal_aperture_mm"])
                    if "horizontal_aperture_mm" in camera_payload
                    else None
                ),
                vertical_aperture_mm=(
                    float(camera_payload["vertical_aperture_mm"])
                    if "vertical_aperture_mm" in camera_payload
                    else None
                ),
                clipping_range_m=_as_optional_float_pair(
                    camera_payload.get("clipping_range_m"),
                    f"{path}.bev_cameras[{index}].clipping_range_m",
                ),
                resolution=resolution,
            )
        )

    return CameraSettings(
        output_resolution=output_resolution,
        robot_camera=RobotCameraSettings(
            mode=str(robot_camera_payload.get("mode", "asset_if_available") or "asset_if_available"),
            fallback_virtual_camera=fallback_camera,
            model_overrides=model_overrides,
            asset_camera_reject_tokens=asset_camera_reject_tokens,
        ),
        bev_cameras=tuple(bev_cameras),
    )


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


def _compute_root_pose_from_base_pose(
    base_pose: TimedPose,
    root_from_base_matrix: np.ndarray,
) -> tuple[tuple[float, float, float], float]:
    base_world = _translation_matrix((base_pose.x, base_pose.y, base_pose.z)) @ _rotation_matrix_z(
        base_pose.yaw
    )
    root_world = base_world @ np.linalg.inv(root_from_base_matrix)
    position_xyz = tuple(float(value) for value in root_world[:3, 3])
    yaw_rad = _yaw_from_matrix(root_world)
    return position_xyz, yaw_rad


def _normalize_robot_infos(
    rollout: RolloutData,
    robot_infos: list[Any],
) -> dict[str, dict[str, Any]]:
    if len(robot_infos) != len(rollout.robots):
        raise RuntimeError(
            f"Stage builder returned {len(robot_infos)} robots for rollout {rollout.rollout_id}, "
            f"but the rollout contains {len(rollout.robots)} robots."
        )

    normalized: dict[str, dict[str, Any]] = {}
    for robot, raw_info in zip(rollout.robots, robot_infos):
        if isinstance(raw_info, dict):
            root_prim_path = str(raw_info.get("prim_path", "")).strip()
            name = str(raw_info.get("name", "") or robot.name).strip()
            model = str(raw_info.get("model", "") or robot.model).strip()
        else:
            root_prim_path = str(raw_info).strip()
            name = robot.name
            model = robot.model
        if not root_prim_path:
            raise RuntimeError(f"Stage builder returned an empty prim path for robot {robot.name}.")
        normalized[robot.name] = {
            "name": name or robot.name,
            "model": model or robot.model,
            "prim_path": root_prim_path,
        }
    return normalized


def _safe_camera_token(value: str) -> str:
    safe = "".join(character if character.isalnum() or character == "_" else "_" for character in value)
    return safe or "camera"


def _normalize_vector(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"Cannot normalize zero-length vector for {label}.")
    return vector / norm


def _look_at_rotation_matrix(
    position_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
    up_axis: tuple[float, float, float],
) -> np.ndarray:
    position = np.asarray(position_xyz, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    up = _normalize_vector(np.asarray(up_axis, dtype=float), "camera up axis")
    forward = _normalize_vector(target - position, "camera look-at direction")
    if abs(float(np.dot(forward, up))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(forward, up))) > 0.98:
            up = np.array([1.0, 0.0, 0.0], dtype=float)

    right = _normalize_vector(np.cross(forward, up), "camera right axis")
    local_z = -forward
    local_y = _normalize_vector(np.cross(local_z, right), "camera local y axis")
    rotation = np.eye(3, dtype=float)
    rotation[:, 0] = right
    rotation[:, 1] = local_y
    rotation[:, 2] = local_z
    return rotation


def _quaternion_xyzw_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif diagonal_index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=float)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    return tuple(float(value) for value in quaternion)


def _set_camera_pose_quaternion(
    prim_path: str,
    position_xyz: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
) -> None:
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Camera prim was not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    qx, qy, qz, qw = orientation_xyzw
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTransform(
        Gf.Rotation(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz))),
        Gf.Vec3d(*position_xyz),
    )
    xformable.AddTransformOp().Set(matrix)


def _set_camera_look_at_pose(
    prim_path: str,
    position_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
    up_axis: tuple[float, float, float],
) -> None:
    rotation = _look_at_rotation_matrix(position_xyz, target_xyz, up_axis)
    _set_camera_pose_quaternion(
        prim_path,
        position_xyz,
        _quaternion_xyzw_from_rotation_matrix(rotation),
    )


def _apply_camera_attributes(
    prim_path: str,
    *,
    focal_length_mm: float | None = None,
    horizontal_aperture_mm: float | None = None,
    vertical_aperture_mm: float | None = None,
    clipping_range_m: tuple[float, float] | None = None,
) -> None:
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Camera prim was not found: {prim_path}")
    camera = UsdGeom.Camera(prim)
    if focal_length_mm is not None:
        camera.GetFocalLengthAttr().Set(float(focal_length_mm))
    if horizontal_aperture_mm is not None:
        camera.GetHorizontalApertureAttr().Set(float(horizontal_aperture_mm))
    if vertical_aperture_mm is not None:
        camera.GetVerticalApertureAttr().Set(float(vertical_aperture_mm))
    if clipping_range_m is not None:
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(float(clipping_range_m[0]), float(clipping_range_m[1])))


def _define_camera_prim(prim_path: str) -> None:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Camera.Define(stage, prim_path)


def _ensure_xform_prim(prim_path: str) -> None:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        UsdGeom.Xform.Define(stage, prim_path)


def _inspect_live_robot_motion_spec(
    goal_sampler_module: Any,
    root_prim_path: str,
    robot_name: str,
    robot_model: str,
) -> LiveRobotMotionSpec:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"Robot prim was not found on the live stage: {root_prim_path}")

    sampler_class = goal_sampler_module.ObjectReachingGoalSampler
    base_prim = sampler_class._resolve_robot_base_prim(root_prim, None)
    root_world = sampler_class._compute_world_transform_matrix(root_prim)
    base_world = sampler_class._compute_world_transform_matrix(base_prim)
    root_from_base = np.linalg.inv(root_world) @ base_world
    base_prim_path = base_prim.GetPath().pathString
    print(
        "[INFO] Live robot base selection: "
        f"robot={robot_name}, model={robot_model}, root_prim={root_prim_path}, "
        f"base_prim={base_prim_path}",
        flush=True,
    )
    return LiveRobotMotionSpec(root_from_base_matrix=root_from_base, base_prim_path=base_prim_path)


def _resolve_explicit_camera_prim_path(root_prim_path: str, camera_prim_path: str) -> str:
    camera_prim_path = str(camera_prim_path).strip()
    if not camera_prim_path:
        raise ValueError("camera_prim_path override must not be empty.")
    if camera_prim_path.startswith("/"):
        return camera_prim_path
    return f"{root_prim_path}/{camera_prim_path.strip('/')}"


def _camera_reject_reason(camera_prim_path: str, reject_tokens: tuple[str, ...]) -> str:
    path_lower = camera_prim_path.lower()
    for token in reject_tokens:
        if token and token in path_lower:
            return f"contains reject token '{token}'"
    return ""


def _asset_camera_prim_path(
    goal_sampler_module: Any,
    root_prim_path: str,
    robot_name: str,
    robot_model: str,
    reject_tokens: tuple[str, ...],
) -> str | None:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"Robot prim was not found on the live stage: {root_prim_path}")
    sampler_class = goal_sampler_module.ObjectReachingGoalSampler
    candidates = sampler_class._sorted_camera_candidates(root_prim)
    if not candidates:
        print(
            "[WARN] Robot asset camera candidates: "
            f"robot={robot_name}, model={robot_model}, root={root_prim_path}, candidates=[]",
            flush=True,
        )
        return None

    candidate_logs: list[str] = []
    for candidate in candidates:
        candidate_path = candidate.GetPath().pathString
        reject_reason = _camera_reject_reason(candidate_path, reject_tokens)
        if reject_reason:
            candidate_logs.append(f"{candidate_path} [rejected: {reject_reason}]")
            continue
        candidate_logs.append(f"{candidate_path} [selected]")
        print(
            "[INFO] Robot asset camera candidates: "
            f"robot={robot_name}, model={robot_model}, candidates={candidate_logs}",
            flush=True,
        )
        return candidate_path

    print(
        "[WARN] Robot asset camera candidates: "
        f"robot={robot_name}, model={robot_model}, candidates={candidate_logs}; "
        "falling back to virtual camera.",
        flush=True,
    )
    return None


def _camera_intrinsics_for_log(camera_prim_path: str) -> dict[str, Any]:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim or not prim.IsValid():
        return {}
    camera = UsdGeom.Camera(prim)
    return {
        "focal_length_mm": camera.GetFocalLengthAttr().Get(),
        "horizontal_aperture_mm": camera.GetHorizontalApertureAttr().Get(),
        "vertical_aperture_mm": camera.GetVerticalApertureAttr().Get(),
        "clipping_range": camera.GetClippingRangeAttr().Get(),
    }


def _robot_fallback_camera_config(
    camera_settings: CameraSettings,
    robot_model: str,
) -> dict[str, Any]:
    config = dict(camera_settings.robot_camera.fallback_virtual_camera)
    override = camera_settings.robot_camera.model_overrides.get(robot_model)
    if override:
        config.update(override)
    return config


def _create_virtual_robot_camera(
    parent_prim_path: str,
    robot_name: str,
    robot_model: str,
    camera_settings: CameraSettings,
) -> str:
    config = _robot_fallback_camera_config(camera_settings, robot_model)
    prim_name = _safe_camera_token(str(config.get("prim_name", "RenderFallbackCamera")))
    camera_prim_path = f"{parent_prim_path}/{prim_name}"
    _define_camera_prim(camera_prim_path)
    _set_camera_look_at_pose(
        camera_prim_path,
        tuple(config["position_xyz"]),
        tuple(config["target_xyz"]),
        tuple(config["up_axis"]),
    )
    _apply_camera_attributes(
        camera_prim_path,
        focal_length_mm=(
            float(config["focal_length_mm"]) if "focal_length_mm" in config else None
        ),
        horizontal_aperture_mm=(
            float(config["horizontal_aperture_mm"])
            if "horizontal_aperture_mm" in config
            else None
        ),
        vertical_aperture_mm=(
            float(config["vertical_aperture_mm"]) if "vertical_aperture_mm" in config else None
        ),
        clipping_range_m=config.get("clipping_range_m"),
    )
    print(
        "[WARN] Falling back to virtual robot camera: "
        f"robot={robot_name}, model={robot_model}, parent_prim={parent_prim_path}, "
        f"camera_prim={camera_prim_path}",
        flush=True,
    )
    return camera_prim_path


def _create_bev_camera_contexts(camera_settings: CameraSettings) -> list[LiveCameraContext]:
    camera_class = _get_camera_class()
    contexts: list[LiveCameraContext] = []
    _ensure_xform_prim("/World/RenderCameras")
    for bev_config in camera_settings.bev_cameras:
        output_name = f"bev_{_safe_camera_token(bev_config.name)}"
        camera_prim_path = f"/World/RenderCameras/{output_name}"
        _define_camera_prim(camera_prim_path)
        _set_camera_look_at_pose(
            camera_prim_path,
            bev_config.position_xyz,
            bev_config.target_xyz,
            bev_config.up_axis,
        )
        _apply_camera_attributes(
            camera_prim_path,
            focal_length_mm=bev_config.focal_length_mm,
            horizontal_aperture_mm=bev_config.horizontal_aperture_mm,
            vertical_aperture_mm=bev_config.vertical_aperture_mm,
            clipping_range_m=bev_config.clipping_range_m,
        )
        resolution = bev_config.resolution or camera_settings.output_resolution
        camera = camera_class(prim_path=camera_prim_path, resolution=resolution)
        contexts.append(
            LiveCameraContext(
                camera_name=bev_config.name,
                output_name=output_name,
                camera_type="bev",
                camera_prim_path=camera_prim_path,
                camera=camera,
                selection_mode="yaml",
            )
        )
        print(
            "[INFO] Added BEV camera: "
            f"name={bev_config.name}, prim={camera_prim_path}, "
            f"position={list(bev_config.position_xyz)}, target={list(bev_config.target_xyz)}, "
            f"resolution={resolution[0]}x{resolution[1]}",
            flush=True,
        )
    return contexts


def _create_robot_camera_contexts(
    rollout: RolloutData,
    robot_infos: dict[str, dict[str, Any]],
    goal_sampler_module: Any,
    camera_settings: CameraSettings,
) -> list[LiveCameraContext]:
    camera_class = _get_camera_class()
    contexts: list[LiveCameraContext] = []
    for robot in rollout.robots:
        info = robot_infos[robot.name]
        root_prim_path = str(info["prim_path"])
        robot_model = str(info.get("model", robot.model) or robot.model)
        motion_spec = _inspect_live_robot_motion_spec(
            goal_sampler_module,
            root_prim_path,
            robot.name,
            robot_model,
        )
        camera_prim_path = None
        selection_mode = "fallback_virtual"
        model_override = dict(camera_settings.robot_camera.model_overrides.get(robot_model, {}) or {})
        camera_mode = str(
            model_override.get("mode", camera_settings.robot_camera.mode) or "asset_if_available"
        )
        explicit_camera_prim_path = str(model_override.get("camera_prim_path", "") or "").strip()
        if explicit_camera_prim_path:
            camera_prim_path = _resolve_explicit_camera_prim_path(
                root_prim_path,
                explicit_camera_prim_path,
            )
            selection_mode = "explicit_asset"
        elif camera_mode == "asset_if_available":
            camera_prim_path = _asset_camera_prim_path(
                goal_sampler_module,
                root_prim_path,
                robot.name,
                robot_model,
                camera_settings.robot_camera.asset_camera_reject_tokens,
            )
            if camera_prim_path:
                selection_mode = "asset"
        elif camera_mode != "virtual_only":
            raise ValueError(
                "robot_camera.mode must be 'asset_if_available' or 'virtual_only', "
                f"got {camera_mode!r} for robot {robot.name} model {robot_model}."
            )
        if camera_prim_path is None:
            camera_prim_path = _create_virtual_robot_camera(
                motion_spec.base_prim_path,
                robot.name,
                robot_model,
                camera_settings,
            )

        camera = camera_class(
            prim_path=camera_prim_path,
            resolution=camera_settings.output_resolution,
        )
        intrinsics = _camera_intrinsics_for_log(camera_prim_path)
        print(
            "[INFO] Robot camera selection: "
            f"robot={robot.name}, model={robot_model}, mode={selection_mode}, "
            f"camera_prim={camera_prim_path}, resolution={camera_settings.output_resolution[0]}x"
            f"{camera_settings.output_resolution[1]}, intrinsics={intrinsics}",
            flush=True,
        )
        contexts.append(
            LiveCameraContext(
                camera_name=robot.name,
                output_name=robot.name,
                camera_type="robot",
                camera_prim_path=camera_prim_path,
                camera=camera,
                robot_name=robot.name,
                robot_model=robot_model,
                root_prim_path=root_prim_path,
                root_from_base_matrix=motion_spec.root_from_base_matrix,
                selection_mode=selection_mode,
            )
        )
    return contexts


def _build_live_camera_contexts(
    rollout: RolloutData,
    robot_infos: list[Any],
    goal_sampler_module: Any,
    camera_settings: CameraSettings,
) -> list[LiveCameraContext]:
    normalized_robot_infos = _normalize_robot_infos(rollout, robot_infos)
    contexts = _create_robot_camera_contexts(
        rollout,
        normalized_robot_infos,
        goal_sampler_module,
        camera_settings,
    )
    contexts.extend(_create_bev_camera_contexts(camera_settings))
    return contexts


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


def _prepare_camera_output_dirs(
    rgb_root: Path,
    depth_root: Path,
    camera_output_names: list[str] | tuple[str, ...],
) -> dict[str, tuple[Path, Path]]:
    output_dirs: dict[str, tuple[Path, Path]] = {}
    for output_name in camera_output_names:
        rgb_dir = rgb_root / output_name
        depth_dir = depth_root / output_name
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[output_name] = (rgb_dir, depth_dir)
    return output_dirs


def _has_global_recorded_pose(rollout: RolloutData, robot: RolloutRobotData) -> bool:
    if not robot.has_recorded_pose:
        return False
    return str(rollout.record_settings.get("source", "")).strip() == "simulator_tf_and_odometry"


def _build_render_timestamps_ns(rollout: RolloutData) -> tuple[int, ...]:
    unique_timestamps_ns = sorted(set(rollout.replay_timestamps_ns))
    if len(unique_timestamps_ns) < 2:
        return tuple(unique_timestamps_ns)

    per_robot_periods_ns: list[int] = []
    for robot in rollout.robots:
        timestamps_ns = [sample.timestamp_ns for sample in robot.velocity_samples]
        deltas_ns = [
            next_timestamp_ns - timestamp_ns
            for timestamp_ns, next_timestamp_ns in zip(timestamps_ns, timestamps_ns[1:])
            if next_timestamp_ns > timestamp_ns
        ]
        if deltas_ns:
            per_robot_periods_ns.append(int(statistics.median(deltas_ns)))

    if not per_robot_periods_ns:
        return tuple(unique_timestamps_ns)

    nominal_period_ns = int(statistics.median(per_robot_periods_ns))
    cluster_gap_ns = max(int(round(nominal_period_ns * 0.4)), 1)
    clustered_timestamps_ns: list[int] = []
    current_cluster = [unique_timestamps_ns[0]]
    for timestamp_ns in unique_timestamps_ns[1:]:
        if timestamp_ns - current_cluster[-1] <= cluster_gap_ns:
            current_cluster.append(timestamp_ns)
            continue
        clustered_timestamps_ns.append(int(round(statistics.median(current_cluster))))
        current_cluster = [timestamp_ns]
    clustered_timestamps_ns.append(int(round(statistics.median(current_cluster))))

    if len(clustered_timestamps_ns) != len(unique_timestamps_ns):
        print(
            f"[INFO] Rollout {rollout.rollout_id}: collapsed {len(unique_timestamps_ns)} raw "
            f"timestamps into {len(clustered_timestamps_ns)} render frames using a "
            f"{cluster_gap_ns * 1e-6:.1f} ms clustering window.",
            flush=True,
        )
    return tuple(clustered_timestamps_ns)


def _prepare_rollout_sampled_poses(
    rollout: RolloutData,
    render_timestamps_ns: tuple[int, ...],
) -> tuple[dict[str, list[TimedPose]], dict[str, str]]:
    sampled_poses_by_robot: dict[str, list[TimedPose]] = {}
    pose_source_by_robot: dict[str, str] = {}
    for robot in rollout.robots:
        if _has_global_recorded_pose(rollout, robot):
            pose_trajectory = build_pose_trajectory(
                [
                    TimedPose(
                        timestamp_ns=sample.timestamp_ns,
                        x=float(sample.x),
                        y=float(sample.y),
                        z=robot.initial_pose.z,
                        yaw=float(sample.yaw),
                    )
                    for sample in robot.velocity_samples
                ],
                source_label=str(robot.velocity_path),
            )
            sampled_poses_by_robot[robot.name] = pose_trajectory.sample(render_timestamps_ns)
            pose_source_by_robot[robot.name] = "recorded_global_pose"
            print(
                f"[INFO] Rollout {rollout.rollout_id} robot {robot.name}: "
                "replaying directly from recorded global poses.",
                flush=True,
            )
            continue

        if robot.has_recorded_pose:
            print(
                f"[WARN] Rollout {rollout.rollout_id} robot {robot.name}: "
                "pose columns were not recorded from a trusted global TF source; "
                "falling back to dead-reckoned velocity integration.",
                flush=True,
            )
        else:
            print(
                f"[WARN] Rollout {rollout.rollout_id} robot {robot.name}: "
                "pose columns were not recorded; falling back to dead-reckoned velocity integration.",
                flush=True,
            )

        integrated_trajectory = integrate_velocity_samples(
            robot.initial_pose,
            robot.velocity_samples,
            source_label=str(robot.velocity_path),
        )
        sampled_poses_by_robot[robot.name] = integrated_trajectory.sample(render_timestamps_ns)
        pose_source_by_robot[robot.name] = "velocity_integration_fallback"

    return sampled_poses_by_robot, pose_source_by_robot


def _median_positive_delta_ns(timestamps_ns: list[int]) -> int:
    deltas_ns = [
        next_timestamp_ns - timestamp_ns
        for timestamp_ns, next_timestamp_ns in zip(timestamps_ns, timestamps_ns[1:])
        if next_timestamp_ns > timestamp_ns
    ]
    if not deltas_ns:
        return 0
    return int(statistics.median(deltas_ns))


def _build_robot_render_active_masks(
    rollout: RolloutData,
    render_timestamps_ns: tuple[int, ...],
) -> dict[str, list[bool]]:
    active_masks: dict[str, list[bool]] = {}

    for robot in rollout.robots:
        sample_timestamps_ns = sorted({int(sample.timestamp_ns) for sample in robot.velocity_samples})
        if not sample_timestamps_ns:
            active_masks[robot.name] = [False for _ in render_timestamps_ns]
            continue

        period_ns = _median_positive_delta_ns(sample_timestamps_ns)
        tolerance_ns = max(int(round(period_ns * 0.6)), 1) if period_ns > 0 else 0
        first_active_ns = sample_timestamps_ns[0] - tolerance_ns
        last_active_ns = sample_timestamps_ns[-1] + tolerance_ns
        active_mask = [
            first_active_ns <= int(timestamp_ns) <= last_active_ns
            for timestamp_ns in render_timestamps_ns
        ]
        active_count = sum(active_mask)
        active_masks[robot.name] = active_mask
        print(
            f"[INFO] Rollout {rollout.rollout_id} robot {robot.name}: "
            f"robot-camera render window has {active_count}/{len(render_timestamps_ns)} "
            f"global frames from recorded samples "
            f"[{sample_timestamps_ns[0]}, {sample_timestamps_ns[-1]}].",
            flush=True,
        )

    return active_masks


def _write_replay_pose_csvs(
    rollout: RolloutData,
    render_timestamps_ns: tuple[int, ...],
    sampled_poses_by_robot: dict[str, list[TimedPose]],
    pose_source_by_robot: dict[str, str],
) -> None:
    elapsed_seconds = replay_elapsed_seconds(render_timestamps_ns)
    for robot in rollout.robots:
        output_path = rollout.rollout_dir / f"replay_pose_{robot.name}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "timestamp_ns",
                    "elapsed_s",
                    "x",
                    "y",
                    "z",
                    "yaw",
                    "qx",
                    "qy",
                    "qz",
                    "qw",
                    "pose_source",
                ]
            )
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
                        pose_source_by_robot[robot.name],
                    ]
                )


def _warm_up_cameras(sim_app: Any, live_contexts: list[LiveCameraContext]) -> None:
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


def _resize_rgb_frame(rgb_frame: np.ndarray, output_resolution: tuple[int, int]) -> np.ndarray:
    rgb_array = np.asarray(rgb_frame, dtype=np.uint8)
    image = Image.fromarray(rgb_array)
    if image.size != output_resolution:
        image = image.resize(output_resolution, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _depth_frame_m_to_uint16_mm(depth_frame_m: np.ndarray) -> np.ndarray:
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
    return depth_mm


def _resize_depth_frame_mm(depth_mm: np.ndarray, output_resolution: tuple[int, int]) -> np.ndarray:
    depth_array = np.asarray(depth_mm, dtype=np.uint16)
    image = Image.fromarray(depth_array)
    if image.size != output_resolution:
        image = image.resize(output_resolution, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint16)


def _save_rgb_png(
    output_path: Path,
    rgb_frame: np.ndarray,
    output_resolution: tuple[int, int],
) -> None:
    Image.fromarray(_resize_rgb_frame(rgb_frame, output_resolution)).save(output_path)


def _save_depth_png_mm(
    output_path: Path,
    depth_frame_m: np.ndarray,
    output_resolution: tuple[int, int],
) -> None:
    depth_mm = _depth_frame_m_to_uint16_mm(depth_frame_m)
    Image.fromarray(_resize_depth_frame_mm(depth_mm, output_resolution)).save(output_path)


def _cleanup_camera(context: LiveCameraContext) -> None:
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


def _initialize_live_cameras(live_contexts: list[LiveCameraContext]) -> None:
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


def _render_rollout(
    sim_app: Any,
    builder_module: Any,
    goal_sampler_module: Any,
    rollout: RolloutData,
    camera_settings: CameraSettings,
) -> None:
    rgb_root, depth_root = _ensure_render_outputs_empty(rollout.rollout_dir)
    render_timestamps_ns = _build_render_timestamps_ns(rollout)
    sampled_poses_by_robot, pose_source_by_robot = _prepare_rollout_sampled_poses(
        rollout,
        render_timestamps_ns,
    )
    robot_camera_active_masks = _build_robot_render_active_masks(rollout, render_timestamps_ns)

    _write_replay_pose_csvs(
        rollout,
        render_timestamps_ns,
        sampled_poses_by_robot,
        pose_source_by_robot,
    )

    scene_usd_path = resolve_rollout_scene_usd_path(rollout)
    print(
        f"[INFO] Rollout {rollout.rollout_id}: loading randomized scene USD {scene_usd_path}",
        flush=True,
    )
    with temporary_team_config_file(rollout) as team_config_path:
        robot_infos = builder_module.build_stage(
            str(team_config_path),
            "",
            scene_usd=str(scene_usd_path),
            enable_ros2=False,
        )

    live_contexts = _build_live_camera_contexts(
        rollout,
        robot_infos,
        goal_sampler_module,
        camera_settings,
    )
    camera_output_dirs = _prepare_camera_output_dirs(
        rgb_root,
        depth_root,
        tuple(context.output_name for context in live_contexts),
    )

    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    try:
        timeline.play()
        _initialize_live_cameras(live_contexts)
        for _ in range(DEFAULT_CAMERA_WARMUP_STEPS):
            sim_app.update()
        _warm_up_cameras(sim_app, live_contexts)

        manifest_path = rollout.rollout_dir / "render_manifest.csv"
        elapsed_seconds = replay_elapsed_seconds(render_timestamps_ns)
        with manifest_path.open("w", encoding="utf-8", newline="") as manifest_stream:
            manifest_writer = csv.writer(manifest_stream)
            manifest_writer.writerow(
                [
                    "frame_index",
                    "timestamp_ns",
                    "elapsed_s",
                    "camera_name",
                    "camera_type",
                    "camera_prim_path",
                    "selection_mode",
                    "rgb_path",
                    "depth_path",
                ]
            )
            total_frames = len(render_timestamps_ns)
            for frame_index, (timestamp_ns, elapsed_s) in enumerate(
                zip(render_timestamps_ns, elapsed_seconds)
            ):
                for context in live_contexts:
                    if context.camera_type != "robot":
                        continue
                    if context.robot_name is None or context.root_prim_path is None:
                        raise RuntimeError(f"Robot camera context is incomplete: {context}")
                    if context.root_from_base_matrix is None:
                        raise RuntimeError(f"Robot camera context has no root/base transform: {context}")
                    base_pose = sampled_poses_by_robot[context.robot_name][frame_index]
                    position_xyz, yaw_rad = _compute_root_pose_from_base_pose(
                        base_pose,
                        context.root_from_base_matrix,
                    )
                    builder_module._set_xform_pose(
                        context.root_prim_path,
                        position_xyz,
                        yaw_deg=math.degrees(yaw_rad),
                    )

                for _ in range(DEFAULT_CAPTURE_UPDATES_PER_FRAME):
                    sim_app.update()

                for context in live_contexts:
                    if (
                        context.camera_type == "robot"
                        and context.robot_name is not None
                        and not robot_camera_active_masks[context.robot_name][frame_index]
                    ):
                        continue
                    rgb_frame = _get_rgb_frame(context.camera)
                    depth_frame_m = _get_depth_frame_m(context.camera)
                    if rgb_frame is None:
                        raise RuntimeError(f"RGB frame was unavailable for camera {context.camera_prim_path}")
                    if depth_frame_m is None:
                        raise RuntimeError(f"Depth frame was unavailable for camera {context.camera_prim_path}")
                    if frame_index == 0:
                        print(
                            "[INFO] First rendered frame source shape: "
                            f"camera={context.output_name}, type={context.camera_type}, "
                            f"rgb_shape={tuple(rgb_frame.shape)}, rgb_dtype={rgb_frame.dtype}, "
                            f"depth_shape={tuple(depth_frame_m.shape)}, depth_dtype={depth_frame_m.dtype}, "
                            f"output_resolution={camera_settings.output_resolution}",
                            flush=True,
                        )

                    rgb_output_dir, depth_output_dir = camera_output_dirs[context.output_name]
                    frame_name = f"frame_{frame_index:06d}.png"
                    rgb_path = rgb_output_dir / frame_name
                    depth_path = depth_output_dir / frame_name
                    _save_rgb_png(rgb_path, rgb_frame, camera_settings.output_resolution)
                    _save_depth_png_mm(depth_path, depth_frame_m, camera_settings.output_resolution)
                    manifest_writer.writerow(
                        [
                            frame_index,
                            timestamp_ns,
                            f"{elapsed_s:.9f}",
                            context.camera_name,
                            context.camera_type,
                            context.camera_prim_path,
                            context.selection_mode,
                            rgb_path.relative_to(rollout.rollout_dir).as_posix(),
                            depth_path.relative_to(rollout.rollout_dir).as_posix(),
                        ]
                    )
                if frame_index == 0 or (frame_index + 1) == total_frames or (frame_index + 1) % 100 == 0:
                    print(
                        f"[INFO] Rollout {rollout.rollout_id}: rendered frame {frame_index + 1}/{total_frames}.",
                        flush=True,
                    )
    finally:
        try:
            timeline.stop()
        except Exception as exc:
            print(
                f"[WARN] Failed to stop Isaac timeline cleanly after rollout "
                f"{rollout.rollout_id}: {exc}",
                flush=True,
            )
        for _ in range(2):
            sim_app.update()
        for context in live_contexts:
            _cleanup_camera(context)
        live_contexts.clear()
        gc.collect()
        sim_app.update()


def main() -> int:
    args = _parse_args()
    try:
        rollouts = load_rollouts(args.experiments_root, rollout_ids=args.rollout_ids)
        camera_settings = _load_camera_settings(args.camera_config)
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
            _render_rollout(sim_app, builder_module, sampler_module, rollout, camera_settings)
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
