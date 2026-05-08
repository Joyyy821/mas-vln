#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaac_sim.rendering.render_rollout_rgbd import (  # noqa: E402
    DEFAULT_CAMERA_CONFIG_PATH,
    _build_live_camera_contexts,
    _enable_extension,
    _get_depth_frame_m,
    _get_rgb_frame,
    _initialize_live_cameras,
    _load_builder_module,
    _load_camera_settings,
    _load_goal_sampler_module,
    _robot_fallback_camera_config,
    _save_depth_png_mm,
    _save_rgb_png,
    _warm_up_cameras,
)
from isaac_sim.rendering.rollout_io import (  # noqa: E402
    load_rollout,
    resolve_rollout_scene_usd_path,
    temporary_team_config_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a randomized warehouse scene with rollout robots and ephemeral BEV cameras "
            "for interactive camera placement validation."
        )
    )
    parser.add_argument(
        "scene",
        help=(
            "Scene id such as scene_1, or a scene bundle path containing scene.usd and "
            "rollouts/<id>/run_config.yaml."
        ),
    )
    parser.add_argument(
        "--camera-config",
        default=str(DEFAULT_CAMERA_CONFIG_PATH),
        help=f"Camera settings YAML. Defaults to {DEFAULT_CAMERA_CONFIG_PATH}.",
    )
    parser.add_argument(
        "--rollout-id",
        type=int,
        default=1,
        help="Rollout id to load for robot spawning. Defaults to 1.",
    )
    parser.add_argument(
        "--save-preview-frame",
        nargs="?",
        const="",
        default=None,
        help=(
            "Save one RGB/depth frame from every resolved camera using the same capture path "
            "as the renderer. Optional value is the output directory; by default writes under "
            "<scene-bundle>/camera_validation_preview."
        ),
    )
    return parser.parse_args()


def _create_gui_sim_app() -> Any:
    try:
        from isaacsim.simulation_app import SimulationApp
    except Exception:
        from omni.isaac.kit import SimulationApp

    return SimulationApp({"headless": False, "anti_aliasing": 0})


def _resolve_scene_bundle(scene_value: str) -> Path:
    raw_path = Path(scene_value).expanduser()
    candidates = []
    if raw_path.is_absolute() or raw_path.exists():
        candidates.append(raw_path)
    candidates.append(REPO_ROOT / "experiments" / "randomized_warehouse" / scene_value)

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "scene.usd").is_file():
            return candidate.resolve()
    candidate_labels = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Unable to resolve scene bundle from {scene_value!r}: {candidate_labels}")


def _matrix_from_gf(gf_matrix: Any) -> np.ndarray:
    return np.array([[float(gf_matrix[i][j]) for j in range(4)] for i in range(4)], dtype=float).T


def _camera_world_pose_snippet(camera_prim_path: str, fallback_config: dict[str, Any]) -> dict[str, Any]:
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Camera prim was not found: {camera_prim_path}")

    matrix = _matrix_from_gf(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    )
    position = matrix[:3, 3]
    forward = matrix[:3, :3] @ np.array([0.0, 0.0, -1.0], dtype=float)
    if abs(float(forward[2])) > 1e-6:
        target = position + forward * ((0.0 - position[2]) / forward[2])
    else:
        target = position + forward

    camera = UsdGeom.Camera(prim)
    return {
        "name": fallback_config["name"],
        "position_xyz": [round(float(value), 6) for value in position],
        "target_xyz": [round(float(value), 6) for value in target],
        "up_axis": fallback_config.get("up_axis", [0.0, 1.0, 0.0]),
        "focal_length_mm": float(camera.GetFocalLengthAttr().Get() or 0.0),
        "horizontal_aperture_mm": float(camera.GetHorizontalApertureAttr().Get() or 0.0),
        "vertical_aperture_mm": float(camera.GetVerticalApertureAttr().Get() or 0.0),
        "clipping_range_m": [
            float(value) for value in (camera.GetClippingRangeAttr().Get() or [0.1, 100.0])
        ],
    }


def _camera_local_pose_snippet(
    camera_prim_path: str,
    robot_model: str,
    fallback_config: dict[str, Any],
) -> dict[str, Any]:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Camera prim was not found: {camera_prim_path}")

    local_transform = UsdGeom.Xformable(prim).GetLocalTransformation()
    if isinstance(local_transform, tuple):
        local_transform = local_transform[0]
    matrix = _matrix_from_gf(local_transform)

    position = matrix[:3, 3]
    forward = matrix[:3, :3] @ np.array([0.0, 0.0, -1.0], dtype=float)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm > 1e-9:
        forward = forward / forward_norm

    up_axis = matrix[:3, :3] @ np.array([0.0, 1.0, 0.0], dtype=float)
    up_norm = float(np.linalg.norm(up_axis))
    if up_norm > 1e-9:
        up_axis = up_axis / up_norm

    default_position = np.asarray(fallback_config.get("position_xyz", [0.0, 0.0, 0.0]), dtype=float)
    default_target = np.asarray(fallback_config.get("target_xyz", [1.0, 0.0, 0.0]), dtype=float)
    target_distance = max(float(np.linalg.norm(default_target - default_position)), 1.0)
    target = position + forward * target_distance

    camera = UsdGeom.Camera(prim)
    return {
        "mode": "virtual_only",
        "prim_name": prim.GetName(),
        "position_xyz": [round(float(value), 6) for value in position],
        "target_xyz": [round(float(value), 6) for value in target],
        "up_axis": [round(float(value), 6) for value in up_axis],
        "focal_length_mm": float(camera.GetFocalLengthAttr().Get() or 0.0),
        "horizontal_aperture_mm": float(camera.GetHorizontalApertureAttr().Get() or 0.0),
        "vertical_aperture_mm": float(camera.GetVerticalApertureAttr().Get() or 0.0),
        "clipping_range_m": [
            float(value) for value in (camera.GetClippingRangeAttr().Get() or [0.05, 100.0])
        ],
    }


def _print_bev_yaml_snippet(camera_config_path: str | Path) -> None:
    settings_path = Path(camera_config_path).expanduser()
    with settings_path.open("r", encoding="utf-8") as stream:
        raw_settings = yaml.safe_load(stream) or {}

    updated_cameras = []
    for camera in raw_settings.get("bev_cameras", []) or []:
        name = str(camera.get("name", "")).strip()
        if not name:
            continue
        prim_path = f"/World/RenderCameras/bev_{name}"
        try:
            updated_cameras.append(_camera_world_pose_snippet(prim_path, camera))
        except Exception as exc:
            print(f"[WARN] Could not read live BEV camera {name}: {exc}", flush=True)

    if not updated_cameras:
        return
    print("\n[INFO] Ready-to-paste BEV camera YAML snippet:", flush=True)
    print(yaml.safe_dump({"bev_cameras": updated_cameras}, sort_keys=False), flush=True)


def _plain_yaml_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain_yaml_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_yaml_value(item) for item in value]
    return value


def _print_virtual_robot_camera_yaml_snippet(contexts: list[Any], camera_settings: Any) -> None:
    model_overrides: dict[str, dict[str, Any]] = {}

    for context in contexts:
        if context.camera_type != "robot" or context.selection_mode != "fallback_virtual":
            continue
        if not context.robot_model:
            continue
        fallback_config = _robot_fallback_camera_config(camera_settings, context.robot_model)
        try:
            model_overrides[context.robot_model] = _camera_local_pose_snippet(
                context.camera_prim_path,
                context.robot_model,
                fallback_config,
            )
        except Exception as exc:
            print(
                f"[WARN] Could not read live virtual camera for {context.robot_model}: {exc}",
                flush=True,
            )

    if not model_overrides:
        return
    merged_model_overrides = _plain_yaml_value(camera_settings.robot_camera.model_overrides)
    merged_model_overrides.update(model_overrides)
    robot_camera_payload = {
        "mode": camera_settings.robot_camera.mode,
        "asset_camera_reject_tokens": list(camera_settings.robot_camera.asset_camera_reject_tokens),
        "fallback_virtual_camera": _plain_yaml_value(
            camera_settings.robot_camera.fallback_virtual_camera
        ),
        "model_overrides": merged_model_overrides,
    }

    print("\n[INFO] Ready-to-paste full robot_camera YAML block:", flush=True)
    print(
        yaml.safe_dump(
            {
                "robot_camera": robot_camera_payload,
            },
            sort_keys=False,
        ),
        flush=True,
    )


def _save_preview_frames(
    *,
    scene_bundle: Path,
    requested_output_dir: str | None,
    contexts: list[Any],
    output_resolution: tuple[int, int],
) -> None:
    output_dir = (
        scene_bundle / "camera_validation_preview"
        if not requested_output_dir
        else Path(requested_output_dir).expanduser()
    )
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    rgb_root = output_dir / "rgb"
    depth_root = output_dir / "depth"
    rgb_root.mkdir(parents=True, exist_ok=True)
    depth_root.mkdir(parents=True, exist_ok=True)

    for context in contexts:
        rgb_frame = _get_rgb_frame(context.camera)
        depth_frame_m = _get_depth_frame_m(context.camera)
        if rgb_frame is None or depth_frame_m is None:
            print(
                f"[WARN] Skipping preview for {context.output_name}: "
                f"rgb_available={rgb_frame is not None}, depth_available={depth_frame_m is not None}",
                flush=True,
            )
            continue
        print(
            "[INFO] Validation preview source shape: "
            f"camera={context.output_name}, type={context.camera_type}, "
            f"rgb_shape={tuple(rgb_frame.shape)}, rgb_dtype={rgb_frame.dtype}, "
            f"depth_shape={tuple(depth_frame_m.shape)}, depth_dtype={depth_frame_m.dtype}, "
            f"output_resolution={output_resolution}",
            flush=True,
        )
        _save_rgb_png(rgb_root / f"{context.output_name}.png", rgb_frame, output_resolution)
        _save_depth_png_mm(depth_root / f"{context.output_name}.png", depth_frame_m, output_resolution)

    print(f"[INFO] Saved camera validation preview frames under {output_dir}", flush=True)


def main() -> int:
    args = _parse_args()
    scene_bundle = _resolve_scene_bundle(args.scene)
    rollout_dir = scene_bundle / "rollouts" / str(args.rollout_id)
    rollout = load_rollout(rollout_dir)
    camera_settings = _load_camera_settings(args.camera_config)
    scene_usd_path = resolve_rollout_scene_usd_path(rollout)

    sim_app = _create_gui_sim_app()
    _enable_extension("isaacsim.sensors.camera")
    for _ in range(8):
        sim_app.update()

    builder_module = _load_builder_module()
    goal_sampler_module = _load_goal_sampler_module()
    contexts = []

    try:
        with temporary_team_config_file(rollout) as team_config_path:
            robot_infos = builder_module.build_stage(
                str(team_config_path),
                "",
                scene_usd=str(scene_usd_path),
                enable_ros2=False,
            )
        contexts = _build_live_camera_contexts(
            rollout,
            robot_infos,
            goal_sampler_module,
            camera_settings,
        )

        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        _initialize_live_cameras(contexts)
        _warm_up_cameras(sim_app, contexts)
        if args.save_preview_frame is not None:
            _save_preview_frames(
                scene_bundle=scene_bundle,
                requested_output_dir=args.save_preview_frame,
                contexts=contexts,
                output_resolution=camera_settings.output_resolution,
            )

        print(
            "[INFO] Validation scene is open. Move BEV camera prims under /World/RenderCameras "
            "or fallback robot camera prims under /World/Robots/*/*/RenderFallbackCamera, "
            "then close Isaac Sim to print updated YAML.",
            flush=True,
        )
        while sim_app.is_running():
            sim_app.update()
    finally:
        _print_bev_yaml_snippet(args.camera_config)
        _print_virtual_robot_camera_yaml_snippet(contexts, camera_settings)
        for context in contexts:
            destroy = getattr(context.camera, "destroy", None)
            if callable(destroy):
                try:
                    destroy()
                except Exception:
                    pass
        sim_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
