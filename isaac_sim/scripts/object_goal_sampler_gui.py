from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from object_goal_sampler_utils import (
    DEFAULT_MANUAL_GOAL_ROBOT_NAME,
    ObjectBBox3D,
    RobotCameraSpec,
    SampledGoalPose,
    _rotation_matrix_z,
    _translation_matrix,
    _yaw_from_matrix,
    format_team_config_goal_yaml,
    pose_array_xyzw_from_position_yaw,
)


class GoalSamplerGuiMixin:
    @staticmethod
    def _load_warehouse_builder_module() -> Any:
        script_dir = Path(__file__).resolve().parent
        builder_path = script_dir.parent / "stage_bringups" / "build_stage_warehouse_carters.py"
        spec = importlib.util.spec_from_file_location("build_stage_warehouse_carters", builder_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import stage builder from {builder_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _apply_xform_ops(
        self,
        prim: Any,
        *,
        translation_xyz: Sequence[float] | None = None,
        yaw_deg: float | None = None,
        scale_xyz: Sequence[float] | None = None,
    ) -> None:
        _, UsdGeom, Gf = self._ensure_usd_imports()
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()

        if translation_xyz is not None:
            op = xformable.AddTranslateOp()
            op.Set(Gf.Vec3d(*[float(value) for value in translation_xyz]))
        if yaw_deg is not None:
            op = xformable.AddRotateXYZOp()
            op.Set(Gf.Vec3f(0.0, 0.0, float(yaw_deg)))
        if scale_xyz is not None:
            op = xformable.AddScaleOp()
            op.Set(Gf.Vec3f(*[float(value) for value in scale_xyz]))

    def _add_live_debug_markers(
        self,
        stage: Any,
        sampled_poses: Sequence[SampledGoalPose],
        chosen_pose: SampledGoalPose,
        object_bbox: ObjectBBox3D,
    ) -> None:
        _, UsdGeom, Gf = self._ensure_usd_imports()

        debug_root = UsdGeom.Xform.Define(stage, "/World/Debug")
        del debug_root

        target_sphere = UsdGeom.Sphere.Define(stage, "/World/Debug/ObjectTarget")
        target_sphere.CreateRadiusAttr(0.12)
        target_sphere.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.2, 0.2)])
        self._apply_xform_ops(
            target_sphere.GetPrim(),
            translation_xyz=chosen_pose.target_point_xyz,
        )

        bbox_visual = UsdGeom.Cube.Define(stage, "/World/Debug/ObjectBBox")
        bbox_visual.CreateSizeAttr(1.0)
        bbox_visual.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.7, 0.2)])
        bbox_visual.CreateDisplayOpacityAttr([0.15])
        bbox_center = object_bbox.center_xyz
        bbox_size = np.maximum(object_bbox.size_xyz, 0.02)
        self._apply_xform_ops(
            bbox_visual.GetPrim(),
            translation_xyz=bbox_center,
            scale_xyz=bbox_size,
        )

        for index, sample in enumerate(sampled_poses):
            sphere = UsdGeom.Sphere.Define(stage, f"/World/Debug/Sample_{index:02d}")
            sphere.CreateRadiusAttr(0.08)
            if sample is chosen_pose:
                sphere.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.2)])
            else:
                sphere.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.4, 1.0)])
            self._apply_xform_ops(
                sphere.GetPrim(),
                translation_xyz=sample.base_position_xyz,
            )

    @staticmethod
    def _maybe_set_viewport_camera(camera_prim_path: str) -> None:
        try:  # pragma: no cover - GUI-only Isaac Sim utility.
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = camera_prim_path
        except Exception:
            pass

    @staticmethod
    def _root_pose_for_base_pose(
        base_position_xyz: Sequence[float],
        base_yaw_rad: float,
        robot_spec: RobotCameraSpec,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        base_world = _translation_matrix(base_position_xyz) @ _rotation_matrix_z(base_yaw_rad)
        root_world = base_world @ np.linalg.inv(robot_spec.root_from_base_matrix)
        root_position_xyz = root_world[:3, 3].copy()
        root_yaw_rad = _yaw_from_matrix(root_world)
        return root_position_xyz, root_yaw_rad, root_world

    def _capture_live_base_pose(
        self,
        stage: Any,
        robot_prim_path: str,
        robot_spec: RobotCameraSpec,
        fallback_pose: SampledGoalPose,
    ) -> tuple[np.ndarray, float]:
        live_base_pose = self._read_live_base_pose(stage, robot_prim_path, robot_spec)
        if live_base_pose is None:
            print(
                "[WARN] Live base prim was not found for manual goal capture. "
                "Falling back to the sampled pose.",
                flush=True,
            )
            return fallback_pose.base_position_xyz.copy(), float(fallback_pose.yaw_rad)

        live_base_path, live_base_position_xyz, live_base_yaw_rad = live_base_pose
        self._print_captured_live_base_pose(
            live_base_path,
            live_base_position_xyz,
            live_base_yaw_rad,
        )
        return live_base_position_xyz, live_base_yaw_rad

    def _read_live_base_pose(
        self,
        stage: Any,
        robot_prim_path: str,
        robot_spec: RobotCameraSpec,
    ) -> tuple[str, np.ndarray, float] | None:
        live_base_path = robot_prim_path + robot_spec.base_rel_path_from_root
        live_base_prim = stage.GetPrimAtPath(live_base_path)
        if not live_base_prim or not live_base_prim.IsValid():
            return None

        live_base_world = self._compute_world_transform_matrix(live_base_prim)
        live_base_position_xyz = live_base_world[:3, 3].copy()
        live_base_yaw_rad = _yaw_from_matrix(live_base_world)
        return live_base_path, live_base_position_xyz, live_base_yaw_rad

    @staticmethod
    def _print_captured_live_base_pose(
        live_base_path: str,
        live_base_position_xyz: Sequence[float],
        live_base_yaw_rad: float,
    ) -> None:
        print(
            "[INFO] Captured live base pose at GUI exit: "
            f"base_path={live_base_path}, "
            f"position_xyz={list(live_base_position_xyz)}, "
            f"yaw_deg={math.degrees(live_base_yaw_rad):.2f}",
            flush=True,
        )

    @staticmethod
    def _print_manual_goal_output(
        robot_name: str,
        base_position_xyz: Sequence[float],
        base_yaw_rad: float,
    ) -> None:
        goal_yaml = format_team_config_goal_yaml(robot_name, base_position_xyz, base_yaw_rad)
        goal_pose_array = pose_array_xyzw_from_position_yaw(base_position_xyz, base_yaw_rad)

        print("[INFO] Manual goal snippet for warehouse_team_config.yaml:", flush=True)
        print(goal_yaml, flush=True)
        print(
            "[INFO] Goal pose array [x, y, z, qx, qy, qz, qw]: "
            f"{json.dumps(goal_pose_array)}",
            flush=True,
        )

    def validate_samples_in_gui(
        self,
        *,
        environment_usd_path: str,
        robot_usd_path: str,
        sampled_poses: Sequence[SampledGoalPose],
        object_bbox: ObjectBBox3D,
        base_prim_path: str | None = None,
        camera_prim_path: str | None = None,
        selected_sample_index: int | None = None,
        robot_name: str = DEFAULT_MANUAL_GOAL_ROBOT_NAME,
        set_viewport_to_robot_camera: bool = True,
        play_timeline: bool = True,
        emit_goal_on_exit: bool = True,
    ) -> SampledGoalPose:
        if not sampled_poses:
            raise ValueError("At least one sampled pose is required for GUI validation.")

        self._ensure_runtime(require_gui=True)
        print("[INFO] Preparing GUI validation stage...", flush=True)
        resolved_environment_usd_path = self._resolve_runtime_asset_path(environment_usd_path)
        resolved_robot_usd_path = self._resolve_runtime_asset_path(robot_usd_path)
        spec = self.inspect_robot_camera_spec(
            resolved_robot_usd_path,
            base_prim_path=base_prim_path,
            camera_prim_path=camera_prim_path,
        )

        if selected_sample_index is None:
            selected_sample_index = int(self._rng.integers(0, len(sampled_poses)))
        if selected_sample_index < 0 or selected_sample_index >= len(sampled_poses):
            raise IndexError(
                f"selected_sample_index={selected_sample_index} is outside [0, {len(sampled_poses) - 1}]"
            )
        chosen_pose = sampled_poses[selected_sample_index]
        root_position_xyz, root_yaw_rad, _ = self._root_pose_for_base_pose(
            chosen_pose.base_position_xyz,
            chosen_pose.yaw_rad,
            spec,
        )

        builder = self._load_warehouse_builder_module()

        sim_app = self._simulation_app
        if sim_app is None:
            raise RuntimeError("GUI validation requested, but the SimulationApp is not available.")
        try:  # pragma: no cover - requires Isaac Sim GUI runtime.
            import omni.timeline
            import omni.usd

            builder._new_stage()
            builder._set_stage_units(1.0)
            builder._define_xform("/World")
            builder._define_xform("/World/Env")
            builder._define_xform("/World/Robots")

            env_prim_path = "/World/Env/Environment"
            robot_prim_path = "/World/Robots/Robot_1"
            builder._add_reference(resolved_environment_usd_path, env_prim_path)
            builder._add_reference(resolved_robot_usd_path, robot_prim_path)
            builder._set_xform_pose(
                robot_prim_path,
                tuple(float(value) for value in root_position_xyz.tolist()),
                yaw_deg=math.degrees(root_yaw_rad),
            )

            for _ in range(20):
                sim_app.update()

            stage = omni.usd.get_context().get_stage()
            self._add_live_debug_markers(stage, sampled_poses, chosen_pose, object_bbox)

            live_camera_path = robot_prim_path + spec.camera_rel_path_from_root
            if set_viewport_to_robot_camera:
                self._maybe_set_viewport_camera(live_camera_path)

            if play_timeline:
                timeline = omni.timeline.get_timeline_interface()
                timeline.play()

            print(
                "[INFO] GUI validation running with sample "
                f"{selected_sample_index}: {json.dumps(chosen_pose.as_dict(), indent=2)}",
                flush=True,
            )
            print(
                "[INFO] GUI validation root pose derived from sampled base pose: "
                f"root_position_xyz={root_position_xyz.tolist()}, "
                f"root_yaw_deg={math.degrees(root_yaw_rad):.2f}, "
                f"sampled_base_yaw_deg={math.degrees(chosen_pose.yaw_rad):.2f}",
                flush=True,
            )
            print(
                f"[INFO] Target object bbox: {json.dumps(object_bbox.as_dict(), indent=2)}",
                flush=True,
            )
            print(f"[INFO] Robot camera prim path: {live_camera_path}", flush=True)

            live_camera_prim = stage.GetPrimAtPath(live_camera_path)
            if live_camera_prim and live_camera_prim.IsValid():
                live_camera_world = self._compute_world_transform_matrix(live_camera_prim)
                live_camera_xyz = live_camera_world[:3, 3].copy()
                live_camera_forward_world = (
                    live_camera_world[:3, :3] @ np.array([0.0, 0.0, -1.0], dtype=float)
                )
                target_direction_world = chosen_pose.target_point_xyz - live_camera_xyz
                target_direction_norm = float(np.linalg.norm(target_direction_world))
                forward_norm = float(np.linalg.norm(live_camera_forward_world))
                if target_direction_norm > 1e-6 and forward_norm > 1e-6:
                    cosine = float(
                        np.clip(
                            np.dot(
                                live_camera_forward_world / forward_norm,
                                target_direction_world / target_direction_norm,
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                    angle_error_deg = math.degrees(math.acos(cosine))
                    print(
                        "[INFO] Live camera/target alignment: "
                        f"camera_xyz={live_camera_xyz.tolist()}, "
                        f"angle_error_deg={angle_error_deg:.2f}",
                        flush=True,
                    )

            for _ in range(30):
                sim_app.update()

            if not sim_app.is_running():
                print(
                    "[WARN] Isaac Sim GUI runtime did not report a running window after stage setup. "
                    "This usually means the process is not attached to a display, the wrong Python "
                    "runtime is being used, or the GUI closed immediately during startup.",
                    flush=True,
                )
                if emit_goal_on_exit:
                    self._print_manual_goal_output(
                        robot_name,
                        chosen_pose.base_position_xyz,
                        chosen_pose.yaw_rad,
                    )
                return chosen_pose

            last_live_base_pose = self._read_live_base_pose(stage, robot_prim_path, spec)
            while sim_app.is_running():
                sim_app.update()
                live_base_pose = self._read_live_base_pose(stage, robot_prim_path, spec)
                if live_base_pose is not None:
                    last_live_base_pose = live_base_pose

            if emit_goal_on_exit:
                if last_live_base_pose is not None:
                    live_base_path, live_base_position_xyz, live_base_yaw_rad = last_live_base_pose
                    self._print_captured_live_base_pose(
                        live_base_path,
                        live_base_position_xyz,
                        live_base_yaw_rad,
                    )
                else:
                    live_base_position_xyz, live_base_yaw_rad = self._capture_live_base_pose(
                        stage,
                        robot_prim_path,
                        spec,
                        chosen_pose,
                    )
                self._print_manual_goal_output(
                    robot_name,
                    live_base_position_xyz,
                    live_base_yaw_rad,
                )
        finally:
            self.close()

        return chosen_pose
