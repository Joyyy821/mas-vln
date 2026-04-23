from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from isaac_sim.stage_bringups.runtime_utils import (
    add_reference,
    compute_world_bbox,
    ensure_xform_path,
    get_stage,
    iter_prims_under,
    set_xform_pose,
)


@dataclass
class RuntimeRobotController:
    model_id: str
    namespace: str
    root_prim_path: str
    base_frame_id: str
    odom_frame_id: str
    articulation: Any
    articulation_controller: Any
    wheel_joint_indices: np.ndarray
    wheel_radius_m: float
    wheel_distance_m: float
    planar_radius_m: float
    max_linear_speed_mps: float
    max_angular_speed_rps: float
    cmd_timeout_sec: float


class RobotAdapter:
    model_id = "robot"
    sensorless_usd_rel = ""
    default_planar_radius_m = 0.45
    default_wheel_radius_m = 0.14
    default_wheel_distance_m = 0.54
    max_linear_speed_mps = 1.5
    max_angular_speed_rps = 1.8
    cmd_timeout_sec = 0.5
    base_frame_id = "base_link"
    odom_frame_id = "odom"

    def resolve_usd_path(self, assets_root_path: str) -> str:
        if not self.sensorless_usd_rel:
            raise NotImplementedError(f"{self.model_id} does not define a sensorless USD path.")
        return f"{assets_root_path}{self.sensorless_usd_rel}"

    def spawn_robot(
        self,
        *,
        assets_root_path: str,
        prim_path: str,
        robot_index: int,
        position_xyz: tuple[float, float, float],
        yaw_deg: float,
    ) -> None:
        add_reference(self.resolve_usd_path(assets_root_path), prim_path)
        set_xform_pose(prim_path, position_xyz, yaw_deg=yaw_deg)
        self.attach_numeric_id_tag(prim_path, robot_index)

    def initialize_runtime_controller(
        self,
        *,
        sim_app,
        namespace: str,
        root_prim_path: str,
    ) -> RuntimeRobotController:
        try:
            from isaacsim.core.prims import SingleArticulation
        except Exception:
            from omni.isaac.core.prims import SingleArticulation

        articulation = SingleArticulation(prim_path=root_prim_path, name=f"{namespace}_{self.model_id}")
        articulation.initialize()
        articulation_controller = articulation.get_articulation_controller()

        left_joint_name, right_joint_name = self._discover_wheel_joint_names(
            root_prim_path=root_prim_path,
            articulation=articulation,
        )
        wheel_joint_indices = np.array(
            [
                articulation.get_dof_index(left_joint_name),
                articulation.get_dof_index(right_joint_name),
            ],
            dtype=np.int32,
        )

        wheel_radius_m, wheel_distance_m = self._discover_wheel_geometry(root_prim_path)
        return RuntimeRobotController(
            model_id=self.model_id,
            namespace=namespace,
            root_prim_path=root_prim_path,
            base_frame_id=self.base_frame_id,
            odom_frame_id=self.odom_frame_id,
            articulation=articulation,
            articulation_controller=articulation_controller,
            wheel_joint_indices=wheel_joint_indices,
            wheel_radius_m=wheel_radius_m,
            wheel_distance_m=wheel_distance_m,
            planar_radius_m=float(self.default_planar_radius_m),
            max_linear_speed_mps=float(self.max_linear_speed_mps),
            max_angular_speed_rps=float(self.max_angular_speed_rps),
            cmd_timeout_sec=float(self.cmd_timeout_sec),
        )

    def _discover_joint_names_from_articulation(self, articulation: Any) -> list[str]:
        articulation_view = getattr(articulation, "_articulation_view", None)
        metadata = getattr(articulation_view, "_metadata", None)
        joint_names = list(getattr(metadata, "joint_names", []) or [])
        if joint_names:
            return [str(name) for name in joint_names]
        return []

    def _discover_wheel_joint_names(self, *, root_prim_path: str, articulation: Any) -> tuple[str, str]:
        joint_names = self._discover_joint_names_from_articulation(articulation)
        if not joint_names:
            for prim in iter_prims_under(root_prim_path):
                prim_name = prim.GetName()
                if "joint" in prim_name.lower():
                    joint_names.append(str(prim_name))

        lower_to_original = {name.lower(): name for name in joint_names}

        def _match(patterns: list[str]) -> str | None:
            for pattern in patterns:
                regex = re.compile(pattern)
                for lower_name, original_name in lower_to_original.items():
                    if regex.search(lower_name):
                        return original_name
            return None

        left_name = _match([r"left.*wheel", r"wheel.*left", r"left.*hub"])
        right_name = _match([r"right.*wheel", r"wheel.*right", r"right.*hub"])
        if left_name and right_name and left_name != right_name:
            return left_name, right_name

        wheel_like = [
            name for name in joint_names if "wheel" in name.lower() or "hub" in name.lower()
        ]
        if len(wheel_like) >= 2:
            return wheel_like[0], wheel_like[1]

        if len(joint_names) >= 2:
            return joint_names[0], joint_names[1]

        raise RuntimeError(
            f"Unable to infer differential-drive wheel joints under {root_prim_path}."
        )

    def _discover_wheel_geometry(self, root_prim_path: str) -> tuple[float, float]:
        wheel_candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
        for prim in iter_prims_under(root_prim_path):
            name = prim.GetName().lower()
            if "wheel" not in name:
                continue
            try:
                bbox_min, bbox_max = compute_world_bbox(prim.GetPath().pathString)
            except Exception:
                continue
            wheel_candidates.append((name, bbox_min, bbox_max))

        left_center = None
        right_center = None
        radius_candidates: list[float] = []
        for name, bbox_min, bbox_max in wheel_candidates:
            size = bbox_max - bbox_min
            center = 0.5 * (bbox_min + bbox_max)
            radius_candidates.append(float(np.sort(size)[1] * 0.5))
            if left_center is None and "left" in name:
                left_center = center
            if right_center is None and "right" in name:
                right_center = center

        wheel_radius_m = (
            float(np.median(radius_candidates))
            if radius_candidates
            else float(self.default_wheel_radius_m)
        )
        if left_center is not None and right_center is not None:
            wheel_distance_m = float(np.linalg.norm(left_center[:2] - right_center[:2]))
        else:
            wheel_distance_m = float(self.default_wheel_distance_m)

        wheel_radius_m = max(wheel_radius_m, 0.02)
        wheel_distance_m = max(wheel_distance_m, 0.05)
        return wheel_radius_m, wheel_distance_m

    def attach_numeric_id_tag(self, root_prim_path: str, robot_index: int) -> None:
        from pxr import Gf, UsdGeom, Vt

        stage = get_stage()
        tag_root = f"{root_prim_path}/IdTag"
        ensure_xform_path(tag_root)
        set_xform_pose(tag_root, (0.0, 0.0, 0.78), yaw_deg=0.0)

        white = Vt.Vec3fArray([Gf.Vec3f(0.98, 0.98, 0.96)])
        black = Vt.Vec3fArray([Gf.Vec3f(0.06, 0.06, 0.06)])
        accent = Vt.Vec3fArray([Gf.Vec3f(0.96, 0.46, 0.12)])

        def _define_cube(path: str, position_xyz: tuple[float, float, float], scale_xyz: tuple[float, float, float], color) -> None:
            cube = UsdGeom.Cube.Define(stage, path)
            set_xform_pose(path, position_xyz, yaw_deg=0.0, scale_xyz=scale_xyz)
            cube.GetDisplayColorAttr().Set(color)

        seven_segment = {
            0: ("a", "b", "c", "d", "e", "f"),
            1: ("b", "c"),
            2: ("a", "b", "g", "e", "d"),
            3: ("a", "b", "g", "c", "d"),
            4: ("f", "g", "b", "c"),
            5: ("a", "f", "g", "c", "d"),
            6: ("a", "f", "g", "c", "d", "e"),
            7: ("a", "b", "c"),
            8: ("a", "b", "c", "d", "e", "f", "g"),
            9: ("a", "b", "c", "d", "f", "g"),
        }

        def _segment_layout_for_plane(plane: str, *, mirror_horizontal: bool) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
            x_sign = -1.0 if mirror_horizontal else 1.0
            if plane == "xy":
                return {
                    "a": ((0.0, 0.055, 0.0), (0.055, 0.008, 0.004)),
                    "b": ((0.040 * x_sign, 0.028, 0.0), (0.008, 0.040, 0.004)),
                    "c": ((0.040 * x_sign, -0.028, 0.0), (0.008, 0.040, 0.004)),
                    "d": ((0.0, -0.055, 0.0), (0.055, 0.008, 0.004)),
                    "e": ((-0.040 * x_sign, -0.028, 0.0), (0.008, 0.040, 0.004)),
                    "f": ((-0.040 * x_sign, 0.028, 0.0), (0.008, 0.040, 0.004)),
                    "g": ((0.0, 0.0, 0.0), (0.055, 0.008, 0.004)),
                }
            if plane == "xz":
                return {
                    "a": ((0.0, 0.0, 0.050), (0.055, 0.004, 0.008)),
                    "b": ((0.040 * x_sign, 0.0, 0.025), (0.008, 0.004, 0.034)),
                    "c": ((0.040 * x_sign, 0.0, -0.025), (0.008, 0.004, 0.034)),
                    "d": ((0.0, 0.0, -0.050), (0.055, 0.004, 0.008)),
                    "e": ((-0.040 * x_sign, 0.0, -0.025), (0.008, 0.004, 0.034)),
                    "f": ((-0.040 * x_sign, 0.0, 0.025), (0.008, 0.004, 0.034)),
                    "g": ((0.0, 0.0, 0.0), (0.055, 0.004, 0.008)),
                }
            return {
                "a": ((0.0, 0.050, 0.0), (0.004, 0.055, 0.008)),
                "b": ((0.0, 0.025 * x_sign, 0.040), (0.004, 0.034, 0.008)),
                "c": ((0.0, -0.025 * x_sign, 0.040), (0.004, 0.034, 0.008)),
                "d": ((0.0, -0.050, 0.0), (0.004, 0.055, 0.008)),
                "e": ((0.0, -0.025 * x_sign, -0.040), (0.004, 0.034, 0.008)),
                "f": ((0.0, 0.025 * x_sign, -0.040), (0.004, 0.034, 0.008)),
                "g": ((0.0, 0.0, 0.0), (0.004, 0.055, 0.008)),
            }

        def _build_digit(panel_path: str, plane: str, *, mirror_horizontal: bool) -> None:
            segment_layout = _segment_layout_for_plane(plane, mirror_horizontal=mirror_horizontal)
            for segment_name in segments:
                center_xyz, scale_xyz = segment_layout[segment_name]
                _define_cube(
                    f"{panel_path}/{segment_name}",
                    center_xyz,
                    scale_xyz,
                    black,
                )

        segments = seven_segment[int(robot_index % 10)]

        _define_cube(f"{tag_root}/Mast", (0.0, 0.0, -0.16), (0.015, 0.015, 0.20), accent)
        _define_cube(f"{tag_root}/TopPlate", (0.0, 0.0, 0.07), (0.175, 0.175, 0.012), white)
        _define_cube(f"{tag_root}/TopRim", (0.0, 0.0, 0.084), (0.185, 0.185, 0.004), accent)

        top_digit_root = f"{tag_root}/TopDigit"
        ensure_xform_path(top_digit_root)
        set_xform_pose(top_digit_root, (0.0, 0.0, 0.085), yaw_deg=0.0)
        _build_digit(top_digit_root, "xy", mirror_horizontal=False)

        side_panels = (
            ("FrontPanel", (0.0, 0.105, 0.01), (0.170, 0.012, 0.125), "xz", False),
            ("BackPanel", (0.0, -0.105, 0.01), (0.170, 0.012, 0.125), "xz", True),
            ("LeftPanel", (-0.105, 0.0, 0.01), (0.012, 0.170, 0.125), "yz", False),
            ("RightPanel", (0.105, 0.0, 0.01), (0.012, 0.170, 0.125), "yz", True),
        )
        for panel_name, position_xyz, scale_xyz, plane, mirror_horizontal in side_panels:
            panel_path = f"{tag_root}/{panel_name}"
            _define_cube(panel_path, position_xyz, scale_xyz, white)
            digit_root = f"{panel_path}/Digit"
            ensure_xform_path(digit_root)
            set_xform_pose(digit_root, (0.0, 0.0, 0.0), yaw_deg=0.0)
            _build_digit(digit_root, plane, mirror_horizontal=mirror_horizontal)


class NovaCarterAdapter(RobotAdapter):
    model_id = "nova_carter"
    sensorless_usd_rel = "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
    default_planar_radius_m = 0.45
    default_wheel_radius_m = 0.14
    default_wheel_distance_m = 0.54
    max_linear_speed_mps = 1.5
    max_angular_speed_rps = 1.8
    cmd_timeout_sec = 0.5


class JackalAdapter(RobotAdapter):
    model_id = "jackal"

    def resolve_usd_path(self, assets_root_path: str) -> str:
        raise NotImplementedError(
            "Jackal adapter scaffolding is present, but v1 randomized bringup only fully supports Nova Carter."
        )


class Turtlebot3Adapter(RobotAdapter):
    model_id = "turtlebot3"

    def resolve_usd_path(self, assets_root_path: str) -> str:
        raise NotImplementedError(
            "TurtleBot3 adapter scaffolding is present, but it still needs a dedicated import/control path."
        )


def build_robot_adapter(model_id: str) -> RobotAdapter:
    clean_model_id = str(model_id).strip().lower()
    if clean_model_id == "nova_carter":
        return NovaCarterAdapter()
    if clean_model_id == "jackal":
        return JackalAdapter()
    if clean_model_id in {"turtlebot3", "turtlebot"}:
        return Turtlebot3Adapter()
    raise ValueError(f"Unsupported robot model '{model_id}'.")
