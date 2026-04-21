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
        set_xform_pose(tag_root, (0.0, 0.0, 0.55), yaw_deg=0.0)

        plaque_path = f"{tag_root}/Plaque"
        plaque = UsdGeom.Cube.Define(stage, plaque_path)
        set_xform_pose(plaque_path, (0.0, 0.0, 0.0), scale_xyz=(0.12, 0.02, 0.18))
        plaque.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.03, 0.03, 0.03)]))

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
        segment_layout = {
            "a": ((0.0, 0.0, 0.07), (0.055, 0.01, 0.01)),
            "b": ((0.045, 0.0, 0.035), (0.01, 0.01, 0.04)),
            "c": ((0.045, 0.0, -0.035), (0.01, 0.01, 0.04)),
            "d": ((0.0, 0.0, -0.07), (0.055, 0.01, 0.01)),
            "e": ((-0.045, 0.0, -0.035), (0.01, 0.01, 0.04)),
            "f": ((-0.045, 0.0, 0.035), (0.01, 0.01, 0.04)),
            "g": ((0.0, 0.0, 0.0), (0.055, 0.01, 0.01)),
        }
        color = Vt.Vec3fArray([Gf.Vec3f(0.95, 0.15, 0.15)])
        segments = seven_segment[int(robot_index % 10)]
        for segment_name in segments:
            center_xyz, scale_xyz = segment_layout[segment_name]
            segment_path = f"{tag_root}/{segment_name}"
            segment_prim = UsdGeom.Cube.Define(stage, segment_path)
            set_xform_pose(segment_path, center_xyz, yaw_deg=0.0, scale_xyz=scale_xyz)
            segment_prim.GetDisplayColorAttr().Set(color)


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
