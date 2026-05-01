from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from isaac_sim.stage_bringups.runtime_utils import (
    add_reference,
    compute_world_bbox,
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
    left_wheel_joint_indices: np.ndarray
    right_wheel_joint_indices: np.ndarray
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

        left_joint_names, right_joint_names = self._discover_wheel_joint_names(
            root_prim_path=root_prim_path,
            articulation=articulation,
        )
        left_wheel_joint_indices = np.array(
            [articulation.get_dof_index(name) for name in left_joint_names],
            dtype=np.int32,
        )
        right_wheel_joint_indices = np.array(
            [articulation.get_dof_index(name) for name in right_joint_names],
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
            left_wheel_joint_indices=left_wheel_joint_indices,
            right_wheel_joint_indices=right_wheel_joint_indices,
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

    def _discover_wheel_joint_names(self, *, root_prim_path: str, articulation: Any) -> tuple[list[str], list[str]]:
        joint_names = self._discover_joint_names_from_articulation(articulation)
        if not joint_names:
            for prim in iter_prims_under(root_prim_path):
                prim_name = prim.GetName()
                if "joint" in prim_name.lower():
                    joint_names.append(str(prim_name))

        lower_to_original = {name.lower(): name for name in joint_names}

        def _match_all(patterns: list[str]) -> list[str]:
            matched: list[str] = []
            seen: set[str] = set()
            for pattern in patterns:
                regex = re.compile(pattern)
                for lower_name, original_name in lower_to_original.items():
                    if original_name in seen:
                        continue
                    if regex.search(lower_name):
                        matched.append(original_name)
                        seen.add(original_name)
            return matched

        left_names = _match_all([r"left.*(wheel|hub)", r"(wheel|hub).*left", r"(^|[_/])l[_/].*(wheel|hub)"])
        right_names = _match_all([r"right.*(wheel|hub)", r"(wheel|hub).*right", r"(^|[_/])r[_/].*(wheel|hub)"])
        if left_names and right_names:
            return left_names, right_names

        wheel_like = [
            name for name in joint_names if "wheel" in name.lower() or "hub" in name.lower()
        ]
        if len(wheel_like) >= 2:
            return [wheel_like[0]], [wheel_like[1]]

        if len(joint_names) >= 2:
            return [joint_names[0]], [joint_names[1]]

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

        left_centers: list[np.ndarray] = []
        right_centers: list[np.ndarray] = []
        radius_candidates: list[float] = []
        for name, bbox_min, bbox_max in wheel_candidates:
            size = bbox_max - bbox_min
            center = 0.5 * (bbox_min + bbox_max)
            radius_candidates.append(float(np.sort(size)[1] * 0.5))
            if "left" in name:
                left_centers.append(center)
            if "right" in name:
                right_centers.append(center)

        wheel_radius_m = (
            float(np.median(radius_candidates))
            if radius_candidates
            else float(self.default_wheel_radius_m)
        )
        if left_centers and right_centers:
            left_center = np.mean(np.vstack(left_centers), axis=0)
            right_center = np.mean(np.vstack(right_centers), axis=0)
            wheel_distance_m = float(np.linalg.norm(left_center[:2] - right_center[:2]))
        else:
            wheel_distance_m = float(self.default_wheel_distance_m)

        wheel_radius_m = max(wheel_radius_m, 0.02)
        wheel_distance_m = max(wheel_distance_m, 0.05)
        return wheel_radius_m, wheel_distance_m


class NovaCarterAdapter(RobotAdapter):
    model_id = "nova_carter"
    sensorless_usd_rel = "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
    default_planar_radius_m = 0.45
    default_wheel_radius_m = 0.14
    default_wheel_distance_m = 0.54
    max_linear_speed_mps = 1.5
    max_angular_speed_rps = 1.8
    cmd_timeout_sec = 0.5


class CarterV1Adapter(RobotAdapter):
    model_id = "carter_v1"
    sensorless_usd_rel = "/Isaac/Robots/NVIDIA/Carter/carter_v1.usd"
    default_planar_radius_m = 0.42
    default_wheel_radius_m = 0.14
    default_wheel_distance_m = 0.54
    max_linear_speed_mps = 1.4
    max_angular_speed_rps = 1.8
    cmd_timeout_sec = 0.5


class JackalAdapter(RobotAdapter):
    model_id = "jackal"
    sensorless_usd_rel = "/Isaac/Robots/Clearpath/Jackal/jackal.usd"
    default_planar_radius_m = 0.35
    default_wheel_radius_m = 0.10
    default_wheel_distance_m = 0.38
    max_linear_speed_mps = 1.2
    max_angular_speed_rps = 1.8
    cmd_timeout_sec = 0.5


class LimoAdapter(RobotAdapter):
    model_id = "limo"
    sensorless_usd_rel = "/Isaac/Robots/AgilexRobotics/limo/limo.usd"
    default_planar_radius_m = 0.32
    default_wheel_radius_m = 0.06
    default_wheel_distance_m = 0.30
    max_linear_speed_mps = 1.0
    max_angular_speed_rps = 1.8
    cmd_timeout_sec = 0.5


def build_robot_adapter(model_id: str) -> RobotAdapter:
    clean_model_id = str(model_id).strip().lower()
    if clean_model_id == "nova_carter":
        return NovaCarterAdapter()
    if clean_model_id == "carter_v1":
        return CarterV1Adapter()
    if clean_model_id == "jackal":
        return JackalAdapter()
    if clean_model_id == "limo":
        return LimoAdapter()
    raise ValueError(f"Unsupported robot model '{model_id}'.")
