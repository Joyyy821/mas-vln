from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from PIL import Image


DEFAULT_REQUIRED_SAMPLES = 8
DEFAULT_DISTANCE_RANGE_M = (1.0, 2.5)
DEFAULT_VISIBILITY_MARGIN_DEG = 5.0
DEFAULT_MIN_VISIBLE_BBOX_CORNERS = 1
DEFAULT_TARGET_HEIGHT_RATIO = 0.5
DEFAULT_SAFETY_MARGIN_M = 0.10
DEFAULT_PROGRESS_LOG_INTERVAL = 500
DEFAULT_MAX_CANDIDATE_EVALUATIONS = 20000
DEFAULT_MAX_ROBOT_GEOMETRY_DISTANCE_M = 3.0
DEFAULT_MAX_ROBOT_GEOMETRY_DIMENSION_M = 3.0
DEFAULT_FALLBACK_ROBOT_RADIUS_M = 0.45
DEFAULT_STAGE_OCCLUSION_TAIL_RELAXATION_M = 0.35
DEFAULT_MANUAL_GOAL_ROBOT_NAME = "robot1"
OBJECT_LIST_MODES = ("representative", "components", "raw")


def _normalize_path_or_url(value: str) -> str:
    expanded = os.path.expanduser(value)
    if "://" in expanded:
        return expanded.rstrip("/")
    return os.path.abspath(expanded)


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


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


def _translation_matrix(xyz: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = np.asarray(xyz, dtype=float)
    return matrix


def _yaw_from_matrix(matrix: np.ndarray) -> float:
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def _transform_point(matrix: np.ndarray, point_xyz: Sequence[float]) -> np.ndarray:
    point_h = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=float)
    return (matrix @ point_h)[:3]


def _matrix_from_gf(gf_matrix: Any) -> np.ndarray:
    # USD / Gf matrices are exposed in row-vector form, while the sampler uses the
    # usual column-vector convention for NumPy transforms (`matrix @ point_h`).
    # Transpose here so translation ends up in `matrix[:3, 3]` and yaw extraction
    # from the upper-left 3x3 stays consistent everywhere else in the codebase.
    return np.array([[float(gf_matrix[i][j]) for j in range(4)] for i in range(4)], dtype=float).T


def _rgb_to_gray01(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32) / 255.0
    if image.shape[-1] == 1:
        return image[..., 0].astype(np.float32) / 255.0
    rgb = image[..., :3].astype(np.float32) / 255.0
    return rgb.mean(axis=2)


def _segment_intersects_aabb(
    start_xyz: Sequence[float],
    end_xyz: Sequence[float],
    min_xyz: Sequence[float],
    max_xyz: Sequence[float],
    epsilon: float = 1e-9,
) -> bool:
    return _segment_aabb_interval(start_xyz, end_xyz, min_xyz, max_xyz, epsilon) is not None


def _segment_aabb_interval(
    start_xyz: Sequence[float],
    end_xyz: Sequence[float],
    min_xyz: Sequence[float],
    max_xyz: Sequence[float],
    epsilon: float = 1e-9,
) -> tuple[float, float] | None:
    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)
    box_min = np.asarray(min_xyz, dtype=float)
    box_max = np.asarray(max_xyz, dtype=float)
    direction = end - start

    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(direction[axis]) < epsilon:
            if start[axis] < box_min[axis] or start[axis] > box_max[axis]:
                return None
            continue

        inv_dir = 1.0 / direction[axis]
        t1 = (box_min[axis] - start[axis]) * inv_dir
        t2 = (box_max[axis] - start[axis]) * inv_dir
        if t1 > t2:
            t1, t2 = t2, t1

        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None

    return float(t_min), float(t_max)


def yaw_to_quaternion(yaw_rad: float) -> list[float]:
    half_yaw = 0.5 * float(yaw_rad)
    return [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)]


def pose_array_xyzw_from_position_yaw(
    position_xyz: Sequence[float],
    yaw_rad: float,
) -> list[float]:
    qx, qy, qz, qw = yaw_to_quaternion(yaw_rad)
    return [
        float(position_xyz[0]),
        float(position_xyz[1]),
        float(position_xyz[2]),
        qx,
        qy,
        qz,
        qw,
    ]


def pose_dict_from_position_yaw(
    position_xyz: Sequence[float],
    yaw_rad: float,
) -> dict[str, float]:
    return {
        "x": float(position_xyz[0]),
        "y": float(position_xyz[1]),
        "z": float(position_xyz[2]),
        "yaw": float(yaw_rad),
    }


def team_config_goal_payload(
    robot_name: str,
    position_xyz: Sequence[float],
    yaw_rad: float,
) -> dict[str, Any]:
    return {
        "robots": [
            {
                "name": str(robot_name),
                "goal_pose": pose_dict_from_position_yaw(position_xyz, yaw_rad),
            }
        ]
    }


def format_team_config_goal_yaml(
    robot_name: str,
    position_xyz: Sequence[float],
    yaw_rad: float,
) -> str:
    return yaml.safe_dump(
        team_config_goal_payload(robot_name, position_xyz, yaw_rad),
        sort_keys=False,
    ).rstrip()


@dataclass
class ObjectBBox3D:
    prim_path: str
    min_xyz: np.ndarray
    max_xyz: np.ndarray

    @property
    def center_xyz(self) -> np.ndarray:
        return 0.5 * (self.min_xyz + self.max_xyz)

    @property
    def size_xyz(self) -> np.ndarray:
        return self.max_xyz - self.min_xyz

    @property
    def corners_xyz(self) -> list[np.ndarray]:
        min_x, min_y, min_z = self.min_xyz.tolist()
        max_x, max_y, max_z = self.max_xyz.tolist()
        return [
            np.array([x, y, z], dtype=float)
            for x in (min_x, max_x)
            for y in (min_y, max_y)
            for z in (min_z, max_z)
        ]

    def target_point(self, height_ratio: float = DEFAULT_TARGET_HEIGHT_RATIO) -> np.ndarray:
        ratio = float(np.clip(height_ratio, 0.0, 1.0))
        return np.array(
            [
                self.center_xyz[0],
                self.center_xyz[1],
                self.min_xyz[2] + ratio * self.size_xyz[2],
            ],
            dtype=float,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "prim_path": self.prim_path,
            "min_xyz": self.min_xyz.tolist(),
            "max_xyz": self.max_xyz.tolist(),
            "center_xyz": self.center_xyz.tolist(),
            "size_xyz": self.size_xyz.tolist(),
        }


@dataclass
class OccluderBBox:
    prim_path: str
    min_xyz: np.ndarray
    max_xyz: np.ndarray


@dataclass
class RobotCameraSpec:
    robot_usd_path: str
    root_prim_path: str
    base_prim_path: str
    camera_prim_path: str
    root_from_base_matrix: np.ndarray
    root_from_base_translation_xyz: np.ndarray
    root_from_base_yaw_rad: float
    camera_rel_matrix: np.ndarray
    camera_rel_translation_xyz: np.ndarray
    camera_forward_base_xyz: np.ndarray
    camera_forward_yaw_rad: float
    camera_horizontal_fov_rad: float
    camera_vertical_fov_rad: float
    camera_clipping_range_m: tuple[float, float]
    robot_planar_radius_m: float
    robot_size_xyz_m: np.ndarray
    robot_bbox_base_frame_min_xyz: np.ndarray
    robot_bbox_base_frame_max_xyz: np.ndarray

    @property
    def camera_rel_path_from_root(self) -> str:
        if self.camera_prim_path.startswith(self.root_prim_path):
            return self.camera_prim_path[len(self.root_prim_path) :]
        return self.camera_prim_path

    @property
    def base_rel_path_from_root(self) -> str:
        if self.base_prim_path.startswith(self.root_prim_path):
            return self.base_prim_path[len(self.root_prim_path) :]
        return self.base_prim_path

    def as_dict(self) -> dict[str, Any]:
        return {
            "robot_usd_path": self.robot_usd_path,
            "root_prim_path": self.root_prim_path,
            "base_prim_path": self.base_prim_path,
            "camera_prim_path": self.camera_prim_path,
            "root_from_base_translation_xyz": self.root_from_base_translation_xyz.tolist(),
            "root_from_base_yaw_rad": self.root_from_base_yaw_rad,
            "camera_rel_translation_xyz": self.camera_rel_translation_xyz.tolist(),
            "camera_forward_base_xyz": self.camera_forward_base_xyz.tolist(),
            "camera_forward_yaw_rad": self.camera_forward_yaw_rad,
            "camera_horizontal_fov_rad": self.camera_horizontal_fov_rad,
            "camera_vertical_fov_rad": self.camera_vertical_fov_rad,
            "camera_clipping_range_m": list(self.camera_clipping_range_m),
            "robot_planar_radius_m": self.robot_planar_radius_m,
            "robot_size_xyz_m": self.robot_size_xyz_m.tolist(),
            "robot_bbox_base_frame_min_xyz": self.robot_bbox_base_frame_min_xyz.tolist(),
            "robot_bbox_base_frame_max_xyz": self.robot_bbox_base_frame_max_xyz.tolist(),
        }


@dataclass
class SampledGoalPose:
    base_position_xyz: np.ndarray
    yaw_rad: float
    camera_position_xyz: np.ndarray
    target_point_xyz: np.ndarray
    distance_to_target_m: float
    visible_bbox_corner_count: int
    source_grid_row: int
    source_grid_col: int

    def as_pose_array(self) -> list[float]:
        return pose_array_xyzw_from_position_yaw(self.base_position_xyz, self.yaw_rad)

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_position_xyz": self.base_position_xyz.tolist(),
            "yaw_rad": self.yaw_rad,
            "pose_array_xyzw": self.as_pose_array(),
            "camera_position_xyz": self.camera_position_xyz.tolist(),
            "target_point_xyz": self.target_point_xyz.tolist(),
            "distance_to_target_m": self.distance_to_target_m,
            "visible_bbox_corner_count": self.visible_bbox_corner_count,
            "source_grid_row": self.source_grid_row,
            "source_grid_col": self.source_grid_col,
        }


@dataclass
class OccupancyMap:
    image_path: str
    resolution_m: float
    origin_xyz: np.ndarray
    negate: bool
    occupied_thresh: float
    free_thresh: float
    grayscale_01: np.ndarray
    occupancy_probability: np.ndarray
    occupied_mask: np.ndarray
    free_mask: np.ndarray
    unknown_mask: np.ndarray

    @property
    def height(self) -> int:
        return int(self.grayscale_01.shape[0])

    @property
    def width(self) -> int:
        return int(self.grayscale_01.shape[1])

    @classmethod
    def load(
        cls,
        occupancy_map_path: str,
        *,
        resolution_m: float | None = None,
        origin_xyz: Sequence[float] | None = None,
        negate: bool | int | None = None,
        occupied_thresh: float = 0.65,
        free_thresh: float = 0.196,
        treat_unknown_as_occupied: bool = True,
    ) -> "OccupancyMap":
        map_path = os.path.abspath(os.path.expanduser(occupancy_map_path))
        suffix = Path(map_path).suffix.lower()

        if suffix in {".yaml", ".yml"}:
            with open(map_path, "r", encoding="utf-8") as stream:
                map_yaml = yaml.safe_load(stream) or {}

            image_value = map_yaml.get("image")
            if not image_value:
                raise ValueError(f"Map YAML does not define an image path: {map_path}")

            image_path = image_value
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.path.dirname(map_path), image_path)

            resolution_m = float(map_yaml.get("resolution", resolution_m))
            origin_xyz = map_yaml.get("origin", origin_xyz)
            negate = bool(int(map_yaml.get("negate", negate if negate is not None else 0)))
            occupied_thresh = float(map_yaml.get("occupied_thresh", occupied_thresh))
            free_thresh = float(map_yaml.get("free_thresh", free_thresh))
        else:
            image_path = map_path
            if resolution_m is None or origin_xyz is None:
                raise ValueError(
                    "A raw occupancy image requires resolution_m and origin_xyz metadata."
                )
            negate = bool(int(negate if negate is not None else 0))

        if resolution_m is None or origin_xyz is None:
            raise ValueError("Occupancy-map resolution and origin must be available.")

        image = np.array(Image.open(image_path))
        grayscale = _rgb_to_gray01(image)
        occupancy_probability = grayscale if negate else (1.0 - grayscale)

        occupied_mask = occupancy_probability >= float(occupied_thresh)
        free_mask = occupancy_probability <= float(free_thresh)
        unknown_mask = ~(occupied_mask | free_mask)
        if treat_unknown_as_occupied:
            occupied_mask = occupied_mask | unknown_mask

        origin_array = np.array(origin_xyz, dtype=float)
        if origin_array.shape[0] < 3:
            origin_array = np.pad(origin_array, (0, 3 - origin_array.shape[0]), mode="constant")

        return cls(
            image_path=os.path.abspath(os.path.expanduser(image_path)),
            resolution_m=float(resolution_m),
            origin_xyz=origin_array[:3],
            negate=bool(negate),
            occupied_thresh=float(occupied_thresh),
            free_thresh=float(free_thresh),
            grayscale_01=grayscale,
            occupancy_probability=occupancy_probability,
            occupied_mask=occupied_mask,
            free_mask=free_mask,
            unknown_mask=unknown_mask,
        )

    @property
    def origin_yaw_rad(self) -> float:
        return float(self.origin_xyz[2])

    def grid_to_world_xy(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=float)
        cols = np.asarray(cols, dtype=float)
        x_local = (cols + 0.5) * self.resolution_m
        y_local = ((self.height - 1 - rows) + 0.5) * self.resolution_m

        cos_yaw = math.cos(self.origin_yaw_rad)
        sin_yaw = math.sin(self.origin_yaw_rad)
        x_world = self.origin_xyz[0] + cos_yaw * x_local - sin_yaw * y_local
        y_world = self.origin_xyz[1] + sin_yaw * x_local + cos_yaw * y_local
        return np.column_stack((x_world, y_world))

    def cell_center_world_xy(self, row: int, col: int) -> np.ndarray:
        return self.grid_to_world_xy(np.array([row]), np.array([col]))[0]

    def world_to_grid(self, x_world: float, y_world: float) -> tuple[int, int]:
        dx = float(x_world) - self.origin_xyz[0]
        dy = float(y_world) - self.origin_xyz[1]

        cos_yaw = math.cos(self.origin_yaw_rad)
        sin_yaw = math.sin(self.origin_yaw_rad)
        x_local = cos_yaw * dx + sin_yaw * dy
        y_local = -sin_yaw * dx + cos_yaw * dy

        col = int(math.floor(x_local / self.resolution_m))
        row_from_bottom = int(math.floor(y_local / self.resolution_m))
        row = self.height - 1 - row_from_bottom
        return row, col

    def contains_world_xy(self, x_world: float, y_world: float) -> bool:
        row, col = self.world_to_grid(x_world, y_world)
        return 0 <= row < self.height and 0 <= col < self.width

    def inflate_occupied_mask(self, inflation_radius_m: float) -> np.ndarray:
        inflation_radius_m = max(0.0, float(inflation_radius_m))
        radius_cells = int(math.ceil(inflation_radius_m / self.resolution_m))
        if radius_cells <= 0:
            return self.occupied_mask.copy()

        occupied_rows, occupied_cols = np.nonzero(self.occupied_mask)
        if occupied_rows.size == 0:
            return self.occupied_mask.copy()

        offsets: list[tuple[int, int]] = []
        radius_sq = radius_cells * radius_cells
        for delta_row in range(-radius_cells, radius_cells + 1):
            for delta_col in range(-radius_cells, radius_cells + 1):
                if delta_row * delta_row + delta_col * delta_col <= radius_sq:
                    offsets.append((delta_row, delta_col))

        inflated = self.occupied_mask.copy()
        for delta_row, delta_col in offsets:
            rows = occupied_rows + delta_row
            cols = occupied_cols + delta_col
            valid = (
                (rows >= 0)
                & (rows < self.height)
                & (cols >= 0)
                & (cols < self.width)
            )
            inflated[rows[valid], cols[valid]] = True

        return inflated

    @staticmethod
    def _bresenham_cells(
        start_row: int,
        start_col: int,
        end_row: int,
        end_col: int,
    ) -> Iterable[tuple[int, int]]:
        row = start_row
        col = start_col
        delta_row = abs(end_row - start_row)
        delta_col = abs(end_col - start_col)
        step_row = 1 if start_row < end_row else -1
        step_col = 1 if start_col < end_col else -1

        error = delta_col - delta_row
        while True:
            yield row, col
            if row == end_row and col == end_col:
                break
            twice_error = 2 * error
            if twice_error > -delta_row:
                error -= delta_row
                col += step_col
            if twice_error < delta_col:
                error += delta_col
                row += step_row

    def line_is_free(
        self,
        start_xy: Sequence[float],
        end_xy: Sequence[float],
        blocked_mask: np.ndarray,
        *,
        allow_end_in_occupied: bool = False,
    ) -> bool:
        if not self.contains_world_xy(start_xy[0], start_xy[1]):
            return False
        if not self.contains_world_xy(end_xy[0], end_xy[1]):
            return False

        start_row, start_col = self.world_to_grid(float(start_xy[0]), float(start_xy[1]))
        end_row, end_col = self.world_to_grid(float(end_xy[0]), float(end_xy[1]))

        for row, col in self._bresenham_cells(start_row, start_col, end_row, end_col):
            if allow_end_in_occupied and row == end_row and col == end_col:
                break
            if blocked_mask[row, col]:
                return False
        return True
