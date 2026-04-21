from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from PIL import Image

from isaac_sim.goal_generator.object_goal_sampler_utils import OccupancyMap


DEFAULT_OCCUPIED_THRESH = 0.65
DEFAULT_FREE_THRESH = 0.196
DEFAULT_UNKNOWN_COLOR = (205, 205, 205, 255)
DEFAULT_OCCUPIED_PIXEL = 0
DEFAULT_FREE_PIXEL = 255
DEFAULT_UNKNOWN_PIXEL = 205


@dataclass(frozen=True)
class MapExportResult:
    yaml_path: Path
    png_path: Path
    resolution_m: float
    origin_xyz: tuple[float, float, float]
    min_bound_xyz: tuple[float, float, float]
    max_bound_xyz: tuple[float, float, float]
    width_px: int
    height_px: int
    occupied_cells: int
    free_cells: int
    unknown_cells: int


def _dim_xy(dims: Any) -> tuple[int, int]:
    width = int(getattr(dims, "x", dims[0]))
    height = int(getattr(dims, "y", dims[1]))
    return width, height


def _expected_dim_xy(
    *,
    resolution_m: float,
    min_bound_xyz: Sequence[float],
    max_bound_xyz: Sequence[float],
) -> tuple[int, int]:
    width = int(round((float(max_bound_xyz[0]) - float(min_bound_xyz[0])) / float(resolution_m)))
    height = int(round((float(max_bound_xyz[1]) - float(min_bound_xyz[1])) / float(resolution_m)))
    return max(width, 1), max(height, 1)


def _reshape_generator_buffer(
    raw_buffer_flat: np.ndarray,
    *,
    dims: Any,
    expected_width_px: int,
    expected_height_px: int,
) -> tuple[np.ndarray, int, int]:
    dim_width_px, dim_height_px = _dim_xy(dims)
    cell_count = int(raw_buffer_flat.size)
    expected_cell_count = int(expected_width_px * expected_height_px)
    if cell_count <= 0:
        raise RuntimeError("Occupancy-map generator returned an empty buffer.")

    if cell_count == expected_cell_count:
        if dim_width_px == expected_width_px and dim_height_px == expected_height_px:
            return raw_buffer_flat.reshape((expected_height_px, expected_width_px)), expected_width_px, expected_height_px
        if dim_width_px == expected_height_px and dim_height_px == expected_width_px:
            return (
                raw_buffer_flat.reshape((dim_height_px, dim_width_px)).T,
                expected_width_px,
                expected_height_px,
            )
        return raw_buffer_flat.reshape((expected_height_px, expected_width_px)), expected_width_px, expected_height_px

    if dim_width_px > 0 and dim_height_px > 0 and cell_count == dim_width_px * dim_height_px:
        return raw_buffer_flat.reshape((dim_height_px, dim_width_px)), dim_width_px, dim_height_px

    raise RuntimeError(
        "Occupancy-map generator returned inconsistent dimensions: "
        f"buffer cells={cell_count}, generator dims=({dim_width_px}, {dim_height_px}), "
        f"expected dims=({expected_width_px}, {expected_height_px})."
    )


def export_occupancy_map(
    *,
    yaml_path: str | Path,
    resolution_m: float,
    origin_hint_xyz: Sequence[float],
    min_bound_xyz: Sequence[float],
    max_bound_xyz: Sequence[float],
    start_location_xyz: Sequence[float] | None = None,
) -> MapExportResult:
    import omni.physx
    import omni.usd
    from isaacsim.asset.gen.omap.bindings import _omap

    yaml_path = Path(yaml_path).expanduser().resolve()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = yaml_path.with_suffix(".png")

    physx = omni.physx.acquire_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()
    generator = _omap.Generator(physx, stage_id)
    generator.update_settings(float(resolution_m), 4, 5, 6)
    if start_location_xyz is None:
        start_location_xyz = (
            0.5 * (float(min_bound_xyz[0]) + float(max_bound_xyz[0])),
            0.5 * (float(min_bound_xyz[1]) + float(max_bound_xyz[1])),
            max(0.2, float(min_bound_xyz[2])),
        )
    generator.set_transform(
        tuple(float(value) for value in start_location_xyz[:3]),
        tuple(float(value) for value in min_bound_xyz[:3]),
        tuple(float(value) for value in max_bound_xyz[:3]),
    )
    generator.generate2d()

    dims = generator.get_dimensions()
    expected_width_px, expected_height_px = _expected_dim_xy(
        resolution_m=resolution_m,
        min_bound_xyz=min_bound_xyz,
        max_bound_xyz=max_bound_xyz,
    )
    raw_buffer_flat = np.asarray(generator.get_buffer(), dtype=np.int32)
    raw_buffer, width_px, height_px = _reshape_generator_buffer(
        raw_buffer_flat,
        dims=dims,
        expected_width_px=expected_width_px,
        expected_height_px=expected_height_px,
    )

    image_buffer = np.full((height_px, width_px), DEFAULT_UNKNOWN_PIXEL, dtype=np.uint8)
    image_buffer[raw_buffer == 4] = DEFAULT_OCCUPIED_PIXEL
    image_buffer[raw_buffer == 5] = DEFAULT_FREE_PIXEL
    image_buffer[raw_buffer == 6] = DEFAULT_UNKNOWN_PIXEL
    # ROS occupancy-map YAML uses a lower-left origin, while image rows are top-first.
    image = Image.fromarray(np.flipud(image_buffer), mode="L")
    image.save(png_path)

    payload = {
        "image": png_path.name,
        "resolution": float(resolution_m),
        "origin": [float(origin_hint_xyz[0]), float(origin_hint_xyz[1]), float(origin_hint_xyz[2])],
        "negate": 0,
        "occupied_thresh": DEFAULT_OCCUPIED_THRESH,
        "free_thresh": DEFAULT_FREE_THRESH,
    }
    with yaml_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)

    return MapExportResult(
        yaml_path=yaml_path,
        png_path=png_path,
        resolution_m=float(resolution_m),
        origin_xyz=(float(origin_hint_xyz[0]), float(origin_hint_xyz[1]), float(origin_hint_xyz[2])),
        min_bound_xyz=(float(min_bound_xyz[0]), float(min_bound_xyz[1]), float(min_bound_xyz[2])),
        max_bound_xyz=(float(max_bound_xyz[0]), float(max_bound_xyz[1]), float(max_bound_xyz[2])),
        width_px=width_px,
        height_px=height_px,
        occupied_cells=int(np.count_nonzero(raw_buffer == 4)),
        free_cells=int(np.count_nonzero(raw_buffer == 5)),
        unknown_cells=int(np.count_nonzero(raw_buffer == 6)),
    )


def largest_connected_mask(free_mask: np.ndarray) -> np.ndarray:
    free_mask = np.asarray(free_mask, dtype=bool)
    if free_mask.size == 0 or not np.any(free_mask):
        return np.zeros_like(free_mask, dtype=bool)

    height, width = free_mask.shape
    visited = np.zeros_like(free_mask, dtype=bool)
    largest_component: list[tuple[int, int]] = []

    for row, col in zip(*np.nonzero(free_mask)):
        if visited[row, col]:
            continue

        queue: deque[tuple[int, int]] = deque([(int(row), int(col))])
        visited[row, col] = True
        current_component: list[tuple[int, int]] = []

        while queue:
            current_row, current_col = queue.popleft()
            current_component.append((current_row, current_col))

            for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row = current_row + delta_row
                next_col = current_col + delta_col
                if not (0 <= next_row < height and 0 <= next_col < width):
                    continue
                if visited[next_row, next_col] or not free_mask[next_row, next_col]:
                    continue
                visited[next_row, next_col] = True
                queue.append((next_row, next_col))

        if len(current_component) > len(largest_component):
            largest_component = current_component

    component_mask = np.zeros_like(free_mask, dtype=bool)
    if not largest_component:
        return component_mask

    rows, cols = zip(*largest_component)
    component_mask[np.asarray(rows), np.asarray(cols)] = True
    return component_mask


def build_navigation_free_mask(
    occupancy_map: OccupancyMap,
    *,
    inflation_radius_m: float,
) -> np.ndarray:
    inflated = occupancy_map.inflate_occupied_mask(float(inflation_radius_m))
    return largest_connected_mask(~inflated)


def _greedy_sample_indices(
    occupancy_map: OccupancyMap,
    candidate_mask: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
    min_separation_m: float,
    reference_points_xy: Sequence[np.ndarray] | None = None,
    min_distance_from_references_m: float = 0.0,
) -> list[tuple[int, int]]:
    candidate_rows, candidate_cols = np.nonzero(candidate_mask)
    if candidate_rows.size == 0:
        return []

    candidate_xy = occupancy_map.grid_to_world_xy(candidate_rows, candidate_cols)
    order = rng.permutation(candidate_rows.size)
    chosen: list[tuple[int, int]] = []
    chosen_xy: list[np.ndarray] = []
    min_separation_sq = float(min_separation_m) * float(min_separation_m)
    min_reference_sq = float(min_distance_from_references_m) * float(min_distance_from_references_m)
    reference_points_xy = list(reference_points_xy or [])

    for idx in order.tolist():
        world_xy = candidate_xy[idx]
        if any(np.sum((world_xy - other_xy) ** 2) < min_separation_sq for other_xy in chosen_xy):
            continue
        if reference_points_xy and any(
            np.sum((world_xy - reference_xy) ** 2) < min_reference_sq
            for reference_xy in reference_points_xy
        ):
            continue
        chosen.append((int(candidate_rows[idx]), int(candidate_cols[idx])))
        chosen_xy.append(world_xy)
        if len(chosen) >= count:
            break

    return chosen


def _pose_dict(x: float, y: float, yaw: float, z: float = 0.0) -> dict[str, float]:
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "yaw": float(yaw),
    }


def sample_multi_robot_rollouts(
    *,
    occupancy_map: OccupancyMap,
    robot_names: Sequence[str],
    rollout_count: int,
    rng: np.random.Generator,
    inflation_radius_m: float,
    min_pairwise_distance_m: float,
    min_goal_distance_m: float,
    focus_xy: Sequence[float] | None = None,
    focus_distance_range_m: tuple[float, float] = (2.5, 5.5),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    navigation_mask = build_navigation_free_mask(
        occupancy_map,
        inflation_radius_m=float(inflation_radius_m),
    )
    if not np.any(navigation_mask):
        raise RuntimeError("No connected free space remains after occupancy inflation.")

    focus_candidate_mask = navigation_mask.copy()
    focus_xy_np = None if focus_xy is None else np.asarray(focus_xy[:2], dtype=float)
    if focus_xy_np is not None:
        goal_rows, goal_cols = np.nonzero(navigation_mask)
        goal_xy = occupancy_map.grid_to_world_xy(goal_rows, goal_cols)
        distances = np.linalg.norm(goal_xy - focus_xy_np, axis=1)
        focus_candidate_mask = np.zeros_like(navigation_mask, dtype=bool)
        valid = (
            distances >= float(focus_distance_range_m[0])
        ) & (
            distances <= float(focus_distance_range_m[1])
        )
        focus_candidate_mask[goal_rows[valid], goal_cols[valid]] = True
        if not np.any(focus_candidate_mask):
            focus_candidate_mask = navigation_mask.copy()

    rollouts: list[dict[str, Any]] = []
    robot_names = [str(name) for name in robot_names]
    for rollout_index in range(int(rollout_count)):
        initial_indices = _greedy_sample_indices(
            occupancy_map,
            navigation_mask,
            count=len(robot_names),
            rng=rng,
            min_separation_m=float(min_pairwise_distance_m),
        )
        if len(initial_indices) != len(robot_names):
            raise RuntimeError(
                "Unable to sample enough collision-free initial poses for the requested robot count."
            )

        initial_xy = [
            occupancy_map.cell_center_world_xy(row, col) for row, col in initial_indices
        ]
        goal_indices = _greedy_sample_indices(
            occupancy_map,
            focus_candidate_mask,
            count=len(robot_names),
            rng=rng,
            min_separation_m=float(min_pairwise_distance_m),
            reference_points_xy=initial_xy,
            min_distance_from_references_m=float(min_goal_distance_m),
        )
        if len(goal_indices) != len(robot_names):
            goal_indices = _greedy_sample_indices(
                occupancy_map,
                navigation_mask,
                count=len(robot_names),
                rng=rng,
                min_separation_m=float(min_pairwise_distance_m),
                reference_points_xy=initial_xy,
                min_distance_from_references_m=float(min_goal_distance_m),
            )
        if len(goal_indices) != len(robot_names):
            raise RuntimeError(
                "Unable to sample enough collision-free goal poses for the requested robot count."
            )

        robots: list[dict[str, Any]] = []
        for robot_name, (init_row, init_col), (goal_row, goal_col) in zip(
            robot_names, initial_indices, goal_indices
        ):
            init_xy = occupancy_map.cell_center_world_xy(init_row, init_col)
            goal_xy = occupancy_map.cell_center_world_xy(goal_row, goal_col)
            initial_yaw = float(rng.uniform(-np.pi, np.pi))
            if focus_xy_np is not None:
                goal_yaw = float(np.arctan2(focus_xy_np[1] - goal_xy[1], focus_xy_np[0] - goal_xy[0]))
            else:
                goal_yaw = float(rng.uniform(-np.pi, np.pi))

            robots.append(
                {
                    "name": robot_name,
                    "initial_pose": _pose_dict(init_xy[0], init_xy[1], initial_yaw),
                    "goal_pose": _pose_dict(goal_xy[0], goal_xy[1], goal_yaw),
                }
            )

        rollouts.append({"id": rollout_index + 1, "robots": robots})

    validation = {
        "inflation_radius_m": float(inflation_radius_m),
        "largest_component_free_cells": int(np.count_nonzero(navigation_mask)),
        "focus_goal_candidates": int(np.count_nonzero(focus_candidate_mask)),
    }
    return rollouts, validation
