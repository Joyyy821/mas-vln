from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
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
    debug: dict[str, Any] = field(default_factory=dict)


def _dim_xy(dims: Any) -> tuple[int, int]:
    if hasattr(dims, "x") and hasattr(dims, "y"):
        width = int(dims.x)
        height = int(dims.y)
    else:
        width = int(dims[0])
        height = int(dims[1])
    return width, height


def _compute_world_xy_bounds(
    env_prim_path: str,
    *,
    z_min: float = 0.1,
    z_max: float = 0.62,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(env_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found for occupancy-map bounds: {env_prim_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    aligned_bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    min_pt = aligned_bbox.GetMin()
    max_pt = aligned_bbox.GetMax()
    min_bound_xyz = (float(min_pt[0]), float(min_pt[1]), float(z_min))
    max_bound_xyz = (float(max_pt[0]), float(max_pt[1]), float(z_max))

    if min_bound_xyz[0] >= max_bound_xyz[0] or min_bound_xyz[1] >= max_bound_xyz[1]:
        raise RuntimeError(
            "Occupancy-map environment bounds are invalid: "
            f"min={min_bound_xyz}, max={max_bound_xyz}"
        )
    if float(z_min) >= float(z_max):
        raise RuntimeError(f"Occupancy-map z bounds are invalid: z_min={z_min}, z_max={z_max}")

    return min_bound_xyz, max_bound_xyz


def _reshape_generator_buffer(
    raw_buffer_flat: np.ndarray,
    *,
    dims: Any,
) -> tuple[np.ndarray, int, int]:
    width_px, height_px = _dim_xy(dims)
    if width_px <= 0 or height_px <= 0:
        raise RuntimeError(
            "Occupancy-map generator returned invalid dimensions: "
            f"width={width_px}, height={height_px}, dims={dims}"
        )

    cell_count = int(raw_buffer_flat.size)
    expected_cell_count = int(width_px * height_px)
    if cell_count <= 0:
        raise RuntimeError("Occupancy-map generator returned an empty buffer.")

    if cell_count == expected_cell_count:
        return raw_buffer_flat.reshape((height_px, width_px)), width_px, height_px

    raise RuntimeError(
        "Occupancy-map generator returned inconsistent dimensions: "
        f"buffer cells={cell_count}, generator dims=({width_px}, {height_px})."
    )


def _pixel_histogram(image_buffer: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(np.asarray(image_buffer, dtype=np.uint8), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def _write_map_artifacts(
    *,
    yaml_path: str | Path,
    resolution_m: float,
    origin_hint_xyz: Sequence[float],
    min_bound_xyz: Sequence[float],
    max_bound_xyz: Sequence[float],
    image_buffer: np.ndarray,
    debug_payload: dict[str, Any] | None = None,
) -> MapExportResult:
    yaml_path = Path(yaml_path).expanduser().resolve()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = yaml_path.with_suffix(".png")

    image = Image.fromarray(np.asarray(image_buffer, dtype=np.uint8), mode="L")
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
    if not png_path.exists():
        raise RuntimeError(f"Occupancy-map image was not written: {png_path}")
    if not yaml_path.exists():
        raise RuntimeError(f"Occupancy-map YAML was not written: {yaml_path}")

    occupied_cells = int(np.count_nonzero(image_buffer == DEFAULT_OCCUPIED_PIXEL))
    free_cells = int(np.count_nonzero(image_buffer == DEFAULT_FREE_PIXEL))
    unknown_cells = int(np.count_nonzero(image_buffer == DEFAULT_UNKNOWN_PIXEL))
    height_px, width_px = image_buffer.shape[:2]

    return MapExportResult(
        yaml_path=yaml_path,
        png_path=png_path,
        resolution_m=float(resolution_m),
        origin_xyz=(float(origin_hint_xyz[0]), float(origin_hint_xyz[1]), float(origin_hint_xyz[2])),
        min_bound_xyz=(float(min_bound_xyz[0]), float(min_bound_xyz[1]), float(min_bound_xyz[2])),
        max_bound_xyz=(float(max_bound_xyz[0]), float(max_bound_xyz[1]), float(max_bound_xyz[2])),
        width_px=int(width_px),
        height_px=int(height_px),
        occupied_cells=occupied_cells,
        free_cells=free_cells,
        unknown_cells=unknown_cells,
        debug={
            **(debug_payload or {}),
            "yaml_path": str(yaml_path),
            "png_path": str(png_path),
            "image_pixel_histogram": _pixel_histogram(image_buffer),
        },
    )


def _export_occupancy_map(
    *,
    env_prim_path: str,
    yaml_path: str | Path,
    resolution_m: float,
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    z_min: float = 0.1,
    z_max: float = 0.62,
    rotate_180: bool = True,
) -> MapExportResult:
    import omni.physx
    import omni.usd
    from isaacsim.asset.gen.omap.bindings import _omap

    get_physx_interface = getattr(omni.physx, "get_physx_interface", None)
    acquire_physx_interface = getattr(omni.physx, "acquire_physx_interface", None)
    if callable(get_physx_interface):
        physx = get_physx_interface()
    elif callable(acquire_physx_interface):
        physx = acquire_physx_interface()
    else:
        raise RuntimeError("Unable to acquire Isaac Sim PhysX interface for occupancy map export.")

    min_bound_xyz, max_bound_xyz = _compute_world_xy_bounds(
        env_prim_path,
        z_min=z_min,
        z_max=z_max,
    )

    stage_id = omni.usd.get_context().get_stage_id()
    generator = _omap.Generator(physx, stage_id)
    generator.update_settings(float(resolution_m), DEFAULT_OCCUPIED_PIXEL, 254, DEFAULT_UNKNOWN_PIXEL)
    generator.set_transform(
        tuple(float(value) for value in origin_xyz[:3]),
        tuple(float(value) for value in min_bound_xyz[:3]),
        tuple(float(value) for value in max_bound_xyz[:3]),
    )
    generator.generate2d()

    dims = generator.get_dimensions()
    raw_buffer_flat = np.asarray(generator.get_buffer(), dtype=np.uint8)
    raw_buffer, width_px, height_px = _reshape_generator_buffer(
        raw_buffer_flat,
        dims=dims,
    )

    raw_buffer = np.asarray(raw_buffer, dtype=np.uint8)
    rotated_buffer = np.rot90(raw_buffer, 2) if rotate_180 else raw_buffer
    image_buffer = np.full((height_px, width_px), DEFAULT_UNKNOWN_PIXEL, dtype=np.uint8)
    image_buffer[rotated_buffer == DEFAULT_OCCUPIED_PIXEL] = DEFAULT_OCCUPIED_PIXEL
    image_buffer[rotated_buffer == 254] = DEFAULT_FREE_PIXEL

    occupied_cells = int(np.count_nonzero(image_buffer == DEFAULT_OCCUPIED_PIXEL))
    free_cells = int(np.count_nonzero(image_buffer == DEFAULT_FREE_PIXEL))
    if occupied_cells <= 0 or free_cells <= 0:
        raise RuntimeError(
            "Occupancy-map generator output is not meaningful: "
            f"occupied_cells={occupied_cells}, free_cells={free_cells}, "
            f"dims=({width_px}, {height_px}), raw_histogram={_pixel_histogram(raw_buffer)}"
        )

    return _write_map_artifacts(
        yaml_path=yaml_path,
        resolution_m=resolution_m,
        origin_hint_xyz=(float(min_bound_xyz[0]), float(min_bound_xyz[1]), 0.0),
        min_bound_xyz=min_bound_xyz,
        max_bound_xyz=max_bound_xyz,
        image_buffer=image_buffer,
        debug_payload={
            "env_prim_path": str(env_prim_path),
            "resolution_m": float(resolution_m),
            "omap_origin_xyz": [float(value) for value in origin_xyz[:3]],
            "min_bound_xyz": [float(value) for value in min_bound_xyz],
            "max_bound_xyz": [float(value) for value in max_bound_xyz],
            "z_min": float(z_min),
            "z_max": float(z_max),
            "rotate_180": bool(rotate_180),
            "generator_dimensions": [int(width_px), int(height_px)],
            "generator_buffer_size": int(raw_buffer_flat.size),
            "raw_pixel_histogram": _pixel_histogram(raw_buffer),
        },
    )


def _export_resampled_occupancy_map(
    *,
    yaml_path: str | Path,
    source_result: MapExportResult,
    resolution_m: float,
) -> MapExportResult:
    if float(resolution_m) <= 0.0:
        raise RuntimeError(f"Invalid resampled occupancy-map resolution: {resolution_m}")
    if float(source_result.resolution_m) <= 0.0:
        raise RuntimeError(f"Invalid source occupancy-map resolution: {source_result.resolution_m}")

    source_image = np.asarray(Image.open(source_result.png_path).convert("L"), dtype=np.uint8)
    source_height_px, source_width_px = source_image.shape[:2]
    source_origin_x = float(source_result.origin_xyz[0])
    source_origin_y = float(source_result.origin_xyz[1])
    source_resolution = float(source_result.resolution_m)
    target_resolution = float(resolution_m)
    min_bound_xyz = tuple(float(value) for value in source_result.min_bound_xyz)
    max_bound_xyz = tuple(float(value) for value in source_result.max_bound_xyz)

    target_width_px = max(
        1,
        int(round((float(max_bound_xyz[0]) - float(min_bound_xyz[0])) / target_resolution)),
    )
    target_height_px = max(
        1,
        int(round((float(max_bound_xyz[1]) - float(min_bound_xyz[1])) / target_resolution)),
    )
    target_image = np.full((target_height_px, target_width_px), DEFAULT_UNKNOWN_PIXEL, dtype=np.uint8)

    for target_row in range(target_height_px):
        bottom_index = target_height_px - 1 - target_row
        y_min = source_origin_y + bottom_index * target_resolution
        y_max = y_min + target_resolution
        source_row_min = source_height_px - int(np.ceil((y_max - source_origin_y) / source_resolution))
        source_row_max = source_height_px - int(np.floor((y_min - source_origin_y) / source_resolution))
        source_row_min = max(0, min(source_height_px, source_row_min))
        source_row_max = max(0, min(source_height_px, source_row_max))
        if source_row_min >= source_row_max:
            continue

        for target_col in range(target_width_px):
            x_min = source_origin_x + target_col * target_resolution
            x_max = x_min + target_resolution
            source_col_min = int(np.floor((x_min - source_origin_x) / source_resolution))
            source_col_max = int(np.ceil((x_max - source_origin_x) / source_resolution))
            source_col_min = max(0, min(source_width_px, source_col_min))
            source_col_max = max(0, min(source_width_px, source_col_max))
            if source_col_min >= source_col_max:
                continue

            source_block = source_image[source_row_min:source_row_max, source_col_min:source_col_max]
            if np.any(source_block == DEFAULT_OCCUPIED_PIXEL):
                target_image[target_row, target_col] = DEFAULT_OCCUPIED_PIXEL
            elif np.any(source_block == DEFAULT_UNKNOWN_PIXEL):
                target_image[target_row, target_col] = DEFAULT_UNKNOWN_PIXEL
            else:
                target_image[target_row, target_col] = DEFAULT_FREE_PIXEL

    occupied_cells = int(np.count_nonzero(target_image == DEFAULT_OCCUPIED_PIXEL))
    free_cells = int(np.count_nonzero(target_image == DEFAULT_FREE_PIXEL))
    if occupied_cells <= 0 or free_cells <= 0:
        raise RuntimeError(
            "Resampled occupancy-map output is not meaningful: "
            f"occupied_cells={occupied_cells}, free_cells={free_cells}, "
            f"dims=({target_width_px}, {target_height_px}), source_histogram={_pixel_histogram(source_image)}"
        )

    return _write_map_artifacts(
        yaml_path=yaml_path,
        resolution_m=target_resolution,
        origin_hint_xyz=source_result.origin_xyz,
        min_bound_xyz=min_bound_xyz,
        max_bound_xyz=max_bound_xyz,
        image_buffer=target_image,
        debug_payload={
            "source_yaml_path": str(source_result.yaml_path),
            "source_png_path": str(source_result.png_path),
            "source_resolution_m": float(source_result.resolution_m),
            "source_dimensions": [int(source_width_px), int(source_height_px)],
            "source_pixel_histogram": _pixel_histogram(source_image),
            "resample_policy": "occupied_over_unknown_over_free",
            "target_resolution_m": target_resolution,
            "target_dimensions": [int(target_width_px), int(target_height_px)],
            "min_bound_xyz": [float(value) for value in min_bound_xyz],
            "max_bound_xyz": [float(value) for value in max_bound_xyz],
        },
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
    min_initial_focus_distance_m: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    navigation_mask = build_navigation_free_mask(
        occupancy_map,
        inflation_radius_m=float(inflation_radius_m),
    )
    if not np.any(navigation_mask):
        raise RuntimeError("No connected free space remains after occupancy inflation.")

    focus_candidate_mask = navigation_mask.copy()
    initial_candidate_mask = navigation_mask.copy()
    focus_xy_np = None if focus_xy is None else np.asarray(focus_xy[:2], dtype=float)
    goal_candidate_count = int(np.count_nonzero(focus_candidate_mask))
    initial_candidate_count = int(np.count_nonzero(initial_candidate_mask))
    resolved_min_initial_focus_distance_m = None
    if focus_xy_np is not None:
        candidate_rows, candidate_cols = np.nonzero(navigation_mask)
        candidate_xy = occupancy_map.grid_to_world_xy(candidate_rows, candidate_cols)
        distances = np.linalg.norm(candidate_xy - focus_xy_np, axis=1)
        focus_candidate_mask = np.zeros_like(navigation_mask, dtype=bool)
        valid = (
            distances >= float(focus_distance_range_m[0])
        ) & (
            distances <= float(focus_distance_range_m[1])
        )
        focus_candidate_mask[candidate_rows[valid], candidate_cols[valid]] = True
        if not np.any(focus_candidate_mask):
            raise RuntimeError(
                "No goal-pose candidates remain inside the focus-distance range after "
                "occupancy inflation."
            )
        goal_candidate_count = int(np.count_nonzero(focus_candidate_mask))

        if min_initial_focus_distance_m is None:
            resolved_min_initial_focus_distance_m = float(focus_distance_range_m[1]) + 1.5
        else:
            resolved_min_initial_focus_distance_m = float(min_initial_focus_distance_m)
        initial_candidate_mask = np.zeros_like(navigation_mask, dtype=bool)
        initial_valid = distances >= resolved_min_initial_focus_distance_m
        initial_candidate_mask[candidate_rows[initial_valid], candidate_cols[initial_valid]] = True
        initial_candidate_count = int(np.count_nonzero(initial_candidate_mask))
        if not np.any(initial_candidate_mask):
            raise RuntimeError(
                "No initial-pose candidates remain after excluding the focus/goal area."
            )

    rollouts: list[dict[str, Any]] = []
    robot_names = [str(name) for name in robot_names]
    observed_initial_focus_distances: list[float] = []
    observed_initial_goal_distances: list[float] = []
    for rollout_index in range(int(rollout_count)):
        initial_indices = _greedy_sample_indices(
            occupancy_map,
            initial_candidate_mask,
            count=len(robot_names),
            rng=rng,
            min_separation_m=float(min_pairwise_distance_m),
        )
        if len(initial_indices) != len(robot_names):
            raise RuntimeError(
                "Unable to sample enough collision-free initial poses outside the focus/goal area."
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
            if focus_xy_np is not None:
                observed_initial_focus_distances.append(
                    float(np.linalg.norm(np.asarray(init_xy, dtype=float) - focus_xy_np))
                )
            observed_initial_goal_distances.append(
                float(np.linalg.norm(np.asarray(init_xy, dtype=float) - np.asarray(goal_xy, dtype=float)))
            )
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
        "initial_pose_candidates": int(initial_candidate_count),
        "focus_goal_candidates": int(np.count_nonzero(focus_candidate_mask)),
        "goal_pose_candidates": int(goal_candidate_count),
        "selected_focus_xy": None if focus_xy_np is None else focus_xy_np.tolist(),
        "min_initial_focus_distance_m": (
            None if resolved_min_initial_focus_distance_m is None else float(resolved_min_initial_focus_distance_m)
        ),
        "observed_min_initial_focus_distance_m": (
            None
            if not observed_initial_focus_distances
            else float(min(observed_initial_focus_distances))
        ),
        "min_initial_goal_distance_m": float(min_goal_distance_m),
        "observed_min_initial_goal_distance_m": (
            None
            if not observed_initial_goal_distances
            else float(min(observed_initial_goal_distances))
        ),
    }
    return rollouts, validation
