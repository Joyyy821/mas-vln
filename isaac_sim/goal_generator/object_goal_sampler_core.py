#!/usr/bin/env python3
"""
Goal sampler for object-reaching navigation in Isaac Sim.

The sampler is intentionally split into two parts:
1. USD / Isaac Sim inspection utilities that locate an object and extract robot metadata.
2. Pure Python occupancy-map sampling that proposes planar robot poses whose camera can
   observe the target object without obvious collisions or occlusions.

This module does not use ROS. Stage inspection and GUI validation are expected to run
from Isaac Sim's Python environment, for example:

    ./python.sh /abs/path/to/object_goal_sampler.py \
        --environment-usd omniverse://localhost/NVIDIA/Assets/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd \
        --robot-usd omniverse://localhost/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd \
        --occupancy-map /abs/path/to/carter_warehouse_navigation.yaml \
        --object-query Forklift \
        --required-samples 8 \
        --validate-gui
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from object_goal_sampler_gui import GoalSamplerGuiMixin
from object_goal_sampler_listing import ObjectListingMixin
from object_goal_sampler_utils import (
    DEFAULT_DISTANCE_RANGE_M,
    DEFAULT_FALLBACK_ROBOT_RADIUS_M,
    DEFAULT_MANUAL_GOAL_ROBOT_NAME,
    DEFAULT_MAX_CANDIDATE_EVALUATIONS,
    DEFAULT_MAX_ROBOT_GEOMETRY_DIMENSION_M,
    DEFAULT_MAX_ROBOT_GEOMETRY_DISTANCE_M,
    DEFAULT_MIN_VISIBLE_BBOX_CORNERS,
    DEFAULT_PROGRESS_LOG_INTERVAL,
    DEFAULT_REQUIRED_SAMPLES,
    DEFAULT_SAFETY_MARGIN_M,
    DEFAULT_STAGE_OCCLUSION_TAIL_RELAXATION_M,
    DEFAULT_TARGET_HEIGHT_RATIO,
    DEFAULT_VISIBILITY_MARGIN_DEG,
    OBJECT_LIST_MODES,
    ObjectBBox3D,
    OccupancyMap,
    OccluderBBox,
    RobotCameraSpec,
    SampledGoalPose,
    _matrix_from_gf,
    _normalize_path_or_url,
    _rotation_matrix_z,
    _segment_aabb_interval,
    _transform_point,
    _translation_matrix,
    _wrap_to_pi,
    _yaw_from_matrix,
)


class ObjectReachingGoalSampler(ObjectListingMixin, GoalSamplerGuiMixin):
    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self._environment_stage: Any | None = None
        self._environment_stage_path: str | None = None
        self._environment_bbox_cache: Any | None = None
        self._environment_root_prim: Any | None = None
        self._last_object_bbox: ObjectBBox3D | None = None
        self._last_object_prim: Any | None = None
        self._occluder_cache_by_object_path: dict[str, list[OccluderBBox]] = {}
        self._robot_specs: dict[tuple[str, str | None, str | None], RobotCameraSpec] = {}
        self._simulation_app: Any | None = None
        self._simulation_app_headless: bool | None = None

    @staticmethod
    def _simulation_app_class() -> Any:
        try:
            from isaacsim.simulation_app import SimulationApp

            return SimulationApp
        except Exception:
            try:
                from omni.isaac.kit import SimulationApp

                return SimulationApp
            except Exception as exc:
                raise RuntimeError(
                    "Isaac Sim runtime is unavailable. Run this module with your Isaac Sim "
                    "`python.sh`, or ensure the Isaac Sim Python packages are on PYTHONPATH."
                ) from exc

    def _ensure_runtime(self, *, require_gui: bool = False) -> tuple[Any, Any, Any]:
        if require_gui:
            if self._simulation_app is not None:
                if self._simulation_app_headless:
                    raise RuntimeError(
                        "This sampler already bootstrapped a headless SimulationApp in the current "
                        "process. For GUI validation, start a fresh process and call the script with "
                        "`--validate-gui` so it can create the GUI app first."
                    )
                return self._ensure_usd_imports()

            simulation_app_class = self._simulation_app_class()
            print("[INFO] Starting Isaac Sim GUI runtime...", flush=True)
            self._simulation_app = simulation_app_class({"headless": False})
            self._simulation_app_headless = False
            for _ in range(10):
                self._simulation_app.update()
            return self._ensure_usd_imports()

        try:
            return self._ensure_usd_imports()
        except RuntimeError as exc:
            if self._simulation_app is not None:
                raise exc

            simulation_app_class = self._simulation_app_class()
            print("[INFO] Starting Isaac Sim headless runtime for USD inspection...", flush=True)
            self._simulation_app = simulation_app_class({"headless": True})
            self._simulation_app_headless = True
            for _ in range(10):
                self._simulation_app.update()
            return self._ensure_usd_imports()

    def close(self) -> None:
        if self._simulation_app is None:
            return

        try:
            self._simulation_app.close()
        finally:
            self._simulation_app = None
            self._simulation_app_headless = None

    @staticmethod
    def _ensure_usd_imports() -> tuple[Any, Any, Any]:
        try:
            from pxr import Gf, Usd, UsdGeom
        except Exception as exc:  # pragma: no cover - requires Isaac Sim / USD runtime.
            raise RuntimeError(
                "USD bindings are unavailable. Run this module from Isaac Sim's Python "
                "environment, for example with ./python.sh."
            ) from exc
        return Usd, UsdGeom, Gf

    @staticmethod
    def _add_reference_to_stage(usd_path: str, prim_path: str) -> None:
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
        except Exception:
            from omni.isaac.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    @staticmethod
    def _update_runtime(sim_app: Any, steps: int = 30) -> None:
        if sim_app is None:
            return
        for _ in range(max(1, steps)):
            sim_app.update()

    @staticmethod
    def _ensure_xform_path(stage: Any, path: str) -> None:
        _, UsdGeom, _ = ObjectReachingGoalSampler._ensure_usd_imports()
        if not path or path == "/":
            return

        current = ""
        for token in [token for token in path.split("/") if token]:
            current += f"/{token}"
            prim = stage.GetPrimAtPath(current)
            if prim and prim.IsValid():
                continue
            UsdGeom.Xform.Define(stage, current)

    @staticmethod
    def _extract_isaac_asset_relative_path(asset_path: str) -> str | None:
        normalized = asset_path.replace("\\", "/")
        if normalized.startswith("/Isaac/"):
            return normalized

        marker = "/NVIDIA/Assets/Isaac/"
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            return normalized[marker_index + len("/NVIDIA/Assets") :]

        marker = "/Isaac/"
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            return normalized[marker_index:]

        return None

    @staticmethod
    def _get_assets_root_path() -> str | None:
        try:
            from isaacsim.storage.native import get_assets_root_path
        except Exception:
            try:
                from omni.isaac.core.utils.nucleus import get_assets_root_path
            except Exception:
                return None

        try:
            assets_root = get_assets_root_path()
        except Exception:
            return None

        if not assets_root:
            return None
        return str(assets_root).rstrip("/")

    def _resolve_runtime_asset_path(self, asset_path: str) -> str:
        normalized = _normalize_path_or_url(asset_path)
        if "://" not in normalized:
            return normalized

        relative_path = self._extract_isaac_asset_relative_path(normalized)
        assets_root = self._get_assets_root_path()
        if not relative_path or not assets_root:
            return normalized

        resolved = f"{assets_root}{relative_path}"
        if resolved != normalized:
            print(
                f"[INFO] Resolved Isaac asset path: {normalized} -> {resolved}",
                flush=True,
            )
        return resolved

    def _compose_single_reference_stage(
        self,
        *,
        usd_path: str,
        prim_path: str,
        stage_label: str,
    ) -> tuple[Any, Any]:
        _, UsdGeom, _ = self._ensure_runtime()
        import omni.usd

        sim_app = self._simulation_app
        print(f"[INFO] Loading {stage_label} into a temporary Isaac Sim stage...", flush=True)
        omni.usd.get_context().new_stage()
        self._update_runtime(sim_app, 5)

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(f"Failed to create a temporary stage for {stage_label}.")

        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        parent_path = str(Path(prim_path).parent)
        self._ensure_xform_path(stage, parent_path)
        print(f"[INFO] Adding reference {usd_path} at {prim_path}...", flush=True)
        self._add_reference_to_stage(usd_path, prim_path)
        print(f"[INFO] Waiting for {stage_label} assets to resolve...", flush=True)
        self._update_runtime(sim_app, 90)

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Referenced {stage_label} prim was not created at {prim_path}")

        print(f"[INFO] {stage_label.capitalize()} loaded at {prim_path}.", flush=True)

        return stage, prim

    @staticmethod
    def _stage_root_prim(stage: Any) -> Any:
        default_prim = stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            return default_prim

        pseudo_root = stage.GetPseudoRoot()
        for child in pseudo_root.GetChildren():
            if child and child.IsValid():
                return child

        raise RuntimeError("Unable to determine the stage root prim.")

    @staticmethod
    def _resolve_prim(
        stage: Any,
        query: str,
        *,
        require_path: bool = False,
        root_prim_path: str | None = None,
    ) -> Any:
        query = query.strip()
        if not query:
            raise ValueError("The prim query must be a non-empty string.")

        if query.startswith("/"):
            prim = stage.GetPrimAtPath(query)
            if prim and prim.IsValid():
                if root_prim_path and not (
                    query == root_prim_path or query.startswith(root_prim_path + "/")
                ):
                    raise RuntimeError(
                        f"Prim path {query} is outside the requested search root {root_prim_path}."
                    )
                return prim
            raise RuntimeError(f"Prim path was not found in the stage: {query}")

        if require_path:
            raise RuntimeError(
                f"Expected an explicit prim path, but received a non-path query: {query}"
            )

        query_lower = query.lower()
        exact_name_matches: list[Any] = []
        fuzzy_matches: list[Any] = []

        for prim in stage.Traverse():
            if not prim or not prim.IsValid() or not prim.IsActive():
                continue
            prim_name = prim.GetName()
            prim_path = prim.GetPath().pathString
            if root_prim_path and not (
                prim_path == root_prim_path or prim_path.startswith(root_prim_path + "/")
            ):
                continue
            if prim_name.lower() == query_lower:
                exact_name_matches.append(prim)
            elif query_lower in prim_name.lower() or query_lower in prim_path.lower():
                fuzzy_matches.append(prim)

        matches = exact_name_matches or fuzzy_matches
        if not matches:
            raise RuntimeError(f"No prim matched query '{query}'.")

        if len(matches) > 1:
            # Prefer the shortest path when the query matches multiple prims.
            matches = sorted(matches, key=lambda prim: len(prim.GetPath().pathString))
            first = matches[0].GetPath().pathString
            second = matches[1].GetPath().pathString
            if len(matches) > 1 and first != second:
                match_preview = ", ".join(
                    prim.GetPath().pathString for prim in matches[:8]
                )
                raise RuntimeError(
                    f"Prim query '{query}' is ambiguous. Matches include: {match_preview}"
                )

        return matches[0]

    @staticmethod
    def _compute_world_bbox(stage: Any, prim: Any, bbox_cache: Any | None = None) -> ObjectBBox3D:
        Usd, UsdGeom, _ = ObjectReachingGoalSampler._ensure_usd_imports()
        if bbox_cache is None:
            bbox_cache = UsdGeom.BBoxCache(
                time=Usd.TimeCode.Default(),
                includedPurposes=[
                    UsdGeom.Tokens.default_,
                    UsdGeom.Tokens.render,
                    UsdGeom.Tokens.proxy,
                ],
                useExtentsHint=True,
                ignoreVisibility=False,
            )

        world_bound = bbox_cache.ComputeWorldBound(prim)
        aligned = world_bound.ComputeAlignedBox()
        min_xyz = np.array(
            [float(value) for value in aligned.GetMin()],
            dtype=float,
        )
        max_xyz = np.array(
            [float(value) for value in aligned.GetMax()],
            dtype=float,
        )
        if np.any(~np.isfinite(min_xyz)) or np.any(~np.isfinite(max_xyz)):
            raise RuntimeError(f"Computed an invalid world-space bbox for {prim.GetPath().pathString}")

        return ObjectBBox3D(
            prim_path=prim.GetPath().pathString,
            min_xyz=min_xyz,
            max_xyz=max_xyz,
        )

    def load_environment(self, environment_usd_path: str) -> Any:
        self._ensure_runtime()
        stage_path = self._resolve_runtime_asset_path(environment_usd_path)
        if (
            self._environment_stage_path == stage_path
            and self._environment_stage is not None
            and self._environment_root_prim is not None
            and self._environment_root_prim.IsValid()
        ):
            print(
                f"[INFO] Reusing cached environment stage for {stage_path}.",
                flush=True,
            )
            return self._environment_stage

        Usd, UsdGeom, _ = self._ensure_runtime()
        stage, environment_root_prim = self._compose_single_reference_stage(
            usd_path=stage_path,
            prim_path="/World/Env/Environment",
            stage_label="environment asset",
        )
        bbox_cache = UsdGeom.BBoxCache(
            time=Usd.TimeCode.Default(),
            includedPurposes=[
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ],
            useExtentsHint=True,
            ignoreVisibility=False,
        )

        self._environment_stage = stage
        self._environment_stage_path = stage_path
        self._environment_bbox_cache = bbox_cache
        self._environment_root_prim = environment_root_prim
        self._last_object_bbox = None
        self._last_object_prim = None
        self._occluder_cache_by_object_path.clear()
        return stage

    def inspect_object_bbox(self, environment_usd_path: str, object_query: str) -> ObjectBBox3D:
        stage = self.load_environment(environment_usd_path)
        search_root_path = (
            self._environment_root_prim.GetPath().pathString
            if self._environment_root_prim is not None
            else None
        )
        prim = self._resolve_prim(stage, object_query, root_prim_path=search_root_path)
        bbox = self._compute_world_bbox(stage, prim, self._environment_bbox_cache)
        self._last_object_bbox = bbox
        self._last_object_prim = prim
        return bbox

    @staticmethod
    def _sorted_camera_candidates(root_prim: Any) -> list[Any]:
        _, UsdGeom, _ = ObjectReachingGoalSampler._ensure_usd_imports()
        cameras: list[tuple[int, Any]] = []
        preferred_tokens = ("front", "hawk", "stereo", "left", "rgb", "camera")

        for prim in root_prim.GetStage().Traverse():
            prim_path = prim.GetPath().pathString
            root_path = root_prim.GetPath().pathString
            if prim_path != root_path and not prim_path.startswith(root_path + "/"):
                continue
            if not prim.IsA(UsdGeom.Camera):
                continue

            path_lower = prim_path.lower()
            score = 0
            for token in preferred_tokens:
                if token in path_lower:
                    score += 1
            cameras.append((score, prim))

        cameras.sort(key=lambda item: (-item[0], len(item[1].GetPath().pathString)))
        return [prim for _, prim in cameras]

    @staticmethod
    def _compute_world_transform_matrix(prim: Any) -> np.ndarray:
        try:
            import omni.usd

            world_matrix = omni.usd.get_world_transform_matrix(prim)
            if world_matrix is not None:
                return _matrix_from_gf(world_matrix)
        except Exception:
            pass

        Usd, UsdGeom, _ = ObjectReachingGoalSampler._ensure_usd_imports()
        return _matrix_from_gf(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )

    @staticmethod
    def _resolve_robot_base_prim(root_prim: Any, explicit_path: str | None) -> Any:
        stage = root_prim.GetStage()
        if explicit_path:
            prim = stage.GetPrimAtPath(explicit_path)
            if prim and prim.IsValid():
                return prim
            raise RuntimeError(f"Robot base prim path was not found: {explicit_path}")

        root_path = root_prim.GetPath().pathString
        preferred_exact = (
            "base_link",
            "base_footprint",
            "chassis_link",
            "chassis",
            "base",
            "root_link",
        )
        ignored_tokens = (
            "sensor",
            "camera",
            "imu",
            "lidar",
            "wheel",
            "caster",
            "visual",
            "collision",
        )
        candidates: list[tuple[int, int, Any]] = []

        for prim in stage.Traverse():
            prim_path = prim.GetPath().pathString
            if prim_path != root_path and not prim_path.startswith(root_path + "/"):
                continue
            if prim_path == root_path:
                continue

            name_lower = prim.GetName().lower()
            path_lower = prim_path.lower()
            if any(token in path_lower for token in ignored_tokens):
                continue

            score = -1
            if name_lower in preferred_exact:
                score = 100
            elif name_lower.endswith("base_link") or name_lower.endswith("base_footprint"):
                score = 95
            elif name_lower.endswith("chassis_link"):
                score = 92
            elif "chassis" in name_lower:
                score = 90
            elif name_lower.startswith("base_") or name_lower.endswith("_base") or "base" in name_lower:
                score = 85
            elif name_lower.endswith("root_link"):
                score = 80

            if score >= 0:
                candidates.append((score, len(prim_path), prim))

        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            return candidates[0][2]

        return root_prim

    @staticmethod
    def _resolve_robot_camera_prim(root_prim: Any, explicit_path: str | None) -> Any:
        stage = root_prim.GetStage()
        if explicit_path:
            prim = stage.GetPrimAtPath(explicit_path)
            if prim and prim.IsValid():
                return prim
            raise RuntimeError(f"Robot camera prim path was not found: {explicit_path}")

        candidates = ObjectReachingGoalSampler._sorted_camera_candidates(root_prim)
        if not candidates:
            raise RuntimeError(
                "No camera prim was found under the robot asset. Pass camera_prim_path explicitly."
            )
        return candidates[0]

    @staticmethod
    def _bbox_in_base_frame(
        stage: Any,
        root_prim: Any,
        base_prim: Any,
    ) -> tuple[np.ndarray, np.ndarray, str, int]:
        Usd, UsdGeom, _ = ObjectReachingGoalSampler._ensure_usd_imports()
        try:
            from pxr import UsdPhysics
        except Exception:
            UsdPhysics = None
        bbox_cache = UsdGeom.BBoxCache(
            time=Usd.TimeCode.Default(),
            includedPurposes=[
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ],
            useExtentsHint=True,
            ignoreVisibility=False,
        )
        base_world = _matrix_from_gf(
            UsdGeom.Xformable(base_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )
        world_to_base = np.linalg.inv(base_world)
        root_path = root_prim.GetPath().pathString
        base_path = base_prim.GetPath().pathString
        ignored_tokens = (
            "sensor",
            "camera",
            "imu",
            "lidar",
            "ros",
            "graph",
            "ogn",
            "actiongraph",
            "renderproduct",
            "render_product",
            "frustum",
            "helper",
            "visualization",
            "viz",
        )

        def collect_bounds(
            search_root_path: str,
            *,
            require_collision_api: bool,
        ) -> tuple[list[np.ndarray], list[np.ndarray], int]:
            bbox_min_list: list[np.ndarray] = []
            bbox_max_list: list[np.ndarray] = []
            included_count = 0

            for prim in stage.Traverse():
                prim_path = prim.GetPath().pathString
                if prim_path != search_root_path and not prim_path.startswith(search_root_path + "/"):
                    continue
                if not prim or not prim.IsValid() or not prim.IsActive():
                    continue
                if not prim.IsA(UsdGeom.Gprim):
                    continue

                if require_collision_api:
                    if UsdPhysics is None or not prim.HasAPI(UsdPhysics.CollisionAPI):
                        continue

                path_lower = prim_path.lower()
                if any(token in path_lower for token in ignored_tokens):
                    continue

                world_bound = bbox_cache.ComputeWorldBound(prim)
                aligned = world_bound.ComputeAlignedBox()
                min_world = np.array([float(value) for value in aligned.GetMin()], dtype=float)
                max_world = np.array([float(value) for value in aligned.GetMax()], dtype=float)
                if np.any(~np.isfinite(min_world)) or np.any(~np.isfinite(max_world)):
                    continue
                if np.any(max_world <= min_world):
                    continue

                corners_world = ObjectBBox3D(
                    prim_path=prim_path,
                    min_xyz=min_world,
                    max_xyz=max_world,
                ).corners_xyz
                corners_base = np.vstack(
                    [_transform_point(world_to_base, corner) for corner in corners_world]
                )
                min_base = corners_base.min(axis=0)
                max_base = corners_base.max(axis=0)
                size_base = max_base - min_base
                center_base = 0.5 * (min_base + max_base)

                if (
                    np.linalg.norm(center_base[:2]) > DEFAULT_MAX_ROBOT_GEOMETRY_DISTANCE_M
                    or np.max(size_base) > DEFAULT_MAX_ROBOT_GEOMETRY_DIMENSION_M
                ):
                    continue

                bbox_min_list.append(min_base)
                bbox_max_list.append(max_base)
                included_count += 1

            return bbox_min_list, bbox_max_list, included_count

        search_strategies = (
            (base_path, True, "base subtree collision geometry"),
            (root_path, True, "robot collision geometry"),
            (base_path, False, "base subtree render/proxy geometry"),
            (root_path, False, "robot render/proxy geometry"),
        )
        for search_root_path, require_collision_api, source_label in search_strategies:
            bbox_min_list, bbox_max_list, included_count = collect_bounds(
                search_root_path,
                require_collision_api=require_collision_api,
            )
            if bbox_min_list and bbox_max_list:
                return (
                    np.min(np.vstack(bbox_min_list), axis=0),
                    np.max(np.vstack(bbox_max_list), axis=0),
                    source_label,
                    included_count,
                )

        bbox = ObjectReachingGoalSampler._compute_world_bbox(stage, root_prim, bbox_cache)
        corners_world = bbox.corners_xyz
        corners_base = np.vstack([_transform_point(world_to_base, corner) for corner in corners_world])
        return (
            corners_base.min(axis=0),
            corners_base.max(axis=0),
            "full robot root bbox fallback",
            1,
        )

    def inspect_robot_camera_spec(
        self,
        robot_usd_path: str,
        *,
        base_prim_path: str | None = None,
        camera_prim_path: str | None = None,
    ) -> RobotCameraSpec:
        self._ensure_runtime()
        resolved_robot_usd_path = self._resolve_runtime_asset_path(robot_usd_path)
        cache_key = (resolved_robot_usd_path, base_prim_path, camera_prim_path)
        cached = self._robot_specs.get(cache_key)
        if cached is not None:
            print(
                f"[INFO] Reusing cached robot camera spec for {cache_key[0]}.",
                flush=True,
            )
            return cached

        Usd, UsdGeom, _ = self._ensure_runtime()
        self._environment_stage = None
        self._environment_stage_path = None
        self._environment_bbox_cache = None
        self._environment_root_prim = None
        self._last_object_bbox = None
        self._last_object_prim = None
        self._occluder_cache_by_object_path.clear()
        stage, root_prim = self._compose_single_reference_stage(
            usd_path=resolved_robot_usd_path,
            prim_path="/World/RobotInspector/Robot",
            stage_label="robot asset",
        )
        base_prim = self._resolve_robot_base_prim(root_prim, base_prim_path)
        camera_prim = self._resolve_robot_camera_prim(root_prim, camera_prim_path)

        root_world = self._compute_world_transform_matrix(root_prim)
        base_world = self._compute_world_transform_matrix(base_prim)
        camera_world = self._compute_world_transform_matrix(camera_prim)
        root_from_base = np.linalg.inv(root_world) @ base_world
        root_from_base_translation = root_from_base[:3, 3].copy()
        root_from_base_yaw = _yaw_from_matrix(root_from_base)
        camera_rel = np.linalg.inv(base_world) @ camera_world
        camera_rel_translation = camera_rel[:3, 3].copy()

        camera_forward_base = camera_rel[:3, :3] @ np.array([0.0, 0.0, -1.0], dtype=float)
        camera_forward_yaw = math.atan2(camera_forward_base[1], camera_forward_base[0])

        camera = UsdGeom.Camera(camera_prim)
        focal_length_mm = float(camera.GetFocalLengthAttr().Get() or 18.0)
        horizontal_aperture_mm = float(camera.GetHorizontalApertureAttr().Get() or 20.955)
        vertical_aperture_mm = float(camera.GetVerticalApertureAttr().Get() or 15.2908)
        clipping_range = camera.GetClippingRangeAttr().Get()
        near_clip = float(clipping_range[0]) if clipping_range else 0.05
        far_clip = float(clipping_range[1]) if clipping_range else 100.0

        bbox_min_base, bbox_max_base, footprint_source, included_prim_count = self._bbox_in_base_frame(
            stage,
            root_prim,
            base_prim,
        )
        robot_size_xyz = bbox_max_base - bbox_min_base
        bbox_xy = np.array(
            [
                [bbox_min_base[0], bbox_min_base[1]],
                [bbox_min_base[0], bbox_max_base[1]],
                [bbox_max_base[0], bbox_min_base[1]],
                [bbox_max_base[0], bbox_max_base[1]],
            ],
            dtype=float,
        )
        robot_planar_radius = float(np.max(np.linalg.norm(bbox_xy, axis=1)))
        print(
            "[INFO] Robot root/base fixed offset: "
            f"root_from_base_translation={root_from_base_translation.tolist()}, "
            f"root_from_base_yaw={math.degrees(root_from_base_yaw):.2f} deg.",
            flush=True,
        )
        print(
            "[INFO] Robot base/camera selection: "
            f"base_prim={base_prim.GetPath().pathString}, "
            f"camera_prim={camera_prim.GetPath().pathString}, "
            f"camera_rel_translation={camera_rel_translation.tolist()}, "
            f"camera_forward_yaw_deg={math.degrees(camera_forward_yaw):.2f}",
            flush=True,
        )
        print(
            "[INFO] Robot footprint bbox in base frame: "
            f"source={footprint_source}, "
            f"included_prims={included_prim_count}, "
            f"min={bbox_min_base.tolist()}, max={bbox_max_base.tolist()}, "
            f"size={robot_size_xyz.tolist()}, raw_radius={robot_planar_radius:.3f} m.",
            flush=True,
        )
        if (
            not np.isfinite(robot_planar_radius)
            or robot_planar_radius <= 0.0
            or robot_planar_radius > DEFAULT_MAX_ROBOT_GEOMETRY_DISTANCE_M
        ):
            print(
                "[WARN] Inferred robot planar radius looks unreasonable; "
                f"falling back to {DEFAULT_FALLBACK_ROBOT_RADIUS_M:.2f} m.",
                flush=True,
            )
            robot_planar_radius = DEFAULT_FALLBACK_ROBOT_RADIUS_M

        spec = RobotCameraSpec(
            robot_usd_path=cache_key[0],
            root_prim_path=root_prim.GetPath().pathString,
            base_prim_path=base_prim.GetPath().pathString,
            camera_prim_path=camera_prim.GetPath().pathString,
            root_from_base_matrix=root_from_base,
            root_from_base_translation_xyz=root_from_base_translation,
            root_from_base_yaw_rad=root_from_base_yaw,
            camera_rel_matrix=camera_rel,
            camera_rel_translation_xyz=camera_rel_translation,
            camera_forward_base_xyz=camera_forward_base,
            camera_forward_yaw_rad=camera_forward_yaw,
            camera_horizontal_fov_rad=2.0 * math.atan(horizontal_aperture_mm / (2.0 * focal_length_mm)),
            camera_vertical_fov_rad=2.0 * math.atan(vertical_aperture_mm / (2.0 * focal_length_mm)),
            camera_clipping_range_m=(near_clip, far_clip),
            robot_planar_radius_m=robot_planar_radius,
            robot_size_xyz_m=robot_size_xyz,
            robot_bbox_base_frame_min_xyz=bbox_min_base,
            robot_bbox_base_frame_max_xyz=bbox_max_base,
        )
        self._robot_specs[cache_key] = spec
        return spec

    def _get_or_build_occluders(self, object_prim: Any) -> list[OccluderBBox]:
        object_path = object_prim.GetPath().pathString
        cached = self._occluder_cache_by_object_path.get(object_path)
        if cached is not None:
            return cached

        stage = object_prim.GetStage()
        Usd, UsdGeom, _ = self._ensure_usd_imports()
        bbox_cache = self._environment_bbox_cache
        if bbox_cache is None:
            raise RuntimeError("Environment bbox cache is not initialized.")

        object_prefix = object_path + "/"
        occluders: list[OccluderBBox] = []
        print("[INFO] Building stage occluder cache for visibility checks...", flush=True)

        for index, prim in enumerate(stage.Traverse(), start=1):
            if index % DEFAULT_PROGRESS_LOG_INTERVAL == 0:
                print(
                    f"[INFO] Occluder scan progress: visited {index} prims, "
                    f"cached {len(occluders)} occluders so far.",
                    flush=True,
                )
                self._update_runtime(self._simulation_app, 1)
            if not prim or not prim.IsValid() or not prim.IsActive():
                continue
            prim_path = prim.GetPath().pathString
            if prim_path == object_path or prim_path.startswith(object_prefix):
                continue
            if not prim.IsA(UsdGeom.Gprim):
                continue

            imageable = UsdGeom.Imageable(prim)
            if (
                imageable
                and imageable.ComputeVisibility(Usd.TimeCode.Default()) == UsdGeom.Tokens.invisible
            ):
                continue

            world_bound = bbox_cache.ComputeWorldBound(prim)
            aligned = world_bound.ComputeAlignedBox()
            min_xyz = np.array([float(value) for value in aligned.GetMin()], dtype=float)
            max_xyz = np.array([float(value) for value in aligned.GetMax()], dtype=float)
            if np.any(~np.isfinite(min_xyz)) or np.any(~np.isfinite(max_xyz)):
                continue
            if np.any(max_xyz <= min_xyz):
                continue

            occluders.append(OccluderBBox(prim_path=prim_path, min_xyz=min_xyz, max_xyz=max_xyz))

        self._occluder_cache_by_object_path[object_path] = occluders
        print(
            f"[INFO] Occluder cache ready with {len(occluders)} entries for {object_path}.",
            flush=True,
        )
        return occluders

    @staticmethod
    def _solve_base_yaw(
        base_xy: np.ndarray,
        target_xyz: np.ndarray,
        robot_spec: RobotCameraSpec,
        base_z_m: float,
        max_iterations: int = 12,
    ) -> float:
        yaw = math.atan2(target_xyz[1] - base_xy[1], target_xyz[0] - base_xy[0]) - robot_spec.camera_forward_yaw_rad
        translation_xy = robot_spec.camera_rel_translation_xyz[:2]

        for _ in range(max_iterations):
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            camera_xy = np.array(
                [
                    base_xy[0] + cos_yaw * translation_xy[0] - sin_yaw * translation_xy[1],
                    base_xy[1] + sin_yaw * translation_xy[0] + cos_yaw * translation_xy[1],
                ],
                dtype=float,
            )
            updated = (
                math.atan2(target_xyz[1] - camera_xy[1], target_xyz[0] - camera_xy[0])
                - robot_spec.camera_forward_yaw_rad
            )
            if abs(_wrap_to_pi(updated - yaw)) < 1e-5:
                yaw = updated
                break
            yaw = updated

        return _wrap_to_pi(yaw)

    @staticmethod
    def _camera_world_matrix(
        base_position_xyz: Sequence[float],
        base_yaw_rad: float,
        robot_spec: RobotCameraSpec,
    ) -> np.ndarray:
        base_matrix = _translation_matrix(base_position_xyz) @ _rotation_matrix_z(base_yaw_rad)
        return base_matrix @ robot_spec.camera_rel_matrix

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

    @staticmethod
    def _count_visible_bbox_corners(
        camera_world: np.ndarray,
        object_bbox: ObjectBBox3D,
        robot_spec: RobotCameraSpec,
        visibility_margin_deg: float,
    ) -> int:
        visibility_margin_rad = math.radians(max(0.0, visibility_margin_deg))
        tan_h = math.tan(max(1e-6, 0.5 * robot_spec.camera_horizontal_fov_rad - visibility_margin_rad))
        tan_v = math.tan(max(1e-6, 0.5 * robot_spec.camera_vertical_fov_rad - visibility_margin_rad))
        near_clip, far_clip = robot_spec.camera_clipping_range_m
        world_to_camera = np.linalg.inv(camera_world)

        visible = 0
        for corner in object_bbox.corners_xyz:
            camera_xyz = _transform_point(world_to_camera, corner)
            depth = -camera_xyz[2]
            if depth <= near_clip or depth >= far_clip:
                continue
            if abs(camera_xyz[0]) > depth * tan_h:
                continue
            if abs(camera_xyz[1]) > depth * tan_v:
                continue
            visible += 1

        return visible

    @staticmethod
    def _visible_bbox_corners(
        camera_world: np.ndarray,
        object_bbox: ObjectBBox3D,
        robot_spec: RobotCameraSpec,
        visibility_margin_deg: float,
    ) -> list[np.ndarray]:
        visibility_margin_rad = math.radians(max(0.0, visibility_margin_deg))
        tan_h = math.tan(max(1e-6, 0.5 * robot_spec.camera_horizontal_fov_rad - visibility_margin_rad))
        tan_v = math.tan(max(1e-6, 0.5 * robot_spec.camera_vertical_fov_rad - visibility_margin_rad))
        near_clip, far_clip = robot_spec.camera_clipping_range_m
        world_to_camera = np.linalg.inv(camera_world)

        visible: list[np.ndarray] = []
        for corner in object_bbox.corners_xyz:
            camera_xyz = _transform_point(world_to_camera, corner)
            depth = -camera_xyz[2]
            if depth <= near_clip or depth >= far_clip:
                continue
            if abs(camera_xyz[0]) > depth * tan_h:
                continue
            if abs(camera_xyz[1]) > depth * tan_v:
                continue
            visible.append(corner)

        return visible

    @staticmethod
    def _target_point_visible(
        camera_world: np.ndarray,
        target_xyz: np.ndarray,
        robot_spec: RobotCameraSpec,
        visibility_margin_deg: float,
    ) -> bool:
        visibility_margin_rad = math.radians(max(0.0, visibility_margin_deg))
        tan_h = math.tan(max(1e-6, 0.5 * robot_spec.camera_horizontal_fov_rad - visibility_margin_rad))
        tan_v = math.tan(max(1e-6, 0.5 * robot_spec.camera_vertical_fov_rad - visibility_margin_rad))
        near_clip, far_clip = robot_spec.camera_clipping_range_m
        world_to_camera = np.linalg.inv(camera_world)
        camera_xyz = _transform_point(world_to_camera, target_xyz)
        depth = -camera_xyz[2]
        if depth <= near_clip or depth >= far_clip:
            return False
        if abs(camera_xyz[0]) > depth * tan_h:
            return False
        if abs(camera_xyz[1]) > depth * tan_v:
            return False
        return True

    @staticmethod
    def _candidate_object_sight_points(
        camera_xyz: np.ndarray,
        object_bbox: ObjectBBox3D,
        preferred_target_xyz: np.ndarray,
        visible_corners_xyz: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        points: list[np.ndarray] = []
        seen_keys: set[tuple[float, float, float]] = set()

        def add_point(point_xyz: np.ndarray) -> None:
            key = tuple(np.round(point_xyz.astype(float), 4).tolist())
            if key in seen_keys:
                return
            seen_keys.add(key)
            points.append(point_xyz.astype(float).copy())

        # Closest point on the bbox to the camera approximates the visible object surface.
        nearest_surface_point = np.minimum(
            np.maximum(camera_xyz, object_bbox.min_xyz),
            object_bbox.max_xyz,
        )
        add_point(nearest_surface_point)
        add_point(preferred_target_xyz)
        for corner_xyz in visible_corners_xyz:
            add_point(corner_xyz)

        return points

    def _has_stage_line_of_sight(
        self,
        camera_xyz: np.ndarray,
        candidate_target_points_xyz: Sequence[np.ndarray],
        object_prim: Any,
        occluders: Sequence[OccluderBBox] | None = None,
        target_tail_relaxation_m: float = 0.0,
    ) -> bool:
        occluders = list(occluders) if occluders is not None else self._get_or_build_occluders(object_prim)
        for target_xyz in candidate_target_points_xyz:
            ray_length = float(np.linalg.norm(np.asarray(target_xyz, dtype=float) - np.asarray(camera_xyz, dtype=float)))
            if ray_length <= 1e-6:
                continue
            blocked = False
            for occluder in occluders:
                hit_interval = _segment_aabb_interval(
                    camera_xyz,
                    target_xyz,
                    occluder.min_xyz,
                    occluder.max_xyz,
                )
                if hit_interval is None:
                    continue
                hit_t_enter, _ = hit_interval
                distance_from_hit_to_target = max(0.0, (1.0 - hit_t_enter) * ray_length)
                if distance_from_hit_to_target <= max(0.0, target_tail_relaxation_m):
                    continue
                blocked = True
                break
            if not blocked:
                return True
        return False

    def sample_goal_poses(
        self,
        *,
        object_bbox: ObjectBBox3D,
        occupancy_map_path: str,
        robot_usd_path: str,
        required_samples: int = DEFAULT_REQUIRED_SAMPLES,
        environment_usd_path: str | None = None,
        object_query: str | None = None,
        occupancy_map_resolution_m: float | None = None,
        occupancy_map_origin_xyz: Sequence[float] | None = None,
        occupancy_map_negate: bool | int | None = None,
        min_distance_m: float = DEFAULT_DISTANCE_RANGE_M[0],
        max_distance_m: float = DEFAULT_DISTANCE_RANGE_M[1],
        min_sample_separation_m: float | None = None,
        base_z_m: float = 0.0,
        target_height_ratio: float = DEFAULT_TARGET_HEIGHT_RATIO,
        visibility_margin_deg: float = DEFAULT_VISIBILITY_MARGIN_DEG,
        min_visible_bbox_corners: int = DEFAULT_MIN_VISIBLE_BBOX_CORNERS,
        safety_margin_m: float = DEFAULT_SAFETY_MARGIN_M,
        base_prim_path: str | None = None,
        camera_prim_path: str | None = None,
        require_stage_occlusion_check: bool = True,
        require_2d_line_of_sight: bool | None = None,
        stage_occlusion_tail_relaxation_m: float = DEFAULT_STAGE_OCCLUSION_TAIL_RELAXATION_M,
        strict_sample_count: bool = True,
        max_candidate_evaluations: int | None = DEFAULT_MAX_CANDIDATE_EVALUATIONS,
    ) -> list[SampledGoalPose]:
        sample_start_time = time.perf_counter()

        def log_sampling_step(message: str) -> None:
            elapsed_s = time.perf_counter() - sample_start_time
            print(
                f"[INFO] [sample_goal_poses +{elapsed_s:.2f}s] {message}",
                flush=True,
            )

        if required_samples <= 0:
            raise ValueError("required_samples must be positive.")
        if max_distance_m <= min_distance_m:
            raise ValueError("max_distance_m must be larger than min_distance_m.")

        log_sampling_step(
            "Starting sampling with "
            f"required_samples={required_samples}, "
            f"distance_range=[{min_distance_m:.2f}, {max_distance_m:.2f}] m, "
            f"stage_occlusion={'on' if require_stage_occlusion_check else 'off'}."
        )
        if require_2d_line_of_sight is None:
            require_2d_line_of_sight = not require_stage_occlusion_check
        log_sampling_step(
            f"2D line-of-sight fallback is {'on' if require_2d_line_of_sight else 'off'}."
        )
        if require_stage_occlusion_check:
            log_sampling_step(
                "Stage occlusion tail relaxation is "
                f"{max(0.0, stage_occlusion_tail_relaxation_m):.2f} m."
            )

        if require_stage_occlusion_check:
            log_sampling_step("Preparing cached environment/object state for stage occlusion checks.")
            if environment_usd_path is not None:
                log_sampling_step("Calling load_environment() from sample_goal_poses.")
                self.load_environment(environment_usd_path)
            if self._environment_stage is None:
                raise RuntimeError(
                    "Stage occlusion checks require an environment USD. Call inspect_object_bbox() "
                    "first or pass environment_usd_path."
                )
            if object_query is not None:
                if (
                    self._last_object_bbox is None
                    or self._last_object_prim is None
                    or self._last_object_bbox.prim_path != object_bbox.prim_path
                ):
                    log_sampling_step(
                        "Cached object bbox/prim is missing or stale; re-running inspect_object_bbox()."
                    )
                    self.inspect_object_bbox(environment_usd_path or "", object_query)
            elif self._environment_stage is not None:
                log_sampling_step("Rebinding target prim from cached environment stage.")
                target_prim = self._environment_stage.GetPrimAtPath(object_bbox.prim_path)
                if target_prim and target_prim.IsValid():
                    self._last_object_prim = target_prim
                    self._last_object_bbox = object_bbox
            if self._last_object_prim is None or self._last_object_bbox is None:
                raise RuntimeError(
                    "Stage occlusion checks require a cached target object. Call inspect_object_bbox() first."
                )
            if self._last_object_bbox.prim_path != object_bbox.prim_path:
                target_prim = self._environment_stage.GetPrimAtPath(object_bbox.prim_path)
                if not target_prim or not target_prim.IsValid():
                    raise RuntimeError(
                        f"The target object prim is not present in the cached environment stage: {object_bbox.prim_path}"
                    )
                self._last_object_prim = target_prim
                self._last_object_bbox = object_bbox
            log_sampling_step(
                f"Stage occlusion target prim ready at {self._last_object_prim.GetPath().pathString}."
            )

        log_sampling_step(f"Loading occupancy map from {occupancy_map_path}.")
        occupancy_map = OccupancyMap.load(
            occupancy_map_path,
            resolution_m=occupancy_map_resolution_m,
            origin_xyz=occupancy_map_origin_xyz,
            negate=occupancy_map_negate,
        )
        log_sampling_step(
            "Occupancy map loaded: "
            f"size={occupancy_map.width}x{occupancy_map.height}, "
            f"resolution={occupancy_map.resolution_m:.3f} m, "
            f"origin={occupancy_map.origin_xyz.tolist()}."
        )
        log_sampling_step(f"Resolving robot camera spec for {robot_usd_path}.")
        robot_spec = self.inspect_robot_camera_spec(
            robot_usd_path,
            base_prim_path=base_prim_path,
            camera_prim_path=camera_prim_path,
        )
        log_sampling_step(
            "Robot camera spec ready: "
            f"camera={robot_spec.camera_prim_path}, "
            f"radius={robot_spec.robot_planar_radius_m:.3f} m."
        )

        min_sample_separation_m = float(
            min_sample_separation_m
            if min_sample_separation_m is not None
            else max(2.0 * robot_spec.robot_planar_radius_m, 0.75)
        )
        log_sampling_step(
            "Inflating occupancy map with robot radius + safety margin: "
            f"{robot_spec.robot_planar_radius_m + float(safety_margin_m):.3f} m."
        )
        inflated_mask = occupancy_map.inflate_occupied_mask(
            robot_spec.robot_planar_radius_m + float(safety_margin_m)
        )
        log_sampling_step("Inflated occupancy mask ready.")

        target_xyz = object_bbox.target_point(target_height_ratio)
        target_xy = target_xyz[:2].copy()
        log_sampling_step(f"Target point selected at {target_xyz.tolist()}.")
        free_rows, free_cols = np.nonzero(~inflated_mask)
        if free_rows.size == 0:
            raise RuntimeError("No collision-free cells remain after occupancy inflation.")
        log_sampling_step(f"Found {free_rows.size} collision-free cells before distance filtering.")

        free_xy = occupancy_map.grid_to_world_xy(free_rows, free_cols)
        distances = np.linalg.norm(free_xy - target_xyz[:2], axis=1)
        candidate_mask = (distances >= float(min_distance_m)) & (distances <= float(max_distance_m))
        candidate_rows = free_rows[candidate_mask]
        candidate_cols = free_cols[candidate_mask]
        candidate_xy = free_xy[candidate_mask]

        if candidate_rows.size == 0:
            raise RuntimeError(
                "No occupancy-map cells remain in the requested sampling annulus "
                f"[{min_distance_m:.2f}, {max_distance_m:.2f}] meters."
            )

        log_sampling_step(f"Candidate free cells in distance range: {candidate_rows.size}.")
        candidate_order = self._rng.permutation(candidate_rows.size)
        sampled: list[SampledGoalPose] = []
        base_xy_history: list[np.ndarray] = []
        occluders: list[OccluderBBox] | None = None
        visibility_blocked_mask = occupancy_map.occupied_mask if require_2d_line_of_sight else None
        object_xy_clearance_m = max(
            0.5 * float(np.max(object_bbox.size_xyz[:2])),
            2.0 * occupancy_map.resolution_m,
        )
        rejection_counts = {
            "too_close_to_other_samples": 0,
            "camera_distance_out_of_range": 0,
            "2d_line_blocked": 0,
            "target_not_in_view": 0,
            "bbox_corner_visibility": 0,
            "occluded_in_stage": 0,
            "accepted": 0,
        }

        if require_stage_occlusion_check:
            log_sampling_step("Building or fetching occluder cache.")
            occluders = self._get_or_build_occluders(self._last_object_prim)
            log_sampling_step(f"Occluder cache available with {len(occluders)} entries.")

        evaluation_budget = (
            int(max_candidate_evaluations)
            if max_candidate_evaluations is not None and int(max_candidate_evaluations) > 0
            else candidate_rows.size
        )
        if evaluation_budget < candidate_rows.size:
            log_sampling_step(
                f"Sampling will evaluate at most {evaluation_budget} randomized candidates."
            )
        else:
            log_sampling_step(
                f"Sampling will evaluate up to all {candidate_rows.size} randomized candidates."
            )

        log_sampling_step("Entering candidate evaluation loop.")
        for evaluated_count, candidate_index in enumerate(candidate_order.tolist(), start=1):
            if evaluated_count > evaluation_budget:
                print(
                    f"[WARN] Reached candidate evaluation budget ({evaluation_budget}) with "
                    f"{len(sampled)} valid sample(s) found.",
                    flush=True,
                )
                break
            if evaluated_count % DEFAULT_PROGRESS_LOG_INTERVAL == 0:
                print(
                    f"[INFO] Sampling progress: evaluated {evaluated_count}/{min(candidate_rows.size, evaluation_budget)} "
                    f"candidates, accepted {len(sampled)} samples.",
                    flush=True,
                )
                self._update_runtime(self._simulation_app, 1)
            base_xy = candidate_xy[candidate_index]
            if base_xy_history:
                nearest_selected = min(
                    np.linalg.norm(base_xy - selected_xy) for selected_xy in base_xy_history
                )
                if nearest_selected < min_sample_separation_m:
                    rejection_counts["too_close_to_other_samples"] += 1
                    continue

            base_yaw = self._solve_base_yaw(base_xy, target_xyz, robot_spec, base_z_m)
            base_position_xyz = np.array([base_xy[0], base_xy[1], float(base_z_m)], dtype=float)
            camera_world = self._camera_world_matrix(base_position_xyz, base_yaw, robot_spec)
            camera_xyz = camera_world[:3, 3].copy()
            distance_to_target = float(np.linalg.norm(target_xyz - camera_xyz))
            base_planar_distance_to_target = float(np.linalg.norm(base_xy - target_xy))

            if (
                base_planar_distance_to_target < min_distance_m
                or base_planar_distance_to_target > max_distance_m
            ):
                rejection_counts["camera_distance_out_of_range"] += 1
                continue

            if require_2d_line_of_sight and visibility_blocked_mask is not None:
                camera_xy = camera_xyz[:2]
                direction_xy = target_xy - camera_xy
                direction_norm = float(np.linalg.norm(direction_xy))
                visibility_target_xy = target_xy.copy()
                if direction_norm > 1e-6:
                    visibility_target_xy = (
                        target_xy - (direction_xy / direction_norm) * object_xy_clearance_m
                    )

                if not occupancy_map.line_is_free(
                    camera_xy,
                    visibility_target_xy,
                    visibility_blocked_mask,
                    allow_end_in_occupied=True,
                ):
                    rejection_counts["2d_line_blocked"] += 1
                    continue

            if not self._target_point_visible(
                camera_world,
                target_xyz,
                robot_spec,
                visibility_margin_deg,
            ):
                rejection_counts["target_not_in_view"] += 1
                continue

            visible_corners_xyz = self._visible_bbox_corners(
                camera_world,
                object_bbox,
                robot_spec,
                visibility_margin_deg,
            )
            visible_corner_count = len(visible_corners_xyz)
            if visible_corner_count < int(min_visible_bbox_corners):
                rejection_counts["bbox_corner_visibility"] += 1
                continue

            if require_stage_occlusion_check:
                candidate_sight_points = self._candidate_object_sight_points(
                    camera_xyz,
                    object_bbox,
                    target_xyz,
                    visible_corners_xyz,
                )
                if not self._has_stage_line_of_sight(
                    camera_xyz,
                    candidate_sight_points,
                    self._last_object_prim,
                    occluders=occluders,
                    target_tail_relaxation_m=stage_occlusion_tail_relaxation_m,
                ):
                    rejection_counts["occluded_in_stage"] += 1
                    continue

            sampled.append(
                SampledGoalPose(
                    base_position_xyz=base_position_xyz,
                    yaw_rad=base_yaw,
                    camera_position_xyz=camera_xyz,
                    target_point_xyz=target_xyz.copy(),
                    distance_to_target_m=distance_to_target,
                    visible_bbox_corner_count=visible_corner_count,
                    source_grid_row=int(candidate_rows[candidate_index]),
                    source_grid_col=int(candidate_cols[candidate_index]),
                )
            )
            base_xy_history.append(base_xy.copy())
            rejection_counts["accepted"] += 1

            if len(sampled) >= required_samples:
                break

        log_sampling_step(f"Rejection summary: {json.dumps(rejection_counts, sort_keys=True)}")
        if len(sampled) < required_samples:
            message = (
                f"Only found {len(sampled)} valid sample(s) out of the requested "
                f"{required_samples}. Try a different --object-query, increase "
                "--stage-occlusion-tail-relaxation, disable stage occlusion, or relax the "
                "distance range, visibility margin, or minimum sample separation."
            )
            if strict_sample_count:
                raise RuntimeError(message)
            warnings.warn(message)

        return sampled


GoalSampler = ObjectReachingGoalSampler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample collision-free object-reaching goal poses for a robot in Isaac Sim."
    )
    parser.add_argument(
        "--environment-usd",
        required=True,
        help="Environment USD to inspect for target-object bbox and GUI validation.",
    )
    parser.add_argument(
        "--robot-usd",
        default=None,
        help="Robot USD used to infer base-to-camera transform, camera FOV, and footprint size. Required for sampling and GUI validation.",
    )
    parser.add_argument(
        "--occupancy-map",
        default=None,
        help="Occupancy-map YAML or image path. A raw image also requires --map-resolution and --map-origin. Required for sampling and GUI validation.",
    )
    parser.add_argument(
        "--object-query",
        default=None,
        help="Target object prim path, exact prim name, or fuzzy name/path query. Required for sampling unless --list-object-prims is used.",
    )
    parser.add_argument(
        "--list-object-prims",
        action="store_true",
        help="List candidate object prims from the environment USD and exit without sampling.",
    )
    parser.add_argument(
        "--object-filter",
        default=None,
        help="Optional case-insensitive substring filter used with --list-object-prims.",
    )
    parser.add_argument(
        "--object-list-mode",
        choices=OBJECT_LIST_MODES,
        default="representative",
        help=(
            "How aggressively to clean up object-list output: "
            "`representative` dedupes and ranks object-like anchors, "
            "`components` keeps meaningful sub-objects, and `raw` exposes broad debug output."
        ),
    )
    parser.add_argument(
        "--object-list-limit",
        type=int,
        default=100,
        help="Maximum number of object-prim candidates to print when using --list-object-prims.",
    )
    parser.add_argument(
        "--required-samples",
        type=int,
        default=DEFAULT_REQUIRED_SAMPLES,
        help=f"Number of goal samples to return. Defaults to {DEFAULT_REQUIRED_SAMPLES}.",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=DEFAULT_DISTANCE_RANGE_M[0],
        help=f"Minimum object distance in meters. Defaults to {DEFAULT_DISTANCE_RANGE_M[0]:.2f}.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=DEFAULT_DISTANCE_RANGE_M[1],
        help=f"Maximum object distance in meters. Defaults to {DEFAULT_DISTANCE_RANGE_M[1]:.2f}.",
    )
    parser.add_argument(
        "--min-sample-separation",
        type=float,
        default=None,
        help="Minimum planar distance between returned samples in meters.",
    )
    parser.add_argument(
        "--visibility-margin-deg",
        type=float,
        default=DEFAULT_VISIBILITY_MARGIN_DEG,
        help=f"Extra image-border margin used by the visibility checks. Defaults to {DEFAULT_VISIBILITY_MARGIN_DEG:.1f}.",
    )
    parser.add_argument(
        "--min-visible-bbox-corners",
        type=int,
        default=DEFAULT_MIN_VISIBLE_BBOX_CORNERS,
        help=f"Minimum visible bbox corners required per sample. Defaults to {DEFAULT_MIN_VISIBLE_BBOX_CORNERS}.",
    )
    parser.add_argument(
        "--target-height-ratio",
        type=float,
        default=DEFAULT_TARGET_HEIGHT_RATIO,
        help=f"Where to aim inside the object bbox, from 0.0=min z to 1.0=max z. Defaults to {DEFAULT_TARGET_HEIGHT_RATIO:.2f}.",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=DEFAULT_SAFETY_MARGIN_M,
        help=f"Additional occupancy-map inflation on top of the inferred robot radius. Defaults to {DEFAULT_SAFETY_MARGIN_M:.2f} m.",
    )
    parser.add_argument(
        "--base-z",
        type=float,
        default=0.0,
        help="Robot base height in world coordinates. Defaults to 0.0.",
    )
    parser.add_argument(
        "--base-prim-path",
        default=None,
        help="Optional explicit robot base prim path inside the robot USD.",
    )
    parser.add_argument(
        "--camera-prim-path",
        default=None,
        help="Optional explicit camera prim path inside the robot USD.",
    )
    parser.add_argument(
        "--map-resolution",
        type=float,
        default=None,
        help="Required when --occupancy-map points at a raw image instead of a YAML file.",
    )
    parser.add_argument(
        "--map-origin",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "YAW"),
        help="Required when --occupancy-map points at a raw image instead of a YAML file.",
    )
    parser.add_argument(
        "--map-negate",
        type=int,
        choices=(0, 1),
        default=None,
        help="Optional negate flag for raw occupancy images.",
    )
    parser.add_argument(
        "--disable-stage-occlusion-check",
        action="store_true",
        help="Skip USD geometry occlusion checks and only rely on the 2D occupancy map.",
    )
    parser.add_argument(
        "--stage-occlusion-tail-relaxation",
        type=float,
        default=DEFAULT_STAGE_OCCLUSION_TAIL_RELAXATION_M,
        help=(
            "Ignore stage-occlusion hits that occur only within the final meters near the target "
            f"object surface. Defaults to {DEFAULT_STAGE_OCCLUSION_TAIL_RELAXATION_M:.2f} m."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return fewer than --required-samples instead of failing when the constraints are too strict.",
    )
    parser.add_argument(
        "--validate-gui",
        action="store_true",
        help="After sampling, open Isaac Sim with one robot placed at a randomly selected sample pose.",
    )
    parser.add_argument(
        "--robot-name",
        default=DEFAULT_MANUAL_GOAL_ROBOT_NAME,
        help=(
            "Robot name used when printing the manual goal snippet in "
            "warehouse_team_config.yaml format."
        ),
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Optional fixed sample index for GUI validation instead of drawing one randomly.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional file path where the object bbox, robot spec, and sampled poses are written as JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sampler = ObjectReachingGoalSampler(seed=args.seed)
    try:
        print("[INFO] Goal sampler started.", flush=True)
        if not args.validate_gui:
            print(
                "[INFO] GUI validation is disabled. Sampling will run headless unless Isaac Sim "
                "needs to be bootstrapped for USD access.",
                flush=True,
            )

        if args.list_object_prims:
            sampler._ensure_runtime(require_gui=False)
            print("[INFO] Listing candidate object prims...", flush=True)
            object_candidates = sampler.list_object_prim_candidates(
                args.environment_usd,
                name_filter=args.object_filter,
                mode=args.object_list_mode,
                limit=args.object_list_limit,
            )
            payload = {
                "environment_usd": args.environment_usd,
                "object_filter": args.object_filter,
                "object_list_mode": args.object_list_mode,
                "object_candidates": object_candidates,
            }
            print(json.dumps(payload, indent=2), flush=True)
            if args.output_json:
                output_path = os.path.abspath(os.path.expanduser(args.output_json))
                with open(output_path, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, indent=2)
                print(f"[INFO] Wrote object candidates to {output_path}", flush=True)
            return

        missing_args: list[str] = []
        if not args.robot_usd:
            missing_args.append("--robot-usd")
        if not args.occupancy_map:
            missing_args.append("--occupancy-map")
        if not args.object_query:
            missing_args.append("--object-query")
        if missing_args:
            raise ValueError(
                "Sampling mode requires the following arguments: "
                + ", ".join(missing_args)
                + ". Use --list-object-prims to inspect target-object options first."
            )

        sampler._ensure_runtime(require_gui=args.validate_gui)

        print("[INFO] Inspecting robot camera and footprint metadata...", flush=True)
        robot_spec = sampler.inspect_robot_camera_spec(
            args.robot_usd,
            base_prim_path=args.base_prim_path,
            camera_prim_path=args.camera_prim_path,
        )
        print("[INFO] Inspecting target object bbox...", flush=True)
        object_bbox = sampler.inspect_object_bbox(args.environment_usd, args.object_query)
        print("[INFO] Sampling goal poses...", flush=True)
        sampled_poses = sampler.sample_goal_poses(
            object_bbox=object_bbox,
            occupancy_map_path=args.occupancy_map,
            robot_usd_path=args.robot_usd,
            required_samples=args.required_samples,
            environment_usd_path=args.environment_usd,
            object_query=args.object_query,
            min_distance_m=args.min_distance,
            max_distance_m=args.max_distance,
            min_sample_separation_m=args.min_sample_separation,
            base_z_m=args.base_z,
            target_height_ratio=args.target_height_ratio,
            visibility_margin_deg=args.visibility_margin_deg,
            min_visible_bbox_corners=args.min_visible_bbox_corners,
            safety_margin_m=args.safety_margin,
            base_prim_path=args.base_prim_path,
            camera_prim_path=args.camera_prim_path,
            require_stage_occlusion_check=not args.disable_stage_occlusion_check,
            require_2d_line_of_sight=None,
            stage_occlusion_tail_relaxation_m=args.stage_occlusion_tail_relaxation,
            strict_sample_count=not args.allow_partial,
            occupancy_map_resolution_m=args.map_resolution,
            occupancy_map_origin_xyz=args.map_origin,
            occupancy_map_negate=args.map_negate,
        )
        print(f"[INFO] Sampled {len(sampled_poses)} pose(s).", flush=True)

        payload = {
            "object_bbox": object_bbox.as_dict(),
            "robot_camera_spec": robot_spec.as_dict(),
            "sampled_poses": [sample.as_dict() for sample in sampled_poses],
        }

        print(json.dumps(payload, indent=2), flush=True)

        if args.output_json:
            output_path = os.path.abspath(os.path.expanduser(args.output_json))
            with open(output_path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
            print(f"[INFO] Wrote sampled poses to {output_path}", flush=True)

        if args.validate_gui:
            if not sampled_poses:
                raise RuntimeError("GUI validation requested, but sampling produced 0 poses.")
            sampler.validate_samples_in_gui(
                environment_usd_path=args.environment_usd,
                robot_usd_path=args.robot_usd,
                sampled_poses=sampled_poses,
                object_bbox=object_bbox,
                base_prim_path=args.base_prim_path,
                camera_prim_path=args.camera_prim_path,
                selected_sample_index=args.sample_index,
                robot_name=args.robot_name,
            )
    except Exception as exc:
        print(f"[ERROR] Goal sampler failed: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
