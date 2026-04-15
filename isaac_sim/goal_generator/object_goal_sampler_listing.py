from __future__ import annotations

import re
from typing import Any

import numpy as np

from object_goal_sampler_utils import OBJECT_LIST_MODES


class ObjectListingMixin:
    @staticmethod
    def _active_child_count(prim: Any) -> int:
        return sum(1 for child in prim.GetChildren() if child and child.IsValid() and child.IsActive())

    @staticmethod
    def _relative_prim_depth(prim_path: str, root_prim_path: str | None) -> int:
        if not root_prim_path:
            return max(0, prim_path.count("/") - 1)
        if prim_path == root_prim_path:
            return 0
        root_tokens = [token for token in root_prim_path.split("/") if token]
        prim_tokens = [token for token in prim_path.split("/") if token]
        return max(0, len(prim_tokens) - len(root_tokens))

    @staticmethod
    def _score_object_list_candidate(
        candidate: dict[str, Any],
        *,
        mode: str,
    ) -> tuple[float, list[str], bool]:
        name_lower = candidate["name"].lower()
        path_lower = candidate["path"].lower()
        path_segments = {token for token in path_lower.split("/") if token}

        min_dim_m = float(candidate["min_dim_m"])
        max_dim_m = float(candidate["max_dim_m"])
        volume_m3 = float(candidate["volume_m3"])
        aspect_ratio = max_dim_m / max(min_dim_m, 1e-6)
        depth = int(candidate["depth"])
        direct_child_count = int(candidate["direct_child_count"])
        is_gprim = bool(candidate["is_gprim"])

        score = 0.0
        tags: list[str] = []

        hard_excluded = False
        if "looks" in path_segments or "materials" in path_segments:
            score -= 12.0
            tags.append("material_subtree")
            hard_excluded = True
        if "visuals" in path_segments:
            score -= 7.0
            tags.append("visual_subtree")
        if re.search(r"(^|[_/])label([_/]|$)", path_lower):
            score -= 8.0
            tags.append("label_like")
        if any(token in path_lower for token in ("decal", "helper", "preview")):
            score -= 6.0
            tags.append("helper_like")
        if re.search(r"(^|_)(fof|mesh)(_|$)", name_lower):
            score -= 4.0
            tags.append("mesh_named")

        if max_dim_m < 0.15:
            score -= 5.0
            tags.append("very_small")
        elif max_dim_m <= 8.0:
            score += 3.0
            tags.append("object_scale")
        elif max_dim_m <= 12.0:
            score += 0.5
            tags.append("large_object")
        else:
            score -= 3.0
            tags.append("scene_scale")

        if min_dim_m >= 0.05:
            score += 1.5
            tags.append("nonflat")
        elif min_dim_m < 0.02:
            score -= 4.0
            tags.append("thin")

        if volume_m3 < 5e-5:
            score -= 6.0
            tags.append("tiny_volume")
        elif volume_m3 < 0.01:
            score -= 2.0
            tags.append("small_volume")
        else:
            score += 1.0
            tags.append("substantial_volume")

        if aspect_ratio > 20.0:
            score -= 3.0
            tags.append("very_slender")
        elif aspect_ratio > 10.0:
            score -= 1.5
            tags.append("slender")
        elif aspect_ratio <= 4.0:
            score += 0.5
            tags.append("compact_shape")

        if is_gprim:
            score -= 0.5
            tags.append("leaf_geometry")
        else:
            score += 1.5
            tags.append("semantic_anchor")

        if 1 <= depth <= 5:
            score += 1.0
            tags.append("usable_depth")
        elif depth >= 8:
            score -= 1.5
            tags.append("deep_hierarchy")

        if direct_child_count == 0:
            score -= 0.5
            tags.append("leaf")
        elif direct_child_count <= 8:
            score += 0.5
            tags.append("compact_group")
        elif direct_child_count <= 20:
            score -= 1.5
            tags.append("busy_group")
        else:
            score -= 4.5
            tags.append("container_like")

        preferred_target_tokens = (
            "forklift",
            "pushcart",
            "cardbox",
            "box",
            "crate",
            "pallet",
            "klt",
            "bin",
            "barrel",
            "cabinet",
            "cart",
            "cone",
            "fusebox",
            "robot",
        )
        structural_tokens = (
            "rackshelf",
            "rackframe",
            "rack",
            "shelf",
            "frame",
            "wall",
            "floor",
            "ceiling",
            "warehouse",
            "pillar",
            "beam",
            "support",
        )

        if any(token in name_lower or token in path_lower for token in preferred_target_tokens):
            score += 2.5
            tags.append("target_like")
        if any(token in name_lower or token in path_lower for token in structural_tokens):
            if mode == "representative":
                score -= 2.5
                tags.append("structural_like")
            elif mode == "components":
                score -= 0.5
                tags.append("structural_like")

        if any(
            token in name_lower or token in path_lower
            for token in (
                "forklift",
                "klt",
                "bin",
                "box",
                "crate",
                "pallet",
                "cart",
                "barrel",
                "cabinet",
                "table",
                "chair",
                "cone",
                "robot",
            )
        ):
            score += 1.0
            tags.append("object_named")

        too_tiny = max_dim_m < 0.12 or volume_m3 < 2e-5
        too_thin = min_dim_m < 0.01 and max_dim_m < 0.5
        large_container = (direct_child_count > 20) and (max_dim_m > 4.0) and (not is_gprim)

        if mode == "raw":
            keep = True
        elif mode == "components":
            keep = (not hard_excluded) and (not (too_tiny and too_thin)) and score >= 0.0
        else:
            keep = (
                (not hard_excluded)
                and (not too_tiny)
                and (not too_thin)
                and (not large_container)
                and score >= 3.0
            )

        return score, tags, keep

    @staticmethod
    def _object_list_candidates_are_duplicate(
        candidate: dict[str, Any],
        kept_candidate: dict[str, Any],
    ) -> bool:
        candidate_path = candidate["path"]
        kept_path = kept_candidate["path"]
        if not (
            candidate_path.startswith(kept_path + "/")
            or kept_path.startswith(candidate_path + "/")
        ):
            return False

        center_distance = float(
            np.linalg.norm(candidate["center_xyz"] - kept_candidate["center_xyz"])
        )
        diagonal_tolerance = max(
            0.05,
            0.08 * max(float(candidate["diag_m"]), float(kept_candidate["diag_m"])),
        )
        size_delta = float(
            np.max(np.abs(candidate["size_xyz"] - kept_candidate["size_xyz"]))
        )
        size_tolerance = max(
            0.05,
            0.12 * max(float(candidate["max_dim_m"]), float(kept_candidate["max_dim_m"])),
        )
        return center_distance <= diagonal_tolerance and size_delta <= size_tolerance

    def list_object_prim_candidates(
        self,
        environment_usd_path: str,
        *,
        name_filter: str | None = None,
        mode: str = "representative",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        stage = self.load_environment(environment_usd_path)
        search_root_path = (
            self._environment_root_prim.GetPath().pathString
            if self._environment_root_prim is not None
            else None
        )
        if mode not in OBJECT_LIST_MODES:
            raise ValueError(
                f"Unsupported object-list mode '{mode}'. Expected one of {OBJECT_LIST_MODES}."
            )

        _, UsdGeom, _ = self._ensure_usd_imports()
        filter_lower = (name_filter or "").strip().lower()
        results: list[dict[str, Any]] = []

        for prim in stage.Traverse():
            if not prim or not prim.IsValid() or not prim.IsActive():
                continue
            prim_path = prim.GetPath().pathString
            if search_root_path and not (
                prim_path == search_root_path or prim_path.startswith(search_root_path + "/")
            ):
                continue

            prim_name = prim.GetName()
            haystack = f"{prim_name} {prim_path}".lower()
            if filter_lower and filter_lower not in haystack:
                continue
            if prim_name in {"Looks", "Materials"}:
                continue

            try:
                bbox = self._compute_world_bbox(stage, prim, self._environment_bbox_cache)
            except Exception:
                continue

            size_xyz = bbox.size_xyz
            if np.any(size_xyz <= 0.0):
                continue
            if np.max(size_xyz) > 20.0:
                continue

            candidate = {
                "name": prim_name,
                "path": prim_path,
                "type_name": str(prim.GetTypeName() or "Xform"),
                "size_xyz": size_xyz.astype(float),
                "center_xyz": bbox.center_xyz.astype(float),
                "min_xyz": bbox.min_xyz.astype(float),
                "max_xyz": bbox.max_xyz.astype(float),
                "volume_m3": float(np.prod(size_xyz)),
                "min_dim_m": float(np.min(size_xyz)),
                "max_dim_m": float(np.max(size_xyz)),
                "diag_m": float(np.linalg.norm(size_xyz)),
                "depth": self._relative_prim_depth(prim_path, search_root_path),
                "direct_child_count": self._active_child_count(prim),
                "is_gprim": bool(prim.IsA(UsdGeom.Gprim)),
            }
            score, tags, keep = self._score_object_list_candidate(candidate, mode=mode)
            if mode != "raw" and not keep:
                continue
            candidate["score"] = round(float(score), 3)
            candidate["tags"] = tags
            results.append(candidate)

        results.sort(
            key=lambda item: (
                -float(item["score"]),
                -float(item["volume_m3"]),
                int(item["depth"]),
                item["path"],
            )
        )

        if mode != "raw":
            deduped_results: list[dict[str, Any]] = []
            for candidate in results:
                if any(
                    self._object_list_candidates_are_duplicate(candidate, kept_candidate)
                    for kept_candidate in deduped_results
                ):
                    continue
                deduped_results.append(candidate)
            results = deduped_results

        formatted_results: list[dict[str, Any]] = []
        for candidate in results[: max(1, int(limit))]:
            formatted_results.append(
                {
                    "name": candidate["name"],
                    "path": candidate["path"],
                    "type_name": candidate["type_name"],
                    "score": candidate["score"],
                    "tags": candidate["tags"],
                    "size_xyz": candidate["size_xyz"].tolist(),
                    "center_xyz": candidate["center_xyz"].tolist(),
                }
            )

        return formatted_results
