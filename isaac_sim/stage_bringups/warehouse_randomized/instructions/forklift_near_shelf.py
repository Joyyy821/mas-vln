from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from isaac_sim.goal_generator.object_goal_sampler_utils import ObjectBBox3D


X_DIFF_TH = 1.0
Y_CENTER = 4.0
FOCUS_SELECTOR_ID = "forklift_near_shelf_x_threshold_then_y_center"
FOCUS_SELECTOR_PATH = str(Path(__file__).resolve())


def select_focus_object(
    candidates: Sequence[ObjectBBox3D],
) -> tuple[ObjectBBox3D | None, dict[str, Any]]:
    """Select the forklift matching the near-shelf language cue."""

    candidate_list = list(candidates)
    debug_candidates = [
        {
            "prim_path": candidate.prim_path,
            "center_xyz": candidate.center_xyz.tolist(),
            "min_xyz": candidate.min_xyz.tolist(),
            "max_xyz": candidate.max_xyz.tolist(),
            "delta_y_from_target_center_m": abs(float(candidate.center_xyz[1]) - Y_CENTER),
        }
        for candidate in candidate_list
    ]
    if not candidate_list:
        return None, {
            "selector_id": FOCUS_SELECTOR_ID,
            "selector_module": __name__,
            "selector_path": FOCUS_SELECTOR_PATH,
            "selection_rule": "x_threshold_then_y_center",
            "x_diff_threshold_m": X_DIFF_TH,
            "target_y_center_m": Y_CENTER,
            "x_span_m": 0.0,
            "candidate_count": 0,
            "candidates": [],
            "selected_prim_path": None,
            "selected_reason": "no_focus_candidates",
        }

    center_x_values = [float(candidate.center_xyz[0]) for candidate in candidate_list]
    x_span = max(center_x_values) - min(center_x_values)
    if x_span > X_DIFF_TH:
        selected = min(
            candidate_list,
            key=lambda candidate: (float(candidate.center_xyz[0]), candidate.prim_path),
        )
        selected_reason = "largest_x_separation_smallest_world_center_x"
    else:
        selected = min(
            candidate_list,
            key=lambda candidate: (
                abs(float(candidate.center_xyz[1]) - Y_CENTER),
                float(candidate.center_xyz[0]),
                candidate.prim_path,
            ),
        )
        selected_reason = "x_centers_close_closest_to_target_y_center"

    return selected, {
        "selector_id": FOCUS_SELECTOR_ID,
        "selector_module": __name__,
        "selector_path": FOCUS_SELECTOR_PATH,
        "selection_rule": "x_threshold_then_y_center",
        "x_diff_threshold_m": X_DIFF_TH,
        "target_y_center_m": Y_CENTER,
        "x_span_m": x_span,
        "candidate_count": len(candidate_list),
        "candidates": debug_candidates,
        "selected_prim_path": selected.prim_path,
        "selected_center_xyz": selected.center_xyz.tolist(),
        "selected_delta_y_from_target_center_m": abs(float(selected.center_xyz[1]) - Y_CENTER),
        "selected_reason": selected_reason,
    }
