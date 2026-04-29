from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from isaac_sim.goal_generator.object_goal_sampler_utils import ObjectBBox3D


FOCUS_SELECTOR_ID = "forklift_near_shelf_min_world_x"
FOCUS_SELECTOR_PATH = str(Path(__file__).resolve())


def select_focus_object(
    candidates: Sequence[ObjectBBox3D],
) -> tuple[ObjectBBox3D | None, dict[str, Any]]:
    """Select the forklift with the smaller world-frame x center."""

    candidate_list = list(candidates)
    debug_candidates = [
        {
            "prim_path": candidate.prim_path,
            "center_xyz": candidate.center_xyz.tolist(),
            "min_xyz": candidate.min_xyz.tolist(),
            "max_xyz": candidate.max_xyz.tolist(),
        }
        for candidate in candidate_list
    ]
    if not candidate_list:
        return None, {
            "selector_id": FOCUS_SELECTOR_ID,
            "selector_module": __name__,
            "selector_path": FOCUS_SELECTOR_PATH,
            "selection_rule": "min_world_center_x",
            "candidate_count": 0,
            "candidates": [],
            "selected_prim_path": None,
            "selected_reason": "no_focus_candidates",
        }

    selected = min(
        candidate_list,
        key=lambda candidate: (float(candidate.center_xyz[0]), candidate.prim_path),
    )
    return selected, {
        "selector_id": FOCUS_SELECTOR_ID,
        "selector_module": __name__,
        "selector_path": FOCUS_SELECTOR_PATH,
        "selection_rule": "min_world_center_x",
        "candidate_count": len(candidate_list),
        "candidates": debug_candidates,
        "selected_prim_path": selected.prim_path,
        "selected_center_xyz": selected.center_xyz.tolist(),
        "selected_reason": "smallest_world_center_x",
    }
