#!/usr/bin/env python3
"""
Thin entrypoint for the Isaac Sim object-reaching goal sampler.

Implementation details now live in focused sibling modules:
- `object_goal_sampler_core.py`: sampling/runtime orchestration and CLI
- `object_goal_sampler_listing.py`: object-listing logic
- `object_goal_sampler_gui.py`: GUI validation and manual goal capture
- `object_goal_sampler_utils.py`: shared helpers, dataclasses, and map utilities
"""

from __future__ import annotations

from object_goal_sampler_core import GoalSampler, ObjectReachingGoalSampler, main

__all__ = ["GoalSampler", "ObjectReachingGoalSampler", "main"]


if __name__ == "__main__":
    main()
