#!/usr/bin/env python3
"""
Thin entrypoint for the Isaac Sim object-reaching goal sampler.

Implementation details now live in focused sibling modules:
- `object_goal_sampler_core.py`: sampling/runtime orchestration and CLI
- `object_goal_sampler_listing.py`: object-listing logic
- `object_goal_sampler_gui.py`: GUI validation and manual goal capture
- `object_goal_sampler_utils.py`: shared helpers, dataclasses, and map utilities

Example usage (from isaac sim root):
$ ./python.sh /home/yjiao/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/isaac_sim/scripts/object_goal_sampler.py \
--environment-usd omniverse://localhost/NVIDIA/Assets/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd \
--robot-usd omniverse://localhost/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd \
--occupancy-map /home/yjiao/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/ros2_ws/src/carters_nav2/maps/carter_warehouse_navigation.yaml \
--object-query Forklift \
--required-samples 8 \
--validate-gui \
--stage-occlusion-tail-relaxation 1.0 \
--min-visible-bbox-corners 0
"""

from __future__ import annotations

from object_goal_sampler_core import GoalSampler, ObjectReachingGoalSampler, main

__all__ = ["GoalSampler", "ObjectReachingGoalSampler", "main"]


if __name__ == "__main__":
    main()
