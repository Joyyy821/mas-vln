# AGENTS.md

## Project context
This repo is for multi-robot navigation / VLN rollout collection with Isaac Sim 5.1 on the host and ROS 2 Humble / Isaac ROS inside Docker.

Host paths:
- Isaac Sim install: ~/isaac-sim
- Main workspace (repo root location): ~/IsaacSim-ros_workspaces/humble_ws/src/mas-vln
- Isaac ROS dev workspace mounted at /workspaces/isaac_ros-dev and /workspaces/IsaacSim-ros_workspaces

Runtime architecture:
- Python script launches/sets up Isaac Sim scene from environment USD, robot USDs, and YAML rollout configs.
- ROS launch files bring up RViz, Nav2/MAPF/tracking, and data recording (robot trajectories).
- YAML config contains environment map info plus multiple initial/goal pose sets per natural-language instruction.
- Simulation-side offline rendering of RGBD camera views from recorded robot trajectories.

## Preferences
- Do not make large refactors without first explaining the plan.
- Prefer minimal, testable changes.
- Preserve ROS 2 Humble compatibility.
- Avoid assuming ROS Jazzy unless explicitly requested.
- When editing launch files, explain namespace/topic/action implications.
- For Isaac Sim code, be careful with USD prim paths and extension/version compatibility (Isaac Sim v5.1.0).

## Validation
Before saying a change is done, suggest or run relevant checks:
- python syntax/import check
- colcon build for affected ROS package
- ros2 launch dry-run reasoning if launch cannot be executed
- small unit-level script/test when possible

## Safety
Do not run destructive commands like rm -rf, git reset --hard, or clean workspace outputs unless explicitly approved.
