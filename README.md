# Data Collection Pipeline for MAS VLN

## Customized team config bringup

- `build_stage_warehouse_carters.py` accepts:
- `--team-config-file`
- `--output-usd`
- CLI args take precedence over the environment-variable fallback.
- The team-config path is normalized and validated before stage build starts.

**How to use it**
From Isaac Sim install dir:

```bash
./python.sh /home/yjiao/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/isaac_sim/stage_bringups/build_stage_warehouse_carters.py \
  --team-config-file /home/yjiao/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/ros2_ws/src/carters_nav2/config/warehouse/warehouse_team_config_2.yaml
```

Optional save path:

```bash
./python.sh /home/yjiao/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/isaac_sim/stage_bringups/build_stage_warehouse_carters.py \
  --team-config-file /home/yjiao/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/ros2_ws/src/carters_nav2/config/warehouse/warehouse_team_config_2.yaml \
  --output-usd /home/admin/stages/warehouse_team_config_2.usd
```

Use the same config file on the ROS side (inside docker):

```bash
ros2 launch carters_nav2 warehouse_team_lightweight.launch.py \
  team_config_file:=/workspaces/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/ros2_ws/src/carters_nav2/config/warehouse/warehouse_team_config_2.yaml
```

```bash
ros2 launch carters_goal isaac_ros_mapf.launch.py \
  team_config_file:=/workspaces/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/ros2_ws/src/carters_nav2/config/warehouse/warehouse_team_config_2.yaml \
  run_plan_executor:=true \
  execution_backend:=timed_tracker
```
