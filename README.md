# Data Collection Pipeline for MAS VLN

## Installation
Tested version: Isaac Sim 5.1 + ROS 2 humble on Ubuntu 22.04

Clone the repo under your Isaac ROS docker mounted space, add MAPF support [here](https://github.com/Joyyy821/mapf_ros), and build the ROS packages.
   
```bash 
git clone https://github.com/Joyyy821/mas-vln.git
cd mas-vln/ros2_ws/src
git clone https://github.com/Joyyy821/mapf_ros.git
cd ..
colcon build --symlink-install
```
Make sure to `source install/setup.bash` before launching.

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
  --team-config-file /home/yjiao/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/data_configs/warehouse/warehouse_forklift.yaml
```

Optional save path:

```bash
./python.sh /home/yjiao/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/isaac_sim/stage_bringups/build_stage_warehouse_carters.py \
  --team-config-file /home/yjiao/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/data_configs/warehouse/warehouse_forklift.yaml \
  --output-usd /home/admin/stages/warehouse_team_config_2.usd
```

Use the same config file on the ROS side (inside docker):

To launch RViz for visualization:
```bash
ros2 launch carters_nav2 warehouse_team_lightweight.launch.py \
  team_config_file:=/workspaces/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/data_configs/warehouse/warehouse_forklift.yaml
```

To launch a single navigation run:
```bash
ros2 launch carters_goal isaac_ros_mapf.launch.py \
  team_config_file:=/workspaces/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/data_configs/warehouse/warehouse_forklift.yaml \
  run_plan_executor:=true \
  execution_backend:=timed_tracker \
  record_velocity:=true \
  record_frequency_hz:=20.0
```

To launch and collect a set of rollouts:
```bash
ros2 launch carters_goal isaac_ros_mapf_rollouts.launch.py \
  team_config_file:=/workspaces/IsaacSim-ros_workspaces/humble_ws/src/mas-vln/data_configs/warehouse/warehouse_forklift.yaml \
  execution_backend:=timed_tracker \
  skip_existed_rollout:=true
```
