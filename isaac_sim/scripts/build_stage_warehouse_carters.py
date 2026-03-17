#!/usr/bin/env python3
"""
Isaac Sim 5.1 stage builder:
- references a warehouse environment USD
- references /Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd for each configured robot
- loads robot namespaces and spawn poses from a shared team config yaml
- (optional) saves the stage

Run standalone:
  ./python.sh /abs/path/to/build_stage_warehouse_carters.py

Env vars (recommended):
  export CARTER_TEAM_CONFIG_FILE="/abs/path/to/warehouse_team_config.yaml"
  export OUTPUT_USD="/home/you/.../warehouse_two_robots.usd"
"""

import importlib.util
import os
from typing import Tuple

STANDALONE = True

# ENV_USD = os.environ.get(
#     "ISAAC_ENV_USD",
#     "omniverse://localhost/NVIDIA/Assets/Isaac/Environments/Simple_Warehouse/warehouse.usd",
# )

# NOVA_CARTER_ROS_USD = os.environ.get(
#     "ISAAC_NOVA_CARTER_ROS_USD",
#     "omniverse://localhost/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd",
# )

# "relative" path for env and nova carter usd (to get assets root dynamically)
ENV_USD_REL = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
NOVA_CARTER_ROS_USD_REL = "/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MAPS_DIR = os.path.join(REPO_ROOT, "ros2_ws", "src", "carters_nav2", "maps")
TEAM_CONFIG_UTILS_PATH = os.path.join(
    REPO_ROOT,
    "ros2_ws",
    "src",
    "carters_nav2",
    "launch",
    "team_config_utils.py",
)
DEFAULT_TEAM_CONFIG_FILE = os.path.join(
    REPO_ROOT,
    "ros2_ws",
    "src",
    "carters_nav2",
    "config",
    "warehouse",
    "warehouse_team_config.yaml",
)
TEAM_CONFIG_FILE = os.environ.get("CARTER_TEAM_CONFIG_FILE", DEFAULT_TEAM_CONFIG_FILE)
OUTPUT_USD = os.environ.get("OUTPUT_USD", "")  # optional


def _load_team_config_utils():
    spec = importlib.util.spec_from_file_location("team_config_utils", TEAM_CONFIG_UTILS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load team config utilities from {TEAM_CONFIG_UTILS_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


team_config_utils = _load_team_config_utils()


##########################################
###### Sim, stage, references utils ######
##########################################

def _maybe_start_sim_app():
    if not STANDALONE:
        return None
    
    # 1. Import and start the SimulationApp FIRST.
    # CRITICAL: Do not import other omni.* or isaacsim.* modules before this!
    # from omni.isaac.kit import SimulationApp
    from isaacsim.simulation_app import SimulationApp
    # sim_app = SimulationApp({"headless": False})
    sim_app = SimulationApp({"headless": True})

    # 2. Import the extension utility
    from omni.isaac.core.utils.extensions import enable_extension

    # 3. Enable the ROS 2 bridge extension
    # Note: In Isaac Sim 5.1, the extension is named "isaacsim.ros2.bridge". 
    # (If you are adapting older 4.x code, it used to be "omni.isaac.ros2_bridge")
    enable_extension("isaacsim.ros2.bridge")

    # 4. Force a simulation update to ensure the extension fully loads into memory
    sim_app.update()

    print("ROS 2 Bridge Enabled Successfully!")

    return sim_app


def _new_stage():
    import omni.usd
    omni.usd.get_context().new_stage()


def _set_stage_units(meters_per_unit: float = 1.0):
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)


def _add_reference(usd_path: str, prim_path: str):
    from omni.isaac.core.utils.stage import add_reference_to_stage
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)


def _define_xform(path: str):
    from omni.isaac.core.utils.prims import define_prim
    define_prim(path, "Xform")


def _set_xform_pose(prim_path: str, position_xyz: Tuple[float, float, float], yaw_deg: float = 0.0):
    import omni.usd
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    x, y, z = position_xyz
    rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(yaw_deg))

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()

    # SetTranslate() clears rotation on Gf.Matrix4d, so compose both in one call.
    mat = Gf.Matrix4d(1.0)
    mat.SetTransform(rot, Gf.Vec3d(x, y, z))
    op.Set(mat)


##########################################
###### Namespace supports           ######
##########################################


def _set_isaac_namespace(prim_path: str, namespace: str):
    import omni.usd
    from pxr import Sdf

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    # Custom attribute used by Isaac Sim ROS2 auto-namespace feature
    attr = prim.GetAttribute("isaac:namespace")
    if not (attr and attr.IsValid()):
        attr = prim.CreateAttribute("isaac:namespace", Sdf.ValueTypeNames.String)
    attr.Set(namespace)


def _is_ros2_omnigraph_node(prim) -> bool:
    # Heuristic: we only touch prims that look like OmniGraph nodes AND have ROS2-ish attributes
    if not prim or not prim.IsValid():
        return False
    # Quick filter: any attribute starting with "inputs:" and containing common ROS2 fields
    names = [a.GetName() for a in prim.GetAttributes()]
    return any(n in ("inputs:nodeNamespace", "inputs:topicName", "inputs:frameId",
                     "inputs:odomFrameId", "inputs:baseFrameId", "inputs:childFrameId",
                     "inputs:parentFrameId") for n in names)

def _prefix_frame(frame: str, ns: str) -> str:
    if not frame:
        return frame
    # Keep global frames as-is (typical shared map setup)
    if frame in ("map", "world"):
        return frame
    # Already prefixed
    if frame.startswith(ns + "/"):
        return frame
    return f"{ns}/{frame}"

def _fix_ros2_graph_under(root_prim_path: str, namespace: str, prefix_frames: bool = False):
    """
    Robust multi-robot patch:
      - inputs:nodeNamespace <- namespace
      - inputs:topicName: strip leading '/' so namespace can apply
        (except /clock, which must remain global for ROS time)
      - frame ids: optionally prefix with 'namespace/' when a stack requires unique TF ids
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Root prim not found: {root_prim_path}")

    # NOTE: stage.Traverse() is fine; we filter by prefix
    touched = {"nodeNamespace": 0, "topicName": 0, "frames": 0}

    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if not (p == root_prim_path or p.startswith(root_prim_path + "/")):
            continue

        # patch for camera topics
        if prim.GetName() == "camera_namespace":
            vattr = prim.GetAttribute("inputs:value")
            if vattr and vattr.IsValid():
                v = vattr.Get()
                if isinstance(v, str) and "camera" in v.lower():
                    base = v.strip("/")
                    vattr.Set(f"/{namespace}/{base}")
                    touched["cameraNamespace"] = touched.get("cameraNamespace", 0) + 1

        if not _is_ros2_omnigraph_node(prim):
            continue

        # 1) node namespace
        ns_attr = prim.GetAttribute("inputs:nodeNamespace")
        if ns_attr and ns_attr.IsValid():
            try:
                ns_attr.Set(namespace)
                touched["nodeNamespace"] += 1
            except Exception:
                pass

        # 2) topicName: remove absolute prefix to allow nodeNamespace to prepend
        topic_attr = prim.GetAttribute("inputs:topicName")
        if topic_attr and topic_attr.IsValid():
            try:
                t = topic_attr.Get()
                if isinstance(t, str) and t.startswith("/"):
                    # Keep /clock global so all Nav2 nodes using use_sim_time
                    # receive ROS time from Isaac Sim.
                    if t != "/clock":
                        topic_attr.Set(t.lstrip("/"))
                        touched["topicName"] += 1
            except Exception:
                pass        

        # 3) frame IDs: prevent TF collisions
        if prefix_frames:
            for frame_key in ("inputs:frameId", "inputs:odomFrameId", "inputs:baseFrameId",
                              "inputs:childFrameId", "inputs:parentFrameId"):
                fattr = prim.GetAttribute(frame_key)
                if not (fattr and fattr.IsValid()):
                    continue
                try:
                    f = fattr.Get()
                    if isinstance(f, str) and f:
                        f2 = _prefix_frame(f, namespace)
                        if f2 != f:
                            fattr.Set(f2)
                            touched["frames"] += 1
                except Exception:
                    pass

    print(f"[OK] Patched ROS2 graph under {root_prim_path}: {touched}")


def _ensure_global_ros2_clock_graph(clock_topic: str = "/clock"):
    """
    Create a global ROS2 clock publisher graph:
      OnPlaybackTick -> ROS2PublishClock(timeStamp <- IsaacReadSimulationTime)
    """
    import omni.graph.core as og

    graph_path = "/World/ROS2ClockGraph"

    try:
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishClock.inputs:topicName", clock_topic),
                ],
            },
        )
        print(f"[OK] Added global ROS2 clock publisher graph at {graph_path} on topic '{clock_topic}'")
    except Exception as e:
        print(f"[WARN] Failed to create ROS2 clock graph at {graph_path}: {e}")


# Old robot namspacing approach: only set nodeNamespace, 
# but topic names and frame ids may still collide if they are absolute (start with '/').
def _set_ros2_node_namespace_under(root_prim_path: str, namespace: str):
    """
    Traverse all prims under root_prim_path and set any attribute named 'inputs:nodeNamespace'
    to `namespace`. This catches ROS2 bridge nodes like ROS2SubscribeTwist, ROS2PublishTF, etc.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Root prim not found: {root_prim_path}")

    count = 0
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if not p.startswith(root_prim_path + "/"):
            continue

        attr = prim.GetAttribute("inputs:nodeNamespace")
        if attr and attr.IsValid():
            try:
                attr.Set(namespace)
                count += 1
            except Exception:
                pass

    print(f"[OK] Set inputs:nodeNamespace='{namespace}' on {count} prims under {root_prim_path}")


##########################################
######          Debug utils         ######
##########################################


def _debug_instanceability(prim_path: str):
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    print(prim_path,
          "instanceable=", prim.IsInstanceable(),
          "instance_proxy=", prim.IsInstanceProxy())


def _find_ros2_camera_publishers():
    """
    Print prim paths that look like ROS2 image/camera publishers, and show their topic-related attributes.
    This helps locate whether the publishers live under a global ActionGraph vs under each robot.
    """
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    hits = []
    for prim in stage.Traverse():
        # Heuristic: look for prims with topicName and something camera-ish
        attrs = {a.GetName(): a for a in prim.GetAttributes()}
        if "inputs:topicName" in attrs:
            t = attrs["inputs:topicName"].Get()
            if isinstance(t, str) and "camera" in t:
                hits.append((prim.GetPath().pathString, t))

        # Another heuristic: if prim name contains 'Camera' and has ROS-ish attrs
        name = prim.GetName().lower()
        if ("camera" in name or "image" in name) and any(k.startswith("inputs:") for k in attrs.keys()):
            # keep it, we’ll print a smaller subset
            hits.append((prim.GetPath().pathString, attrs.get("inputs:topicName").Get() if "inputs:topicName" in attrs else None))

    # de-dup
    uniq = []
    seen = set()
    for p, t in hits:
        if p in seen:
            continue
        seen.add(p)
        uniq.append((p, t))

    print("\n=== Candidate ROS2 camera publisher prims ===")
    for p, t in uniq[:200]:
        print("  ", p, "topic=", t)
    print("=== End ===\n")


def _print_prim_inputs(prim_path: str):
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"[WARN] Prim not found: {prim_path}")
        return

    print(f"\n=== Inputs for {prim_path} ===")
    for a in prim.GetAttributes():
        n = a.GetName()
        if n.startswith("inputs:"):
            try:
                v = a.Get()
            except Exception:
                v = "<unreadable>"
            print(f"  {n} = {v}")
    print("=== End ===\n")


##########################################
#### Stage builder and main function #####
##########################################


def build_stage():
    from isaacsim.storage.native import get_assets_root_path
    assets_root_path = get_assets_root_path()
    team_config = team_config_utils.load_team_config(TEAM_CONFIG_FILE, maps_dir=MAPS_DIR)

    _new_stage()
    _set_stage_units(1.0)

    # Create prims
    _define_xform("/World")
    _define_xform("/World/Env")
    _define_xform("/World/Robots")

    # 1) Environment
    env_prim = "/World/Env/Warehouse"
    _add_reference(f"{assets_root_path}{ENV_USD_REL}", env_prim)

    # 2) Robots from the shared team config.
    carter_usd = f"{assets_root_path}{NOVA_CARTER_ROS_USD_REL}"
    for index, robot in enumerate(team_config["robots"], start=1):
        robot_prim = f"/World/Robots/NovaCarter_{index}"
        initial_pose = team_config_utils.pose_array_to_pose_dict(robot["initial_pose"])

        _add_reference(carter_usd, robot_prim)
        _set_isaac_namespace(robot_prim, robot["name"])
        _set_xform_pose(
            robot_prim,
            (
                initial_pose["x"],
                initial_pose["y"],
                initial_pose["z"],
            ),
            yaw_deg=initial_pose["yaw"] * 180.0 / 3.141592653589793,
        )

        # Match Isaac Sim's multi-robot Nav2 example: namespace ROS topics per robot,
        # but keep TF frame ids as the asset defaults (map / odom / base_link / sensor frames).
        _fix_ros2_graph_under(robot_prim, robot["name"], prefix_frames=False)

    _ensure_global_ros2_clock_graph("/clock")

    # 4) Save (optional)
    if OUTPUT_USD:
        import omni.usd
        omni.usd.get_context().save_as_stage(OUTPUT_USD)
        print(f"[OK] Saved stage to: {OUTPUT_USD}")

    print(
        "[OK] Stage built: warehouse + "
        f"{team_config['agent_num']} Nova_Carter_ROS robots from {TEAM_CONFIG_FILE}"
    )

    # debug camera topic collisions
    # _find_ros2_camera_publishers()
    # _print_prim_inputs("/World/Robots/NovaCarter_1/front_hawk/camera_namespace")
    #


def main():
    sim_app = _maybe_start_sim_app()
    # warm-up updates
    for _ in range(10):
        sim_app.update()

    try:
        build_stage()

        # one or two updates after stage creation helps graphs/materialization settle
        for _ in range(10):
            sim_app.update()

        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        print("[OK] Timeline started from Python")

        while sim_app.is_running():
            sim_app.update()
        # # In standalone, keep UI alive so you can press Play and inspect the graph/topics.
        # if sim_app is not None:
        #     import time
        #     while sim_app.is_running():
        #         sim_app.update()
        #         # time.sleep(0.1)
    finally:
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    # usage (tested on ubuntu 22.04 with ROS 2 Humble and Isaac Sim 5.1):
    # $ cd isaac-sim # Note: your isaac sim installation dir, not this script's dir
    # $ conda deactivate  # (optional) ensure base conda env is not active, to avoid conflicts with ROS 2 Humble's Python 3.10
    # $ source /opt/ros/humble/setup.bash  # source ROS 2 Humble setup to enable ROS2 bridge
    # $ ./python.sh /abs/path/to/mas-vln/isaac_sim/scripts/build_stage_warehouse_carters.py
    main()
