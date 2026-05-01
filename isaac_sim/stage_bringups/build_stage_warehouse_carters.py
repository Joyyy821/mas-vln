#!/usr/bin/env python3
"""
Isaac Sim 5.1 stage builder:
- references a warehouse environment USD
- references the robot USD model configured for each rollout robot
- loads robot namespaces and spawn poses from a shared team config yaml
- runs ROS 2 Humble helper processes with system Python for rollout reset and
  heterogeneous robot cmd_vel/odom/tf topics
- (optional) saves the stage

Run standalone:
  ./python.sh /abs/path/to/build_stage_warehouse_carters.py \
    --team-config-file /abs/path/to/warehouse_forklift.yaml

CLI args (preferred):
  --team-config-file /abs/path/to/warehouse_forklift.yaml
  --scene-usd /abs/path/to/randomized_warehouse_scene.usd
  --output-usd /home/you/.../warehouse_two_robots.usd

Env vars (fallback):
  export CARTER_TEAM_CONFIG_FILE="/abs/path/to/warehouse_forklift.yaml"
  export CARTER_SCENE_USD="/abs/path/to/randomized_warehouse_scene.usd"
  export OUTPUT_USD="/home/you/.../warehouse_two_robots.usd"
  export CARTER_ROLLOUT_CONTROL_TOPIC="/carters_goal/rollout_control"
  export CARTER_ROLLOUT_RESET_DONE_TOPIC="/carters_goal/rollout_reset_done"
"""

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import time
from typing import Any, Tuple

import numpy as np

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
NOVA_CARTER_RENDER_USD_REL = "/Isaac/Samples/AnimRobot/nova_carter.usd"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
ROLLOUT_HELPER_PATH = os.path.join(SCRIPT_DIR, "isaac_rollout_reset_helper.py")
SYSTEM_ROS_ROBOT_HELPER_PATH = os.path.join(
    SCRIPT_DIR,
    "isaac_system_ros_robot_bridge_helper.py",
)
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
    "warehouse_forklift.yaml",
)
ENV_TEAM_CONFIG_FILE = os.environ.get("CARTER_TEAM_CONFIG_FILE", DEFAULT_TEAM_CONFIG_FILE)
ENV_SCENE_USD = os.environ.get("CARTER_SCENE_USD", "")
ENV_OUTPUT_USD = os.environ.get("OUTPUT_USD", "")  # optional
ENV_ROLLOUT_CONTROL_TOPIC = os.environ.get(
    "CARTER_ROLLOUT_CONTROL_TOPIC",
    "/carters_goal/rollout_control",
)
ENV_ROLLOUT_RESET_DONE_TOPIC = os.environ.get(
    "CARTER_ROLLOUT_RESET_DONE_TOPIC",
    "/carters_goal/rollout_reset_done",
)
ENV_ROLLOUT_ID = os.environ.get("CARTER_ROLLOUT_ID", "")
ENV_GUI = os.environ.get("CARTER_GUI", "").strip().lower() in {"1", "true", "yes", "on"}
ENV_CMD_ANGULAR_SIGN = float(os.environ.get("CARTER_CMD_ANGULAR_SIGN", "1.0"))


def _load_team_config_utils():
    spec = importlib.util.spec_from_file_location("team_config_utils", TEAM_CONFIG_UTILS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load team config utilities from {TEAM_CONFIG_UTILS_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


team_config_utils = _load_team_config_utils()


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the Isaac Sim warehouse stage using a shared Carter team configuration. "
            "CLI arguments override the environment-variable defaults."
        )
    )
    parser.add_argument(
        "--team-config-file",
        default=ENV_TEAM_CONFIG_FILE,
        help=(
            "Full path to the shared team configuration YAML. "
            "Defaults to CARTER_TEAM_CONFIG_FILE or the repo's warehouse_forklift.yaml."
        ),
    )
    parser.add_argument(
        "--output-usd",
        default=ENV_OUTPUT_USD,
        help=(
            "Optional output USD path. Defaults to OUTPUT_USD when set."
        ),
    )
    parser.add_argument(
        "--scene-usd",
        default=ENV_SCENE_USD,
        help=(
            "Optional scene USD to open before spawning Carter robots. "
            "Defaults to CARTER_SCENE_USD when set; otherwise references the built-in warehouse."
        ),
    )
    parser.add_argument(
        "--rollout-id",
        default=ENV_ROLLOUT_ID,
        help=(
            "Optional rollout id to load from a multi-rollout team_config.yaml. "
            "Defaults to CARTER_ROLLOUT_ID when set; otherwise uses the first rollout."
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=ENV_GUI,
        help="Run Isaac Sim with the GUI visible. Defaults to headless unless CARTER_GUI=1 is set.",
    )
    parser.add_argument(
        "--rollout-control-topic",
        default=ENV_ROLLOUT_CONTROL_TOPIC,
        help="PoseArray topic used by ROS to request rollout resets inside Isaac Sim.",
    )
    parser.add_argument(
        "--rollout-reset-done-topic",
        default=ENV_ROLLOUT_RESET_DONE_TOPIC,
        help="Int32 topic published by Isaac Sim after a rollout reset completes.",
    )
    args, unknown = parser.parse_known_args()

    team_config_file = os.path.abspath(os.path.expanduser(args.team_config_file))
    if not os.path.exists(team_config_file):
        raise FileNotFoundError(f"Team config file not found: {team_config_file}")

    output_usd = os.path.abspath(os.path.expanduser(args.output_usd)) if args.output_usd else ""
    scene_usd = os.path.abspath(os.path.expanduser(args.scene_usd)) if args.scene_usd else ""
    if scene_usd and not os.path.exists(scene_usd):
        raise FileNotFoundError(f"Scene USD not found: {scene_usd}")

    if unknown:
        print(f"[INFO] Ignoring unknown CLI args passed through Isaac Sim: {unknown}")

    rollout_control_topic = str(args.rollout_control_topic).strip()
    rollout_reset_done_topic = str(args.rollout_reset_done_topic).strip()
    rollout_id_text = str(args.rollout_id).strip()
    rollout_id = int(rollout_id_text) if rollout_id_text else None
    headless = not bool(args.gui)

    return (
        team_config_file,
        output_usd,
        scene_usd,
        rollout_control_topic,
        rollout_reset_done_topic,
        rollout_id,
        headless,
    )


##########################################
###### Sim, stage, references utils ######
##########################################

def _maybe_start_sim_app(enable_ros2_bridge: bool = True, *, headless: bool = True):
    if not STANDALONE:
        return None
    
    # 1. Import and start the SimulationApp FIRST.
    # CRITICAL: Do not import other omni.* or isaacsim.* modules before this!
    # from omni.isaac.kit import SimulationApp
    from isaacsim.simulation_app import SimulationApp
    sim_app = SimulationApp({"headless": bool(headless)})

    if enable_ros2_bridge:
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
    try:
        from isaacsim.core.utils.stage import create_new_stage
    except Exception:
        import omni.usd

        omni.usd.get_context().new_stage()
    else:
        create_new_stage()


def _set_stage_units(meters_per_unit: float = 1.0):
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)


def _add_reference(usd_path: str, prim_path: str):
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage
    except Exception:
        from omni.isaac.core.utils.stage import add_reference_to_stage

    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)


def _define_xform(path: str):
    try:
        from isaacsim.core.utils.prims import define_prim
    except Exception:
        from omni.isaac.core.utils.prims import define_prim

    define_prim(path, "Xform")


def _ensure_xform_path(path: str):
    if not path or path == "/":
        return
    import omni.usd

    current = ""
    stage = omni.usd.get_context().get_stage()
    for token in [token for token in path.split("/") if token]:
        current += f"/{token}"
        prim = stage.GetPrimAtPath(current)
        if prim and prim.IsValid():
            continue
        _define_xform(current)


def _open_stage(stage_usd: str):
    import omni.usd

    context = omni.usd.get_context()
    opened = context.open_stage(stage_usd)
    if opened is False:
        raise RuntimeError(f"Failed to open scene USD: {stage_usd}")


def _clear_prim_children(prim_path: str):
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    for child in list(prim.GetChildren()):
        stage.RemovePrim(child.GetPath())


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


def _disable_lidar_debug_visualization(root_prim_path: str) -> None:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Root prim not found: {root_prim_path}")

    lidar_path_tokens = ("lidar", "laser", "scan")
    debug_visual_tokens = ("debug", "visual", "points", "pointcloud", "rays", "lines")
    disabled_attrs = 0
    hidden_prims = 0

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not (path == root_prim_path or path.startswith(root_prim_path + "/")):
            continue

        lower_path = path.lower()
        if not any(token in lower_path for token in lidar_path_tokens):
            continue

        for attr_name in (
            "inputs:showDebugView",
            "drawPoints",
            "drawLines",
            "inputs:drawPoints",
            "inputs:drawLines",
        ):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                try:
                    attr.Set(False)
                    disabled_attrs += 1
                except Exception:
                    pass

        if any(token in lower_path for token in debug_visual_tokens):
            try:
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
                    hidden_prims += 1
            except Exception:
                pass

    print(
        f"[OK] Disabled lidar debug visualization under {root_prim_path}: "
        f"attrs={disabled_attrs}, hidden_prims={hidden_prims}"
    )


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


class IsaacRolloutResetBridge:
    def __init__(
        self,
        sim_app,
        robot_prim_paths: list[str],
        rollout_control_topic: str,
        rollout_reset_done_topic: str,
    ) -> None:
        self._sim_app = sim_app
        self._robot_prim_paths = robot_prim_paths
        self._rollout_control_topic = rollout_control_topic
        self._rollout_reset_done_topic = rollout_reset_done_topic
        self._pending_reset: tuple[int, list[float]] | None = None
        self._last_applied_rollout_id: int | None = None
        self._helper_process: subprocess.Popen | None = None
        self._client_socket: socket.socket | None = None
        self._recv_buffer = ""
        self._helper_failure_logged = False

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(("127.0.0.1", 0))
        self._server_socket.listen(1)
        self._server_socket.setblocking(False)
        self._server_host, self._server_port = self._server_socket.getsockname()

        self._start_helper_process()
        print(
            "[OK] Isaac rollout reset bridge listening on "
            f"{self._server_host}:{self._server_port} for ROS topic {rollout_control_topic}"
        )

    def _start_helper_process(self) -> None:
        if not os.path.exists(ROLLOUT_HELPER_PATH):
            raise FileNotFoundError(f"Rollout reset helper not found: {ROLLOUT_HELPER_PATH}")

        helper_command = (
            "source /opt/ros/humble/setup.bash && "
            f"exec /usr/bin/python3 {shlex.quote(ROLLOUT_HELPER_PATH)} "
            f"--host {shlex.quote(self._server_host)} "
            f"--port {self._server_port} "
            f"--rollout-control-topic {shlex.quote(self._rollout_control_topic)} "
            f"--rollout-reset-done-topic {shlex.quote(self._rollout_reset_done_topic)}"
        )
        helper_env = self._build_helper_env()
        self._helper_process = subprocess.Popen(
            ["/bin/bash", "-lc", helper_command],
            cwd=REPO_ROOT,
            env=helper_env,
        )

    def _build_helper_env(self) -> dict[str, str]:
        helper_env: dict[str, str] = {}

        for key in (
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "TERM",
            "LANG",
            "LC_ALL",
            "DISPLAY",
            "XAUTHORITY",
            "ROS_DOMAIN_ID",
            "ROS_LOCALHOST_ONLY",
            "RMW_IMPLEMENTATION",
            "FASTRTPS_DEFAULT_PROFILES_FILE",
            "CYCLONEDDS_URI",
            "ROS_AUTOMATIC_DISCOVERY_RANGE",
        ):
            value = os.environ.get(key)
            if value:
                helper_env[key] = value

        helper_env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        helper_env["PYTHONNOUSERSITE"] = "1"
        return helper_env

    def spin_once(self) -> None:
        self._check_helper_process()
        self._accept_client_if_needed()
        self._read_client_messages()

    def _check_helper_process(self) -> None:
        if self._helper_process is None or self._helper_failure_logged:
            return

        return_code = self._helper_process.poll()
        if return_code is None:
            return

        self._helper_failure_logged = True
        print(
            "[WARN] Isaac rollout reset helper exited early with code "
            f"{return_code}. Rollout resets will not work until that helper starts correctly."
        )

    def process_pending_reset(self) -> int | None:
        if self._pending_reset is None:
            return None

        rollout_id, flat_pose_array = self._pending_reset
        self._pending_reset = None

        if rollout_id == self._last_applied_rollout_id:
            self._publish_ack(rollout_id)
            return rollout_id

        self._apply_reset(rollout_id, flat_pose_array)
        self._publish_ack(rollout_id)
        return rollout_id

    def _apply_reset(self, rollout_id: int, flat_pose_array: list[float]) -> None:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        was_playing = timeline.is_playing()
        if was_playing:
            pause_timeline = getattr(timeline, "pause", None)
            if callable(pause_timeline):
                pause_timeline()
            else:
                print(
                    "[WARN] omni.timeline interface does not expose pause(); "
                    "falling back to stop(), which may rewind /clock during rollout resets."
                )
                timeline.stop()
            for _ in range(5):
                self._sim_app.update()

        for idx, robot_prim in enumerate(self._robot_prim_paths):
            offset = idx * 7
            pose_dict = team_config_utils.pose_array_to_pose_dict(
                flat_pose_array[offset : offset + 7]
            )
            _set_xform_pose(
                robot_prim,
                (pose_dict["x"], pose_dict["y"], pose_dict["z"]),
                yaw_deg=pose_dict["yaw"] * 180.0 / 3.141592653589793,
            )

        for _ in range(5):
            self._sim_app.update()

        if was_playing:
            timeline.play()
            for _ in range(10):
                self._sim_app.update()

        self._last_applied_rollout_id = rollout_id
        print(f"[OK] Applied Isaac Sim rollout reset for rollout {rollout_id}.")

    def _publish_ack(self, rollout_id: int) -> None:
        self._send_message(
            {
                "type": "reset_done",
                "rollout_id": int(rollout_id),
            }
        )

    def _accept_client_if_needed(self) -> None:
        if self._client_socket is not None:
            return

        try:
            client_socket, _ = self._server_socket.accept()
        except BlockingIOError:
            return

        client_socket.setblocking(False)
        self._client_socket = client_socket
        print("[OK] Rollout reset helper connected to Isaac Sim.")

    def _read_client_messages(self) -> None:
        if self._client_socket is None:
            return

        try:
            chunk = self._client_socket.recv(65536)
        except BlockingIOError:
            return
        except OSError:
            self._close_client_socket()
            return

        if not chunk:
            self._close_client_socket()
            return

        self._recv_buffer += chunk.decode("utf-8")
        while "\n" in self._recv_buffer:
            line, self._recv_buffer = self._recv_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self._handle_client_message(line)

    def _handle_client_message(self, line: str) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Failed to decode rollout reset message '{line}': {exc}")
            return

        if message.get("type") != "reset":
            return

        rollout_id = int(message.get("rollout_id", -1))
        flat_pose_array = [float(value) for value in message.get("poses", [])]
        if rollout_id <= 0:
            return

        expected_values = len(self._robot_prim_paths) * 7
        if len(flat_pose_array) != expected_values:
            print(
                f"[WARN] Ignoring rollout {rollout_id} reset request because it contains "
                f"{len(flat_pose_array)} pose values and {expected_values} were expected."
            )
            return

        self._pending_reset = (rollout_id, flat_pose_array)

    def _send_message(self, message: dict) -> None:
        if self._client_socket is None:
            return

        payload = (json.dumps(message) + "\n").encode("utf-8")
        try:
            self._client_socket.sendall(payload)
        except BlockingIOError:
            return
        except OSError:
            self._close_client_socket()

    def _close_client_socket(self) -> None:
        if self._client_socket is None:
            return
        try:
            self._client_socket.close()
        finally:
            self._client_socket = None
            self._recv_buffer = ""
            print("[WARN] Rollout reset helper disconnected from Isaac Sim.")

    def shutdown(self) -> None:
        self._close_client_socket()
        try:
            self._server_socket.close()
        except Exception:
            pass

        if self._helper_process is not None:
            if self._helper_process.poll() is None:
                self._helper_process.terminate()
                try:
                    self._helper_process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._helper_process.kill()


@dataclass
class _RobotStateCache:
    sim_time_sec: float | None = None
    world_position_xy: np.ndarray | None = None
    yaw_rad: float | None = None


@dataclass
class _OdomOrigin:
    world_position_xy: np.ndarray
    world_yaw_rad: float


def _wheel_action_arrays(controller: Any, *, linear_x: float, angular_z: float) -> tuple[np.ndarray, np.ndarray]:
    right_velocity = (
        (2.0 * float(linear_x)) + (float(angular_z) * controller.wheel_distance_m)
    ) / (2.0 * controller.wheel_radius_m)
    left_velocity = (
        (2.0 * float(linear_x)) - (float(angular_z) * controller.wheel_distance_m)
    ) / (2.0 * controller.wheel_radius_m)

    joint_indices = np.concatenate(
        [
            np.asarray(controller.left_wheel_joint_indices, dtype=np.int32),
            np.asarray(controller.right_wheel_joint_indices, dtype=np.int32),
        ]
    )
    joint_velocities = np.concatenate(
        [
            np.full(len(controller.left_wheel_joint_indices), left_velocity, dtype=np.float32),
            np.full(len(controller.right_wheel_joint_indices), right_velocity, dtype=np.float32),
        ]
    )
    return joint_indices, joint_velocities


class IsaacSystemRosRobotBridge:
    """Bridge ROS Humble cmd_vel/odom/tf through a system-Python helper process."""

    def __init__(
        self,
        sim_app,
        robot_controllers: list[Any],
        *,
        cmd_angular_sign: float = ENV_CMD_ANGULAR_SIGN,
    ) -> None:
        if not robot_controllers:
            raise ValueError("At least one robot controller is required for ROS topic bridging.")

        try:
            from isaacsim.core.utils.types import ArticulationAction
        except Exception:
            from omni.isaac.core.utils.types import ArticulationAction

        self._sim_app = sim_app
        self._robot_controllers = list(robot_controllers)
        self._cmd_angular_sign = float(cmd_angular_sign)
        self._ArticulationAction = ArticulationAction
        self._helper_process: subprocess.Popen | None = None
        self._client_socket: socket.socket | None = None
        self._recv_buffer = ""
        self._helper_failure_logged = False
        self._latest_cmd_by_namespace: dict[str, tuple[float, float, float]] = {}
        self._state_cache_by_namespace: dict[str, _RobotStateCache] = {}
        self._odom_origin_by_namespace: dict[str, _OdomOrigin] = {}

        for controller in self._robot_controllers:
            self._latest_cmd_by_namespace[controller.namespace] = (0.0, 0.0, float("-inf"))
            self._state_cache_by_namespace[controller.namespace] = _RobotStateCache()
            self._reset_odom_origin(controller)

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(("127.0.0.1", 0))
        self._server_socket.listen(1)
        self._server_socket.setblocking(False)
        self._server_host, self._server_port = self._server_socket.getsockname()

        self._start_helper_process()
        print(
            "[OK] Isaac system ROS robot bridge listening on "
            f"{self._server_host}:{self._server_port} for namespaces "
            f"{[controller.namespace for controller in self._robot_controllers]}"
        )
        print(f"[OK] ROS cmd_vel angular sign for Isaac wheel control: {self._cmd_angular_sign:+.1f}")

    def _start_helper_process(self) -> None:
        if not os.path.exists(SYSTEM_ROS_ROBOT_HELPER_PATH):
            raise FileNotFoundError(
                f"System ROS robot bridge helper not found: {SYSTEM_ROS_ROBOT_HELPER_PATH}"
            )

        namespace_args = " ".join(
            f"--robot-namespace {shlex.quote(controller.namespace)}"
            for controller in self._robot_controllers
        )
        helper_command = (
            "source /opt/ros/humble/setup.bash && "
            f"exec /usr/bin/python3 {shlex.quote(SYSTEM_ROS_ROBOT_HELPER_PATH)} "
            f"--host {shlex.quote(self._server_host)} "
            f"--port {self._server_port} "
            f"{namespace_args}"
        )
        helper_env = self._build_helper_env()
        self._helper_process = subprocess.Popen(
            ["/bin/bash", "-lc", helper_command],
            cwd=REPO_ROOT,
            env=helper_env,
        )

    def _build_helper_env(self) -> dict[str, str]:
        helper_env: dict[str, str] = {}

        for key in (
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "TERM",
            "LANG",
            "LC_ALL",
            "DISPLAY",
            "XAUTHORITY",
            "ROS_DOMAIN_ID",
            "ROS_LOCALHOST_ONLY",
            "RMW_IMPLEMENTATION",
            "FASTRTPS_DEFAULT_PROFILES_FILE",
            "CYCLONEDDS_URI",
            "ROS_AUTOMATIC_DISCOVERY_RANGE",
        ):
            value = os.environ.get(key)
            if value:
                helper_env[key] = value

        helper_env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        helper_env["PYTHONNOUSERSITE"] = "1"
        return helper_env

    def spin_once(self) -> None:
        self._check_helper_process()
        self._accept_client_if_needed()
        self._read_client_messages()

    def step(self, sim_time_sec: float) -> None:
        self._apply_wheel_commands()
        self._publish_robot_state(sim_time_sec)

    def reset_after_pose_change(self) -> None:
        now = time.monotonic()
        for controller in self._robot_controllers:
            self._latest_cmd_by_namespace[controller.namespace] = (0.0, 0.0, now)
            self._state_cache_by_namespace[controller.namespace] = _RobotStateCache()
            self._reset_odom_origin(controller)
            joint_indices, joint_velocities = _wheel_action_arrays(
                controller,
                linear_x=0.0,
                angular_z=0.0,
            )
            action = self._ArticulationAction(
                joint_velocities=joint_velocities,
                joint_indices=joint_indices,
            )
            controller.articulation_controller.apply_action(action)

    def _reset_odom_origin(self, controller: Any) -> None:
        from isaac_sim.stage_bringups.runtime_utils import quaternion_xyzw_to_yaw

        world_position_xyz, world_orientation_xyzw = self._get_controller_world_pose_xyzw(controller)
        world_yaw_rad = quaternion_xyzw_to_yaw(world_orientation_xyzw)
        self._odom_origin_by_namespace[controller.namespace] = _OdomOrigin(
            world_position_xy=world_position_xyz[:2].copy(),
            world_yaw_rad=float(world_yaw_rad),
        )

    def _get_controller_world_pose_xyzw(self, controller: Any) -> tuple[np.ndarray, np.ndarray]:
        try:
            position_xyz, orientation_wxyz = controller.articulation.get_world_pose()
            position_xyz = np.asarray(position_xyz, dtype=np.float64)
            orientation_wxyz = np.asarray(orientation_wxyz, dtype=np.float64)
            if position_xyz.shape[0] >= 3 and orientation_wxyz.shape[0] >= 4:
                return (
                    position_xyz[:3].copy(),
                    np.array(
                        [
                            float(orientation_wxyz[1]),
                            float(orientation_wxyz[2]),
                            float(orientation_wxyz[3]),
                            float(orientation_wxyz[0]),
                        ],
                        dtype=np.float64,
                    ),
                )
        except Exception:
            pass

        from isaac_sim.stage_bringups.runtime_utils import get_world_pose_xyzw

        return get_world_pose_xyzw(controller.root_prim_path)

    def _world_pose_to_odom_pose(
        self,
        controller: Any,
        world_position_xyz: np.ndarray,
        world_orientation_xyzw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        origin = self._odom_origin_by_namespace[controller.namespace]
        world_yaw_rad = float(
            math.atan2(
                2.0
                * (
                    float(world_orientation_xyzw[3]) * float(world_orientation_xyzw[2])
                    + float(world_orientation_xyzw[0]) * float(world_orientation_xyzw[1])
                ),
                1.0
                - 2.0
                * (
                    float(world_orientation_xyzw[1]) * float(world_orientation_xyzw[1])
                    + float(world_orientation_xyzw[2]) * float(world_orientation_xyzw[2])
                ),
            )
        )
        delta_world_xy = world_position_xyz[:2] - origin.world_position_xy
        cos_origin = math.cos(origin.world_yaw_rad)
        sin_origin = math.sin(origin.world_yaw_rad)
        odom_x = cos_origin * float(delta_world_xy[0]) + sin_origin * float(delta_world_xy[1])
        odom_y = -sin_origin * float(delta_world_xy[0]) + cos_origin * float(delta_world_xy[1])
        odom_yaw = math.atan2(
            math.sin(world_yaw_rad - origin.world_yaw_rad),
            math.cos(world_yaw_rad - origin.world_yaw_rad),
        )
        half_yaw = 0.5 * odom_yaw
        odom_position_xyz = np.array(
            [odom_x, odom_y, float(world_position_xyz[2])],
            dtype=np.float64,
        )
        odom_orientation_xyzw = np.array(
            [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)],
            dtype=np.float64,
        )
        return odom_position_xyz, odom_orientation_xyzw, world_yaw_rad

    def _apply_wheel_commands(self) -> None:
        now_monotonic = time.monotonic()
        for controller in self._robot_controllers:
            linear_x, angular_z, stamp_sec = self._latest_cmd_by_namespace[controller.namespace]
            if now_monotonic - stamp_sec > controller.cmd_timeout_sec:
                linear_x = 0.0
                angular_z = 0.0

            angular_z *= self._cmd_angular_sign
            linear_x = float(
                np.clip(linear_x, -controller.max_linear_speed_mps, controller.max_linear_speed_mps)
            )
            angular_z = float(
                np.clip(
                    angular_z,
                    -controller.max_angular_speed_rps,
                    controller.max_angular_speed_rps,
                )
            )
            joint_indices, joint_velocities = _wheel_action_arrays(
                controller,
                linear_x=linear_x,
                angular_z=angular_z,
            )
            action = self._ArticulationAction(
                joint_velocities=joint_velocities,
                joint_indices=joint_indices,
            )
            controller.articulation_controller.apply_action(action)

    def _publish_robot_state(self, sim_time_sec: float) -> None:
        if self._client_socket is None:
            return

        for controller in self._robot_controllers:
            world_position_xyz, world_orientation_xyzw = self._get_controller_world_pose_xyzw(controller)
            position_xyz, orientation_xyzw, yaw_rad = self._world_pose_to_odom_pose(
                controller,
                world_position_xyz,
                world_orientation_xyzw,
            )
            cache = self._state_cache_by_namespace[controller.namespace]

            linear_vx = 0.0
            linear_vy = 0.0
            angular_vz = 0.0
            if (
                cache.sim_time_sec is not None
                and cache.world_position_xy is not None
                and cache.yaw_rad is not None
            ):
                dt = float(sim_time_sec) - float(cache.sim_time_sec)
                if dt > 1e-6:
                    delta_xy = world_position_xyz[:2] - cache.world_position_xy
                    world_vx = float(delta_xy[0] / dt)
                    world_vy = float(delta_xy[1] / dt)
                    cos_yaw = math.cos(yaw_rad)
                    sin_yaw = math.sin(yaw_rad)
                    linear_vx = cos_yaw * world_vx + sin_yaw * world_vy
                    linear_vy = -sin_yaw * world_vx + cos_yaw * world_vy
                    angular_vz = math.atan2(
                        math.sin(yaw_rad - cache.yaw_rad),
                        math.cos(yaw_rad - cache.yaw_rad),
                    ) / dt

            cache.sim_time_sec = float(sim_time_sec)
            cache.world_position_xy = world_position_xyz[:2].copy()
            cache.yaw_rad = float(yaw_rad)

            self._send_message(
                {
                    "type": "robot_state",
                    "namespace": controller.namespace,
                    "sim_time_sec": float(sim_time_sec),
                    "position": [float(value) for value in position_xyz[:3]],
                    "orientation_xyzw": [float(value) for value in orientation_xyzw[:4]],
                    "linear": [float(linear_vx), float(linear_vy), 0.0],
                    "angular_z": float(angular_vz),
                    "odom_frame_id": controller.odom_frame_id,
                    "base_frame_id": controller.base_frame_id,
                }
            )

    def _check_helper_process(self) -> None:
        if self._helper_process is None or self._helper_failure_logged:
            return

        return_code = self._helper_process.poll()
        if return_code is None:
            return

        self._helper_failure_logged = True
        print(
            "[WARN] Isaac system ROS robot bridge helper exited early with code "
            f"{return_code}. Robot ROS topics will not work until that helper starts correctly."
        )

    def _accept_client_if_needed(self) -> None:
        if self._client_socket is not None:
            return

        try:
            client_socket, _ = self._server_socket.accept()
        except BlockingIOError:
            return

        client_socket.setblocking(False)
        self._client_socket = client_socket
        print("[OK] System ROS robot bridge helper connected to Isaac Sim.")

    def _read_client_messages(self) -> None:
        if self._client_socket is None:
            return

        try:
            chunk = self._client_socket.recv(65536)
        except BlockingIOError:
            return
        except OSError:
            self._close_client_socket()
            return

        if not chunk:
            self._close_client_socket()
            return

        self._recv_buffer += chunk.decode("utf-8")
        while "\n" in self._recv_buffer:
            line, self._recv_buffer = self._recv_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self._handle_client_message(line)

    def _handle_client_message(self, line: str) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Failed to decode robot bridge message '{line}': {exc}")
            return

        if message.get("type") != "cmd_vel":
            return

        namespace = str(message.get("namespace", "")).strip("/")
        if namespace not in self._latest_cmd_by_namespace:
            return

        self._latest_cmd_by_namespace[namespace] = (
            float(message.get("linear_x", 0.0)),
            float(message.get("angular_z", 0.0)),
            time.monotonic(),
        )

    def _send_message(self, message: dict) -> None:
        if self._client_socket is None:
            return

        payload = (json.dumps(message) + "\n").encode("utf-8")
        try:
            self._client_socket.sendall(payload)
        except BlockingIOError:
            return
        except OSError:
            self._close_client_socket()

    def _close_client_socket(self) -> None:
        if self._client_socket is None:
            return
        try:
            self._client_socket.close()
        finally:
            self._client_socket = None
            self._recv_buffer = ""
            print("[WARN] System ROS robot bridge helper disconnected from Isaac Sim.")

    def shutdown(self) -> None:
        self._close_client_socket()
        try:
            self._server_socket.close()
        except Exception:
            pass

        if self._helper_process is not None:
            if self._helper_process.poll() is None:
                self._helper_process.terminate()
                try:
                    self._helper_process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._helper_process.kill()


##########################################
#### Stage builder and main function #####
##########################################


def build_stage(
    team_config_file: str,
    output_usd: str,
    scene_usd: str = "",
    enable_ros2: bool = True,
    rollout_id: int | None = None,
    robot_usd_rel: str | None = None,
):
    from isaacsim.storage.native import get_assets_root_path
    from isaac_sim.stage_bringups.warehouse_randomized.robots import build_robot_adapter

    print(f"[INFO] Loading team config: {team_config_file}")
    assets_root_path = get_assets_root_path()
    team_config = team_config_utils.load_team_config(
        team_config_file,
        maps_dir=MAPS_DIR,
        rollout_id=rollout_id,
    )
    print(
        f"[INFO] Selected rollout {team_config['rollout_id']} with "
        f"{team_config['agent_num']} robots: {team_config['robot_namespaces']}"
    )

    if scene_usd:
        print(f"[INFO] Opening scene USD: {scene_usd}")
        _open_stage(scene_usd)
        _ensure_xform_path("/World")
        _ensure_xform_path("/World/Robots")
        _clear_prim_children("/World/Robots")
        print(f"[OK] Opened environment scene USD: {scene_usd}")
    else:
        _new_stage()
        _set_stage_units(1.0)

        # Create prims
        _define_xform("/World")
        _define_xform("/World/Env")
        _define_xform("/World/Robots")

        # 1) Environment
        env_prim = "/World/Env/Warehouse"
        _add_reference(f"{assets_root_path}{ENV_USD_REL}", env_prim)

    def _safe_robot_prim_name(robot_name: str, index: int) -> str:
        safe_name = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in str(robot_name).strip("/")
        )
        return safe_name or f"robot_{index}"

    # 2) Robots from the selected rollout. Heterogeneous robots use sensorless
    # built-in USD assets and system-ROS helper topics instead of per-asset
    # ROS OmniGraphs.
    robot_infos: list[dict[str, Any]] = []
    for index, robot in enumerate(team_config["robots"], start=1):
        robot_name = str(robot["name"]).strip("/") or f"robot_{index}"
        robot_model = str(robot.get("model") or "nova_carter").strip() or "nova_carter"
        adapter = build_robot_adapter(robot_model)
        robot_prim = f"/World/Robots/{_safe_robot_prim_name(robot_name, index)}"
        initial_pose = team_config_utils.pose_array_to_pose_dict(robot["initial_pose"])
        print(
            f"[INFO] Spawning robot {index}: name={robot_name}, model={robot_model}, "
            f"prim={robot_prim}"
        )

        if robot_usd_rel:
            _add_reference(f"{assets_root_path}{robot_usd_rel}", robot_prim)
            _set_xform_pose(
                robot_prim,
                (
                    initial_pose["x"],
                    initial_pose["y"],
                    initial_pose["z"],
                ),
                yaw_deg=initial_pose["yaw"] * 180.0 / 3.141592653589793,
            )
        else:
            adapter.spawn_robot(
                assets_root_path=assets_root_path,
                prim_path=robot_prim,
                robot_index=index,
                position_xyz=(
                    initial_pose["x"],
                    initial_pose["y"],
                    initial_pose["z"],
                ),
                yaw_deg=initial_pose["yaw"] * 180.0 / 3.141592653589793,
            )
        _set_isaac_namespace(robot_prim, robot["name"])
        if robot_model.lower() == "jackal":
            _disable_lidar_debug_visualization(robot_prim)

        robot_infos.append(
            {
                "name": robot_name,
                "model": robot_model,
                "prim_path": robot_prim,
                "adapter": adapter,
            }
        )

    if enable_ros2:
        _ensure_global_ros2_clock_graph("/clock")

    # 4) Save (optional)
    if output_usd:
        import omni.usd
        omni.usd.get_context().save_as_stage(output_usd)
        print(f"[OK] Saved stage to: {output_usd}")

    print(
        "[OK] Stage built: "
        f"{scene_usd or 'built-in warehouse'} + {team_config['agent_num']} robots "
        f"from {team_config_file} rollout {team_config['rollout_id']}."
    )
    print(
        "[OK] Robot ROS topics: "
        + ", ".join(
            f"/{robot_info['name']}/cmd_vel, /{robot_info['name']}/chassis/odom, /{robot_info['name']}/tf"
            for robot_info in robot_infos
        )
    )

    # debug camera topic collisions
    # _find_ros2_camera_publishers()
    # _print_prim_inputs("/World/Robots/NovaCarter_1/front_hawk/camera_namespace")
    #
    return robot_infos


def main():
    (
        team_config_file,
        output_usd,
        scene_usd,
        rollout_control_topic,
        rollout_reset_done_topic,
        rollout_id,
        headless,
    ) = _parse_args()
    sim_app = _maybe_start_sim_app(headless=headless)
    rollout_reset_bridge = None
    robot_ros_bridge = None
    # warm-up updates
    for _ in range(10):
        sim_app.update()

    try:
        robot_infos = build_stage(
            team_config_file,
            output_usd,
            scene_usd=scene_usd,
            rollout_id=rollout_id,
        )
        robot_prim_paths = [robot_info["prim_path"] for robot_info in robot_infos]
        rollout_reset_bridge = IsaacRolloutResetBridge(
            sim_app=sim_app,
            robot_prim_paths=robot_prim_paths,
            rollout_control_topic=rollout_control_topic,
            rollout_reset_done_topic=rollout_reset_done_topic,
        )

        # one or two updates after stage creation helps graphs/materialization settle
        for _ in range(10):
            sim_app.update()

        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        for _ in range(8):
            sim_app.update()

        robot_controllers = []
        for robot_info in robot_infos:
            print(
                f"[INFO] Initializing articulation controller for "
                f"{robot_info['name']} at {robot_info['prim_path']}"
            )
            robot_controllers.append(
                robot_info["adapter"].initialize_runtime_controller(
                    sim_app=sim_app,
                    namespace=robot_info["name"],
                    root_prim_path=robot_info["prim_path"],
                )
            )
        robot_ros_bridge = IsaacSystemRosRobotBridge(
            sim_app=sim_app,
            robot_controllers=robot_controllers,
        )

        print("[OK] Timeline started from Python")

        while sim_app.is_running():
            rollout_reset_bridge.spin_once()
            applied_rollout_id = rollout_reset_bridge.process_pending_reset()
            if applied_rollout_id is not None and robot_ros_bridge is not None:
                robot_ros_bridge.reset_after_pose_change()
            robot_ros_bridge.spin_once()
            sim_app.update()
            robot_ros_bridge.step(float(timeline.get_current_time()))
        # # In standalone, keep UI alive so you can press Play and inspect the graph/topics.
        # if sim_app is not None:
        #     import time
        #     while sim_app.is_running():
        #         sim_app.update()
        #         # time.sleep(0.1)
    except Exception:
        import traceback

        print("\n[ERROR] build_stage_warehouse_carters.py failed with exception:")
        traceback.print_exc()
        raise
    finally:
        if robot_ros_bridge is not None:
            robot_ros_bridge.shutdown()
        if rollout_reset_bridge is not None:
            rollout_reset_bridge.shutdown()
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    # usage (tested on ubuntu 22.04 with ROS 2 Humble and Isaac Sim 5.1):
    # $ cd isaac-sim # Note: your isaac sim installation dir, not this script's dir
    # $ conda deactivate  # (optional) ensure base conda env is not active, to avoid conflicts with ROS 2 Humble's Python 3.10
    # $ source /opt/ros/humble/setup.bash  # source ROS 2 Humble setup to enable ROS2 bridge
    # $ ./python.sh /abs/path/to/mas-vln/isaac_sim/scripts/build_stage_warehouse_carters.py
    main()
