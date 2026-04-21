#!/usr/bin/env python3
"""Shared Isaac Sim stage/runtime helpers used by stage bringup scripts."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def maybe_start_sim_app(
    *,
    headless: bool = True,
    enable_ros2_bridge: bool = False,
    extra_extensions: Sequence[str] | None = None,
):
    """Start a SimulationApp and enable any requested extensions."""

    from isaacsim.simulation_app import SimulationApp

    sim_app = SimulationApp({"headless": bool(headless)})

    requested_extensions = list(extra_extensions or [])
    if enable_ros2_bridge:
        requested_extensions.insert(0, "isaacsim.ros2.bridge")

    if requested_extensions:
        try:
            from isaacsim.core.utils.extensions import enable_extension
        except Exception:
            from omni.isaac.core.utils.extensions import enable_extension

        seen: set[str] = set()
        for extension_name in requested_extensions:
            clean_name = str(extension_name).strip()
            if not clean_name or clean_name in seen:
                continue
            enable_extension(clean_name)
            seen.add(clean_name)

        sim_app.update()

    return sim_app


def get_stage():
    import omni.usd

    return omni.usd.get_context().get_stage()


def new_stage() -> None:
    try:
        from isaacsim.core.utils.stage import create_new_stage
    except Exception:
        import omni.usd

        omni.usd.get_context().new_stage()
    else:
        create_new_stage()


def set_stage_units(meters_per_unit: float = 1.0) -> None:
    from pxr import UsdGeom

    UsdGeom.SetStageMetersPerUnit(get_stage(), float(meters_per_unit))


def add_reference(usd_path: str, prim_path: str) -> None:
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage
    except Exception:
        from omni.isaac.core.utils.stage import add_reference_to_stage

    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)


def define_xform(path: str) -> None:
    try:
        from isaacsim.core.utils.prims import define_prim
    except Exception:
        from omni.isaac.core.utils.prims import define_prim

    define_prim(path, "Xform")


def ensure_xform_path(path: str) -> None:
    if not path or path == "/":
        return

    stage = get_stage()
    current = ""
    for token in [token for token in path.split("/") if token]:
        current += f"/{token}"
        prim = stage.GetPrimAtPath(current)
        if prim and prim.IsValid():
            continue
        define_xform(current)


def set_xform_pose(
    prim_path: str,
    position_xyz: Sequence[float],
    yaw_deg: float = 0.0,
    *,
    scale_xyz: Sequence[float] | None = None,
) -> None:
    from pxr import Gf, UsdGeom

    stage = get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    position_values = [float(value) for value in position_xyz[:3]]
    scale_values = None if scale_xyz is None else [float(value) for value in scale_xyz[:3]]

    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    rotate_xyz_op = None
    orient_op = None
    scale_op = None

    for op in xformable.GetOrderedXformOps():
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ and rotate_xyz_op is None:
            rotate_xyz_op = op
        elif op_type == UsdGeom.XformOp.TypeOrient and orient_op is None:
            orient_op = op
        elif op_type == UsdGeom.XformOp.TypeScale and scale_op is None:
            scale_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    if rotate_xyz_op is None and orient_op is None:
        rotate_xyz_op = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
    if scale_values is not None and scale_op is None:
        scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)

    ordered_ops = [translate_op]
    if orient_op is not None:
        ordered_ops.append(orient_op)
    else:
        ordered_ops.append(rotate_xyz_op)
    if scale_op is not None:
        ordered_ops.append(scale_op)
    xformable.SetXformOpOrder(ordered_ops)

    if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        translate_op.Set(Gf.Vec3f(*position_values))
    else:
        translate_op.Set(Gf.Vec3d(*position_values))

    if orient_op is not None:
        yaw_rad = np.deg2rad(float(yaw_deg))
        half_yaw = 0.5 * float(yaw_rad)
        quat_xyzw = (0.0, 0.0, float(np.sin(half_yaw)), float(np.cos(half_yaw)))
        if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            orient_op.Set(Gf.Quatf(quat_xyzw[3], Gf.Vec3f(*quat_xyzw[:3])))
        else:
            orient_op.Set(Gf.Quatd(quat_xyzw[3], Gf.Vec3d(*quat_xyzw[:3])))
    else:
        rotation_values = [0.0, 0.0, float(yaw_deg)]
        if rotate_xyz_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            rotate_xyz_op.Set(Gf.Vec3f(*rotation_values))
        else:
            rotate_xyz_op.Set(Gf.Vec3d(*rotation_values))

    if scale_values is not None and scale_op is not None:
        if scale_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            scale_op.Set(Gf.Vec3f(*scale_values))
        else:
            scale_op.Set(Gf.Vec3d(*scale_values))


def get_world_transform_matrix(prim_path: str):
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    try:
        world_matrix = omni.usd.get_world_transform_matrix(prim)
        if world_matrix is not None:
            return world_matrix
    except Exception:
        pass

    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def get_world_pose_xyzw(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = get_world_transform_matrix(prim_path)
    translation = matrix.ExtractTranslation()
    quat = matrix.ExtractRotationQuat()
    position = np.array([float(translation[0]), float(translation[1]), float(translation[2])])
    orientation = np.array(
        [
            float(quat.GetImaginary()[0]),
            float(quat.GetImaginary()[1]),
            float(quat.GetImaginary()[2]),
            float(quat.GetReal()),
        ]
    )
    return position, orientation


def quaternion_xyzw_to_yaw(quaternion_xyzw: Sequence[float]) -> float:
    x = float(quaternion_xyzw[0])
    y = float(quaternion_xyzw[1])
    z = float(quaternion_xyzw[2])
    w = float(quaternion_xyzw[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def compute_world_bbox(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom

    stage = get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
    )
    world_bound = bbox_cache.ComputeWorldBound(prim)
    aligned_range = world_bound.ComputeAlignedRange()
    bbox_min = aligned_range.GetMin()
    bbox_max = aligned_range.GetMax()
    return (
        np.array([float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])]),
        np.array([float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])]),
    )


def iter_prims_under(root_prim_path: str) -> Iterable[Any]:
    stage = get_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        return []

    root_prefix = root_prim_path.rstrip("/")
    hits: list[Any] = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path == root_prefix or path.startswith(f"{root_prefix}/"):
            hits.append(prim)
    return hits


def set_isaac_namespace(prim_path: str, namespace: str) -> None:
    from pxr import Sdf

    stage = get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    attr = prim.GetAttribute("isaac:namespace")
    if not (attr and attr.IsValid()):
        attr = prim.CreateAttribute("isaac:namespace", Sdf.ValueTypeNames.String)
    attr.Set(str(namespace))


def is_ros2_omnigraph_node(prim) -> bool:
    if not prim or not prim.IsValid():
        return False
    attribute_names = [attribute.GetName() for attribute in prim.GetAttributes()]
    return any(
        name
        in (
            "inputs:nodeNamespace",
            "inputs:topicName",
            "inputs:frameId",
            "inputs:odomFrameId",
            "inputs:baseFrameId",
            "inputs:childFrameId",
            "inputs:parentFrameId",
        )
        for name in attribute_names
    )


def prefix_frame(frame: str, namespace: str) -> str:
    if not frame:
        return frame
    if frame in {"map", "world"}:
        return frame
    if frame.startswith(f"{namespace}/"):
        return frame
    return f"{namespace}/{frame}"


def fix_ros2_graph_under(
    root_prim_path: str,
    namespace: str,
    *,
    prefix_frames: bool = False,
) -> None:
    stage = get_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Root prim not found: {root_prim_path}")

    touched: dict[str, int] = {"nodeNamespace": 0, "topicName": 0, "frames": 0}

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not (path == root_prim_path or path.startswith(f"{root_prim_path}/")):
            continue

        if prim.GetName() == "camera_namespace":
            value_attr = prim.GetAttribute("inputs:value")
            if value_attr and value_attr.IsValid():
                value = value_attr.Get()
                if isinstance(value, str) and "camera" in value.lower():
                    base = value.strip("/")
                    value_attr.Set(f"/{namespace}/{base}")
                    touched["cameraNamespace"] = touched.get("cameraNamespace", 0) + 1

        if not is_ros2_omnigraph_node(prim):
            continue

        namespace_attr = prim.GetAttribute("inputs:nodeNamespace")
        if namespace_attr and namespace_attr.IsValid():
            try:
                namespace_attr.Set(namespace)
                touched["nodeNamespace"] += 1
            except Exception:
                pass

        topic_attr = prim.GetAttribute("inputs:topicName")
        if topic_attr and topic_attr.IsValid():
            try:
                topic_name = topic_attr.Get()
                if isinstance(topic_name, str) and topic_name.startswith("/") and topic_name != "/clock":
                    topic_attr.Set(topic_name.lstrip("/"))
                    touched["topicName"] += 1
            except Exception:
                pass

        if not prefix_frames:
            continue

        for frame_key in (
            "inputs:frameId",
            "inputs:odomFrameId",
            "inputs:baseFrameId",
            "inputs:childFrameId",
            "inputs:parentFrameId",
        ):
            frame_attr = prim.GetAttribute(frame_key)
            if not (frame_attr and frame_attr.IsValid()):
                continue
            try:
                frame = frame_attr.Get()
                if isinstance(frame, str) and frame:
                    prefixed = prefix_frame(frame, namespace)
                    if prefixed != frame:
                        frame_attr.Set(prefixed)
                        touched["frames"] += 1
            except Exception:
                pass

    print(f"[OK] Patched ROS2 graph under {root_prim_path}: {touched}")


def ensure_global_ros2_clock_graph(clock_topic: str = "/clock") -> None:
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
                    ("PublishClock.inputs:topicName", str(clock_topic)),
                ],
            },
        )
        print(f"[OK] Added global ROS2 clock publisher graph at {graph_path} on topic '{clock_topic}'")
    except Exception as exc:
        print(f"[WARN] Failed to create ROS2 clock graph at {graph_path}: {exc}")
