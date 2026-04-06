#!/usr/bin/env python3
"""
Isaac Sim 5.1 stage builder:
- references the same warehouse environment used by build_stage_warehouse_carters.py
- imports Dexmate Vega from the installed dexmate_urdf package
- places Vega at the origin under /World/Robots/Vega
- optionally saves the composed stage

Run standalone:
  ./python.sh /abs/path/to/build_stage_dexmate_example.py
  ./python.sh /abs/path/to/build_stage_dexmate_example.py \
    --output-usd /home/you/.../warehouse_vega.usd
"""

import argparse
import copy
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, Tuple

STANDALONE = True

ENV_USD_REL = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
VEGA_PRIM_PATH = "/World/Robots/Vega"
ENV_OUTPUT_USD = os.environ.get("OUTPUT_USD", "")
ENV_ASSETS_ROOT_PATH = os.environ.get("ISAAC_ASSETS_ROOT", "")
ENV_VEGA_JOINT_PRESET = os.environ.get("VEGA_JOINT_PRESET", "compact")

# Dexmate's URDF zero configuration leaves Vega's torso sprawled outward.
# This preset folds the torso up over the base and slightly relaxes the arms
# so the imported robot reads as a single assembled platform at startup.
VEGA_JOINT_PRESETS: Dict[str, Dict[str, float]] = {
    "zero": {},
    "compact": {
        "torso_j1": 0.131,
        "torso_j2": 0.262,
        "torso_j3": -0.523,
        "L_arm_j1": 0.300,
        "L_arm_j2": 0.150,
        "L_arm_j4": -0.200,
        "R_arm_j1": -0.300,
        "R_arm_j2": -0.150,
        "R_arm_j4": -0.200,
    },
}

_USD_EXCLUDED_PATH_TOKENS = {"/visuals/", "/collisions/", "/joints/", "/Looks/"}


def _normalize_path_or_url(value: str) -> str:
    if not value:
        return ""

    expanded = os.path.expanduser(value)
    if "://" in expanded:
        return expanded.rstrip("/")
    return os.path.abspath(expanded).rstrip("/")


def _parse_args():
    default_joint_preset = ENV_VEGA_JOINT_PRESET.strip().lower() or "compact"
    if default_joint_preset not in VEGA_JOINT_PRESETS:
        print(
            f"[WARN] Unsupported VEGA_JOINT_PRESET={ENV_VEGA_JOINT_PRESET!r}; "
            "falling back to 'compact'."
        )
        default_joint_preset = "compact"

    parser = argparse.ArgumentParser(
        description=(
            "Build the Isaac Sim warehouse stage and import the Dexmate Vega robot at the origin."
        )
    )
    parser.add_argument(
        "--output-usd",
        default=ENV_OUTPUT_USD,
        help="Optional output USD path. Defaults to OUTPUT_USD when set.",
    )
    parser.add_argument(
        "--assets-root-path",
        default=ENV_ASSETS_ROOT_PATH,
        help=(
            "Optional Isaac Sim assets root. Defaults to ISAAC_ASSETS_ROOT when set; "
            "otherwise the script asks Isaac Sim to resolve the configured assets root."
        ),
    )
    parser.add_argument(
        "--headless",
        default=False,
        action="store_true",
        help="Run Isaac Sim headless instead of opening the UI.",
    )
    parser.add_argument(
        "--floating-base",
        action="store_true",
        help="Import Vega with a floating base instead of the fixed base used by this example.",
    )
    parser.add_argument(
        "--play-on-start",
        action="store_true",
        help="Start the simulation timeline immediately after building the stage.",
    )
    parser.add_argument(
        "--use-collision-visuals",
        action="store_true",
        help=(
            "Debug Vega visual-frame issues by replacing visual meshes with collision meshes "
            "in a temporary URDF before import."
        ),
    )
    parser.add_argument(
        "--joint-preset",
        choices=tuple(VEGA_JOINT_PRESETS.keys()),
        default=default_joint_preset,
        help=(
            "Startup joint pose for Vega. 'compact' folds the torso over the base so the robot "
            "looks assembled at rest; 'zero' keeps the raw URDF q=0 pose."
        ),
    )
    args, unknown = parser.parse_known_args()

    output_usd = os.path.abspath(os.path.expanduser(args.output_usd)) if args.output_usd else ""
    assets_root_path = _normalize_path_or_url(args.assets_root_path)

    if unknown:
        print(f"[INFO] Ignoring unknown CLI args passed through Isaac Sim: {unknown}")

    return (
        output_usd,
        assets_root_path,
        args.headless,
        not args.floating_base,
        args.play_on_start,
        args.use_collision_visuals,
        args.joint_preset,
    )


def _maybe_start_sim_app(headless: bool):
    if not STANDALONE:
        return None

    from isaacsim.simulation_app import SimulationApp

    sim_app = SimulationApp({"headless": headless})
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
    from isaacsim.core.utils.stage import add_reference_to_stage

    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)


def _define_xform(path: str):
    from isaacsim.core.utils.prims import define_prim

    define_prim(path, "Xform")


def _set_xform_pose(prim_path: str, position_xyz: Tuple[float, float, float], yaw_deg: float = 0.0):
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    x, y, z = position_xyz
    rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(yaw_deg))

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()

    mat = Gf.Matrix4d(1.0)
    mat.SetTransform(rot, Gf.Vec3d(x, y, z))
    op.Set(mat)


def _default_vega_robot_usd_path(output_usd: str, use_collision_visuals: bool) -> str:
    suffix = "_vega_collision_visuals" if use_collision_visuals else "_vega"
    if output_usd:
        output_dir = os.path.dirname(output_usd)
        output_name = os.path.splitext(os.path.basename(output_usd))[0]
        return os.path.join(output_dir, f"{output_name}{suffix}.usd")

    basename = "vega_1_f5d6_collision_visuals.usd" if use_collision_visuals else "vega_1_f5d6.usd"
    return os.path.join(tempfile.gettempdir(), basename)


def _ensure_symlink(link_path: str, target_path: str):
    if os.path.islink(link_path):
        if os.readlink(link_path) == target_path:
            return
        os.unlink(link_path)
    elif os.path.exists(link_path):
        raise RuntimeError(f"Cannot create symlink at existing path: {link_path}")

    os.symlink(target_path, link_path)


def _build_collision_visuals_debug_urdf(original_urdf_path: str) -> Tuple[str, int]:
    original_urdf_path = os.path.abspath(os.path.expanduser(original_urdf_path))
    original_vega_dir = os.path.dirname(original_urdf_path)
    original_robots_dir = os.path.abspath(os.path.join(original_vega_dir, "..", ".."))

    debug_root = os.path.join(tempfile.gettempdir(), "dexmate_urdf_debug")
    debug_robots_dir = os.path.join(debug_root, "robots")
    debug_vega_dir = os.path.join(debug_robots_dir, "humanoid", "vega_1")
    os.makedirs(debug_vega_dir, exist_ok=True)

    # Preserve Dexmate's relative mesh references:
    #   meshes/...                     -> ./meshes
    #   ../../hands/f5d6_hand/...     -> ../../hands/f5d6_hand/...
    _ensure_symlink(os.path.join(debug_vega_dir, "meshes"), os.path.join(original_vega_dir, "meshes"))
    _ensure_symlink(os.path.join(debug_robots_dir, "hands"), os.path.join(original_robots_dir, "hands"))

    tree = ET.parse(original_urdf_path)
    root = tree.getroot()
    swapped_links = 0

    for link in root.findall("link"):
        collision_elems = link.findall("collision")
        if not collision_elems:
            continue

        for visual_elem in list(link.findall("visual")):
            link.remove(visual_elem)

        insert_index = 0
        for idx, child in enumerate(list(link)):
            if child.tag != "inertial":
                insert_index = idx
                break
            insert_index = idx + 1

        for collision_elem in collision_elems:
            visual_elem = copy.deepcopy(collision_elem)
            visual_elem.tag = "visual"
            link.insert(insert_index, visual_elem)
            insert_index += 1

        swapped_links += 1

    debug_urdf_path = os.path.join(debug_vega_dir, "vega_1_f5d6_collision_visuals.urdf")
    tree.write(debug_urdf_path, encoding="utf-8", xml_declaration=True)

    return debug_urdf_path, swapped_links


def _parse_xyz_or_rpy(attr_value: str) -> Tuple[float, float, float]:
    if not attr_value:
        return (0.0, 0.0, 0.0)

    parts = attr_value.split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values, got {attr_value!r}")
    return tuple(float(part) for part in parts)


def _rpy_to_rotation_matrix(rpy: Tuple[float, float, float]):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = [
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ]
    ry = [
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ]
    rz = [
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ]

    return _mat3_mul(_mat3_mul(rz, ry), rx)


def _axis_angle_to_rotation_matrix(axis_xyz: Tuple[float, float, float], angle_rad: float):
    axis_norm = math.sqrt(sum(value * value for value in axis_xyz))
    if axis_norm <= 1e-9:
        return _identity_matrix3()

    x, y, z = (value / axis_norm for value in axis_xyz)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_minus_c = 1.0 - c

    return [
        [x * x * one_minus_c + c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
        [y * x * one_minus_c + z * s, y * y * one_minus_c + c, y * z * one_minus_c - x * s],
        [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, z * z * one_minus_c + c],
    ]


def _make_transform_matrix(
    xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    mat = _identity_matrix4()
    rot = _rpy_to_rotation_matrix(rpy)
    for row in range(3):
        for col in range(3):
            mat[row][col] = rot[row][col]
        mat[row][3] = xyz[row]
    return mat


def _rotation_matrix_to_quat_wxyz(rot):
    trace = rot[0][0] + rot[1][1] + rot[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2][1] - rot[1][2]) / s
        y = (rot[0][2] - rot[2][0]) / s
        z = (rot[1][0] - rot[0][1]) / s
    elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
        s = math.sqrt(1.0 + rot[0][0] - rot[1][1] - rot[2][2]) * 2.0
        w = (rot[2][1] - rot[1][2]) / s
        x = 0.25 * s
        y = (rot[0][1] + rot[1][0]) / s
        z = (rot[0][2] + rot[2][0]) / s
    elif rot[1][1] > rot[2][2]:
        s = math.sqrt(1.0 + rot[1][1] - rot[0][0] - rot[2][2]) * 2.0
        w = (rot[0][2] - rot[2][0]) / s
        x = (rot[0][1] + rot[1][0]) / s
        y = 0.25 * s
        z = (rot[1][2] + rot[2][1]) / s
    else:
        s = math.sqrt(1.0 + rot[2][2] - rot[0][0] - rot[1][1]) * 2.0
        w = (rot[1][0] - rot[0][1]) / s
        x = (rot[0][2] + rot[2][0]) / s
        y = (rot[1][2] + rot[2][1]) / s
        z = 0.25 * s

    quat = (w, x, y, z)
    quat_norm = math.sqrt(sum(value * value for value in quat))
    if quat_norm <= 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(value / quat_norm) for value in quat)


def _identity_matrix3():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _identity_matrix4():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat3_mul(left, right):
    result = []
    for row in range(3):
        result.append([])
        for col in range(3):
            value = sum(left[row][k] * right[k][col] for k in range(3))
            result[row].append(float(value))
    return result


def _mat4_mul(left, right):
    result = []
    for row in range(4):
        result.append([])
        for col in range(4):
            value = sum(left[row][k] * right[k][col] for k in range(4))
            result[row].append(float(value))
    return result


def _set_translate_orient_scale_on_prim(prim, translation_xyz, quat_wxyz, scale_xyz=(1.0, 1.0, 1.0)):
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    orient_op = None
    scale_op = None
    recreate_ops = False

    for op in xformable.GetOrderedXformOps():
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif op_type == UsdGeom.XformOp.TypeOrient and orient_op is None:
            orient_op = op
        elif op_type == UsdGeom.XformOp.TypeScale and scale_op is None:
            scale_op = op
        else:
            recreate_ops = True

    if recreate_ops or translate_op is None or orient_op is None or scale_op is None:
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    else:
        xformable.SetXformOpOrder([translate_op, orient_op, scale_op])

    if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        translate_op.Set(Gf.Vec3f(*translation_xyz))
    else:
        translate_op.Set(Gf.Vec3d(*translation_xyz))

    orient_real = quat_wxyz[0]
    orient_imag = quat_wxyz[1:]
    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        orient_op.Set(Gf.Quatf(float(orient_real), Gf.Vec3f(*orient_imag)))
    else:
        orient_op.Set(Gf.Quatd(float(orient_real), Gf.Vec3d(*orient_imag)))

    if scale_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        scale_op.Set(Gf.Vec3f(*scale_xyz))
    else:
        scale_op.Set(Gf.Vec3d(*scale_xyz))


def _should_skip_usd_prim_path(path: str) -> bool:
    return any(token in path for token in _USD_EXCLUDED_PATH_TOKENS)


def _fix_vega_base_visual_in_generated_asset(robot_usd_path: str):
    from pxr import Usd

    robot_dir = os.path.dirname(robot_usd_path)
    robot_name = os.path.splitext(os.path.basename(robot_usd_path))[0]
    base_asset_path = os.path.join(robot_dir, "configuration", f"{robot_name}_base.usd")
    if not os.path.exists(base_asset_path):
        print(f"[WARN] Vega base asset not found for visual correction: {base_asset_path}")
        return

    stage = Usd.Stage.Open(base_asset_path)
    if stage is None:
        raise RuntimeError(f"Failed to open Vega base asset USD: {base_asset_path}")

    base_visual_prim = stage.GetPrimAtPath("/visuals/base/base/base")
    if not base_visual_prim or not base_visual_prim.IsValid():
        print(f"[WARN] Vega base visual prim not found in asset: {base_asset_path}")
        return

    _set_translate_orient_scale_on_prim(
        base_visual_prim,
        translation_xyz=(-0.008999999612569809, 0.01298999972641468, 0.1550000011920929),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    stage.Save()
    print(f"[OK] Corrected Vega base visual orientation in asset: {base_asset_path}")


def _compute_urdf_link_transforms(urdf_path: str, joint_positions: Dict[str, float]):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    child_joint_map = {}
    children_by_parent = {}
    child_link_names = set()
    link_names = []

    for link_elem in root.findall("link"):
        link_names.append(link_elem.attrib["name"])

    for joint_elem in root.findall("joint"):
        parent_link = joint_elem.find("parent").attrib["link"]
        child_link = joint_elem.find("child").attrib["link"]
        joint_type = joint_elem.attrib["type"]

        origin_elem = joint_elem.find("origin")
        axis_elem = joint_elem.find("axis")

        origin_xyz = _parse_xyz_or_rpy(origin_elem.attrib.get("xyz", "")) if origin_elem is not None else (0.0, 0.0, 0.0)
        origin_rpy = _parse_xyz_or_rpy(origin_elem.attrib.get("rpy", "")) if origin_elem is not None else (0.0, 0.0, 0.0)
        axis_xyz = _parse_xyz_or_rpy(axis_elem.attrib.get("xyz", "")) if axis_elem is not None else (0.0, 0.0, 1.0)

        joint_data = {
            "name": joint_elem.attrib["name"],
            "type": joint_type,
            "parent": parent_link,
            "child": child_link,
            "origin_xyz": origin_xyz,
            "origin_rpy": origin_rpy,
            "axis_xyz": axis_xyz,
        }
        child_joint_map[child_link] = joint_data
        children_by_parent.setdefault(parent_link, []).append(child_link)
        child_link_names.add(child_link)

    root_link_candidates = [name for name in link_names if name not in child_link_names]
    if len(root_link_candidates) != 1:
        raise RuntimeError(f"Expected exactly one URDF root link, found: {root_link_candidates}")

    root_link = root_link_candidates[0]
    local_transforms = {root_link: _identity_matrix4()}
    world_transforms = {root_link: _identity_matrix4()}

    def _walk(parent_link: str):
        parent_world = world_transforms[parent_link]
        for child_link in children_by_parent.get(parent_link, []):
            joint_data = child_joint_map[child_link]
            joint_local = _make_transform_matrix(
                xyz=joint_data["origin_xyz"],
                rpy=joint_data["origin_rpy"],
            )

            if joint_data["type"] in {"revolute", "continuous"}:
                q = float(joint_positions.get(joint_data["name"], 0.0))
                joint_rotation = _identity_matrix4()
                rot = _axis_angle_to_rotation_matrix(joint_data["axis_xyz"], q)
                for row in range(3):
                    for col in range(3):
                        joint_rotation[row][col] = rot[row][col]
                joint_local = _mat4_mul(joint_local, joint_rotation)
            elif joint_data["type"] == "prismatic":
                q = float(joint_positions.get(joint_data["name"], 0.0))
                translation = _identity_matrix4()
                for idx, value in enumerate(joint_data["axis_xyz"]):
                    translation[idx][3] = float(value) * q
                joint_local = _mat4_mul(joint_local, translation)

            local_transforms[child_link] = joint_local
            world_transforms[child_link] = _mat4_mul(parent_world, joint_local)
            _walk(child_link)

    _walk(root_link)
    return root_link, child_joint_map, world_transforms, local_transforms


def _bake_vega_rest_pose_into_robot_usd(robot_usd_path: str, urdf_path: str, joint_preset: str):
    from pxr import Usd

    preset = VEGA_JOINT_PRESETS[joint_preset]
    root_link, child_joint_map, world_transforms, local_transforms = _compute_urdf_link_transforms(
        urdf_path=urdf_path,
        joint_positions=preset,
    )

    stage = Usd.Stage.Open(robot_usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open Vega robot USD for baking: {robot_usd_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError(f"Robot USD does not have a valid default prim: {robot_usd_path}")

    robot_root_path = str(default_prim.GetPath())
    link_prim_paths = {}

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Xform":
            continue

        prim_path = str(prim.GetPath())
        if _should_skip_usd_prim_path(prim_path):
            continue
        if not prim_path.startswith(f"{robot_root_path}/"):
            continue

        prim_name = prim.GetName()
        if prim_name in world_transforms:
            link_prim_paths[prim_name] = prim_path

    baked_links = 0
    for link_name, prim_path in link_prim_paths.items():
        prim = stage.GetPrimAtPath(prim_path)
        parent_name = child_joint_map.get(link_name, {}).get("parent")
        usd_parent_name = prim.GetParent().GetName() if prim.GetParent() and prim.GetParent().IsValid() else ""

        if link_name == root_link:
            transform = world_transforms[link_name]
        elif usd_parent_name == parent_name:
            transform = local_transforms[link_name]
        else:
            transform = world_transforms[link_name]

        translation = (
            float(transform[0][3]),
            float(transform[1][3]),
            float(transform[2][3]),
        )
        quat_wxyz = _rotation_matrix_to_quat_wxyz([row[:3] for row in transform[:3]])
        _set_translate_orient_scale_on_prim(prim, translation_xyz=translation, quat_wxyz=quat_wxyz)
        baked_links += 1

    stage.Save()
    _fix_vega_base_visual_in_generated_asset(robot_usd_path)
    print(f"[OK] Baked Vega rest pose '{joint_preset}' into USD: {robot_usd_path} ({baked_links} link prims)")


def _create_vega_import_config(fix_base: bool):
    import omni.kit.commands
    from isaacsim.asset.importer.urdf import _urdf

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create URDF import configuration.")

    # Match Isaac Sim's URDF importer defaults closely, while skipping
    # per-robot physics scene creation because this script composes into
    # a larger warehouse stage.
    import_config.set_merge_fixed_joints(True)
    import_config.set_replace_cylinders_with_capsules(False)
    import_config.set_convex_decomp(False)
    import_config.set_import_inertia_tensor(True)
    import_config.set_fix_base(fix_base)
    import_config.set_self_collision(False)
    import_config.set_density(0.0)
    import_config.set_distance_scale(1.0)
    import_config.set_default_drive_type(1)
    import_config.set_default_drive_strength(1e3)
    import_config.set_default_position_drive_damping(1e2)
    import_config.set_up_vector(0, 0, 1)
    import_config.set_make_default_prim(True)
    import_config.set_parse_mimic(True)
    import_config.set_create_physics_scene(False)
    import_config.set_collision_from_visuals(False)

    return import_config


def _apply_vega_joint_preset(sim_app, joint_preset: str):
    if joint_preset == "zero":
        print("[OK] Leaving Vega in the raw URDF zero pose (--joint-preset zero).")
        return

    import numpy as np
    import omni.timeline
    from isaacsim.core.prims import SingleArticulation

    preset = VEGA_JOINT_PRESETS[joint_preset]
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        sim_app.update()

    robot = SingleArticulation(
        prim_path=VEGA_PRIM_PATH,
        name="vega",
        reset_xform_properties=False,
    )
    robot.initialize()

    dof_names = list(robot.dof_names)
    dof_name_to_index = {name: idx for idx, name in enumerate(dof_names)}
    missing = [name for name in preset if name not in dof_name_to_index]
    if missing:
        raise RuntimeError(f"Vega preset references missing DOFs: {missing}")

    positions = robot.get_joint_positions()
    if positions is None:
        positions = np.zeros(len(dof_names), dtype=np.float32)
    else:
        positions = np.asarray(positions, dtype=np.float32).copy()

    for name, value in preset.items():
        positions[dof_name_to_index[name]] = float(value)

    velocities = np.zeros_like(positions)
    robot.set_joints_default_state(positions=positions, velocities=velocities)
    robot.set_joint_positions(positions)
    robot.set_joint_velocities(velocities)
    robot.post_reset()

    for _ in range(5):
        sim_app.update()

    applied_summary = ", ".join(f"{name}={value:.3f}" for name, value in preset.items())
    print(f"[OK] Applied Vega joint preset '{joint_preset}': {applied_summary}")


def _build_vega_robot_usd(
    robot_usd_path: str,
    fix_base: bool,
    use_collision_visuals: bool,
    joint_preset: str,
) -> str:
    import omni.kit.commands
    from dexmate_urdf import robots

    urdf_path = robots.humanoid.vega_1.vega_1_f5d6.urdf
    if use_collision_visuals:
        urdf_path, swapped_links = _build_collision_visuals_debug_urdf(urdf_path)
        print(f"[OK] Generated collision-visuals debug URDF: {urdf_path} (swapped {swapped_links} links)")

    import_config = _create_vega_import_config(fix_base=fix_base)

    robot_usd_path = os.path.abspath(os.path.expanduser(robot_usd_path))
    os.makedirs(os.path.dirname(robot_usd_path), exist_ok=True)

    result, imported_prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        dest_path=robot_usd_path,
    )
    if not result:
        raise RuntimeError(f"Failed to export Vega URDF to USD: {urdf_path}")

    _bake_vega_rest_pose_into_robot_usd(
        robot_usd_path=robot_usd_path,
        urdf_path=urdf_path,
        joint_preset=joint_preset,
    )

    print(f"[OK] Exported Vega robot USD: {robot_usd_path}")
    print(f"[OK] Vega default prim inside robot USD: {imported_prim_path}")

    return robot_usd_path


def build_stage(
    output_usd: str,
    assets_root_path: str,
    fix_base: bool,
    use_collision_visuals: bool,
    joint_preset: str,
):
    from isaacsim.storage.native import get_assets_root_path

    if not assets_root_path:
        assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Isaac Sim assets root path could not be resolved.")

    _new_stage()
    _set_stage_units(1.0)

    _define_xform("/World")
    _define_xform("/World/Env")
    _define_xform("/World/Robots")

    env_prim = "/World/Env/Warehouse"
    _add_reference(f"{assets_root_path}{ENV_USD_REL}", env_prim)

    robot_usd_path = _build_vega_robot_usd(
        _default_vega_robot_usd_path(output_usd, use_collision_visuals=use_collision_visuals),
        fix_base=fix_base,
        use_collision_visuals=use_collision_visuals,
        joint_preset=joint_preset,
    )
    _add_reference(robot_usd_path, VEGA_PRIM_PATH)
    _set_xform_pose(VEGA_PRIM_PATH, (0.0, 0.0, 0.0))

    print(
        "[OK] Stage built: warehouse + Dexmate Vega at the origin "
        f"(fix_base={fix_base}, collision_visuals={use_collision_visuals}, robot_usd={robot_usd_path})"
    )


def main():
    (
        output_usd,
        assets_root_path,
        headless,
        fix_base,
        play_on_start,
        use_collision_visuals,
        joint_preset,
    ) = _parse_args()
    sim_app = _maybe_start_sim_app(headless=headless)

    if sim_app is None:
        raise RuntimeError("SimulationApp failed to start")

    for _ in range(10):
        sim_app.update()

    try:
        build_stage(
            output_usd=output_usd,
            assets_root_path=assets_root_path,
            fix_base=fix_base,
            use_collision_visuals=use_collision_visuals,
            joint_preset=joint_preset,
        )
        _apply_vega_joint_preset(sim_app=sim_app, joint_preset=joint_preset)

        if output_usd:
            import omni.usd

            omni.usd.get_context().save_as_stage(output_usd)
            print(f"[OK] Saved stage to: {output_usd}")

        for _ in range(10):
            sim_app.update()

        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        if play_on_start:
            timeline.play()
            print("[OK] Timeline started from Python")
        else:
            timeline.pause()
            print("[OK] Timeline paused for inspection. Use --play-on-start to auto-play.")

        while sim_app.is_running():
            sim_app.update()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
