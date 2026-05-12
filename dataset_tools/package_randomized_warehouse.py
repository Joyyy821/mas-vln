from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml


REQUIRED_SCENE_FILES = (
    "mapf_map.png",
    "mapf_map.yaml",
    "nav2_map.png",
    "nav2_map.yaml",
    "scene_manifest.yaml",
    "team_config.yaml",
)
OPTIONAL_SCENE_FILES = ("collection_metadata.yaml",)
REQUIRED_ROLLOUT_ITEMS = ("run_config.yaml", "render_manifest.csv", "rgb", "depth")
METADATA_FILENAMES = {
    "scenes": "scenes.parquet",
    "rollouts": "rollouts.parquet",
    "frames": "frames.parquet",
}

ParquetWriter = Callable[[list[dict[str, Any]], Path], None]


class PackageError(RuntimeError):
    """Raised when the dataset cannot be packaged safely."""


@dataclass(frozen=True)
class RolloutInput:
    scene_id: int
    scene_dir: Path
    rollout_id: int
    rollout_dir: Path


@dataclass(frozen=True)
class SceneInput:
    scene_id: int
    scene_dir: Path
    rollouts: tuple[RolloutInput, ...]


@dataclass(frozen=True)
class VelocitySample:
    timestamp_ns: int
    vx: float | None
    vy: float | None
    wz: float | None
    x: float | None
    y: float | None
    yaw: float | None
    cmd_vel_timestamp_ns: int | None
    cmd_vx: float | None
    cmd_vy: float | None
    cmd_wz: float | None


class VelocityIndex:
    def __init__(self, samples: Iterable[VelocitySample]) -> None:
        self.samples = tuple(sorted(samples, key=lambda sample: sample.timestamp_ns))
        self.timestamps = tuple(sample.timestamp_ns for sample in self.samples)

    def nearest(self, timestamp_ns: int) -> VelocitySample | None:
        if not self.samples:
            return None
        insert_at = bisect.bisect_left(self.timestamps, timestamp_ns)
        candidates: list[VelocitySample] = []
        if insert_at < len(self.samples):
            candidates.append(self.samples[insert_at])
        if insert_at > 0:
            candidates.append(self.samples[insert_at - 1])
        return min(candidates, key=lambda sample: abs(sample.timestamp_ns - timestamp_ns))


def package_randomized_warehouse(
    raw_root: str | Path,
    out_root: str | Path,
    version: str,
    *,
    append: bool = False,
    include_scene_usd: bool = False,
    overwrite_rollout_tars: bool = False,
    split: str = "train",
    isaac_sim_version: str = "5.1.0",
    ros_distro: str | None = None,
    parquet_writer: ParquetWriter | None = None,
) -> dict[str, Any]:
    """Package randomized warehouse rollouts into an HF-friendly release folder."""

    raw_root_path = _resolve_dir(raw_root, "raw root")
    out_root_path = Path(out_root).expanduser()
    if not out_root_path.is_absolute():
        out_root_path = (Path.cwd() / out_root_path).resolve()

    if parquet_writer is None:
        _ensure_parquet_support()
        parquet_writer = _write_parquet

    discovery = discover_release_inputs(raw_root_path)
    out_root_path.mkdir(parents=True, exist_ok=True)
    (out_root_path / "metadata").mkdir(parents=True, exist_ok=True)
    (out_root_path / "scenes").mkdir(parents=True, exist_ok=True)
    (out_root_path / "rollouts").mkdir(parents=True, exist_ok=True)

    if (raw_root_path / "collection_metadata.yaml").is_file():
        shutil.copy2(raw_root_path / "collection_metadata.yaml", out_root_path / "collection_metadata.yaml")

    scenes_rows: list[dict[str, Any]] = []
    rollouts_rows: list[dict[str, Any]] = []
    frames_rows: list[dict[str, Any]] = []
    tar_actions: list[dict[str, Any]] = []

    for scene in discovery["scenes"]:
        scene_out_dir = out_root_path / "scenes" / scene_key(scene.scene_id)
        _copy_scene_files(scene.scene_dir, scene_out_dir, include_scene_usd=include_scene_usd)
        scene_manifest = _load_yaml(scene.scene_dir / "scene_manifest.yaml")
        team_config = _load_yaml(scene.scene_dir / "team_config.yaml")

        packaged_rollouts = 0
        scene_robot_models: set[str] = set()
        scene_robot_names: set[str] = set()
        scene_camera_names: set[str] = set()
        for rollout in scene.rollouts:
            rollout_rows = _build_rollout_metadata(
                raw_root_path=raw_root_path,
                rollout=rollout,
                split=split,
            )
            tar_path = out_root_path / rollout_rows["tar_path"]
            tar_action = _write_rollout_tar(
                rollout.rollout_dir,
                tar_path,
                append=append,
                overwrite=overwrite_rollout_tars,
            )
            tar_actions.append(
                {
                    "scene_id": rollout.scene_id,
                    "rollout_id": rollout.rollout_id,
                    "tar_path": rollout_rows["tar_path"],
                    "action": tar_action,
                }
            )
            packaged_rollouts += 1
            rollouts_rows.append(rollout_rows)
            frames = _build_frame_metadata(rollout=rollout, tar_path=rollout_rows["tar_path"])
            frames_rows.extend(frames)
            scene_robot_models.update(json.loads(rollout_rows["robot_models"]))
            scene_robot_names.update(json.loads(rollout_rows["robot_names"]))
            scene_camera_names.update(json.loads(rollout_rows["camera_names"]))

        scenes_rows.append(
            {
                "scene_id": scene.scene_id,
                "scene_name": scene.scene_dir.name,
                "scene_key": scene_key(scene.scene_id),
                "raw_scene_path": _relative_or_absolute(scene.scene_dir, raw_root_path),
                "scene_path": f"scenes/{scene_key(scene.scene_id)}",
                "mapf_map_path": f"scenes/{scene_key(scene.scene_id)}/mapf_map.yaml",
                "nav2_map_path": f"scenes/{scene_key(scene.scene_id)}/nav2_map.yaml",
                "scene_usd_path": (
                    f"scenes/{scene_key(scene.scene_id)}/scene.usd"
                    if include_scene_usd and (scene.scene_dir / "scene.usd").is_file()
                    else ""
                ),
                "team_config_path": f"scenes/{scene_key(scene.scene_id)}/team_config.yaml",
                "scene_manifest_path": f"scenes/{scene_key(scene.scene_id)}/scene_manifest.yaml",
                "instruction": str(
                    scene_manifest.get("language_instruction")
                    or team_config.get("language_instruction")
                    or ""
                ),
                "robot_names": _json_list(scene_robot_names),
                "robot_models": _json_list(scene_robot_models),
                "camera_names": _json_list(scene_camera_names),
                "num_packaged_rollouts": packaged_rollouts,
            }
        )

    parquet_writer(scenes_rows, out_root_path / "metadata" / METADATA_FILENAMES["scenes"])
    parquet_writer(rollouts_rows, out_root_path / "metadata" / METADATA_FILENAMES["rollouts"])
    parquet_writer(frames_rows, out_root_path / "metadata" / METADATA_FILENAMES["frames"])

    release = _build_release_summary(
        raw_root=raw_root_path,
        version=version,
        append=append,
        include_scene_usd=include_scene_usd,
        overwrite_rollout_tars=overwrite_rollout_tars,
        isaac_sim_version=isaac_sim_version,
        ros_distro=ros_distro or os.environ.get("ROS_DISTRO", "humble"),
        scenes_rows=scenes_rows,
        rollouts_rows=rollouts_rows,
        frames_rows=frames_rows,
        discovery=discovery,
        tar_actions=tar_actions,
    )
    _write_json(out_root_path / "metadata" / "dataset_release.json", release)
    _write_schema(out_root_path / "metadata" / "schema.md")
    _write_dataset_card(out_root_path / "README.md", release)
    _copy_load_example(out_root_path / "tools" / "load_example.py")
    return release


def discover_release_inputs(raw_root: str | Path) -> dict[str, Any]:
    root = _resolve_dir(raw_root, "raw root")
    scenes: list[SceneInput] = []
    skipped_scenes: list[dict[str, Any]] = []
    skipped_failed_rollouts: list[dict[str, Any]] = []
    skipped_rollouts: list[dict[str, Any]] = []

    for scene_dir in sorted(root.iterdir(), key=lambda path: _scene_sort_key(path.name)):
        if not scene_dir.is_dir():
            continue
        scene_id = _parse_scene_id(scene_dir.name)
        if scene_id is None:
            continue
        missing_scene_files = [
            name for name in REQUIRED_SCENE_FILES if not (scene_dir / name).is_file()
        ]
        if missing_scene_files:
            skipped_scenes.append(
                {
                    "scene": scene_dir.name,
                    "reason": "missing_scene_files",
                    "missing": missing_scene_files,
                }
            )
            continue

        rollouts_root = scene_dir / "rollouts"
        if not rollouts_root.is_dir():
            skipped_scenes.append(
                {"scene": scene_dir.name, "reason": "missing_rollouts_directory"}
            )
            continue

        rollouts: list[RolloutInput] = []
        for rollout_dir in sorted(rollouts_root.iterdir(), key=lambda path: _rollout_sort_key(path.name)):
            if not rollout_dir.is_dir():
                continue
            if rollout_dir.name.endswith("_failed"):
                skipped_failed_rollouts.append(
                    {
                        "scene_id": scene_id,
                        "scene": scene_dir.name,
                        "rollout_dir": rollout_dir.name,
                        "reason": "failed_rollout_directory",
                    }
                )
                continue
            try:
                rollout_id = int(rollout_dir.name)
            except ValueError:
                skipped_rollouts.append(
                    {
                        "scene_id": scene_id,
                        "scene": scene_dir.name,
                        "rollout_dir": rollout_dir.name,
                        "reason": "non_numeric_rollout_directory",
                    }
                )
                continue

            missing_items = [
                name for name in REQUIRED_ROLLOUT_ITEMS if not (rollout_dir / name).exists()
            ]
            if missing_items:
                skipped_rollouts.append(
                    {
                        "scene_id": scene_id,
                        "scene": scene_dir.name,
                        "rollout_id": rollout_id,
                        "rollout_dir": rollout_dir.name,
                        "reason": "missing_required_rollout_items",
                        "missing": missing_items,
                    }
                )
                continue
            rollouts.append(
                RolloutInput(
                    scene_id=scene_id,
                    scene_dir=scene_dir,
                    rollout_id=rollout_id,
                    rollout_dir=rollout_dir,
                )
            )

        if rollouts:
            scenes.append(
                SceneInput(
                    scene_id=scene_id,
                    scene_dir=scene_dir,
                    rollouts=tuple(rollouts),
                )
            )
        else:
            skipped_scenes.append({"scene": scene_dir.name, "reason": "no_valid_rollouts"})

    return {
        "root": str(root),
        "scenes": tuple(scenes),
        "skipped_scenes": skipped_scenes,
        "skipped_failed_rollouts": skipped_failed_rollouts,
        "skipped_rollouts": skipped_rollouts,
    }


def scene_key(scene_id: int) -> str:
    return f"scene_{scene_id:03d}"


def rollout_tar_name(rollout_id: int) -> str:
    return f"rollout_{rollout_id:03d}.tar"


def _build_rollout_metadata(
    *,
    raw_root_path: Path,
    rollout: RolloutInput,
    split: str,
) -> dict[str, Any]:
    run_config = _load_yaml(rollout.rollout_dir / "run_config.yaml")
    manifest_rows = _load_render_manifest(rollout.rollout_dir / "render_manifest.csv")
    timestamps = _manifest_timestamps(manifest_rows)
    camera_names = sorted({str(row.get("camera_name", "")) for row in manifest_rows if row.get("camera_name")})
    camera_types = sorted({str(row.get("camera_type", "")) for row in manifest_rows if row.get("camera_type")})
    robot_names, robot_models = _rollout_robot_names_and_models(run_config)
    tar_path = f"rollouts/{scene_key(rollout.scene_id)}/{rollout_tar_name(rollout.rollout_id)}"
    return {
        "scene_id": rollout.scene_id,
        "scene_name": rollout.scene_dir.name,
        "scene_key": scene_key(rollout.scene_id),
        "rollout_id": rollout.rollout_id,
        "rollout_key": f"rollout_{rollout.rollout_id:03d}",
        "split": split,
        "instruction": str(run_config.get("language_instruction", "") or ""),
        "tar_path": tar_path,
        "raw_rollout_path": _relative_or_absolute(rollout.rollout_dir, raw_root_path),
        "robot_names": _json_list(robot_names),
        "robot_models": _json_list(robot_models),
        "camera_names": _json_list(camera_names),
        "camera_types": _json_list(camera_types),
        "num_robots": len(robot_names),
        "num_cameras": len(camera_names),
        "num_manifest_rows": len(manifest_rows),
        "num_render_frames": len(timestamps),
        "num_rgb_frames": _count_manifest_paths(manifest_rows, "rgb_path"),
        "num_depth_frames": _count_manifest_paths(manifest_rows, "depth_path"),
        "start_timestamp_ns": min(timestamps) if timestamps else None,
        "end_timestamp_ns": max(timestamps) if timestamps else None,
        "duration_sec": _duration_seconds(timestamps),
        "has_render_manifest": True,
        "success": True,
        "package_status": "packaged",
    }


def _build_frame_metadata(*, rollout: RolloutInput, tar_path: str) -> list[dict[str, Any]]:
    manifest_rows = _load_render_manifest(rollout.rollout_dir / "render_manifest.csv")
    velocity_indexes = _load_velocity_indexes(rollout.rollout_dir)
    frames: list[dict[str, Any]] = []
    for row in manifest_rows:
        timestamp_ns = _optional_int(row.get("timestamp_ns"))
        camera_name = str(row.get("camera_name", "") or "")
        camera_type = str(row.get("camera_type", "") or "")
        nearest = (
            velocity_indexes.get(camera_name).nearest(timestamp_ns)
            if timestamp_ns is not None and camera_name in velocity_indexes
            else None
        )
        frames.append(
            {
                "scene_id": rollout.scene_id,
                "scene_name": rollout.scene_dir.name,
                "scene_key": scene_key(rollout.scene_id),
                "rollout_id": rollout.rollout_id,
                "rollout_key": f"rollout_{rollout.rollout_id:03d}",
                "tar_path": tar_path,
                "frame_index": _optional_int(row.get("frame_index")),
                "timestamp_ns": timestamp_ns,
                "elapsed_s": _optional_float(row.get("elapsed_s")),
                "camera_name": camera_name,
                "camera_type": camera_type,
                "camera_prim_path": str(row.get("camera_prim_path", "") or ""),
                "selection_mode": str(row.get("selection_mode", "") or ""),
                "rgb_path_in_tar": str(row.get("rgb_path", "") or ""),
                "depth_path_in_tar": str(row.get("depth_path", "") or ""),
                "robot_name": camera_name if camera_name in velocity_indexes else "",
                "nearest_velocity_timestamp_ns": nearest.timestamp_ns if nearest else None,
                "nearest_velocity_dt_ns": (
                    abs(nearest.timestamp_ns - timestamp_ns)
                    if nearest and timestamp_ns is not None
                    else None
                ),
                "vx": nearest.vx if nearest else None,
                "vy": nearest.vy if nearest else None,
                "wz": nearest.wz if nearest else None,
                "x": nearest.x if nearest else None,
                "y": nearest.y if nearest else None,
                "yaw": nearest.yaw if nearest else None,
                "cmd_vel_timestamp_ns": nearest.cmd_vel_timestamp_ns if nearest else None,
                "cmd_vx": nearest.cmd_vx if nearest else None,
                "cmd_vy": nearest.cmd_vy if nearest else None,
                "cmd_wz": nearest.cmd_wz if nearest else None,
            }
        )
    return frames


def _write_rollout_tar(
    rollout_dir: Path,
    tar_path: Path,
    *,
    append: bool,
    overwrite: bool,
) -> str:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{tar_path.stem}.",
        suffix=".tar",
        dir=tar_path.parent,
        delete=False,
    ) as temp_file:
        temp_tar = Path(temp_file.name)
    try:
        _create_deterministic_rollout_tar(rollout_dir, temp_tar)
        if tar_path.exists():
            if _sha256_file(tar_path) == _sha256_file(temp_tar):
                temp_tar.unlink()
                return "reused_existing"
            if not overwrite:
                mode = "append" if append else "package"
                raise PackageError(
                    f"Existing rollout tar differs during {mode} mode: {tar_path}. "
                    "Use --overwrite-rollout-tars only if you intend to replace it."
                )
            tar_path.unlink()
            shutil.move(str(temp_tar), str(tar_path))
            return "overwritten"
        shutil.move(str(temp_tar), str(tar_path))
        return "created"
    finally:
        if temp_tar.exists():
            temp_tar.unlink()


def _create_deterministic_rollout_tar(rollout_dir: Path, tar_path: Path) -> None:
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tar:
        for source_path, arcname in _iter_rollout_tar_files(rollout_dir):
            info = tar.gettarinfo(str(source_path), arcname=arcname)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with source_path.open("rb") as stream:
                tar.addfile(info, stream)


def _iter_rollout_tar_files(rollout_dir: Path) -> list[tuple[Path, str]]:
    files: list[Path] = [
        rollout_dir / "render_manifest.csv",
        rollout_dir / "run_config.yaml",
    ]
    files.extend(sorted(rollout_dir.glob("*_velocity.csv")))
    for folder_name in ("rgb", "depth"):
        files.extend(sorted(path for path in (rollout_dir / folder_name).rglob("*") if path.is_file()))
    return [
        (path, path.relative_to(rollout_dir).as_posix())
        for path in sorted(files, key=lambda item: item.relative_to(rollout_dir).as_posix())
    ]


def _copy_scene_files(scene_dir: Path, scene_out_dir: Path, *, include_scene_usd: bool) -> None:
    scene_out_dir.mkdir(parents=True, exist_ok=True)
    names = list(REQUIRED_SCENE_FILES)
    if include_scene_usd:
        names.append("scene.usd")
    for name in names:
        source = scene_dir / name
        if source.is_file():
            shutil.copy2(source, scene_out_dir / name)


def _load_velocity_indexes(rollout_dir: Path) -> dict[str, VelocityIndex]:
    indexes: dict[str, VelocityIndex] = {}
    for velocity_path in sorted(rollout_dir.glob("*_velocity.csv")):
        robot_name = velocity_path.name[: -len("_velocity.csv")]
        indexes[robot_name] = VelocityIndex(_load_velocity_samples(velocity_path))
    return indexes


def _load_velocity_samples(velocity_path: Path) -> list[VelocitySample]:
    with velocity_path.open("r", encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        samples: list[VelocitySample] = []
        for row in rows:
            timestamp_ns = _optional_int(row.get("timestamp_ns"))
            if timestamp_ns is None:
                continue
            samples.append(
                VelocitySample(
                    timestamp_ns=timestamp_ns,
                    vx=_optional_float(row.get("vx")),
                    vy=_optional_float(row.get("vy")),
                    wz=_optional_float(row.get("wz")),
                    x=_optional_float(row.get("x")),
                    y=_optional_float(row.get("y")),
                    yaw=_optional_float(row.get("yaw")),
                    cmd_vel_timestamp_ns=_optional_int(row.get("cmd_vel_timestamp_ns")),
                    cmd_vx=_optional_float(row.get("cmd_vx")),
                    cmd_vy=_optional_float(row.get("cmd_vy")),
                    cmd_wz=_optional_float(row.get("cmd_wz")),
                )
            )
    return samples


def _load_render_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"frame_index", "timestamp_ns", "camera_name", "rgb_path", "depth_path"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise PackageError(f"{manifest_path} is missing required columns: {missing}")
        return [dict(row) for row in reader]


def _rollout_robot_names_and_models(run_config: dict[str, Any]) -> tuple[list[str], list[str]]:
    robots = list(((run_config.get("team_config") or {}).get("robots") or []))
    names: list[str] = []
    models: list[str] = []
    for robot in robots:
        name = str(robot.get("name", "") or "").strip()
        model = str(robot.get("model", "") or name).strip()
        if name:
            names.append(name)
        if model:
            models.append(model)
    return sorted(set(names)), sorted(set(models))


def _manifest_timestamps(rows: list[dict[str, str]]) -> list[int]:
    timestamps = {
        timestamp
        for row in rows
        for timestamp in [_optional_int(row.get("timestamp_ns"))]
        if timestamp is not None
    }
    return sorted(timestamps)


def _count_manifest_paths(rows: list[dict[str, str]], key: str) -> int:
    return sum(1 for row in rows if str(row.get(key, "") or "").strip())


def _duration_seconds(timestamps_ns: list[int]) -> float | None:
    if len(timestamps_ns) < 2:
        return 0.0 if timestamps_ns else None
    return (timestamps_ns[-1] - timestamps_ns[0]) / 1_000_000_000.0


def _build_release_summary(
    *,
    raw_root: Path,
    version: str,
    append: bool,
    include_scene_usd: bool,
    overwrite_rollout_tars: bool,
    isaac_sim_version: str,
    ros_distro: str,
    scenes_rows: list[dict[str, Any]],
    rollouts_rows: list[dict[str, Any]],
    frames_rows: list[dict[str, Any]],
    discovery: dict[str, Any],
    tar_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "dataset_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "source_github_repo": "mas-vln",
        "collection_code_commit": _git_commit(),
        "packaging_code_commit": _git_commit(),
        "isaac_sim_version": isaac_sim_version,
        "ros_distro": ros_distro,
        "append": append,
        "include_scene_usd": include_scene_usd,
        "overwrite_rollout_tars": overwrite_rollout_tars,
        "counts": {
            "scenes": len(scenes_rows),
            "rollouts": len(rollouts_rows),
            "frames": len(frames_rows),
            "skipped_scenes": len(discovery["skipped_scenes"]),
            "skipped_failed_rollouts": len(discovery["skipped_failed_rollouts"]),
            "skipped_rollouts": len(discovery["skipped_rollouts"]),
            "created_tars": sum(1 for item in tar_actions if item["action"] == "created"),
            "reused_existing_tars": sum(
                1 for item in tar_actions if item["action"] == "reused_existing"
            ),
            "overwritten_tars": sum(1 for item in tar_actions if item["action"] == "overwritten"),
        },
        "tar_actions": tar_actions,
        "skipped": {
            "scenes": discovery["skipped_scenes"],
            "failed_rollouts": discovery["skipped_failed_rollouts"],
            "rollouts": discovery["skipped_rollouts"],
        },
    }


def _write_dataset_card(path: Path, release: dict[str, Any]) -> None:
    counts = release["counts"]
    text = f"""---
pretty_name: MAS-VLN Randomized Warehouse RGBD
task_categories:
- robotics
- visual-question-answering
tags:
- isaac-sim
- ros2
- rgbd
- multi-robot-navigation
---

# MAS-VLN Randomized Warehouse RGBD

This dataset contains Isaac Sim randomized warehouse multi-robot rollouts packaged
for Hugging Face release. RGB and metric depth frames are stored inside one plain
tar file per rollout. Metadata tables index scenes, rollouts, and rendered camera
frames.

## Release

- Dataset version: `{release["dataset_version"]}`
- Created at: `{release["created_at"]}`
- Isaac Sim: `{release["isaac_sim_version"]}`
- ROS distro: `{release["ros_distro"]}`
- Scenes: {counts["scenes"]}
- Rollouts: {counts["rollouts"]}
- Frame rows: {counts["frames"]}

## Layout

- `metadata/scenes.parquet`: scene-level metadata.
- `metadata/rollouts.parquet`: one row per packaged rollout.
- `metadata/frames.parquet`: one row per rendered camera frame.
- `scenes/`: map/config/manifest files copied once per scene.
- `rollouts/`: one tar per successful rollout.

Depth PNGs are encoded as uint16 millimeters by the rendering pipeline. Large
rollout tar files should be treated as immutable after release; future updates
append new tars and regenerate the small metadata tables.
"""
    path.write_text(text, encoding="utf-8")


def _write_schema(path: Path) -> None:
    path.write_text(
        """# Dataset Schema

## metadata/scenes.parquet

One row per packaged scene. List-like fields are JSON strings for broad Parquet
reader compatibility.

## metadata/rollouts.parquet

One row per packaged rollout. `tar_path` points to a plain tar file under
`rollouts/scene_XXX/rollout_YYY.tar`.

## metadata/frames.parquet

One row per rendered camera frame from `render_manifest.csv`. `rgb_path_in_tar`
and `depth_path_in_tar` point to files inside `tar_path`. For robot cameras, the
nearest recorded robot pose and cmd_vel sample are included when the matching
`<robot>_velocity.csv` exists.
""",
        encoding="utf-8",
    )


def _copy_load_example(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("load_example.py")
    if source.is_file():
        shutil.copy2(source, destination)


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Writing Parquet metadata requires pyarrow or fastparquet. "
            "Install one in the release environment, e.g. `pip install pyarrow`."
        ) from exc


def _ensure_parquet_support() -> None:
    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("Packaging metadata requires pandas.") from exc
    try:
        import pyarrow  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    try:
        import fastparquet  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Packaging metadata requires a Parquet engine. Install pyarrow "
            "or fastparquet, e.g. `pip install pyarrow`."
        ) from exc


def _resolve_dir(path_value: str | Path, label: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream) or {})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(sorted({str(value) for value in values if str(value)}))


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _parse_scene_id(name: str) -> int | None:
    if not name.startswith("scene_"):
        return None
    suffix = name[len("scene_") :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _scene_sort_key(name: str) -> tuple[int, str]:
    scene_id = _parse_scene_id(name)
    return (scene_id if scene_id is not None else 10**9, name)


def _rollout_sort_key(name: str) -> tuple[int, str]:
    try:
        return (int(name), name)
    except ValueError:
        return (10**9, name)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package randomized warehouse RGBD rollouts for Hugging Face release."
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Path to experiments/randomized_warehouse.",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output folder for the HF-ready dataset.",
    )
    parser.add_argument("--version", required=True, help="Dataset release version, e.g. v0.1.0.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new rollout tars while treating existing tars as immutable.",
    )
    parser.add_argument(
        "--include-scene-usd",
        action="store_true",
        help="Copy scene.usd into scene metadata folders. Disabled by default for licensing caution.",
    )
    parser.add_argument(
        "--overwrite-rollout-tars",
        action="store_true",
        help="Replace existing rollout tar files when their contents differ.",
    )
    parser.add_argument("--split", default="train", help="Dataset split assigned to packaged rollouts.")
    parser.add_argument("--isaac-sim-version", default="5.1.0")
    parser.add_argument("--ros-distro", default=os.environ.get("ROS_DISTRO", "humble"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        release = package_randomized_warehouse(
            raw_root=args.raw_root,
            out_root=args.out_root,
            version=args.version,
            append=args.append,
            include_scene_usd=args.include_scene_usd,
            overwrite_rollout_tars=args.overwrite_rollout_tars,
            split=args.split,
            isaac_sim_version=args.isaac_sim_version,
            ros_distro=args.ros_distro,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    counts = release["counts"]
    print(
        "Packaged randomized warehouse dataset: "
        f"{counts['scenes']} scenes, {counts['rollouts']} rollouts, "
        f"{counts['frames']} frame rows."
    )
    if counts["skipped_failed_rollouts"] or counts["skipped_rollouts"] or counts["skipped_scenes"]:
        print(
            "Skipped: "
            f"{counts['skipped_scenes']} scenes, "
            f"{counts['skipped_failed_rollouts']} failed rollout dirs, "
            f"{counts['skipped_rollouts']} rollout dirs."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
