from __future__ import annotations

import csv
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RobotPose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class VelocitySample:
    timestamp_ns: int
    vx: float
    vy: float
    wz: float


@dataclass(frozen=True)
class TrackerSample:
    elapsed_s: float
    actual_x: float
    actual_y: float
    actual_yaw: float


@dataclass(frozen=True)
class RolloutRobotData:
    name: str
    initial_pose: RobotPose
    goal_pose: RobotPose
    velocity_path: Path
    velocity_samples: tuple[VelocitySample, ...]
    tracker_path: Path | None
    tracker_samples: tuple[TrackerSample, ...]


@dataclass(frozen=True)
class RolloutData:
    rollout_id: int
    rollout_dir: Path
    run_config_path: Path
    created_at: str
    language_instruction: str
    team_config_snapshot: dict[str, Any]
    robots: tuple[RolloutRobotData, ...]
    replay_timestamps_ns: tuple[int, ...]


def resolve_experiments_root(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def discover_rollout_dirs(
    experiments_root: str | Path,
    rollout_ids: list[int] | None = None,
) -> list[Path]:
    root = resolve_experiments_root(experiments_root)
    if not root.exists():
        raise FileNotFoundError(f"Experiments root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Experiments root is not a directory: {root}")

    requested_ids = set(rollout_ids or [])
    discovered: list[tuple[int, Path]] = []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        if not candidate.name.isdigit():
            continue
        rollout_id = int(candidate.name)
        if requested_ids and rollout_id not in requested_ids:
            continue
        if not (candidate / "run_config.yaml").is_file():
            continue
        discovered.append((rollout_id, candidate.resolve()))

    discovered.sort(key=lambda item: item[0])
    found_ids = {rollout_id for rollout_id, _ in discovered}
    missing_ids = sorted(requested_ids.difference(found_ids))
    if missing_ids:
        missing_label = ", ".join(str(value) for value in missing_ids)
        raise FileNotFoundError(
            f"Requested rollout ids were not found under {root}: {missing_label}"
        )
    return [path for _, path in discovered]


def load_rollouts(
    experiments_root: str | Path,
    rollout_ids: list[int] | None = None,
) -> list[RolloutData]:
    rollout_dirs = discover_rollout_dirs(experiments_root, rollout_ids=rollout_ids)
    if not rollout_dirs:
        raise FileNotFoundError(
            f"No rollout directories with run_config.yaml were found under {resolve_experiments_root(experiments_root)}."
        )
    return [load_rollout(rollout_dir) for rollout_dir in rollout_dirs]


def load_rollout(rollout_dir: str | Path) -> RolloutData:
    rollout_path = Path(rollout_dir).expanduser().resolve()
    run_config_path = rollout_path / "run_config.yaml"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing rollout metadata file: {run_config_path}")

    with run_config_path.open("r", encoding="utf-8") as stream:
        run_config = yaml.safe_load(stream) or {}

    team_config_snapshot = dict(run_config.get("team_config", {}) or {})
    robot_configs = list(team_config_snapshot.get("robots", []) or [])
    if not robot_configs:
        raise ValueError(f"No robots are defined in {run_config_path}.")

    robots: list[RolloutRobotData] = []
    replay_timestamps: set[int] = set()
    for robot_config in robot_configs:
        name = str(robot_config.get("name", "")).strip()
        if not name:
            raise ValueError(f"Encountered a robot entry without a name in {run_config_path}.")

        velocity_path = rollout_path / f"{name}_velocity.csv"
        if not velocity_path.is_file():
            raise FileNotFoundError(f"Missing velocity log for {name}: {velocity_path}")
        velocity_samples = tuple(_load_velocity_samples(velocity_path))
        replay_timestamps.update(sample.timestamp_ns for sample in velocity_samples)

        tracker_path, tracker_samples = _load_tracker_samples(rollout_path, name)
        robots.append(
            RolloutRobotData(
                name=name,
                initial_pose=_pose_from_config(robot_config.get("initial_pose"), f"{name}.initial_pose"),
                goal_pose=_pose_from_config(robot_config.get("goal_pose"), f"{name}.goal_pose"),
                velocity_path=velocity_path,
                velocity_samples=velocity_samples,
                tracker_path=tracker_path,
                tracker_samples=tracker_samples,
            )
        )

    rollout_id = _resolve_rollout_id(run_config.get("run_id"), rollout_path)
    return RolloutData(
        rollout_id=rollout_id,
        rollout_dir=rollout_path,
        run_config_path=run_config_path,
        created_at=str(run_config.get("created_at", "") or ""),
        language_instruction=str(run_config.get("language_instruction", "") or ""),
        team_config_snapshot=team_config_snapshot,
        robots=tuple(robots),
        replay_timestamps_ns=tuple(sorted(replay_timestamps)),
    )


@contextmanager
def temporary_team_config_file(rollout: RolloutData) -> Iterator[Path]:
    environment = dict(rollout.team_config_snapshot.get("environment", {}) or {})
    payload = {
        "language_instruction": rollout.language_instruction,
        "environment": environment,
        "robots": [
            {
                "name": robot.name,
                "initial_pose": pose_to_dict(robot.initial_pose),
                "goal_pose": pose_to_dict(robot.goal_pose),
            }
            for robot in rollout.robots
        ],
    }

    file_descriptor, raw_path = tempfile.mkstemp(
        prefix=f"rollout_{rollout.rollout_id:03d}_",
        suffix=".yaml",
    )
    temp_path = Path(raw_path)
    try:
        os.close(file_descriptor)
        with temp_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
        yield temp_path
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def pose_to_dict(pose: RobotPose) -> dict[str, float]:
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "yaw": float(pose.yaw),
    }


def replay_elapsed_seconds(timestamps_ns: tuple[int, ...] | list[int]) -> list[float]:
    if not timestamps_ns:
        return []
    first_timestamp_ns = int(timestamps_ns[0])
    return [(int(timestamp_ns) - first_timestamp_ns) * 1e-9 for timestamp_ns in timestamps_ns]


def _pose_from_config(pose_config: Any, pose_label: str) -> RobotPose:
    if not isinstance(pose_config, dict):
        raise ValueError(f"{pose_label} must be a mapping with x/y/z/yaw fields.")
    return RobotPose(
        x=float(pose_config.get("x", 0.0)),
        y=float(pose_config.get("y", 0.0)),
        z=float(pose_config.get("z", 0.0)),
        yaw=float(pose_config.get("yaw", 0.0)),
    )


def _resolve_rollout_id(run_id_value: Any, rollout_path: Path) -> int:
    if rollout_path.name.isdigit():
        return int(rollout_path.name)
    if run_id_value is None:
        raise ValueError(
            f"Unable to determine a numeric rollout id for {rollout_path}: directory name is not numeric."
        )
    try:
        return int(run_id_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unable to parse rollout id from run_id={run_id_value!r} in {rollout_path / 'run_config.yaml'}."
        ) from exc


def _load_velocity_samples(csv_path: Path) -> list[VelocitySample]:
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty or missing a header row.")

        required_columns = {"timestamp_ns", "vx", "vy", "wz"}
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            missing_label = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required columns: {missing_label}")

        samples = [
            VelocitySample(
                timestamp_ns=int(row["timestamp_ns"]),
                vx=float(row["vx"]),
                vy=float(row["vy"]),
                wz=float(row["wz"]),
            )
            for row in reader
        ]
    if not samples:
        raise ValueError(f"{csv_path} has no velocity data rows.")
    return samples


def _load_tracker_samples(
    rollout_dir: Path,
    robot_name: str,
) -> tuple[Path | None, tuple[TrackerSample, ...]]:
    matches = sorted(rollout_dir.glob(f"mapf_timed_tracker_*_{robot_name}.csv"))
    if not matches:
        return None, ()
    if len(matches) > 1:
        match_list = ", ".join(path.name for path in matches)
        raise ValueError(
            f"Expected at most one tracker CSV for {robot_name} in {rollout_dir}, found: {match_list}"
        )

    csv_path = matches[0]
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty or missing a header row.")

        required_columns = {"elapsed", "actual_x", "actual_y", "actual_yaw"}
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            missing_label = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required columns: {missing_label}")

        samples = tuple(
            TrackerSample(
                elapsed_s=float(row["elapsed"]),
                actual_x=float(row["actual_x"]),
                actual_y=float(row["actual_y"]),
                actual_yaw=float(row["actual_yaw"]),
            )
            for row in reader
        )
    return csv_path, samples
