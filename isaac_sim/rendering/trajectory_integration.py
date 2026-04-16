from __future__ import annotations

import math
import warnings
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable, Sequence

try:
    from .rollout_io import RobotPose, VelocitySample
except ImportError:
    from rollout_io import RobotPose, VelocitySample


SMALL_ANGULAR_VELOCITY_RAD_PER_SEC = 1e-6


@dataclass(frozen=True)
class TimedPose:
    timestamp_ns: int
    x: float
    y: float
    z: float
    yaw: float

    @property
    def pose(self) -> RobotPose:
        return RobotPose(x=self.x, y=self.y, z=self.z, yaw=self.yaw)


@dataclass(frozen=True)
class IntegratedTrajectory:
    source_label: str
    initial_pose: RobotPose
    velocity_samples: tuple[VelocitySample, ...]
    keyframes: tuple[TimedPose, ...]

    @property
    def first_timestamp_ns(self) -> int:
        return self.keyframes[0].timestamp_ns

    @property
    def last_timestamp_ns(self) -> int:
        return self.keyframes[-1].timestamp_ns

    def pose_at(self, timestamp_ns: int) -> TimedPose:
        if timestamp_ns <= self.first_timestamp_ns:
            return TimedPose(
                timestamp_ns=int(timestamp_ns),
                x=self.initial_pose.x,
                y=self.initial_pose.y,
                z=self.initial_pose.z,
                yaw=self.initial_pose.yaw,
            )
        if timestamp_ns >= self.last_timestamp_ns:
            final_pose = self.keyframes[-1]
            return TimedPose(
                timestamp_ns=int(timestamp_ns),
                x=final_pose.x,
                y=final_pose.y,
                z=final_pose.z,
                yaw=final_pose.yaw,
            )

        keyframe_timestamps = [pose.timestamp_ns for pose in self.keyframes]
        segment_index = bisect_right(keyframe_timestamps, int(timestamp_ns)) - 1
        segment_index = max(0, min(segment_index, len(self.velocity_samples) - 1))
        start_pose = self.keyframes[segment_index]
        sample = self.velocity_samples[segment_index]
        delta_t_sec = (int(timestamp_ns) - start_pose.timestamp_ns) * 1e-9
        return integrate_step(
            start_pose,
            sample.vx,
            sample.vy,
            sample.wz,
            delta_t_sec,
            timestamp_ns=int(timestamp_ns),
        )

    def sample(self, timestamps_ns: Sequence[int]) -> list[TimedPose]:
        return [self.pose_at(int(timestamp_ns)) for timestamp_ns in timestamps_ns]


def integrate_velocity_samples(
    initial_pose: RobotPose,
    velocity_samples: Sequence[VelocitySample],
    *,
    source_label: str,
) -> IntegratedTrajectory:
    samples = sanitize_velocity_samples(velocity_samples, source_label=source_label)
    keyframes: list[TimedPose] = [
        TimedPose(
            timestamp_ns=samples[0].timestamp_ns,
            x=initial_pose.x,
            y=initial_pose.y,
            z=initial_pose.z,
            yaw=wrap_to_pi(initial_pose.yaw),
        )
    ]

    current_pose = keyframes[0]
    for index in range(len(samples) - 1):
        current_sample = samples[index]
        next_sample = samples[index + 1]
        delta_t_sec = (next_sample.timestamp_ns - current_sample.timestamp_ns) * 1e-9
        current_pose = integrate_step(
            current_pose,
            current_sample.vx,
            current_sample.vy,
            current_sample.wz,
            delta_t_sec,
            timestamp_ns=next_sample.timestamp_ns,
        )
        keyframes.append(current_pose)

    return IntegratedTrajectory(
        source_label=source_label,
        initial_pose=RobotPose(
            x=initial_pose.x,
            y=initial_pose.y,
            z=initial_pose.z,
            yaw=wrap_to_pi(initial_pose.yaw),
        ),
        velocity_samples=tuple(samples),
        keyframes=tuple(keyframes),
    )


def sanitize_velocity_samples(
    velocity_samples: Sequence[VelocitySample],
    *,
    source_label: str,
) -> list[VelocitySample]:
    if not velocity_samples:
        raise ValueError(f"{source_label} contains no velocity samples.")

    raw_samples = list(velocity_samples)
    timestamps = [sample.timestamp_ns for sample in raw_samples]
    if timestamps != sorted(timestamps):
        warnings.warn(
            f"{source_label} has non-monotonic timestamps; sorting samples before integration.",
            stacklevel=2,
        )

    deduped_by_timestamp: dict[int, VelocitySample] = {}
    duplicate_count = 0
    for sample in raw_samples:
        if sample.timestamp_ns in deduped_by_timestamp:
            duplicate_count += 1
        deduped_by_timestamp[sample.timestamp_ns] = sample

    if duplicate_count:
        warnings.warn(
            f"{source_label} has {duplicate_count} duplicate timestamps; keeping the last sample for each timestamp.",
            stacklevel=2,
        )

    samples = [deduped_by_timestamp[timestamp_ns] for timestamp_ns in sorted(deduped_by_timestamp)]
    if not samples:
        raise ValueError(f"{source_label} has no usable velocity samples after sanitization.")
    return samples


def sample_trajectories_on_timestamps(
    trajectories: dict[str, IntegratedTrajectory],
    timestamps_ns: Sequence[int],
) -> dict[str, list[TimedPose]]:
    return {
        robot_name: trajectory.sample(timestamps_ns)
        for robot_name, trajectory in trajectories.items()
    }


def integrate_step(
    pose: TimedPose | RobotPose,
    vx: float,
    vy: float,
    wz: float,
    delta_t_sec: float,
    *,
    timestamp_ns: int,
) -> TimedPose:
    if delta_t_sec < 0.0:
        raise ValueError(f"Integration delta_t_sec must be non-negative, got {delta_t_sec}.")

    start_pose = pose.pose if isinstance(pose, TimedPose) else pose
    theta = float(wz) * float(delta_t_sec)
    if abs(float(wz)) < SMALL_ANGULAR_VELOCITY_RAD_PER_SEC:
        body_dx = float(vx) * float(delta_t_sec)
        body_dy = float(vy) * float(delta_t_sec)
    else:
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        body_dx = (sin_theta * float(vx) - (1.0 - cos_theta) * float(vy)) / float(wz)
        body_dy = ((1.0 - cos_theta) * float(vx) + sin_theta * float(vy)) / float(wz)

    cos_yaw = math.cos(start_pose.yaw)
    sin_yaw = math.sin(start_pose.yaw)
    world_dx = cos_yaw * body_dx - sin_yaw * body_dy
    world_dy = sin_yaw * body_dx + cos_yaw * body_dy
    return TimedPose(
        timestamp_ns=int(timestamp_ns),
        x=start_pose.x + world_dx,
        y=start_pose.y + world_dy,
        z=start_pose.z,
        yaw=wrap_to_pi(start_pose.yaw + theta),
    )


def wrap_to_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def translation_errors_m(
    poses_a: Iterable[TimedPose | RobotPose],
    poses_b: Iterable[TimedPose | RobotPose],
) -> list[float]:
    errors: list[float] = []
    for pose_a, pose_b in zip(poses_a, poses_b):
        a = pose_a.pose if isinstance(pose_a, TimedPose) else pose_a
        b = pose_b.pose if isinstance(pose_b, TimedPose) else pose_b
        errors.append(math.hypot(a.x - b.x, a.y - b.y))
    return errors


def yaw_errors_rad(
    poses_a: Iterable[TimedPose | RobotPose],
    poses_b: Iterable[TimedPose | RobotPose],
) -> list[float]:
    errors: list[float] = []
    for pose_a, pose_b in zip(poses_a, poses_b):
        a = pose_a.pose if isinstance(pose_a, TimedPose) else pose_a
        b = pose_b.pose if isinstance(pose_b, TimedPose) else pose_b
        errors.append(wrap_to_pi(a.yaw - b.yaw))
    return errors
