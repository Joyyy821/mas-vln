#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from .rollout_io import RobotPose, RolloutData, TrackerSample, load_rollouts
    from .trajectory_integration import (
        IntegratedTrajectory,
        TimedPose,
        integrate_velocity_samples,
        wrap_to_pi,
    )
except ImportError:
    from rollout_io import RobotPose, RolloutData, TrackerSample, load_rollouts
    from trajectory_integration import (
        IntegratedTrajectory,
        TimedPose,
        integrate_velocity_samples,
        wrap_to_pi,
    )


ALIGNMENT_WINDOW_S = 2.0
ALIGNMENT_MAX_SAMPLES = 20


@dataclass(frozen=True)
class RobotEvaluationResult:
    robot_name: str
    status: str
    tracker_path: str | None
    aligned_start_timestamp_ns: int | None
    sample_count: int
    duration_s: float
    translation_ate_rmse_m: float | None
    translation_ate_mae_m: float | None
    translation_ate_max_m: float | None
    yaw_rmse_rad: float | None
    yaw_mae_rad: float | None
    yaw_max_rad: float | None
    final_translation_drift_m: float | None
    final_yaw_drift_rad: float | None


@dataclass(frozen=True)
class AlignedComparison:
    elapsed_s: float
    timestamp_ns: int
    integrated_pose: TimedPose
    actual_pose: RobotPose
    translation_error_m: float
    yaw_error_rad: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate integrated trajectories against optional MAPF timed-tracker CSV logs "
            "stored beside each rollout."
        )
    )
    parser.add_argument(
        "--experiments-root",
        required=True,
        help="Directory containing rollout subdirectories with run_config.yaml files.",
    )
    parser.add_argument(
        "--rollout-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional rollout ids to evaluate. Defaults to every rollout under the experiments root.",
    )
    return parser.parse_args()


def _import_matplotlib() -> tuple[object, bool]:
    try:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str((Path(tempfile.gettempdir()) / "mas_vln_matplotlib").resolve()),
        )
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import matplotlib

        display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        headless = sys.platform.startswith("linux") and not display_available
        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for velocity integration evaluation. Install python3-matplotlib."
        ) from exc
    return plt, headless


def _fit_xy_axis(axis, x_values: list[float], y_values: list[float]) -> None:
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    span_x = max(max_x - min_x, 0.1)
    span_y = max(max_y - min_y, 0.1)
    margin_x = max(0.05, span_x * 0.12)
    margin_y = max(0.05, span_y * 0.12)
    axis.set_xlim(min_x - margin_x, max_x + margin_x)
    axis.set_ylim(min_y - margin_y, max_y + margin_y)
    axis.set_aspect("auto")


def _rmse(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _mae(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _max_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(abs(value) for value in values)


def _actual_pose_from_tracker_sample(sample: TrackerSample, z_value: float) -> RobotPose:
    return RobotPose(
        x=sample.actual_x,
        y=sample.actual_y,
        z=z_value,
        yaw=sample.actual_yaw,
    )


def _estimate_alignment_start_timestamp_ns(
    trajectory: IntegratedTrajectory,
    tracker_samples: tuple[TrackerSample, ...],
    z_value: float,
) -> int:
    first_elapsed_s = tracker_samples[0].elapsed_s
    window_samples = list(tracker_samples[:ALIGNMENT_MAX_SAMPLES])
    if window_samples:
        window_limit_s = window_samples[0].elapsed_s + ALIGNMENT_WINDOW_S
        window_samples = [
            sample
            for sample in window_samples
            if sample.elapsed_s <= window_limit_s
        ]
    if not window_samples:
        window_samples = [tracker_samples[0]]

    relative_offsets_ns = [
        int(round((sample.elapsed_s - first_elapsed_s) * 1e9))
        for sample in window_samples
    ]
    candidate_timestamps = [pose.timestamp_ns for pose in trajectory.keyframes]
    best_score = float("inf")
    best_timestamp_ns = candidate_timestamps[0]
    for candidate_timestamp_ns in candidate_timestamps:
        if candidate_timestamp_ns + relative_offsets_ns[-1] > trajectory.last_timestamp_ns:
            continue

        translation_errors: list[float] = []
        yaw_errors: list[float] = []
        for sample, relative_offset_ns in zip(window_samples, relative_offsets_ns):
            predicted_pose = trajectory.pose_at(candidate_timestamp_ns + relative_offset_ns)
            actual_pose = _actual_pose_from_tracker_sample(sample, z_value)
            translation_errors.append(
                math.hypot(predicted_pose.x - actual_pose.x, predicted_pose.y - actual_pose.y)
            )
            yaw_errors.append(wrap_to_pi(predicted_pose.yaw - actual_pose.yaw))

        score = _rmse(translation_errors) + 0.25 * _rmse(yaw_errors)
        if score < best_score:
            best_score = score
            best_timestamp_ns = candidate_timestamp_ns

    return best_timestamp_ns


def _build_aligned_comparisons(
    trajectory: IntegratedTrajectory,
    tracker_samples: tuple[TrackerSample, ...],
    z_value: float,
    aligned_start_timestamp_ns: int,
) -> list[AlignedComparison]:
    first_elapsed_s = tracker_samples[0].elapsed_s
    comparisons: list[AlignedComparison] = []
    for sample in tracker_samples:
        timestamp_ns = aligned_start_timestamp_ns + int(round((sample.elapsed_s - first_elapsed_s) * 1e9))
        if timestamp_ns > trajectory.last_timestamp_ns:
            break
        if timestamp_ns < trajectory.first_timestamp_ns:
            continue

        integrated_pose = trajectory.pose_at(timestamp_ns)
        actual_pose = _actual_pose_from_tracker_sample(sample, z_value)
        translation_error_m = math.hypot(
            integrated_pose.x - actual_pose.x,
            integrated_pose.y - actual_pose.y,
        )
        yaw_error_rad = wrap_to_pi(integrated_pose.yaw - actual_pose.yaw)
        comparisons.append(
            AlignedComparison(
                elapsed_s=sample.elapsed_s - first_elapsed_s,
                timestamp_ns=timestamp_ns,
                integrated_pose=integrated_pose,
                actual_pose=actual_pose,
                translation_error_m=translation_error_m,
                yaw_error_rad=yaw_error_rad,
            )
        )
    return comparisons


def _write_comparison_csv(output_path: Path, comparisons: list[AlignedComparison]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "elapsed_s",
                "timestamp_ns",
                "integrated_x",
                "integrated_y",
                "integrated_yaw",
                "actual_x",
                "actual_y",
                "actual_yaw",
                "translation_error_m",
                "yaw_error_rad",
            ]
        )
        for comparison in comparisons:
            writer.writerow(
                [
                    f"{comparison.elapsed_s:.9f}",
                    comparison.timestamp_ns,
                    f"{comparison.integrated_pose.x:.9f}",
                    f"{comparison.integrated_pose.y:.9f}",
                    f"{comparison.integrated_pose.yaw:.9f}",
                    f"{comparison.actual_pose.x:.9f}",
                    f"{comparison.actual_pose.y:.9f}",
                    f"{comparison.actual_pose.yaw:.9f}",
                    f"{comparison.translation_error_m:.9f}",
                    f"{comparison.yaw_error_rad:.9f}",
                ]
            )


def _plot_trajectory_overlay(robot_name: str, comparisons: list[AlignedComparison], output_path: Path, plt) -> None:
    integrated_x = [comparison.integrated_pose.x for comparison in comparisons]
    integrated_y = [comparison.integrated_pose.y for comparison in comparisons]
    actual_x = [comparison.actual_pose.x for comparison in comparisons]
    actual_y = [comparison.actual_pose.y for comparison in comparisons]

    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    axis.plot(integrated_x, integrated_y, label="integrated", color="tab:blue", linewidth=2.0)
    axis.plot(actual_x, actual_y, label="actual", color="tab:orange", linewidth=2.0)
    axis.scatter(integrated_x[0], integrated_y[0], color="tab:blue", marker="o", s=40, label="integrated start")
    axis.scatter(integrated_x[-1], integrated_y[-1], color="tab:blue", marker="X", s=55, label="integrated end")
    axis.scatter(actual_x[0], actual_y[0], color="tab:orange", marker="o", s=40, label="actual start")
    axis.scatter(actual_x[-1], actual_y[-1], color="tab:orange", marker="X", s=55, label="actual end")
    axis.set_title(f"{robot_name} Integrated vs Actual Trajectory")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    _fit_xy_axis(axis, integrated_x + actual_x, integrated_y + actual_y)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_error_timeseries(robot_name: str, comparisons: list[AlignedComparison], output_path: Path, plt) -> None:
    elapsed_s = [comparison.elapsed_s for comparison in comparisons]
    translation_errors = [comparison.translation_error_m for comparison in comparisons]
    yaw_errors = [comparison.yaw_error_rad for comparison in comparisons]

    figure, (axis_translation, axis_yaw) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        constrained_layout=True,
        sharex=True,
    )
    axis_translation.plot(elapsed_s, translation_errors, color="tab:red", linewidth=2.0)
    axis_translation.set_title(f"{robot_name} Translation Error")
    axis_translation.set_ylabel("translation error [m]")
    axis_translation.grid(True, alpha=0.3)

    axis_yaw.plot(elapsed_s, yaw_errors, color="tab:purple", linewidth=2.0)
    axis_yaw.set_title(f"{robot_name} Yaw Error")
    axis_yaw.set_xlabel("aligned elapsed [s]")
    axis_yaw.set_ylabel("yaw error [rad]")
    axis_yaw.grid(True, alpha=0.3)

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _evaluate_rollout(rollout: RolloutData, plt) -> list[RobotEvaluationResult]:
    tracked_robots = [robot for robot in rollout.robots if robot.tracker_path is not None]
    if not tracked_robots:
        print(
            f"[INFO] Skipping rollout {rollout.rollout_id}: no mapf_timed_tracker CSV files were found.",
            flush=True,
        )
        return []

    output_dir = rollout.rollout_dir / "integration_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[RobotEvaluationResult] = []
    for robot in rollout.robots:
        if robot.tracker_path is None or not robot.tracker_samples:
            results.append(
                RobotEvaluationResult(
                    robot_name=robot.name,
                    status="skipped_missing_tracker",
                    tracker_path=None,
                    aligned_start_timestamp_ns=None,
                    sample_count=0,
                    duration_s=0.0,
                    translation_ate_rmse_m=None,
                    translation_ate_mae_m=None,
                    translation_ate_max_m=None,
                    yaw_rmse_rad=None,
                    yaw_mae_rad=None,
                    yaw_max_rad=None,
                    final_translation_drift_m=None,
                    final_yaw_drift_rad=None,
                )
            )
            continue

        trajectory = integrate_velocity_samples(
            robot.initial_pose,
            robot.velocity_samples,
            source_label=str(robot.velocity_path),
        )
        aligned_start_timestamp_ns = _estimate_alignment_start_timestamp_ns(
            trajectory,
            robot.tracker_samples,
            robot.initial_pose.z,
        )
        comparisons = _build_aligned_comparisons(
            trajectory,
            robot.tracker_samples,
            robot.initial_pose.z,
            aligned_start_timestamp_ns,
        )
        if not comparisons:
            results.append(
                RobotEvaluationResult(
                    robot_name=robot.name,
                    status="skipped_no_overlap",
                    tracker_path=str(robot.tracker_path),
                    aligned_start_timestamp_ns=aligned_start_timestamp_ns,
                    sample_count=0,
                    duration_s=0.0,
                    translation_ate_rmse_m=None,
                    translation_ate_mae_m=None,
                    translation_ate_max_m=None,
                    yaw_rmse_rad=None,
                    yaw_mae_rad=None,
                    yaw_max_rad=None,
                    final_translation_drift_m=None,
                    final_yaw_drift_rad=None,
                )
            )
            continue

        translation_errors = [comparison.translation_error_m for comparison in comparisons]
        yaw_errors = [comparison.yaw_error_rad for comparison in comparisons]
        duration_s = comparisons[-1].elapsed_s - comparisons[0].elapsed_s if len(comparisons) > 1 else 0.0

        _write_comparison_csv(output_dir / f"{robot.name}_comparison.csv", comparisons)
        _plot_trajectory_overlay(
            robot.name,
            comparisons,
            output_dir / f"{robot.name}_trajectory_overlay.png",
            plt,
        )
        _plot_error_timeseries(
            robot.name,
            comparisons,
            output_dir / f"{robot.name}_error_timeseries.png",
            plt,
        )

        results.append(
            RobotEvaluationResult(
                robot_name=robot.name,
                status="evaluated",
                tracker_path=str(robot.tracker_path),
                aligned_start_timestamp_ns=aligned_start_timestamp_ns,
                sample_count=len(comparisons),
                duration_s=duration_s,
                translation_ate_rmse_m=_rmse(translation_errors),
                translation_ate_mae_m=_mae(translation_errors),
                translation_ate_max_m=max(translation_errors),
                yaw_rmse_rad=_rmse(yaw_errors),
                yaw_mae_rad=_mae(yaw_errors),
                yaw_max_rad=_max_abs(yaw_errors),
                final_translation_drift_m=translation_errors[-1],
                final_yaw_drift_rad=abs(yaw_errors[-1]),
            )
        )

    _write_summary_files(output_dir, results)
    return results


def _write_summary_files(output_dir: Path, results: list[RobotEvaluationResult]) -> None:
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"
    fieldnames = list(asdict(results[0]).keys()) if results else list(asdict(
        RobotEvaluationResult(
            robot_name="",
            status="",
            tracker_path=None,
            aligned_start_timestamp_ns=None,
            sample_count=0,
            duration_s=0.0,
            translation_ate_rmse_m=None,
            translation_ate_mae_m=None,
            translation_ate_max_m=None,
            yaw_rmse_rad=None,
            yaw_mae_rad=None,
            yaw_max_rad=None,
            final_translation_drift_m=None,
            final_yaw_drift_rad=None,
        )
    ).keys())

    with summary_csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    with summary_json_path.open("w", encoding="utf-8") as stream:
        json.dump([asdict(result) for result in results], stream, indent=2)


def main() -> int:
    args = _parse_args()
    try:
        rollouts = load_rollouts(args.experiments_root, rollout_ids=args.rollout_ids)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"EvaluateVelocityIntegration: {exc}", file=sys.stderr)
        return 1

    plt, _ = _import_matplotlib()
    evaluated_rollout_count = 0
    for rollout in rollouts:
        print(
            f"[INFO] Evaluating rollout {rollout.rollout_id} from {rollout.rollout_dir}...",
            flush=True,
        )
        results = _evaluate_rollout(rollout, plt)
        if not results:
            continue
        evaluated_rollout_count += 1
        evaluated_count = sum(1 for result in results if result.status == "evaluated")
        print(
            f"[INFO] Finished rollout {rollout.rollout_id}: evaluated {evaluated_count} robot trajectories.",
            flush=True,
        )

    if evaluated_rollout_count == 0:
        print("[INFO] No rollouts produced integration evaluation outputs.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
