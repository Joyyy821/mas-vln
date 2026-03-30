#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class TrackingLog:
    path: Path
    label: str
    elapsed: list[float]
    ref_x: list[float]
    ref_y: list[float]
    ref_yaw: list[float]
    ref_linear_velocity: list[float]
    ref_angular_velocity: list[float]
    actual_x: list[float]
    actual_y: list[float]
    actual_yaw: list[float]
    cmd_linear_x: list[float]
    cmd_angular_z: list[float]
    position_error: list[float]
    yaw_error: list[float]
    linear_saturated: list[int]
    angular_saturated: list[int]
    actual_linear_velocity: list[float]
    actual_angular_velocity: list[float]

    @property
    def sample_count(self) -> int:
        return len(self.elapsed)

    @property
    def duration(self) -> float:
        if not self.elapsed:
            return 0.0
        return self.elapsed[-1] - self.elapsed[0]


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _float_field(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except KeyError as exc:
        raise ValueError(f"Missing required CSV column '{key}'.") from exc


def _int_field(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except KeyError as exc:
        raise ValueError(f"Missing required CSV column '{key}'.") from exc


def _compute_actual_velocities(
    elapsed: list[float],
    xs: list[float],
    ys: list[float],
    yaws: list[float],
) -> tuple[list[float], list[float]]:
    if not elapsed:
        return [], []

    linear_velocity = [0.0]
    angular_velocity = [0.0]
    for idx in range(1, len(elapsed)):
        dt_sec = max(elapsed[idx] - elapsed[idx - 1], 1e-6)
        dx = xs[idx] - xs[idx - 1]
        dy = ys[idx] - ys[idx - 1]
        dyaw = _normalize_angle(yaws[idx] - yaws[idx - 1])
        linear_velocity.append(math.hypot(dx, dy) / dt_sec)
        angular_velocity.append(dyaw / dt_sec)

    if len(linear_velocity) > 1:
        linear_velocity[0] = linear_velocity[1]
        angular_velocity[0] = angular_velocity[1]
    return linear_velocity, angular_velocity


def _load_tracking_log(path: Path) -> TrackingLog:
    required_columns = {
        "elapsed",
        "ref_x",
        "ref_y",
        "ref_yaw",
        "ref_linear_velocity",
        "ref_angular_velocity",
        "actual_x",
        "actual_y",
        "actual_yaw",
        "cmd_linear_x",
        "cmd_angular_z",
        "position_error",
        "yaw_error",
        "linear_saturated",
        "angular_saturated",
    }

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or missing a header row.")
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required columns: {missing_list}.")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} has no data rows.")

    elapsed = [_float_field(row, "elapsed") for row in rows]
    ref_x = [_float_field(row, "ref_x") for row in rows]
    ref_y = [_float_field(row, "ref_y") for row in rows]
    ref_yaw = [_float_field(row, "ref_yaw") for row in rows]
    ref_linear_velocity = [_float_field(row, "ref_linear_velocity") for row in rows]
    ref_angular_velocity = [_float_field(row, "ref_angular_velocity") for row in rows]
    actual_x = [_float_field(row, "actual_x") for row in rows]
    actual_y = [_float_field(row, "actual_y") for row in rows]
    actual_yaw = [_float_field(row, "actual_yaw") for row in rows]
    cmd_linear_x = [_float_field(row, "cmd_linear_x") for row in rows]
    cmd_angular_z = [_float_field(row, "cmd_angular_z") for row in rows]
    position_error = [_float_field(row, "position_error") for row in rows]
    yaw_error = [_float_field(row, "yaw_error") for row in rows]
    linear_saturated = [_int_field(row, "linear_saturated") for row in rows]
    angular_saturated = [_int_field(row, "angular_saturated") for row in rows]
    actual_linear_velocity, actual_angular_velocity = _compute_actual_velocities(
        elapsed,
        actual_x,
        actual_y,
        actual_yaw,
    )

    return TrackingLog(
        path=path,
        label=path.stem,
        elapsed=elapsed,
        ref_x=ref_x,
        ref_y=ref_y,
        ref_yaw=ref_yaw,
        ref_linear_velocity=ref_linear_velocity,
        ref_angular_velocity=ref_angular_velocity,
        actual_x=actual_x,
        actual_y=actual_y,
        actual_yaw=actual_yaw,
        cmd_linear_x=cmd_linear_x,
        cmd_angular_z=cmd_angular_z,
        position_error=position_error,
        yaw_error=yaw_error,
        linear_saturated=linear_saturated,
        angular_saturated=angular_saturated,
        actual_linear_velocity=actual_linear_velocity,
        actual_angular_velocity=actual_angular_velocity,
    )


def _rmse(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _mae(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _final_translation_position_error(log: TrackingLog) -> float:
    return math.hypot(
        log.actual_x[-1] - log.ref_x[-1],
        log.actual_y[-1] - log.ref_y[-1],
    )


def _final_translation_yaw_error(log: TrackingLog) -> float:
    return abs(_normalize_angle(log.actual_yaw[-1] - log.ref_yaw[-1]))


def _summary_lines(log: TrackingLog) -> list[str]:
    return [
        f"samples: {log.sample_count}",
        f"duration: {log.duration:.2f} s",
        f"position_rmse: {_rmse(log.position_error):.3f} m",
        f"position_mae: {_mae(log.position_error):.3f} m",
        f"position_max: {max(log.position_error):.3f} m",
        f"yaw_rmse: {_rmse(log.yaw_error):.3f} rad",
        f"yaw_mae: {_mae(log.yaw_error):.3f} rad",
        f"yaw_max: {max(abs(value) for value in log.yaw_error):.3f} rad",
        f"linear_saturation_ratio: {_mean(log.linear_saturated):.2%}",
        f"angular_saturation_ratio: {_mean(log.angular_saturated):.2%}",
        f"max_ref_linear_velocity: {max(abs(v) for v in log.ref_linear_velocity):.3f} m/s",
        f"max_cmd_linear_velocity: {max(abs(v) for v in log.cmd_linear_x):.3f} m/s",
        f"max_actual_linear_velocity: {max(abs(v) for v in log.actual_linear_velocity):.3f} m/s",
        f"max_cmd_angular_velocity: {max(abs(v) for v in log.cmd_angular_z):.3f} rad/s",
        f"max_actual_angular_velocity: {max(abs(v) for v in log.actual_angular_velocity):.3f} rad/s",
        f"final_translation_position_error: {_final_translation_position_error(log):.3f} m",
        f"final_translation_yaw_error: {_final_translation_yaw_error(log):.3f} rad",
        "note: CSV covers timed translation only; post-rotation is not logged here.",
    ]


def _find_repo_root() -> Path:
    search_roots = [Path.cwd().resolve(), Path(__file__).resolve()]
    for root in search_roots:
        for candidate in [root] + list(root.parents):
            if (candidate / ".gitignore").exists() and (candidate / "ros2_ws").exists():
                return candidate
    return Path.cwd().resolve()


def _run_timestamp(csv_paths: list[Path]) -> str:
    earliest_mtime = min(path.stat().st_mtime for path in csv_paths)
    return dt.datetime.fromtimestamp(earliest_mtime).strftime("%Y%m%d_%H%M%S")


def _import_matplotlib(force_headless: bool):
    try:
        import matplotlib

        display_available = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        headless = force_headless or (sys.platform.startswith("linux") and not display_available)
        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for PlotMapfTrackingLogs. "
            "Install python3-matplotlib in the ROS environment."
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


def _plot_single_log(log: TrackingLog, output_dir: Path, dpi: int, plt) -> tuple[Path, object]:
    figure = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = figure.add_gridspec(3, 2)
    ax_xy = figure.add_subplot(grid[0, 0])
    ax_error = figure.add_subplot(grid[0, 1])
    ax_linear = figure.add_subplot(grid[1, 0])
    ax_angular = figure.add_subplot(grid[1, 1])
    ax_saturation = figure.add_subplot(grid[2, 0])
    ax_summary = figure.add_subplot(grid[2, 1])

    figure.suptitle(log.label, fontsize=14, fontweight="bold")

    ax_xy.plot(log.ref_x, log.ref_y, label="reference", color="tab:blue", linewidth=2.0)
    ax_xy.plot(log.actual_x, log.actual_y, label="actual", color="tab:orange", linewidth=2.0)
    ax_xy.scatter(log.ref_x[0], log.ref_y[0], color="tab:blue", marker="o", s=45, label="ref start")
    ax_xy.scatter(log.ref_x[-1], log.ref_y[-1], color="tab:blue", marker="X", s=60, label="ref end")
    ax_xy.scatter(
        log.actual_x[0],
        log.actual_y[0],
        color="tab:orange",
        marker="o",
        facecolors="none",
        s=55,
        label="actual start",
    )
    ax_xy.scatter(
        log.actual_x[-1],
        log.actual_y[-1],
        color="tab:orange",
        marker="X",
        s=70,
        label="actual end",
    )
    ax_xy.set_title("XY Overlay")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    _fit_xy_axis(ax_xy, log.ref_x + log.actual_x, log.ref_y + log.actual_y)
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best")

    ax_error.plot(
        log.elapsed,
        log.position_error,
        color="tab:red",
        linewidth=2.0,
        label="position error [m]",
    )
    ax_error.set_title("Tracking Error")
    ax_error.set_xlabel("elapsed [s]")
    ax_error.set_ylabel("position error [m]", color="tab:red")
    ax_error.tick_params(axis="y", labelcolor="tab:red")
    ax_error.grid(True, alpha=0.3)
    ax_error_yaw = ax_error.twinx()
    ax_error_yaw.plot(
        log.elapsed,
        [abs(value) for value in log.yaw_error],
        color="tab:purple",
        linewidth=2.0,
        label="|yaw error| [rad]",
    )
    ax_error_yaw.set_ylabel("|yaw error| [rad]", color="tab:purple")
    ax_error_yaw.tick_params(axis="y", labelcolor="tab:purple")
    lines = ax_error.get_lines() + ax_error_yaw.get_lines()
    ax_error.legend(lines, [line.get_label() for line in lines], loc="best")

    ax_linear.plot(
        log.elapsed,
        log.ref_linear_velocity,
        label="reference",
        color="tab:blue",
        linewidth=2.0,
    )
    ax_linear.plot(
        log.elapsed,
        log.cmd_linear_x,
        label="commanded",
        color="tab:orange",
        linewidth=1.8,
    )
    ax_linear.plot(
        log.elapsed,
        log.actual_linear_velocity,
        label="actual (finite diff)",
        color="tab:green",
        linewidth=1.5,
    )
    ax_linear.set_title("Linear Velocity")
    ax_linear.set_xlabel("elapsed [s]")
    ax_linear.set_ylabel("linear velocity [m/s]")
    ax_linear.grid(True, alpha=0.3)
    ax_linear.legend(loc="best")

    ax_angular.plot(
        log.elapsed,
        log.ref_angular_velocity,
        label="reference",
        color="tab:blue",
        linewidth=2.0,
    )
    ax_angular.plot(
        log.elapsed,
        log.cmd_angular_z,
        label="commanded",
        color="tab:orange",
        linewidth=1.8,
    )
    ax_angular.plot(
        log.elapsed,
        log.actual_angular_velocity,
        label="actual (finite diff)",
        color="tab:green",
        linewidth=1.5,
    )
    ax_angular.set_title("Angular Velocity")
    ax_angular.set_xlabel("elapsed [s]")
    ax_angular.set_ylabel("angular velocity [rad/s]")
    ax_angular.grid(True, alpha=0.3)
    ax_angular.legend(loc="best")

    linear_events = [time for time, saturated in zip(log.elapsed, log.linear_saturated) if saturated]
    angular_events = [time for time, saturated in zip(log.elapsed, log.angular_saturated) if saturated]
    ax_saturation.eventplot(
        [linear_events, angular_events],
        colors=["tab:orange", "tab:purple"],
        lineoffsets=[1.0, 0.0],
        linelengths=0.8,
        linewidths=1.5,
    )
    ax_saturation.set_title("Command Saturation Events")
    ax_saturation.set_xlabel("elapsed [s]")
    ax_saturation.set_yticks([0.0, 1.0], labels=["angular", "linear"])
    ax_saturation.grid(True, axis="x", alpha=0.3)

    ax_summary.axis("off")
    ax_summary.set_title("Summary")
    ax_summary.text(
        0.0,
        1.0,
        "\n".join(_summary_lines(log)),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{log.path.stem}.png"
    figure.savefig(output_path, dpi=dpi)
    return output_path, figure


def _plot_combined_xy(logs: list[TrackingLog], output_dir: Path, dpi: int, plt) -> tuple[Path | None, object | None]:
    if len(logs) < 2:
        return None, None

    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    color_map = plt.get_cmap("tab10")
    all_x: list[float] = []
    all_y: list[float] = []
    for idx, log in enumerate(logs):
        color = color_map(idx % 10)
        short_label = log.label.replace("mapf_timed_tracker_", "")
        axis.plot(
            log.ref_x,
            log.ref_y,
            linestyle="--",
            linewidth=2.0,
            color=color,
            label=f"{short_label} ref",
        )
        axis.plot(
            log.actual_x,
            log.actual_y,
            linestyle="-",
            linewidth=2.0,
            color=color,
            alpha=0.9,
            label=f"{short_label} actual",
        )
        axis.scatter(log.ref_x[0], log.ref_y[0], color=color, marker="o", s=40)
        axis.scatter(log.ref_x[-1], log.ref_y[-1], color=color, marker="X", s=55)
        all_x.extend(log.ref_x)
        all_x.extend(log.actual_x)
        all_y.extend(log.ref_y)
        all_y.extend(log.actual_y)

    axis.set_title("Combined XY Overlay")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    _fit_xy_axis(axis, all_x, all_y)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined_xy_overlay.png"
    figure.savefig(output_path, dpi=dpi)
    return output_path, figure


def _resolve_csv_paths(inputs: list[str], pattern: str) -> list[Path]:
    csv_paths: list[Path] = []
    for raw_input in inputs:
        path = Path(raw_input).expanduser()
        if path.is_dir():
            csv_paths.extend(sorted(candidate for candidate in path.glob(pattern) if candidate.is_file()))
            continue
        if path.is_file():
            csv_paths.append(path)
            continue
        raise FileNotFoundError(f"Input path does not exist: {path}")

    deduped = []
    seen = set()
    for path in csv_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _default_output_dir(csv_paths: list[Path]) -> Path:
    return _find_repo_root() / "experiments" / _run_timestamp(csv_paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot MAPF timed-tracker CSV logs. Pass one or more CSV files or a directory "
            "containing the logs emitted by MapfTimedTracker."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV files or directories to plot.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern used when an input is a directory. Default: *.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated PNG files. Defaults to <repo>/experiments/<run_timestamp>/.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output figure DPI. Default: 160",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots without opening interactive windows.",
    )
    args = parser.parse_args()

    try:
        csv_paths = _resolve_csv_paths(args.inputs, args.pattern)
        if not csv_paths:
            raise ValueError("No CSV files matched the provided inputs.")
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir is not None
            else _default_output_dir(csv_paths)
        )
        logs = [_load_tracking_log(path) for path in csv_paths]
    except (FileNotFoundError, ValueError) as exc:
        print(f"PlotMapfTrackingLogs: {exc}", file=sys.stderr)
        return 1

    plt, headless = _import_matplotlib(force_headless=args.no_show)
    figures = []

    for log in logs:
        output_path, figure = _plot_single_log(log, output_dir, args.dpi, plt)
        figures.append(figure)
        print(f"Saved plot: {output_path}")

    combined_output_path, combined_figure = _plot_combined_xy(logs, output_dir, args.dpi, plt)
    if combined_output_path is not None:
        print(f"Saved combined plot: {combined_output_path}")
    if combined_figure is not None:
        figures.append(combined_figure)

    if not args.no_show:
        if headless:
            print(
                "PlotMapfTrackingLogs: no interactive display detected, so the figures were saved only.",
                file=sys.stderr,
            )
        else:
            plt.show()

    for figure in figures:
        plt.close(figure)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
