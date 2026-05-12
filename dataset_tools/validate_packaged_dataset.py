from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any


class ValidationError(RuntimeError):
    """Raised when a packaged dataset is internally inconsistent."""


REQUIRED_ROLLOUT_COLUMNS = {
    "scene_id",
    "rollout_id",
    "tar_path",
    "robot_names",
    "camera_names",
    "num_render_frames",
    "split",
    "success",
    "package_status",
}
REQUIRED_FRAME_COLUMNS = {
    "scene_id",
    "rollout_id",
    "tar_path",
    "frame_index",
    "timestamp_ns",
    "camera_name",
    "camera_type",
    "rgb_path_in_tar",
    "depth_path_in_tar",
}


def validate_packaged_dataset(root: str | Path, *, max_tar_checks: int | None = None) -> dict[str, Any]:
    dataset_root = Path(root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    required_files = [
        dataset_root / "README.md",
        dataset_root / "metadata" / "dataset_release.json",
        dataset_root / "metadata" / "schema.md",
        dataset_root / "metadata" / "scenes.parquet",
        dataset_root / "metadata" / "rollouts.parquet",
        dataset_root / "metadata" / "frames.parquet",
    ]
    missing = [str(path.relative_to(dataset_root)) for path in required_files if not path.is_file()]
    if missing:
        raise ValidationError(f"Missing required packaged dataset files: {missing}")

    rollouts = _read_parquet_records(
        dataset_root / "metadata" / "rollouts.parquet",
        required_columns=REQUIRED_ROLLOUT_COLUMNS,
    )
    frames = _read_parquet_records(
        dataset_root / "metadata" / "frames.parquet",
        required_columns=REQUIRED_FRAME_COLUMNS,
    )
    release = json.loads(
        (dataset_root / "metadata" / "dataset_release.json").read_text(encoding="utf-8")
    )

    frames_by_tar: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        tar_path = str(frame.get("tar_path", "") or "")
        if not tar_path:
            raise ValidationError("Frame row is missing tar_path.")
        timestamp = frame.get("timestamp_ns")
        if timestamp is None:
            raise ValidationError(f"Frame row is missing timestamp_ns: {frame}")
        try:
            int(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Frame timestamp is not parseable: {timestamp}") from exc
        frames_by_tar.setdefault(tar_path, []).append(frame)

    checked_tars = 0
    for rollout in rollouts:
        tar_rel = str(rollout.get("tar_path", "") or "")
        if not tar_rel:
            raise ValidationError(f"Rollout row is missing tar_path: {rollout}")
        tar_path = dataset_root / tar_rel
        if not tar_path.is_file():
            raise ValidationError(f"Rollout tar does not exist: {tar_rel}")
        if max_tar_checks is not None and checked_tars >= max_tar_checks:
            continue
        _validate_rollout_tar(tar_path, frames_by_tar.get(tar_rel, []), dataset_root)
        checked_tars += 1

    return {
        "dataset_root": str(dataset_root),
        "dataset_version": release.get("dataset_version", ""),
        "rollouts": len(rollouts),
        "frames": len(frames),
        "checked_tars": checked_tars,
    }


def _validate_rollout_tar(
    tar_path: Path,
    frames: list[dict[str, Any]],
    dataset_root: Path,
) -> None:
    with tarfile.open(tar_path, "r") as tar:
        names = set(tar.getnames())
        required = {"run_config.yaml", "render_manifest.csv"}
        missing = sorted(required.difference(names))
        if missing:
            raise ValidationError(
                f"{tar_path.relative_to(dataset_root)} is missing required members: {missing}"
            )
        if not any(name.startswith("rgb/") for name in names):
            raise ValidationError(f"{tar_path.relative_to(dataset_root)} has no rgb/ members.")
        if not any(name.startswith("depth/") for name in names):
            raise ValidationError(f"{tar_path.relative_to(dataset_root)} has no depth/ members.")

        for frame in frames:
            rgb_path = str(frame.get("rgb_path_in_tar", "") or "")
            depth_path = str(frame.get("depth_path_in_tar", "") or "")
            if rgb_path and rgb_path not in names:
                raise ValidationError(
                    f"{tar_path.relative_to(dataset_root)} missing referenced RGB frame {rgb_path}"
                )
            if depth_path and depth_path not in names:
                raise ValidationError(
                    f"{tar_path.relative_to(dataset_root)} missing referenced depth frame {depth_path}"
                )


def _read_parquet_records(
    path: Path,
    *,
    required_columns: set[str],
) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Packaged dataset validation requires pandas.") from exc
    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Reading Parquet metadata requires pyarrow or fastparquet. "
            "Install one in the validation environment, e.g. `pip install pyarrow`."
        ) from exc
    missing = sorted(required_columns.difference(str(column) for column in frame.columns))
    if missing:
        raise ValidationError(f"{path.name} is missing required columns: {missing}")
    return frame.to_dict(orient="records")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an HF-ready MAS-VLN dataset folder.")
    parser.add_argument("--root", required=True, help="Packaged dataset root.")
    parser.add_argument(
        "--max-tar-checks",
        type=int,
        default=None,
        help="Limit tar member validation for a quicker smoke check.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = validate_packaged_dataset(args.root, max_tar_checks=args.max_tar_checks)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
