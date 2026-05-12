from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path


def load_first_rgb_frame(
    dataset_root: str | Path,
    *,
    scene_id: int | None = None,
    rollout_id: int | None = None,
) -> bytes:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("This example requires pandas.") from exc

    root = Path(dataset_root).expanduser()
    frames = pd.read_parquet(root / "metadata" / "frames.parquet")
    if scene_id is not None:
        frames = frames[frames["scene_id"] == scene_id]
    if rollout_id is not None:
        frames = frames[frames["rollout_id"] == rollout_id]
    if frames.empty:
        raise RuntimeError("No matching frame rows found.")

    row = frames.iloc[0]
    tar_path = root / str(row["tar_path"])
    rgb_path = str(row["rgb_path_in_tar"])
    with tarfile.open(tar_path, "r") as tar:
        member = tar.getmember(rgb_path)
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Could not extract {rgb_path} from {tar_path}")
        return extracted.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read one RGB frame from a packaged dataset tar.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene-id", type=int, default=None)
    parser.add_argument("--rollout-id", type=int, default=None)
    parser.add_argument("--out", default=None, help="Optional path to write the extracted PNG.")
    args = parser.parse_args(argv)

    try:
        data = load_first_rgb_frame(
            args.dataset_root,
            scene_id=args.scene_id,
            rollout_id=args.rollout_id,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    if args.out:
        Path(args.out).write_bytes(data)
        print(f"Wrote {len(data)} bytes to {args.out}")
    else:
        image = io.BytesIO(data)
        print(f"Loaded RGB frame with {len(image.getvalue())} bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

