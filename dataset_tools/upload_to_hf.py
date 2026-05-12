from __future__ import annotations

import argparse
from pathlib import Path


def upload_to_hf(
    *,
    dataset_root: str | Path,
    repo_id: str,
    revision: str | None = None,
    private: bool | None = None,
    num_workers: int | None = None,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Uploading requires huggingface_hub. Install it with `pip install huggingface_hub`."
        ) from exc

    folder_path = Path(dataset_root).expanduser()
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {folder_path}")

    api = HfApi()
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder_path),
        revision=revision,
        private=private,
        num_workers=num_workers,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload a packaged dataset to Hugging Face.")
    parser.add_argument("--dataset-root", required=True, help="HF-ready dataset folder.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id.")
    parser.add_argument("--revision", default=None, help="Branch or revision to upload to.")
    parser.add_argument("--private", action="store_true", help="Create repo as private if needed.")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        upload_to_hf(
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            revision=args.revision,
            private=True if args.private else None,
            num_workers=args.num_workers,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

