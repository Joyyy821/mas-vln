from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def _load_team_config_utils():
    helper_path = os.path.join(
        get_package_share_directory("carters_nav2"),
        "launch",
        "team_config_utils.py",
    )
    spec = importlib.util.spec_from_file_location("team_config_utils", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load team config utilities from {helper_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


team_config_utils = _load_team_config_utils()


def find_repo_root(search_paths: list[Path]) -> Path:
    for root in search_paths:
        resolved = root.resolve()
        candidates = [resolved] + list(resolved.parents)
        for candidate in candidates:
            if (candidate / ".gitignore").exists() and (candidate / "ros2_ws").exists():
                return candidate
    return Path.cwd().resolve()


def resolve_experiments_root(experiments_dir: str, team_config_path: Path | str) -> Path:
    team_config_path = Path(team_config_path).expanduser().resolve()
    if experiments_dir:
        return Path(experiments_dir).expanduser().resolve()

    repo_root = find_repo_root([Path.cwd(), team_config_path, Path(__file__)])
    return repo_root / "experiments"


def rollout_run_dir(experiments_dir: str, team_config_path: Path | str, rollout_id: int) -> Path:
    team_config_path = Path(team_config_path).expanduser().resolve()
    experiments_root = resolve_experiments_root(experiments_dir, team_config_path)
    return experiments_root / team_config_path.stem.strip() / str(rollout_id)
