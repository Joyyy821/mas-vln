from __future__ import annotations

import copy
import math
import os
import tempfile
from typing import Any

import yaml


POSE_ARRAY_LENGTH = 7


class NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def load_yaml_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_temp_yaml(prefix: str, data: Any) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".yaml")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as stream:
        yaml.dump(data, stream, Dumper=NoAliasSafeDumper, sort_keys=False)
    return path


def yaw_to_quaternion(yaw: float) -> list[float]:
    half_yaw = yaw * 0.5
    return [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)]


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def pose_config_to_list(pose_config: Any, pose_label: str) -> list[float]:
    if isinstance(pose_config, (list, tuple)):
        if len(pose_config) == 4:
            x, y, z, yaw = pose_config
            qx, qy, qz, qw = yaw_to_quaternion(float(yaw))
            return [float(x), float(y), float(z), qx, qy, qz, qw]
        if len(pose_config) == POSE_ARRAY_LENGTH:
            return [float(value) for value in pose_config]
        raise ValueError(
            f"{pose_label} must contain either 4 values [x, y, z, yaw] "
            f"or {POSE_ARRAY_LENGTH} values [x, y, z, qx, qy, qz, qw]."
        )

    if not isinstance(pose_config, dict):
        raise ValueError(f"{pose_label} must be a dict, list, or tuple.")

    x = float(pose_config.get("x", 0.0))
    y = float(pose_config.get("y", 0.0))
    z = float(pose_config.get("z", 0.0))

    if "orientation" in pose_config:
        orientation = pose_config["orientation"]
        if isinstance(orientation, (list, tuple)) and len(orientation) == 4:
            qx, qy, qz, qw = [float(value) for value in orientation]
        elif isinstance(orientation, dict):
            qx = float(orientation.get("x", 0.0))
            qy = float(orientation.get("y", 0.0))
            qz = float(orientation.get("z", 0.0))
            qw = float(orientation.get("w", 1.0))
        else:
            raise ValueError(
                f"{pose_label}.orientation must be a 4-value list or a dict with x/y/z/w."
            )
        return [x, y, z, qx, qy, qz, qw]

    if {"qx", "qy", "qz", "qw"}.issubset(pose_config.keys()):
        return [
            x,
            y,
            z,
            float(pose_config["qx"]),
            float(pose_config["qy"]),
            float(pose_config["qz"]),
            float(pose_config["qw"]),
        ]

    yaw = float(pose_config.get("yaw", 0.0))
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    return [x, y, z, qx, qy, qz, qw]


def pose_array_to_pose_dict(pose_array: list[float]) -> dict[str, float]:
    if len(pose_array) != POSE_ARRAY_LENGTH:
        raise ValueError(
            f"Expected a {POSE_ARRAY_LENGTH}-value pose array, got {len(pose_array)} values."
        )

    return {
        "x": float(pose_array[0]),
        "y": float(pose_array[1]),
        "z": float(pose_array[2]),
        "qx": float(pose_array[3]),
        "qy": float(pose_array[4]),
        "qz": float(pose_array[5]),
        "qw": float(pose_array[6]),
        "yaw": quaternion_to_yaw(
            float(pose_array[3]),
            float(pose_array[4]),
            float(pose_array[5]),
            float(pose_array[6]),
        ),
    }


def flatten_pose_arrays(robots: list[dict[str, Any]], pose_key: str) -> list[float]:
    return [
        value
        for robot in robots
        for value in pose_config_to_list(robot[pose_key], f"{robot['name']}.{pose_key}")
    ]


def resolve_optional_path(path_value: str | None, base_dir: str | None) -> str | None:
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value
    if base_dir is None:
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))


def _normalize_robots_config(robots_config: Any, source_label: str) -> list[dict[str, Any]]:
    if not robots_config:
        raise ValueError(f"No robots were configured in {source_label}.")

    robots: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, robot_config in enumerate(robots_config):
        if not isinstance(robot_config, dict):
            raise ValueError(f"Robot entry {index} in {source_label} must be a mapping.")

        name = str(robot_config.get("name", f"robot{index + 1}"))
        if name in seen_names:
            raise ValueError(f"Robot name '{name}' is duplicated in {source_label}.")
        seen_names.add(name)

        initial_pose = pose_config_to_list(
            robot_config.get("initial_pose", robot_config.get("start_pose")),
            f"{source_label}.robots[{index}].initial_pose",
        )
        goal_pose = pose_config_to_list(
            robot_config.get("goal_pose", robot_config.get("goal")),
            f"{source_label}.robots[{index}].goal_pose",
        )

        robots.append(
            {
                "name": name,
                "initial_pose": initial_pose,
                "goal_pose": goal_pose,
            }
        )

    return robots


def _normalize_rollout_id(raw_rollout_id: Any, default_rollout_id: int, source_label: str) -> int:
    if raw_rollout_id is None:
        return default_rollout_id

    try:
        rollout_id = int(raw_rollout_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Rollout id in {source_label} must be an integer.") from exc

    if rollout_id <= 0:
        raise ValueError(f"Rollout id in {source_label} must be positive, got {rollout_id}.")
    return rollout_id


def _rollout_payload(
    *,
    rollout_id: int,
    robots: list[dict[str, Any]],
    rollout_index: int,
) -> dict[str, Any]:
    return {
        "id": rollout_id,
        "rollout_index": rollout_index,
        "robots": robots,
        "initial_pose_array": flatten_pose_arrays(robots, "initial_pose"),
        "goal_pose_array": flatten_pose_arrays(robots, "goal_pose"),
    }


def load_multi_rollout_config(team_config_path: str, maps_dir: str | None = None) -> dict[str, Any]:
    config = load_yaml_file(team_config_path) or {}

    environment = config.get("environment", {})
    nav2_map = resolve_optional_path(environment.get("nav2_map"), maps_dir)
    mapf_map = resolve_optional_path(environment.get("mapf_map"), maps_dir)

    language_instruction = str(config.get("language_instruction", "") or "").strip()
    raw_rollouts = config.get("rollouts")

    canonical_rollouts: list[dict[str, Any]] = []
    if raw_rollouts is None:
        robots = _normalize_robots_config(config.get("robots", []), team_config_path)
        canonical_rollouts.append(
            _rollout_payload(
                rollout_id=_normalize_rollout_id(config.get("id"), 1, team_config_path),
                robots=robots,
                rollout_index=0,
            )
        )
    else:
        if not isinstance(raw_rollouts, list) or not raw_rollouts:
            raise ValueError(
                f"'rollouts' in {team_config_path} must be a non-empty list when provided."
            )

        canonical_robot_names: list[str] | None = None
        seen_rollout_ids: set[int] = set()
        for rollout_index, rollout_config in enumerate(raw_rollouts):
            if not isinstance(rollout_config, dict):
                raise ValueError(
                    f"Rollout entry {rollout_index} in {team_config_path} must be a mapping."
                )

            rollout_id = _normalize_rollout_id(
                rollout_config.get("id"),
                rollout_index + 1,
                f"{team_config_path}.rollouts[{rollout_index}]",
            )
            if rollout_id in seen_rollout_ids:
                raise ValueError(
                    f"Rollout id '{rollout_id}' is duplicated in {team_config_path}."
                )
            seen_rollout_ids.add(rollout_id)

            robots = _normalize_robots_config(
                rollout_config.get("robots", []),
                f"{team_config_path}.rollouts[{rollout_index}]",
            )
            rollout_robot_names = [robot["name"] for robot in robots]
            if canonical_robot_names is None:
                canonical_robot_names = rollout_robot_names
            else:
                if set(rollout_robot_names) != set(canonical_robot_names):
                    raise ValueError(
                        "Every rollout must define the same robot namespaces. "
                        f"Expected {canonical_robot_names}, got {rollout_robot_names} in "
                        f"{team_config_path}.rollouts[{rollout_index}]."
                    )
                robot_by_name = {robot["name"]: robot for robot in robots}
                robots = [robot_by_name[name] for name in canonical_robot_names]

            canonical_rollouts.append(
                _rollout_payload(
                    rollout_id=rollout_id,
                    robots=robots,
                    rollout_index=rollout_index,
                )
            )

    first_rollout = canonical_rollouts[0]

    return {
        "team_config_path": team_config_path,
        "language_instruction": language_instruction,
        "environment": environment,
        "rollouts": canonical_rollouts,
        "rollout_ids": [rollout["id"] for rollout in canonical_rollouts],
        "robot_namespaces": [robot["name"] for robot in first_rollout["robots"]],
        "agent_num": len(first_rollout["robots"]),
        "nav2_map": nav2_map,
        "mapf_map": mapf_map,
        "first_rollout": first_rollout,
    }


def load_team_config(
    team_config_path: str,
    maps_dir: str | None = None,
    *,
    rollout_id: int | None = None,
    rollout_index: int = 0,
) -> dict[str, Any]:
    config = load_multi_rollout_config(team_config_path, maps_dir=maps_dir)
    rollouts = config["rollouts"]

    selected_rollout = None
    if rollout_id is not None:
        selected_rollout = next((rollout for rollout in rollouts if rollout["id"] == rollout_id), None)
        if selected_rollout is None:
            raise ValueError(
                f"Rollout id {rollout_id} was not found in {team_config_path}. "
                f"Available ids: {config['rollout_ids']}"
            )
    else:
        if rollout_index < 0 or rollout_index >= len(rollouts):
            raise IndexError(
                f"rollout_index {rollout_index} is out of range for {team_config_path} "
                f"({len(rollouts)} rollouts available)."
            )
        selected_rollout = rollouts[rollout_index]

    return {
        **config,
        "selected_rollout": selected_rollout,
        "rollout_id": selected_rollout["id"],
        "rollout_index": selected_rollout["rollout_index"],
        "robots": selected_rollout["robots"],
        "initial_pose_array": selected_rollout["initial_pose_array"],
        "goal_pose_array": selected_rollout["goal_pose_array"],
    }


def _replace_robot_namespace(value: Any, template_namespace: str, robot_namespace: str) -> Any:
    if isinstance(value, str):
        return value.replace(f"/{template_namespace}/", f"/{robot_namespace}/").replace(
            f"/{template_namespace}",
            f"/{robot_namespace}",
        )
    if isinstance(value, list):
        return [
            _replace_robot_namespace(item, template_namespace, robot_namespace) for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _replace_robot_namespace(item, template_namespace, robot_namespace)
            for key, item in value.items()
        }
    return value


def render_nav2_params(
    template_path: str,
    robot_namespace: str,
    initial_pose: list[float],
    template_namespace: str = "robot1",
) -> dict[str, Any]:
    params = copy.deepcopy(load_yaml_file(template_path))
    params = _replace_robot_namespace(params, template_namespace, robot_namespace)

    amcl_params = params.get("amcl", {}).get("ros__parameters")
    if amcl_params is not None:
        initial_pose_dict = pose_array_to_pose_dict(initial_pose)
        amcl_params["initial_pose"] = {
            "x": initial_pose_dict["x"],
            "y": initial_pose_dict["y"],
            "z": initial_pose_dict["z"],
            "yaw": initial_pose_dict["yaw"],
        }
    return params


def build_agent_indexed_map(
    robot_namespaces: list[str],
    suffix: str,
    leading_slash: bool = False,
) -> dict[str, str]:
    topic_prefix = "/" if leading_slash else ""
    return {
        f"agent_{index}": f"{topic_prefix}{namespace}/{suffix}"
        for index, namespace in enumerate(robot_namespaces)
    }
