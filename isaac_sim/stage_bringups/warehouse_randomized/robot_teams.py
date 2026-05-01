from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_ROBOT_TEAM_MODE = "heterogeneous_priority_v1"
DEFAULT_ROBOT_NAMESPACE_SCHEME = "model_name"
DEFAULT_ROBOT_MODEL_PRIORITY = ("nova_carter", "carter_v1", "jackal", "limo")
DEFAULT_ROBOT_COUNT_DISTRIBUTION = ((2, 0.2), (3, 0.5), (4, 0.3))


@dataclass(frozen=True)
class RobotTeamPolicy:
    mode: str = DEFAULT_ROBOT_TEAM_MODE
    count_distribution: tuple[tuple[int, float], ...] = DEFAULT_ROBOT_COUNT_DISTRIBUTION
    model_priority: tuple[str, ...] = DEFAULT_ROBOT_MODEL_PRIORITY
    namespace_scheme: str = DEFAULT_ROBOT_NAMESPACE_SCHEME


def _normalize_model_id(model_id: Any) -> str:
    return str(model_id).strip().lower()


def _normalize_count_distribution(raw_value: Any) -> tuple[tuple[int, float], ...]:
    if raw_value is None:
        return DEFAULT_ROBOT_COUNT_DISTRIBUTION

    items: list[tuple[Any, Any]]
    if isinstance(raw_value, Mapping):
        items = list(raw_value.items())
    else:
        items = []
        for entry in raw_value:
            if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) != 2:
                raise ValueError(
                    "robot_team.count_distribution entries must be two-value "
                    "[robot_count, probability] pairs."
                )
            items.append((entry[0], entry[1]))

    normalized: list[tuple[int, float]] = []
    for raw_count, raw_probability in items:
        count = int(raw_count)
        probability = float(raw_probability)
        if count <= 0:
            raise ValueError(f"robot_team.count_distribution count must be positive, got {count}.")
        if probability < 0.0:
            raise ValueError(
                f"robot_team.count_distribution probability must be non-negative, got {probability}."
            )
        normalized.append((count, probability))

    probability_sum = sum(probability for _, probability in normalized)
    if not normalized or probability_sum <= 0.0:
        raise ValueError("robot_team.count_distribution must contain at least one positive weight.")

    return tuple(sorted(normalized, key=lambda item: item[0]))


def parse_robot_team_policy(raw_value: Mapping[str, Any] | None) -> RobotTeamPolicy:
    raw = dict(raw_value or {})
    model_priority = tuple(
        _normalize_model_id(value)
        for value in raw.get("model_priority", DEFAULT_ROBOT_MODEL_PRIORITY)
        if _normalize_model_id(value)
    )
    if not model_priority:
        raise ValueError("robot_team.model_priority must contain at least one model id.")

    namespace_scheme = str(raw.get("namespace_scheme", DEFAULT_ROBOT_NAMESPACE_SCHEME)).strip()
    if namespace_scheme != DEFAULT_ROBOT_NAMESPACE_SCHEME:
        raise ValueError(
            "Only robot_team.namespace_scheme='model_name' is supported for heterogeneous teams."
        )

    count_distribution = _normalize_count_distribution(raw.get("count_distribution"))
    max_count = max(count for count, _ in count_distribution)
    if max_count > len(model_priority):
        raise ValueError(
            "robot_team.count_distribution requests more unique robots than "
            f"robot_team.model_priority provides: max_count={max_count}, "
            f"model_priority={list(model_priority)}."
        )

    return RobotTeamPolicy(
        mode=str(raw.get("mode", DEFAULT_ROBOT_TEAM_MODE)).strip() or DEFAULT_ROBOT_TEAM_MODE,
        count_distribution=count_distribution,
        model_priority=model_priority,
        namespace_scheme=namespace_scheme,
    )


def robot_team_policy_payload(policy: RobotTeamPolicy) -> dict[str, Any]:
    return {
        "mode": policy.mode,
        "count_distribution": {int(count): float(probability) for count, probability in policy.count_distribution},
        "model_priority": list(policy.model_priority),
        "namespace_scheme": policy.namespace_scheme,
    }


def sample_priority_robot_team(
    *,
    policy: RobotTeamPolicy,
    rng: np.random.Generator,
) -> list[dict[str, str]]:
    counts = np.array([count for count, _ in policy.count_distribution], dtype=np.int32)
    weights = np.array([probability for _, probability in policy.count_distribution], dtype=float)
    probabilities = weights / float(np.sum(weights))
    robot_count = int(rng.choice(counts, p=probabilities))
    models = list(policy.model_priority[:robot_count])
    return [{"name": model_id, "model": model_id} for model_id in models]


def build_fixed_robot_team(
    *,
    model_ids: Sequence[str],
    robot_count: int,
) -> list[dict[str, str]]:
    models = [_normalize_model_id(model_id) for model_id in model_ids if _normalize_model_id(model_id)]
    if not models:
        models = ["nova_carter"]
    if len(models) == 1:
        models = models * int(robot_count)
    if len(models) != int(robot_count):
        raise ValueError("robot_models must contain either one model or exactly robot_count models.")

    if len(set(models)) == len(models):
        names = list(models)
    else:
        names = [f"robot{index}" for index in range(1, int(robot_count) + 1)]

    return [
        {
            "name": name,
            "model": model_id,
        }
        for name, model_id in zip(names, models)
    ]
