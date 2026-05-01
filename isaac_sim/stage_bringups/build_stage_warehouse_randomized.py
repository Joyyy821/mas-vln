#!/usr/bin/env python3
"""Build randomized Isaac Sim warehouse scene bundles from manual USD templates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import re
import sys
import time
import traceback

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaac_sim.stage_bringups.warehouse_randomized.templates import (  # noqa: E402
    DEFAULT_BASE_VARIANT_ID,
    DEFAULT_VARIANT_IDS,
    WarehouseTemplate,
    compose_warehouse_template,
    discover_template_assets,
    load_randomization_presets,
    load_shared_warehouse_defaults,
)
from isaac_sim.stage_bringups.warehouse_randomized.robot_teams import (  # noqa: E402
    DEFAULT_ROBOT_TEAM_MODE,
)


DEFAULT_SCENE_ROOT_DIR = REPO_ROOT / "experiments" / "randomized_warehouse"
DEFAULT_COLLECTION_LANGUAGE_INSTRUCTION = "go to the forklift near the shelf"
DEFAULT_FOCUS_SELECTOR_ID = "forklift_near_shelf_min_world_x"
DEFAULT_COLLECTION_METADATA_FILENAME = "collection_metadata.yaml"
DEFAULT_RANDOMIZATION_STRENGTH = "balanced"
DEFAULT_SCENES_PER_TEMPLATE = 5
DEFAULT_ROLLOUT_CONTROL_TOPIC = os.environ.get(
    "CARTER_ROLLOUT_CONTROL_TOPIC",
    "/carters_goal/rollout_control",
)
DEFAULT_ROLLOUT_RESET_DONE_TOPIC = os.environ.get(
    "CARTER_ROLLOUT_RESET_DONE_TOPIC",
    "/carters_goal/rollout_reset_done",
)


@dataclass(frozen=True)
class BundleBuildSpec:
    template_id: str
    variant_id: str
    scene_id: str
    seed: int
    template_scene_index: int
    scene_number: int


@dataclass(frozen=True)
class FailedSceneAttempt:
    attempt_index: int
    seed: int
    exception_type: str
    exception_message: str


@dataclass(frozen=True)
class FailedSceneSpec:
    spec: BundleBuildSpec
    attempts: tuple[FailedSceneAttempt, ...]
    failure_snapshot_path: Path


def _load_builder_class():
    from isaac_sim.stage_bringups.warehouse_randomized.builder import RandomizedWarehouseBuilder

    return RandomizedWarehouseBuilder


def _load_language_instruction(reference_config_path: Path) -> str:
    if not reference_config_path.exists():
        return ""
    with reference_config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    return str(payload.get("language_instruction", "") or "").strip()


def build_collection_metadata_payload(language_instruction: str) -> dict[str, str]:
    return {
        "language_instruction": str(language_instruction).strip(),
        "focus_selector": DEFAULT_FOCUS_SELECTOR_ID,
        "robot_team_mode": DEFAULT_ROBOT_TEAM_MODE,
    }


def _collection_metadata_path(scene_root_dir: Path) -> Path:
    return Path(scene_root_dir).expanduser().resolve() / DEFAULT_COLLECTION_METADATA_FILENAME


def _load_collection_metadata_language(scene_root_dir: Path) -> str:
    metadata_path = _collection_metadata_path(scene_root_dir)
    if not metadata_path.exists():
        return ""
    with metadata_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    return str(payload.get("language_instruction", "") or "").strip()


def _write_collection_metadata(scene_root_dir: Path, language_instruction: str) -> Path:
    metadata_path = _collection_metadata_path(scene_root_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            build_collection_metadata_payload(language_instruction),
            stream,
            sort_keys=False,
        )
    return metadata_path


def _parse_robot_models(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _template_sort_key(template_id: str) -> tuple[int, str]:
    clean_template_id = str(template_id).strip()
    if clean_template_id.isdigit():
        return (0, f"{int(clean_template_id):010d}")
    return (1, clean_template_id)


def _derive_bundle_seed(
    base_seed: int,
    template_id: str,
    variant_id: str,
    template_scene_index: int = 1,
    attempt_index: int = 1,
) -> int:
    seed_material = (
        f"{int(base_seed)}::{str(template_id).strip()}::"
        f"{str(variant_id).strip()}::{int(template_scene_index)}"
    )
    if int(attempt_index) > 1:
        seed_material = f"{seed_material}::{int(attempt_index)}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    return derived_seed or 1


def _build_scene_id(scene_number: int, scene_id_prefix: str = "") -> str:
    base_scene_id = f"scene_{int(scene_number)}"
    clean_prefix = str(scene_id_prefix).strip()
    return base_scene_id if not clean_prefix else f"{clean_prefix}_{base_scene_id}"


def _normalize_variant_ids(variant_ids: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    raw_variant_ids = tuple(variant_ids or DEFAULT_VARIANT_IDS)
    normalized_variant_ids: list[str] = []
    seen: set[str] = set()
    for variant_id in raw_variant_ids:
        clean_variant_id = str(variant_id).strip()
        if not clean_variant_id or clean_variant_id in seen:
            continue
        normalized_variant_ids.append(clean_variant_id)
        seen.add(clean_variant_id)
    if not normalized_variant_ids:
        normalized_variant_ids = list(DEFAULT_VARIANT_IDS)

    invalid_variant_ids = [variant_id for variant_id in normalized_variant_ids if variant_id not in DEFAULT_VARIANT_IDS]
    if invalid_variant_ids:
        raise ValueError(
            f"Unknown variant id(s) {invalid_variant_ids}. Available variants: {list(DEFAULT_VARIANT_IDS)}"
        )
    return tuple(normalized_variant_ids)


def _parse_selected_variant_ids(*, variants: list[str], base_only: bool) -> tuple[str, ...]:
    if base_only and variants:
        raise ValueError("--base-only cannot be combined with --variant.")
    if base_only:
        return (DEFAULT_BASE_VARIANT_ID,)

    requested_variant_ids: list[str] = []
    for value in variants:
        for token in str(value).split(","):
            clean_token = token.strip()
            if clean_token:
                requested_variant_ids.append(clean_token)
    if requested_variant_ids:
        selected = _normalize_variant_ids(tuple(requested_variant_ids))
        if len(selected) != 1:
            raise ValueError(
                "Randomized warehouse generation now uses one randomization strength per batch. "
                "Pass a single --variant value or use --randomization-strength."
            )
        return selected
    return (DEFAULT_RANDOMIZATION_STRENGTH,)


def _parse_randomization_strength(
    *,
    randomization_strength: str,
    variants: list[str],
    base_only: bool,
) -> str:
    if base_only or variants:
        return _parse_selected_variant_ids(variants=variants, base_only=base_only)[0]

    clean_strength = str(randomization_strength).strip() or DEFAULT_RANDOMIZATION_STRENGTH
    selected = _normalize_variant_ids((clean_strength,))
    if len(selected) != 1:
        raise ValueError(f"Expected one randomization strength, got {selected}.")
    return selected[0]


def plan_bundle_specs(
    *,
    available_template_ids: list[str],
    requested_template_id: str = "",
    all_template: bool = False,
    scene_id_prefix: str = "",
    base_seed: int,
    randomization_strength: str = DEFAULT_RANDOMIZATION_STRENGTH,
    scenes_per_template: int = DEFAULT_SCENES_PER_TEMPLATE,
) -> list[BundleBuildSpec]:
    available_ids = sorted(
        {str(value).strip() for value in available_template_ids if str(value).strip()},
        key=_template_sort_key,
    )
    if not available_ids:
        raise ValueError("No template ids are available for planning.")
    if all_template and str(scene_id_prefix).strip():
        raise ValueError("--scene-id cannot be used together with --all_template.")
    if int(scenes_per_template) <= 0:
        raise ValueError(f"scenes_per_template must be positive, got {scenes_per_template}.")
    clean_strength = str(randomization_strength).strip() or DEFAULT_RANDOMIZATION_STRENGTH
    selected_variant_id = _normalize_variant_ids((clean_strength,))[0]

    clean_requested_template_id = str(requested_template_id).strip()
    if all_template:
        selected_template_ids = available_ids
    else:
        if not clean_requested_template_id:
            clean_requested_template_id = available_ids[0]
        if clean_requested_template_id not in set(available_ids):
            raise ValueError(
                f"Unknown template_id '{clean_requested_template_id}'. Available templates: {available_ids}"
            )
        selected_template_ids = [clean_requested_template_id]

    specs: list[BundleBuildSpec] = []
    for template_id in selected_template_ids:
        template_offset = available_ids.index(template_id) * int(scenes_per_template)
        for template_scene_index in range(1, int(scenes_per_template) + 1):
            scene_number = template_offset + template_scene_index
            specs.append(
                BundleBuildSpec(
                    template_id=template_id,
                    variant_id=selected_variant_id,
                    scene_id=_build_scene_id(scene_number, scene_id_prefix),
                    seed=_derive_bundle_seed(
                        base_seed,
                        template_id,
                        selected_variant_id,
                        template_scene_index,
                    ),
                    template_scene_index=template_scene_index,
                    scene_number=scene_number,
                )
            )
    return specs


def _scene_bundle_dir(scene_root_dir: Path, scene_id: str) -> Path:
    return Path(scene_root_dir).expanduser().resolve() / str(scene_id)


def _selected_template_ids(
    *,
    available_template_ids: list[str],
    requested_template_id: str = "",
    all_template: bool = False,
    scene_id_prefix: str = "",
) -> list[str]:
    available_ids = sorted(
        {str(value).strip() for value in available_template_ids if str(value).strip()},
        key=_template_sort_key,
    )
    if not available_ids:
        raise ValueError("No template ids are available for planning.")
    if all_template and str(scene_id_prefix).strip():
        raise ValueError("--scene-id cannot be used together with --all_template.")

    clean_requested_template_id = str(requested_template_id).strip()
    if all_template:
        return available_ids
    if not clean_requested_template_id:
        clean_requested_template_id = available_ids[0]
    if clean_requested_template_id not in set(available_ids):
        raise ValueError(
            f"Unknown template_id '{clean_requested_template_id}'. Available templates: {available_ids}"
        )
    return [clean_requested_template_id]


def _existing_scene_numbers(scene_root_dir: Path, scene_id_prefix: str = "") -> list[int]:
    scene_root = Path(scene_root_dir).expanduser().resolve()
    if not scene_root.exists():
        return []

    clean_prefix = str(scene_id_prefix).strip()
    if clean_prefix:
        pattern = re.compile(rf"^{re.escape(clean_prefix)}_scene_(\d+)$")
    else:
        pattern = re.compile(r"^scene_(\d+)$")

    scene_numbers: list[int] = []
    for path in scene_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.fullmatch(path.name)
        if match is not None:
            scene_numbers.append(int(match.group(1)))
    return sorted(scene_numbers)


def select_bundle_specs_for_run(
    *,
    scene_root_dir: Path,
    available_template_ids: list[str],
    requested_template_id: str = "",
    all_template: bool = False,
    scene_id_prefix: str = "",
    base_seed: int,
    randomization_strength: str = DEFAULT_RANDOMIZATION_STRENGTH,
    scenes_per_template: int = DEFAULT_SCENES_PER_TEMPLATE,
    overwrite: bool = False,
) -> list[BundleBuildSpec]:
    requested_count = int(scenes_per_template)
    if requested_count <= 0:
        raise ValueError(f"scenes_per_template must be positive, got {scenes_per_template}.")
    selected_template_ids = _selected_template_ids(
        available_template_ids=available_template_ids,
        requested_template_id=requested_template_id,
        all_template=all_template,
        scene_id_prefix=scene_id_prefix,
    )
    if overwrite:
        return plan_bundle_specs(
            available_template_ids=available_template_ids,
            requested_template_id=requested_template_id,
            all_template=all_template,
            scene_id_prefix=scene_id_prefix,
            base_seed=base_seed,
            randomization_strength=randomization_strength,
            scenes_per_template=requested_count,
        )

    clean_strength = str(randomization_strength).strip() or DEFAULT_RANDOMIZATION_STRENGTH
    selected_variant_id = _normalize_variant_ids((clean_strength,))[0]
    next_scene_number = max(_existing_scene_numbers(scene_root_dir, scene_id_prefix), default=0) + 1
    selected: list[BundleBuildSpec] = []
    for template_id in selected_template_ids:
        for _ in range(requested_count):
            scene_number = next_scene_number
            next_scene_number += 1
            selected.append(
                BundleBuildSpec(
                    template_id=template_id,
                    variant_id=selected_variant_id,
                    scene_id=_build_scene_id(scene_number, scene_id_prefix),
                    seed=_derive_bundle_seed(
                        base_seed,
                        template_id,
                        selected_variant_id,
                        scene_number,
                    ),
                    template_scene_index=scene_number,
                    scene_number=scene_number,
                )
            )
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create runnable randomized warehouse scene bundles from manually authored "
            "warehouse_template_<id>.usd assets."
        )
    )
    parser.add_argument(
        "--scene-root-dir",
        default=str(DEFAULT_SCENE_ROOT_DIR),
        help="Directory that will contain flat sibling scene bundles.",
    )
    parser.add_argument(
        "--scene-id",
        default="",
        help="Optional scene-id prefix, producing folders like <prefix>_scene_1.",
    )
    parser.add_argument(
        "--template-registry-dir",
        action="append",
        default=[],
        help=(
            "Additional template registry directory containing the shared warehouse YAML and preset YAMLs. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--template-id",
        default="",
        help=(
            "Manual warehouse template id parsed from warehouse_template_<id>.usd. "
            "Defaults to the first discovered template when omitted."
        ),
    )
    parser.add_argument(
        "--randomization-strength",
        default=DEFAULT_RANDOMIZATION_STRENGTH,
        help=(
            "Randomization strength preset to use for every generated scene. "
            f"Defaults to '{DEFAULT_RANDOMIZATION_STRENGTH}'."
        ),
    )
    parser.add_argument(
        "--scenes-per-template",
        "--scene-per-template",
        "--scene-per-id",
        dest="scenes_per_template",
        type=int,
        default=DEFAULT_SCENES_PER_TEMPLATE,
        help=(
            "Number of new randomized scenes to generate per selected manual template. "
            f"Defaults to {DEFAULT_SCENES_PER_TEMPLATE}."
        ),
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "Deprecated alias for --randomization-strength. Pass only one value."
        ),
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Fast debug shortcut that only generates the base variant.",
    )
    parser.add_argument(
        "--all_template",
        "--all-templates",
        dest="all_template",
        action="store_true",
        help="Generate bundles for every discovered manual warehouse template.",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List discovered manual warehouse template ids and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic base seed. Each bundle derives its own stable per-template/per-variant seed from this.",
    )
    parser.add_argument(
        "--robot-count",
        type=int,
        default=0,
        help=(
            "Optional fixed robot count override. Leave at 0 to sample each rollout's "
            "heterogeneous team from the template robot_team policy."
        ),
    )
    parser.add_argument(
        "--robot-models",
        default="",
        help=(
            "Optional comma-separated fixed robot model override. Leave empty to use the "
            "template robot_team policy. Supported ids: nova_carter, carter_v1, jackal, limo."
        ),
    )
    parser.add_argument(
        "--rollout-count",
        type=int,
        default=5,
        help="How many rollouts to sample into each generated team config.",
    )
    parser.add_argument(
        "--scene-only",
        action="store_true",
        help=(
            "Generate only the randomized warehouse scene USD and maps. "
            "Skip robot pose sampling, team_config.yaml, and robot spawning."
        ),
    )
    parser.add_argument(
        "--spawn-robots-only",
        action="store_true",
        help=(
            "Load an existing scene bundle and spawn robots into the live stage without saving "
            "them into scene.usd. Existing team_config.yaml is reused unless --overwrite is set; "
            "missing/overwritten team configs are sampled from the existing scene maps."
        ),
    )
    parser.add_argument(
        "--reference-config-file",
        default="",
        help=(
            "Optional legacy YAML language fallback used only when --language-instruction "
            "and collection_metadata.yaml are absent."
        ),
    )
    parser.add_argument(
        "--language-instruction",
        default="",
        help="Language instruction to embed in the generated team config.",
    )
    parser.add_argument(
        "--rollout-control-topic",
        default=DEFAULT_ROLLOUT_CONTROL_TOPIC,
        help="PoseArray topic used by ROS to request rollout resets inside Isaac Sim.",
    )
    parser.add_argument(
        "--rollout-reset-done-topic",
        default=DEFAULT_ROLLOUT_RESET_DONE_TOPIC,
        help="Int32 topic published by Isaac Sim after each rollout reset completes.",
    )
    parser.add_argument(
        "--enable-ros2-runtime",
        action="store_true",
        help="Start the internal Isaac-side ROS runtime bridge for the final live Isaac bundle.",
    )
    parser.add_argument(
        "--run-sim",
        action="store_true",
        help="Keep the final generated Isaac Sim bundle running after batch generation completes.",
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Run Isaac Sim headless.",
    )
    parser.add_argument(
        "--gui",
        dest="headless",
        action="store_false",
        help="Run Isaac Sim with the GUI visible.",
    )
    parser.set_defaults(headless=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing scene bundle directory if it is already present.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring unknown Isaac Sim passthrough args: {unknown}")
    return args


def _run_interactive_loop(sim_app, ros_bridge) -> None:
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    try:
        while sim_app.is_running():
            if ros_bridge is not None:
                ros_bridge.spin_once()
            sim_app.update()
            if ros_bridge is not None:
                ros_bridge.step(float(timeline.get_current_time()))
    except KeyboardInterrupt:
        pass


def _close_runtime(result, sim_app) -> None:
    if result is not None and result.ros_bridge is not None:
        result.ros_bridge.shutdown()
    if sim_app is not None:
        sim_app.close()


def _build_scene_bundle(
    *,
    template: WarehouseTemplate,
    scene_root_dir: Path,
    scene_id: str,
    seed: int,
    robot_models: list[str],
    robot_count: int,
    rollout_count: int,
    language_instruction: str,
    enable_ros2_runtime: bool,
    rollout_control_topic: str,
    rollout_reset_done_topic: str,
    overwrite: bool,
    headless: bool,
    keep_sim_running: bool,
    sim_app=None,
    close_sim_app_when_done: bool = False,
    scene_only: bool = False,
    spawn_robots_only: bool = False,
):
    builder_cls = _load_builder_class()
    builder = builder_cls(
        sim_app=sim_app,
        repo_root=REPO_ROOT,
        scene_root_dir=scene_root_dir,
        scene_id=scene_id,
        template=template,
        seed=seed,
        robot_models=robot_models,
        robot_count=robot_count,
        rollout_count=rollout_count,
        language_instruction=language_instruction,
        enable_ros2_runtime=enable_ros2_runtime,
        rollout_control_topic=rollout_control_topic,
        rollout_reset_done_topic=rollout_reset_done_topic,
        overwrite=overwrite,
        scene_only=scene_only,
        spawn_robots_only=spawn_robots_only,
    )

    active_sim_app = sim_app
    result = None
    try:
        builder.ensure_sim_app(headless=headless)
        active_sim_app = builder.sim_app
        result = builder.build()
        if keep_sim_running or not close_sim_app_when_done:
            return result, active_sim_app
        _close_runtime(result, active_sim_app)
        return result, None
    except Exception as exc:
        if close_sim_app_when_done:
            _close_runtime(result, active_sim_app)
        else:
            setattr(exc, "_randomized_warehouse_sim_app", active_sim_app)
        raise


def _build_scene_bundle_with_retries(
    *,
    bundle_spec: BundleBuildSpec,
    base_seed: int,
    max_scene_attempts: int,
    template: WarehouseTemplate,
    scene_root_dir: Path,
    robot_models: list[str],
    robot_count: int,
    rollout_count: int,
    language_instruction: str,
    enable_ros2_runtime: bool,
    rollout_control_topic: str,
    rollout_reset_done_topic: str,
    overwrite: bool,
    headless: bool,
    keep_sim_running: bool,
    sim_app=None,
    scene_only: bool = False,
    spawn_robots_only: bool = False,
):
    attempts: list[FailedSceneAttempt] = []
    attempt_budget = max(1, int(max_scene_attempts))

    for attempt_index in range(1, attempt_budget + 1):
        attempt_seed = _derive_bundle_seed(
            base_seed,
            bundle_spec.template_id,
            bundle_spec.variant_id,
            bundle_spec.template_scene_index,
            attempt_index,
        )
        attempt_spec = replace(bundle_spec, seed=attempt_seed)
        attempt_overwrite = bool(overwrite) or attempt_index > 1
        if attempt_index > 1:
            print(
                f"[INFO] Retrying {bundle_spec.scene_id} "
                f"(attempt {attempt_index}/{attempt_budget}, seed={attempt_seed}).",
                flush=True,
            )
        try:
            result, active_sim_app = _build_scene_bundle(
                template=template,
                scene_root_dir=scene_root_dir,
                scene_id=attempt_spec.scene_id,
                seed=attempt_spec.seed,
                robot_models=robot_models,
                robot_count=robot_count,
                rollout_count=rollout_count,
                language_instruction=language_instruction,
                enable_ros2_runtime=enable_ros2_runtime,
                rollout_control_topic=rollout_control_topic,
                rollout_reset_done_topic=rollout_reset_done_topic,
                overwrite=attempt_overwrite,
                headless=headless,
                keep_sim_running=keep_sim_running,
                sim_app=sim_app,
                close_sim_app_when_done=False,
                scene_only=scene_only,
                spawn_robots_only=spawn_robots_only,
            )
            if attempts:
                print(
                    f"[OK] {bundle_spec.scene_id} succeeded after {attempt_index} attempts.",
                    flush=True,
                )
            return result, active_sim_app, None
        except Exception as exc:
            failed_sim_app = getattr(exc, "_randomized_warehouse_sim_app", None)
            if failed_sim_app is not None:
                sim_app = failed_sim_app
            attempts.append(
                FailedSceneAttempt(
                    attempt_index=attempt_index,
                    seed=attempt_seed,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
            )
            print(
                f"[WARN] {bundle_spec.scene_id} attempt {attempt_index}/{attempt_budget} failed: {exc}",
                flush=True,
            )

    failure = FailedSceneSpec(
        spec=bundle_spec,
        attempts=tuple(attempts),
        failure_snapshot_path=(
            _scene_bundle_dir(scene_root_dir, bundle_spec.scene_id) / "build_failure_snapshot.yaml"
        ),
    )
    print(
        f"[ERROR] Skipping {bundle_spec.scene_id} after {attempt_budget} failed attempts.",
        flush=True,
    )
    return None, sim_app, failure


def _print_bundle_result(result) -> None:
    print("[OK] Randomized warehouse scene bundle created.", flush=True)
    print(f"  scene_id: {result.scene_id}", flush=True)
    print(f"  seed: {result.seed}", flush=True)
    print(f"  template: {result.template_id}", flush=True)
    print(f"  variant: {result.variant_id}", flush=True)
    print(f"  bundle_dir: {result.bundle_dir}", flush=True)
    print(f"  scene_usd: {result.scene_usd_path}", flush=True)
    has_team_config = result.team_config_path.exists()
    if has_team_config:
        print(f"  team_config: {result.team_config_path}", flush=True)
    else:
        print(f"  team_config: skipped", flush=True)
    print(f"  nav2_map: {result.nav2_map_path}", flush=True)
    print(f"  mapf_map: {result.mapf_map_path}", flush=True)
    if has_team_config:
        print(
            "  ros_launch_hint: "
            f"ros2 launch carters_goal isaac_ros_mapf_rollouts.launch.py "
            f"team_config_file:={result.team_config_path} experiments_dir:={result.rollouts_dir}",
            flush=True,
        )


def _print_batch_summary(
    *,
    successful_results: list,
    failed_scenes: list[FailedSceneSpec],
) -> None:
    print("[SUMMARY] Randomized warehouse batch complete.", flush=True)
    print(f"  generated: {len(successful_results)}", flush=True)
    print(f"  failed: {len(failed_scenes)}", flush=True)
    if failed_scenes:
        print("  unsuccessful scene bundles:", flush=True)
        for failure in failed_scenes:
            last_attempt = failure.attempts[-1] if failure.attempts else None
            if last_attempt is None:
                print(
                    f"    - {failure.spec.scene_id} "
                    f"(template={failure.spec.template_id}, attempts=0)",
                    flush=True,
                )
                continue
            print(
                f"    - {failure.spec.scene_id} "
                f"(template={failure.spec.template_id}, attempts={len(failure.attempts)}, "
                f"last_seed={last_attempt.seed}, snapshot={failure.failure_snapshot_path})",
                flush=True,
            )
            print(
                f"      last_error: {last_attempt.exception_type}: "
                f"{last_attempt.exception_message}",
                flush=True,
            )


def main() -> int:
    args = _parse_args()
    if args.scene_only and args.spawn_robots_only:
        print("[ERROR] --scene-only cannot be combined with --spawn-robots-only.", file=sys.stderr)
        return 2
    scene_root_dir = Path(args.scene_root_dir).expanduser().resolve()
    language_instruction = str(args.language_instruction).strip()
    if not language_instruction:
        language_instruction = _load_collection_metadata_language(scene_root_dir)
    if not language_instruction and str(args.reference_config_file).strip():
        language_instruction = _load_language_instruction(
            Path(args.reference_config_file).expanduser().resolve()
        )
    if not language_instruction:
        language_instruction = DEFAULT_COLLECTION_LANGUAGE_INSTRUCTION

    template_registry_dirs = args.template_registry_dir or None
    shared_defaults = load_shared_warehouse_defaults(
        REPO_ROOT,
        template_registry_dirs=template_registry_dirs,
    )
    template_assets = discover_template_assets(
        REPO_ROOT,
        shared_defaults=shared_defaults,
        template_registry_dirs=template_registry_dirs,
    )
    template_ids = sorted(template_assets, key=_template_sort_key)
    if args.list_templates:
        print("Available warehouse templates:")
        for template_id in template_ids:
            template_asset = template_assets[template_id]
            print(f"  {template_id}: {template_asset.usd_path}", flush=True)
        return 0

    presets = load_randomization_presets(
        REPO_ROOT,
        template_registry_dirs=template_registry_dirs,
    )
    try:
        selected_randomization_strength = _parse_randomization_strength(
            randomization_strength=args.randomization_strength,
            variants=list(args.variant),
            base_only=bool(args.base_only),
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    required_preset_ids = sorted({selected_randomization_strength} - {DEFAULT_BASE_VARIANT_ID})
    missing_presets = sorted(set(required_preset_ids) - set(presets))
    if missing_presets:
        print(
            f"[ERROR] Missing required warehouse presets: {missing_presets}.",
            file=sys.stderr,
        )
        return 2

    run_sim = bool(args.run_sim)
    if args.scene_only and args.enable_ros2_runtime:
        print("[INFO] Ignoring --enable-ros2-runtime because --scene-only does not spawn robots.")
        args.enable_ros2_runtime = False
    if args.enable_ros2_runtime and not run_sim:
        print("[INFO] Enabling --run-sim because --enable-ros2-runtime requires a live Isaac process.")
        run_sim = True

    requested_headless = bool(args.headless) if args.headless is not None else not run_sim
    base_seed = int(args.seed) if args.seed is not None else int(time.time_ns() % (2**31 - 1))
    robot_models = _parse_robot_models(args.robot_models)

    try:
        bundle_specs = select_bundle_specs_for_run(
            scene_root_dir=scene_root_dir,
            available_template_ids=template_ids,
            requested_template_id=args.template_id,
            all_template=bool(args.all_template),
            scene_id_prefix=args.scene_id,
            base_seed=base_seed,
            randomization_strength=selected_randomization_strength,
            scenes_per_template=int(args.scenes_per_template),
            overwrite=bool(args.overwrite) or bool(args.spawn_robots_only),
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not bundle_specs:
        _write_collection_metadata(scene_root_dir, language_instruction)
        _print_batch_summary(
            successful_results=[],
            failed_scenes=[],
        )
        return 0

    if run_sim and len(bundle_specs) > 1:
        final_spec = bundle_specs[-1]
        print(
            "[INFO] Multiple bundles requested; only the final bundle will keep Isaac Sim running: "
            f"{final_spec.scene_id}",
            flush=True,
        )
    if args.enable_ros2_runtime and len(bundle_specs) > 1:
        print(
            "[INFO] ROS2 runtime will be enabled only for the final live bundle in this batch.",
            flush=True,
        )

    _write_collection_metadata(scene_root_dir, language_instruction)
    final_result = None
    successful_results = []
    failed_scenes: list[FailedSceneSpec] = []
    shared_sim_app = None
    shared_headless = requested_headless
    try:
        for index, bundle_spec in enumerate(bundle_specs):
            template_asset = template_assets[bundle_spec.template_id]
            preset = (
                None
                if bundle_spec.variant_id == DEFAULT_BASE_VARIANT_ID
                else presets[bundle_spec.variant_id]
            )
            template = compose_warehouse_template(
                template_asset,
                shared_defaults=shared_defaults,
                preset=preset,
            )
            is_final_bundle = index == (len(bundle_specs) - 1)
            keep_sim_running = is_final_bundle and run_sim
            result, sim_app, failure = _build_scene_bundle_with_retries(
                bundle_spec=bundle_spec,
                base_seed=base_seed,
                max_scene_attempts=template.scene_generation_max_attempts,
                template=template,
                scene_root_dir=scene_root_dir,
                robot_models=robot_models,
                robot_count=int(args.robot_count),
                rollout_count=int(args.rollout_count),
                language_instruction=language_instruction,
                enable_ros2_runtime=bool(args.enable_ros2_runtime) and keep_sim_running,
                rollout_control_topic=args.rollout_control_topic,
                rollout_reset_done_topic=args.rollout_reset_done_topic,
                overwrite=bool(args.overwrite),
                headless=shared_headless,
                keep_sim_running=keep_sim_running,
                sim_app=shared_sim_app,
                scene_only=bool(args.scene_only),
                spawn_robots_only=bool(args.spawn_robots_only),
            )
            if failure is not None:
                shared_sim_app = sim_app
                failed_scenes.append(failure)
                continue
            _print_bundle_result(result)
            successful_results.append(result)
            shared_sim_app = sim_app
            if is_final_bundle:
                final_result = result

        if run_sim and shared_sim_app is not None and final_result is not None:
            _run_interactive_loop(shared_sim_app, final_result.ros_bridge)
        _print_batch_summary(
            successful_results=successful_results,
            failed_scenes=failed_scenes,
        )
        return 1 if failed_scenes else 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        _close_runtime(final_result, shared_sim_app)


if __name__ == "__main__":
    raise SystemExit(main())
