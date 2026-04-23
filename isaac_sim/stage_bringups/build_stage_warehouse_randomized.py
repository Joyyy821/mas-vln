#!/usr/bin/env python3
"""Build randomized Isaac Sim warehouse scene bundles from manual USD templates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
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
    DEFAULT_RANDOMIZATION_VARIANT_IDS,
    DEFAULT_VARIANT_IDS,
    WarehouseTemplate,
    compose_warehouse_template,
    discover_template_assets,
    load_randomization_presets,
    load_shared_warehouse_defaults,
)


DEFAULT_SCENE_ROOT_DIR = REPO_ROOT / "experiments" / "randomized_warehouse"
DEFAULT_REFERENCE_CONFIG = REPO_ROOT / "data_configs" / "warehouse" / "warehouse_forklift.yaml"
DEFAULT_MAP_EXPORT_MODE = os.environ.get("WAREHOUSE_MAP_EXPORT_MODE", "bbox").strip().lower() or "bbox"
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


def _load_builder_class():
    from isaac_sim.stage_bringups.warehouse_randomized.builder import RandomizedWarehouseBuilder

    return RandomizedWarehouseBuilder


def _load_language_instruction(reference_config_path: Path) -> str:
    if not reference_config_path.exists():
        return ""
    with reference_config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    return str(payload.get("language_instruction", "") or "").strip()


def _parse_robot_models(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _template_sort_key(template_id: str) -> tuple[int, str]:
    clean_template_id = str(template_id).strip()
    if clean_template_id.isdigit():
        return (0, f"{int(clean_template_id):010d}")
    return (1, clean_template_id)


def _derive_bundle_seed(base_seed: int, template_id: str, variant_id: str) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}::{str(template_id).strip()}::{str(variant_id).strip()}".encode("utf-8")
    ).digest()
    derived_seed = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    return derived_seed or 1


def _build_scene_id(template_id: str, variant_id: str, scene_id_prefix: str = "") -> str:
    base_scene_id = f"template_{str(template_id).strip()}_{str(variant_id).strip()}"
    clean_prefix = str(scene_id_prefix).strip()
    return base_scene_id if not clean_prefix else f"{clean_prefix}_{base_scene_id}"


def plan_bundle_specs(
    *,
    available_template_ids: list[str],
    requested_template_id: str = "",
    all_template: bool = False,
    scene_id_prefix: str = "",
    base_seed: int,
    variant_ids: tuple[str, ...] = DEFAULT_VARIANT_IDS,
) -> list[BundleBuildSpec]:
    available_ids = sorted({str(value).strip() for value in available_template_ids if str(value).strip()}, key=_template_sort_key)
    if not available_ids:
        raise ValueError("No template ids are available for planning.")
    if all_template and str(scene_id_prefix).strip():
        raise ValueError("--scene-id cannot be used together with --all_template.")

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
        for variant_id in variant_ids:
            specs.append(
                BundleBuildSpec(
                    template_id=template_id,
                    variant_id=variant_id,
                    scene_id=_build_scene_id(template_id, variant_id, scene_id_prefix),
                    seed=_derive_bundle_seed(base_seed, template_id, variant_id),
                )
            )
    return specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create runnable randomized warehouse scene bundles from manually authored "
            "warehouse_template_<id>.usd assets, including base, balanced, messy, and open variants."
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
        help="Optional per-template scene-id prefix, producing folders like <prefix>_template_1_balanced.",
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
        default=3,
        help="How many robots to spawn after map validation. Must be between 2 and 5.",
    )
    parser.add_argument(
        "--robot-models",
        default="nova_carter",
        help=(
            "Comma-separated robot model ids. Provide one model to reuse it for every robot, "
            "or one entry per robot. v1 fully supports 'nova_carter'."
        ),
    )
    parser.add_argument(
        "--rollout-count",
        type=int,
        default=5,
        help="How many rollouts to sample into each generated team config.",
    )
    parser.add_argument(
        "--reference-config-file",
        default=str(DEFAULT_REFERENCE_CONFIG),
        help=(
            "YAML file used to source a default language instruction when --language-instruction is omitted."
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
        "--map-export-mode",
        choices=("bbox", "omap"),
        default=DEFAULT_MAP_EXPORT_MODE,
        help=(
            "Map export backend. 'bbox' rasterizes collision/render bounds in-process and avoids "
            "the crashing OMAP extension; 'omap' uses isaacsim.asset.gen.omap."
        ),
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
    map_export_mode: str,
    overwrite: bool,
    headless: bool,
    keep_sim_running: bool,
    sim_app=None,
    close_sim_app_when_done: bool = False,
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
        map_export_mode=map_export_mode,
        overwrite=overwrite,
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
    except Exception:
        if close_sim_app_when_done:
            _close_runtime(result, active_sim_app)
        raise


def _print_bundle_result(result) -> None:
    print("[OK] Randomized warehouse scene bundle created.", flush=True)
    print(f"  scene_id: {result.scene_id}", flush=True)
    print(f"  seed: {result.seed}", flush=True)
    print(f"  template: {result.template_id}", flush=True)
    print(f"  variant: {result.variant_id}", flush=True)
    print(f"  bundle_dir: {result.bundle_dir}", flush=True)
    print(f"  scene_usd: {result.scene_usd_path}", flush=True)
    print(f"  team_config: {result.team_config_path}", flush=True)
    print(f"  nav2_map: {result.nav2_map_path}", flush=True)
    print(f"  mapf_map: {result.mapf_map_path}", flush=True)
    print(
        "  ros_launch_hint: "
        f"ros2 launch carters_goal isaac_ros_mapf_rollouts.launch.py "
        f"team_config_file:={result.team_config_path} experiments_dir:={result.rollouts_dir}",
        flush=True,
    )


def main() -> int:
    args = _parse_args()
    reference_config_path = Path(args.reference_config_file).expanduser().resolve()
    language_instruction = str(args.language_instruction).strip()
    if not language_instruction:
        language_instruction = _load_language_instruction(reference_config_path)

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
    missing_presets = sorted(set(DEFAULT_RANDOMIZATION_VARIANT_IDS) - set(presets))
    if missing_presets:
        print(
            f"[ERROR] Missing required warehouse presets: {missing_presets}.",
            file=sys.stderr,
        )
        return 2

    run_sim = bool(args.run_sim)
    if args.enable_ros2_runtime and not run_sim:
        print("[INFO] Enabling --run-sim because --enable-ros2-runtime requires a live Isaac process.")
        run_sim = True

    requested_headless = bool(args.headless) if args.headless is not None else not run_sim
    base_seed = int(args.seed) if args.seed is not None else int(time.time_ns() % (2**31 - 1))
    robot_models = _parse_robot_models(args.robot_models)

    try:
        bundle_specs = plan_bundle_specs(
            available_template_ids=template_ids,
            requested_template_id=args.template_id,
            all_template=bool(args.all_template),
            scene_id_prefix=args.scene_id,
            base_seed=base_seed,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

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

    scene_root_dir = Path(args.scene_root_dir).expanduser().resolve()
    final_result = None
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
            result, sim_app = _build_scene_bundle(
                template=template,
                scene_root_dir=scene_root_dir,
                scene_id=bundle_spec.scene_id,
                seed=bundle_spec.seed,
                robot_models=robot_models,
                robot_count=int(args.robot_count),
                rollout_count=int(args.rollout_count),
                language_instruction=language_instruction,
                enable_ros2_runtime=bool(args.enable_ros2_runtime) and keep_sim_running,
                rollout_control_topic=args.rollout_control_topic,
                rollout_reset_done_topic=args.rollout_reset_done_topic,
                map_export_mode=args.map_export_mode,
                overwrite=bool(args.overwrite),
                headless=shared_headless,
                keep_sim_running=keep_sim_running,
                sim_app=shared_sim_app,
                close_sim_app_when_done=False,
            )
            _print_bundle_result(result)
            shared_sim_app = sim_app
            if is_final_bundle:
                final_result = result

        if run_sim and shared_sim_app is not None and final_result is not None:
            _run_interactive_loop(shared_sim_app, final_result.ros_bridge)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        _close_runtime(final_result, shared_sim_app)


if __name__ == "__main__":
    raise SystemExit(main())
