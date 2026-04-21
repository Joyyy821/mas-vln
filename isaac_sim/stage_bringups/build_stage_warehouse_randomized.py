#!/usr/bin/env python3
"""Build a randomized Isaac Sim warehouse scene bundle and optionally keep it running."""

from __future__ import annotations

import argparse
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

from isaac_sim.stage_bringups.warehouse_randomized.builder import (  # noqa: E402
    RandomizedWarehouseBuilder,
)
from isaac_sim.stage_bringups.warehouse_randomized.templates import (  # noqa: E402
    build_template_catalog,
)


DEFAULT_SCENE_ROOT_DIR = REPO_ROOT / "experiments" / "randomized_warehouse"
DEFAULT_REFERENCE_CONFIG = REPO_ROOT / "data_configs" / "warehouse" / "warehouse_forklift.yaml"
DEFAULT_ROLLOUT_CONTROL_TOPIC = os.environ.get(
    "CARTER_ROLLOUT_CONTROL_TOPIC",
    "/carters_goal/rollout_control",
)
DEFAULT_ROLLOUT_RESET_DONE_TOPIC = os.environ.get(
    "CARTER_ROLLOUT_RESET_DONE_TOPIC",
    "/carters_goal/rollout_reset_done",
)


def _load_language_instruction(reference_config_path: Path) -> str:
    if not reference_config_path.exists():
        return ""
    with reference_config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    return str(payload.get("language_instruction", "") or "").strip()


def _default_scene_id(template_id: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{template_id}_{timestamp}"


def _parse_robot_models(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _parse_args() -> argparse.Namespace:
    template_catalog = build_template_catalog(REPO_ROOT)
    template_ids = sorted(template_catalog)

    parser = argparse.ArgumentParser(
        description=(
            "Create one randomized warehouse scene bundle for Isaac Sim 5.1, export occupancy maps "
            "and rollout configs, and optionally keep the scene running for GUI inspection or ROS-driven rollouts."
        )
    )
    parser.add_argument(
        "--scene-root-dir",
        default=str(DEFAULT_SCENE_ROOT_DIR),
        help="Directory that will contain per-scene bundles.",
    )
    parser.add_argument(
        "--scene-id",
        default="",
        help="Optional scene bundle name. Defaults to <template>_<timestamp>.",
    )
    parser.add_argument(
        "--template-id",
        choices=template_ids,
        default="warehouse_balanced",
        help=f"Warehouse layout template to randomize from. Available: {', '.join(template_ids)}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic random seed. Defaults to a time-based seed.",
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
        help="How many rollouts to sample into the generated team config.",
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
        help="Start the internal Isaac-side ROS runtime bridge for /cmd_vel, odom, TF, clock, and rollout reset.",
    )
    parser.add_argument(
        "--run-sim",
        action="store_true",
        help="Keep Isaac Sim running after the scene bundle is generated.",
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


def main() -> int:
    args = _parse_args()

    reference_config_path = Path(args.reference_config_file).expanduser().resolve()
    language_instruction = str(args.language_instruction).strip()
    if not language_instruction:
        language_instruction = _load_language_instruction(reference_config_path)

    run_sim = bool(args.run_sim)
    if args.enable_ros2_runtime and not run_sim:
        print("[INFO] Enabling --run-sim because --enable-ros2-runtime requires a live Isaac process.")
        run_sim = True

    headless = bool(args.headless) if args.headless is not None else not run_sim
    seed = int(args.seed) if args.seed is not None else int(time.time_ns() % (2**31 - 1))
    scene_id = str(args.scene_id).strip() or _default_scene_id(args.template_id)
    robot_models = _parse_robot_models(args.robot_models)

    builder = RandomizedWarehouseBuilder(
        repo_root=REPO_ROOT,
        scene_root_dir=Path(args.scene_root_dir).expanduser().resolve(),
        scene_id=scene_id,
        template_id=args.template_id,
        seed=seed,
        robot_models=robot_models,
        robot_count=int(args.robot_count),
        rollout_count=int(args.rollout_count),
        language_instruction=language_instruction,
        enable_ros2_runtime=bool(args.enable_ros2_runtime),
        rollout_control_topic=args.rollout_control_topic,
        rollout_reset_done_topic=args.rollout_reset_done_topic,
        overwrite=bool(args.overwrite),
    )

    sim_app = None
    result = None
    try:
        builder.ensure_sim_app(headless=headless)
        sim_app = builder.sim_app
        result = builder.build()

        print("[OK] Randomized warehouse scene bundle created.", flush=True)
        print(f"  scene_id: {result.scene_id}", flush=True)
        print(f"  seed: {result.seed}", flush=True)
        print(f"  template: {result.template_id}", flush=True)
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

        if run_sim and sim_app is not None:
            _run_interactive_loop(sim_app, result.ros_bridge)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if result is not None and result.ros_bridge is not None:
            result.ros_bridge.shutdown()
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
