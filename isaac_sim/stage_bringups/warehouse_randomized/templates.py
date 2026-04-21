from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from PIL import Image


SelectorMode = Literal["exact_path", "glob", "regex", "semantic"]
RandomizationPolicy = Literal["fixed", "appearance_only", "jittered_existing", "spawned_floor_prop"]


@dataclass(frozen=True)
class SelectorSpec:
    mode: SelectorMode
    value: str


@dataclass(frozen=True)
class KeepoutZone:
    zone_id: str
    min_xy: tuple[float, float]
    max_xy: tuple[float, float]


@dataclass(frozen=True)
class TemplateMapSpec:
    reference_yaml_path: Path
    resolution_m: float
    origin_hint_xyz: tuple[float, float, float]
    min_bound_xyz: tuple[float, float, float]
    max_bound_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class LightRandomizationSpec:
    name: str
    selectors: tuple[SelectorSpec, ...]
    intensity_range: tuple[float, float]
    color_value_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    temperature_range: tuple[float, float] = (2800.0, 6500.0)


@dataclass(frozen=True)
class ObjectRandomizationSpec:
    name: str
    policy: RandomizationPolicy
    selectors: tuple[SelectorSpec, ...] = ()
    source_selectors: tuple[SelectorSpec, ...] = ()
    required: bool = False
    xy_jitter_m: tuple[float, float] = (0.0, 0.0)
    yaw_jitter_deg: tuple[float, float] = (0.0, 0.0)
    uniform_scale_range: tuple[float, float] = (1.0, 1.0)
    color_value_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.25, 0.95),
        (0.25, 0.95),
        (0.25, 0.95),
    )
    support_surface: Literal["current_z", "ground"] = "current_z"
    collision_margin_m: float = 0.15
    max_attempts: int = 20
    keepout_zone_ids: tuple[str, ...] = ()
    spawn_count_range: tuple[int, int] = (0, 0)
    spawn_size_range_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.35, 1.0),
        (0.35, 1.0),
        (0.35, 1.2),
    )


@dataclass(frozen=True)
class WarehouseTemplate:
    template_id: str
    description: str
    base_environment_usd_rel: str
    nav2_map: TemplateMapSpec
    mapf_map: TemplateMapSpec
    light_randomizers: tuple[LightRandomizationSpec, ...]
    object_randomizers: tuple[ObjectRandomizationSpec, ...]
    focus_object_selectors: tuple[SelectorSpec, ...] = ()
    focus_distance_range_m: tuple[float, float] = (2.5, 5.5)
    keepout_zones: tuple[KeepoutZone, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


def _reference_map_spec(
    reference_yaml_path: Path,
    resolution_m: float,
    *,
    lower_bound_z_m: float = 0.1,
    upper_bound_z_m: float = 2.0,
) -> TemplateMapSpec:
    with reference_yaml_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}

    image_value = payload.get("image")
    if not image_value:
        raise ValueError(f"Map yaml is missing an image field: {reference_yaml_path}")

    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = reference_yaml_path.parent / image_path

    image = Image.open(image_path)
    width_px, height_px = image.size
    origin = payload.get("origin") or [0.0, 0.0, 0.0]
    origin_x = float(origin[0])
    origin_y = float(origin[1])
    origin_yaw = float(origin[2]) if len(origin) > 2 else 0.0

    max_x = origin_x + float(width_px) * float(resolution_m)
    max_y = origin_y + float(height_px) * float(resolution_m)
    return TemplateMapSpec(
        reference_yaml_path=reference_yaml_path,
        resolution_m=float(resolution_m),
        origin_hint_xyz=(origin_x, origin_y, origin_yaw),
        min_bound_xyz=(origin_x, origin_y, float(lower_bound_z_m)),
        max_bound_xyz=(max_x, max_y, float(upper_bound_z_m)),
    )


def build_template_catalog(repo_root: str | Path) -> dict[str, WarehouseTemplate]:
    repo_root = Path(repo_root).expanduser().resolve()
    maps_dir = repo_root / "ros2_ws" / "src" / "carters_nav2" / "maps"
    nav2_map = _reference_map_spec(maps_dir / "carter_warehouse_navigation.yaml", resolution_m=0.05)
    mapf_map = _reference_map_spec(maps_dir / "carter_warehouse_navigation_mapf.yaml", resolution_m=0.2)

    common_focus = (
        SelectorSpec("regex", r"(^|/).*(Forklift|forklift).*"),
    )
    light_specs = (
        LightRandomizationSpec(
            name="warehouse_lights",
            selectors=(
                SelectorSpec("regex", r"(^|/).*(Light|light).*"),
            ),
            intensity_range=(250.0, 5000.0),
            color_value_range=((0.65, 1.0), (0.65, 1.0), (0.65, 1.0)),
            temperature_range=(2800.0, 7200.0),
        ),
    )

    appearance_props = ObjectRandomizationSpec(
        name="appearance_props",
        policy="appearance_only",
        selectors=(
            SelectorSpec("regex", r"(^|/).*(Pallet|pallet|Bin|bin|Box|box|Crate|crate|Cardbox).*"),
        ),
        color_value_range=((0.20, 0.95), (0.20, 0.95), (0.20, 0.95)),
    )
    forklift_jitter = ObjectRandomizationSpec(
        name="forklift_layout",
        policy="jittered_existing",
        selectors=(
            SelectorSpec("regex", r"(^|/).*(Forklift|forklift).*"),
        ),
        xy_jitter_m=(0.75, 0.75),
        yaw_jitter_deg=(-25.0, 25.0),
        uniform_scale_range=(1.0, 1.0),
        collision_margin_m=0.25,
        max_attempts=24,
    )
    pallet_jitter_balanced = ObjectRandomizationSpec(
        name="balanced_floor_props",
        policy="jittered_existing",
        selectors=(
            SelectorSpec("regex", r"(^|/).*(Pallet|pallet|Bin|bin|Crate|crate|Cardbox).*"),
        ),
        xy_jitter_m=(1.0, 1.75),
        yaw_jitter_deg=(-90.0, 90.0),
        uniform_scale_range=(0.85, 1.2),
        collision_margin_m=0.18,
        max_attempts=18,
    )
    pallet_jitter_open = ObjectRandomizationSpec(
        name="open_floor_props",
        policy="jittered_existing",
        selectors=pallet_jitter_balanced.selectors,
        xy_jitter_m=(0.5, 1.0),
        yaw_jitter_deg=(-60.0, 60.0),
        uniform_scale_range=(0.95, 1.1),
        collision_margin_m=0.16,
        max_attempts=16,
    )
    pallet_jitter_messy = ObjectRandomizationSpec(
        name="messy_floor_props",
        policy="jittered_existing",
        selectors=pallet_jitter_balanced.selectors,
        xy_jitter_m=(1.8, 2.8),
        yaw_jitter_deg=(-180.0, 180.0),
        uniform_scale_range=(0.75, 1.35),
        collision_margin_m=0.20,
        max_attempts=26,
    )
    spawned_clutter_open = ObjectRandomizationSpec(
        name="open_spawned_clutter",
        policy="spawned_floor_prop",
        support_surface="ground",
        spawn_count_range=(1, 3),
        spawn_size_range_m=((0.30, 0.60), (0.30, 0.60), (0.35, 0.75)),
        color_value_range=((0.35, 0.90), (0.35, 0.90), (0.35, 0.90)),
        collision_margin_m=0.15,
        max_attempts=50,
    )
    spawned_clutter_balanced = ObjectRandomizationSpec(
        name="balanced_spawned_clutter",
        policy="spawned_floor_prop",
        support_surface="ground",
        spawn_count_range=(3, 7),
        spawn_size_range_m=((0.35, 0.85), (0.35, 0.85), (0.40, 1.00)),
        color_value_range=((0.30, 0.95), (0.30, 0.95), (0.30, 0.95)),
        collision_margin_m=0.16,
        max_attempts=80,
    )
    spawned_clutter_messy = ObjectRandomizationSpec(
        name="messy_spawned_clutter",
        policy="spawned_floor_prop",
        support_surface="ground",
        spawn_count_range=(8, 16),
        spawn_size_range_m=((0.35, 1.10), (0.35, 1.10), (0.45, 1.25)),
        color_value_range=((0.20, 0.95), (0.20, 0.95), (0.20, 0.95)),
        collision_margin_m=0.18,
        max_attempts=120,
    )

    base_environment = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"

    return {
        "warehouse_open": WarehouseTemplate(
            template_id="warehouse_open",
            description="Lighter randomization with mostly clear lanes and sparse clutter.",
            base_environment_usd_rel=base_environment,
            nav2_map=nav2_map,
            mapf_map=mapf_map,
            light_randomizers=light_specs,
            object_randomizers=(
                appearance_props,
                forklift_jitter,
                pallet_jitter_open,
                spawned_clutter_open,
            ),
            focus_object_selectors=common_focus,
            focus_distance_range_m=(2.75, 5.0),
            metadata={"layout_class": "open"},
        ),
        "warehouse_balanced": WarehouseTemplate(
            template_id="warehouse_balanced",
            description="Moderate clutter and moderate pose jitter around the reference warehouse layout.",
            base_environment_usd_rel=base_environment,
            nav2_map=nav2_map,
            mapf_map=mapf_map,
            light_randomizers=light_specs,
            object_randomizers=(
                appearance_props,
                forklift_jitter,
                pallet_jitter_balanced,
                spawned_clutter_balanced,
            ),
            focus_object_selectors=common_focus,
            focus_distance_range_m=(2.5, 5.25),
            metadata={"layout_class": "balanced"},
        ),
        "warehouse_messy": WarehouseTemplate(
            template_id="warehouse_messy",
            description="Heavier clutter and broader prop perturbations while still requiring collision-free navigation space.",
            base_environment_usd_rel=base_environment,
            nav2_map=nav2_map,
            mapf_map=mapf_map,
            light_randomizers=light_specs,
            object_randomizers=(
                appearance_props,
                forklift_jitter,
                pallet_jitter_messy,
                spawned_clutter_messy,
            ),
            focus_object_selectors=common_focus,
            focus_distance_range_m=(2.5, 5.5),
            metadata={"layout_class": "messy"},
        ),
    }
