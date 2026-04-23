from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import yaml


DEFAULT_TEMPLATE_CONFIG_DIRNAME = "template_configs"
DEFAULT_SHARED_TEMPLATE_CONFIG_BASENAME = "warehouse_shared.yaml"
DEFAULT_RANDOMIZATION_VARIANT_IDS = ("balanced", "messy", "open")
DEFAULT_BASE_VARIANT_ID = "base"
DEFAULT_VARIANT_IDS = (DEFAULT_BASE_VARIANT_ID,) + DEFAULT_RANDOMIZATION_VARIANT_IDS


def _as_float_tuple(values: Sequence[float], *, length: int, field_name: str) -> tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values, got {values}.")
    return tuple(float(value) for value in values)


def _as_range_tuple(values: Sequence[float], *, field_name: str) -> tuple[float, float]:
    lower, upper = _as_float_tuple(values, length=2, field_name=field_name)
    if lower > upper:
        raise ValueError(f"{field_name} lower bound must be <= upper bound, got {values}.")
    return (lower, upper)


def _as_rgb_range(
    values: Sequence[Sequence[float]] | None,
    *,
    field_name: str,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if not values:
        return ((0.65, 1.0), (0.65, 1.0), (0.65, 1.0))
    if len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 channel ranges.")
    channels: list[tuple[float, float]] = []
    for index, channel_range in enumerate(values):
        channels.append(_as_range_tuple(channel_range, field_name=f"{field_name}[{index}]"))
    return (channels[0], channels[1], channels[2])


def _path_list(paths: Iterable[str | Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        clean = Path(path).expanduser()
        if clean in seen:
            continue
        deduped.append(clean)
        seen.add(clean)
    return deduped


def _load_yaml_mapping(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved_path = Path(config_path).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML config must contain a mapping: {resolved_path}")
    return resolved_path, dict(payload)


def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if key == "enabled":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_repo_path(
    repo_root: str | Path,
    raw_path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    repo_root_path = Path(repo_root).expanduser().resolve()
    clean_path = Path(raw_path).expanduser()
    if clean_path.is_absolute():
        return clean_path.resolve()

    repo_candidate = (repo_root_path / clean_path).resolve()
    if base_dir is None:
        return repo_candidate

    local_candidate = (Path(base_dir).expanduser().resolve() / clean_path).resolve()
    if local_candidate.exists() and not repo_candidate.exists():
        return local_candidate
    return repo_candidate


@dataclass(frozen=True)
class SelectorSpec:
    mode: str
    value: str


@dataclass(frozen=True)
class TemplateMapSpec:
    resolution_m: float
    origin_hint_xyz: tuple[float, float, float]
    min_bound_xyz: tuple[float, float, float]
    max_bound_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class KeepoutZone:
    zone_id: str
    min_xy: tuple[float, float]
    max_xy: tuple[float, float]
    description: str = ""


@dataclass(frozen=True)
class PlacementZone:
    zone_id: str
    zone_type: str
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    description: str = ""


@dataclass(frozen=True)
class LightRandomizationSpec:
    name: str
    selectors: tuple[SelectorSpec, ...]
    intensity_range: tuple[float, float]
    color_value_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    temperature_range: tuple[float, float]


@dataclass(frozen=True)
class ObjectGroupSpec:
    group_id: str
    selectors: tuple[SelectorSpec, ...]
    required: bool = True
    root_mode: str = "nearest_xform"
    description: str = ""


@dataclass(frozen=True)
class ObjectRandomizationSpec:
    name: str
    policy: str
    target_group_name: str = ""
    source_group_name: str = ""
    selectors: tuple[SelectorSpec, ...] = ()
    anchor_zone_ids: tuple[str, ...] = ()
    support_zone_ids: tuple[str, ...] = ()
    keepout_zone_ids: tuple[str, ...] = ()
    snapped_yaw_deg: tuple[float, ...] = ()
    xy_jitter_m: tuple[float, float] = (0.0, 0.0)
    yaw_jitter_deg: tuple[float, float] = (0.0, 0.0)
    uniform_scale_range: tuple[float, float] = (1.0, 1.0)
    spawn_count_range: tuple[int, int] = (0, 0)
    spawn_size_range_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.25, 0.45),
        (0.25, 0.45),
        (0.25, 0.45),
    )
    collision_margin_m: float = 0.15
    max_attempts: int = 12
    color_value_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.65, 1.0),
        (0.65, 1.0),
        (0.65, 1.0),
    )
    required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WarehouseTemplate:
    template_id: str
    variant_id: str
    description: str
    shared_defaults_config_path: Path
    preset_config_path: Path | None
    source_template_usd_path: Path
    base_environment_usd: str
    nav2_map: TemplateMapSpec
    mapf_map: TemplateMapSpec
    light_randomizers: tuple[LightRandomizationSpec, ...]
    object_groups: tuple[ObjectGroupSpec, ...]
    placement_zones: tuple[PlacementZone, ...]
    keepout_zones: tuple[KeepoutZone, ...]
    object_randomizers: tuple[ObjectRandomizationSpec, ...]
    focus_group_names: tuple[str, ...]
    focus_distance_range_m: tuple[float, float]
    metadata: dict[str, Any]

    @property
    def template_config_path(self) -> Path:
        return self.shared_defaults_config_path

    @property
    def group_map(self) -> dict[str, ObjectGroupSpec]:
        return {group.group_id: group for group in self.object_groups}

    @property
    def zone_map(self) -> dict[str, PlacementZone]:
        return {zone.zone_id: zone for zone in self.placement_zones}

    @property
    def base_environment_usd_rel(self) -> str:
        return self.base_environment_usd

    @property
    def focus_object_selectors(self) -> tuple[SelectorSpec, ...]:
        selectors: list[SelectorSpec] = []
        group_map = self.group_map
        for group_name in self.focus_group_names:
            group = group_map.get(group_name)
            if group is None:
                continue
            selectors.extend(group.selectors)
        return tuple(selectors)


@dataclass(frozen=True)
class WarehouseTemplateAsset:
    template_id: str
    usd_path: Path


@dataclass(frozen=True)
class WarehouseSharedDefaults:
    shared_defaults_config_path: Path
    usd_root_dir: Path
    usd_filename_glob: str
    template_id_regex: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class WarehouseRandomizationPreset:
    variant_id: str
    description: str
    preset_config_path: Path
    focus_distance_range_m: tuple[float, float] | None
    light_randomizer_overrides: dict[str, dict[str, Any]]
    object_randomizer_overrides: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


def _parse_selectors(raw_value: Sequence[Any] | None, *, field_name: str) -> tuple[SelectorSpec, ...]:
    selectors: list[SelectorSpec] = []
    for index, item in enumerate(raw_value or ()):
        if isinstance(item, str):
            selectors.append(SelectorSpec(mode="exact_path", value=item))
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{index}] must be a mapping or string, got {item!r}.")
        mode = str(item.get("mode", "exact_path")).strip()
        value = str(item.get("value", "")).strip()
        if not mode or not value:
            raise ValueError(f"{field_name}[{index}] must define non-empty mode and value.")
        selectors.append(SelectorSpec(mode=mode, value=value))
    return tuple(selectors)


def _parse_map_spec(raw_value: Mapping[str, Any], *, field_name: str) -> TemplateMapSpec:
    if not isinstance(raw_value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return TemplateMapSpec(
        resolution_m=float(raw_value["resolution_m"]),
        origin_hint_xyz=_as_float_tuple(
            raw_value["origin_hint_xyz"], length=3, field_name=f"{field_name}.origin_hint_xyz"
        ),
        min_bound_xyz=_as_float_tuple(
            raw_value["min_bound_xyz"], length=3, field_name=f"{field_name}.min_bound_xyz"
        ),
        max_bound_xyz=_as_float_tuple(
            raw_value["max_bound_xyz"], length=3, field_name=f"{field_name}.max_bound_xyz"
        ),
    )


def _parse_keepout_zones(raw_value: Sequence[Mapping[str, Any]] | None) -> tuple[KeepoutZone, ...]:
    zones: list[KeepoutZone] = []
    for index, item in enumerate(raw_value or ()):
        if not isinstance(item, Mapping):
            raise ValueError(f"keepout_zones[{index}] must be a mapping.")
        zone_id = str(item.get("zone_id", "")).strip()
        if not zone_id:
            raise ValueError(f"keepout_zones[{index}] must define zone_id.")
        zones.append(
            KeepoutZone(
                zone_id=zone_id,
                min_xy=_as_float_tuple(item["min_xy"], length=2, field_name=f"keepout_zones[{index}].min_xy"),
                max_xy=_as_float_tuple(item["max_xy"], length=2, field_name=f"keepout_zones[{index}].max_xy"),
                description=str(item.get("description", "")).strip(),
            )
        )
    return tuple(zones)


def _parse_placement_zones(raw_value: Sequence[Mapping[str, Any]] | None) -> tuple[PlacementZone, ...]:
    zones: list[PlacementZone] = []
    for index, item in enumerate(raw_value or ()):
        if not isinstance(item, Mapping):
            raise ValueError(f"placement_zones[{index}] must be a mapping.")
        zone_id = str(item.get("zone_id", "")).strip()
        zone_type = str(item.get("zone_type", "")).strip()
        if not zone_id or not zone_type:
            raise ValueError(f"placement_zones[{index}] must define zone_id and zone_type.")
        zones.append(
            PlacementZone(
                zone_id=zone_id,
                zone_type=zone_type,
                min_xyz=_as_float_tuple(
                    item["min_xyz"], length=3, field_name=f"placement_zones[{index}].min_xyz"
                ),
                max_xyz=_as_float_tuple(
                    item["max_xyz"], length=3, field_name=f"placement_zones[{index}].max_xyz"
                ),
                description=str(item.get("description", "")).strip(),
            )
        )
    return tuple(zones)


def _parse_light_randomizers(
    raw_value: Sequence[Mapping[str, Any]] | None,
) -> tuple[LightRandomizationSpec, ...]:
    items: list[LightRandomizationSpec] = []
    for index, entry in enumerate(raw_value or ()):
        if not isinstance(entry, Mapping):
            raise ValueError(f"light_randomizers[{index}] must be a mapping.")
        items.append(
            LightRandomizationSpec(
                name=str(entry.get("name", f"light_{index + 1}")).strip(),
                selectors=_parse_selectors(
                    entry.get("selectors", ()),
                    field_name=f"light_randomizers[{index}].selectors",
                ),
                intensity_range=_as_range_tuple(
                    entry.get("intensity_range", (500.0, 900.0)),
                    field_name=f"light_randomizers[{index}].intensity_range",
                ),
                color_value_range=_as_rgb_range(
                    entry.get("color_value_range"),
                    field_name=f"light_randomizers[{index}].color_value_range",
                ),
                temperature_range=_as_range_tuple(
                    entry.get("temperature_range", (3300.0, 5200.0)),
                    field_name=f"light_randomizers[{index}].temperature_range",
                ),
            )
        )
    return tuple(items)


def _parse_object_groups(raw_value: Sequence[Mapping[str, Any]] | None) -> tuple[ObjectGroupSpec, ...]:
    groups: list[ObjectGroupSpec] = []
    for index, entry in enumerate(raw_value or ()):
        if not isinstance(entry, Mapping):
            raise ValueError(f"object_groups[{index}] must be a mapping.")
        group_id = str(entry.get("group_id", "")).strip()
        if not group_id:
            raise ValueError(f"object_groups[{index}] must define group_id.")
        groups.append(
            ObjectGroupSpec(
                group_id=group_id,
                selectors=_parse_selectors(
                    entry.get("selectors", ()),
                    field_name=f"object_groups[{index}].selectors",
                ),
                required=bool(entry.get("required", True)),
                root_mode=str(entry.get("root_mode", "nearest_xform")).strip() or "nearest_xform",
                description=str(entry.get("description", "")).strip(),
            )
        )
    return tuple(groups)


def _parse_object_randomizers(
    raw_value: Sequence[Mapping[str, Any]] | None,
) -> tuple[ObjectRandomizationSpec, ...]:
    items: list[ObjectRandomizationSpec] = []
    for index, entry in enumerate(raw_value or ()):
        if not isinstance(entry, Mapping):
            raise ValueError(f"object_randomizers[{index}] must be a mapping.")
        items.append(
            ObjectRandomizationSpec(
                name=str(entry.get("name", f"randomizer_{index + 1}")).strip(),
                policy=str(entry.get("policy", "")).strip(),
                target_group_name=str(entry.get("target_group_name", "")).strip(),
                source_group_name=str(entry.get("source_group_name", "")).strip(),
                selectors=_parse_selectors(
                    entry.get("selectors", ()),
                    field_name=f"object_randomizers[{index}].selectors",
                ),
                anchor_zone_ids=tuple(
                    str(value).strip()
                    for value in entry.get("anchor_zone_ids", ())
                    if str(value).strip()
                ),
                support_zone_ids=tuple(
                    str(value).strip()
                    for value in entry.get("support_zone_ids", ())
                    if str(value).strip()
                ),
                keepout_zone_ids=tuple(
                    str(value).strip()
                    for value in entry.get("keepout_zone_ids", ())
                    if str(value).strip()
                ),
                snapped_yaw_deg=tuple(float(value) for value in entry.get("snapped_yaw_deg", ())),
                xy_jitter_m=_as_range_tuple(
                    entry.get("xy_jitter_m", (0.0, 0.0)),
                    field_name=f"object_randomizers[{index}].xy_jitter_m",
                ),
                yaw_jitter_deg=_as_range_tuple(
                    entry.get("yaw_jitter_deg", (0.0, 0.0)),
                    field_name=f"object_randomizers[{index}].yaw_jitter_deg",
                ),
                uniform_scale_range=_as_range_tuple(
                    entry.get("uniform_scale_range", (1.0, 1.0)),
                    field_name=f"object_randomizers[{index}].uniform_scale_range",
                ),
                spawn_count_range=tuple(
                    int(value)
                    for value in _as_float_tuple(
                        entry.get("spawn_count_range", (0, 0)),
                        length=2,
                        field_name=f"object_randomizers[{index}].spawn_count_range",
                    )
                ),
                spawn_size_range_m=(
                    _as_range_tuple(
                        entry.get("spawn_size_range_m", ((0.25, 0.45), (0.25, 0.45), (0.25, 0.45)))[0],
                        field_name=f"object_randomizers[{index}].spawn_size_range_m[0]",
                    ),
                    _as_range_tuple(
                        entry.get("spawn_size_range_m", ((0.25, 0.45), (0.25, 0.45), (0.25, 0.45)))[1],
                        field_name=f"object_randomizers[{index}].spawn_size_range_m[1]",
                    ),
                    _as_range_tuple(
                        entry.get("spawn_size_range_m", ((0.25, 0.45), (0.25, 0.45), (0.25, 0.45)))[2],
                        field_name=f"object_randomizers[{index}].spawn_size_range_m[2]",
                    ),
                ),
                collision_margin_m=float(entry.get("collision_margin_m", 0.15)),
                max_attempts=int(entry.get("max_attempts", 12)),
                color_value_range=_as_rgb_range(
                    entry.get("color_value_range"),
                    field_name=f"object_randomizers[{index}].color_value_range",
                ),
                required=bool(entry.get("required", False)),
                metadata=dict(entry.get("metadata", {}) or {}),
            )
        )
    return tuple(items)


def _warehouse_template_from_payload(
    payload: Mapping[str, Any],
    *,
    shared_defaults_config_path: Path,
    preset_config_path: Path | None,
    source_template_usd_path: Path,
    variant_id: str,
) -> WarehouseTemplate:
    template_id = str(payload.get("template_id", "")).strip()
    if not template_id:
        raise ValueError("Resolved warehouse template payload is missing template_id.")

    base_environment_usd = str(
        payload.get("base_environment_usd", payload.get("base_environment_usd_rel", ""))
    ).strip()
    if not base_environment_usd:
        raise ValueError(f"Template '{template_id}' is missing base_environment_usd.")

    object_groups = _parse_object_groups(payload.get("object_groups"))
    if not object_groups:
        raise ValueError(f"Template '{template_id}' must define at least one object group.")

    focus_group_names = tuple(
        str(value).strip() for value in payload.get("focus_group_names", ()) if str(value).strip()
    )
    if not focus_group_names:
        if any(group.group_id == "focus_objects" for group in object_groups):
            focus_group_names = ("focus_objects",)
        elif any(group.group_id == "forklifts" for group in object_groups):
            focus_group_names = ("forklifts",)

    return WarehouseTemplate(
        template_id=template_id,
        variant_id=str(variant_id).strip() or DEFAULT_BASE_VARIANT_ID,
        description=str(payload.get("description", "")).strip(),
        shared_defaults_config_path=Path(shared_defaults_config_path).expanduser().resolve(),
        preset_config_path=(
            None if preset_config_path is None else Path(preset_config_path).expanduser().resolve()
        ),
        source_template_usd_path=Path(source_template_usd_path).expanduser().resolve(),
        base_environment_usd=base_environment_usd,
        nav2_map=_parse_map_spec(payload["nav2_map"], field_name="nav2_map"),
        mapf_map=_parse_map_spec(payload["mapf_map"], field_name="mapf_map"),
        light_randomizers=_parse_light_randomizers(payload.get("light_randomizers")),
        object_groups=object_groups,
        placement_zones=_parse_placement_zones(payload.get("placement_zones")),
        keepout_zones=_parse_keepout_zones(payload.get("keepout_zones")),
        object_randomizers=_parse_object_randomizers(payload.get("object_randomizers")),
        focus_group_names=focus_group_names,
        focus_distance_range_m=_as_range_tuple(
            payload.get("focus_distance_range_m", (2.5, 5.25)),
            field_name="focus_distance_range_m",
        ),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def _parse_named_override_map(
    raw_value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    if not raw_value:
        return overrides
    if not isinstance(raw_value, Mapping):
        raise ValueError(f"{field_name} must be a mapping of randomizer names to override mappings.")
    for name, override in raw_value.items():
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError(f"{field_name} contains an empty randomizer name.")
        if override is None:
            overrides[clean_name] = {}
            continue
        if not isinstance(override, Mapping):
            raise ValueError(f"{field_name}.{clean_name} must be a mapping.")
        overrides[clean_name] = deepcopy(dict(override))
    return overrides


def default_template_registry_dirs(repo_root: str | Path) -> list[Path]:
    repo_root_path = Path(repo_root).expanduser().resolve()
    return [repo_root_path / "isaac_sim" / "stage_bringups" / "warehouse_randomized" / DEFAULT_TEMPLATE_CONFIG_DIRNAME]


def _resolve_registry_dirs(
    repo_root: str | Path,
    template_registry_dirs: Sequence[str | Path] | None,
) -> list[Path]:
    repo_root_path = Path(repo_root).expanduser().resolve()
    raw_dirs = list(template_registry_dirs) if template_registry_dirs else default_template_registry_dirs(repo_root_path)

    resolved_dirs: list[Path] = []
    for raw_dir in _path_list(raw_dirs):
        clean_dir = raw_dir
        if not clean_dir.is_absolute():
            clean_dir = (repo_root_path / clean_dir).resolve()
        else:
            clean_dir = clean_dir.resolve()
        resolved_dirs.append(clean_dir)
    return resolved_dirs


def load_shared_warehouse_defaults(
    repo_root: str | Path,
    *,
    template_registry_dirs: Sequence[str | Path] | None = None,
) -> WarehouseSharedDefaults:
    searched_dirs = _resolve_registry_dirs(repo_root, template_registry_dirs)
    shared_config_path = None
    for registry_dir in searched_dirs:
        candidate = registry_dir / DEFAULT_SHARED_TEMPLATE_CONFIG_BASENAME
        if candidate.exists():
            shared_config_path = candidate
            break

    if shared_config_path is None:
        joined_dirs = ", ".join(str(path) for path in searched_dirs)
        raise RuntimeError(
            f"No shared warehouse template config named '{DEFAULT_SHARED_TEMPLATE_CONFIG_BASENAME}' found in: {joined_dirs}"
        )

    config_path, payload = _load_yaml_mapping(shared_config_path)
    template_assets = payload.get("template_assets", {})
    if not isinstance(template_assets, Mapping):
        raise ValueError(f"{config_path}: template_assets must be a mapping.")

    raw_usd_root_dir = str(template_assets.get("usd_root_dir", "")).strip()
    if not raw_usd_root_dir:
        raise ValueError(f"{config_path}: template_assets.usd_root_dir must be set.")

    usd_root_dir = _resolve_repo_path(
        repo_root,
        raw_usd_root_dir,
        base_dir=config_path.parent,
    )
    usd_filename_glob = str(template_assets.get("usd_filename_glob", "warehouse_template_*.usd")).strip()
    template_id_regex = str(
        template_assets.get("template_id_regex", r"^warehouse_template_(\d+)\.usd$")
    ).strip()
    if not usd_filename_glob or not template_id_regex:
        raise ValueError(f"{config_path}: template_assets must define usd_filename_glob and template_id_regex.")

    return WarehouseSharedDefaults(
        shared_defaults_config_path=config_path,
        usd_root_dir=usd_root_dir,
        usd_filename_glob=usd_filename_glob,
        template_id_regex=template_id_regex,
        payload=payload,
    )


def _normalize_template_id(raw_template_id: str) -> str:
    clean_template_id = str(raw_template_id).strip()
    if clean_template_id.isdigit():
        return str(int(clean_template_id))
    return clean_template_id


def discover_template_assets(
    repo_root: str | Path,
    *,
    shared_defaults: WarehouseSharedDefaults | None = None,
    template_registry_dirs: Sequence[str | Path] | None = None,
) -> dict[str, WarehouseTemplateAsset]:
    defaults = shared_defaults or load_shared_warehouse_defaults(
        repo_root,
        template_registry_dirs=template_registry_dirs,
    )
    pattern = re.compile(defaults.template_id_regex)

    assets: dict[str, WarehouseTemplateAsset] = {}
    for usd_path in sorted(defaults.usd_root_dir.glob(defaults.usd_filename_glob)):
        if not usd_path.is_file():
            continue
        match = pattern.fullmatch(usd_path.name)
        if match is None:
            continue
        if "id" in match.re.groupindex:
            template_id = match.group("id")
        elif match.lastindex:
            template_id = match.group(1)
        else:
            template_id = match.group(0)
        clean_template_id = _normalize_template_id(template_id)
        if clean_template_id in assets:
            raise RuntimeError(
                f"Duplicate warehouse template id '{clean_template_id}' discovered under {defaults.usd_root_dir}."
            )
        assets[clean_template_id] = WarehouseTemplateAsset(
            template_id=clean_template_id,
            usd_path=usd_path.resolve(),
        )

    if not assets:
        raise RuntimeError(
            f"No warehouse template USD assets discovered in {defaults.usd_root_dir} "
            f"matching '{defaults.usd_filename_glob}'."
        )
    return assets


def load_randomization_presets(
    repo_root: str | Path,
    *,
    template_registry_dirs: Sequence[str | Path] | None = None,
) -> dict[str, WarehouseRandomizationPreset]:
    presets: dict[str, WarehouseRandomizationPreset] = {}
    searched_dirs = _resolve_registry_dirs(repo_root, template_registry_dirs)
    for registry_dir in searched_dirs:
        if not registry_dir.exists():
            continue
        yaml_paths = sorted(registry_dir.glob("*.yaml")) + sorted(registry_dir.glob("*.yml"))
        for config_path in yaml_paths:
            if config_path.name == DEFAULT_SHARED_TEMPLATE_CONFIG_BASENAME:
                continue
            resolved_path, payload = _load_yaml_mapping(config_path)
            variant_id = str(payload.get("variant_id", "")).strip()
            if not variant_id:
                continue
            if variant_id == DEFAULT_BASE_VARIANT_ID:
                raise ValueError(
                    f"{resolved_path}: '{DEFAULT_BASE_VARIANT_ID}' is built-in and must not be defined as a preset."
                )
            presets[variant_id] = WarehouseRandomizationPreset(
                variant_id=variant_id,
                description=str(payload.get("description", "")).strip(),
                preset_config_path=resolved_path,
                focus_distance_range_m=(
                    None
                    if payload.get("focus_distance_range_m") is None
                    else _as_range_tuple(
                        payload["focus_distance_range_m"],
                        field_name=f"{resolved_path}.focus_distance_range_m",
                    )
                ),
                light_randomizer_overrides=_parse_named_override_map(
                    payload.get("light_randomizer_overrides"),
                    field_name=f"{resolved_path}.light_randomizer_overrides",
                ),
                object_randomizer_overrides=_parse_named_override_map(
                    payload.get("object_randomizer_overrides"),
                    field_name=f"{resolved_path}.object_randomizer_overrides",
                ),
                metadata=dict(payload.get("metadata", {}) or {}),
            )

    if not presets:
        joined_dirs = ", ".join(str(path) for path in searched_dirs)
        raise RuntimeError(f"No warehouse preset configs found in: {joined_dirs}")
    return presets


def _apply_randomizer_overrides(
    base_items: Sequence[Mapping[str, Any]] | None,
    overrides: Mapping[str, Mapping[str, Any]],
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    items_by_name: dict[str, dict[str, Any]] = {}
    ordered_names: list[str] = []
    for index, item in enumerate(base_items or ()):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{index}] must be a mapping.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"{field_name}[{index}] must define name.")
        items_by_name[name] = deepcopy(dict(item))
        ordered_names.append(name)

    unknown_override_names = sorted(name for name in overrides if name not in items_by_name)
    if unknown_override_names:
        raise KeyError(f"{field_name} overrides reference unknown randomizers: {unknown_override_names}")

    resolved_items: list[dict[str, Any]] = []
    for name in ordered_names:
        override = overrides.get(name, {})
        if not bool(override.get("enabled", True)):
            continue
        resolved_items.append(_merge_dicts(items_by_name[name], override))
    return resolved_items


def compose_warehouse_template(
    template_asset: WarehouseTemplateAsset,
    *,
    shared_defaults: WarehouseSharedDefaults,
    preset: WarehouseRandomizationPreset | None = None,
) -> WarehouseTemplate:
    variant_id = DEFAULT_BASE_VARIANT_ID if preset is None else str(preset.variant_id).strip()
    payload = deepcopy(shared_defaults.payload)
    payload["template_id"] = str(template_asset.template_id)
    payload["base_environment_usd"] = str(template_asset.usd_path)

    if preset is None:
        payload["description"] = (
            f"Base warehouse template {template_asset.template_id} without additional randomization."
        )
        payload["light_randomizers"] = []
        payload["object_randomizers"] = []
    else:
        payload["description"] = preset.description or str(payload.get("description", "")).strip()
        payload["light_randomizers"] = _apply_randomizer_overrides(
            payload.get("light_randomizers"),
            preset.light_randomizer_overrides,
            field_name="light_randomizers",
        )
        payload["object_randomizers"] = _apply_randomizer_overrides(
            payload.get("object_randomizers"),
            preset.object_randomizer_overrides,
            field_name="object_randomizers",
        )
        if preset.focus_distance_range_m is not None:
            payload["focus_distance_range_m"] = list(preset.focus_distance_range_m)

    metadata = dict(payload.get("metadata", {}) or {})
    metadata["variant_id"] = variant_id
    metadata["source_template_usd_path"] = str(template_asset.usd_path)
    if preset is None:
        metadata.setdefault("layout_class", DEFAULT_BASE_VARIANT_ID)
    else:
        metadata = _merge_dicts(metadata, preset.metadata)
    payload["metadata"] = metadata

    return _warehouse_template_from_payload(
        payload,
        shared_defaults_config_path=shared_defaults.shared_defaults_config_path,
        preset_config_path=None if preset is None else preset.preset_config_path,
        source_template_usd_path=template_asset.usd_path,
        variant_id=variant_id,
    )


def build_template_catalog(
    repo_root: str | Path,
    *,
    template_registry_dirs: Sequence[str | Path] | None = None,
    variant_ids: Sequence[str] | None = None,
) -> dict[str, WarehouseTemplate]:
    shared_defaults = load_shared_warehouse_defaults(
        repo_root,
        template_registry_dirs=template_registry_dirs,
    )
    template_assets = discover_template_assets(
        repo_root,
        shared_defaults=shared_defaults,
        template_registry_dirs=template_registry_dirs,
    )
    presets = load_randomization_presets(
        repo_root,
        template_registry_dirs=template_registry_dirs,
    )

    selected_variants = tuple(variant_ids or DEFAULT_VARIANT_IDS)
    catalog: dict[str, WarehouseTemplate] = {}
    for template_id in sorted(template_assets, key=lambda value: int(value) if str(value).isdigit() else value):
        template_asset = template_assets[template_id]
        for variant_id in selected_variants:
            preset = None if variant_id == DEFAULT_BASE_VARIANT_ID else presets[variant_id]
            catalog[f"{template_id}:{variant_id}"] = compose_warehouse_template(
                template_asset,
                shared_defaults=shared_defaults,
                preset=preset,
            )

    if not catalog:
        raise RuntimeError("No resolved warehouse templates were composed.")
    return catalog
