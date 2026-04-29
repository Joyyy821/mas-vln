from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import yaml

from isaac_sim.goal_generator.object_goal_sampler_utils import OccupancyMap
from isaac_sim.stage_bringups.warehouse_randomized import maps as maps_module
from isaac_sim.stage_bringups.warehouse_randomized.maps import (
    MapExportResult,
    _export_occupancy_map,
    _export_resampled_occupancy_map,
    sample_multi_robot_rollouts,
)


@contextmanager
def _fake_omap_modules(generator_cls):
    module_names = [
        "omni",
        "omni.physx",
        "omni.usd",
        "isaacsim",
        "isaacsim.asset",
        "isaacsim.asset.gen",
        "isaacsim.asset.gen.omap",
        "isaacsim.asset.gen.omap.bindings",
    ]
    originals = {name: sys.modules.get(name) for name in module_names}

    omni = ModuleType("omni")
    omni.__path__ = []
    omni_physx = ModuleType("omni.physx")
    omni_usd = ModuleType("omni.usd")
    omni_physx.get_physx_interface = lambda: object()
    omni_usd.get_context = lambda: SimpleNamespace(get_stage_id=lambda: 41)
    omni.physx = omni_physx
    omni.usd = omni_usd

    isaacsim = ModuleType("isaacsim")
    isaacsim.__path__ = []
    isaacsim_asset = ModuleType("isaacsim.asset")
    isaacsim_asset.__path__ = []
    isaacsim_asset_gen = ModuleType("isaacsim.asset.gen")
    isaacsim_asset_gen.__path__ = []
    isaacsim_omap = ModuleType("isaacsim.asset.gen.omap")
    isaacsim_omap.__path__ = []
    isaacsim_bindings = ModuleType("isaacsim.asset.gen.omap.bindings")
    isaacsim_bindings._omap = SimpleNamespace(Generator=generator_cls)

    sys.modules["omni"] = omni
    sys.modules["omni.physx"] = omni_physx
    sys.modules["omni.usd"] = omni_usd
    sys.modules["isaacsim"] = isaacsim
    sys.modules["isaacsim.asset"] = isaacsim_asset
    sys.modules["isaacsim.asset.gen"] = isaacsim_asset_gen
    sys.modules["isaacsim.asset.gen.omap"] = isaacsim_omap
    sys.modules["isaacsim.asset.gen.omap.bindings"] = isaacsim_bindings

    try:
        yield
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class _FakeGenerator:
    instances: list["_FakeGenerator"] = []

    def __init__(self, physx, stage_id: int) -> None:
        self.physx = physx
        self.stage_id = stage_id
        self.settings = None
        self.transform = None
        self.generated = False
        self.instances.append(self)

    def update_settings(self, resolution, occupied, free, unknown) -> None:
        self.settings = (resolution, occupied, free, unknown)

    def set_transform(self, origin, min_bound, max_bound) -> None:
        self.transform = (origin, min_bound, max_bound)

    def generate2d(self) -> None:
        self.generated = True

    def get_buffer(self):
        return [0, 254, 205, 254]

    def get_dimensions(self):
        return (2, 2, 1)


class _BadBufferGenerator(_FakeGenerator):
    def get_buffer(self):
        return [0, 254, 205]


def _free_occupancy_map(width: int = 20, height: int = 20, resolution_m: float = 1.0) -> OccupancyMap:
    grayscale = np.ones((height, width), dtype=float)
    occupied = np.zeros((height, width), dtype=bool)
    free = np.ones((height, width), dtype=bool)
    unknown = np.zeros((height, width), dtype=bool)
    return OccupancyMap(
        image_path="",
        resolution_m=float(resolution_m),
        origin_xyz=np.array([0.0, 0.0, 0.0], dtype=float),
        negate=False,
        occupied_thresh=0.65,
        free_thresh=0.196,
        grayscale_01=grayscale,
        occupancy_probability=np.zeros_like(grayscale, dtype=float),
        occupied_mask=occupied,
        free_mask=free,
        unknown_mask=unknown,
    )


class WarehouseRandomizedMapsTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeGenerator.instances = []

    def test_export_occupancy_map_uses_omap_generator_and_records_debug(self) -> None:
        original_bounds = maps_module._compute_world_xy_bounds
        maps_module._compute_world_xy_bounds = lambda env_prim_path, z_min=0.1, z_max=0.62: (
            (-1.0, -2.0, z_min),
            (1.0, 0.0, z_max),
        )
        try:
            with tempfile.TemporaryDirectory() as tmpdir, _fake_omap_modules(_FakeGenerator):
                yaml_path = Path(tmpdir) / "omap.yaml"
                result = _export_occupancy_map(
                    env_prim_path="/World/Env/Warehouse",
                    yaml_path=yaml_path,
                    resolution_m=1.0,
                    origin_xyz=(0.0, 0.0, 0.0),
                    z_min=0.1,
                    z_max=0.62,
                )
                payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        finally:
            maps_module._compute_world_xy_bounds = original_bounds

        generator = _FakeGenerator.instances[0]
        self.assertEqual(generator.stage_id, 41)
        self.assertEqual(generator.settings, (1.0, 0, 254, 205))
        self.assertEqual(
            generator.transform,
            ((0.0, 0.0, 0.0), (-1.0, -2.0, 0.1), (1.0, 0.0, 0.62)),
        )
        self.assertTrue(generator.generated)

        self.assertEqual(result.width_px, 2)
        self.assertEqual(result.height_px, 2)
        self.assertEqual(result.occupied_cells, 1)
        self.assertEqual(result.free_cells, 2)
        self.assertEqual(result.unknown_cells, 1)
        self.assertEqual(result.origin_xyz, (-1.0, -2.0, 0.0))
        self.assertEqual(result.debug["raw_pixel_histogram"], {0: 1, 205: 1, 254: 2})
        self.assertEqual(result.debug["generator_dimensions"], [2, 2])
        self.assertEqual(result.debug["generator_buffer_size"], 4)

        self.assertEqual(payload["origin"], [-1.0, -2.0, 0.0])

    def test_export_occupancy_map_rejects_buffer_dimension_mismatch(self) -> None:
        original_bounds = maps_module._compute_world_xy_bounds
        maps_module._compute_world_xy_bounds = lambda env_prim_path, z_min=0.1, z_max=0.62: (
            (-1.0, -2.0, z_min),
            (1.0, 0.0, z_max),
        )
        try:
            with tempfile.TemporaryDirectory() as tmpdir, _fake_omap_modules(_BadBufferGenerator):
                with self.assertRaisesRegex(RuntimeError, "inconsistent dimensions"):
                    _export_occupancy_map(
                        env_prim_path="/World/Env/Warehouse",
                        yaml_path=Path(tmpdir) / "omap.yaml",
                        resolution_m=1.0,
                    )
        finally:
            maps_module._compute_world_xy_bounds = original_bounds

    def test_export_resampled_occupancy_map_preserves_obstacles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_yaml_path = tmp_path / "source.yaml"
            source_png_path = tmp_path / "source.png"
            source_image = np.array(
                [
                    [255, 255, 255, 255],
                    [255, 0, 255, 255],
                    [255, 255, 205, 255],
                    [255, 255, 255, 255],
                ],
                dtype=np.uint8,
            )
            maps_module.Image.fromarray(source_image, mode="L").save(source_png_path)
            source_yaml_path.write_text(
                "\n".join(
                    [
                        "image: source.png",
                        "resolution: 0.5",
                        "origin: [0.0, 0.0, 0.0]",
                        "negate: 0",
                        "occupied_thresh: 0.65",
                        "free_thresh: 0.196",
                    ]
                ),
                encoding="utf-8",
            )
            source_result = MapExportResult(
                yaml_path=source_yaml_path,
                png_path=source_png_path,
                resolution_m=0.5,
                origin_xyz=(0.0, 0.0, 0.0),
                min_bound_xyz=(0.0, 0.0, 0.1),
                max_bound_xyz=(2.0, 2.0, 0.62),
                width_px=4,
                height_px=4,
                occupied_cells=1,
                free_cells=14,
                unknown_cells=1,
            )

            result = _export_resampled_occupancy_map(
                yaml_path=tmp_path / "resampled.yaml",
                source_result=source_result,
                resolution_m=1.0,
            )

            resampled = np.asarray(maps_module.Image.open(result.png_path).convert("L"), dtype=np.uint8)

        self.assertEqual(result.width_px, 2)
        self.assertEqual(result.height_px, 2)
        self.assertEqual(result.occupied_cells, 1)
        self.assertEqual(result.unknown_cells, 1)
        self.assertEqual(result.free_cells, 2)
        self.assertEqual(resampled[0, 0], 0)
        self.assertEqual(resampled[1, 1], 205)

    def test_rollout_sampler_excludes_initial_poses_near_focus_area(self) -> None:
        occupancy_map = _free_occupancy_map()
        focus_xy = np.array([5.0, 5.0], dtype=float)
        rollouts, validation = sample_multi_robot_rollouts(
            occupancy_map=occupancy_map,
            robot_names=["robot1", "robot2", "robot3"],
            rollout_count=3,
            rng=np.random.default_rng(7),
            inflation_radius_m=0.0,
            min_pairwise_distance_m=1.0,
            min_goal_distance_m=2.0,
            focus_xy=focus_xy,
            focus_distance_range_m=(2.0, 3.0),
            min_initial_focus_distance_m=6.0,
        )

        self.assertEqual(len(rollouts), 3)
        self.assertGreater(validation["initial_pose_candidates"], 0)
        self.assertGreater(validation["goal_pose_candidates"], 0)
        self.assertEqual(validation["selected_focus_xy"], [5.0, 5.0])
        self.assertGreaterEqual(validation["observed_min_initial_focus_distance_m"], 6.0)
        self.assertGreaterEqual(validation["observed_min_initial_goal_distance_m"], 2.0)
        for rollout in rollouts:
            for robot in rollout["robots"]:
                initial = robot["initial_pose"]
                distance_to_focus = float(
                    np.linalg.norm(np.array([initial["x"], initial["y"]], dtype=float) - focus_xy)
                )
                self.assertGreaterEqual(distance_to_focus, 6.0)

    def test_rollout_sampler_fails_when_focus_annulus_has_no_candidates(self) -> None:
        occupancy_map = _free_occupancy_map()

        with self.assertRaisesRegex(RuntimeError, "focus-distance range"):
            sample_multi_robot_rollouts(
                occupancy_map=occupancy_map,
                robot_names=["robot1", "robot2", "robot3"],
                rollout_count=1,
                rng=np.random.default_rng(7),
                inflation_radius_m=0.0,
                min_pairwise_distance_m=1.0,
                min_goal_distance_m=2.0,
                focus_xy=(5.0, 5.0),
                focus_distance_range_m=(50.0, 60.0),
            )


if __name__ == "__main__":
    unittest.main()
