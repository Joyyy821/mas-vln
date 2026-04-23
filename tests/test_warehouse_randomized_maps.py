from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from isaac_sim.goal_generator.object_goal_sampler_utils import OccupancyMap
from isaac_sim.stage_bringups.warehouse_randomized.maps import export_bbox_occupancy_map


class WarehouseRandomizedMapsTests(unittest.TestCase):
    def test_export_bbox_occupancy_map_marks_obstacle_cells_and_no_unknowns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "bbox_map.yaml"
            result = export_bbox_occupancy_map(
                yaml_path=yaml_path,
                resolution_m=1.0,
                origin_hint_xyz=(0.0, 0.0, 0.0),
                min_bound_xyz=(0.0, 0.0, 0.0),
                max_bound_xyz=(5.0, 5.0, 2.0),
                obstacle_bboxes=(
                    ((1.0, 1.0, 0.0), (3.0, 3.0, 2.0)),
                ),
            )

            self.assertEqual(result.width_px, 5)
            self.assertEqual(result.height_px, 5)
            self.assertEqual(result.unknown_cells, 0)
            self.assertEqual(result.occupied_cells, 4)
            self.assertEqual(result.free_cells, 21)

            occupancy_map = OccupancyMap.load(str(yaml_path), treat_unknown_as_occupied=False)
            occupied_row, occupied_col = occupancy_map.world_to_grid(1.5, 1.5)
            free_row, free_col = occupancy_map.world_to_grid(4.5, 4.5)

            self.assertTrue(occupancy_map.occupied_mask[occupied_row, occupied_col])
            self.assertTrue(occupancy_map.free_mask[free_row, free_col])


if __name__ == "__main__":
    unittest.main()
