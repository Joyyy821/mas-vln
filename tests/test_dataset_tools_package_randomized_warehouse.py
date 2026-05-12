from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import yaml

from dataset_tools.package_randomized_warehouse import (
    PackageError,
    discover_release_inputs,
    package_randomized_warehouse,
    rollout_tar_name,
    scene_key,
)


class CapturingParquetWriter:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, object]]] = {}

    def __call__(self, rows: list[dict[str, object]], path: Path) -> None:
        self.tables[path.name] = list(rows)
        path.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")


class DatasetToolsPackageRandomizedWarehouseTest(unittest.TestCase):
    def test_discovery_skips_failed_non_numeric_and_missing_rollouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "randomized_warehouse"
            self._write_scene(root, 1)
            self._write_rollout(root, 1, 1)
            (root / "scene_1" / "rollouts" / "2_failed").mkdir()
            (root / "scene_1" / "rollouts" / "old").mkdir()
            (root / "scene_1" / "rollouts" / "3").mkdir()

            discovery = discover_release_inputs(root)

            self.assertEqual(len(discovery["scenes"]), 1)
            self.assertEqual(discovery["scenes"][0].rollouts[0].rollout_id, 1)
            self.assertEqual(len(discovery["skipped_failed_rollouts"]), 1)
            skipped_reasons = {item["reason"] for item in discovery["skipped_rollouts"]}
            self.assertIn("non_numeric_rollout_directory", skipped_reasons)
            self.assertIn("missing_required_rollout_items", skipped_reasons)

    def test_package_creates_zero_padded_tar_and_metadata_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "randomized_warehouse"
            out_root = Path(temp_dir) / "hf"
            self._write_scene(root, 2)
            self._write_rollout(root, 2, 4)
            writer = CapturingParquetWriter()

            release = package_randomized_warehouse(
                root,
                out_root,
                "v0.1.0",
                parquet_writer=writer,
            )

            tar_path = out_root / "rollouts" / scene_key(2) / rollout_tar_name(4)
            self.assertTrue(tar_path.is_file())
            with tarfile.open(tar_path, "r") as tar:
                names = set(tar.getnames())
            self.assertIn("run_config.yaml", names)
            self.assertIn("render_manifest.csv", names)
            self.assertIn("nova_carter_velocity.csv", names)
            self.assertIn("rgb/nova_carter/frame_100.png", names)
            self.assertIn("depth/nova_carter/frame_100.png", names)

            rollouts = writer.tables["rollouts.parquet"]
            frames = writer.tables["frames.parquet"]
            self.assertEqual(rollouts[0]["tar_path"], "rollouts/scene_002/rollout_004.tar")
            self.assertEqual(rollouts[0]["num_render_frames"], 1)
            self.assertEqual(frames[0]["nearest_velocity_timestamp_ns"], 100)
            self.assertEqual(frames[0]["x"], 1.0)
            self.assertEqual(frames[0]["cmd_vx"], 0.3)
            self.assertEqual(release["counts"]["rollouts"], 1)

    def test_append_mode_reuses_same_tar_and_rejects_changed_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "randomized_warehouse"
            out_root = Path(temp_dir) / "hf"
            self._write_scene(root, 1)
            self._write_rollout(root, 1, 1)
            writer = CapturingParquetWriter()

            first = package_randomized_warehouse(
                root,
                out_root,
                "v0.1.0",
                parquet_writer=writer,
            )
            second = package_randomized_warehouse(
                root,
                out_root,
                "v0.1.1",
                append=True,
                parquet_writer=writer,
            )
            self.assertEqual(first["counts"]["created_tars"], 1)
            self.assertEqual(second["counts"]["reused_existing_tars"], 1)

            (root / "scene_1" / "rollouts" / "1" / "rgb" / "nova_carter" / "frame_100.png").write_bytes(
                b"changed"
            )
            with self.assertRaises(PackageError):
                package_randomized_warehouse(
                    root,
                    out_root,
                    "v0.1.2",
                    append=True,
                    parquet_writer=writer,
                )

    def test_include_scene_usd_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "randomized_warehouse"
            out_root = Path(temp_dir) / "hf"
            self._write_scene(root, 3)
            self._write_rollout(root, 3, 1)
            writer = CapturingParquetWriter()

            package_randomized_warehouse(root, out_root, "v0.1.0", parquet_writer=writer)
            self.assertFalse((out_root / "scenes" / "scene_003" / "scene.usd").exists())

            package_randomized_warehouse(
                root,
                out_root,
                "v0.1.1",
                include_scene_usd=True,
                overwrite_rollout_tars=True,
                parquet_writer=writer,
            )
            self.assertTrue((out_root / "scenes" / "scene_003" / "scene.usd").is_file())

    def _write_scene(self, root: Path, scene_id: int) -> None:
        scene_dir = root / f"scene_{scene_id}"
        scene_dir.mkdir(parents=True)
        (scene_dir / "rollouts").mkdir()
        for filename in ("mapf_map.png", "nav2_map.png", "scene.usd"):
            (scene_dir / filename).write_bytes(filename.encode("utf-8"))
        for filename in ("mapf_map.yaml", "nav2_map.yaml"):
            (scene_dir / filename).write_text("image: map.png\n", encoding="utf-8")
        (scene_dir / "scene_manifest.yaml").write_text(
            yaml.safe_dump({"language_instruction": "go to the shelf"}, sort_keys=False),
            encoding="utf-8",
        )
        (scene_dir / "team_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "robots": [
                        {
                            "name": "nova_carter",
                            "model": "nova_carter",
                        }
                    ]
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def _write_rollout(self, root: Path, scene_id: int, rollout_id: int) -> None:
        rollout_dir = root / f"scene_{scene_id}" / "rollouts" / str(rollout_id)
        rollout_dir.mkdir(parents=True)
        (rollout_dir / "rgb" / "nova_carter").mkdir(parents=True)
        (rollout_dir / "depth" / "nova_carter").mkdir(parents=True)
        (rollout_dir / "rgb" / "nova_carter" / "frame_100.png").write_bytes(b"rgb")
        (rollout_dir / "depth" / "nova_carter" / "frame_100.png").write_bytes(b"depth")
        (rollout_dir / "render_manifest.csv").write_text(
            "frame_index,timestamp_ns,elapsed_s,camera_name,camera_type,camera_prim_path,"
            "selection_mode,rgb_path,depth_path\n"
            "0,100,0.0,nova_carter,robot,/World/Camera,asset,"
            "rgb/nova_carter/frame_100.png,depth/nova_carter/frame_100.png\n",
            encoding="utf-8",
        )
        (rollout_dir / "nova_carter_velocity.csv").write_text(
            "timestamp_ns,vx,vy,wz,x,y,yaw,cmd_vel_timestamp_ns,cmd_vx,cmd_vy,cmd_wz\n"
            "100,0.1,0.0,0.2,1.0,2.0,0.5,90,0.3,0.0,0.4\n",
            encoding="utf-8",
        )
        (rollout_dir / "run_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "run_id": rollout_id,
                    "language_instruction": "go to the shelf",
                    "team_config": {
                        "robots": [
                            {
                                "name": "nova_carter",
                                "model": "nova_carter",
                                "initial_pose": {"x": 0, "y": 0, "z": 0, "yaw": 0},
                                "goal_pose": {"x": 1, "y": 0, "z": 0, "yaw": 0},
                            }
                        ]
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
