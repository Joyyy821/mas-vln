from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CARTERS_GOAL_SRC = REPO_ROOT / "ros2_ws" / "src" / "carters_goal"
sys.path.insert(0, str(CARTERS_GOAL_SRC))
try:
    from carters_goal import plot_mapf_tracking_logs
finally:
    try:
        sys.path.remove(str(CARTERS_GOAL_SRC))
    except ValueError:
        pass


class PlotMapfTrackingLogsTests(unittest.TestCase):
    def test_combined_only_writes_overlay_without_per_agent_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            logs_dir = tmp_path / "logs"
            output_dir = tmp_path / "plots"
            mpl_config_dir = tmp_path / "mpl"
            logs_dir.mkdir()
            output_dir.mkdir()
            mpl_config_dir.mkdir()
            self._write_log(logs_dir / "mapf_timed_tracker_pid1_exec001_nova_carter.csv", 0.0)
            self._write_log(logs_dir / "mapf_timed_tracker_pid1_exec001_jackal.csv", 1.0)

            argv = [
                "PlotMapfTrackingLogs",
                str(logs_dir),
                "--pattern",
                "mapf_timed_tracker_*.csv",
                "--output-dir",
                str(output_dir),
                "--no-show",
                "--combined-only",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(
                os.environ,
                {"MPLCONFIGDIR": str(mpl_config_dir)},
            ):
                exit_code = plot_mapf_tracking_logs.main()

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "combined_xy_overlay.png").is_file())
            self.assertFalse(
                (output_dir / "mapf_timed_tracker_pid1_exec001_nova_carter.png").exists()
            )
            self.assertFalse((output_dir / "mapf_timed_tracker_pid1_exec001_jackal.png").exists())

    def test_combined_only_with_one_log_exits_cleanly_without_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            logs_dir = tmp_path / "logs"
            output_dir = tmp_path / "plots"
            mpl_config_dir = tmp_path / "mpl"
            logs_dir.mkdir()
            output_dir.mkdir()
            mpl_config_dir.mkdir()
            self._write_log(logs_dir / "mapf_timed_tracker_pid1_exec001_nova_carter.csv", 0.0)

            argv = [
                "PlotMapfTrackingLogs",
                str(logs_dir),
                "--pattern",
                "mapf_timed_tracker_*.csv",
                "--output-dir",
                str(output_dir),
                "--no-show",
                "--combined-only",
            ]
            stdout = io.StringIO()
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(
                os.environ,
                {"MPLCONFIGDIR": str(mpl_config_dir)},
            ), redirect_stdout(stdout):
                exit_code = plot_mapf_tracking_logs.main()

            self.assertEqual(exit_code, 0)
            self.assertIn("requires at least two logs", stdout.getvalue())
            self.assertFalse((output_dir / "combined_xy_overlay.png").exists())

    def _write_log(self, path: Path, y_offset: float) -> None:
        path.write_text(
            "elapsed,ref_x,ref_y,ref_yaw,ref_linear_velocity,ref_angular_velocity,"
            "actual_x,actual_y,actual_yaw,cmd_linear_x,cmd_angular_z,"
            "position_error,yaw_error,linear_saturated,angular_saturated\n"
            f"0.0,0.0,{y_offset},0.0,0.1,0.0,0.0,{y_offset},0.0,0.1,0.0,0.0,0.0,0,0\n"
            f"1.0,1.0,{y_offset},0.0,0.1,0.0,0.9,{y_offset},0.0,0.1,0.0,0.1,0.0,0,0\n"
            f"2.0,2.0,{y_offset},0.0,0.1,0.0,1.9,{y_offset},0.0,0.1,0.0,0.1,0.0,0,0\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
