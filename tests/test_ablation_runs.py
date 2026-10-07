"""Offline tests for the read-only ablation-runs report."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HELPERS = Path(__file__).resolve().parents[1] / ".pi/prompts/scripts/vast-train"
sys.path.insert(0, str(HELPERS))
import ablation_runs
from ablation_runs import config_row, read_report, render_table, s3_objects


class ReportTests(unittest.TestCase):
    def make_iteration(self, *, root: Path) -> Path:
        iteration = root / "ablations/example/20261007T000000Z-aaaaaaaaaaaa"
        configs = iteration / "configs"
        configs.mkdir(parents=True)
        (iteration / "inputs").mkdir()
        (iteration / "inputs/000-example.toml").write_text(
            'description = "example"\n'
            'group = "configs/train/act_v2/BS-32"\n'
            "[fixed]\n"
            "lr = 1e-05\n"
            "[grid]\n"
            "use_z = [true, false]\n")
        (configs / "a.toml").write_text(
            'version = "v4"\n'
            "action_chunk_size = 100\n"
            "batch_size = 32\n"
            "steps = 1000\n"
            "lr = 1e-05\n"
            "use_z = true\n"
            'dataset = "datasets/x.hdf5"\n'
            "[rollout]\n"
            "n_action_steps = [10]\n")
        run_dir = root / "toy-act-111"
        run_dir.mkdir()
        (run_dir / "watcher.log").write_text(
            "[2026-10-07T00:00:00+00:00] progress: epoch 5 step 100/1000: 1%| | 1/10 "
            "[00:00<00:00, 10.00it/s, loss=0.5]\n")
        (run_dir / "watcher.pid").write_text("999999\n")
        (run_dir / "report.txt").write_text(
            "toy-act vast-train report\n"
            "generated: 2026-10-07T00:10:00+00:00\n"
            "outcome: success\n"
            "failure_reason: none\n"
            "elapsed_seconds: 3600\n"
            "s3_verification: verified 50 snapshots\n"
            "tensorboard_verification: verified: ok\n"
            "final_cleanup_status: destroyed_and_verified\n")
        (run_dir / "handoff.json").write_text(json.dumps({"status": "ready"}))
        (iteration / "manifest.json").write_text(json.dumps({
            "version": 1, "id": "20261007T000000Z-aaaaaaaaaaaa", "kind": "sweep",
            "created_at": "2026-10-07T00:00:00+00:00", "git_commit": "a" * 40,
            "source": "configs/sweep/example.toml",
            "inputs": [{"path": "inputs/000-example.toml",
                        "source": "configs/sweep/example.toml", "sha256": "b" * 64}],
            "combos": [{"id": "0000", "config": "configs/a.toml", "slug": "a"}]}))
        (iteration / "state.json").write_text(json.dumps({
            "version": 1, "driver_status": "handed_off",
            "combos": [{"id": "0000", "status": "done", "instance_id": 111,
                        "run_dir": str(run_dir), "remote_run_name": "run-a",
                        "run": {"offer": {"dph_total": 0.5}, "actual_price": 0.5}}]}))
        tensorboard = iteration / "tensorboard"
        (tensorboard / "logs/0000/run-a").mkdir(parents=True)
        (tensorboard / "logs/0000/run-a/events.out.tfevents.1").write_bytes(b"x" * 10)
        (tensorboard / "service.json").write_text(json.dumps(
            {"directory": str(iteration), "port": 16006, "url": "http://localhost:16006/",
             "server_session": "ablate-tb-x", "sync_session": "ablate-sync-x"}))
        (tensorboard / "sync-status.json").write_text(json.dumps(
            {"0000": {"checked_at": "2026-10-07T00:05:00+00:00", "ok": True, "run": "run-a"}}))
        return iteration

    def test_format_param_renders_nested_values(self) -> None:
        self.assertEqual(ablation_runs.format_param(value=[10, 100]), "10, 100")
        self.assertEqual(ablation_runs.format_param(value={"episodes": 30, "seed": 42}),
                         "episodes=30, seed=42")
        self.assertEqual(ablation_runs.format_param(value=True), "true")
        self.assertEqual(ablation_runs.format_param(value="datasets/x.hdf5"), "datasets/x.hdf5")

    def test_runs_report_lists_params_and_bucket_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration = self.make_iteration(root=Path(temporary))
            with patch.object(ablation_runs, "s3_folders", return_value=set()):
                text = ablation_runs.runs_report(directory=iteration)
            self.assertIn("## Runs", text)
            self.assertIn("## Fixed params", text)
            self.assertIn("## Ablated params", text)
            self.assertNotIn("Checkpoints folder", text)
            self.assertIn("s3://toy-act/runs/act_v2/run-a/", text)
            self.assertNotIn("s3://toy-act/checkpoints/act_v2/run-a/", text)
            self.assertIn("| use_z", text)
            self.assertIn("| true, false", text)

    def test_started_run_names_unions_state_and_s3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration = self.make_iteration(root=Path(temporary))
            discovered = {"2026-10-07_00-10-00_b-iaaaaaaaaaaaa",
                          "2026-10-07_00-20-00_c-iaaaaaaaaaaaa"}
            with patch.object(ablation_runs, "s3_folders", return_value=discovered) as folders:
                names = ablation_runs.started_run_names(
                    directory=iteration,
                    manifest=ablation_runs.read_json(path=iteration / "manifest.json"),
                    state=ablation_runs.read_json(path=iteration / "state.json"))
            self.assertEqual(set(names), discovered | {"run-a"})
            self.assertEqual(folders.call_count, 2)

    def test_run_name_from_record_recovers_from_report_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "report.txt").write_text("run_name: recovered-run\n")
            record = {"run_dir": str(run_dir), "status": "failed"}
            self.assertEqual(ablation_runs.run_name_from_record(record=record), "recovered-run")

    def test_runs_report_rejects_unsaved_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Not a saved iteration"):
                ablation_runs.runs_report(directory=Path(temporary))

    def test_table_alignment_and_headers(self) -> None:
        table = render_table(headers=["A", "BB"], rows=[["1", "2"], ["333", "4"]])
        self.assertEqual(table.splitlines()[0], "| A   | BB |")
        self.assertEqual(table.splitlines()[1], "| --- | -- |")
        self.assertEqual(table.splitlines()[2], "| 1   | 2  |")
        self.assertEqual(table.splitlines()[3], "| 333 | 4  |")

    def test_config_row_reads_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "a.toml"
            config.write_text('lr = 1e-05\nuse_z = false\nbatch_size = 8\n'
                              'dataset = "datasets/x.hdf5"\n'
                              "action_chunk_size = 10\nsteps = 100\n"
                              "[rollout]\nn_action_steps = [10, 100]\n")
            row = config_row(config_path=config)
            self.assertEqual(row["lr"], 1e-05)
            self.assertEqual(row["use_z"], False)
            self.assertEqual(row["n_action_steps"], "10,100")
            self.assertEqual(row["dataset"], "datasets/x.hdf5")

    def test_read_report_keeps_first_value_and_ignores_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "report.txt").write_text("header line\noutcome: success\noutcome: failure\n")
            report = read_report(run_dir=run_dir)
            self.assertEqual(report, {"outcome": "success"})

    def test_s3_objects_parses_listing_and_fails_closed(self) -> None:
        listing = ("2026-10-07 17:24:21  123 step_000002000.pt\n"
                   "2026-10-07 17:29:18  456 config.json\n")
        with patch.object(ablation_runs.subprocess, "run",
                          return_value=SimpleNamespace(returncode=0, stdout=listing)):
            objects = s3_objects(prefix="s3://toy-act/checkpoints/act_v2/run-a/")
        self.assertEqual(objects, [
            {"modified": "2026-10-07T17:24:21Z", "size": 123, "key": "step_000002000.pt"},
            {"modified": "2026-10-07T17:29:18Z", "size": 456, "key": "config.json"}])
        with patch.object(ablation_runs.subprocess, "run",
                          return_value=SimpleNamespace(returncode=1, stdout="")):
            self.assertIsNone(s3_objects(prefix="s3://toy-act/checkpoints/act_v2/run-a/"))

    def test_report_includes_every_section_and_links_s3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration = self.make_iteration(root=Path(temporary))

            def listing(*, prefix: str, timeout: float = 120):
                if "checkpoints" in prefix:
                    return [{"modified": "2026-10-07T00:05:00Z", "size": 1024, "key": "step_000001000.pt"}]
                return [{"modified": "2026-10-07T00:06:00Z", "size": 2048, "key": "events.out.tfevents.1"},
                        {"modified": "2026-10-07T00:06:01Z", "size": 10, "key": "config.json"},
                        {"modified": "2026-10-07T00:06:02Z", "size": 20, "key": "training-log-tail.txt"}]

            with patch.object(ablation_runs, "session_exists", return_value=False), \
                    patch.object(ablation_runs, "port_state", return_value="free"), \
                    patch.object(ablation_runs, "watcher_alive", return_value=True), \
                    patch.object(ablation_runs, "s3_objects", side_effect=listing):
                text = ablation_runs.report(directory=iteration, combo=None, files=True, no_s3=False)

            for heading in ("## Iteration", "## Configs", "## Execution", "## Verification",
                            "## Local records", "## S3 artifacts", "## S3 prefixes",
                            "## TensorBoard cache", "## File inventory"):
                self.assertIn(heading, text)
            self.assertIn("s3://toy-act/checkpoints/act_v2/run-a/", text)
            self.assertIn("step_000001000.pt", text)
            self.assertIn("| 0000  | done", text)
            self.assertIn("destroyed_and_verified", text)
            self.assertIn("| 0000  | local | report.txt", text)

    def test_report_no_s3_skips_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration = self.make_iteration(root=Path(temporary))
            with patch.object(ablation_runs, "port_state", return_value="free"):
                text = ablation_runs.report(directory=iteration, combo=None, files=False, no_s3=True)
            self.assertIn("skipped", text)
            self.assertNotIn("## File inventory", text)

    def test_report_rejects_unknown_combo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration = self.make_iteration(root=Path(temporary))
            with self.assertRaisesRegex(ValueError, "Unknown combo"):
                ablation_runs.report(directory=iteration, combo="9999", files=False, no_s3=True)


if __name__ == "__main__":
    unittest.main()
