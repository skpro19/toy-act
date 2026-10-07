"""Offline tests for iteration-scoped TensorBoard services and S3 downloads."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HELPERS = Path(__file__).resolve().parents[1] / ".pi/prompts/scripts/vast-train"
sys.path.insert(0, str(HELPERS))
import ablation_tensorboard as dashboard
from workflow_common import Blocked


class DashboardTests(unittest.TestCase):
    def make_iteration(self, *, directory: Path, kind: str = "sweep"):
        directory.mkdir(parents=True, exist_ok=True)
        journal = Mock()
        journal.run.return_value = SimpleNamespace(returncode=0, stdout="")
        return SimpleNamespace(directory=directory, manifest={"kind": kind},
                               state={}, journal=journal, save=Mock())

    def test_new_iterations_are_isolated_and_resume_reuses_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.make_iteration(directory=root / "first")
            second = self.make_iteration(directory=root / "second")
            with patch.object(dashboard, "REPO", root), \
                    patch.object(dashboard, "session_exists", return_value=False), \
                    patch.object(dashboard, "port_available", side_effect=lambda **kw: True), \
                    patch.object(dashboard, "ensure_session") as sessions:
                dashboard.ensure_dashboard(first)
                saved = first.state["tensorboard"].copy()
                # Model the first server's occupied port during the next invocation.
                with patch.object(dashboard, "port_available", side_effect=lambda **kw: kw["port"] != saved["port"]):
                    dashboard.ensure_dashboard(second)
                dashboard.ensure_dashboard(first)
            self.assertEqual(first.state["tensorboard"], saved)
            self.assertNotEqual(saved["port"], second.state["tensorboard"]["port"])
            self.assertNotEqual(saved["server_session"], second.state["tensorboard"]["server_session"])
            self.assertEqual(sessions.call_count, 6)

    def test_summary_reports_same_dashboard_for_every_combo(self) -> None:
        from iteration import Iteration

        with tempfile.TemporaryDirectory() as temporary:
            iteration = Iteration.__new__(Iteration)
            iteration.directory = Path(temporary)
            iteration.manifest = {"git_commit": "a" * 40, "combos": [
                {"slug": "first"}, {"slug": "second"}]}
            url = "http://localhost:16006/"
            iteration.state = {"driver_status": "running", "tensorboard": {"url": url},
                               "combos": [{"id": "0000", "status": "running", "lease": {"TB_PORT": 6007}},
                                          {"id": "0001", "status": "running", "lease": {"TB_PORT": 6008}}]}
            summary = iteration.summary()
            self.assertEqual(summary.count(url), 3)
            self.assertNotIn("localhost:6007", summary)
            self.assertNotIn("localhost:6008", summary)

    def test_single_config_does_not_start_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration = self.make_iteration(directory=Path(temporary), kind="single")
            dashboard.ensure_dashboard(iteration)
            iteration.save.assert_not_called()
            iteration.journal.run.assert_not_called()

    def test_recovery_never_steals_occupied_port(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iteration = self.make_iteration(directory=root / "iteration")
            with patch.object(dashboard, "REPO", root), \
                    patch.object(dashboard, "session_exists", return_value=False), \
                    patch.object(dashboard, "port_available", return_value=True), \
                    patch.object(dashboard, "ensure_session"):
                dashboard.ensure_dashboard(iteration)
            with patch.object(dashboard, "REPO", root), \
                    patch.object(dashboard, "session_exists", return_value=False), \
                    patch.object(dashboard, "port_available", return_value=False):
                with self.assertRaisesRegex(Blocked, "occupied"):
                    dashboard.ensure_dashboard(iteration)

    def test_sync_uses_only_registered_events_and_retries_failed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "state.json").write_text(json.dumps({"combos": [
                {"id": "0000", "remote_run_name": "run-a", "status": "running"},
                {"id": "0001", "remote_run_name": "run-b", "status": "done"},
                {"id": "0002", "status": "pending"}]}))
            journal = Mock()
            journal.run.side_effect = [SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]
            results = dashboard.sync_once(directory=root, journal=journal)
            self.assertFalse(results["0000"]["ok"])
            self.assertTrue(results["0001"]["ok"])
            self.assertEqual(journal.run.call_count, 2)
            command = journal.run.call_args_list[0].kwargs["args"]
            self.assertIn("s3://toy-act/runs/act_v2/run-a/", command)
            self.assertIn("*tfevents*", command)
            self.assertNotIn("--delete", command)
            self.assertIn("toy-pickplace-backup", command)
            journal.run.side_effect = None
            journal.run.return_value = SimpleNamespace(returncode=0)
            self.assertTrue(dashboard.sync_once(directory=root, journal=journal)["0000"]["ok"])

    def test_unrelated_session_is_not_reused(self) -> None:
        journal = Mock()
        journal.run.return_value = SimpleNamespace(stdout="/other/iteration")
        with patch.object(dashboard, "session_exists", return_value=True):
            with self.assertRaisesRegex(Blocked, "Unrelated"):
                dashboard.ensure_session(journal=journal, name="occupied", directory=Path("/expected"), command=["true"])
        self.assertEqual(journal.run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
