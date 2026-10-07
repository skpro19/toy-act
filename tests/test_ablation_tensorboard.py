"""Offline tests for iteration-scoped TensorBoard services and S3 downloads."""

import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import urllib.parse
import urllib.request
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

    def test_sync_stages_downloads_and_publishes_only_successful_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "state.json").write_text(json.dumps({"combos": [
                {"id": "0000", "remote_run_name": "run-a"}]}))
            journal = Mock()

            def download(**kwargs):
                staging = Path(kwargs["args"][4])
                self.assertNotIn("logs", staging.parts)
                (staging / "events.out.tfevents.test").write_bytes(b"first snapshot")
                return SimpleNamespace(returncode=0)

            journal.run.side_effect = download
            dashboard.sync_once(directory=root, journal=journal)
            published = root / "tensorboard/logs/0000/run-a/events.out.tfevents.test"
            self.assertEqual(published.read_bytes(), b"first snapshot")
            inode = published.stat().st_ino
            journal.run.side_effect = None
            journal.run.return_value = SimpleNamespace(returncode=0)
            dashboard.sync_once(directory=root, journal=journal)
            self.assertEqual(published.stat().st_ino, inode)
            staging = root / "tensorboard/staging/0000/run-a/events.out.tfevents.test"
            staging.write_bytes(b"partial failed download")
            journal.run.return_value = SimpleNamespace(returncode=1)
            results = dashboard.sync_once(directory=root, journal=journal)
            self.assertFalse(results["0000"]["ok"])
            self.assertEqual(published.read_bytes(), b"first snapshot")
            journal.run.return_value = SimpleNamespace(returncode=0)
            staging.write_bytes(b"first snapshot plus new records")
            dashboard.sync_once(directory=root, journal=journal)
            self.assertEqual(published.read_bytes(), b"first snapshot plus new records")

    def test_snapshot_growth_preserves_inode_and_refuses_rewritten_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "snapshot"
            target = root / "logs/events.out.tfevents.test"
            source.write_bytes(b"existing records")
            dashboard.publish_snapshot(source=source, destination=target)
            inode = target.stat().st_ino
            source.write_bytes(b"existing records plus new records")
            dashboard.publish_snapshot(source=source, destination=target)
            self.assertEqual(target.stat().st_ino, inode)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            source.write_bytes(b"short")
            with self.assertRaisesRegex(Blocked, "shrank"):
                dashboard.publish_snapshot(source=source, destination=target)
            source.write_bytes(b"different records plus new records")
            with self.assertRaisesRegex(Blocked, "changed"):
                dashboard.publish_snapshot(source=source, destination=target)
            self.assertEqual(target.read_bytes(), b"existing records plus new records")

    def test_refresh_refuses_unrelated_session_before_stopping_anything(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "tensorboard").mkdir()
            service = {"directory": str(root),
                       "server_session": f"ablate-tb-{root.name}",
                       "sync_session": f"ablate-sync-{root.name}"}
            (root / "tensorboard/service.json").write_text(json.dumps(service))
            iteration = SimpleNamespace(manifest={"kind": "sweep"})
            journal = Mock()
            journal.run.return_value = SimpleNamespace(stdout="/unrelated")
            with patch("iteration.Iteration", return_value=iteration), \
                    patch.object(dashboard, "Journal", return_value=journal), \
                    patch.object(dashboard, "session_exists", return_value=True):
                with self.assertRaisesRegex(Blocked, "Unrelated"):
                    dashboard.refresh_dashboard(directory=root)
            self.assertEqual(journal.run.call_count, 1)
            self.assertNotIn("kill-session", journal.run.call_args.kwargs["args"])

    def test_live_server_reads_snapshot_growth_without_restart(self) -> None:
        try:
            from tensorboard.compat.proto import event_pb2, summary_pb2
            from tensorboard.summary.writer.record_writer import RecordWriter
        except ImportError:
            self.skipTest("TensorBoard requires the train dependency group")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tensorboard/staging").mkdir(parents=True)
            with socket.socket() as connection:
                connection.bind(("127.0.0.1", 0))
                port = connection.getsockname()[1]
            (root / "tensorboard/service.json").write_text(json.dumps({"port": port}))
            source = root / "tensorboard/staging/events.out.tfevents.test"
            target = root / "tensorboard/logs/test-run" / source.name

            def append_event(*, step: int) -> None:
                with source.open("ab") as output:
                    writer = RecordWriter(output)
                    event = event_pb2.Event(wall_time=time.time(), step=step,
                                           summary=summary_pb2.Summary(value=[
                                               summary_pb2.Summary.Value(tag="loss", simple_value=0.5)]))
                    writer.write(event.SerializeToString())
                    writer.flush()

            def wait_for_step(*, expected: int) -> None:
                query = urllib.parse.urlencode({"run": "test-run", "tag": "loss"})
                url = f"http://127.0.0.1:{port}/data/plugin/scalars/scalars?{query}"
                deadline = time.monotonic() + 30
                latest = -1
                while time.monotonic() < deadline:
                    self.assertIsNone(process.poll(), "Diagnostic TensorBoard exited")
                    try:
                        with urllib.request.urlopen(url, timeout=2) as response:
                            points = json.load(response)
                        latest = max((point[1] for point in points), default=-1)
                        if latest == expected:
                            return
                    except OSError:
                        pass
                    time.sleep(0.5)
                self.fail(f"Dashboard step {latest} did not reach {expected}")

            append_event(step=0)
            dashboard.publish_snapshot(source=source, destination=target)
            process = subprocess.Popen([sys.executable, str(HELPERS / "ablation_tensorboard.py"),
                                        "serve", str(root)], stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            try:
                wait_for_step(expected=0)
                inode = target.stat().st_ino
                for step in (50, 100):
                    append_event(step=step)
                    dashboard.publish_snapshot(source=source, destination=target)
                    self.assertEqual(target.stat().st_ino, inode)
                    wait_for_step(expected=step)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    def test_unrelated_session_is_not_reused(self) -> None:
        journal = Mock()
        journal.run.return_value = SimpleNamespace(stdout="/other/iteration")
        with patch.object(dashboard, "session_exists", return_value=True):
            with self.assertRaisesRegex(Blocked, "Unrelated"):
                dashboard.ensure_session(journal=journal, name="occupied", directory=Path("/expected"), command=["true"])
        self.assertEqual(journal.run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
