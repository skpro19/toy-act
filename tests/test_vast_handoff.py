"""The handoff gate is read-only and requires every readiness condition."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch


HELPER = Path(__file__).resolve().parents[1] / ".pi/prompts/scripts/vast-train/check-handoff.py"
SPEC = importlib.util.spec_from_file_location("vast_handoff", HELPER)
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


class HandoffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name)
        self.config = {
            key: "" for key in handoff.SETUP_KEYS
        }
        self.config.update(
            INSTANCE_ID="123", SSH_HOST="example.invalid", SSH_PORT="2222",
            TB_URL="http://localhost:6007/", TB_SESSION="act-tb-1")
        self.remote = {key: "yes" for key in handoff.REMOTE_KEYS}
        self.remote.update(run_status="running", completed="no", failed="no", backup_failed="no")
        self.commands = []
        self.ssh_error = None
        self.local_failure = None
        self.write_ack()

    def write_ack(self) -> None:
        pid = os.getpid()
        ticks = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        (self.run_dir / "watcher.pid").write_text(str(pid))
        (self.run_dir / "watcher.ready").write_text(
            f"instance_id=123\npid={pid}\nprocess_start_ticks={ticks}\nobserved_at=1\n")

    def fake_run(self, args, **kwargs):
        self.commands.append(args)
        self.assertGreater(kwargs["timeout"], 0)
        if args[0] == "ssh":
            if self.ssh_error is not None:
                raise self.ssh_error
            output = "\n".join(f"{key}={value}" for key, value in self.remote.items())
            return subprocess.CompletedProcess(args, 0, output, "")
        return subprocess.CompletedProcess(args, int(args[0] == self.local_failure), "", "")

    def check(self) -> dict:
        with patch.object(handoff.subprocess, "run", side_effect=self.fake_run):
            report = handoff.check_once(
                run_dir=self.run_dir, config=self.config, deadline=handoff.Deadline(seconds=35))
        # The helper has no Vast API call or commands that mutate training/leases.
        self.assertTrue(all(command[0] in ("ssh", "curl", "tmux") for command in self.commands))
        self.assertNotIn("destroy", str(self.commands))
        self.assertNotIn("kill-session", str(self.commands))
        self.assertNotIn("release", str(self.commands))
        return report

    def test_all_conditions_pass(self) -> None:
        self.assertEqual(self.check()["status"], "ready")
        ssh = next(command for command in self.commands if command[0] == "ssh")
        self.assertIn("StrictHostKeyChecking=yes", ssh)
        self.assertIn("BatchMode=yes", ssh)

    def test_each_missing_remote_condition_blocks_handoff(self) -> None:
        for key in ("train_session", "backup_session", "backup_running",
                    "backup_artifact_ready", "backup_last_succeeded"):
            with self.subTest(key=key):
                self.remote[key] = "no"
                self.assertEqual(self.check()["status"], "not_ready")
                self.remote[key] = "yes"
        self.remote["run_status"] = "starting"
        self.assertEqual(self.check()["status"], "not_ready")

    def test_each_local_condition_blocks_handoff(self) -> None:
        for command in ("curl", "tmux"):
            with self.subTest(command=command):
                self.local_failure = command
                self.assertEqual(self.check()["status"], "not_ready")

    def test_backup_failure_blocks_handoff(self) -> None:
        self.remote["backup_failed"] = "yes"
        self.assertEqual(self.check()["checks"]["backup_health"]["status"], "failed")

    def test_terminal_markers_are_distinct_from_not_ready(self) -> None:
        for marker in ("completed", "failed"):
            with self.subTest(marker=marker):
                self.remote[marker] = "yes"
                self.assertEqual(self.check()["status"], "terminal")
                self.remote[marker] = "no"

    def test_connectivity_failure_is_unknown(self) -> None:
        self.ssh_error = subprocess.TimeoutExpired("ssh", 20)
        report = self.check()
        self.assertEqual(report["status"], "not_ready")
        self.assertEqual(report["checks"]["remote"]["status"], "unknown")

    def test_malformed_remote_output_is_unknown(self) -> None:
        self.remote.pop("backup_last_succeeded")
        self.assertEqual(self.check()["checks"]["remote"]["status"], "unknown")

    def test_missing_acknowledgement_is_pending(self) -> None:
        (self.run_dir / "watcher.ready").unlink()
        self.assertEqual(self.check()["checks"]["watcher"]["status"], "pending")

    def test_stale_instance_or_pid_acknowledgement_fails(self) -> None:
        for original, replacement in (("instance_id=123", "instance_id=456"),
                                      (f"pid={os.getpid()}", "pid=999999")):
            with self.subTest(replacement=replacement):
                self.write_ack()
                path = self.run_dir / "watcher.ready"
                path.write_text(path.read_text().replace(original, replacement))
                self.assertEqual(self.check()["checks"]["watcher"]["status"], "failed")

    def test_pid_reuse_fails(self) -> None:
        path = self.run_dir / "watcher.ready"
        lines = path.read_text().splitlines()
        path.write_text("\n".join("process_start_ticks=0" if line.startswith("process_start_ticks=") else line for line in lines))
        self.assertEqual(self.check()["checks"]["watcher"]["status"], "failed")

    def test_dead_watcher_fails(self) -> None:
        with patch.object(handoff.os, "kill", side_effect=ProcessLookupError):
            self.assertEqual(self.check()["checks"]["watcher"]["status"], "failed")

    def test_expired_overall_deadline_does_not_run_commands(self) -> None:
        deadline = handoff.Deadline(seconds=0)
        with patch.object(handoff.subprocess, "run") as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                deadline.run(args=["ssh"], maximum=20)
            run.assert_not_called()

    def test_setup_loads_only_allowlisted_fields(self) -> None:
        setup = self.run_dir / "setup.env"
        setup.write_text(
            "INSTANCE_ID=123\nSSH_HOST=example.invalid\nSSH_PORT=2222\n"
            "TB_URL=http://localhost:6007/\nTB_SESSION=act-tb-1\nSECRET=must-not-return\n")
        config = handoff.load_setup(run_dir=self.run_dir, deadline=handoff.Deadline(seconds=5))
        self.assertEqual(config["INSTANCE_ID"], "123")
        self.assertNotIn("SECRET", config)
        setup.write_text(setup.read_text().replace("http://localhost:6007/", "http://example.invalid/"))
        with self.assertRaises(ValueError):
            handoff.load_setup(run_dir=self.run_dir, deadline=handoff.Deadline(seconds=5))

    def test_real_watcher_publishes_ack_only_after_arming_gate(self) -> None:
        bin_dir = self.run_dir / "bin"
        bin_dir.mkdir()
        ssh = bin_dir / "ssh"
        ssh.write_text(
            "#!/bin/bash\nprintf '%s\\n' TRAIN_SESSION=yes COMPLETED=no FAILED=no RUN_STATUS=running RUN_NAME=test LAST_LOG=training\n")
        ssh.chmod(0o755)
        (self.run_dir / "setup.env").write_text(
            "INSTANCE_ID=123\nINSTANCE_LABEL=test-instance\nSSH_HOST=example.invalid\n"
            "SSH_PORT=2222\nSSH_KNOWN_HOSTS=\nCHECKPOINT_EVERY=10\nSTEPS=100\n"
            "UV=/usr/bin/true\nWATCHER_DRY_RUN=yes\nPOLL_SECONDS=0.02\n")
        # Existing acknowledgement from another process must be replaced.
        (self.run_dir / "watcher.ready").write_text("instance_id=stale\n")
        watcher = HELPER.with_name("local-watcher.sh")
        env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", VAST_API_KEY="test-only")
        process = subprocess.Popen(
            ["bash", str(watcher), str(self.run_dir)], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            end = time.monotonic() + 5
            while time.monotonic() < end:
                result = handoff.watcher_condition(run_dir=self.run_dir, config=self.config)
                if result["status"] == "passed":
                    break
                time.sleep(0.02)
            self.assertEqual(result["status"], "passed")
            record = handoff.read_record(path=self.run_dir / "watcher.ready")
            self.assertEqual(record["pid"], str(process.pid))
            log = (self.run_dir / "watcher.log").read_text()
            self.assertIn("cleanup gate armed", log)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        # A signal after handoff must still leave unconfirmed training untouched.
        self.assertIn("cleanup refused", (self.run_dir / "watcher.log").read_text())

    def test_cli_exit_codes_and_wait_deadline(self) -> None:
        cases = (("ready", 0), ("not_ready", 1), ("terminal", 3))
        for status, expected_code in cases:
            with self.subTest(status=status):
                report = {"status": status, "checks": {"remote": {"status": "pending"}}}
                with patch("sys.argv", [str(HELPER), str(self.run_dir), "--wait-seconds", "0.01"]), \
                        patch.object(handoff, "load_setup", return_value=self.config), \
                        patch.object(handoff, "check_once", return_value=report), \
                        patch("builtins.print") as output:
                    self.assertEqual(handoff.main(), expected_code)
                    self.assertEqual(json.loads(output.call_args.args[0])["status"], status)
        with patch("sys.argv", [str(HELPER), str(self.run_dir)]), \
                patch.object(handoff, "load_setup", side_effect=ValueError), \
                patch("builtins.print"):
            self.assertEqual(handoff.main(), 2)


if __name__ == "__main__":
    unittest.main()
