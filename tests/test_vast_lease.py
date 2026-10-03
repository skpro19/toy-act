"""Forwarding lease recovery cannot steal or release another iteration's slot."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HELPERS = Path(__file__).resolve().parents[1] / ".pi/prompts/scripts/vast-train"


class LeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.actions = self.root / "tmux-actions.txt"
        self.stub(name="tmux", text=(
            'if [ "$1" = has-session ]; then test "${FAKE_TMUX_BUSY:-no}" = yes; '
            f'else printf "%s\\n" "$*" >> "{self.actions}"; fi\n'))
        self.stub(name="ss", text='if [ "${FAKE_PORT_BUSY:-no}" = yes ]; then echo LISTEN; fi\n')
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}")
        self.script = self.root / "lease.sh"
        # Isolate the committed helper's lock/owner files, not real /tmp leases.
        self.script.write_text((HELPERS / "local-wrapper-lease.sh").read_text().replace(
            "/tmp/toy-act-local-wrapper", str(self.root / "toy-act-local-wrapper")))
        self.owner = self.root / "toy-act-local-wrapper-0.owner"

    def stub(self, *, name: str, text: str) -> None:
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + text)
        path.chmod(0o755)

    def invoke(self, *, action: str, owner: str = "toy-act-123", index: str = "0") -> subprocess.CompletedProcess:
        args = ["bash", str(self.script), action, owner]
        if action != "allocate":
            args.append(index)
        return subprocess.run(args, env=self.env, text=True, capture_output=True, timeout=5)

    def test_restore_reclaims_same_free_slot_after_reboot(self) -> None:
        result = self.invoke(action="restore")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["0", "act-ssh-0", "act-tb-0", "6006"])
        self.assertEqual(self.owner.read_text(), "toy-act-123\n")
        self.assertEqual(self.owner.stat().st_mode & 0o777, 0o600)

    def test_restore_reuses_matching_ownership(self) -> None:
        self.owner.write_text("toy-act-123\n")
        self.env["FAKE_TMUX_BUSY"] = "yes"
        self.assertEqual(self.invoke(action="restore").returncode, 0)

    def test_restore_never_steals_another_owner(self) -> None:
        self.owner.write_text("toy-act-999\n")
        self.assertEqual(self.invoke(action="restore").returncode, 1)
        self.assertEqual(self.owner.read_text(), "toy-act-999\n")
        self.assertFalse(self.actions.exists())

    def test_restore_refuses_unowned_busy_sessions_and_ports(self) -> None:
        for variable in ("FAKE_TMUX_BUSY", "FAKE_PORT_BUSY"):
            self.env[variable] = "yes"
            self.assertEqual(self.invoke(action="restore").returncode, 1)
            self.assertFalse(self.owner.exists())
            del self.env[variable]

    def test_allocate_and_release_remain_owner_scoped(self) -> None:
        self.owner.write_text("toy-act-999\n")
        result = self.invoke(action="allocate")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[0], "1")
        self.assertEqual(self.invoke(action="release-unstarted", index="1").returncode, 0)
        self.assertEqual(self.owner.read_text(), "toy-act-999\n")
        self.assertFalse((self.root / "toy-act-local-wrapper-1.owner").exists())
        self.assertIn("kill-session -t act-tb-1", self.actions.read_text())

    def test_watcher_cleanup_cannot_release_a_reassigned_slot(self) -> None:
        self.stub(name="ssh", text=(
            'printf "%s\\n" TRAIN_SESSION=no COMPLETED=no FAILED=yes RUN_STATUS=running '
            'RUN_NAME=test LAST_LOG=failure\n'))
        self.stub(name="vastai", text='if [ "$1" = show ]; then echo "[]"; else echo destroyed; fi\n')
        self.stub(name="jq", text=(
            'input=$(< /dev/stdin)\nif [[ "$*" == *\'type=="array"\'* ]]; then exit 0; else exit 1; fi\n'))
        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "setup.env").write_text(
            'INSTANCE_ID=123\nINSTANCE_LABEL=test\nSSH_HOST=example.invalid\nSSH_PORT=22\n'
            'SSH_KNOWN_HOSTS=\nCHECKPOINT_EVERY=5\nSTEPS=10\nUV=/usr/bin/true\n'
            f'LOCAL_OWNER_FILE={self.owner}\nSSH_SESSION=act-ssh-0\nTB_SESSION=act-tb-0\n'
            'WATCHER_ASSUME_STARTED=yes\n')
        watcher = run_dir / "watcher.sh"
        watcher.write_text((HELPERS / "local-watcher.sh").read_text().replace(
            "/tmp/toy-act-local-wrapper.lock", str(self.root / "toy-act-local-wrapper.lock")))
        self.owner.write_text("toy-act-999\n")
        result = subprocess.run(["bash", str(watcher), str(run_dir)],
                                env=dict(self.env, VAST_API_KEY="test-only"),
                                text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("destroyed_and_verified", (run_dir / "report.txt").read_text())
        self.assertEqual(self.owner.read_text(), "toy-act-999\n")
        self.assertFalse(self.actions.exists())
        self.assertIn("owned by another run", (run_dir / "watcher.log").read_text())


if __name__ == "__main__":
    unittest.main()
