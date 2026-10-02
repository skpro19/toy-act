"""Exercise the provisioning Git gate against disposable local repositories."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


HELPER = Path(__file__).resolve().parents[1] / ".pi/prompts/scripts/vast-train/git-preflight.sh"


class GitPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.remote = root / "remote.git"
        self.repo = root / "local"
        self.env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        self.git(cwd=root, args=["init", "--bare", str(self.remote)])
        self.git(cwd=root, args=["init", "-b", "act-v2", str(self.repo)])
        self.git(cwd=self.repo, args=["config", "user.name", "Test"])
        self.git(cwd=self.repo, args=["config", "user.email", "test@example.invalid"])
        (self.repo / ".gitignore").write_text("configs/\n.vast-train-local/\n")
        (self.repo / "code.txt").write_text("initial\n")
        self.commit()
        self.git(cwd=self.repo, args=["remote", "add", "origin", str(self.remote)])
        self.push()
        self.initial = self.git(cwd=self.repo, args=["rev-parse", "HEAD"]).strip()

    def git(self, *, cwd: Path, args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args], cwd=cwd, env=self.env, text=True,
            capture_output=True, check=True)
        return result.stdout

    def commit(self) -> None:
        self.git(cwd=self.repo, args=["add", "."])
        self.git(cwd=self.repo, args=["-c", "commit.gpgsign=false", "commit", "-m", "test change"])

    def push(self) -> None:
        self.git(cwd=self.repo, args=["push", "origin", "act-v2"])

    def gate(self, *, expected: str | None = None, remote: Path | None = None) -> subprocess.CompletedProcess[str]:
        args = ["bash", str(HELPER), "--remote-url", str(remote or self.remote)]
        if expected is not None:
            args.extend(["--expected-commit", expected])
        return subprocess.run(
            args, cwd=self.repo, env=self.env, text=True, capture_output=True, timeout=15)

    def assert_rejected(self, *, message: str, expected: str | None = None, remote: Path | None = None) -> None:
        result = self.gate(expected=expected, remote=remote)
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn(message, result.stderr)

    def test_clean_synced_and_ignored_configs_pass(self) -> None:
        for directory in ["configs", ".vast-train-local"]:
            path = self.repo / directory
            path.mkdir()
            (path / "local.toml").write_text("local only\n")
        result = self.gate(expected=self.initial)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), self.initial)

    def test_unstaged_changes_fail(self) -> None:
        (self.repo / "code.txt").write_text("modified\n")
        self.assert_rejected(message="Working tree is dirty")

    def test_staged_changes_fail(self) -> None:
        (self.repo / "code.txt").write_text("modified\n")
        self.git(cwd=self.repo, args=["add", "code.txt"])
        self.assert_rejected(message="Working tree is dirty")

    def test_nonignored_untracked_files_fail(self) -> None:
        (self.repo / "new.py").write_text("# untracked\n")
        self.assert_rejected(message="Working tree is dirty")

    def test_wrong_branch_fails(self) -> None:
        self.git(cwd=self.repo, args=["checkout", "-b", "other"])
        self.assert_rejected(message="training requires act-v2")

    def test_detached_head_fails(self) -> None:
        self.git(cwd=self.repo, args=["checkout", "--detach"])
        self.assert_rejected(message="Detached HEAD")

    def test_ahead_fails_without_resetting(self) -> None:
        (self.repo / "code.txt").write_text("ahead\n")
        self.commit()
        head = self.git(cwd=self.repo, args=["rev-parse", "HEAD"])
        self.assert_rejected(message="ahead=1, behind=0")
        self.assertEqual(self.git(cwd=self.repo, args=["rev-parse", "HEAD"]), head)

    def test_behind_fails_without_pulling(self) -> None:
        (self.repo / "code.txt").write_text("remote update\n")
        self.commit()
        self.push()
        self.git(cwd=self.repo, args=["reset", "--hard", self.initial])
        self.assert_rejected(message="ahead=0, behind=1")
        self.assertEqual(self.git(cwd=self.repo, args=["rev-parse", "HEAD"]).strip(), self.initial)

    def test_diverged_fails(self) -> None:
        (self.repo / "code.txt").write_text("remote update\n")
        self.commit()
        self.push()
        self.git(cwd=self.repo, args=["reset", "--hard", self.initial])
        (self.repo / "code.txt").write_text("different local update\n")
        self.commit()
        self.assert_rejected(message="ahead=1, behind=1")

    def test_remote_failure_has_no_stale_ref_fallback(self) -> None:
        self.assert_rejected(message="Cannot read", remote=self.remote.parent / "missing.git")

    def test_missing_remote_branch_fails(self) -> None:
        self.git(cwd=self.repo, args=["push", "origin", "--delete", "act-v2"])
        self.assert_rejected(message="Cannot read")

    def test_changed_sweep_commit_fails_even_when_synced(self) -> None:
        (self.repo / "code.txt").write_text("next revision\n")
        self.commit()
        self.push()
        self.assert_rejected(message="Sweep pinned", expected=self.initial)

    def test_branch_tip_move_does_not_prevent_pinned_checkout(self) -> None:
        (self.repo / "code.txt").write_text("next revision\n")
        self.commit()
        self.push()
        instance = self.remote.parent / "instance"
        self.git(cwd=self.remote.parent, args=[
            "clone", "--branch", "act-v2", "--single-branch", str(self.remote), str(instance)])
        self.git(cwd=instance, args=["fetch", "--no-tags", "origin", self.initial])
        self.git(cwd=instance, args=["checkout", "--detach", self.initial])
        self.assertEqual(self.git(cwd=instance, args=["rev-parse", "HEAD"]).strip(), self.initial)


if __name__ == "__main__":
    unittest.main()
