"""Offline cancellation checks: only the saved watcher may remove a rental."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HELPERS = Path(__file__).resolve().parents[1] / '.pi/prompts/scripts/vast-train'
sys.path.insert(0, str(HELPERS))
import cancel_run
import workflow_common as common


class CancellationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.combo = {'id': '0000', 'instance_id': 123, 'training_launch_requested': True,
                      'run_dir': '/saved/run', 'remote_run_name': 'saved-run', 'status': 'running',
                      'run': {'label': 'saved-label', 'instance_id': 123}}
        self.journal = Mock()
        self.iteration = SimpleNamespace(state={'combos': [self.combo]}, journal=self.journal,
                                        manifest={'git_commit': 'a' * 40})

    def test_wrong_iteration_or_unlaunched_instance_never_cancels(self) -> None:
        with self.assertRaises(common.Blocked):
            cancel_run.cancel(iteration=self.iteration, instance_id=456, wait_seconds=0)
        self.combo['training_launch_requested'] = False
        with self.assertRaises(common.Blocked):
            cancel_run.cancel(iteration=self.iteration, instance_id=123, wait_seconds=0)
        self.journal.run.assert_not_called()

    def test_identity_mismatch_or_missing_watcher_blocks(self) -> None:
        with patch.object(cancel_run, 'reconcile', return_value=False), \
                patch.object(cancel_run, 'instances', return_value=[{'id': 123, 'label': 'other'}]):
            with self.assertRaises(common.Blocked):
                cancel_run.cancel(iteration=self.iteration, instance_id=123, wait_seconds=0)
        with patch.object(cancel_run, 'reconcile', return_value=False), \
                patch.object(cancel_run, 'instances', return_value=[{'id': 123, 'label': 'saved-label'}]), \
                patch.object(cancel_run, 'ensure_watcher'), patch.object(cancel_run, 'watcher_alive', return_value=False):
            with self.assertRaises(common.Blocked):
                cancel_run.cancel(iteration=self.iteration, instance_id=123, wait_seconds=0)
        self.journal.run.assert_not_called()

    def test_cancel_signals_runner_and_waits_for_watcher_reconciliation(self) -> None:
        with patch.object(cancel_run, 'reconcile', side_effect=[False, True]) as reconcile, \
                patch.object(cancel_run, 'instances', return_value=[{'id': 123, 'label': 'saved-label'}]), \
                patch.object(cancel_run, 'ensure_watcher'), patch.object(cancel_run, 'watcher_alive', return_value=True), \
                patch.object(cancel_run, 'ssh_args', return_value=['ssh', 'example', 'python -']), \
                patch.object(cancel_run, 'digest', return_value='b' * 64):
            cancel_run.cancel(iteration=self.iteration, instance_id=123, wait_seconds=0)
        self.assertEqual(reconcile.call_count, 2)
        self.journal.run.assert_called_once()
        script = self.journal.run.call_args.kwargs['input_text']
        self.assertIn('pidfd_send_signal', script)
        self.assertIn('expected_run', script)
        self.assertIn('expected_runner_hash', script)
        self.assertNotIn('destroy', script)
        self.assertNotIn('write_text', script)
        self.assertNotIn('write_bytes', script)

    def test_terminal_report_is_reconciled_without_remote_signal(self) -> None:
        with patch.object(cancel_run, 'reconcile', return_value=True):
            cancel_run.cancel(iteration=self.iteration, instance_id=123, wait_seconds=0)
        self.journal.run.assert_not_called()

    def test_remote_script_is_valid_python(self) -> None:
        compile(cancel_run.REMOTE_CANCEL, '<remote-cancellation>', 'exec')


if __name__ == '__main__':
    unittest.main()
