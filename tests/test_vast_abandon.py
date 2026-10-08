"""Mocked provisional abandonment; no live rentals or API mutations."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HELPERS = Path(__file__).resolve().parents[1] / ".pi/prompts/scripts/vast-train"
sys.path.insert(0, str(HELPERS))
import abandon_provisional
import provision
from workflow_common import Blocked


class AbandonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.combo = {"id": "0000", "status": "provisioning", "instance_id": 123,
                      "attempts": [{"instance_id": 123, "label": "exact-label", "status": "created"}]}
        self.other = {"id": "0001", "status": "pending", "attempts": []}
        self.snapshots = []
        self.iteration = SimpleNamespace(state={"combos": [self.combo, self.other]},
                                         journal=Mock(), save=lambda: self.snapshots.append(deepcopy(self.combo)))
        self.record = {"id": 123, "label": "exact-label"}
        self.iteration.journal.directory = Path("/unused/mock-directory")
        self.iteration.journal.run.return_value = subprocess.CompletedProcess([], 0, "", "")

    def abandon(self) -> None:
        abandon_provisional.abandon(iteration=self.iteration, instance_id=123)

    def test_intent_precedes_exact_removal_and_verified_terminal_state(self) -> None:
        def command(**kwargs):
            self.assertEqual(kwargs["args"], ["vastai", "destroy", "instance", "123", "-y"])
            self.assertTrue(self.snapshots[-1]["provisional_abandon_requested"])
            return subprocess.CompletedProcess([], 0, "", "")
        self.iteration.journal.run.side_effect = command
        with patch.object(abandon_provisional, "instances", return_value=[self.record]), \
                patch.object(provision, "instances", side_effect=[[self.record], []]):
            self.abandon()
        self.assertEqual(self.combo["status"], "abandoned")
        self.assertEqual(self.combo["attempts"][0]["status"], "removed")
        self.assertTrue(self.combo["provisional_abandon_verified_at"])
        self.assertEqual(self.other, {"id": "0001", "status": "pending", "attempts": []})

    def test_launch_intent_and_accepted_attempts_block_without_api_or_destroy(self) -> None:
        variants = [{"training_launch_requested": True}, {"services_started": True},
                    {"status": "setup"}, {"run": {"instance_id": 123}},
                    {"attempts": [{"instance_id": 123, "label": "exact-label", "status": "accepted"}]}]
        original = deepcopy(self.combo)
        for variant in variants:
            self.combo.clear()
            self.combo.update(deepcopy(original), **variant)
            with patch.object(abandon_provisional, "instances") as records:
                with self.assertRaises(Blocked):
                    self.abandon()
                records.assert_not_called()
        self.iteration.journal.run.assert_not_called()

    def test_wrong_id_label_duplicate_label_and_api_error_block_removal(self) -> None:
        variants = [[{"id": 123, "label": "other"}],
                    [self.record, {"id": 456, "label": "exact-label"}]]
        for records in variants:
            with patch.object(abandon_provisional, "instances", return_value=records):
                with self.assertRaises(Blocked):
                    self.abandon()
        with patch.object(abandon_provisional, "instances", side_effect=Blocked("API unavailable")):
            with self.assertRaises(Blocked):
                self.abandon()
        self.iteration.journal.run.assert_not_called()
        self.assertEqual(self.snapshots, [])

    def test_failed_verification_preserves_intent_and_blocks_driver_provisioning(self) -> None:
        with patch.object(abandon_provisional, "instances", return_value=[self.record]), \
                patch.object(provision, "instances", side_effect=[[self.record], Blocked("API down")]):
            with self.assertRaises(Blocked):
                self.abandon()
        self.assertEqual(self.combo["status"], "provisioning")
        self.assertEqual(self.combo["attempts"][0]["status"], "created")
        with self.assertRaises(Blocked):
            provision.provision(journal=self.iteration.journal, combo=self.combo, directory=Path("/unused"),
                                commit="a" * 40, save=self.iteration.save)

    def test_absence_recovery_and_repeated_abandon_never_destroy_again(self) -> None:
        self.combo["provisional_abandon_requested"] = "saved-time"
        with patch.object(abandon_provisional, "instances", return_value=[]), \
                patch.object(provision, "instances", return_value=[]):
            self.abandon()
            self.abandon()
        self.iteration.journal.run.assert_not_called()
        self.assertEqual(self.combo["status"], "abandoned")

    def test_terminal_removed_instance_reappearing_blocks(self) -> None:
        self.combo.update(status="abandoned", provisional_abandon_requested="saved-time")
        self.combo["attempts"][0]["status"] = "removed"
        with patch.object(abandon_provisional, "instances", return_value=[self.record]):
            with self.assertRaises(Blocked):
                self.abandon()
        self.iteration.journal.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
