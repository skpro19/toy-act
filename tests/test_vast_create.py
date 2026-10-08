"""Offline create transport, receipt replay and provider-confirmed recovery tests."""

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
import unittest
from unittest.mock import Mock, patch

HELPERS = Path(__file__).resolve().parents[1] / ".pi/prompts/scripts/vast-train"
sys.path.insert(0, str(HELPERS))
import create_request
import provision
import resolve_create
import workflow_common as common

LABEL = "toy-act-train-actv2-2026-10-08_17-14-07-abcdef123456"


class ResponseTests(unittest.TestCase):
    def test_created_and_explicit_rejection(self) -> None:
        created = create_request.classify_response(status=200, body=b'{"success":true,"new_contract":123}')
        self.assertEqual(created["outcome"], "created")
        self.assertEqual(created["instance_id"], 123)
        rejected = create_request.classify_response(status=200, body=b'{"success":false}')
        self.assertEqual(rejected["outcome"], "rejected")

    def test_ambiguous_responses_never_authorize_replacement(self) -> None:
        bodies = (b'Failed with error 400: unavailable', b'null', b'[]', b'{}',
                  b'{"success":true,"new_contract":true}', b'{"success":true,"new_contract":0}',
                  b'{"success":false,"new_contract":123}', b'{"success":false,"new_contract":false}',
                  b'{"success":false,"new_contract":"123"}', b'\xff',
                  b'{"success":true,"success":false}', b'{"success":false,"new_contract":0.0}')
        for body in bodies:
            with self.subTest(body=body):
                self.assertEqual(create_request.classify_response(status=200, body=body)["outcome"], "unknown")
        for status in (302, 400, 408, 429, 500, 503):
            self.assertEqual(create_request.classify_response(status=status, body=b'{"success":false}')["outcome"], "unknown")

    def test_only_allowlisted_diagnostics_are_retained(self) -> None:
        body = json.dumps({"success": False, "error": "offer_unavailable",
                           "msg": "secret-provider-token", "api_key": "secret-provider-token",
                           "instance": {"jupyter_token": "secret-provider-token"}}).encode()
        result = create_request.classify_response(status=400, body=body)
        self.assertNotIn("secret-provider-token", json.dumps(result))
        self.assertEqual(result["provider_error_code"], "offer_unavailable")
        self.assertEqual(result["http_status"], 400)
        self.assertEqual(len(result["response_sha256"]), 64)
        unknown_code = create_request.classify_response(status=400, body=b'{"error":"secret_provider_token"}')
        self.assertNotIn("secret_provider_token", json.dumps(unknown_code))

    def test_single_shot_transport_and_payload(self) -> None:
        response = Mock(code=200)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'{"success":true,"new_contract":123}'
        opener = Mock()
        opener.open.return_value = response
        with patch.dict(os.environ, {"VAST_API_KEY": "test-secret-key"}), \
                patch.object(create_request, "build_opener", return_value=opener):
            result = create_request.send_create(offer_id=42, label=LABEL)
        self.assertEqual(result["outcome"], "created")
        opener.open.assert_called_once()
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "PUT")
        self.assertNotIn("test-secret-key", request.full_url)
        payload = json.loads(request.data)
        self.assertEqual(payload["label"], LABEL)
        self.assertEqual(payload["disk"], 100)
        self.assertTrue(payload["cancel_unavail"])
        self.assertEqual(payload["runtype"], "ssh_direc ssh_proxy")
        self.assertEqual(payload["env"], {"-p 22:22": "1"})

    def test_transport_errors_do_not_retry_or_leak_exception(self) -> None:
        for error in (URLError("secret-provider-token"), TimeoutError("secret-provider-token"),
                      HTTPError("https://example.invalid/?api_key=secret-provider-token", 400,
                                "error", {}, io.BytesIO(b'{"success":false}'))):
            opener = Mock()
            opener.open.side_effect = error
            with patch.dict(os.environ, {"VAST_API_KEY": "test-secret-key"}), \
                    patch.object(create_request, "build_opener", return_value=opener):
                result = create_request.send_create(offer_id=42, label=LABEL)
            self.assertEqual(result["outcome"], "unknown")
            self.assertNotIn("secret-provider-token", json.dumps(result))
            opener.open.assert_called_once()

    def test_oversized_and_interrupted_reads_remain_unknown(self) -> None:
        for body, error in ((b'x' * (create_request.MAX_RESPONSE_BYTES + 1), None),
                            (None, TimeoutError("secret-provider-token"))):
            response = Mock(code=200)
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.read.return_value = body
            response.read.side_effect = error
            opener = Mock()
            opener.open.return_value = response
            with patch.dict(os.environ, {"VAST_API_KEY": "test-secret-key"}), \
                    patch.object(create_request, "build_opener", return_value=opener):
                result = create_request.send_create(offer_id=42, label=LABEL)
            self.assertEqual(result["outcome"], "unknown")
            self.assertNotIn("secret-provider-token", json.dumps(result))
            opener.open.assert_called_once()

    def test_redirects_are_not_followed(self) -> None:
        self.assertIsNone(create_request.NoRedirects().redirect_request(None, None, 302, "", {}, "https://elsewhere.invalid"))


class ReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.journal = common.Journal(directory=self.directory)
        self.attempt = {"label": LABEL, "offer": {"id": 42}, "status": "create_requested"}
        self.path = self.directory / f"create-{LABEL}.json"

    def test_result_is_durable_before_parent_receives_stdout(self) -> None:
        args = ["create_request.py", "--offer-id", "42", "--label", LABEL, "--result", str(self.path)]
        with patch.object(sys, "argv", args), \
                patch.object(create_request, "send_create", return_value={"outcome": "rejected", "reason": "explicit_rejection"}), \
                patch("builtins.print", side_effect=BrokenPipeError):
            with self.assertRaises(BrokenPipeError):
                create_request.main()
        self.assertEqual(json.loads(self.path.read_text())["outcome"], "rejected")
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        with patch.object(sys, "argv", args), patch.object(create_request, "send_create") as send:
            with self.assertRaises(SystemExit):
                create_request.main()
            send.assert_not_called()

    def test_receipt_replays_after_interruption_without_create(self) -> None:
        common.write_json(path=self.path, value={"version": 1, "label": LABEL, "offer_id": 42,
                                                "outcome": "rejected", "reason": "explicit_rejection"})
        save = Mock()
        provision.restore_create_result(journal=self.journal, attempt=self.attempt, directory=self.directory, save=save)
        self.assertTrue(self.attempt["create_rejected"])
        save.assert_called_once()

    def test_known_id_replays_but_conflicting_receipt_blocks(self) -> None:
        common.write_json(path=self.path, value={"version": 1, "label": LABEL, "offer_id": 42,
                                                "outcome": "created", "instance_id": 123})
        provision.restore_create_result(journal=self.journal, attempt=self.attempt, directory=self.directory, save=Mock())
        self.assertEqual(self.attempt["returned_instance_id"], 123)
        common.write_json(path=self.path, value={"version": 1, "label": LABEL, "offer_id": 42,
                                                "outcome": "rejected"})
        with self.assertRaises(common.Blocked):
            provision.restore_create_result(journal=self.journal, attempt=self.attempt, directory=self.directory, save=Mock())
        self.assertNotIn("create_rejected", self.attempt)

    def test_unknown_receipt_does_not_invent_rejection(self) -> None:
        common.write_json(path=self.path, value={"version": 1, "label": LABEL, "offer_id": 42,
                                                "outcome": "unknown", "reason": "http_error", "http_status": 400})
        provision.restore_create_result(journal=self.journal, attempt=self.attempt, directory=self.directory, save=Mock())
        self.assertNotIn("create_rejected", self.attempt)
        self.assertNotIn("returned_instance_id", self.attempt)

    def test_missing_historical_receipt_does_not_invent_rejection(self) -> None:
        provision.restore_create_result(journal=self.journal, attempt=self.attempt, directory=self.directory, save=Mock())
        self.assertNotIn("create_rejected", self.attempt)

    def test_mismatched_receipt_is_blocked(self) -> None:
        common.write_json(path=self.path, value={"version": 1, "label": "other", "offer_id": 42, "outcome": "rejected"})
        with self.assertRaises(common.Blocked):
            provision.restore_create_result(journal=self.journal, attempt=self.attempt, directory=self.directory, save=Mock())

    def test_delayed_visibility_only_repeats_read_queries(self) -> None:
        record = {"id": 123, "label": LABEL}
        with patch.object(provision, "instances", side_effect=[[], [], [record]]) as query, \
                patch.object(provision.time, "sleep"), patch.object(self.journal, "run") as mutation:
            result = provision.reconcile_create(journal=self.journal, attempt=self.attempt, save=Mock(), wait_seconds=60)
        self.assertEqual(result, record)
        self.assertEqual(query.call_count, 3)
        mutation.assert_not_called()

    def test_duplicate_labels_are_blocked_immediately(self) -> None:
        with patch.object(provision, "instances", return_value=[{"id": 1, "label": LABEL}, {"id": 2, "label": LABEL}]), \
                patch.object(provision.time, "sleep") as sleep:
            with self.assertRaises(common.Blocked):
                provision.reconcile_create(journal=self.journal, attempt=self.attempt, save=Mock(), wait_seconds=60)
        sleep.assert_not_called()


class ProviderRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.attempt = {"label": LABEL, "offer": {"id": 42}, "status": "create_requested", "created_at": "2026-10-08T17:14:07Z"}
        self.combo = {"id": "0001", "status": "provisioning", "attempts": [self.attempt]}
        self.iteration = SimpleNamespace(directory=self.directory, state={"combos": [self.combo]}, save=Mock(),
                                        journal=common.Journal(directory=self.directory))
        self.evidence = {"iteration_id": self.directory.name, "combo_id": "0001", "label": LABEL,
                         "offer_id": 42, "created_at": self.attempt["created_at"],
                         "conclusion": "provider_confirmed_no_rental", "provider_reference": "support-ticket-123",
                         "reviewed_by": "operator"}

    def resolve(self) -> None:
        resolve_create.resolve_rejection(iteration=self.iteration, combo_id="0001", evidence=self.evidence)

    def test_cli_requires_explicit_provider_attestation(self) -> None:
        args = ["resolve_create.py", "--resume", "saved", "--combo", "0001", "--evidence", "evidence.json"]
        with patch.object(sys, "argv", args), patch.object(resolve_create, "locate") as locate:
            with self.assertRaises(SystemExit) as error:
                resolve_create.main()
        self.assertEqual(error.exception.code, 2)
        locate.assert_not_called()

    def test_cli_cannot_update_state_while_another_driver_holds_lock(self) -> None:
        args = ["resolve_create.py", "--resume", "saved", "--combo", "0001", "--evidence", "evidence.json",
                "--confirm-provider-rejection"]
        with common.lock(self.directory / "driver.lock"), patch.object(sys, "argv", args), \
                patch.object(resolve_create, "locate", return_value=self.directory), \
                patch.object(resolve_create, "Iteration") as load:
            self.assertEqual(resolve_create.main(), 1)
        load.assert_not_called()

    def test_reviewed_rejection_preserves_history_and_launch_status(self) -> None:
        with patch.object(resolve_create, "instances", return_value=[]):
            self.resolve()
        self.assertEqual(self.attempt["status"], "removed")
        self.assertEqual(self.combo["status"], "provisioning")
        self.assertEqual(len(self.combo["attempts"]), 1)
        snapshot = self.directory / self.attempt["provider_resolution"]["evidence"]
        self.assertEqual(json.loads(snapshot.read_text()), self.evidence)
        self.assertEqual(common.digest(snapshot), self.attempt["provider_resolution"]["sha256"])
        self.iteration.save.assert_called_once()

    def test_absence_without_provider_attestation_is_insufficient(self) -> None:
        self.evidence["conclusion"] = "not_in_instance_list"
        with patch.object(resolve_create, "instances", return_value=[]) as query:
            with self.assertRaises(common.Blocked):
                self.resolve()
        query.assert_not_called()
        self.assertEqual(self.attempt["status"], "create_requested")

    def test_mismatched_identity_existing_instance_or_api_failure_blocks(self) -> None:
        for key in ("label", "offer_id", "iteration_id", "created_at"):
            evidence = deepcopy(self.evidence)
            evidence[key] = "wrong"
            with self.assertRaises(common.Blocked):
                resolve_create.resolve_rejection(iteration=self.iteration, combo_id="0001", evidence=evidence)
        for response in ([{"id": 123, "label": LABEL}], common.Blocked("API unavailable")):
            options = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
            with patch.object(resolve_create, "instances", **options):
                with self.assertRaises(common.Blocked):
                    self.resolve()
        self.assertEqual(self.attempt["status"], "create_requested")
        self.iteration.save.assert_not_called()

    def test_receipt_with_confirmed_creation_blocks_manual_resolution(self) -> None:
        path = self.directory / "combos/0001" / f"create-{LABEL}.json"
        common.write_json(path=path, value={"version": 1, "label": LABEL, "offer_id": 42,
                                          "outcome": "created", "instance_id": 123})
        with patch.object(resolve_create, "instances") as query:
            with self.assertRaises(common.Blocked):
                self.resolve()
        query.assert_not_called()

    def test_launch_intent_and_known_instance_never_allow_resolution(self) -> None:
        self.combo["training_launch_requested"] = True
        with self.assertRaises(common.Blocked):
            self.resolve()
        del self.combo["training_launch_requested"]
        self.attempt["returned_instance_id"] = 123
        with self.assertRaises(common.Blocked):
            self.resolve()


if __name__ == "__main__":
    unittest.main()
