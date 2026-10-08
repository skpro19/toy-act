"""Offline SSH endpoint selection and pinning regressions."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

HELPERS = Path(__file__).resolve().parents[1] / ".pi/prompts/scripts/vast-train"
sys.path.insert(0, str(HELPERS))
import ssh_endpoint
from workflow_common import Blocked


class EndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.known_hosts = Path(self.temp.name) / "known_hosts"
        self.record = {"public_ipaddr": "8.8.8.8", "ports": {"22/tcp": [{"HostPort": "41685"}]},
                       "ssh_host": "ssh6.vast.ai", "ssh_port": 31270, "image_runtype": "ssh_direc ssh_proxy"}
        self.attempt = {}
        self.journal = Mock()
        self.save = Mock()

    def select(self) -> dict:
        with patch.object(ssh_endpoint.time, "sleep"):
            return ssh_endpoint.select_endpoint(journal=self.journal, record=self.record, attempt=self.attempt,
                                                known_hosts=self.known_hosts, save=self.save)

    def scan(self, *, args: list, **kwargs) -> subprocess.CompletedProcess:
        host, port = args[-1], int(args[-2])
        key_host = host if port == 22 else f"[{host}]:{port}"
        return subprocess.CompletedProcess(args, 0, f"{key_host} ssh-ed25519 test-key\n", "")

    def test_direct_success_does_not_contact_dead_proxy_or_cli_cache(self) -> None:
        self.journal.run.side_effect = self.scan
        selected = self.select()
        self.assertEqual(selected, {"host": "8.8.8.8", "port": 41685})
        self.assertEqual(self.attempt["ssh_endpoint"], selected)
        self.assertEqual(self.journal.run.call_count, 1)
        self.assertEqual(self.known_hosts.stat().st_mode & 0o777, 0o600)
        self.save.assert_called_once()

    def test_dual_stack_bindings_with_same_port_select_and_pin_direct(self) -> None:
        self.record["ports"] = {"22/tcp": [
            {"HostIp": "0.0.0.0", "HostPort": "3028"},
            {"HostIp": "::", "HostPort": "3028"}]}
        self.journal.run.side_effect = self.scan
        selected = self.select()
        self.assertEqual(selected, {"host": "8.8.8.8", "port": 3028})
        self.assertEqual(self.attempt["ssh_endpoint"], selected)
        self.assertEqual(self.journal.run.call_count, 1)
        self.save.assert_called_once()
        self.assertIn("[8.8.8.8]:3028", self.known_hosts.read_text())

    def test_duplicate_ports_are_compared_after_normalization(self) -> None:
        self.record["ports"] = {"22/tcp": [{"HostPort": "3028"}, {"HostPort": 3028}]}
        self.assertEqual(ssh_endpoint.candidates(self.record)[0], {"host": "8.8.8.8", "port": 3028})

    def test_dual_stack_direct_failure_still_uses_proxy(self) -> None:
        self.record["ports"] = {"22/tcp": [
            {"HostIp": "0.0.0.0", "HostPort": "41685"},
            {"HostIp": "::", "HostPort": "41685"}]}
        def scan(**kwargs):
            if kwargs["args"][-1] == "8.8.8.8":
                return subprocess.CompletedProcess(kwargs["args"], 1, "", "")
            return self.scan(**kwargs)
        self.journal.run.side_effect = scan
        self.assertEqual(self.select(), {"host": "ssh6.vast.ai", "port": 31270})
        self.assertEqual(self.journal.run.call_count, 2)

    def test_direct_failure_falls_back_to_proxy(self) -> None:
        def scan(**kwargs):
            if kwargs["args"][-1] == "8.8.8.8":
                return subprocess.CompletedProcess(kwargs["args"], 1, "", "")
            return self.scan(**kwargs)
        self.journal.run.side_effect = scan
        self.assertEqual(self.select(), {"host": "ssh6.vast.ai", "port": 31270})
        self.assertEqual(self.journal.run.call_count, 2)

    def test_missing_mapping_uses_proxy_not_direct_port_start(self) -> None:
        self.record.pop("ports")
        self.record["direct_port_start"] = 41685
        self.journal.run.side_effect = self.scan
        self.assertEqual(self.select()["host"], "ssh6.vast.ai")

    def test_invalid_provider_fields_fail_before_scanning(self) -> None:
        variants = [{"ports": []}, {"ports": {"22/tcp": []}},
                    {"ports": {"22/tcp": [{"HostPort": "0"}]}},
                    {"ports": {"22/tcp": [{"HostPort": "41685"}, {"HostPort": "41686"}]}},
                    {"ports": {"22/tcp": [{"HostPort": "41685"}, {}]}},
                    {"ports": {"22/tcp": [{"HostPort": "41685"}, None]}},
                    {"ports": {"22/tcp": [{"HostPort": "41685"}, {"HostPort": True}]}},
                    {"ports": {"22/tcp": [{"HostPort": "41685"}, {"HostPort": "0"}]}},
                    {"public_ipaddr": "127.0.0.1"}, {"public_ipaddr": "bad"},
                    {"ssh_host": "-evil"}, {"ssh_port": True}, {"ssh_port": 65536}]
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(Blocked):
                    ssh_endpoint.candidates({**self.record, **variant})
        self.journal.run.assert_not_called()

    def test_both_endpoints_failed_is_bounded_and_does_not_pin(self) -> None:
        self.journal.run.return_value = subprocess.CompletedProcess([], 1, "", "")
        with self.assertRaises(Blocked):
            self.select()
        self.assertEqual(self.journal.run.call_count, 24)
        self.assertEqual(self.attempt, {})
        self.save.assert_not_called()
        self.assertFalse(self.known_hosts.exists())

    def test_saved_endpoint_does_not_fall_back_on_failure(self) -> None:
        self.attempt["ssh_endpoint"] = {"host": "ssh6.vast.ai", "port": 31270}
        self.journal.run.return_value = subprocess.CompletedProcess([], 1, "", "")
        with self.assertRaises(Blocked):
            self.select()
        self.assertEqual(self.journal.run.call_count, 12)
        self.assertTrue(all(call.kwargs["args"][-1] == "ssh6.vast.ai" for call in self.journal.run.call_args_list))

    def test_changed_endpoint_or_host_keys_are_not_replaced(self) -> None:
        self.attempt["ssh_endpoint"] = {"host": "old.vast.ai", "port": 1234}
        with self.assertRaises(Blocked):
            self.select()
        self.journal.run.assert_not_called()
        self.attempt.clear()
        original = "[8.8.8.8]:41685 ssh-ed25519 original-key\n"
        self.known_hosts.write_text(original)
        self.journal.run.side_effect = self.scan
        with self.assertRaises(Blocked):
            self.select()
        self.assertEqual(self.known_hosts.read_text(), original)
        self.assertEqual(self.attempt, {})

    def test_malformed_scan_output_fails_closed(self) -> None:
        self.journal.run.return_value = subprocess.CompletedProcess([], 0, "other-host ssh-ed25519 key\n", "")
        with self.assertRaises(Blocked):
            self.select()
        self.assertFalse(self.known_hosts.exists())

    def test_jupyter_proxy_port_adjustment_matches_upstream(self) -> None:
        record = deepcopy(self.record)
        record.pop("ports")
        record["image_runtype"] = "jupyter"
        self.assertEqual(ssh_endpoint.candidates(record)[0]["port"], 31271)


if __name__ == "__main__":
    unittest.main()
