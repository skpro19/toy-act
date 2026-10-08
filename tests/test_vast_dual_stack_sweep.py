"""Run the sweep CLI offline with real resolution, provisioning and SSH selection.

Cloud, SSH, dashboard and remote training boundaries are mocked. This fixture
never resumes saved rentals or bypasses Git checks in a live driver process.
"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_vast_workflow import COMMIT, fixture_offer, fixture_probe
import iteration
import provision
import workflow
import workflow_common as common

REPO = Path(__file__).resolve().parents[1]
SPEC_NAME = "ph100-k100-beta1-z1+k.toml"


class DualStackSweepTests(unittest.TestCase):
    def test_fresh_sweep_cli_hands_off_all_combos_with_dual_stack_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Sweep specs may be ignored local files. Keep this regression
            # reproducible in a clean checkout with the same requested grid.
            spec = root / SPEC_NAME
            group = REPO / "configs/train/act_v2/BS-32"
            spec.write_text(
                f'group = "{group}"\n[fixed]\n'
                'dataset = "datasets/can/ph/2026-09-29_02-01-42_agentview_robot0_eye_in_hand_ph100.hdf5"\n'
                'rollout = { episodes = 30, horizon = 200, seed = 42, n_action_steps = [10] }\n'
                'lr = 1e-5\nbeta = 1\nuse_z = true\n'
                '[grid]\naction_chunk_size = [10, 50, 100]\n')
            directory = iteration.new_directory(source=spec, root=root / "iterations")
            records = []
            gates = []
            command_log = []
            real_run = common.Journal.run

            def command(journal, **kwargs):
                args = kwargs["args"]
                command_log.append(args)
                # Only local config resolution executes a real subprocess.
                if args[:6] == ["uv", "run", "--frozen", "python", "-m", "scripts.ablate"]:
                    return real_run(journal, **kwargs)
                if len(args) > 1 and Path(args[1]).name == "create_request.py":
                    records.append({
                        "id": 123 + len(records), "label": args[args.index("--label") + 1],
                        "actual_status": "running", "status_msg": "", "dph_total": 0.55,
                        "public_ipaddr": "8.8.8.8", "ssh_host": "example.invalid", "ssh_port": 2222,
                        "ports": {"22/tcp": [
                            {"HostIp": "0.0.0.0", "HostPort": "3028"},
                            {"HostIp": "::", "HostPort": "3028"}]}})
                    # Exercise exact-label reconciliation without a live create.
                    output = "mocked create response reconciled by exact label"
                elif args[0] == "ssh-keyscan":
                    self.assertEqual(args[-3:], ["-p", "3028", "8.8.8.8"])
                    output = "[8.8.8.8]:3028 ssh-ed25519 test-key\n"
                elif args[0] == "ssh":
                    output = json.dumps(fixture_probe())
                else:
                    self.fail(f"Unexpected external command: {args[0]}")
                return subprocess.CompletedProcess(args, 0, output, "")

            def configure(*, journal, combo, config_path, commit, save) -> None:
                self.assertEqual(combo["run"]["host"], "8.8.8.8")
                self.assertEqual(combo["run"]["port"], 3028)
                self.assertTrue(Path(combo["run"]["known_hosts"]).exists())
                combo["status"] = "ready_to_launch"
                save()

            def launch(*, journal, combo, save) -> None:
                combo.update(status="awaiting_handoff", training_launch_requested=True, services_started=True)
                save()

            def handoff(*, journal, combo, save, wait_seconds=None) -> int:
                gates.append((combo["id"], wait_seconds))
                # A new combo may only rent after both previous fresh checks.
                self.assertEqual(len(records), int(combo["id"]) + 1)
                combo["status"] = "running"
                save()
                return 0

            with patch.object(sys, "argv", ["workflow.py", "--spec", str(spec)]), \
                    patch.object(workflow, "new_directory", return_value=directory), \
                    patch.object(iteration, "git_preflight", return_value=COMMIT), \
                    patch.object(workflow, "git_preflight", return_value=COMMIT), \
                    patch.object(provision, "git_preflight", return_value=COMMIT), \
                    patch.object(workflow, "local_preflight"), \
                    patch.object(workflow, "validate_dataset_source"), \
                    patch.object(workflow, "ensure_dashboard"), \
                    patch.object(provision, "REPO", root), \
                    patch.object(provision, "search_offers", return_value=[fixture_offer()]), \
                    patch.object(provision, "instances", side_effect=lambda journal: deepcopy(records)), \
                    patch.object(provision, "network_gate", return_value=True), \
                    patch.object(common.Journal, "run", new=command), \
                    patch.object(workflow, "setup", side_effect=configure), \
                    patch.object(workflow, "launch", side_effect=launch), \
                    patch.object(workflow, "handoff", side_effect=handoff):
                self.assertEqual(workflow.main(), 0)

            saved = iteration.Iteration(directory=directory)
            self.assertEqual(saved.manifest["kind"], "sweep")
            self.assertEqual(saved.state["driver_status"], "handed_off")
            self.assertEqual([combo["status"] for combo in saved.state["combos"]], ["running"] * 3)
            self.assertEqual(gates, [("0000", None), ("0000", 0), ("0001", None),
                                     ("0001", 0), ("0002", None), ("0002", 0)])
            for combo in saved.state["combos"]:
                self.assertEqual(len(combo["attempts"]), 1)
                attempt = combo["attempts"][0]
                self.assertEqual(attempt["status"], "accepted")
                self.assertEqual(attempt["ssh_endpoint"], {"host": "8.8.8.8", "port": 3028})
            self.assertFalse(any(args[:2] == ["vastai", "destroy"] for args in command_log))


if __name__ == "__main__":
    unittest.main()
