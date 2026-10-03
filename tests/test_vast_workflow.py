"""Offline safety/recovery fixtures; these tests never rent or query real instances."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

HELPERS = Path(__file__).resolve().parents[1] / ".pi/prompts/scripts/vast-train"
sys.path.insert(0, str(HELPERS))
import hardware_gate
import iteration
import provision
import setup_run
import workflow
import workflow_common as common

COMMIT = "a" * 40


def fixture_offer() -> dict:
    return dict(id=1, machine_id=10, cpu_name="AMD EPYC 9554 64-Core Processor",
                dph_total=0.5, disk_bw=2000, reliability=0.999, cpu_cores_effective=48,
                cpu_ram=128000, pcie_bw=24, gpu_name="RTX 4090", num_gpus=1,
                rentable=True, verification="verified", gpu_display_active=False,
                gpu_ram=24564, gpu_max_power=450, compute_cap=890, pci_gen=4,
                inet_down=800, inet_up=500, disk_space=200)


def fixture_probe() -> dict:
    return dict(cpu_model="AMD EPYC 9554 64-Core Processor", physical_cores=24, logical_cpus=48,
                cpu_quota=48, memory_bytes=128_000_000_000, workspace_available_bytes=110_000_000_000,
                gpus=[dict(name="NVIDIA GeForce RTX 4090", memory_mib=24564, power_watts=450,
                           pcie_gen=4, pcie_width=16, thermal_slowdown="Not Active",
                           power_brake_slowdown="Not Active")])


class HardwareTests(unittest.TestCase):
    def test_complete_probe_passes(self) -> None:
        self.assertTrue(hardware_gate.evaluate(probe=fixture_probe(), offer=fixture_offer())["passed"])

    def test_every_missing_measurement_fails_closed(self) -> None:
        for key in fixture_probe():
            probe = fixture_probe()
            del probe[key]
            with self.subTest(key=key):
                self.assertFalse(hardware_gate.evaluate(probe=probe, offer=fixture_offer())["passed"])
        for key in fixture_probe()["gpus"][0]:
            probe = fixture_probe()
            del probe["gpus"][0][key]
            with self.subTest(gpu_key=key):
                self.assertFalse(hardware_gate.evaluate(probe=probe, offer=fixture_offer())["passed"])

    def test_gpu_count_and_cpu_generation(self) -> None:
        for count in (0, 2):
            probe = fixture_probe()
            probe["gpus"] *= count
            self.assertFalse(hardware_gate.evaluate(probe=probe, offer=fixture_offer())["passed"])
        probe = fixture_probe()
        probe["cpu_model"] = "AMD EPYC 7742"
        self.assertFalse(hardware_gate.evaluate(probe=probe, offer=fixture_offer())["passed"])

    def test_quota_ram_disk_and_unknown_gpu_values(self) -> None:
        for key, value in (("cpu_quota", 8), ("memory_bytes", 63_000_000_000),
                           ("workspace_available_bytes", 99_000_000_000)):
            probe = fixture_probe()
            probe[key] = value
            self.assertFalse(hardware_gate.evaluate(probe=probe, offer=fixture_offer())["passed"])
        for value in (0, "N/A", float("nan")):
            probe = fixture_probe()
            probe["gpus"][0]["power_watts"] = value
            self.assertFalse(hardware_gate.evaluate(probe=probe, offer=fixture_offer())["passed"])
        probe = fixture_probe()
        probe["cpu_quota"] = None  # Explicit measured unlimited, not missing.
        self.assertTrue(hardware_gate.evaluate(probe=probe, offer=fixture_offer())["passed"])

    def test_cpu_ranking(self) -> None:
        names = ("AMD EPYC 9655", "AMD EPYC 9554", "AMD Ryzen Threadripper PRO 7995WX",
                 "AMD EPYC 7R13", "AMD Ryzen 9 9950X")
        self.assertEqual([hardware_gate.cpu_rank(name) for name in names], list(range(5)))
        self.assertIsNone(hardware_gate.cpu_rank("AMD EPYC 7K62"))


class LocalFixture(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "example.toml"
        self.source.write_text('name = "example"\n')
        self.directory = iteration.new_directory(source=self.source, root=self.root / "iterations")
        self.journal = common.Journal(directory=self.directory)

    def make_iteration(self) -> iteration.Iteration:
        def resolve(**kwargs):
            args = kwargs["args"]
            target = Path(args[args.index("--out") + 1])
            target.write_text('name = "example"\ndataset = "datasets/test.hdf5"\nsteps = 10\ncheckpoint_every = 5\n')
            return subprocess.CompletedProcess(args, 0, str(target), "")
        with patch.object(iteration, "git_preflight", return_value=COMMIT), \
                patch.object(self.journal, "run", side_effect=resolve):
            iteration.create(directory=self.directory, source=self.source, kind="single", journal=self.journal)
        return iteration.Iteration(directory=self.directory)


class IterationTests(LocalFixture):
    def test_identical_sources_create_independent_iterations(self) -> None:
        first = self.make_iteration()
        second = iteration.new_directory(source=self.source, root=self.root / "iterations")
        self.assertNotEqual(first.directory.name, second.name)
        same_basename = self.root / "other" / self.source.name
        same_basename.parent.mkdir()
        same_basename.write_text(self.source.read_text())
        third = iteration.new_directory(source=same_basename, root=self.root / "iterations")
        self.assertNotEqual(second, third)

    def test_source_changes_do_not_change_saved_iteration(self) -> None:
        saved = self.make_iteration()
        snapshot = saved.directory / saved.manifest["combos"][0]["config"]
        before = snapshot.read_bytes()
        self.source.write_text('name = "different"\n')
        saved.validate()
        self.assertEqual(before, snapshot.read_bytes())
        self.assertIn(saved.directory.name.rsplit("-", 1)[1], saved.manifest["combos"][0]["name"])

    def test_input_snapshots_preserve_original_line_endings(self) -> None:
        self.source.write_bytes(b'name = "example"\r\n')
        saved = self.make_iteration()
        snapshot = saved.directory / saved.manifest["inputs"][0]["path"]
        self.assertEqual(snapshot.read_bytes(), self.source.read_bytes())
        saved.validate()

    def test_config_manifest_and_input_mutations_fail(self) -> None:
        saved = self.make_iteration()
        for name in (saved.manifest["combos"][0]["config"], saved.manifest["inputs"][0]["path"], "manifest.json"):
            path = saved.directory / name
            original = path.read_text()
            path.write_text(original + "\n ")
            with self.subTest(path=name), self.assertRaises(common.Blocked):
                iteration.Iteration(directory=saved.directory)
            path.write_text(original)

    def test_stale_configs_are_not_enumerated(self) -> None:
        saved = self.make_iteration()
        (saved.directory / "configs/stale.toml").write_text('name = "stale"')
        saved.validate()
        self.assertEqual(len(saved.manifest["combos"]), 1)

    def test_unknown_and_duplicate_state_fails(self) -> None:
        saved = self.make_iteration()
        saved.state["combos"][0]["status"] = "mystery"
        with self.assertRaises(common.Blocked):
            saved.validate()
        saved.state["combos"].append(saved.state["combos"][0])
        with self.assertRaises(common.Blocked):
            saved.validate()

    def test_launch_intent_cannot_be_reset_to_pending(self) -> None:
        saved = self.make_iteration()
        saved.state["combos"][0]["training_launch_requested"] = True
        with self.assertRaises(common.Blocked):
            saved.validate()

    def test_concurrent_drivers_cannot_acquire_same_lock(self) -> None:
        path = self.directory / "driver.lock"
        with common.lock(path):
            with self.assertRaises(common.Blocked):
                with common.lock(path):
                    self.fail("Second lock acquired")

    def test_atomic_state_and_logs_are_private(self) -> None:
        saved = self.make_iteration()
        saved.state["driver_status"] = "interrupted"
        saved.save()
        restored = iteration.Iteration(directory=saved.directory)
        self.assertEqual(restored.state["driver_status"], "interrupted")
        self.assertEqual((saved.directory / "state.json").stat().st_mode & 0o777, 0o600)
        self.assertTrue((saved.directory / "summary.txt").exists())

    def test_git_failure_precedes_resolution(self) -> None:
        with patch.object(iteration, "git_preflight", side_effect=common.Blocked("dirty")), \
                patch.object(self.journal, "run") as run:
            with self.assertRaises(common.Blocked):
                iteration.create(directory=self.directory, source=self.source, kind="single", journal=self.journal)
            run.assert_not_called()

    def test_explicit_resume_locates_unique_identity(self) -> None:
        self.make_iteration()
        root = self.root / "iterations"
        self.assertEqual(iteration.locate(identity=self.directory.name, root=root), self.directory)
        with self.assertRaises(common.Blocked):
            iteration.locate(identity="../guess", root=root)

    def test_sweep_manifest_preserves_order_and_input_chain(self) -> None:
        group = self.root / "group"
        group.mkdir()
        (group / "parent.toml").write_text('name = "base"\n')
        (group / "BASE.toml").write_text('base_config = "parent.toml"\n')
        self.source.write_text(f'group = "{group}"\n[grid]\nseed = [2, 1]\n')
        def resolve(**kwargs):
            args = kwargs["args"]
            configs = Path(args[args.index("--resolve-dir") + 1])
            paths = [configs / "z.toml", configs / "a.toml"]
            for path in paths:
                path.write_text(f'name = "{path.stem}"\n')
            manifest = Path(args[args.index("--manifest") + 1])
            manifest.write_text(json.dumps([str(path) for path in paths]))
            return subprocess.CompletedProcess(args, 0, "", "")
        with patch.object(iteration, "git_preflight", return_value=COMMIT), \
                patch.object(self.journal, "run", side_effect=resolve):
            manifest = iteration.create(directory=self.directory, source=self.source, kind="sweep", journal=self.journal)
        self.assertEqual([combo["slug"] for combo in manifest["combos"]], ["z", "a"])
        self.assertEqual(len(manifest["inputs"]), 3)
        self.assertTrue(all((self.directory / record["path"]).is_file() for record in manifest["inputs"]))

    def test_real_sweep_resolution_fixture_without_cloud(self) -> None:
        group = self.root / "group"
        group.mkdir()
        (group / "BASE.toml").write_text(
            'version = "v4"\naction_loss = "l1"\nbatch_size = 8\nsteps = 10\n'
            'image_keys = ["agentview_image", "robot0_eye_in_hand_image"]\n'
            'dataset = "datasets/test.hdf5"\nlr = 0.0001\nseed = 0\nbeta = 0.01\n'
            'checkpoint_every = 5\n[tensorboard]\n')
        self.source.write_text(f'group = "{group}"\n[grid]\nseed = [2, 1]\n')
        with patch.object(iteration, "git_preflight", return_value=COMMIT), \
                patch.object(common, "instances", side_effect=AssertionError("Cloud access in plan")):
            manifest = iteration.create(directory=self.directory, source=self.source, kind="sweep", journal=self.journal)
        saved = iteration.Iteration(directory=self.directory)
        self.assertEqual(len(saved.state["combos"]), 2)
        import tomllib
        seeds = [tomllib.loads((self.directory / entry["config"]).read_text())["seed"]
                 for entry in manifest["combos"]]
        self.assertEqual(seeds, [2, 1])
        self.assertTrue(all(entry["sha256"] == common.digest(self.directory / entry["config"])
                            for entry in manifest["combos"]))

    def test_cli_plan_does_not_touch_cloud_or_historical_state(self) -> None:
        self.make_iteration()
        # Existing historical corruption must not enter a fresh iteration's path.
        historical = self.root / "ablate-example.state"
        historical.write_text("stalled ancient-run\n")
        fresh = iteration.new_directory(source=self.source, root=self.root / "iterations")
        args = argparse.Namespace(spec=None, config=self.source, resume=None, expected_commit=None, plan=True)
        def create_fixture(**kwargs):
            # Use the same fixture-building mechanism in the fresh directory.
            original = self.directory
            self.directory = fresh
            self.journal = common.Journal(directory=fresh)
            try:
                self.make_iteration()
            finally:
                self.directory = original
        with patch.object(workflow, "parse_args", return_value=args), \
                patch.object(workflow, "new_directory", return_value=fresh), \
                patch.object(workflow, "create", side_effect=create_fixture), \
                patch.object(workflow, "execute") as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.main(), 0)
            execute.assert_not_called()
        self.assertEqual(json.loads((fresh / "state.json").read_text())["driver_status"], "planned")
        self.assertEqual(historical.read_text(), "stalled ancient-run\n")

    def test_resume_does_not_require_original_source_file(self) -> None:
        self.make_iteration()
        self.source.unlink()
        args = argparse.Namespace(spec=None, config=self.source, resume=self.directory.name,
                                  expected_commit=None, plan=False)
        with patch.object(workflow, "parse_args", return_value=args), \
                patch.object(workflow, "locate", return_value=self.directory), \
                patch.object(workflow, "execute") as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.main(), 0)
            execute.assert_called_once()

    def test_interruption_records_state_while_lock_is_held(self) -> None:
        self.make_iteration()
        args = argparse.Namespace(spec=None, config=None, resume=self.directory.name, expected_commit=None, plan=False)
        def interrupt(saved):
            with self.assertRaises(common.Blocked):
                with common.lock(self.directory / "driver.lock"):
                    pass
            raise KeyboardInterrupt
        with patch.object(workflow, "parse_args", return_value=args), \
                patch.object(workflow, "locate", return_value=self.directory), \
                patch.object(workflow, "execute", side_effect=interrupt), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(workflow.main(), 130)
        self.assertEqual(json.loads((self.directory / "state.json").read_text())["driver_status"], "interrupted")


class ProvisionTests(LocalFixture):
    def test_offer_selection_rank_price_and_quarantine(self) -> None:
        first = fixture_offer()
        second = dict(first, id=2, machine_id=20, cpu_name="AMD EPYC 9655", dph_total=0.6)
        expensive = dict(first, id=3, machine_id=30, dph_total=0.81)
        old = dict(first, id=4, machine_id=40, cpu_name="AMD EPYC 7742")
        choices = provision.shortlist(offers=[first, second, expensive, old], quarantine={}, epoch=100000)
        self.assertEqual([offer["id"] for offer in choices], [2, 1])
        choices = provision.shortlist(offers=[first, second], quarantine={"20": 99999}, epoch=100000)
        self.assertEqual([offer["id"] for offer in choices], [1])

    def test_api_failure_or_malformed_output_is_not_absence(self) -> None:
        with patch.object(common, "load_vast_key"):
            for output in ("not-json", "{}", '[{"id":"123"}]'):
                with patch.object(self.journal, "run", return_value=subprocess.CompletedProcess([], 0, output, "")):
                    with self.assertRaises(common.Blocked):
                        common.instances(self.journal)
            with patch.object(self.journal, "run", side_effect=common.Blocked("API down")):
                with self.assertRaises(common.Blocked):
                    common.instances(self.journal)

    def test_create_intent_is_saved_before_external_request(self) -> None:
        combo = {"id": "0000", "status": "pending"}
        saved = []
        def save():
            saved.append(deepcopy(combo))
        def command(**kwargs):
            self.assertEqual(kwargs["args"][:3], ["vastai", "create", "instance"])
            self.assertEqual(saved[-1]["attempts"][0]["status"], "create_requested")
            self.assertTrue(saved[-1]["attempts"][0]["label"].startswith("toy-act-train-actv2-"))
            raise common.Blocked("create response lost")
        with patch.object(provision, "search_offers", return_value=[fixture_offer()]), \
                patch.object(provision, "git_preflight", return_value=COMMIT), \
                patch.object(provision, "instances", return_value=[]), \
                patch.object(self.journal, "run", side_effect=command):
            with self.assertRaises(common.Blocked):
                provision.provision(journal=self.journal, combo=combo, directory=self.directory / "combo", commit=COMMIT, save=save)
        self.assertEqual(len(combo["attempts"]), 1)

    def test_explicit_rejection_with_confirmed_absence_tries_only_three_offers(self) -> None:
        combo = {"id": "0000", "status": "pending"}
        commands = []
        def offers(**kwargs):
            index = len(combo.get("attempts", [])) + 1
            return [dict(fixture_offer(), id=index, machine_id=index * 10)]
        def command(**kwargs):
            commands.append(kwargs["args"])
            return subprocess.CompletedProcess(kwargs["args"], 0, '{"success":false}', "")
        with patch.object(provision, "search_offers", side_effect=offers), \
                patch.object(provision, "git_preflight", return_value=COMMIT), \
                patch.object(provision, "instances", return_value=[]), \
                patch.object(provision.time, "sleep"), patch.object(self.journal, "run", side_effect=command):
            with self.assertRaises(common.Blocked):
                provision.provision(journal=self.journal, combo=combo, directory=self.directory,
                                    commit=COMMIT, save=lambda: None)
        self.assertEqual(len(commands), 3)
        self.assertTrue(all(attempt["status"] == "removed" for attempt in combo["attempts"]))

    def test_preexisting_label_is_not_adopted_or_created_twice(self) -> None:
        combo = {"id": "0000", "status": "pending"}
        def records(journal):
            return [{"id": 123, "label": combo["attempts"][0]["label"]}]
        with patch.object(provision, "search_offers", return_value=[fixture_offer()]), \
                patch.object(provision, "git_preflight", return_value=COMMIT), \
                patch.object(provision, "instances", side_effect=records), patch.object(self.journal, "run") as run:
            with self.assertRaises(common.Blocked):
                provision.provision(journal=self.journal, combo=combo, directory=self.directory,
                                    commit=COMMIT, save=lambda: None)
            self.assertTrue(combo["attempts"][0]["preexisting_label"])
            with self.assertRaises(common.Blocked):
                provision.provision(journal=self.journal, combo=combo, directory=self.directory,
                                    commit=COMMIT, save=lambda: None)
            run.assert_not_called()

    def test_uncertain_create_resume_does_not_create_again(self) -> None:
        combo = {"id": "0000", "status": "provisioning", "attempts": [
            {"label": "saved-label", "offer": fixture_offer(), "status": "create_requested"}]}
        with patch.object(provision, "instances", return_value=[]), patch.object(self.journal, "run") as run:
            with self.assertRaises(common.Blocked):
                provision.provision(journal=self.journal, combo=combo, directory=self.directory, commit=COMMIT, save=lambda: None)
            run.assert_not_called()

    def test_label_reconciliation_recovers_id_without_parsing_create_output(self) -> None:
        attempt = {"label": "saved-label", "status": "create_requested"}
        record = {"id": 123, "label": "saved-label"}
        save = Mock()
        with patch.object(provision, "instances", return_value=[record]):
            self.assertEqual(provision.reconcile_create(journal=self.journal, attempt=attempt, save=save), record)
        self.assertEqual(attempt["instance_id"], 123)
        save.assert_called_once()

    def test_provisioning_fixture_records_instance_before_setup(self) -> None:
        combo = {"id": "0000", "status": "pending"}
        snapshots = []
        requested = []
        def save():
            snapshots.append(deepcopy(combo))
        def records(journal):
            if not requested:
                return []
            return [{"id": 123, "label": combo["attempts"][0]["label"], "actual_status": "running",
                     "status_msg": "", "dph_total": 0.55, "jupyter_token": "sensitive-test-token"}]
        def command(**kwargs):
            args = kwargs["args"]
            if args[:3] == ["vastai", "create", "instance"]:
                self.assertEqual(snapshots[-1]["status"], "provisioning")
                requested.append(True)
                output = "unparseable response recovered by label"
            elif args[:2] == ["vastai", "ssh-url"]:
                output = "ssh://root@example.invalid:2222\n"
            elif args[0] == "ssh-keyscan":
                output = "[example.invalid]:2222 ssh-ed25519 test-key\n"
            elif args[0] == "ssh":
                run_dir = self.root / ".vast-train-local/toy-act-123"
                self.assertTrue((run_dir / "instance.json").exists())
                self.assertEqual(snapshots[-1]["instance_id"], 123)
                output = json.dumps(fixture_probe())
            else:
                self.fail(f"Unexpected external command: {args[0]}")
            return subprocess.CompletedProcess(args, 0, output, "")
        with patch.object(provision, "REPO", self.root), \
                patch.object(provision, "search_offers", return_value=[fixture_offer()]), \
                patch.object(provision, "git_preflight", return_value=COMMIT), \
                patch.object(provision, "instances", side_effect=records), \
                patch.object(provision, "network_gate", return_value=True), \
                patch.object(provision.time, "sleep"), patch.object(self.journal, "run", side_effect=command):
            run = provision.provision(journal=self.journal, combo=combo, directory=self.directory / "combo",
                                      commit=COMMIT, save=save)
        self.assertEqual(combo["status"], "setup")
        self.assertEqual(combo["attempts"][0]["status"], "accepted")
        self.assertEqual(run["instance_id"], 123)
        self.assertEqual(Path(run["run_dir"]).joinpath("instance.json").stat().st_mode & 0o777, 0o600)
        self.assertNotIn('sensitive-test-token', (self.directory / 'events.jsonl').read_text())

    def test_cleanup_failure_cannot_authorize_replacement(self) -> None:
        attempt = {"instance_id": 123, "label": "saved-label", "status": "created"}
        with patch.object(provision, "instances", side_effect=[[{"id": 123, "label": "saved-label"}], common.Blocked("API down")]), \
                patch.object(self.journal, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            with self.assertRaises(common.Blocked):
                provision.destroy_provisional(journal=self.journal, attempt=attempt, save=lambda: None)
        self.assertEqual(attempt["status"], "created")

    def test_interrupted_provisional_removal_can_reconcile_absence(self) -> None:
        attempt = {"instance_id": 123, "label": "saved-label", "status": "created", "rejected": "hardware gate"}
        saved = Mock()
        with patch.object(provision, "instances", return_value=[]), patch.object(self.journal, "run") as run:
            provision.destroy_provisional(journal=self.journal, attempt=attempt, save=saved)
        self.assertEqual(attempt["status"], "removed")
        saved.assert_called_once()
        run.assert_not_called()

    def test_cleanup_rejects_unrelated_identity(self) -> None:
        attempt = {"instance_id": 123, "label": "saved-label"}
        with patch.object(provision, "instances", return_value=[{"id": 123, "label": "other"}]), \
                patch.object(self.journal, "run") as run:
            with self.assertRaises(common.Blocked):
                provision.destroy_provisional(journal=self.journal, attempt=attempt, save=lambda: None)
            run.assert_not_called()


class RecoveryTests(LocalFixture):
    def active_combo(self) -> tuple[iteration.Iteration, dict]:
        saved = self.make_iteration()
        combo = saved.state["combos"][0]
        run_dir = self.directory / "run"
        run_dir.mkdir()
        combo.update(status="awaiting_handoff", training_launch_requested=True, instance_id=123,
                     run_dir=str(run_dir), run={"instance_id": 123, "label": "saved-label", "run_dir": str(run_dir)})
        return saved, combo

    def report(self, *, combo: dict, verification: bool = True) -> None:
        (Path(combo["run_dir"]) / "report.txt").write_text(
            'instance_id: 123\ninstance_label: saved-label\noutcome: success\n'
            'final_cleanup_status: destroyed_and_verified\n'
            f's3_verification: {"verified snapshots" if verification else "failed"}\n'
            'tensorboard_verification: verified: scalars\ns3_uri: s3://toy-act/checkpoints/act_v2/test/\n')

    def test_terminal_report_requires_confirmed_api_absence(self) -> None:
        saved, combo = self.active_combo()
        self.report(combo=combo)
        with patch.object(workflow, "watcher_alive", return_value=False), \
                patch.object(workflow, "instances", side_effect=common.Blocked("unknown")):
            with self.assertRaises(common.Blocked):
                workflow.reconcile(iteration=saved, combo=combo)
        self.assertEqual(combo["status"], "awaiting_handoff")

    def test_terminal_and_verification_outcomes_are_distinct(self) -> None:
        saved, combo = self.active_combo()
        for verified, expected in ((True, "done"), (False, "verification_failed")):
            combo["status"] = "awaiting_handoff"
            self.report(combo=combo, verification=verified)
            with patch.object(workflow, "watcher_alive", return_value=False), \
                    patch.object(workflow, "instances", return_value=[]):
                self.assertTrue(workflow.reconcile(iteration=saved, combo=combo))
            self.assertEqual(combo["status"], expected)

    def test_dead_watcher_restarts_without_reprovisioning(self) -> None:
        saved, combo = self.active_combo()
        with patch.object(workflow, "watcher_alive", return_value=False), \
                patch.object(workflow, "ensure_watcher") as restart, patch.object(workflow, "provision") as rent:
            self.assertFalse(workflow.reconcile(iteration=saved, combo=combo))
            restart.assert_called_once()
            rent.assert_not_called()

    def test_interrupted_watcher_report_is_archived_and_recovered(self) -> None:
        saved, combo = self.active_combo()
        path = Path(combo["run_dir"]) / "report.txt"
        path.write_text('instance_id: 123\ninstance_label: saved-label\noutcome: unknown\n'
                        'final_cleanup_status: refused_gate_closed\n')
        with patch.object(workflow, "watcher_alive", return_value=False), \
                patch.object(workflow, "ensure_watcher") as restart, \
                patch.object(workflow, "instances") as query:
            self.assertFalse(workflow.reconcile(iteration=saved, combo=combo))
            restart.assert_called_once()
            query.assert_not_called()
        history = list((Path(combo["run_dir"]) / "report-history").glob('*.txt'))
        self.assertEqual(len(history), 1)
        self.assertFalse(path.exists())
        self.assertIn('outcome: unknown', history[0].read_text())

    def test_unverified_cleanup_can_be_verified_on_resume(self) -> None:
        saved, combo = self.active_combo()
        self.report(combo=combo)
        path = Path(combo["run_dir"]) / "report.txt"
        path.write_text(path.read_text().replace('destroyed_and_verified', 'destroy_unverified'))
        with patch.object(workflow, "watcher_alive", return_value=False), \
                patch.object(workflow, "instances", return_value=[]):
            self.assertTrue(workflow.reconcile(iteration=saved, combo=combo))
        self.assertEqual(combo["status"], "done")
        self.assertTrue(combo["cleanup_verified_on_resume"])
        self.assertEqual(combo["report"]["final_cleanup_status"], "destroy_unverified")

    def test_launch_intent_and_watcher_precede_ssh(self) -> None:
        _, combo = self.active_combo()
        combo["status"] = "ready_to_launch"
        combo["training_launch_requested"] = False
        saved, events = [], []
        def save():
            saved.append(deepcopy(combo))
        def remote(**kwargs):
            self.assertTrue(saved[-1]["training_launch_requested"])
            self.assertEqual(events, ["watcher"])
            raise common.Blocked("SSH response lost")
        combo["run"].update(host="example.invalid", port=22, known_hosts="/tmp/not-used")
        with patch.object(setup_run, "ensure_forwarding"), \
                patch.object(setup_run, "ensure_watcher", side_effect=lambda **kwargs: events.append("watcher")), \
                patch.object(self.journal, "run", side_effect=remote):
            with self.assertRaises(common.Blocked):
                setup_run.launch(journal=self.journal, combo=combo, save=save)
        self.assertEqual(combo["status"], "awaiting_handoff")

    def test_bad_handoff_never_marks_running(self) -> None:
        _, combo = self.active_combo()
        for code, status in ((1, "not_ready"), (2, "invalid"), (3, "terminal")):
            result = subprocess.CompletedProcess([], code, json.dumps({"status": status}), "")
            with patch.object(self.journal, "run", return_value=result):
                if code == 3:
                    self.assertEqual(setup_run.handoff(journal=self.journal, combo=combo, save=lambda: None), 3)
                else:
                    with self.assertRaises(common.Blocked):
                        setup_run.handoff(journal=self.journal, combo=combo, save=lambda: None)
            self.assertEqual(combo["status"], "awaiting_handoff")

    def test_dataset_access_error_is_not_missing_object(self) -> None:
        result = subprocess.CompletedProcess([], 1, "", "An error occurred (403) Forbidden")
        with patch.object(self.journal, "run", return_value=result):
            with self.assertRaises(common.Blocked):
                setup_run.dataset_head(journal=self.journal, key="datasets/test.hdf5")

    def test_dataset_traversal_rejected(self) -> None:
        for path in ("/datasets/test.hdf5", "datasets/../test.hdf5", "elsewhere/test.hdf5"):
            with self.assertRaises(common.Blocked):
                setup_run.dataset_path({"dataset": path})

    def test_credentials_are_allowlisted_private_and_removed_even_on_failure(self) -> None:
        _, combo = self.active_combo()
        run = combo["run"]
        run.update(host="example.invalid", port=22, known_hosts="/tmp/not-used")
        credentials = ('AWS_ACCESS_KEY_ID="test-access"\nAWS_SECRET_ACCESS_KEY=\'test-secret\'\n'
                       'AWS_SESSION_TOKEN=test-token\nVAST_API_KEY=never-transfer\n')
        for fail_transfer in (False, True):
            paths = []
            def command(**kwargs):
                args = kwargs["args"]
                if args[0] == 'aws':
                    self.assertTrue(kwargs["sensitive"])
                    return subprocess.CompletedProcess(args, 0, credentials, "")
                if args[0] == 'scp':
                    source = Path(args[-2])
                    paths.append(source)
                    self.assertEqual(source.stat().st_mode & 0o777, 0o600)
                    text = source.read_text()
                    self.assertIn('AWS_SECRET_ACCESS_KEY=test-secret', text)
                    self.assertNotIn('VAST_API_KEY', text)
                    if fail_transfer:
                        raise common.Blocked("transfer failed")
                return subprocess.CompletedProcess(args, 0, "", "")
            with patch.object(self.journal, "run", side_effect=command):
                if fail_transfer:
                    with self.assertRaises(common.Blocked):
                        setup_run.transfer_credentials(journal=self.journal, run=run)
                else:
                    setup_run.transfer_credentials(journal=self.journal, run=run)
            self.assertEqual(len(paths), 1)
            self.assertFalse(paths[0].exists())

    def test_modified_watcher_snapshot_cannot_be_replaced_or_restarted(self) -> None:
        saved, combo = self.active_combo()
        run_dir = Path(combo["run_dir"])
        common.atomic_write(path=run_dir / "setup.env", text='WATCHER_ASSUME_STARTED=yes\n')
        common.atomic_write(path=run_dir / "watcher.sh", text='#!/bin/bash\n')
        combo["setup_sha256"] = common.digest(run_dir / "setup.env")
        combo["watcher_sha256"] = common.digest(run_dir / "watcher.sh")
        saved.validate()
        (run_dir / "watcher.sh").write_text('modified\n')
        with self.assertRaises(common.Blocked):
            saved.validate()
        with patch.object(setup_run.subprocess, "Popen") as spawn:
            with self.assertRaises(common.Blocked):
                setup_run.ensure_watcher(journal=self.journal, combo=combo)
            spawn.assert_not_called()

    def test_structured_events_remain_valid_json_after_redaction(self) -> None:
        self.journal.event(kind="error", reason='AWS_ACCESS_KEY_ID=secret \"jupyter_token\": \"hidden\"',
                           fields={"AWS_SESSION_TOKEN": "hidden"})
        event = json.loads((self.directory / "events.jsonl").read_text().splitlines()[-1])
        self.assertNotIn('secret', json.dumps(event))
        self.assertNotIn('hidden', json.dumps(event))
        self.assertEqual(event["kind"], "error")

    def test_command_logs_stream_before_process_finishes(self) -> None:
        log_path = self.directory / "setup.log"
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.journal.run, args=["bash", "-c", "echo first-stage; sleep 1; echo second-stage"],
                                     timeout=5, log_path=log_path)
            end = time.monotonic() + 0.8
            while time.monotonic() < end:
                if log_path.exists() and "first-stage" in log_path.read_text():
                    break
                time.sleep(0.01)
            self.assertIn("first-stage", log_path.read_text())
            self.assertFalse(future.done())
            result = future.result(timeout=5)
        self.assertEqual(result.stdout, "first-stage\nsecond-stage\n")
        self.assertIn("second-stage", log_path.read_text())

    def test_timeout_preserves_partial_log_and_sensitive_output_is_omitted(self) -> None:
        log_path = self.directory / "setup.log"
        with self.assertRaises(common.Blocked):
            self.journal.run(args=["bash", "-c", "echo started; sleep 5"], timeout=0.2, log_path=log_path)
        self.assertIn("started", log_path.read_text())
        self.assertIn("timed out", log_path.read_text())
        result = self.journal.run(args=["bash", "-c", "echo must-not-be-logged"], sensitive=True, log_path=log_path)
        self.assertEqual(result.stdout.strip(), "must-not-be-logged")
        self.assertNotIn("must-not-be-logged", log_path.read_text())

    def test_logging_redacts_credentials_and_signed_urls(self) -> None:
        text = 'AWS_SECRET_ACCESS_KEY=secret VAST_API_KEY=token https://example.invalid/key?X-Amz-Signature=secret'
        cleaned = common.redact(text)
        self.assertNotIn("secret", cleaned)
        self.assertNotIn("=token", cleaned)
        self.assertNotIn("Signature", cleaned)

    def planned_pair(self) -> iteration.Iteration:
        saved = self.make_iteration()
        entry = deepcopy(saved.manifest["combos"][0])
        entry.update(id="0001", slug="second", config="configs/second.toml")
        source = saved.directory / saved.manifest["combos"][0]["config"]
        target = saved.directory / entry["config"]
        target.write_text(source.read_text().replace('name = "example-', 'name = "second-'))
        entry["sha256"] = common.digest(target)
        saved.manifest["combos"].append(entry)
        saved.state["combos"].append({"id": "0001", "status": "pending", "attempts": []})
        common.write_json(path=saved.directory / "manifest.json", value=saved.manifest)
        saved.state["manifest_sha256"] = common.digest(saved.directory / "manifest.json")
        saved.save()
        return saved

    def run_pipeline_fixture(self, *, saved: iteration.Iteration, fail_gate: bool = False) -> list[tuple]:
        events = []
        def rental(**kwargs):
            combo = kwargs["combo"]
            events.append(("provision", combo["id"]))
            run_dir = self.directory / f"run-{combo['id']}"
            run_dir.mkdir()
            combo.update(status="setup", run_dir=str(run_dir), instance_id=int(combo["id"]) + 123)
            kwargs["save"]()
        def configure(**kwargs):
            combo = kwargs["combo"]
            events.append(("setup", combo["id"]))
            combo.update(status="ready_to_launch", lease={"TB_PORT": 6007 + int(combo["id"])})
            kwargs["save"]()
        def start(**kwargs):
            combo = kwargs["combo"]
            events.append(("launch", combo["id"]))
            combo.update(status="awaiting_handoff", training_launch_requested=True, services_started=True)
            kwargs["save"]()
        def gate(**kwargs):
            combo = kwargs["combo"]
            events.append(("handoff", combo["id"]))
            if fail_gate:
                raise common.Blocked("unknown handoff")
            combo["status"] = "running"
            kwargs["save"]()
            return 0
        with patch.object(workflow, "git_preflight", return_value=COMMIT), \
                patch.object(workflow, "local_preflight"), patch.object(workflow, "validate_dataset_source"), \
                patch.object(workflow, "provision", side_effect=rental), \
                patch.object(workflow, "setup", side_effect=configure), \
                patch.object(workflow, "launch", side_effect=start), \
                patch.object(workflow, "handoff", side_effect=gate):
            if fail_gate:
                with self.assertRaises(common.Blocked):
                    workflow.execute(saved)
            else:
                workflow.execute(saved)
        return events

    def test_pipeline_requires_two_fresh_gates_before_next_rental(self) -> None:
        saved = self.planned_pair()
        events = self.run_pipeline_fixture(saved=saved)
        self.assertEqual(events, [
            ("provision", "0000"), ("setup", "0000"), ("launch", "0000"),
            ("handoff", "0000"), ("handoff", "0000"),
            ("provision", "0001"), ("setup", "0001"), ("launch", "0001"),
            ("handoff", "0001"), ("handoff", "0001")])
        self.assertEqual([combo["status"] for combo in saved.state["combos"]], ["running", "running"])

    def test_pipeline_stops_new_rentals_on_unknown_handoff(self) -> None:
        saved = self.planned_pair()
        events = self.run_pipeline_fixture(saved=saved, fail_gate=True)
        self.assertEqual(events[-1], ("handoff", "0000"))
        self.assertNotIn(("provision", "0001"), events)
        self.assertEqual([combo["status"] for combo in saved.state["combos"]], ["awaiting_handoff", "pending"])

    def test_terminal_only_resume_does_not_require_new_git_revision(self) -> None:
        saved = self.make_iteration()
        saved.state["combos"][0]["status"] = "done"
        with patch.object(workflow, "git_preflight") as git, patch.object(workflow, "local_preflight") as preflight:
            workflow.execute(saved)
            git.assert_not_called()
            preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
