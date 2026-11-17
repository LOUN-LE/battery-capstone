"""Offline tests: no AWS calls, real solver, systemd mutation, or shutdown."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import venv
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

NOTIFY = module("configure_stop_email")
CLOUD = module("run_lifetime_cloud")

class CloudControls(unittest.TestCase):
    def test_notification_scope_and_no_role(self):
        plan = NOTIFY.build_plan("123456789012", "us-east-1", "i-1234567890abcdef0", "example@example.com")
        self.assertEqual(plan["event_pattern"]["detail"]["instance-id"], ["i-1234567890abcdef0"])
        self.assertEqual(plan["event_pattern"]["detail"]["state"], ["stopped", "terminated"])
        self.assertEqual(plan["publish_statement"]["Condition"]["ArnEquals"]["aws:SourceArn"], plan["rule_arn"])
        self.assertEqual(plan["publish_statement"]["Resource"], plan["topic_arn"])
        self.assertNotIn("RoleArn", plan["target"])
        self.assertIn("실험 성공을 뜻하지", json.loads(plan["target"]["InputTransformer"]["InputTemplate"]))

    def test_notification_rejects_wrong_ids(self):
        with self.assertRaises(ValueError):
            NOTIFY.build_plan("1234", "us-east-1", "i-1234567890abcdef0", "example@example.com")
        with self.assertRaises(ValueError):
            NOTIFY.build_plan("123456789012", "us-east-1", "i-1234567890abcdef0", "bad address")

    def test_pending_subscription_arn_is_not_confirmation(self):
        sub = {"SubscriptionArn":"arn:aws:sns:us-east-1:123456789012:topic:pending-id"}
        with patch.object(NOTIFY, "aws", return_value={"Attributes":{"PendingConfirmation":"true"}}):
            self.assertFalse(NOTIFY.inspect_subscription("us-east-1", sub)["confirmed"])
        with patch.object(NOTIFY, "aws", return_value={"Attributes":{"PendingConfirmation":"false"}}):
            self.assertTrue(NOTIFY.inspect_subscription("us-east-1", sub)["confirmed"])
        with patch.object(NOTIFY, "aws", return_value={"Attributes":{}}):
            self.assertFalse(NOTIFY.inspect_subscription("us-east-1", sub)["confirmed"])

    def test_compact_outcome_does_not_export_history(self):
        huge = {"status":"max_efc_censored", "ledger":{"efc":3000},
            "cycles":[{"large":"x"*1000}]*10000, "checks":[1]*10000,
            "diagnostics":[{"efc":2990,"SOH_percent":95.0}]*302,
            "files":["file"]*10000, "elapsed_wall_s":70000,
            "eol":None}
        result = CLOUD.compact_outcome(huge, None, Path("results"))
        self.assertLess(len(json.dumps(result)), 1000)
        self.assertEqual(result["last_valid_EFC"], 3000)
        self.assertEqual(result["last_SOH_percent"], 95.0)
        self.assertNotIn("cycles", result)

    def observed_failure_fixture(self):
        # Shape and numbers from the real 2026-10-29 stale-outcome incident.
        old = {"status":"paused_verified", "right_censored":True,
            "last_valid_EFC":100.43647188455031, "elapsed_wall_s":2441.7482158109997,
            "ledger":{"efc":100.43647188455031},
            "diagnostics":[{"efc":100.0,"SOH_percent":99.79109053020528}]}
        summary = {"status":"running", "elapsed_wall_s":58433.699568414995,
            "max_EFC":3000, "ledger":{"efc":1614.3393723356705},
            "diagnostics":[{"efc":100.0,"SOH_percent":99.79109053020528},
                           {"efc":1610.0,"SOH_percent":98.94352059914654}]}
        failure = {"status":"failed", "elapsed_wall_s":58455.26863223,
                   "error":"RuntimeError: Physical state connection mismatch"}
        return old, summary, failure

    def eol_fixture(self):
        return {"status":"eol_observed", "last_valid_EFC":1620.,
            "elapsed_wall_s":60000., "ledger":{"efc":1620.},
            "diagnostics":[{"efc":1610.,"SOH_percent":80.1},
                           {"efc":1620.,"SOH_percent":79.9}],
            "eol":{"threshold_percent":80., "observed_at_EFC":1620.,
                   "interval_lower_EFC":1610., "interval_upper_EFC":1620.,
                   "exact_crossing_known":False, "rul_from_100_observation_EFC":1520.}}

    def test_actual_stale_pause_failure_uses_latest_checkpoint_and_measurement(self):
        old, summary, failure = self.observed_failure_fixture()
        original = copy.deepcopy((old, summary, failure))
        result = CLOUD.compact_outcome(old, summary, Path("results"),
            failure=failure, runner_exit_code=1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["outcome_ignored"], "stale")
        self.assertAlmostEqual(result["last_valid_EFC"], 1614.3393723356705)
        self.assertEqual(result["last_SOH_measurement_EFC"], 1610.)
        self.assertAlmostEqual(result["last_SOH_percent"], 98.94352059914654)
        self.assertAlmostEqual(result["elapsed_wall_s"], 58455.26863223)
        self.assertIsNone(result["right_censored"])
        self.assertIsNone(result["eol"])
        self.assertEqual((old, summary, failure), original)

    def test_failing_exit_cannot_reuse_apparently_current_pause(self):
        old, _, _ = self.observed_failure_fixture()
        result = CLOUD.compact_outcome(old, old, Path("results"), runner_exit_code=1)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["right_censored"])

    def test_old_verification_pause_does_not_explain_timeout_exit(self):
        old, _, _ = self.observed_failure_fixture()
        result = CLOUD.compact_outcome(old, old, Path("results"), runner_exit_code=124)
        self.assertEqual(result["status"], "runner_exit_nonzero_without_valid_outcome")
        self.assertIsNone(result["right_censored"])

    def test_valid_capacity_eol_and_old_failure_are_distinct(self):
        observed = self.eol_fixture()
        result = CLOUD.compact_outcome(observed, observed, Path("results"),
            failure={"status":"failed", "elapsed_wall_s":100.}, runner_exit_code=0)
        self.assertEqual(result["status"], "eol_observed")
        self.assertFalse(result["right_censored"])
        self.assertEqual(result["last_SOH_measurement_EFC"], 1620.)
        self.assertEqual(result["eol"]["interval_lower_EFC"], 1610.)

    def test_valid_maximum_and_time_censoring_remain_censored(self):
        for status, code in (("max_efc_censored", 0), ("wall_time_censored", 124)):
            with self.subTest(status=status):
                final = {"status":status, "max_EFC":3000.,
                    "ledger":{"efc":3000. if code == 0 else 1614.3393723356705},
                    "elapsed_wall_s":72000.,
                    "diagnostics":[{"efc":1610.,"SOH_percent":98.94}]}
                result = CLOUD.compact_outcome(final, final, Path("results"), runner_exit_code=code)
                self.assertEqual(result["status"], status)
                self.assertTrue(result["right_censored"])
                self.assertIsNone(result["eol"])

    def test_incomplete_eol_or_missing_outcome_does_not_create_eol(self):
        _, summary, _ = self.observed_failure_fixture()
        for outcome in (None, {"status":"eol_observed"}):
            with self.subTest(outcome=outcome):
                result = CLOUD.compact_outcome(outcome, summary, Path("results"), runner_exit_code=0)
                self.assertEqual(result["status"], "runner_exit_zero_without_valid_outcome")
                self.assertIsNone(result["eol"])
                self.assertIsNone(result["right_censored"])
                self.assertEqual(result["last_SOH_measurement_EFC"], 1610.)

    def test_eol_claim_requires_an_actual_threshold_crossing(self):
        final = self.eol_fixture()
        final["diagnostics"][-1]["SOH_percent"] = 98.94
        result = CLOUD.compact_outcome(final, final, Path("results"), runner_exit_code=0)
        self.assertEqual(result["status"], "invalid_eol_evidence")
        self.assertIsNone(result["eol"])
        self.assertIsNone(result["right_censored"])

    def test_outcome_cannot_invent_state_beyond_committed_checkpoint(self):
        final = {"status":"max_efc_censored", "max_EFC":3000.,
                 "last_valid_EFC":3000., "elapsed_wall_s":72000.,
                 "diagnostics":[{"efc":3000.,"SOH_percent":95.}]}
        summary = {"status":"running", "ledger":{"efc":100.}, "elapsed_wall_s":100.,
                   "diagnostics":[{"efc":100.,"SOH_percent":99.}]}
        result = CLOUD.compact_outcome(final, summary, Path("results"), runner_exit_code=0)
        self.assertEqual(result["last_valid_EFC"], 100.)
        self.assertEqual(result["last_SOH_measurement_EFC"], 100.)
        self.assertEqual(result["status"], "invalid_termination_evidence")

    def test_supervisor_resource_guard_keeps_safe_state_without_claiming_eol(self):
        old, summary, _ = self.observed_failure_fixture()
        result = CLOUD.compact_outcome(old, summary, Path("results"),
            runner_exit_code=-9, stop_reason="disk_guard")
        self.assertEqual(result["status"], "resource_limit_censored:disk")
        self.assertTrue(result["right_censored"])
        self.assertIsNone(result["eol"])
        self.assertEqual(result["last_SOH_measurement_EFC"], 1610.)

    def test_terminate_waits_for_worker_after_parent_exit(self):
        proc=SimpleNamespace(pid=1234,poll=lambda:0,wait=lambda **kwargs:0)
        fake_os=SimpleNamespace(killpg=lambda *args:None)
        fake_signal=SimpleNamespace(SIGTERM=15,SIGKILL=9)
        with patch.object(CLOUD,"os",fake_os), patch.object(CLOUD,"signal",fake_signal), \
             patch.object(CLOUD,"process_group_alive",side_effect=[True,True,False,False,False]), \
             patch.object(CLOUD.time,"sleep") as sleeper:
            result=CLOUD.terminate_tree(proc)
        self.assertTrue(result["term_sent"])
        self.assertTrue(result["group_empty"])
        self.assertTrue(sleeper.called)
        self.assertFalse(result["kill_sent"])

    def test_terminate_escalates_and_waits_for_group(self):
        proc=SimpleNamespace(pid=1234,poll=lambda:0,wait=lambda **kwargs:0)
        killed=[]
        fake_os=SimpleNamespace(killpg=lambda *args:killed.append(args))
        fake_signal=SimpleNamespace(SIGTERM=15,SIGKILL=9)
        with patch.object(CLOUD,"os",fake_os), patch.object(CLOUD,"signal",fake_signal), \
             patch.object(CLOUD,"process_group_alive",side_effect=[True,True,True,False,False]), \
             patch.object(CLOUD.time,"monotonic",side_effect=[0,1,2]), \
             patch.object(CLOUD.time,"sleep"):
            result=CLOUD.terminate_tree(proc,grace_s=0)
        self.assertEqual(killed,[(1234,15),(1234,9)])
        self.assertTrue(result["kill_sent"])
        self.assertTrue(result["group_empty"])

    def settings(self, tmp):
        repo = Path(tmp) / "repo"
        (repo / "simulation").mkdir(parents=True)
        (repo / "simulation/run_lifetime.py").write_text("# mock", encoding="utf-8")
        run = Path(tmp) / "run"
        return SimpleNamespace(repo_root=str(repo), python=sys.executable,
            run_root=str(run), cell=4, current=3.2, mesh_scale=4, efc=3000,
            max_wall_s=72000, rpt_every_efc=10, resume=False,
            import_100efc_dir=str(Path(tmp) / "seed"),
            hard_stop_extra_s=900, guard_unit="capstone-lifetime-hard-stop",
            minimum_free_gib=2, minimum_available_ram_fraction=.08, plan=False)

    def test_cli_contract_and_separate_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.settings(tmp)
            command = CLOUD.runner_command(args)
            self.assertEqual(command[command.index("--efc")+1], "3000")
            self.assertEqual(command[command.index("--max-wall-s")+1], "72000")
            self.assertEqual(command[command.index("--output-dir")+1], str(Path(args.run_root) / "trajectory"))
            self.assertIn("--import-100efc-dir", command)
            args.resume = True
            self.assertIn("--resume", CLOUD.runner_command(args))
            self.assertNotIn("--import-100efc-dir", CLOUD.runner_command(args))
            self.assertIn("--on-active=72900s", CLOUD.hard_stop_command(args))

    @unittest.skipUnless(sys.platform.startswith("linux"), "real venv symlink fixture requires Linux")
    def test_venv_symlink_preserves_prefix_and_venv_only_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.settings(tmp)
            prefix = Path(tmp) / "fixture-venv"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(prefix)
            launcher = prefix / "bin/python"
            self.assertTrue(launcher.is_symlink())
            args.python = str(launcher)
            command = CLOUD.runner_command(args)
            self.assertEqual(command[0], str(launcher.absolute()))
            purelib = subprocess.run([command[0], "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))"],
                check=True, capture_output=True, text=True).stdout.strip()
            module_name = "capstone_venv_only_regression_fixture"
            (Path(purelib) / (module_name + ".py")).write_text("VALUE = 'venv-only'\n", encoding="utf-8")
            probe = subprocess.run([command[0], "-c",
                "import json,sys," + module_name + "; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'value':" + module_name + ".VALUE}))"],
                check=True, capture_output=True, text=True)
            value = json.loads(probe.stdout)
            self.assertEqual(Path(value["prefix"]).resolve(), prefix.resolve())
            self.assertNotEqual(value["prefix"], value["base_prefix"])
            self.assertEqual(value["value"], "venv-only")
            # The old resolve() command would choose this base interpreter.
            bad_probe = subprocess.run([str(launcher.resolve()), "-c",
                "import importlib.util; print(importlib.util.find_spec('" + module_name + "') is None)"],
                check=True, capture_output=True, text=True)
            self.assertEqual(bad_probe.stdout.strip(), "True")

    def test_result_preserved_then_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.settings(tmp)
            child = Path(args.run_root) / "trajectory"
            child.mkdir(parents=True)
            final_outcome = self.eol_fixture()
            (child / "outcome.json").write_text(json.dumps(final_outcome), encoding="utf-8")
            (child / "summary.json").write_text(json.dumps(final_outcome), encoding="utf-8")
            calls = []
            saved = []
            fake_os = SimpleNamespace(name="posix", geteuid=lambda: 0,
                environ={}, fsync=CLOUD.os.fsync)
            proc = SimpleNamespace(pid=1234, poll=lambda: 0, wait=lambda **kwargs: 0)
            def call(cmd, **kwargs):
                calls.append(cmd)
                return SimpleNamespace(returncode=0)
            with patch.object(CLOUD, "os", fake_os), patch.object(CLOUD.subprocess, "run", side_effect=call), \
                 patch.object(CLOUD.subprocess, "Popen", return_value=proc), \
                 patch.object(CLOUD.shutil, "disk_usage", return_value=SimpleNamespace(free=10*CLOUD.GIB)), \
                 patch.object(CLOUD, "terminate_tree", return_value={"term_sent":False,"kill_sent":False,"group_empty":True}), \
                 patch.object(CLOUD, "open", create=True), \
                 patch.object(CLOUD, "save_json", side_effect=lambda path,value: saved.append(copy.deepcopy(value))):
                self.assertEqual(CLOUD.run(args), 0)
            final = saved[-1]
            self.assertEqual(final["status"], "eol_observed")
            self.assertTrue(final["shutdown_requested"])
            self.assertEqual(calls[0][0], "systemd-run")
            self.assertEqual(calls[-2], ["sync"])
            self.assertEqual(calls[-1], ["/usr/sbin/shutdown", "-h", "now"])

    def test_real_failure_shape_is_saved_as_failed_and_shutdown_still_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.settings(tmp)
            child = Path(args.run_root) / "trajectory"
            child.mkdir(parents=True)
            old, summary, failure = self.observed_failure_fixture()
            for name, value in (("outcome.json",old),("summary.json",summary),("FAILED.json",failure)):
                (child/name).write_text(json.dumps(value),encoding="utf-8")
            before = {p.name:p.read_bytes() for p in child.iterdir()}
            calls, saved = [], []
            fake_os = SimpleNamespace(name="posix",geteuid=lambda:0,environ={},fsync=CLOUD.os.fsync)
            proc = SimpleNamespace(pid=1234,poll=lambda:1,wait=lambda **kwargs:1)
            def call(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0)
            with patch.object(CLOUD,"os",fake_os), patch.object(CLOUD.subprocess,"run",side_effect=call), \
                 patch.object(CLOUD.subprocess,"Popen",return_value=proc), \
                 patch.object(CLOUD.shutil,"disk_usage",return_value=SimpleNamespace(free=10*CLOUD.GIB)), \
                 patch.object(CLOUD,"terminate_tree",return_value={"term_sent":False,"kill_sent":False,"group_empty":True}), \
                 patch.object(CLOUD,"open",create=True), \
                 patch.object(CLOUD,"save_json",side_effect=lambda path,value:saved.append(copy.deepcopy(value))):
                self.assertEqual(CLOUD.run(args),1)
            final = saved[-1]
            self.assertEqual(final["status"],"failed")
            self.assertEqual(final["outcome"]["status"],"failed")
            self.assertEqual(final["outcome"]["last_SOH_measurement_EFC"],1610.)
            self.assertIsNone(final["outcome"]["right_censored"])
            self.assertEqual(calls[-2:],[['sync'],['/usr/sbin/shutdown','-h','now']])
            self.assertEqual({p.name:p.read_bytes() for p in child.iterdir()},before)

    def test_runner_launch_failure_still_records_and_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.settings(tmp)
            calls = []
            saved = []
            fake_os = SimpleNamespace(name="posix", geteuid=lambda: 0,
                environ={}, fsync=CLOUD.os.fsync)
            def call(cmd, **kwargs):
                calls.append(cmd)
                return SimpleNamespace(returncode=0)
            with patch.object(CLOUD, "os", fake_os), patch.object(CLOUD.subprocess, "run", side_effect=call), \
                 patch.object(CLOUD.subprocess, "Popen", side_effect=RuntimeError("mock launch failure")), \
                 patch.object(CLOUD.shutil, "disk_usage", return_value=SimpleNamespace(free=10*CLOUD.GIB)), \
                 patch.object(CLOUD, "terminate_tree", return_value={"term_sent":False,"kill_sent":False,"group_empty":True}), \
                 patch.object(CLOUD, "open", create=True), \
                 patch.object(CLOUD, "save_json", side_effect=lambda path,value: saved.append(copy.deepcopy(value))):
                self.assertEqual(CLOUD.run(args), 1)
            final = saved[-1]
            self.assertEqual(final["status"], "supervisor_failed")
            self.assertEqual(calls[-1], ["/usr/sbin/shutdown", "-h", "now"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
