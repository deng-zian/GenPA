"""CPU-only regression tests for evaluation completeness and aggregation."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "experiments/libero/summarize_results.py"
SPEC = importlib.util.spec_from_file_location("genpa_result_summary", SUMMARY_PATH)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class ResultIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.results = Path(self.temporary.name)
        for suite, task_id in summary.full_task_set("libero"):
            self.write_result(
                self.make_result(suite, task_id, trials=50, successes=49)
            )

    @staticmethod
    def make_result(suite, task_id, *, trials, successes, benchmark="libero"):
        return {
            "benchmark": benchmark,
            "task_suite": suite,
            "task_id": task_id,
            "total_episodes": trials,
            "successes": successes,
            "success_episodes": list(range(successes)),
            "failure_episodes": list(range(successes, trials)),
            "duration": 1.0,
            "instruction_protocol": "official",
            "evaluation_seed": 42,
            "checkpoint_sha256": "a" * 64,
        }

    def write_result(self, row, *, gpu=0):
        directory = self.results / row["task_suite"]
        directory.mkdir(exist_ok=True)
        path = directory / f"gpu{gpu}_task{row['task_id']}_results.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        return path

    def first_result(self):
        path = self.results / "libero_spatial/gpu0_task0_results.json"
        return path, json.loads(path.read_text(encoding="utf-8"))

    def clear_results(self):
        for path in self.results.rglob("gpu*_task*_results.json"):
            path.unlink()

    def collect(self, *, allow_partial=False):
        return summary.collect_results(
            self.results,
            benchmark="libero",
            expected=summary.full_task_set("libero"),
            num_trials=50,
            allow_partial=allow_partial,
        )

    def summarize(self, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return summary.summarize_results(self.results, **options)

    def test_complete_standard_benchmark(self):
        result = self.summarize(benchmark="libero", num_trials=50)
        self.assertTrue(result["complete"])
        self.assertTrue(result["formal"])
        self.assertEqual(result["overall"]["total_trials"], 2000)
        self.assertEqual(result["overall"]["total_successes"], 1960)
        self.assertEqual(result["overall"]["average_success_rate"], 98.0)
        self.assertEqual(result["seed"], 42)
        self.assertEqual(result["evaluation_seed"], 42)
        self.assertEqual(result["checkpoint_sha256"], "a" * 64)
        self.assertTrue(result["identity_verified"])
        for name in ("summary.json", "summary.csv", "task_success_rates.csv"):
            self.assertTrue((self.results / name).is_file())

    def test_missing_task_requires_explicit_partial(self):
        self.first_result()[0].unlink()
        with self.assertRaises(ValueError):
            self.collect()
        rows, missing = self.collect(allow_partial=True)
        self.assertEqual(len(rows), 39)
        self.assertEqual(len(missing), 1)
        result = self.summarize(
            benchmark="libero", num_trials=50, allow_partial=True
        )
        self.assertFalse(result["complete"])
        self.assertFalse(result["formal"])

    def test_duplicate_task_from_different_gpu_is_rejected(self):
        _, row = self.first_result()
        self.write_result(row, gpu=1)
        with self.assertRaises(ValueError):
            self.collect()

    def test_bad_episode_identities_are_rejected(self):
        path, original = self.first_result()
        for failure_ids in ([48], [50], [], [True]):
            with self.subTest(failure_ids=failure_ids):
                row = copy.deepcopy(original)
                row["failure_episodes"] = failure_ids
                path.write_text(json.dumps(row), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.collect()

    def test_bad_counts_are_rejected(self):
        path, original = self.first_result()
        changes = [
            ("successes", 51),
            ("successes", -1),
            ("successes", 48),
            ("total_episodes", 49),
            ("total_episodes", 50.0),
            ("successes", True),
        ]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                row = copy.deepcopy(original)
                row[field] = value
                path.write_text(json.dumps(row), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.collect()

    def test_result_identity_must_match_filename(self):
        path, row = self.first_result()
        row["task_id"] = 1
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.collect()

    def test_unknown_task_is_rejected(self):
        self.write_result(
            self.make_result("libero_spatial", 10, trials=50, successes=49)
        )
        with self.assertRaises(ValueError):
            self.collect()

    def test_mixed_protocol_seed_and_checkpoint_are_rejected(self):
        path, original = self.first_result()
        changes = [
            ("instruction_protocol", "clean"),
            ("evaluation_seed", 1042),
            ("checkpoint_sha256", "b" * 64),
            ("benchmark", "libero_plus"),
        ]
        for field, value in changes:
            with self.subTest(field=field):
                row = copy.deepcopy(original)
                row[field] = value
                path.write_text(json.dumps(row), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.collect()

    def test_conflicting_legacy_seed_alias_is_rejected(self):
        path, row = self.first_result()
        row["seed"] = 1042
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.collect()

    def test_consistent_legacy_seed_alias_is_supported(self):
        for path in self.results.rglob("gpu*_task*_results.json"):
            row = json.loads(path.read_text())
            row["seed"] = row.pop("evaluation_seed")
            path.write_text(json.dumps(row), encoding="utf-8")
        result = self.summarize(benchmark="libero", num_trials=50)
        self.assertEqual(result["evaluation_seed"], 42)
        self.assertTrue(result["identity_verified"])

    def test_missing_fingerprint_among_identified_rows_is_rejected(self):
        path, row = self.first_result()
        del row["checkpoint_sha256"]
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.collect()

    def test_old_actual_rows_without_fingerprint_are_not_identity_verified(self):
        # The frozen evaluator emits evaluation_seed, not seed, and no SHA.
        for path in self.results.rglob("gpu*_task*_results.json"):
            row = json.loads(path.read_text())
            del row["checkpoint_sha256"]
            path.write_text(json.dumps(row), encoding="utf-8")
        result = self.summarize(benchmark="libero", num_trials=50)
        self.assertTrue(result["formal"])
        self.assertEqual(result["evaluation_seed"], 42)
        self.assertFalse(result["identity_verified"])
        self.assertIsNone(result["checkpoint_sha256"])
        path, row = self.first_result()
        row["evaluation_seed"] = 1042
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.collect()

    def test_malformed_identity_fields_are_rejected(self):
        path, original = self.first_result()
        for field, value in [("evaluation_seed", True),
                             ("evaluation_seed", "42"),
                             ("checkpoint_sha256", "not-a-checkpoint-hash")]:
            with self.subTest(field=field, value=value):
                row = copy.deepcopy(original)
                row[field] = value
                path.write_text(json.dumps(row), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.collect()

    def test_malformed_json_is_rejected(self):
        self.first_result()[0].write_text("{", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.collect()

    def test_missing_episode_lists_are_rejected(self):
        path, row = self.first_result()
        del row["failure_episodes"]
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.collect()

    def test_duplicate_task_manifest_is_rejected(self):
        path = self.results / "tasks.txt"
        path.write_text("libero_spatial,0\nlibero_spatial,0\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            summary.read_task_file(path)

    def test_scoped_smoke_is_complete_but_not_formal(self):
        self.clear_results()
        self.write_result(
            self.make_result("libero_spatial", 0, trials=1, successes=0)
        )
        (self.results / "tasks.txt").write_text(
            "libero_spatial,0\n", encoding="utf-8"
        )
        result = self.summarize(benchmark="libero", num_trials=1)
        self.assertTrue(result["complete"])
        self.assertFalse(result["formal"])
        self.assertFalse(result["full_benchmark"])

    def test_full_standard_cli_rejects_narrowed_manifest(self):
        self.clear_results()
        self.write_result(
            self.make_result("libero_spatial", 0, trials=50, successes=49)
        )
        (self.results / "tasks.txt").write_text(
            "libero_spatial,0\n", encoding="utf-8"
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/libero/summarize_standard.py"),
                "--results",
                str(self.results),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Incomplete evaluation", completed.stderr)
        self.assertFalse((self.results / "summary.json").exists())

    def test_complete_plus_is_instance_pooled_and_missing_task_rejected(self):
        self.clear_results()
        for suite, task_id in summary.full_task_set("libero_plus"):
            self.write_result(
                self.make_result(
                    suite,
                    task_id,
                    trials=1,
                    successes=int(task_id % 2 == 0),
                    benchmark="libero_plus",
                )
            )
        result = self.summarize(benchmark="libero_plus", num_trials=1)
        expected_successes = sum(
            (count + 1) // 2 for count in summary.PLUS_COUNTS.values()
        )
        self.assertTrue(result["formal"])
        self.assertEqual(result["overall"]["total_trials"], 10030)
        self.assertEqual(result["overall"]["aggregation"], "instance_pooled")
        self.assertAlmostEqual(
            result["overall"]["average_success_rate"],
            100 * expected_successes / 10030,
        )
        self.first_result()[0].unlink()
        with self.assertRaises(ValueError):
            summary.collect_results(
                self.results,
                benchmark="libero_plus",
                expected=summary.full_task_set("libero_plus"),
                num_trials=1,
            )


if __name__ == "__main__":
    unittest.main()
