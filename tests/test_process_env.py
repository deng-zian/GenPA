"""Exercise the real spawn/RPC adapter with a small CPU simulator fixture."""

import importlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None


ROOT = Path(os.environ.get("GENPA_TEST_REPOSITORY", Path(__file__).resolve().parents[1]))
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "fake_process_env_backend.py"


@unittest.skipIf(np is None, "ProcessEnv tests require NumPy; no simulator or GPU is needed")
class ProcessEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package_dir = tempfile.TemporaryDirectory(prefix="genpa-process-backend-")
        cls.addClassCleanup(cls.package_dir.cleanup)
        cls.addClassCleanup(sys.path.__setitem__, slice(None), sys.path[:])
        cls.addClassCleanup(cls.restore_rng, (random.getstate(), np.random.get_state()))
        package = Path(cls.package_dir.name)
        for directory in (package / "libero", package / "libero" / "libero"):
            directory.mkdir()
            (directory / "__init__.py").write_text("", encoding="utf-8")
        shutil.copyfile(FIXTURE, package / "libero" / "libero" / "envs.py")
        sys.path[:0] = [str(package), str(ROOT / "experiments" / "libero")]
        cls.adapter = importlib.import_module("process_env")
        expected = ROOT / "experiments" / "libero" / "process_env.py"
        if Path(cls.adapter.__file__).resolve() != expected.resolve():
            raise RuntimeError("A different process_env module is already imported")
        spec = importlib.util.spec_from_file_location("cpu_simulator_fixture", FIXTURE)
        backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend)
        cls.direct_backend = backend.OffScreenRenderEnv

    def setUp(self):
        self.saved_rng = (random.getstate(), np.random.get_state())
        self.temp = tempfile.TemporaryDirectory(prefix="genpa-process-test-")
        self.directory = Path(self.temp.name)
        random.seed(193)
        np.random.seed(71)
        self.initial_rng = (random.getstate(), np.random.get_state())
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.restore_rng, self.saved_rng)

    @staticmethod
    def restore_rng(state):
        random.setstate(state[0])
        np.random.set_state(state[1])

    def arguments(self, name):
        return {"event_file": str(self.directory / f"{name}.jsonl"),
                "camera_heights": 12, "camera_widths": 16}

    def events(self, name):
        path = self.directory / f"{name}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def spawned(self, name):
        env = self.adapter.ProcessEnv(self.arguments(name), seed=42, timeout=10.0)
        self.addCleanup(env.close)
        return env

    @staticmethod
    def exercise(env):
        return [env.reset(offset=2.0),
                env.set_init_state(np.array([3.0, -2.0], dtype=np.float32)),
                env.step(np.array([0.5, 1.0], dtype=np.float32), scale=2.0),
                env.seed(99), env.reset()]

    def assert_closed(self, env):
        self.assertFalse(env._process.is_alive(), "Simulator child survived cleanup")
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            env.reset()

    def test_rpc_matches_direct_calls_and_close_is_idempotent(self):
        direct = self.direct_backend(**self.arguments("direct"))
        direct.seed(42)
        expected = self.exercise(direct)
        direct.close()

        self.restore_rng(self.initial_rng)
        env = self.spawned("child")
        self.assertNotEqual(env._process.pid, os.getpid())
        actual = self.exercise(env)
        for expected_value, actual_value in zip(expected, actual):
            np.testing.assert_equal(actual_value, expected_value)
        env.close()
        self.assert_closed(env)
        env.close()
        direct_events = [row["event"] for row in self.events("direct")]
        child_events = [row["event"] for row in self.events("child")]
        self.assertEqual(child_events, direct_events)
        self.assertEqual(child_events.count("construct"), 1)
        self.assertEqual(child_events.count("close"), 1)

    def test_rng_handoff_matches_two_consecutive_direct_environments(self):
        expected = []
        for index in range(2):
            env = self.direct_backend(**self.arguments(f"direct{index}"))
            env.seed(42)
            expected.extend(self.exercise(env))
            env.close()
        expected_next = (random.random(), np.random.random(4))

        self.restore_rng(self.initial_rng)
        actual = []
        child_pids = []
        for index in range(2):
            env = self.spawned(f"child{index}")
            child_pids.append(env._process.pid)
            actual.extend(self.exercise(env))
            env.close()
            self.assert_closed(env)
        for expected_value, actual_value in zip(expected, actual):
            np.testing.assert_equal(actual_value, expected_value)
        np.testing.assert_equal((random.random(), np.random.random(4)), expected_next)
        self.assertEqual(len(set(child_pids)), 2)

    def test_method_error_reaches_parent_and_closes_child(self):
        env = self.spawned("error")
        with self.assertRaisesRegex(RuntimeError, "synthetic step failure"):
            env.step("raise")
        self.assert_closed(env)
        self.assertEqual([row["event"] for row in self.events("error")],
                         ["construct", "seed", "step", "close"])

    def test_constructor_error_does_not_leave_a_child_process(self):
        previous_children = {child.pid for child in multiprocessing.active_children()}
        kwargs = self.arguments("constructor_error")
        kwargs["fail_constructor"] = True
        with self.assertRaisesRegex(RuntimeError, "synthetic constructor failure"):
            self.adapter.ProcessEnv(kwargs, seed=42, timeout=10.0)
        remaining_children = {child.pid for child in multiprocessing.active_children()}
        self.assertFalse(remaining_children - previous_children)
        self.assertEqual([row["event"] for row in self.events("constructor_error")],
                         ["construct"])

    def test_abrupt_child_exit_is_reported_and_reaped(self):
        env = self.spawned("exit")
        with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
            env.step("exit")
        self.assert_closed(env)
        self.assertEqual(env._process.exitcode, 17)


if __name__ == "__main__":
    unittest.main()
