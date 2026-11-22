"""Exercise actual subprocesses, task coverage, failure exit and cancellation."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT/'experiments/libero/run_libero_workers.py'
FIXTURE = ROOT/'tests/fixtures/fake_eval_worker.py'


class EvaluationLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='genpa launcher ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        entry = self.root/'experiments/libero/eval_entry.py'
        entry.parent.mkdir(parents=True)
        shutil.copyfile(FIXTURE, entry)
        self.tasks = self.root/'input tasks.txt'
        self.tasks.write_text(''.join(f'libero_spatial,{i}\n' for i in range(8)))
        self.output = self.root/'results'
        self.env = os.environ.copy()
        self.env.update(ROOT_DIR=str(self.root), OUTPUT_DIR=str(self.output),
                        CUDA_VISIBLE_DEVICES='0', MAX_TASKS_PER_GPU='1', NUM_TRIALS='1',
                        TASK_CHUNK_SIZE='1', CONFIG='libero_genpa', CONFIG_NAME='eval_libero',
                        CKPT=str(self.root/'checkpoint with spaces.pt'), GENPA_EVAL_BENCHMARK='libero',
                        GENPA_LIBERO_SIM_PROCESS='0', WORLD_SIZE='32', MASTER_PORT='12345',
                        EXTRA_ARGS="'++EVALUATION.sentinel=value with spaces'")
        for key in ('GENPA_LIBERO_RENDER_GPU', 'GENPA_LIBERO_RENDER_GPU_MAP'):
            self.env.pop(key, None)

    def invoke(self):
        return subprocess.run([sys.executable, str(RUNNER), str(self.tasks)], env=self.env,
                              text=True, capture_output=True, timeout=25)

    def test_one_two_four_eight_gpu_assignments(self):
        for count in (1, 2, 4, 8):
            with self.subTest(gpus=count):
                self.output = self.root/f'results_{count}'
                self.env.update(OUTPUT_DIR=str(self.output), CUDA_VISIBLE_DEVICES=','.join(map(str, range(count))))
                result = self.invoke()
                self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
                summary = json.loads((self.output/'summary.json').read_text())
                self.assertTrue(summary['complete'])
                self.assertFalse(summary['formal'])
                records = [json.loads(p.read_text()) for p in (self.output/'events').glob('*.finished.json')]
                self.assertEqual(len(records), 8)
                self.assertEqual({r['gpu'] for r in records}, set(range(count)))
                for r in records:
                    self.assertEqual(r['cuda'], str(r['gpu']))
                    self.assertEqual(r['egl'], str(r['gpu']))
                    self.assertIsNone(r['world_size'])
                    self.assertIsNone(r['master_port'])
                    self.assertEqual(r['sentinel'], 'value with spaces')

    def test_per_gpu_limit_sparse_ids_and_chunk_reuse(self):
        self.env.update(CUDA_VISIBLE_DEVICES='2,5', MAX_TASKS_PER_GPU='2', TASK_CHUNK_SIZE='2', FAKE_SLEEP='0.2')
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
        records = [json.loads(p.read_text()) for p in (self.output/'events').glob('*.finished.json')]
        self.assertEqual(len(records), 4)
        self.assertEqual(len(list(self.output.rglob('gpu*_task*_results.json'))), 8)
        for gpu in (2, 5):
            events = sorted((r[k], delta) for r in records if r['gpu']==gpu
                            for k, delta in [('start', 1), ('end', -1)])
            active = maximum = 0
            for _, delta in events:
                active += delta
                maximum = max(maximum, active)
            self.assertLessEqual(maximum, 2)
            self.assertEqual(active, 0)

    def test_failure_does_not_report_complete(self):
        self.env.update(FAKE_FAIL_TASK='libero_spatial,3', CUDA_VISIBLE_DEVICES='0,1')
        result = self.invoke()
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.output/'summary.json').exists())
        failures = (self.output/'failed_tasks.txt').read_text().splitlines()
        self.assertEqual(len(failures), 1)
        self.assertEqual(json.loads(failures[0])['exit_code'], 23)

    def test_existing_results_are_not_overwritten(self):
        self.assertEqual(self.invoke().returncode, 0)
        before = (self.output/'parallel_launch.json').read_bytes()
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already contains a run', result.stderr)
        self.assertEqual((self.output/'parallel_launch.json').read_bytes(), before)

    def test_invalid_task_is_rejected_before_launch(self):
        self.tasks.write_text('libero_spatial,10\n')
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unknown benchmark tasks', result.stderr)
        self.assertFalse((self.output/'parallel_launch.json').exists())

    def test_progress_updates_while_last_worker_is_still_running(self):
        self.tasks.write_text('libero_spatial,0\nlibero_spatial,1\n')
        self.env.update(CUDA_VISIBLE_DEVICES='0,1', FAKE_SLOW_CHUNK='chunk_000001', FAKE_SLOW_SECONDS='5')
        with (self.root/'controller.log').open('w') as log:
            manager = subprocess.Popen([sys.executable,str(RUNNER),str(self.tasks)], env=self.env,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic()+4
                observed = False
                while time.monotonic()<deadline and manager.poll() is None:
                    manifest = self.output/'parallel_launch.json'
                    if manifest.exists():
                        rows = json.loads(manifest.read_text())['launches']
                        if len(rows)==2 and rows[0].get('exit_code')==0 and 'exit_code' not in rows[1]:
                            observed = True
                            break
                    time.sleep(0.03)
                self.assertTrue(observed, 'Completed worker was not published until the entire run ended')
                self.assertEqual(manager.wait(timeout=8),0)
            finally:
                if manager.poll() is None:
                    manager.send_signal(signal.SIGTERM)
                    manager.wait(timeout=8)

    def test_sigterm_reaps_owned_workers(self):
        self.env.update(FAKE_SLEEP='30')
        with (self.root/'controller.log').open('w') as log:
            manager = subprocess.Popen([sys.executable, str(RUNNER), str(self.tasks)], env=self.env,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic()+10
                records = []
                while not records and time.monotonic()<deadline:
                    records = list((self.output/'events').glob('*.started.json'))
                    time.sleep(0.02)
                self.assertTrue(records, 'Child never started')
                pid = json.loads(records[0].read_text())['pid']
                manager.send_signal(signal.SIGTERM)
                self.assertEqual(manager.wait(timeout=8), 130)
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            finally:
                if manager.poll() is None:
                    manager.kill()
                    manager.wait()

    def test_help_does_not_require_assets(self):
        result = subprocess.run(['bash', str(ROOT/'scripts/libero/evaluate.sh'), '--help'],
                                text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--workers-per-gpu', result.stdout)

    def test_public_entry_defaults_and_overrides(self):
        # Exercise the public shell entry point while capturing its Python launch.
        binary = self.root/'bin'
        binary.mkdir()
        stub = binary/'python'
        stub.write_text(f'#!{sys.executable}\nimport json,os,sys\njson.dump(sys.argv[1:],open(os.environ["ARG_CAPTURE"],"w"))\n')
        stub.chmod(0o755)
        capture = self.root/'arguments.json'
        env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ['PATH'], ARG_CAPTURE=str(capture))
        for key in ('GENPA_LIBERO_CKPT','GENPA_EVAL_OUTPUT','GENPA_LIBERO_BENCHMARK',
                    'GENPA_LIBERO_QWEN_CACHE','GENPA_FLUX2_SRC','GENPA_FLUX2_WEIGHTS',
                    'GENPA_FLUX2_AE','GENPA_QWEN3','GENPA_CLEAN_INSTRUCTION_MANIFEST'):
            env[key] = str(self.root/'asset with spaces')
        for mode, chunk, trials in [('standard',1,50),('plus',20,1),('plus_released',20,1)]:
            with self.subTest(mode=mode):
                for flags, expected_chunk, expected_trials in [([],chunk,trials), (['--chunk-size','3','--trials','2'],3,2)]:
                    result = subprocess.run(['bash',str(ROOT/'scripts/libero/evaluate.sh'),mode,*flags],
                                            env=env, text=True, capture_output=True, timeout=5)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    options = dict(arg.split('=',1) for arg in json.loads(capture.read_text()) if '=' in arg)
                    self.assertEqual(options['MULTIRUN.chunk_size'], str(expected_chunk))
                    self.assertEqual(options['EVALUATION.num_trials'], str(expected_trials))
                    self.assertEqual(options['ckpt'], env['GENPA_LIBERO_CKPT'])


if __name__ == '__main__':
    unittest.main()
