"""CPU-only checks for relocated configuration, installation and launcher paths."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HAS_CONFIG_DEPS = all(importlib.util.find_spec(name) for name in ('hydra', 'yaml'))


@unittest.skipUnless(HAS_CONFIG_DEPS, 'Install the package with pip install -e . for configuration checks')
class RepositoryLayoutTests(unittest.TestCase):
    def test_hydra_training_and_evaluation_search_paths(self):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(ROOT/'configs/libero'), version_base='1.3'):
            depth = compose(config_name='depth')
            action = compose(config_name='action_only')
            evaluation = compose(config_name='eval_libero', overrides=['task=libero_genpa'])
        self.assertEqual(depth.model._target_, 'genpa.models.backbones.genpa.create_genpa')
        self.assertEqual(action.model.lambda_visual, 0.0)
        self.assertEqual(depth.max_steps, 36750)
        self.assertEqual(evaluation.seed, 42)
        self.assertEqual(evaluation.EVALUATION.num_trials, 50)
        self.assertEqual(evaluation.EVALUATION.replan_steps, 12)
        self.assertEqual(evaluation.model._target_, depth.model._target_)

    def test_accelerate_config_resolves_deepspeed_from_root(self):
        import yaml
        for benchmark in ('libero', 'robotwin'):
            config = yaml.safe_load((ROOT/f'scripts/{benchmark}/accelerate_configs/accelerate_zero1_ds.yaml').read_text())
            path = ROOT/config['deepspeed_config']['deepspeed_config_file']
            self.assertEqual(json.loads(path.read_text())['zero_optimization']['stage'], 1)

    def test_editable_package_import_outside_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            result = subprocess.run([sys.executable, '-c', 'import genpa; print(genpa.__file__)'],
                                    cwd=temp, env=env, text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()).resolve(), (ROOT/'src/genpa/__init__.py').resolve())

    def test_summaries_work_outside_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ('summarize_standard.py', 'summarize_plus.py'):
                result = subprocess.run([sys.executable, str(ROOT/'scripts/libero'/name), '--help'],
                                        cwd=temp, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--results', result.stdout)

    def test_robotwin_dry_run_never_loads_policy_or_starts_workers(self):
        import yaml
        with tempfile.TemporaryDirectory(prefix='genpa relocated ') as temp:
            root = Path(temp)
            tasks = root/'benchmark/task_config'
            tasks.mkdir(parents=True)
            (tasks/'_eval_step_limit.yml').write_text('adjust_bottle: 100\n')
            checkpoint = root/'checkpoint.pt'
            checkpoint.write_bytes(b'CPU fixture, not loadable model weights')
            config = dict(robotwin_root=str(tasks.parent), checkpoint=str(checkpoint),
                          train_config=str(ROOT/'configs/robotwin/depth.yaml'),
                          stats=str(ROOT/'configs/robotwin/absolute_stats.json'))
            config_path = root/'eval.yaml'
            config_path.write_text(yaml.safe_dump(config))
            output = root/'results'
            result = subprocess.run([sys.executable, '-m', 'experiments.robotwin.run',
                                     '--config', str(config_path), '--output', str(output),
                                     '--tasks', 'adjust_bottle', '--gpus', '2,5', '--dry-run'],
                                    cwd=ROOT, text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout.splitlines()[0])
            self.assertEqual(plan['episodes'], 200)
            self.assertEqual(plan['gpus'], [2, 5])
            self.assertEqual(plan['workers_per_gpu'], 1)
            self.assertFalse(output.exists())

    def test_robotwin_vendor_copy_preserves_upstream_and_input_configs(self):
        import importlib
        sys.path.insert(0, str(ROOT))
        try:
            runtime = importlib.import_module('experiments.robotwin.run')
        finally:
            sys.path.remove(str(ROOT))
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'source'
            (source/'task_config').mkdir(parents=True)
            for name in ('demo_clean', 'demo_randomized'):
                (source/'task_config'/f'{name}.yml').write_text('eval_video_log: true\nother: 42\n')
            destination = Path(temp)/'runtime'
            runtime.prepare_runtime(source, destination, False)
            provenance = json.loads((ROOT/'third_party/robotwin/provenance.json').read_text())
            self.assertEqual(runtime.sha256(destination/'script/eval_policy.py'), provenance['sha256'])
            self.assertIn('eval_video_log: true', (source/'task_config/demo_clean.yml').read_text())
            self.assertIn('eval_video_log: false', (destination/'task_config/demo_clean.yml').read_text())

    def test_shell_entry_paths_without_launching_training_or_inference(self):
        with tempfile.TemporaryDirectory(prefix='genpa shell ') as temp:
            root = Path(temp)
            capture = root/'capture.json'
            stub = root/'capture.py'
            stub.write_text('import json,os,sys\nfrom pathlib import Path\n'
                            'Path(os.environ["CAPTURE"]).write_text(json.dumps(dict(argv=sys.argv[1:],cwd=os.getcwd(),'
                            'pythonpath=os.environ["PYTHONPATH"],flux=os.environ["GENPA_FLUX2_SRC"])))\n')
            for command in ('python', 'accelerate'):
                path = root/command
                path.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(stub))+' "$@"\n')
                path.chmod(0o755)
            env = os.environ.copy()
            env.update(PATH=str(root)+os.pathsep+env['PATH'], CAPTURE=str(capture))
            for key in ('OUTPUT_DIR', 'LIBERO_DATA', 'LIBERO_DEPTH_LABELS', 'LIBERO_QWEN_CACHE',
                        'ROBOTWIN_DATA', 'ROBOTWIN_NUMERIC', 'ROBOTWIN_LANGUAGE',
                        'ROBOTWIN_QWEN_0', 'ROBOTWIN_QWEN_1', 'ROBOTWIN_QWEN_2',
                        'FLUX2_WEIGHTS', 'FLUX2_AE', 'QWEN3', 'ROBOTWIN_BENCHMARK',
                        'ROBOTWIN_CKPT', 'EVAL_OUTPUT', 'CUROBO_SRC'):
                env['GENPA_'+key] = str(root/'placeholder')
            env.pop('GENPA_FLUX2_SRC', None)
            env.pop('GENPA_ACCELERATE_CONFIG', None)
            env.pop('GENPA_GPUS', None)
            for bench, entry in [('libero', 'train.sh'), ('robotwin', 'train.sh'), ('robotwin', 'evaluate.sh')]:
                result = subprocess.run(['bash', str(ROOT/f'scripts/{bench}/{entry}')], cwd=temp,
                                        env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(capture.read_text())
                self.assertEqual(Path(record['cwd']).resolve(), ROOT.resolve())
                self.assertEqual(Path(record['flux']).resolve(), (ROOT/'third_party/flux2').resolve())
                paths = record['pythonpath'].split(os.pathsep)
                self.assertIn(str(ROOT/'src'), paths)
                self.assertNotIn(str(ROOT/bench/'src'), paths)
                if entry == 'train.sh':
                    args = record['argv']
                    self.assertTrue((ROOT/args[args.index('--config_file')+1]).is_file())
                    self.assertEqual(args[args.index('--num_processes')+1], '8')
                    self.assertIn(f'scripts/{bench}/train.py', args)
                else:
                    self.assertEqual(record['argv'][:2], ['-m', 'experiments.robotwin.run'])
                    self.assertIn('configs/robotwin/eval.yaml', record['argv'])


if __name__ == '__main__':
    unittest.main()
