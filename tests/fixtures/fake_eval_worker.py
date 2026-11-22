"""CPU-only child process for testing GPU assignment and scheduler lifecycle."""
import json
import os
from pathlib import Path
import sys
import time

options = dict(arg.lstrip('+').split('=', 1) for arg in sys.argv[1:] if '=' in arg)
chunk = Path(options['EVALUATION.task_chunk_file'])
output = Path(options['EVALUATION.output_dir'])
gpu = int(options['gpu_id'])
trials = int(options['EVALUATION.num_trials'])
events = output / 'events'
events.mkdir(exist_ok=True)
event = dict(pid=os.getpid(), gpu=gpu, cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),
             egl=os.environ.get('MUJOCO_EGL_DEVICE_ID'), world_size=os.environ.get('WORLD_SIZE'),
             master_port=os.environ.get('MASTER_PORT'),
             sentinel=options.get('EVALUATION.sentinel'), start=time.monotonic())
(events / (chunk.stem+'.started.json')).write_text(json.dumps(event))
delay = os.environ.get('FAKE_SLOW_SECONDS','5') if chunk.stem==os.environ.get('FAKE_SLOW_CHUNK') else os.environ.get('FAKE_SLEEP','0.08')
time.sleep(float(delay))
if os.environ.get('FAKE_FAIL_TASK') in chunk.read_text().splitlines():
    sys.exit(23)
for line in chunk.read_text().splitlines():
    suite, task = line.split(',')
    directory = output / suite
    directory.mkdir(exist_ok=True)
    payload = dict(benchmark='libero', task_suite=suite, task_id=int(task), gpu_id=gpu,
                   total_episodes=trials, successes=trials, success_episodes=list(range(trials)),
                   failure_episodes=[], evaluation_seed=42, checkpoint_sha256='a'*64,
                   instruction_protocol='official', task_description='fixture', duration=0.01)
    (directory/f'gpu{gpu}_task{task}_results.json').write_text(json.dumps(payload))
event['end'] = time.monotonic()
(events / (chunk.stem+'.finished.json')).write_text(json.dumps(event))
