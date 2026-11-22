"""Task-level scheduling only. Scene/expert/policy scoring stays upstream."""

import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import yaml

from .results import audit_attempt, summarize, write_json
from .runtime import WorkerWatchdog, compiler_environment

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_runtime(source, destination, video):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name in ("script", "task_config", "__pycache__", ".git"):
            continue
        link = destination / child.name
        if not link.exists():
            link.symlink_to(child.resolve(), target_is_directory=child.is_dir())
    scripts = destination / "script"
    scripts.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "third_party/robotwin/eval_policy.py", scripts / "eval_policy.py")
    config = destination / "task_config"
    shutil.copytree(source / "task_config", config, dirs_exist_ok=True)
    for name in ("demo_clean", "demo_randomized"):
        path = config / f"{name}.yml"
        data = yaml.safe_load(path.read_text())
        data["eval_video_log"] = bool(video)
        path.write_text(yaml.safe_dump(data, sort_keys=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--phases", default="clean,random")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--stall-seconds", type=int, default=3600)
    return parser.parse_args()


def main():
    args = parse_args()
    # Resolve reviewer-supplied asset paths without embedding site-specific locations.
    config_text = os.path.expandvars(Path(args.config).read_text())
    if "$GENPA_" in config_text:
        raise ValueError("Unresolved GENPA_* path in evaluation config")
    config = yaml.safe_load(config_text)
    gpus = [int(g) for g in args.gpus.split(",")]
    if args.trials < 1 or args.workers_per_gpu < 1 or len(set(gpus)) != len(gpus) or args.stall_seconds <= 0:
        raise ValueError("Invalid trials, workers or GPU IDs")
    phases = args.phases.split(",")
    if len(set(phases)) != len(phases) or not set(phases) <= {"clean", "random"}:
        raise ValueError("Invalid phases")
    task_map = yaml.safe_load((Path(config["robotwin_root"]) / "task_config/_eval_step_limit.yml").read_text())
    tasks = list(task_map) if args.tasks == "all" else args.tasks.split(",")
    if len(set(tasks)) != len(tasks) or not set(tasks) <= set(task_map):
        raise ValueError("Unknown or duplicate task")
    if args.tasks == "all" and len(tasks) != 50:
        raise ValueError(f"Expected 50 tasks, found {len(tasks)}")
    for name in ("checkpoint", "train_config", "stats"):
        if not Path(config[name]).is_file():
            raise FileNotFoundError(config[name])
    source = json.loads((ROOT / "third_party/robotwin/provenance.json").read_text())
    if sha256(ROOT / "third_party/robotwin/eval_policy.py") != source["sha256"]:
        raise ValueError("Pinned upstream evaluator has changed")
    ckpt = Path(config["checkpoint"]).stat()
    contract = {"config": config, "trials": args.trials, "seed": args.seed,
                "tasks": tasks, "phases": phases, "video": args.video,
                "upstream": source, "checkpoint_size": ckpt.st_size,
                "checkpoint_mtime_ns": ckpt.st_mtime_ns,
                "train_config_sha256": sha256(config["train_config"]), "stats_sha256": sha256(config["stats"]),
                "pipeline_sources": {str(p.relative_to(ROOT)): sha256(p)
                                     for folder in ("experiments/robotwin", "experiments/robotwin/policy")
                                     for p in sorted((ROOT / folder).glob("*.py"))}}
    jobs = [(task, phase) for phase in phases for task in tasks]
    print(json.dumps({"jobs": len(jobs), "episodes": len(jobs) * args.trials,
                      "gpus": gpus, "workers_per_gpu": args.workers_per_gpu,
                      "checkpoint": config["checkpoint"], "online_qwen": True,
                      "upstream_commit": source["commit"]}), flush=True)
    output = Path(args.output).resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        lock = (output / ".manager.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract_path = output / "protocol.json"
    upgrade = False
    if contract_path.exists():
        if not args.resume:
            raise ValueError("Existing run needs --resume")
        previous = json.loads(contract_path.read_text())
        if previous != contract:
            raise ValueError("Resume requires the same configuration and source; use a new output directory")
    else:
        if args.resume:
            raise ValueError("No run to resume")
    statuses = {}
    pending = collections.deque()
    for task, phase in jobs:
        status_path = output / task / phase / "status.json"
        old = json.loads(status_path.read_text()) if status_path.exists() else {}
        if old.get("status") == "complete":
            audited = audit_attempt(old["attempt_dir"], phase, args.trials)
            if any(old[k] != v for k, v in audited.items()):
                raise ValueError(f"Stored status disagrees with episode records: {task}/{phase}")
            statuses[task + "/" + phase] = old
        else:
            pending.append((task, phase))
    print(json.dumps({"audited_complete_jobs": len(statuses), "pending_jobs": len(pending),
                      "runtime_upgrade": upgrade, "dry_run": args.dry_run}), flush=True)
    if args.dry_run:
        return 0
    write_json(contract_path, contract)
    runtime = output / "runtime"
    prepare_runtime(config["robotwin_root"], runtime, args.video)
    cache_root = Path(tempfile.mkdtemp(prefix="genpa_online_compile_"))
    slots = [(gpu, index) for gpu in gpus for index in range(args.workers_per_gpu)]
    active = {}
    watchdogs = {}
    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def record(key, value):
        statuses[key] = value
        write_json(output / key / "status.json", value)
        write_json(output / "summary.json", summarize(jobs, statuses, args.trials))

    def launch(slot, task, phase):
        key = task + "/" + phase
        parent = output / key
        parent.mkdir(parents=True, exist_ok=True)
        attempt = parent / f"attempt_{len(list(parent.glob('attempt_*'))) + 1:03d}"
        attempt.mkdir()
        eval_args = {"task_name": task, "task_config": "demo_clean" if phase == "clean" else "demo_randomized",
                     "ckpt_setting": config["checkpoint"], "train_config": config["train_config"],
                     "dataset_stats_path": config["stats"], "model_overrides": config.get("model_overrides", {}),
                     "policy_name": "experiments.robotwin.policy", "instruction_type": "unseen", "seed": args.seed,
                     "eval_num_episodes": args.trials, "eval_output_dir": str(attempt),
                     "skip_get_obs_within_replan": False}
        cache_env = compiler_environment(cache_root, slot)
        job = {"runtime_root": str(runtime), "cpu_threads": args.cpu_threads,
               "eval_args": eval_args, "compiler_environment": cache_env}
        write_json(attempt / "job.json", job)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(slot[0])
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *config.get("python_paths", []), env.get("PYTHONPATH", "")])
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = str(args.cpu_threads)
        env.update(cache_env)
        log = (attempt / "worker.log").open("w")
        process = subprocess.Popen([sys.executable, "-m", "experiments.robotwin.worker", str(attempt / "job.json")],
                                   cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[slot] = (process, log, key, phase, attempt, time.time())
        watchdogs[slot] = WorkerWatchdog(attempt, active[slot][-1])
        record(key, {"status": "running", "attempt_dir": str(attempt), "gpu": slot[0], "pid": process.pid})
        print(f"[start] {key} gpu={slot[0]} pid={process.pid}", flush=True)

    try:
        while pending or active:
            if stopping:
                break
            for slot in slots:
                if pending and slot not in active:
                    launch(slot, *pending.popleft())
            for slot, (process, log, key, phase, attempt, started) in list(active.items()):
                rc = process.poll()
                failure = watchdogs[slot].fatal_error()
                if rc is None:
                    if not failure and watchdogs[slot].stalled(time.time(), args.stall_seconds):
                        failure = f"No completed episode for {args.stall_seconds} seconds"
                    if failure:
                        print(f"[abort] {key}: {failure}", flush=True)
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        rc = process.returncode
                    else:
                        continue
                log.close()
                status = {"status": "failed", "attempt_dir": str(attempt), "returncode": rc,
                          "elapsed_s": time.time() - started}
                try:
                    if failure:
                        raise RuntimeError(failure)
                    if rc != 0:
                        raise RuntimeError(f"Worker exit code {rc}; see worker.log")
                    status.update(audit_attempt(attempt, phase, args.trials), status="complete")
                except Exception as exc:
                    status["error"] = str(exc)
                record(key, status)
                print(f"[{status['status']}] {key} {status}", flush=True)
                del active[slot]
                del watchdogs[slot]
            time.sleep(2)
    finally:
        for process, log, key, _, attempt, started in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log, key, _, attempt, started in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
            record(key, {"status": "interrupted", "attempt_dir": str(attempt)})
        shutil.rmtree(cache_root)
    final = summarize(jobs, statuses, args.trials)
    write_json(output / "summary.json", final)
    return 0 if final["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
