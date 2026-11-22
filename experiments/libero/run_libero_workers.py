"""Run independent, same-GPU policy/simulator workers; no distributed training."""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from summarize_results import full_task_set, read_task_file, summarize_results

DISTRIBUTED_ENV = {
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_ERROR_FILE", "MPI_LOCALRANKID", "PMI_RANK", "PMI_SIZE", "PMIX_RANK",
    "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_RANK", "MV2_COMM_WORLD_SIZE",
    "MV2_COMM_WORLD_LOCAL_RANK",
}


def positive(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def gpu_ids() -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    values = visible.split(",") if visible else list(range(positive("NUM_GPUS", 1)))
    ids = [int(str(value).strip()) for value in values]
    if not ids or any(gpu < 0 for gpu in ids) or len(set(ids)) != len(ids):
        raise ValueError("GPU IDs must be distinct nonnegative physical indices")
    return ids


def ensure_fresh_output(output: Path) -> None:
    if (output / "parallel_launch.json").exists() or any(output.rglob("gpu*_task*_results.json")):
        raise ValueError(f"Evaluation output already contains a run: {output}. Use a new directory.")


def write_manifest(output: Path, manifest: dict) -> None:
    temporary = output / ".parallel_launch.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output / "parallel_launch.json")


def worker_environment(gpu: int) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in DISTRIBUTED_ENV}
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["PYOPENGL_PLATFORM"] = env.get("PYOPENGL_PLATFORM", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = str(positive("WORKER_OMP_NUM_THREADS", 1))
    env["MKL_NUM_THREADS"] = str(positive("WORKER_MKL_NUM_THREADS", int(env["OMP_NUM_THREADS"])))
    env["OPENBLAS_NUM_THREADS"] = str(positive("WORKER_OPENBLAS_NUM_THREADS", 1))
    if env.get("WORKER_PYTHONPATH"):
        env["PYTHONPATH"] = env["WORKER_PYTHONPATH"]
    if env.get("GENPA_LIBERO_SIM_PROCESS", "0") == "0":
        if env.get("GENPA_LIBERO_RENDER_GPU") or env.get("GENPA_LIBERO_RENDER_GPU_MAP"):
            raise ValueError("Rendering GPU overrides require GENPA_LIBERO_SIM_PROCESS=1")
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    else:
        # ProcessEnv assigns the child renderer independently.
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
    return env


def stop_workers(active: list[dict]) -> None:
    """Signal only process groups created by this invocation, never saved PIDs."""
    for item in active:
        try:
            os.killpg(item["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for item in active:
        process = item["process"]
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        item["log_handle"].close()


def run(task_file: Path) -> int:
    root = Path(os.environ.get("ROOT_DIR", Path(__file__).resolve().parents[2])).resolve()
    output = Path(os.environ["OUTPUT_DIR"]).resolve()
    ensure_fresh_output(output)
    ids = gpu_ids()
    workers = positive("MAX_TASKS_PER_GPU", 1)
    trials = positive("NUM_TRIALS", 50)
    chunk_size = positive("TASK_CHUNK_SIZE", 1)
    selected = read_task_file(task_file)
    unknown = selected - full_task_set(os.environ.get("GENPA_EVAL_BENCHMARK", "libero"))
    if unknown:
        raise ValueError(f"Unknown benchmark tasks: {sorted(unknown)}")
    tasks = [line.strip() for line in task_file.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    # GPU assignment changes concurrency only; task order and trial settings stay intact.
    output.mkdir(parents=True, exist_ok=True)
    canonical_tasks = output / "tasks.txt"
    canonical_tasks.write_text("\n".join(tasks) + "\n")
    for directory in ("task_chunks", "task_logs", "task_status"):
        (output / directory).mkdir(exist_ok=True)
    pending = deque()
    for index, start in enumerate(range(0, len(tasks), chunk_size)):
        name = f"chunk_{index:06d}"
        path = output / "task_chunks" / f"{name}.txt"
        path.write_text("\n".join(tasks[start:start + chunk_size]) + "\n")
        suite, task = tasks[start].split(",")
        pending.append((name, path, suite.strip(), int(task)))
    available = deque((gpu, slot) for slot in range(workers) for gpu in ids)
    environments = {gpu: worker_environment(gpu) for gpu in ids}
    extra = shlex.split(os.environ.get("EXTRA_ARGS", ""))
    active, failures, launches = [], [], []
    started = time.monotonic()
    manifest = {"gpus": ids, "workers_per_gpu": workers, "chunk_size": chunk_size,
                "num_tasks": len(tasks), "num_trials": trials,
                "simulator_mode": "separate_process" if os.environ.get("GENPA_LIBERO_SIM_PROCESS") == "1" else "same_process_same_gpu",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "launches": launches}
    write_manifest(output, manifest)
    (output / "failed_tasks.txt").write_text("")
    print(f"Evaluation: GPUs={ids}, workers/GPU={workers}, tasks={len(tasks)}, chunk={chunk_size}", flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    old_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        while pending or active:
            while pending and available:
                gpu, slot = available.popleft()
                name, chunk, suite, task = pending.popleft()
                command = [sys.executable, "experiments/libero/eval_entry.py",
                           "--config-name", os.environ.get("CONFIG_NAME", "eval_libero"),
                           f"task={os.environ['CONFIG']}", f"ckpt={os.environ['CKPT']}",
                           f"EVALUATION.task_chunk_file={chunk}",
                           f"EVALUATION.task_suite_name={suite}", f"EVALUATION.task_id={task}",
                           f"gpu_id={gpu}", f"EVALUATION.num_trials={trials}",
                           f"EVALUATION.output_dir={output}", *extra]
                log_path = output / "task_logs" / f"{name}_{suite}_task{task}_gpu{gpu}.log"
                log = log_path.open("w")
                try:
                    process = subprocess.Popen(command, cwd=root, env=environments[gpu],
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                record = dict(chunk=name, gpu=gpu, slot=slot, pid=process.pid,
                              log=str(log_path), started_seconds=time.monotonic()-started)
                launches.append(record)
                active.append(dict(process=process, log_handle=log, record=record))
                write_manifest(output, manifest)
                print(f"Start {name}: GPU {gpu}, worker {slot}, PID {process.pid}", flush=True)
            completed = False
            for item in active[:]:
                code = item["process"].poll()
                if code is None:
                    continue
                record = item["record"]
                completed = True
                record.update(exit_code=code, finished_seconds=time.monotonic()-started)
                item["log_handle"].close()
                active.remove(item)
                available.append((record["gpu"], record["slot"]))
                status = "SUCCESS" if code == 0 else "FAILED"
                (output / "task_status" / f"{record['chunk']}.status").write_text(
                    f"{status}|{record['gpu']}|{code}|{int(time.time())}|{record['log']}\n")
                if code:
                    failures.append(record)
                    with (output / "failed_tasks.txt").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                print(f"{status} {record['chunk']}: GPU {record['gpu']}, exit={code}", flush=True)
            if completed:
                write_manifest(output, manifest)
            if active:
                time.sleep(0.2)
        manifest.update(wall_seconds=time.monotonic()-started, failed_chunks=len(failures))
        write_manifest(output, manifest)
        if failures:
            print(f"Incomplete evaluation: {len(failures)} failed chunks; see failed_tasks.txt", file=sys.stderr)
            return 1
        summarize_results(output, benchmark=os.environ.get("GENPA_EVAL_BENCHMARK", "libero"),
                          task_file=canonical_tasks, num_trials=trials)
        return 0
    except BaseException:
        stop_workers(active)
        manifest.update(interrupted=True, wall_seconds=time.monotonic()-started)
        write_manifest(output, manifest)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_file", type=Path)
    try:
        sys.exit(run(parser.parse_args().task_file))
    except KeyboardInterrupt:
        sys.exit(130)
