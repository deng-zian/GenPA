"""Validate complete task/trial coverage before reporting LIBERO success rates."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PLUS_COUNTS = dict(zip(SUITES, (2402, 2518, 2591, 2519)))
RESULT_NAME = re.compile(r"gpu(\d+)_task(\d+)_results\.json$")


def full_task_set(benchmark: str) -> set[tuple[str, int]]:
    if benchmark not in {"libero", "libero_plus"}:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return {(suite, i) for suite in SUITES
            for i in range(10 if benchmark == "libero" else PLUS_COUNTS[suite])}


def read_task_file(path: Path) -> set[tuple[str, int]]:
    tasks = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"Invalid task record {path}:{number}")
        key = (parts[0], int(parts[1]))
        if key in tasks:
            raise ValueError(f"Duplicate task in manifest: {key}")
        tasks.add(key)
    if not tasks:
        raise ValueError(f"Empty task manifest: {path}")
    return tasks


def validate_trial_record(row: dict, key: tuple[str, int], trials: int) -> None:
    # bool is deliberately not an integer count in this artifact format.
    for field in ("task_id", "total_episodes", "successes"):
        if type(row.get(field)) is not int:
            raise ValueError(f"Non-integer {field} for {key}")
    if row["total_episodes"] != trials or not 0 <= row["successes"] <= trials:
        raise ValueError(f"Invalid trial/success count for {key}: expected {trials}")
    success = row.get("success_episodes")
    failure = row.get("failure_episodes")
    if not isinstance(success, list) or not isinstance(failure, list):
        raise ValueError(f"Missing episode identity lists for {key}")
    all_ids = success + failure
    if any(type(i) is not int for i in all_ids):
        raise ValueError(f"Non-integer episode identity for {key}")
    if len(all_ids) != trials or len(set(all_ids)) != trials or set(all_ids) != set(range(trials)):
        raise ValueError(f"Missing, duplicated, or out-of-range episode identity for {key}")
    if len(success) != row["successes"]:
        raise ValueError(f"Success list/count mismatch for {key}")


def result_identity(row: dict) -> tuple[int | None, str | None]:
    """Read actual evaluator fields and an unambiguous legacy seed alias."""
    if "evaluation_seed" in row and "seed" in row and row["evaluation_seed"] != row["seed"]:
        raise ValueError("Conflicting evaluation_seed and legacy seed fields")
    seed = row.get("evaluation_seed", row.get("seed"))
    if seed is not None and type(seed) is not int:
        raise ValueError("Evaluation seed must be an integer when present")
    checkpoint = row.get("checkpoint_sha256")
    if checkpoint is not None:
        if not isinstance(checkpoint, str) or re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint) is None:
            raise ValueError("Invalid checkpoint_sha256; expected 64 hexadecimal characters")
        checkpoint = checkpoint.lower()
    return seed, checkpoint


def identity_summary(rows: dict) -> dict:
    """Identity means seed/checkpoint consistency, not benchmark correctness."""
    identities = {result_identity(row) for row in rows.values()}
    if len(identities) != 1:
        raise ValueError("Mixed or missing evaluation seeds/checkpoint fingerprints")
    seed, checkpoint = next(iter(identities))
    return {"evaluation_seed": seed, "seed": seed,
            "checkpoint_sha256": checkpoint,
            "identity_verified": seed is not None and checkpoint is not None}


def collect_results(output_dir: Path, *, benchmark: str, expected: set[tuple[str, int]],
                    num_trials: int, allow_partial: bool = False) -> tuple[dict, set]:
    if type(num_trials) is not int or num_trials < 1:
        raise ValueError("num_trials must be a positive integer")
    valid = full_task_set(benchmark)
    if not expected or not expected <= valid:
        raise ValueError(f"Task manifest contains unknown/out-of-range {benchmark} tasks: {expected - valid}")
    rows = {}
    protocols, seeds, checkpoint_ids = set(), set(), set()
    for path in sorted(output_dir.rglob("gpu*_task*_results.json")):
        match = RESULT_NAME.fullmatch(path.name)
        if not match:
            raise ValueError(f"Malformed result filename: {path}")
        row = json.loads(path.read_text())
        if not isinstance(row, dict):
            raise ValueError(f"Result must be an object: {path}")
        key = (row.get("task_suite"), row.get("task_id"))
        if type(key[1]) is not int or key[0] != path.parent.name or key[1] != int(match[2]):
            raise ValueError(f"Result identity disagrees with filename/directory: {path}")
        if key not in expected or key in rows:
            raise ValueError(f"Unknown or duplicated task result: {key}")
        if row.get("benchmark", benchmark) != benchmark:
            raise ValueError(f"Mixed benchmark results at {path}")
        validate_trial_record(row, key, num_trials)
        protocols.add(row.get("instruction_protocol", "official"))
        evaluation_seed, checkpoint = result_identity(row)
        seeds.add(evaluation_seed)
        checkpoint_ids.add(checkpoint)
        rows[key] = row
    if not rows:
        raise ValueError("No result files found")
    if len(protocols) > 1 or len(seeds) > 1 or len(checkpoint_ids) > 1:
        raise ValueError("Mixed instruction protocols, seeds, or checkpoint fingerprints")
    missing = expected - rows.keys()
    if missing and not allow_partial:
        raise ValueError(f"Incomplete evaluation: {len(rows)}/{len(expected)} tasks; "
                         "use --allow-partial for progress only")
    return rows, missing


def summarize_results(output_dir, *, benchmark=None, task_file=None, num_trials=None,
                      allow_partial=False, auto_task_file=True):
    output_dir = Path(output_dir)
    benchmark = benchmark or os.environ.get("GENPA_EVAL_BENCHMARK", "libero")
    full = full_task_set(benchmark)
    if auto_task_file and task_file is None and (output_dir / "tasks.txt").is_file():
        task_file = output_dir / "tasks.txt"
    expected = read_task_file(Path(task_file)) if task_file else full
    num_trials = num_trials if num_trials is not None else int(os.environ.get("NUM_TRIALS", "1" if benchmark == "libero_plus" else "50"))
    rows, missing = collect_results(output_dir, benchmark=benchmark, expected=expected,
                                   num_trials=num_trials, allow_partial=allow_partial)
    suite_stats, task_results = {}, {}
    for (suite, task_id), row in sorted(rows.items()):
        stats = suite_stats.setdefault(suite, dict(total_tasks=0, total_trials=0, total_successes=0, total_time=0.0, max_time=0.0))
        duration = float(row.get("duration", 0))
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f"Invalid duration for {(suite, task_id)}")
        stats["total_tasks"] += 1
        stats["total_trials"] += num_trials
        stats["total_successes"] += row["successes"]
        stats["total_time"] += duration
        stats["max_time"] = max(stats["max_time"], duration)
        task_results[f"{suite}_{task_id}"] = dict(success_rate=100*row["successes"]/num_trials,
             duration=duration, total_episodes=num_trials, successes=row["successes"],
             task_description=row.get("task_description", ""))
    trials = len(rows)*num_trials
    successes = sum(row["successes"] for row in rows.values())
    pooled = 100*successes/trials
    suite_mean = sum(100*s["total_successes"]/s["total_trials"] for s in suite_stats.values())/len(suite_stats)
    full_benchmark = expected == full and num_trials == (1 if benchmark == "libero_plus" else 50)
    payload = dict(run_id=output_dir.name, ckpt=os.environ.get("CKPT", ""), config=os.environ.get("CONFIG", ""),
        complete=not missing, full_benchmark=full_benchmark, formal=full_benchmark and not missing,
        instruction_protocol=next(iter(rows.values())).get("instruction_protocol", "official"),
        **identity_summary(rows),
        expected_tasks=len(expected), completed_tasks=len(rows),
        expected_trials=len(expected)*num_trials, missing_tasks=[f"{s}:{i}" for s,i in sorted(missing)],
        suite_stats=suite_stats, task_results=task_results,
        overall=dict(benchmark=benchmark, aggregation="instance_pooled" if benchmark == "libero_plus" else "suite_mean",
            total_trials=trials, total_successes=successes,
            average_success_rate=pooled if benchmark == "libero_plus" else suite_mean,
            pooled_success_rate=pooled, total_time=sum(s["total_time"] for s in suite_stats.values()),
            average_task_time=sum(s["total_time"] for s in suite_stats.values())/len(rows)))
    temporary = output_dir / ".summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2)+"\n")
    temporary.replace(output_dir / "summary.json")
    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer=csv.writer(f); writer.writerow(["suite","successes","trials","success_rate_percent"])
        for suite, s in suite_stats.items():
            writer.writerow([suite,s["total_successes"],s["total_trials"],100*s["total_successes"]/s["total_trials"]])
        writer.writerow(["Overall",successes,trials,payload["overall"]["average_success_rate"]])
    with (output_dir / "task_success_rates.csv").open("w", newline="") as f:
        writer=csv.writer(f); writer.writerow(["Task","Description","Success Rate (%)"])
        for key, row in task_results.items():
            writer.writerow([key, row["task_description"], row["success_rate"]])
    print(json.dumps({k:v for k,v in payload.items() if k not in {"suite_stats","task_results"}}, indent=2))
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", "--results", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("libero","libero_plus"))
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    args=parser.parse_args()
    summarize_results(args.output_dir, benchmark=args.benchmark, task_file=args.task_file,
                      num_trials=args.num_trials, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
