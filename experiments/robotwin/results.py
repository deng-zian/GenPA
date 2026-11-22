import json
import math
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def audit_attempt(path, phase, trials):
    path = Path(path)
    records = [json.loads(line) for line in (path / "episode_end_states.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (path / "policy_metrics.jsonl").read_text().splitlines()]
    if len(records) != trials or len(metrics) != trials:
        raise ValueError("Incomplete episode or policy-metric coverage")
    if [x["episode_idx"] for x in records] != list(range(trials)):
        raise ValueError("Episode IDs are missing or duplicated")
    if len({x["seed"] for x in records}) != trials:
        raise ValueError("Duplicate environment seed")
    for row, metric in zip(records, metrics):
        if type(row["success"]) is not bool or not row.get("instruction"):
            raise ValueError("Missing success flag or instruction")
        if any(row[key] != metric[key] for key in ("episode_idx", "seed", "success")):
            raise ValueError("Policy metrics do not align with official results")
        if metric["qwen_calls"] != 1 or metric["distinct_instructions"] != 1:
            raise ValueError("Expected one online Qwen encoding per episode")
    success = sum(row["success"] for row in records)
    rate = float((path / f"_result_{phase}.txt").read_text().strip().splitlines()[-1])
    if not math.isclose(rate, success / trials, abs_tol=1e-10):
        raise ValueError("Official rate and episode records disagree")
    return {"episodes": trials, "successes": success, "success_rate": success / trials}


def summarize(jobs, statuses, trials):
    complete = [job for job in jobs if statuses.get(job[0] + "/" + job[1], {}).get("status") == "complete"]
    phases = {}
    for phase in sorted({p for _, p in jobs}):
        rows = [statuses[t + "/" + p] for t, p in complete if p == phase]
        successes = sum(row["successes"] for row in rows)
        episodes = sum(row["episodes"] for row in rows)
        phases[phase] = {"completed_tasks": len(rows), "episodes": episodes,
                         "successes": successes, "success_rate": successes / episodes if episodes else None}
    finished = len(complete) == len(jobs)
    return {"status": "complete" if finished else "partial", "trials_per_task_phase": trials,
            "complete_jobs": len(complete), "expected_jobs": len(jobs), "phases": phases,
            "overall": sum(x["success_rate"] for x in phases.values()) / len(phases) if finished else None,
            "jobs": statuses}
