"""Build absolute joint statistics from the existing immutable H16 training index."""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

from genpa.robotwin.depth.contract import ABS_STATS, GRIPPERS, HORIZON, REVISION, load_stats, sha256


def build_statistics(numeric_dir):
    numeric_dir = Path(numeric_dir)
    base = load_stats(numeric_dir / "relative_stats.json", "relative_depth")
    manifest = json.loads((numeric_dir / "manifest.json").read_text())
    if manifest["revision"] != REVISION or manifest["train_rows"] != base["train_rows"]:
        raise ValueError("Numeric index and existing delta statistics differ")
    actions = np.load(numeric_dir / "actions.npy", mmap_mode="r")
    kept_rows = np.load(numeric_dir / "kept_rows.npy", mmap_mode="r")
    kept_offsets = np.load(numeric_dir / "kept_offsets.npy", mmap_mode="r")
    episodes = np.load(numeric_dir / "train_episodes.npy")
    refs = np.load(numeric_dir / "train_refs.npy", mmap_mode="r")
    sums = np.zeros((HORIZON, 14), np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(HORIZON, np.int64)
    consumed = 0
    for episode in episodes:
        lo, hi = map(int, kept_offsets[int(episode):int(episode) + 2])
        expected_refs = np.arange(lo, hi, dtype=np.int64)
        if not np.array_equal(refs[consumed:consumed + hi - lo], expected_refs):
            raise ValueError("Training references differ from the frozen episode split")
        consumed += hi - lo
        a = np.asarray(actions[kept_rows[lo:hi]], dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 14 or not np.isfinite(a).all():
            raise ValueError(f"Invalid absolute actions in episode {episode}")
        # Use precisely the same valid H16 targets as delta; exclude terminal padding.
        for k in range(min(HORIZON, len(a))):
            targets = a[k:]
            sums[k] += targets.sum(0)
            squares[k] += np.square(targets).sum(0)
            counts[k] += len(targets)
    if consumed != len(refs) or consumed != base["train_rows"] or len(episodes) != base["train_episodes"]:
        raise ValueError("Incomplete training split")
    if counts.tolist() != base["valid_targets_per_horizon"] or counts.sum() == 0:
        raise ValueError("Absolute and delta statistics must use identical valid target counts")
    mean = sums.sum(0) / counts.sum()
    std = np.sqrt(np.maximum(squares.sum(0) / counts.sum() - mean ** 2, 0))
    # Grippers remain absolute in both modes, so preserve their exact existing scaling.
    mean[list(GRIPPERS)] = np.asarray(base["action"]["mean"])[list(GRIPPERS)]
    std[list(GRIPPERS)] = np.asarray(base["action"]["std"])[list(GRIPPERS)]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 1e-8):
        raise ValueError("Degenerate absolute action statistics")
    result = copy.deepcopy(base)
    result.update(format=ABS_STATS, action={"mean": mean.tolist(), "std": std.tolist()},
                  joint_stats_use="absolute targets; all valid H16 targets of training episodes; no tail padding or clipping",
                  source_numeric_manifest_sha256=sha256(numeric_dir / "manifest.json"),
                  source_delta_stats_sha256=sha256(numeric_dir / "relative_stats.json"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    stats = build_statistics(args.numeric_dir)
    if args.output.exists():
        if json.loads(args.output.read_text()) != stats:
            raise ValueError("Existing absolute statistics differ; use a new versioned output")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
        with partial.open("x") as stream:
            stream.write(json.dumps(stats, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(args.output)
    print(json.dumps({"passed": True, "statistics": str(args.output), "sha256": sha256(args.output),
                      "train_rows": stats["train_rows"], "valid_targets_per_horizon": stats["valid_targets_per_horizon"],
                      "seconds": round(time.monotonic() - start, 3)}), flush=True)


if __name__ == "__main__":
    main()
