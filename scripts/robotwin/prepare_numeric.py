"""Build immutable new-dataset numeric arrays, official filtering and delta statistics."""
import argparse
import importlib.util
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from genpa.robotwin.depth.contract import HORIZON, JOINTS, GRIPPERS, REVISION, sha256


def load_filter(path):
    spec = importlib.util.spec_from_file_location("official_nonidle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numeric_column(table, key):
    column = table[key].combine_chunks()
    return np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(len(table), 14)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--official-filter", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        manifest = json.loads((args.output / "manifest.json").read_text())
        if manifest["revision"] != REVISION:
            raise ValueError("Existing numeric index is from a different dataset")
        print(json.dumps({"state": "already_complete", **manifest}), flush=True)
        return
    work = args.output.with_name(args.output.name + ".partial")
    if work.exists():
        raise FileExistsError(f"Inspect the partial build before rerunning: {work}")
    work.mkdir(parents=True)
    start = time.monotonic()
    official = load_filter(args.official_filter)
    episodes = [json.loads(line) for line in (args.root / "meta/episodes.jsonl").open()]
    episodes.sort(key=lambda row: row["episode_index"])
    if [r["episode_index"] for r in episodes] != list(range(27500)):
        raise ValueError("Expected all 27,500 episodes")
    lengths = np.array([r["length"] for r in episodes], dtype=np.int64)
    offsets = np.concatenate([[0], lengths.cumsum()])
    if offsets[-1] != 6183813:
        raise ValueError(f"Unexpected raw frame count: {offsets[-1]}")
    action = np.lib.format.open_memmap(work / "actions.npy", mode="w+", dtype=np.float32,
                                     shape=(int(offsets[-1]), 14))
    state = np.lib.format.open_memmap(work / "states.npy", mode="w+", dtype=np.float32,
                                    shape=action.shape)
    shuffled = np.arange(len(episodes))
    np.random.default_rng(42).shuffle(shuffled)
    train_episodes = set(shuffled[:int(len(episodes) * 0.99)].tolist())
    kept_rows, train_refs, val_refs, kept_offsets = [], [], [], [0]
    ranges_by_episode = {}
    sums = np.zeros((HORIZON, 14), np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(HORIZON, np.int64)
    prompt_db = sqlite3.connect(work / "episode_prompts.sqlite3")
    prompt_db.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, prompts TEXT NOT NULL)")

    def read(row):
        episode = row["episode_index"]
        path = args.root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
        table = pq.read_table(path, columns=["action", "observation.state", "frame_index", "episode_index"],
                              use_threads=False)
        a, s = numeric_column(table, "action"), numeric_column(table, "observation.state")
        if len(table) != row["length"] or not np.isfinite(a).all() or not np.isfinite(s).all():
            raise ValueError(f"Invalid numeric episode {episode}")
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(len(table)))
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.full(len(table), episode))
        idle = official._idle_mask(a.astype(np.float64), s.astype(np.float64), idle_l2_threshold=1e-3,
                                  idle_arm_l2_threshold=1e-3, idle_gripper_l2_threshold=1e-3)
        ranges = official._compute_keep_ranges(idle, min_idle_len=5, min_non_idle_len=1,
                                              filter_last_n_in_ranges=0)
        kept = np.concatenate([np.arange(lo, hi, dtype=np.int64) for lo, hi in ranges])
        return episode, a, s, ranges, kept

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Bound in-flight Parquet outputs; do not retain the full corpus as Python objects.
        for block in range(0, len(episodes), 128):
            for episode, a, s, ranges, kept in pool.map(read, episodes[block:block + 128]):
                lo, hi = offsets[episode:episode + 2]
                action[lo:hi], state[lo:hi] = a, s
                ranges_by_episode[str(episode)] = ranges
                kept_rows.append((kept + lo).astype(np.int64))
                refs = np.arange(kept_offsets[-1], kept_offsets[-1] + len(kept), dtype=np.int64)
                (train_refs if episode in train_episodes else val_refs).append(refs)
                kept_offsets.append(kept_offsets[-1] + len(kept))
                prompt_db.execute("INSERT INTO episodes VALUES (?, ?)",
                                  (episode, json.dumps(episodes[episode]["tasks"], ensure_ascii=False)))
                if episode in train_episodes:
                    a, s = a[kept].astype(np.float64), s[kept].astype(np.float64)
                    for k in range(HORIZON):
                        n = len(a) - k
                        if n <= 0:
                            break
                        delta = a[k:].copy()
                        delta[:, JOINTS] -= s[:n, JOINTS]
                        sums[k] += delta.sum(0)
                        squares[k] += np.square(delta).sum(0)
                        counts[k] += n
            if block % 1024 == 0:
                print(json.dumps({"episodes": min(block + 128, len(episodes)), "kept_rows": kept_offsets[-1],
                                  "seconds": round(time.monotonic() - start, 1)}), flush=True)
    action.flush()
    state.flush()
    prompt_db.commit()
    prompt_db.close()
    arrays = {"raw_offsets": offsets, "kept_offsets": np.array(kept_offsets, np.int64),
              "kept_rows": np.concatenate(kept_rows), "train_refs": np.concatenate(train_refs),
              "val_refs": np.concatenate(val_refs), "train_episodes": np.array(sorted(train_episodes), np.int32)}
    for key, value in arrays.items():
        np.save(work / (key + ".npy"), value)
    filter_payload = {"format": "genpa_nonidle_ranges_v1", "dataset_dir": str(args.root),
                      "episodes": ranges_by_episode,
                      "thresholds": {"idle_l2_threshold": 1e-3, "idle_arm_l2_threshold": 1e-3,
                                     "idle_gripper_l2_threshold": 1e-3, "min_idle_len": 5,
                                     "min_non_idle_len": 1, "terminal_idle_kept": True},
                      "summary": {"episodes": len(episodes), "total_steps": int(offsets[-1]),
                                  "kept_steps": kept_offsets[-1]}}
    (work / "nonidle_ranges.json").write_text(json.dumps(filter_payload) + "\n")
    mean = sums.sum(0) / counts.sum()
    std = np.sqrt(np.maximum(squares.sum(0) / counts.sum() - mean ** 2, 0))
    public = json.loads((args.root / "dataset_stats.json").read_text())
    for key, target in (("global_mean", mean), ("global_std", std)):
        target[list(GRIPPERS)] = np.asarray(public["action"]["default"][key])[list(GRIPPERS)]
    if not np.isfinite(std).all() or np.any(std < 1e-8):
        raise ValueError("Degenerate action statistics")
    stats = {"format": "robotwin_flexpi_delta_stats_v1", "revision": REVISION,
             "train_rows": len(arrays["train_refs"]), "val_rows": len(arrays["val_refs"]),
             "train_episodes": len(train_episodes), "seed": 42, "horizon": HORIZON,
             "valid_targets_per_horizon": counts.tolist(),
             "action": {"mean": mean.tolist(), "std": std.tolist()},
             "state": {"min": public["state"]["default"]["global_min"],
                       "max": public["state"]["default"]["global_max"]},
             "filter_sha256": sha256(work / "nonidle_ranges.json"),
             "public_stats_sha256": sha256(args.root / "dataset_stats.json"),
             "public_stats_use": "absolute proprio min/max and continuous gripper mean/std only",
             "joint_stats_use": "all valid H16 targets of training episodes; no tail padding or clipping"}
    (work / "relative_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    manifest = {"format": "robotwin_flexpi_numeric_v1", "revision": REVISION, "source_root": str(args.root),
                "raw_rows": int(offsets[-1]), "kept_rows": kept_offsets[-1],
                "train_rows": stats["train_rows"], "val_rows": stats["val_rows"],
                "train_episodes": len(train_episodes), "val_episodes": len(episodes) - len(train_episodes),
                "official_filter_sha256": sha256(args.official_filter),
                "metadata_sha256": sha256(args.root / "meta/episodes.jsonl"),
                "statistics_sha256": sha256(work / "relative_stats.json"),
                "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in work.iterdir()},
                "elapsed_seconds": time.monotonic() - start}
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    work.rename(args.output)
    print(json.dumps({"state": "complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
