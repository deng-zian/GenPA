"""Index per-frame instructions and find reusable Qwen embeddings."""
import argparse
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from genpa.robotwin.depth.contract import DEPTH_PROMPT, REVISION, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--numeric", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-cache", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    work = args.output.with_name(args.output.name + ".partial")
    work.mkdir(parents=True, exist_ok=False)
    numeric = json.loads((args.numeric / "manifest.json").read_text())
    if numeric["revision"] != REVISION:
        raise ValueError("Unexpected numeric dataset version")
    offsets = np.load(args.numeric / "raw_offsets.npy")
    task_ids = np.lib.format.open_memmap(work / "task_index.npy", mode="w+", dtype=np.int64,
                                       shape=(int(offsets[-1]),))

    def read(episode):
        path = args.root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
        return episode, pq.read_table(path, columns=["task_index"], use_threads=False)["task_index"].to_numpy()

    with ThreadPoolExecutor(max_workers=12) as pool:
        for episode, values in pool.map(read, range(len(offsets) - 1)):
            task_ids[offsets[episode]:offsets[episode + 1]] = values
    task_ids.flush()
    needed = set(np.unique(task_ids[np.load(args.numeric / "kept_rows.npy", mmap_mode="r")]).tolist())
    connection = sqlite3.connect(work / "tasks.sqlite3")
    connection.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, prompt TEXT NOT NULL)")
    prompts = {DEPTH_PROMPT}
    missing_ids = needed.copy()
    with (args.root / "meta/tasks.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            key = int(record["task_index"])
            if key not in needed:
                continue
            prompt = str(record["task"]).strip()
            if not prompt:
                raise ValueError(f"Empty instruction for task_index={key}")
            connection.execute("INSERT INTO tasks VALUES (?, ?)", (key, prompt))
            prompts.add(prompt)
            missing_ids.remove(key)
    if missing_ids:
        raise ValueError(f"Missing {len(missing_ids)} frame instructions")
    connection.commit()
    connection.close()
    available = set()
    reusable = []
    for cache in args.reuse_cache:
        meta = json.loads((cache / "manifest.json").read_text())
        if meta.get("state") != "complete" or meta.get("format") != "qwen3_flux2_packed_v1":
            raise ValueError(f"Incomplete or incompatible reusable Qwen cache: {cache}")
        if meta["context_length"] != 128 or meta["hidden_dim"] != 7680:
            raise ValueError("Reusable Qwen cache dimensions differ")
        for shard in range(meta["num_shards"]):
            db = sqlite3.connect(f"file:{cache}/shard_{shard:03d}.sqlite3?mode=ro&immutable=1", uri=True)
            available.update(bytes(row[0]) for row in db.execute("SELECT key FROM entries"))
            db.close()
        reusable.append({"path": str(cache), "manifest_sha256": sha256(cache / "manifest.json")})
    missing = [p for p in sorted(prompts) if hashlib.sha256(p.encode()).digest() not in available]
    for name, values in (("all_prompts.jsonl", sorted(prompts)), ("missing_prompts.jsonl", missing)):
        with (work / name).open("w") as stream:
            for i, prompt in enumerate(values):
                stream.write(json.dumps({"task_index": i, "task": prompt}, ensure_ascii=False) + "\n")
    report = {"format": "robotwin_flexpi_language_v1", "revision": REVISION,
              "selection": "Parquet task_index of the selected current frame, not episode tasks[0]",
              "frame_instruction_ids": len(needed), "unique_prompts_including_depth": len(prompts),
              "reused_exact_prompts": len(prompts) - len(missing), "missing_prompts": len(missing),
              "reusable_caches": reusable, "numeric_manifest_sha256": sha256(args.numeric / "manifest.json"),
              "tasks_metadata_sha256": sha256(args.root / "meta/tasks.jsonl"),
              "all_prompts_sha256": sha256(work / "all_prompts.jsonl"),
              "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in work.iterdir()}}
    (work / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    work.rename(args.output)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
