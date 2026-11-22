#!/usr/bin/env python3
"""Check that a complete packed Qwen cache covers every processed task prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genpa.datasets.lerobot.packed_qwen_cache import PackedQwenCacheReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    prompts = [
        str(json.loads(line)["task"])
        for line in args.tasks_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts or len(prompts) != len(set(prompts)):
        raise ValueError("Prompt list must be nonempty and unique")
    reader = PackedQwenCacheReader(args.cache_dir)
    try:
        if int(reader.manifest["prompt_count"]) != len(prompts):
            raise ValueError("Cache manifest prompt count differs from processed task index")
        for prompt in prompts:
            hidden, mask = reader.get(prompt)
            if hidden.shape != (reader.context_length, reader.hidden_dim):
                raise ValueError(f"Invalid hidden shape for prompt: {prompt[:80]}")
            if mask.shape != (reader.context_length,) or not bool(mask.any()):
                raise ValueError(f"Invalid attention mask for prompt: {prompt[:80]}")
    finally:
        reader.close()
    print(f"validated={len(prompts)} context_length={reader.context_length} hidden_dim={reader.hidden_dim}")


if __name__ == "__main__":
    main()
