#!/usr/bin/env python3
"""Validate exact prompt coverage and tensor contracts for a Qwen cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--context-len", type=int, default=128)
    args = parser.parse_args()

    payload = json.loads(args.prompt_manifest.read_text(encoding="utf-8"))
    prompts = [str(prompt) for prompt in payload["prompts"]]
    expected = {
        f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}.qwen3_flux2_len{args.context_len}.pt"
        for prompt in prompts
    }
    actual = {path.name for path in args.cache_dir.glob("*.qwen3_flux2_len*.pt")}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(
            f"Qwen cache mismatch: missing={len(missing)} "
            f"missing_examples={missing[:3]}"
        )

    hidden_shape = None
    for filename in sorted(expected):
        item = torch.load(args.cache_dir / filename, map_location="cpu")
        hidden = item["text_hidden_states"]
        mask = item["text_attention_mask"]
        if hidden.ndim != 2 or tuple(hidden.shape) != (args.context_len, 7680):
            raise ValueError(f"Bad hidden shape in {filename}: {tuple(hidden.shape)}")
        if tuple(mask.shape) != (args.context_len,) or mask.dtype != torch.bool:
            raise ValueError(f"Bad mask contract in {filename}: {tuple(mask.shape)} {mask.dtype}")
        if hidden_shape is None:
            hidden_shape = tuple(hidden.shape)
        elif tuple(hidden.shape) != hidden_shape:
            raise ValueError(
                f"Inconsistent hidden shape in {filename}: {tuple(hidden.shape)} != {hidden_shape}"
            )
    print(f"validated={len(expected)} hidden_shape={hidden_shape} context_len={args.context_len} unused_entries={len(extra)}")


if __name__ == "__main__":
    main()
