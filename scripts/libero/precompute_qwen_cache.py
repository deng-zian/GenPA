#!/usr/bin/env python3
"""Precompute Qwen3/FLUX.2 text states for a supplied prompt manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from genpa.models.backbones.flux2_imports import ensure_flux2_importable


def _atomic_save(payload: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--flux2-src", type=str, required=True)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    payload = json.loads(args.prompt_manifest.read_text(encoding="utf-8"))
    # Each distinct task/depth prompt maps to one immutable SHA-256 cache file.
    # Action-prefix tokens are learned by the policy and are not cached here.
    prompts = [str(prompt) for prompt in payload["prompts"]]
    if not prompts or len(set(prompts)) != len(prompts):
        raise ValueError("Prompt manifest must contain nonempty, unique prompts.")
    prompts = prompts[rank::world_size]
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    ensure_flux2_importable(args.flux2_src)
    from flux2.text_encoder import OUTPUT_LAYERS_QWEN3

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
    ).to(device).eval()

    written = 0
    skipped = 0
    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            needed = []
            for prompt in batch_prompts:
                digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                path = args.cache_dir / f"{digest}.qwen3_flux2_len{args.context_len}.pt"
                if path.exists() and not args.overwrite:
                    skipped += 1
                else:
                    needed.append(prompt)
            if not needed:
                continue
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for prompt in needed
            ]
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.context_len,
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = torch.stack(
                [outputs.hidden_states[index] for index in OUTPUT_LAYERS_QWEN3],
                dim=1,
            )
            hidden = rearrange(hidden, "b c l d -> b l (c d)")
            for index, prompt in enumerate(needed):
                digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                path = args.cache_dir / f"{digest}.qwen3_flux2_len{args.context_len}.pt"
                _atomic_save(
                    {
                        "text_hidden_states": hidden[index].detach().cpu().to(torch.bfloat16),
                        "text_attention_mask": attention_mask[index].detach().cpu().bool(),
                    },
                    path,
                )
                written += 1
    print(f"rank={rank} written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
