#!/usr/bin/env python3
"""Build an exact-length, sharded FLUX.2 Qwen3 cache for RoboTwin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer


HIDDEN_DIM = 7680
BYTES_PER_VALUE = 2
FORMAT_NAME = "qwen3_flux2_packed_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--flux2-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--prefix-length", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=256)
    parser.add_argument("--tokenize-batch-size", type=int, default=4096)
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument("--commit-every", type=int, default=512)
    return parser.parse_args()


def atomic_json_dump(payload: dict, path: Path) -> None:
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def iter_batches(values: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


class ShardWriter:
    def __init__(self, output_dir: Path, shard_id: int, hidden_dim: int, commit_every: int):
        self.output_dir = output_dir
        self.shard_id = int(shard_id)
        self.hidden_dim = int(hidden_dim)
        self.commit_every = int(commit_every)
        stem = f"shard_{self.shard_id:03d}"
        self.final_data_path = output_dir / f"{stem}.bin"
        self.final_index_path = output_dir / f"{stem}.sqlite3"
        self.meta_path = output_dir / f"{stem}.meta.json"
        self.partial_data_path = output_dir / f".{stem}.partial.bin"
        self.partial_index_path = output_dir / f".{stem}.partial.sqlite3"
        self.is_final = self.final_data_path.is_file() and self.final_index_path.is_file()
        if self.final_data_path.exists() != self.final_index_path.exists():
            raise RuntimeError(f"Incomplete final shard pair for {stem}")
        self.connection: sqlite3.Connection | None = None
        self.data_file = None
        self.pending = 0
        if self.is_final:
            return
        self.connection = sqlite3.connect(self.partial_index_path, timeout=120.0)
        if not self._table_exists():
            self.connection.execute("PRAGMA page_size=65536")
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "key BLOB PRIMARY KEY, "
            "byte_offset INTEGER NOT NULL, "
            "token_length INTEGER NOT NULL"
            ") WITHOUT ROWID"
        )
        self.connection.commit()
        self.partial_data_path.parent.mkdir(parents=True, exist_ok=True)
        had_data_file = self.partial_data_path.is_file()
        self.data_file = self.partial_data_path.open("a+b", buffering=0)
        safe_end = self._safe_end()
        current_size = self.partial_data_path.stat().st_size
        if safe_end > 0 and (not had_data_file or current_size < safe_end):
            raise RuntimeError(
                f"Shard {self.shard_id} index references {safe_end} bytes, "
                f"but partial data has only {current_size} bytes"
            )
        self.data_file.truncate(safe_end)
        self.data_file.seek(safe_end)

    def _table_exists(self) -> bool:
        assert self.connection is not None
        row = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entries'"
        ).fetchone()
        return row is not None

    def _safe_end(self) -> int:
        assert self.connection is not None
        row = self.connection.execute(
            "SELECT byte_offset, token_length FROM entries ORDER BY byte_offset DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return 0
        return int(row[0]) + int(row[1]) * self.hidden_dim * BYTES_PER_VALUE

    def existing_keys(self) -> set[bytes]:
        path = self.final_index_path if self.is_final else self.partial_index_path
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            return {bytes(row[0]) for row in connection.execute("SELECT key FROM entries")}
        finally:
            connection.close()

    def append(self, digest: bytes, token_length: int, raw_hidden: bytes) -> None:
        if self.is_final:
            raise RuntimeError(f"Cannot append to completed shard {self.shard_id}")
        assert self.connection is not None and self.data_file is not None
        expected_bytes = int(token_length) * self.hidden_dim * BYTES_PER_VALUE
        if len(raw_hidden) != expected_bytes:
            raise ValueError(
                f"Shard {self.shard_id} payload mismatch: expected={expected_bytes}, got={len(raw_hidden)}"
            )
        byte_offset = self.data_file.tell()
        self.data_file.write(raw_hidden)
        self.connection.execute(
            "INSERT INTO entries(key, byte_offset, token_length) VALUES (?, ?, ?)",
            (sqlite3.Binary(digest), byte_offset, int(token_length)),
        )
        self.pending += 1
        if self.pending >= self.commit_every:
            self.commit()

    def commit(self) -> None:
        if self.is_final or self.pending == 0:
            return
        assert self.connection is not None and self.data_file is not None
        self.data_file.flush()
        os.fsync(self.data_file.fileno())
        self.connection.commit()
        self.pending = 0

    def summarize(self) -> tuple[int, int]:
        path = self.final_index_path if self.is_final else self.partial_index_path
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(token_length), 0) FROM entries"
            ).fetchone()
            return int(row[0]), int(row[1])
        finally:
            connection.close()

    def finalize(self, expected_count: int, expected_tokens: int) -> dict:
        if not self.is_final:
            self.commit()
            assert self.connection is not None and self.data_file is not None
            integrity = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"SQLite integrity failure for shard {self.shard_id}: {integrity}")
            self.connection.close()
            self.connection = None
            self.data_file.close()
            self.data_file = None
        count, token_count = self.summarize()
        data_path = self.final_data_path if self.is_final else self.partial_data_path
        payload_bytes = data_path.stat().st_size
        if count != int(expected_count) or token_count != int(expected_tokens):
            raise RuntimeError(
                f"Shard {self.shard_id} count mismatch: actual=({count},{token_count}), "
                f"expected=({expected_count},{expected_tokens})"
            )
        expected_bytes = token_count * self.hidden_dim * BYTES_PER_VALUE
        if payload_bytes != expected_bytes:
            raise RuntimeError(
                f"Shard {self.shard_id} byte mismatch: actual={payload_bytes}, expected={expected_bytes}"
            )
        if not self.is_final:
            os.replace(self.partial_data_path, self.final_data_path)
            os.replace(self.partial_index_path, self.final_index_path)
            self.is_final = True
        payload = {
            "shard_id": self.shard_id,
            "prompt_count": count,
            "valid_tokens": token_count,
            "payload_bytes": payload_bytes,
            "data_file": self.final_data_path.name,
            "index_file": self.final_index_path.name,
        }
        atomic_json_dump(payload, self.meta_path)
        return payload


def main() -> None:
    args = parse_args()
    if args.context_length <= args.prefix_length:
        raise ValueError("context length must be larger than prefix length")
    if args.num_shards <= 0 or args.num_shards > 256:
        raise ValueError("num_shards must be in [1,256]")
    if not torch.cuda.is_available():
        raise RuntimeError("Packed Qwen cache generation requires CUDA")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("state") == "complete" and int(manifest.get("prompt_count", -1)) == args.expected_prompts:
            print(f"[rank{rank}] complete cache already exists: {manifest_path}", flush=True)
            dist.barrier()
            dist.destroy_process_group()
            return

    flux2_src = Path(args.flux2_src).resolve()
    for candidate in (flux2_src / "src", flux2_src):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from flux2.text_encoder import OUTPUT_LAYERS_QWEN3

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        use_fast=True,
    )
    owned_shards = [shard for shard in range(args.num_shards) if shard % world_size == rank]
    writers = {
        shard: ShardWriter(output_dir, shard, HIDDEN_DIM, args.commit_every)
        for shard in owned_shards
    }
    existing = set().union(*(writer.existing_keys() for writer in writers.values()))
    buckets: dict[int, list[tuple[bytes, list[int]]]] = defaultdict(list)
    expected_count = CounterLike(owned_shards)
    expected_tokens = CounterLike(owned_shards)
    seen: set[bytes] = set()
    selected: list[tuple[bytes, str]] = []
    raw_records = 0
    scan_started = time.perf_counter()

    def tokenize_selected() -> None:
        nonlocal selected
        if not selected:
            return
        conversations = [[{"role": "user", "content": prompt}] for _, prompt in selected]
        token_rows = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            padding=False,
            truncation=False,
        )
        if len(token_rows) != len(selected):
            raise RuntimeError("Tokenizer batch-size mismatch")
        for (digest, _), token_ids in zip(selected, token_rows, strict=True):
            token_length = len(token_ids)
            if token_length > args.context_length - args.prefix_length:
                raise ValueError(
                    f"Prompt {digest.hex()} has {token_length} tokens; "
                    f"only {args.context_length - args.prefix_length} fit after Prefix"
                )
            shard_id = int(digest[0]) % args.num_shards
            expected_count[shard_id] += 1
            expected_tokens[shard_id] += token_length
            if digest not in existing:
                buckets[token_length].append((digest, token_ids))
        selected = []

    with Path(args.tasks_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            raw_records += 1
            prompt = str(record["task"]).strip()
            if not prompt:
                raise ValueError(f"Empty prompt at line {line_number}")
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            shard_id = int(digest[0]) % args.num_shards
            if shard_id % world_size != rank:
                continue
            if digest in seen:
                raise ValueError(f"Duplicate prompt digest at line {line_number}: {digest.hex()}")
            seen.add(digest)
            selected.append((digest, prompt))
            if len(selected) >= args.tokenize_batch_size:
                tokenize_selected()
        tokenize_selected()

    local_prompt_count = sum(expected_count.values())
    local_prompt_tensor = torch.tensor([local_prompt_count], dtype=torch.int64, device=device)
    dist.all_reduce(local_prompt_tensor, op=dist.ReduceOp.SUM)
    global_prompt_count = int(local_prompt_tensor.item())
    if global_prompt_count != args.expected_prompts or raw_records != args.expected_prompts:
        raise RuntimeError(
            f"Prompt count mismatch: raw={raw_records}, distributed={global_prompt_count}, "
            f"expected={args.expected_prompts}"
        )
    pending_count = sum(len(values) for values in buckets.values())
    print(
        f"[rank{rank}] prompt scan complete: owned={local_prompt_count:,}, "
        f"existing={len(existing):,}, pending={pending_count:,}, "
        f"seconds={time.perf_counter() - scan_started:.1f}",
        flush=True,
    )

    if pending_count:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device).eval()
        model.requires_grad_(False)
        completed = 0
        started = time.perf_counter()
        with torch.inference_mode():
            for token_length in sorted(buckets):
                records = sorted(buckets[token_length], key=lambda item: (item[0][0], item[0]))
                for batch in iter_batches(records, args.model_batch_size):
                    input_ids = torch.full(
                        (len(batch), args.context_length),
                        fill_value=int(tokenizer.pad_token_id),
                        dtype=torch.long,
                        device=device,
                    )
                    input_ids[:, :token_length] = torch.tensor(
                        [token_ids for _, token_ids in batch],
                        dtype=torch.long,
                        device=device,
                    )
                    attention_mask = torch.zeros_like(input_ids)
                    attention_mask[:, :token_length] = 1
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
                    hidden = rearrange(hidden, "b c l d -> b l (c d)")[:, :token_length]
                    if hidden.shape[-1] != HIDDEN_DIM:
                        raise RuntimeError(f"Unexpected Qwen hidden dim: {tuple(hidden.shape)}")
                    hidden_cpu = hidden.to(device="cpu", dtype=torch.bfloat16).contiguous()
                    for row_index, (digest, _) in enumerate(batch):
                        shard_id = int(digest[0]) % args.num_shards
                        raw_hidden = hidden_cpu[row_index].view(torch.uint16).numpy().tobytes()
                        writers[shard_id].append(digest, token_length, raw_hidden)
                    completed += len(batch)
                    if completed % (args.model_batch_size * 100) < len(batch):
                        elapsed = time.perf_counter() - started
                        print(
                            f"[rank{rank}] generated={completed:,}/{pending_count:,} "
                            f"rate={completed / max(elapsed, 1e-6):.1f} prompts/s",
                            flush=True,
                        )
        del model

    shard_payloads = []
    for shard_id in owned_shards:
        shard_payloads.append(
            writers[shard_id].finalize(
                expected_count=expected_count[shard_id],
                expected_tokens=expected_tokens[shard_id],
            )
        )
    dist.barrier()
    if rank == 0:
        shard_meta = []
        for shard_id in range(args.num_shards):
            meta_path = output_dir / f"shard_{shard_id:03d}.meta.json"
            if not meta_path.is_file():
                raise FileNotFoundError(f"Missing shard metadata: {meta_path}")
            shard_meta.append(json.loads(meta_path.read_text(encoding="utf-8")))
        total_prompts = sum(int(item["prompt_count"]) for item in shard_meta)
        total_tokens = sum(int(item["valid_tokens"]) for item in shard_meta)
        total_bytes = sum(int(item["payload_bytes"]) for item in shard_meta)
        if total_prompts != args.expected_prompts:
            raise RuntimeError(f"Final prompt count mismatch: {total_prompts}")
        if total_bytes != total_tokens * HIDDEN_DIM * BYTES_PER_VALUE:
            raise RuntimeError("Final payload byte count mismatch")
        manifest = {
            "state": "complete",
            "format": FORMAT_NAME,
            "prompt_count": total_prompts,
            "context_length": args.context_length,
            "prefix_length": args.prefix_length,
            "hidden_dim": HIDDEN_DIM,
            "dtype": "bfloat16",
            "bytes_per_value": BYTES_PER_VALUE,
            "num_shards": args.num_shards,
            "valid_tokens": total_tokens,
            "payload_bytes": total_bytes,
            "qwen_model": str(Path(args.model).resolve()),
            "output_layers": list(OUTPUT_LAYERS_QWEN3),
            "chat_template": {
                "role": "user",
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
            "qwen_forward_context_length": args.context_length,
            "stored_padding_tokens": False,
            "tasks_jsonl": str(Path(args.tasks_jsonl).resolve()),
            "shards": shard_meta,
        }
        atomic_json_dump(manifest, manifest_path)
        print(
            f"[complete] prompts={total_prompts:,} tokens={total_tokens:,} "
            f"payload_bytes={total_bytes:,} manifest={manifest_path}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


class CounterLike(dict[int, int]):
    def __init__(self, keys: Iterable[int]):
        super().__init__((int(key), 0) for key in keys)

    def __missing__(self, key: int) -> int:
        return 0


if __name__ == "__main__":
    main()
