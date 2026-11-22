from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path

import torch


class PackedQwenCacheReader:
    """Read variable-length Qwen states from immutable binary shards."""

    def __init__(self, cache_dir: str | os.PathLike[str], max_open_shards: int = 16):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing packed Qwen cache manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("state") != "complete":
            raise RuntimeError(f"Packed Qwen cache is not complete: {manifest_path}")
        if self.manifest.get("format") != "qwen3_flux2_packed_v1":
            raise RuntimeError(f"Unsupported packed Qwen cache format: {self.manifest}")
        self.context_length = int(self.manifest["context_length"])
        self.hidden_dim = int(self.manifest["hidden_dim"])
        self.num_shards = int(self.manifest["num_shards"])
        self.max_open_shards = int(max_open_shards)
        if self.max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        self._pid = os.getpid()
        self._handles: OrderedDict[int, tuple[sqlite3.Connection, int]] = OrderedDict()

    def _close_handles(self) -> None:
        for connection, data_fd in self._handles.values():
            connection.close()
            os.close(data_fd)
        self._handles.clear()

    def _ensure_process(self) -> None:
        current_pid = os.getpid()
        if current_pid != self._pid:
            self._close_handles()
            self._pid = current_pid

    def _open_shard(self, shard_id: int) -> tuple[sqlite3.Connection, int]:
        self._ensure_process()
        cached = self._handles.pop(shard_id, None)
        if cached is not None:
            self._handles[shard_id] = cached
            return cached
        index_path = (self.cache_dir / f"shard_{shard_id:03d}.sqlite3").resolve()
        data_path = self.cache_dir / f"shard_{shard_id:03d}.bin"
        if not index_path.is_file() or not data_path.is_file():
            raise FileNotFoundError(
                f"Missing packed Qwen shard {shard_id}: {index_path}, {data_path}"
            )
        connection = sqlite3.connect(
            f"file:{index_path}?mode=ro&immutable=1",
            uri=True,
            check_same_thread=False,
        )
        connection.execute("PRAGMA query_only=ON")
        data_fd = os.open(data_path, os.O_RDONLY)
        opened = (connection, data_fd)
        self._handles[shard_id] = opened
        while len(self._handles) > self.max_open_shards:
            _, (old_connection, old_fd) = self._handles.popitem(last=False)
            old_connection.close()
            os.close(old_fd)
        return opened

    def get(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        shard_id = int(digest[0]) % self.num_shards
        connection, data_fd = self._open_shard(shard_id)
        row = connection.execute(
            "SELECT byte_offset, token_length FROM entries WHERE key = ?",
            (sqlite3.Binary(digest),),
        ).fetchone()
        if row is None:
            raise FileNotFoundError(
                f"Prompt is absent from packed Qwen cache {self.cache_dir}: "
                f"sha256={digest.hex()}"
            )
        byte_offset, token_length = int(row[0]), int(row[1])
        if token_length <= 0 or token_length > self.context_length:
            raise RuntimeError(
                f"Invalid token length {token_length} for sha256={digest.hex()}"
            )
        expected_bytes = token_length * self.hidden_dim * 2
        raw = os.pread(data_fd, expected_bytes, byte_offset)
        if len(raw) != expected_bytes:
            raise OSError(
                f"Short packed Qwen read for sha256={digest.hex()}: "
                f"expected={expected_bytes}, got={len(raw)}"
            )
        valid_hidden = torch.frombuffer(
            bytearray(raw),
            dtype=torch.bfloat16,
        ).reshape(token_length, self.hidden_dim)
        context = torch.zeros(
            self.context_length,
            self.hidden_dim,
            dtype=torch.bfloat16,
        )
        context[:token_length].copy_(valid_hidden)
        context_mask = torch.zeros(self.context_length, dtype=torch.bool)
        context_mask[:token_length] = True
        return context, context_mask

    def close(self) -> None:
        self._close_handles()

    def __del__(self) -> None:
        try:
            self._close_handles()
        except Exception:
            pass


_READERS: dict[str, PackedQwenCacheReader] = {}


def get_packed_qwen_context(
    cache_dir: str | os.PathLike[str],
    prompt: str,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = str(Path(cache_dir).resolve())
    reader = _READERS.get(cache_key)
    if reader is None:
        reader = PackedQwenCacheReader(cache_key)
        _READERS[cache_key] = reader
    if reader.context_length != int(context_length):
        raise ValueError(
            f"Packed Qwen context mismatch: cache={reader.context_length}, "
            f"requested={context_length}"
        )
    return reader.get(prompt)
