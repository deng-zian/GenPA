import json
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from genpa.datasets.lerobot.packed_qwen_cache import PackedQwenCacheReader
from genpa.datasets.lerobot.transforms.image import VideoAugmentation
from .contract import HORIZON, REVISION, MODES, DEPTH_MODES, DEPTH_PROMPT, Normalization
from .spatial import CAMERAS, prepare_views, unpack_depth

AUGMENTATION = dict(
    p=0.65, augment_types=["corrupt_only", "color_only", "both", "both"],
    color_jitter=dict(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.04),
    gamma={"range": [0.8, 1.25]}, exposure={"ev_range": [-0.25, 0.25]}, gaussian_noise={"std": 0.015},
    random_resized_crop={"scale": [0.92, 1.0], "ratio": "preserve"}, rotate={"degrees": 8.0, "fill": "mean"},
)


class DepthDataset(Dataset):
    def __init__(self, root, numeric_dir, language_dir, stats_path, qwen_cache_dirs,
                 mode, height=240, is_training_set=True, max_decoders=6, sample_indices_path=None):
        if mode not in MODES:
            raise ValueError(mode)
        self.root, self.numeric_dir, self.language_dir = Path(root), Path(numeric_dir), Path(language_dir)
        manifest = json.loads((self.numeric_dir / "manifest.json").read_text())
        if manifest["revision"] != REVISION or Path(manifest["source_root"]).resolve() != self.root.resolve():
            raise ValueError("Numeric arrays must belong to this flex-pi RGB-D dataset")
        self.actions = np.load(self.numeric_dir / "actions.npy", mmap_mode="r")
        self.states = np.load(self.numeric_dir / "states.npy", mmap_mode="r")
        self.kept_rows = np.load(self.numeric_dir / "kept_rows.npy", mmap_mode="r")
        self.kept_offsets = np.load(self.numeric_dir / "kept_offsets.npy", mmap_mode="r")
        self.raw_offsets = np.load(self.numeric_dir / "raw_offsets.npy", mmap_mode="r")
        self.refs = np.load(self.numeric_dir / ("train_refs.npy" if is_training_set else "val_refs.npy"), mmap_mode="r")
        if sample_indices_path is not None:
            self.refs = self.refs[np.load(sample_indices_path)]
        self.task_ids = np.load(self.language_dir / "task_index.npy", mmap_mode="r")
        self.normalization = Normalization(stats_path, mode)
        self.qwen_cache_dirs = list(qwen_cache_dirs)
        self.mode, self.height, self.max_decoders = mode, int(height), int(max_decoders)
        self.video_augmentation = VideoAugmentation(**AUGMENTATION) if is_training_set else None
        self._pid = None
        self._decoders, self._prompts = OrderedDict(), OrderedDict()
        self._db, self._qwen = None, []
        self._depth_context = None

    def __len__(self):
        return len(self.refs)

    def _worker(self):
        if self._pid != os.getpid():
            self._decoders, self._prompts = OrderedDict(), OrderedDict()
            self._db = sqlite3.connect(f"file:{self.language_dir}/tasks.sqlite3?mode=ro&immutable=1", uri=True)
            self._qwen = [PackedQwenCacheReader(path, max_open_shards=8) for path in self.qwen_cache_dirs]
            self._depth_context = None
            self._pid = os.getpid()

    def _prompt(self, task_id):
        if task_id not in self._prompts:
            row = self._db.execute("SELECT prompt FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(f"No instruction for task_index={task_id}")
            self._prompts[task_id] = row[0]
            if len(self._prompts) > 512:
                self._prompts.popitem(last=False)
        return self._prompts[task_id]

    def _context(self, prompt):
        last = None
        for reader in self._qwen:
            try:
                return reader.get(prompt)
            except FileNotFoundError as error:
                if "Prompt is absent" not in str(error):
                    raise
                last = error
        raise FileNotFoundError("Instruction is missing from the configured Qwen caches") from last

    def _frame(self, episode, frame, camera, depth=False):
        kind = "observation.depth_x264rgb" if depth else "observation.images"
        path = self.root / f"videos/chunk-{episode // 1000:03d}/{kind}.{camera}/episode_{episode:06d}.mp4"
        key = str(path)
        decoder = self._decoders.pop(key, None)
        if decoder is None:
            from torchcodec.decoders import VideoDecoder

            decoder = VideoDecoder(key, device="cpu", seek_mode="approximate", num_ffmpeg_threads=1)
            expected = int(self.raw_offsets[episode + 1] - self.raw_offsets[episode])
            if len(decoder) != expected:
                raise ValueError(f"RGB/Depth/Parquet frame count differs: {path}")
        self._decoders[key] = decoder
        while len(self._decoders) > self.max_decoders:
            self._decoders.popitem(last=False)
        return decoder.get_frame_at(frame).data

    def numeric_sample(self, index):
        # The packed numeric view stores one 14-D absolute joint target and
        # 14-D request state per retained frame. One sample yields a 16-step
        # absolute action chunk, normalized only after selecting its frames.
        position = int(self.refs[index])
        episode = int(np.searchsorted(self.kept_offsets, position, side="right") - 1)
        last = int(self.kept_offsets[episode + 1]) - 1
        row = int(self.kept_rows[position])
        future = self.kept_rows[np.minimum(np.arange(position, position + HORIZON), last)]
        state = torch.from_numpy(self.states[row].copy())
        absolute = torch.from_numpy(self.actions[future].copy())
        return {"episode_index": episode, "frame_index": row - int(self.raw_offsets[episode]),
                "task_index": int(self.task_ids[row]), "raw_state": state,
                "action_absolute": absolute, "action": self.normalization.action(absolute, state),
                "proprio": self.normalization.proprio(state),
                "action_is_pad": torch.arange(position, position + HORIZON) > last}

    def _prepare_views(self, rgb, depth):
        return prepare_views(rgb, depth, self.height, augmentation=self.video_augmentation)

    def __getitem__(self, index):
        self._worker()
        sample = self.numeric_sample(index)
        episode, frame = sample["episode_index"], sample["frame_index"]
        rgb = torch.stack([self._frame(episode, frame, camera) for camera in CAMERAS])
        depth = None
        if self.mode in DEPTH_MODES:
            depth = torch.stack([unpack_depth(self._frame(episode, frame, camera, True)) for camera in CAMERAS])
        prepared = self._prepare_views(rgb, depth)
        prompt = self._prompt(sample["task_index"])
        context, mask = self._context(prompt)
        sample.update(prompt=prompt, text_hidden_states=context, text_attention_mask=mask,
                      video=prepared["reference"].unsqueeze(1))
        if depth is not None:
            if self._depth_context is None:
                self._depth_context = self._context(DEPTH_PROMPT)
            sample.update(video=torch.stack([prepared["reference"], prepared["target"]], dim=1),
                          visual_pixel_valid=prepared["visual_pixel_valid"],
                          visual_text_hidden_states=self._depth_context[0],
                          visual_text_attention_mask=self._depth_context[1])
        return sample
