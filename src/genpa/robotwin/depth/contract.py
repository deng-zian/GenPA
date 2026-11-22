import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .spatial import CAMERAS, Resolution

REVISION = "bee164afe94041d8c3d7dd1203725c41163fc3f4"
DATASET = "flex-pi/robotwin_3d"
JOINTS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPERS = (6, 13)
HORIZON = 16
DEPTH_PROMPT = "Generate a full-scene metric depth map for the current observation."
MODES = ("relative_action_only", "relative_depth", "absolute_depth")
DEPTH_MODES = ("relative_depth", "absolute_depth")
DELTA_STATS = "robotwin_flexpi_delta_stats_v1"
ABS_STATS = "robotwin_flexpi_absolute_stats_v1"


def encode_chunk(actions, state):
    if actions.shape[-1] != 14 or state.shape[-1] != 14:
        raise ValueError("Expected 14D action and raw request state")
    result = actions.clone() if isinstance(actions, torch.Tensor) else np.array(actions, copy=True)
    result[..., JOINTS] -= state[..., None, JOINTS]
    return result


def decode_chunk(actions, state):
    result = actions.clone() if isinstance(actions, torch.Tensor) else np.array(actions, copy=True)
    result[..., JOINTS] += state[..., None, JOINTS]
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stats(path, mode=None):
    stats = json.loads(Path(path).read_text())
    if stats.get("format") not in (DELTA_STATS, ABS_STATS) or stats.get("revision") != REVISION:
        raise ValueError("Statistics are not from the new flex-pi RGB-D dataset")
    if mode is not None:
        if mode not in MODES:
            raise ValueError(mode)
        expected = ABS_STATS if mode == "absolute_depth" else DELTA_STATS
        if stats["format"] != expected:
            raise ValueError("Action representation and normalization statistics differ")
    return stats


def checkpoint_contract(stats_path, height, mode):
    stats = load_stats(stats_path, mode)
    if mode not in MODES:
        raise ValueError(mode)
    absolute = mode == "absolute_depth"
    return {"version": "robotwin_flexpi_rgbd_abs_v1" if absolute else "robotwin_flexpi_rgbd_delta_v1",
            "dataset": DATASET, "revision": REVISION,
            "action_mode": "absolute_joint_target" if absolute else "current_request_state_joint_delta",
            "action_dim": 14, "horizon": HORIZON,
            "grippers": "absolute_continuous", "joint_indices": list(JOINTS),
            "action_normalization": "global_zscore_no_clip", "proprio_normalization": "minmax",
            "camera_order": list(CAMERAS), "resolution": Resolution.from_height(height).metadata(),
            "mode": mode, "stats_sha256": sha256(stats_path), "train_rows": stats["train_rows"],
            "frame_sequence": "official_nonidle_retained_rows"}


class Normalization:
    def __init__(self, path, mode=None):
        self.stats = load_stats(path, mode)
        self.absolute = self.stats["format"] == ABS_STATS
        self.mean = torch.tensor(self.stats["action"]["mean"], dtype=torch.float32)
        self.std = torch.tensor(self.stats["action"]["std"], dtype=torch.float32).clamp_min(1e-6)
        self.low = torch.tensor(self.stats["state"]["min"], dtype=torch.float32)
        self.high = torch.tensor(self.stats["state"]["max"], dtype=torch.float32)

    def proprio(self, state):
        span = (self.high - self.low).to(state)
        return torch.where(span > 1e-8, (state - self.low.to(state)) * 2 / span.clamp_min(1e-8) - 1,
                           torch.zeros_like(state)).clamp(-1, 1)

    def action(self, absolute, state):
        target = absolute if self.absolute else encode_chunk(absolute, state)
        return (target - self.mean.to(absolute)) / self.std.to(absolute)

    def decode(self, normalized, raw_request_state):
        target = normalized * self.std.to(normalized) + self.mean.to(normalized)
        if self.absolute:
            return target
        return decode_chunk(target, raw_request_state.to(normalized))
