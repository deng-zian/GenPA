"""Read the image-form supervision in Appendix B from one extracted replay task."""
from functools import lru_cache
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

from .geometry import camera_rays, surface_normal_target
from .pointmap import point_map_target

CAMERAS = ("agentview", "robot0_eye_in_hand")
VIDEO_KEYS = ("observation.images.image", "observation.images.wrist_image")
TARGETS = ("depth", "normals", "pointmap", "segmentation", "grounding", "future_rgb", "future_depth")


def _records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _rgb_frame(path, frame_index):
    import av
    with av.open(str(path)) as container:
        for index, image in enumerate(container.decode(video=0)):
            if index == frame_index:
                return image.to_ndarray(format="rgb24")
    raise IndexError(f"Missing video frame {frame_index} in {path.name}")


def _depth_rgb(depth):
    from genpa.datasets.lerobot.depth_targets import encode_depth
    return encode_depth(depth).transpose(1, 2, 0)


class VisualTargets:
    """Return dual-view targets as float32 [camera, channel, height, width] in [0, 1]."""

    def __init__(self, root):
        self.root = Path(root)
        rows = _records(self.root / "labels/episodes.jsonl")
        self.episodes = {r["source_episode_id"]: r for r in rows}
        classes = json.loads((self.root / "labels/segmentation.json").read_text())["labels"]
        self.palette = np.asarray([r["rgb"] for r in sorted(classes, key=lambda r: r["id"])], np.uint8)

    @lru_cache(maxsize=1)
    def _training_index(self):
        rows = _records(self.root / "paper/manifests/episode_map.jsonl")
        mapping = {r["source_episode_id"]: str(r["episode_index"]) for r in rows}
        ranges = json.loads((self.root / "paper/policy/action_keep_ranges.json").read_text())["episodes"]
        return mapping, ranges

    def future_frame(self, episode, frame):
        mapping, ranges = self._training_index()
        if episode not in mapping:
            raise ValueError("Future targets use episodes in the paper training view")
        keep = np.concatenate([np.arange(start, end) for start, end in ranges[mapping[episode]]])
        pos = int(np.searchsorted(keep, frame))
        if pos == len(keep) or keep[pos] != frame:
            raise ValueError(f"Frame {frame} is not a retained action anchor for {episode}")
        return int(keep[min(pos + 16, len(keep) - 1)])

    @lru_cache(maxsize=4)
    def _grounding_table(self, suite):
        cols = ["global_episode_index", "frame_index", "phase_id", "grounding_valid_bits",
                "agent_union_pixels", "wrist_union_pixels"]
        return pq.read_table(self.root / "labels/grounding" / (suite + ".parquet"), columns=cols)

    def _grounding_row(self, record, frame):
        row = self._grounding_table(record["suite"]).slice(record["grounding_row_start"] + frame, 1).to_pylist()[0]
        if (row["global_episode_index"], row["frame_index"]) != (record["grounding_episode_index"], frame):
            raise ValueError("Grounding metadata and geometry are not aligned")
        phase = row["phase_id"]
        roles = record["phases"][phase]["resolved_roles"] if phase >= 0 else {}
        return row, roles

    def read(self, target, episode, frame):
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}. Choose from {TARGETS}")
        rec = self.episodes[episode]
        if not 0 <= frame < rec["frames"]:
            raise IndexError(frame)
        selected = self.future_frame(episode, frame) if target.startswith("future_") else frame
        base = self.root / "canonical" / rec["suite"]
        name = "episode_%06d" % rec["suite_episode_index"]
        valid_cameras = np.ones(2, dtype=bool)
        if target == "future_rgb":
            rgb = [_rgb_frame(base / "videos/chunk-000" / key / (name + ".mp4"), selected) for key in VIDEO_KEYS]
            values = np.stack(rgb).astype(np.float32) / 255.0
            masks = np.ones(values.shape[:3], dtype=bool)
        else:
            images, masks_list = [], []
            if target == "grounding":
                grounding, roles = self._grounding_row(rec, frame)
            with h5py.File(base / "visual_gt/chunk-000" / (name + ".hdf5"), "r") as handle:
                for k, cam in enumerate(CAMERAS):
                    group = handle[cam]
                    if target in ("depth", "future_depth"):
                        d = group["depth_m"][selected]
                        image = _depth_rgb(d)
                        mask = np.isfinite(d) & (d > 0)
                    elif target in ("normals", "pointmap"):
                        d = group["depth_m"][frame]
                        rays = camera_rays(group["intrinsics_raw"][:], group["stored_to_raw_pixel"][:], *d.shape)
                        if target == "normals":
                            encoded, mask, _ = surface_normal_target(d, rays)
                        else:
                            encoded, mask, _ = point_map_target(d, rays, group["camera_to_world_raw"][frame],
                                                                 rec["world_from_base"], group["geom_id_plus1"][frame])
                        image = encoded.astype(np.float32) / 255.0
                    elif target == "segmentation":
                        geom = group["geom_id_plus1"][frame]
                        classes = np.asarray(rec["segmentation_lut"], dtype=np.uint16)[geom]
                        image = self.palette[classes].astype(np.float32) / 255.0
                        mask = np.ones(geom.shape, dtype=bool)
                    else:
                        geom = group["geom_id_plus1"][frame]
                        image = np.stack([np.isin(geom, roles.get(role, []))
                                          for role in ("operation", "goal", "active_part")], axis=-1)
                        count = grounding[("agent_union_pixels", "wrist_union_pixels")[k]]
                        if image.any(-1).sum() != count:
                            raise ValueError("Grounding visibility differs from the stored projection")
                        valid_cameras[k] = bool(grounding["grounding_valid_bits"] & (1 << k))
                        mask = np.full(geom.shape, valid_cameras[k], dtype=bool)
                        image = (image & mask[..., None]).astype(np.float32)
                    images.append(image)
                    masks_list.append(mask)
            values, masks = np.stack(images), np.stack(masks_list)
        return {"target": values.transpose(0, 3, 1, 2), "valid_pixels": masks,
                "valid_cameras": valid_cameras, "source_episode_id": episode,
                "anchor_frame": frame, "target_frame": selected}
