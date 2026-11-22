"""State restoration and file encoders used to build the LIBERO visual replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = ("agentview", "robot0_eye_in_hand")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_noop(action: np.ndarray, previous_kept: np.ndarray | None, threshold: float = 1e-4) -> bool:
    motion_is_zero = np.linalg.norm(action[:-1]) < threshold
    if previous_kept is None:
        return bool(motion_is_zero)
    return bool(motion_is_zero and action[-1] == previous_kept[-1])


def noop_mask(actions: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    mask = np.zeros(len(actions), dtype=bool)
    previous_kept = None
    for index, action in enumerate(actions):
        if is_noop(action, previous_kept, threshold):
            mask[index] = True
            continue
        previous_kept = action
    return mask


def stored_action(actions: np.ndarray) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32).copy()
    result[..., -1] = (result[..., -1] < 0).astype(np.float32)
    return result


def rewrite_model_assets(model_xml: str, libero_root: Path) -> str:
    import robosuite

    libero_assets = (libero_root / "libero/libero/assets").resolve()
    robosuite_assets = (Path(robosuite.__file__).parent / "models/assets").resolve()
    root = ET.fromstring(model_xml)
    asset = root.find("asset")
    if asset is None:
        raise ValueError("MuJoCo XML has no <asset> section")
    elements = list(asset.findall("mesh")) + list(asset.findall("texture"))
    for element in elements:
        old = element.get("file")
        if not old:
            continue
        if "/chiliocosm/assets/" in old:
            new = libero_assets / old.split("/chiliocosm/assets/", 1)[1]
        elif "/libero/libero/assets/" in old:
            new = libero_assets / old.split("/libero/libero/assets/", 1)[1]
        elif "/robosuite/models/assets/" in old:
            new = robosuite_assets / old.split("/robosuite/models/assets/", 1)[1]
        else:
            continue
        if not new.exists():
            raise FileNotFoundError(new)
        element.set("file", str(new))
    return ET.tostring(root, encoding="unicode")


def make_render_env(bddl: Path, gpu_id: int, height: int, width: int):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=height,
        camera_widths=width,
        camera_depths=True,
        camera_segmentations="element",
        render_gpu_device_id=gpu_id,
    )
    env.seed(0)
    return env


def metric_depth_m(env: Any, normalized_depth: np.ndarray) -> np.ndarray:
    from robosuite.utils.camera_utils import get_real_depth_map

    depth = get_real_depth_map(env.sim, normalized_depth.squeeze(-1)).astype(np.float32)
    if not np.all(np.isfinite(depth)) or np.any(depth <= 0):
        raise ValueError("Depth contains non-finite or non-positive values")
    return depth


def geom_metadata(env: Any) -> list[dict[str, Any]]:
    model = env.env.model
    sim_model = env.sim.model
    rows: list[dict[str, Any]] = []
    for geom_id in range(int(sim_model.ngeom)):
        body_id = int(sim_model.geom_bodyid[geom_id])
        joint_start = int(sim_model.body_jntadr[body_id])
        joint_count = int(sim_model.body_jntnum[body_id])
        rows.append(
            {
                "geom_id": geom_id,
                "stored_geom_id": geom_id + 1,
                "geom_name": sim_model.geom_id2name(geom_id),
                "instance_name": model.geom_ids_to_instances.get(geom_id),
                "class_name": model.geom_ids_to_classes.get(geom_id),
                "body_id": body_id,
                "body_name": sim_model.body_id2name(body_id),
                "joint_names": [
                    sim_model.joint_id2name(joint_id)
                    for joint_id in range(joint_start, joint_start + joint_count)
                ],
            }
        )
    return rows


class VideoWriter:
    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: int,
        codec: str,
        crf: int,
        keyframe_interval: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.container = av.open(str(path), "w", format="mp4")
        self.stream = self.container.add_stream(
            codec,
            fps,
            options={"g": str(keyframe_interval), "crf": str(crf)},
        )
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.closed = False

    def write(self, image: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self.closed:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.closed = True

    def abort(self) -> None:
        if self.closed:
            return
        try:
            self.container.close()
        finally:
            self.closed = True


def create_sidecar(path: Path, length: int, state_dim: int, height: int, width: int, attrs: dict[str, Any]) -> h5py.File:
    handle = h5py.File(path, "w")
    for key, value in attrs.items():
        handle.attrs[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
    vector = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    handle.create_dataset("source_frame_index", data=np.arange(length, dtype=np.int32), **vector)
    handle.create_dataset("source_noop_mask", shape=(length,), dtype=np.uint8, **vector)
    handle.create_dataset("source_keep_rank", shape=(length,), dtype=np.int32, **vector)
    handle.create_dataset("action", shape=(length, 7), dtype=np.float32, **vector)
    handle.create_dataset("observation_state", shape=(length, 8), dtype=np.float32, **vector)
    handle.create_dataset("sim_state", shape=(length, state_dim), dtype=np.float64, **vector)
    handle.create_dataset("source_reward", shape=(length,), dtype=np.float32, **vector)
    handle.create_dataset("source_done", shape=(length,), dtype=np.uint8, **vector)
    handle.create_dataset("episode_end", shape=(length,), dtype=np.uint8, **vector)
    handle.create_dataset("task_success", shape=(length,), dtype=np.uint8, **vector)
    dense = {
        "shape": (length, height, width),
        "chunks": (1, height, width),
        "compression": "gzip",
        "compression_opts": 1,
        "shuffle": True,
    }
    for camera in CAMERAS:
        group = handle.create_group(camera)
        group.create_dataset("depth_m", dtype=np.float32, **dense)
        group.create_dataset("geom_id_plus1", dtype=np.uint16, **dense)
        group.create_dataset("camera_to_world_raw", shape=(length, 4, 4), dtype=np.float32, **vector)
        group.create_dataset("intrinsics_raw", shape=(3, 3), dtype=np.float32)
        group.create_dataset("stored_to_raw_pixel", shape=(3, 3), dtype=np.float32)
    return handle


def write_episode_parquet(
    path: Path,
    observation_state: np.ndarray,
    actions: np.ndarray,
    episode_index: int,
    task_index: int,
    global_row_start: int,
    fps: int,
) -> None:
    length = len(actions)
    frame_index = np.arange(length, dtype=np.int64)
    table = pa.table(
        {
            "observation.state": pa.array(
                np.asarray(observation_state, dtype=np.float32).tolist(),
                type=pa.list_(pa.float32(), 8),
            ),
            "action": pa.array(
                np.asarray(actions, dtype=np.float32).tolist(),
                type=pa.list_(pa.float32(), 7),
            ),
            "timestamp": pa.array(frame_index.astype(np.float32) / float(fps), type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
            "index": pa.array(np.arange(global_row_start, global_row_start + length, dtype=np.int64)),
            "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
