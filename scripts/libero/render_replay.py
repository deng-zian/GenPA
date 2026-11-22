#!/usr/bin/env python3
"""Restore original LIBERO demonstration states and render a canonical replay."""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from genpa.datasets.libero_replay import (
    CAMERAS, VideoWriter, create_sidecar, geom_metadata, make_render_env, metric_depth_m,
    noop_mask, rewrite_model_assets, sha256_file, stored_action, write_episode_parquet,
)

HEIGHT = WIDTH = 256
FPS = 20


def portable_xml(xml, benchmark_root):
    import robosuite
    assets = {
        "libero": (benchmark_root / "libero/libero/assets").resolve(),
        "robosuite": (Path(robosuite.__file__).parent / "models/assets").resolve(),
    }
    tree = ET.fromstring(xml)
    for node in tree.iter():
        value = node.get("file")
        if value and Path(value).is_absolute():
            path = Path(value).resolve()
            for name, root in assets.items():
                if path.is_relative_to(root):
                    node.set("file", str(Path("assets") / name / path.relative_to(root)))
                    break
            else:
                raise ValueError(f"Unrecognized simulator asset: {path.name}")
    return ET.tostring(tree, encoding="utf-8")


def render(row, row_start, raw_root, benchmark, root, gpu):
    import robosuite.utils.transform_utils as transforms
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

    eid = row["suite_episode_index"]
    name = f"episode_{eid:06d}"
    suite_root = root / "canonical" / row["suite"]
    chunk = f"chunk-{eid // 1000:03d}"
    for directory in ("data", "visual_gt", "geom_metadata", "models"):
        (suite_root / directory / chunk).mkdir(parents=True, exist_ok=True)

    with h5py.File(raw_root / row["source_hdf5"], "r") as original:
        demo = original["data"][row["source_demo"]]
        actions, states = demo["actions"][:], demo["states"][:]
        rewards, dones = demo["rewards"][:], demo["dones"][:]
        source_xml = str(demo.attrs["model_file"])

    length = row["frames"]
    if not all(len(data) == length for data in (states, actions, rewards, dones)):
        raise ValueError(f"Source length disagrees with the index for {row['source_episode_id']}")
    xml = rewrite_model_assets(source_xml, benchmark)
    actions_stored = stored_action(actions)
    noops = noop_mask(actions)
    keep_rank = np.full(length, -1, dtype=np.int32)
    keep_rank[~noops] = np.arange((~noops).sum(), dtype=np.int32)
    obs_states = np.empty((length, 8), dtype=np.float32)
    attrs = {
        "schema_version": "genpa-libero-visual-data-v1",
        "source_episode_id": row["source_episode_id"], "task": row["task"],
        "source_hdf5": row["source_hdf5"], "source_demo": row["source_demo"],
        "source_hdf5_sha256": row["source_hdf5_sha256"],
        "source_model_xml_sha256": hashlib.sha256(source_xml.encode()).hexdigest(),
        "bddl": row["bddl"], "bddl_sha256": row["bddl_sha256"], "fps": FPS,
        "image_size": [HEIGHT, WIDTH], "storage_image_transform": "rotate_180",
        "temporal_alignment": "render state[i] and pair it with action[i]",
        "canonical_filtering": "none; every raw action row is retained",
        "depth_encoding": "float32 metric meters",
        "geom_encoding": "uint16 MuJoCo geom_id + 1; 0 reserved for invalid/background",
    }

    with ExitStack() as stack:
        env = make_render_env(benchmark / row["bddl"], gpu, HEIGHT, WIDTH)
        stack.callback(env.close)
        env.reset()
        env.reset_from_xml_string(xml)
        env.sim.reset()
        sidecar = stack.enter_context(create_sidecar(
            suite_root/"visual_gt"/chunk/(name+".hdf5"), length, states.shape[1], HEIGHT, WIDTH, attrs
        ))
        for key, vals in (("action", actions_stored), ("sim_state", states),
                          ("source_reward", rewards), ("source_done", dones),
                          ("source_noop_mask", noops), ("source_keep_rank", keep_rank)):
            sidecar[key][:] = vals
        sidecar["episode_end"][-1] = 1
        writers = {}
        for camera, suffix in zip(CAMERAS, ("image", "wrist_image")):
            path = suite_root/"videos"/chunk/("observation.images."+suffix)/(name+".mp4")
            writer = VideoWriter(path, WIDTH, HEIGHT, FPS, "libsvtav1", 30, 2)
            stack.callback(writer.abort)
            writers[camera] = writer
            group = sidecar[camera]
            group["intrinsics_raw"][:] = get_camera_intrinsic_matrix(env.sim, camera, HEIGHT, WIDTH)
            group["stored_to_raw_pixel"][:] = [[-1, 0, WIDTH-1], [0, -1, HEIGHT-1], [0, 0, 1]]
        for i, state in enumerate(states):
            obs = env.set_init_state(state)
            obs_states[i] = np.concatenate((obs["robot0_eef_pos"],
                                             transforms.quat2axisangle(obs["robot0_eef_quat"]),
                                             obs["robot0_gripper_qpos"]))
            sidecar["observation_state"][i] = obs_states[i]
            sidecar["task_success"][i] = int(env.check_success())
            for camera, writer in writers.items():
                group = sidecar[camera]
                writer.write(obs[f"{camera}_image"][::-1, ::-1])
                group["depth_m"][i] = metric_depth_m(env, obs[f"{camera}_depth"])[::-1, ::-1]
                ids = obs[f"{camera}_segmentation_element"].squeeze(-1).astype(np.int64)
                group["geom_id_plus1"][i] = np.where(ids >= 0, ids + 1, 0)[::-1, ::-1].astype(np.uint16)
                group["camera_to_world_raw"][i] = get_camera_extrinsic_matrix(env.sim, camera)
        for writer in writers.values():
            writer.close()
        geom = {"schema_version": attrs["schema_version"], "source_episode_id": row["source_episode_id"],
                "task": row["task"], "geom_table": geom_metadata(env)}
    (suite_root/"geom_metadata"/chunk/(name+".json")).write_text(json.dumps(geom, indent=2)+"\n")
    (suite_root/"models"/chunk/(name+".xml.gz")).write_bytes(gzip.compress(portable_xml(xml, benchmark), mtime=0))
    write_episode_parquet(suite_root/"data"/chunk/(name+".parquet"), obs_states, actions_stored,
                          eid, row["suite_task_index"], row_start, FPS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path, help="GenPA-LIBERO canonical/episodes.jsonl")
    parser.add_argument("--raw-root", required=True, type=Path, help="Original LIBERO demonstration root")
    parser.add_argument("--benchmark", required=True, type=Path, help="Root of the LIBERO checkout")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode", action="append", help="Source episode ID; repeat for multiple episodes")
    selection.add_argument("--suite", choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"))
    parser.add_argument("--output", required=True, type=Path, help="New directory for the canonical files")
    parser.add_argument("--gpu", type=int, default=0, help="Renderer device index")
    args = parser.parse_args()
    if "GENPA_LIBERO_CONFIG_PATH" in os.environ:
        os.environ.setdefault("LIBERO_CONFIG_PATH", os.environ["GENPA_LIBERO_CONFIG_PATH"])
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ["MUJOCO_GL"] in {"egl", "osmesa"}:
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must name a new directory")
    rows = [json.loads(line) for line in args.index.read_text().splitlines() if line.strip()]
    wanted = set(args.episode or ())
    missing = wanted.difference(row["source_episode_id"] for row in rows)
    if missing:
        parser.error(f"Unknown episodes: {sorted(missing)}")
    selected, starts = [], {}
    for row in rows:
        suite = row["suite"]
        offset = starts.get(suite, 0)
        if row["source_episode_id"] in wanted or suite == args.suite:
            selected.append((row, offset))
        starts[suite] = offset + row["frames"]
    if not selected:
        parser.error("The index contains no matching episodes")
    output.parent.mkdir(parents=True, exist_ok=True)
    verified = {}
    with tempfile.TemporaryDirectory(prefix=".genpa-replay-", dir=output.parent) as tmp:
        root = Path(tmp)/"replay"
        for row, offset in selected:
            for folder, key in ((args.raw_root, "source_hdf5"), (args.benchmark, "bddl")):
                path = folder/row[key]
                if path not in verified:
                    verified[path] = sha256_file(path)
                if verified[path] != row[key+"_sha256"]:
                    raise ValueError(f"Source file differs from the published index: {row[key]}")
            render(row, offset, args.raw_root, args.benchmark.resolve(), root, args.gpu)
            print(row["source_episode_id"], row["frames"], flush=True)
        os.rename(root, output)


if __name__ == "__main__":
    main()
