"""Current metric-depth supervision aligned with the two LIBERO RGB views."""
from collections import OrderedDict
import json
import os
from pathlib import Path
import h5py
import numpy as np

DEPTH_PROMPT = "Generate a full-scene metric depth map for the current observation."
CAMERAS = ("agentview", "robot0_eye_in_hand")

def encode_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Depth must be finite and strictly positive, in meters")
    x = (np.log(3.5) - np.log(np.clip(depth, 0.04, 3.5))) / (np.log(3.5) - np.log(0.04))
    return np.repeat(np.clip(x, 0, 1)[None], 3, axis=0).astype(np.float32)

class DepthTargetLoader:
    target_type = "depth"
    def __init__(self, config, dataset_dirs):
        if len(dataset_dirs) != 1:
            raise ValueError("Use one merged LIBERO dataset")
        self.root = Path(config["depth_root"])
        self.chunk_size = int(json.loads((Path(dataset_dirs[0])/"meta/info.json").read_text()).get("chunks_size",1000))
        self.max_open = int(config.get("max_open_files",8))
        self.handles = OrderedDict()
        self.pid = os.getpid()
    def close(self):
        for handle in self.handles.values(): handle.close()
        self.handles.clear()
    def __del__(self):
        try:
            if hasattr(self,"handles"): self.close()
        except Exception:
            pass  # h5py may already be finalized during interpreter shutdown.
    @staticmethod
    def build_prompt(task, task_id):
        if task_id != 0: raise ValueError("This release supports current-depth supervision")
        return DEPTH_PROMPT
    def load(self, sample, num_cameras, output_size):
        import torch
        import torchvision.transforms.functional as TF

        if num_cameras != 2: raise ValueError("Expected the agent and wrist cameras")
        if self.pid != os.getpid():
            self.close(); self.pid=os.getpid()
        episode=int(sample["episode_index"]); frame=int(sample["frame_index"])
        path=self.root/f"chunk-{episode//self.chunk_size:03d}/episode_{episode:06d}.hdf5"
        handle=self.handles.pop(str(path),None)
        if handle is None: handle=h5py.File(path,"r")
        self.handles[str(path)]=handle
        while len(self.handles)>self.max_open: self.handles.popitem(last=False)[1].close()
        target=torch.from_numpy(np.stack([encode_depth(handle[c]["depth_m"][frame]) for c in CAMERAS]))
        target=TF.resize(target,list(output_size),interpolation=TF.InterpolationMode.BILINEAR,antialias=True).clamp(0,1)
        return target,np.ones(2,dtype=np.float32),None,0,"depth"
