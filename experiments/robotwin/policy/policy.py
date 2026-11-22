"""The RoboTwin policy interface; no scene selection or benchmark logic."""

import importlib
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.transforms import functional as F

from experiments.robotwin.prompt import EpisodePrompt


def compact_reference(observation):
    views = []
    cameras = ("head_camera", "left_camera", "right_camera")
    for name, size in zip(cameras, ((192, 256), (96, 128), (96, 128))):
        rgb = np.asarray(observation["observation"][name]["rgb"])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected uint8 HWC RGB for {name}")
        value = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255
        value = F.resize(value, [240, 320], F.InterpolationMode.BILINEAR, antialias=True)
        views.append(F.resize(value, list(size), F.InterpolationMode.BILINEAR, antialias=True))
    return (torch.cat([views[0], torch.cat(views[1:], dim=-1)], dim=-2) * 2 - 1).unsqueeze(0)


class GenPAPolicy:
    def __init__(self, args):
        from genpa.robotwin.depth.contract import Normalization

        config = yaml.safe_load(Path(args["train_config"]).read_text())["model"].copy()
        factory = config.pop("_target_")
        if config.get("mode") != "genpa-depth" or factory != "genpa.robotwin.policy.model.create_policy":
            raise ValueError("Expected a GenPA RoboTwin depth checkpoint")
        config.update(args.get("model_overrides", {}))
        config.update(device="cuda", model_dtype=torch.bfloat16, checkpoint_blocks=False,
                      stats_path=args["dataset_stats_path"])
        module, name = factory.rsplit(".", 1)
        self.model = getattr(importlib.import_module(module), name)(**config)
        self.model.load_checkpoint(args["ckpt_setting"])
        self.model.eval().requires_grad_(False)
        self.normalization = Normalization(args["dataset_stats_path"], mode="absolute_depth")
        self.horizon = self.replan_steps = 16
        self.pending = deque()
        self.text = EpisodePrompt(self._encode)
        self.metrics_path = Path(args["eval_output_dir"]) / "policy_metrics.jsonl"
        self.model._load_text_encoder()
        self.reset()

    @torch.inference_mode()
    def _encode(self, instruction):
        # Raw task text, not ImageWAM's future-video template. Prefix packing is
        # performed by the trained model, with its original tokenizer/mask.
        before = time.perf_counter()
        context, mask = self.model.encode_action_prompt(instruction)
        torch.cuda.synchronize()
        self.encode_seconds += time.perf_counter() - before
        return context.detach(), mask.detach()

    @torch.inference_mode()
    def _chunk(self, observation, instruction):
        state = torch.as_tensor(np.array(observation["joint_action"]["vector"], copy=True),
                                dtype=torch.float32)
        if state.shape != (14,) or not torch.isfinite(state).all():
            raise ValueError("Expected a finite 14D joint state")
        context, mask = self.text.get(instruction)
        start = time.perf_counter()
        prediction = self.model.infer_action_from_packed_context(
            packed_context=context, packed_context_mask=mask,
            input_image=compact_reference(observation).to(self.model.device, self.model.torch_dtype),
            action_horizon=self.horizon, proprio=self.normalization.proprio(state).unsqueeze(0),
        )
        action = self.normalization.decode(prediction["action"].detach().float().cpu().reshape(1, 16, 14), None)
        if action.shape != (1, 16, 14) or not torch.isfinite(action).all():
            raise ValueError("Invalid decoded absolute joint action")
        self.infer_seconds += time.perf_counter() - start
        self.infer_calls += 1
        return action[0].numpy()

    def step(self, env, observation):
        if not self.pending:
            for action in self._chunk(observation, env.get_instruction())[:self.replan_steps]:
                self.pending.append(np.asarray(action, dtype=np.float32))
        env.take_action(self.pending.popleft(), action_type="qpos")
        self.env_steps += 1

    def reset(self):
        self.pending.clear()
        self.text.reset()
        self.encode_seconds = self.infer_seconds = 0.0
        self.infer_calls = self.env_steps = 0
        torch.cuda.reset_peak_memory_stats()

    def print_timing_rollout(self, *, episode_idx, seed, success):
        record = dict(episode_idx=episode_idx, seed=seed, success=bool(success),
                      qwen_calls=self.text.calls, distinct_instructions=len(self.text.values),
                      action_chunk_calls=self.infer_calls, env_steps=self.env_steps,
                      qwen_seconds=self.encode_seconds, action_seconds=self.infer_seconds,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print("[genpa-policy] " + json.dumps(record), flush=True)


def get_model(args):
    return GenPAPolicy(args)


def eval(task_env, model, observation):
    model.step(task_env, observation)


def reset_model(model):
    model.reset()
