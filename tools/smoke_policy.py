#!/usr/bin/env python3
"""Run a GenPA checkpoint on one normalized observation and cached text embedding."""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["libero", "robotwin"], required=True)
    parser.add_argument("--variant", choices=["depth", "action_only"], default="depth")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-npz", type=Path, help="Use an example with image, proprio, context_bf16 and mask")
    parser.add_argument("--output-npy", type=Path, help="Save normalized actions for comparison")
    args = parser.parse_args()
    if args.benchmark == "robotwin" and args.variant != "depth":
        parser.error("The RoboTwin implementation supports the depth policy")

    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("GENPA_FLUX2_SRC", str(root / "third_party/flux2"))
    # Both example formats carry the Qwen context alongside the observation.
    os.environ.setdefault("GENPA_QWEN3", "Qwen/Qwen3-4B")
    os.environ.setdefault("GENPA_ROBOTWIN_STATS", str(root / "configs/robotwin/absolute_stats.json"))
    sys.path[:0] = [str(root / "src"), str(root / "third_party/flux2/src")]
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    torch.set_num_threads(4)
    cfg = OmegaConf.load(root / "configs" / args.benchmark / f"{args.variant}.yaml")
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=args.device)
    model.load_checkpoint(str(args.checkpoint))
    model.eval().requires_grad_(False)
    sample = args.input_npz or root / "examples" / f"{args.benchmark}.npz"
    with np.load(sample, allow_pickle=False) as data:
        context = torch.from_numpy(data["context_bf16"].copy()).view(torch.bfloat16)
        output = model.infer_action(
            prompt=None, input_image=torch.from_numpy(data["image"].copy()), action_horizon=16,
            proprio=torch.from_numpy(data["proprio"].copy()), context=context,
            context_mask=torch.from_numpy(data["mask"].copy()),
        )["action"]
    expected = (16, 7 if args.benchmark == "libero" else 14)
    if tuple(output.shape) != expected or not torch.isfinite(output).all():
        raise ValueError(f"Expected finite actions of shape {expected}, got {tuple(output.shape)}")
    if args.output_npy:
        args.output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_npy, output.detach().cpu().float().numpy())
    print(json.dumps(dict(checkpoint=args.checkpoint.name, benchmark=args.benchmark, variant=args.variant,
                          action_shape=list(output.shape), all_finite=True), indent=2))


if __name__ == "__main__":
    main()
