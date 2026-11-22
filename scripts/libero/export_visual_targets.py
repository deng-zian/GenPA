"""Export the selected Appendix B targets for one canonical frame as an NPZ."""
import argparse
from pathlib import Path

import numpy as np

from genpa.datasets.visual_targets.reader import TARGETS, VisualTargets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Root of the extracted GenPA-LIBERO snapshot")
    parser.add_argument("--episode", required=True, help="Canonical ID, e.g. libero_spatial/task_00/demo_0")
    parser.add_argument("--frame", type=int, required=True, help="Zero-based frame on the canonical timeline")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reader = VisualTargets(args.data)
    fields = {"source_episode_id": args.episode, "anchor_frame": args.frame,
              "cameras": np.asarray(["agentview", "robot0_eye_in_hand"])}
    for target in args.targets:
        result = reader.read(target, args.episode, args.frame)
        fields[target] = result["target"]
        for key in ("valid_cameras", "valid_pixels", "target_frame"):
            fields[target + "_" + key] = result[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **fields)
    print(f"Saved {', '.join(args.targets)} for {args.episode}, frame {args.frame}, to {args.output}")


if __name__ == "__main__":
    main()
