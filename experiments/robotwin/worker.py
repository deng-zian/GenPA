"""Run an unmodified, pinned upstream evaluator in an isolated working tree."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    os.environ.update(job["compiler_environment"])
    os.chdir(job["runtime_root"])
    sys.path.insert(0, str(Path.cwd()))
    import torch
    import warp
    # Set both the supported environment variable and explicit Warp config before
    # CuRobo import/initialization. Inductor's cache variable does not cover Warp.
    warp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    torch.set_num_threads(job["cpu_threads"])
    print("[compiler-isolation] " + json.dumps(job["compiler_environment"]), flush=True)
    # Do not disable process-wide autograd: the CuRobo expert needs it.
    spec = importlib.util.spec_from_file_location("official_robotwin_eval", Path("script/eval_policy.py"))
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    print("[official-online] " + json.dumps(job["eval_args"]), flush=True)
    official.main(job["eval_args"])


if __name__ == "__main__":
    main()
