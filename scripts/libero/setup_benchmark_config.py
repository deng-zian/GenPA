#!/usr/bin/env python3
"""Create a noninteractive path configuration for LIBERO or LIBERO-Plus."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--datasets", type=Path)
    args = parser.parse_args()

    checkout = args.benchmark_root.expanduser().resolve()
    package = checkout / "libero" / "libero"
    required = {"benchmark_root": package, "assets": package / "assets",
                "bddl_files": package / "bddl_files", "init_states": package / "init_files"}
    for name, path in required.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {name}: {path}; install the benchmark and its assets first")

    config_dir = args.config_dir.expanduser().resolve()
    datasets = (args.datasets or config_dir / "datasets").expanduser().resolve()
    config = {name: str(path) for name, path in required.items()}
    config["datasets"] = str(datasets)
    target = config_dir / "config.yaml"
    if target.exists() and yaml.safe_load(target.read_text()) != config:
        raise FileExistsError(f"A different configuration exists at {target}; choose a new directory")
    config_dir.mkdir(parents=True, exist_ok=True)
    if args.datasets is None:
        datasets.mkdir(exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=True))
    print(f"Created {target}")
    print(f"Set GENPA_LIBERO_CONFIG_PATH={config_dir}")


if __name__ == "__main__":
    main()
