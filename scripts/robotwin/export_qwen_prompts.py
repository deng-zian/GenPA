#!/usr/bin/env python3
"""Export the processed RoboTwin training prompts for the packed Qwen cache."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


DEPTH_PROMPT = "Generate a full-scene metric depth map for the current observation."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    database = args.language_dir / "tasks.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(database)
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT prompt FROM tasks ORDER BY id").fetchall()
    prompts = list(dict.fromkeys(str(row[0]).strip() for row in rows))
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("Processed task index has no prompts or contains an empty prompt")
    if DEPTH_PROMPT not in prompts:
        prompts.append(DEPTH_PROMPT)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for prompt in prompts:
            stream.write(json.dumps({"task": prompt}, ensure_ascii=False) + "\n")
    print(f"prompts={len(prompts)} output={args.output}")


if __name__ == "__main__":
    main()
