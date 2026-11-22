#!/usr/bin/env python3
"""Build the LIBERO-Plus Clean-Instruction manifest.

For non-language perturbations, preserve the benchmark's canonical task wording
and remove only the perturbation metadata appended by the benchmark filename.
For language perturbations, preserve the benchmark-provided rewritten language
exactly. BDDL ``(:language ...)`` fields are deliberately not used as a rewrite
source because they are not textually equivalent to the benchmark instruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

from libero.libero import benchmark, get_libero_path


SUITES = ("libero_10", "libero_goal", "libero_spatial", "libero_object")
FORBIDDEN_IDENTIFIER = re.compile(
    r"(?ix)"
    r"(?:^|[\s_-])"
    r"(?:table|tb|view|initstate|light|noise|layout|add|level|sample|moved)"
    r"(?:[\s_-]*\d+)"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def load_canonical_tasks(bddl_root: Path, suite_name: str) -> list[tuple[str, Path, str]]:
    info_path = bddl_root / suite_name / "tasks_info.txt"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    canonical: list[tuple[str, Path, str]] = []
    for raw_line in info_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        filename = Path(raw_line).name
        path = bddl_root / suite_name / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        stem = Path(filename).stem
        official_instruction = benchmark.grab_language_from_filename(
            suite_name,
            filename,
        ).strip()
        if not official_instruction:
            raise ValueError(f"Empty canonical benchmark instruction: {suite_name}:{filename}")
        canonical.append((stem, path, official_instruction))
    if len(canonical) != 10:
        raise ValueError(f"Expected 10 canonical tasks for {suite_name}, got {len(canonical)}")
    return sorted(canonical, key=lambda item: len(item[0]), reverse=True)


def resolve_canonical_task(
    task_stem: str,
    canonical_tasks: list[tuple[str, Path, str]],
) -> tuple[str, Path, str]:
    matches = [
        item
        for item in canonical_tasks
        if task_stem == item[0] or task_stem.startswith(f"{item[0]}_")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one canonical task prefix for {task_stem!r}, "
            f"got {[item[0] for item in matches]}"
        )
    return matches[0]


def metadata_suffix_text(task_stem: str, canonical_stem: str) -> tuple[str, str]:
    if task_stem == canonical_stem:
        return "", ""
    prefix = f"{canonical_stem}_"
    if not task_stem.startswith(prefix):
        raise ValueError(
            f"Task stem does not extend canonical stem: {task_stem!r} vs {canonical_stem!r}"
        )
    suffix = task_stem[len(canonical_stem) :]
    humanized = " ".join(suffix.lstrip("_").split("_"))
    if not humanized:
        raise ValueError(f"Empty perturbation metadata suffix: {task_stem!r}")
    return suffix, humanized


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction-manifest", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--released-prompt-manifest", type=Path,
                        help="Also export exact released instructions for text-cache preparation.")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--expected-task-count", type=int, default=10030)
    args = parser.parse_args()

    if not 0.0 < args.sample_ratio <= 1.0:
        raise ValueError(f"sample_ratio must be in (0, 1], got {args.sample_ratio}")

    benchmark_dict = benchmark.get_benchmark_dict()
    bddl_root = Path(get_libero_path("bddl_files")).resolve()
    entries: list[dict[str, object]] = []

    for suite_name in SUITES:
        task_suite = benchmark_dict[suite_name]()
        canonical_tasks = load_canonical_tasks(bddl_root, suite_name)
        task_ids = list(range(int(task_suite.n_tasks)))
        if args.sample_ratio < 1.0:
            sample_count = max(1, int(math.ceil(len(task_ids) * args.sample_ratio)))
            rng = random.Random(f"{args.sample_seed}:{suite_name}")
            task_ids = sorted(rng.sample(task_ids, sample_count))

        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            official_instruction = str(task.language).strip()
            task_stem = Path(task.bddl_file).stem
            base_stem, base_bddl_path, canonical_official_instruction = resolve_canonical_task(
                task_stem,
                canonical_tasks,
            )
            is_language_perturbation = "_language_" in task_stem.lower()
            if is_language_perturbation:
                clean_instruction = official_instruction
                clean_source = "official_language_perturbation_instruction"
                perturbation_metadata_suffix = ""
                perturbation_metadata_text = ""
                expected_official_instruction = official_instruction
            else:
                perturbation_metadata_suffix, perturbation_metadata_text = metadata_suffix_text(
                    task_stem,
                    base_stem,
                )
                clean_instruction = canonical_official_instruction
                clean_source = "canonical_benchmark_filename_instruction"
                expected_official_instruction = clean_instruction
                if perturbation_metadata_text:
                    expected_official_instruction = (
                        f"{clean_instruction} {perturbation_metadata_text}"
                    )
                if official_instruction != expected_official_instruction:
                    raise ValueError(
                        "Official instruction is not the canonical instruction plus only "
                        f"the filename metadata suffix for {suite_name}:{task_id}: "
                        f"official={official_instruction!r} expected={expected_official_instruction!r}"
                    )

            if is_language_perturbation and clean_instruction != official_instruction:
                raise ValueError(
                    f"Language perturbation changed under Clean-Instruction for {suite_name}:{task_id}"
                )
            forbidden_matches = [match.group(0).strip(" _-") for match in FORBIDDEN_IDENTIFIER.finditer(clean_instruction)]
            if forbidden_matches:
                raise ValueError(
                    f"Clean instruction still contains perturbation identifiers for "
                    f"{suite_name}:{task_id}: {forbidden_matches}: {clean_instruction!r}"
                )

            entries.append(
                {
                    "key": f"{suite_name}:{task_id}",
                    "suite": suite_name,
                    "task_id": task_id,
                    "problem_folder": str(task.problem_folder),
                    "bddl_file": str(task.bddl_file),
                    "canonical_task_stem": base_stem,
                    "canonical_bddl_file": str(base_bddl_path.relative_to(bddl_root)),
                    "canonical_bddl_sha256": sha256_bytes(base_bddl_path.read_bytes()),
                    "is_language_perturbation": is_language_perturbation,
                    "official_instruction": official_instruction,
                    "expected_official_instruction": expected_official_instruction,
                    "canonical_official_instruction": canonical_official_instruction,
                    "clean_instruction": clean_instruction,
                    "clean_instruction_source": clean_source,
                    "perturbation_metadata_suffix": perturbation_metadata_suffix,
                    "perturbation_metadata_text": perturbation_metadata_text,
                }
            )

    if len(entries) != args.expected_task_count:
        raise ValueError(f"Expected {args.expected_task_count} sampled tasks, got {len(entries)}")
    keys = [str(item["key"]) for item in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate suite/task_id keys in clean-instruction manifest")

    clean_prompts = sorted({str(item["clean_instruction"]) for item in entries})
    official_with_identifiers = sum(
        bool(FORBIDDEN_IDENTIFIER.search(str(item["official_instruction"]))) for item in entries
    )
    clean_with_digits = [str(item["key"]) for item in entries if re.search(r"\d", str(item["clean_instruction"]))]
    non_language_entries = [item for item in entries if not bool(item["is_language_perturbation"])]
    language_entries = [item for item in entries if bool(item["is_language_perturbation"])]
    strict_contract_violations = [
        str(item["key"])
        for item in non_language_entries
        if str(item["clean_instruction"]) != str(item["canonical_official_instruction"])
        or str(item["official_instruction"]) != str(item["expected_official_instruction"])
    ]
    language_preservation_violations = [
        str(item["key"])
        for item in language_entries
        if str(item["clean_instruction"]) != str(item["official_instruction"])
    ]
    if strict_contract_violations or language_preservation_violations:
        raise ValueError(
            "Clean-Instruction contract violation: "
            f"non_language={strict_contract_violations[:3]} "
            f"language={language_preservation_violations[:3]}"
        )

    manifest_entries_sha256 = stable_json_sha256(entries)
    instruction_manifest = {
        "schema_version": 2,
        "protocol": "clean",
        "instruction_source": {
            "non_language": (
                "the exact canonical task instruction produced by the benchmark's "
                "filename-to-language function; only appended perturbation metadata is removed"
            ),
            "language": "the exact benchmark-provided rewritten language instruction",
        },
        "suite_order": list(SUITES),
        "sample_ratio": args.sample_ratio,
        "sample_seed": args.sample_seed,
        "task_count": len(entries),
        "unique_clean_instruction_count": len(clean_prompts),
        "non_language_task_count": len(non_language_entries),
        "language_perturbation_task_count": len(language_entries),
        "non_language_exact_suffix_contract_count": len(non_language_entries),
        "language_exact_preservation_count": len(language_entries),
        "strict_contract_violation_count": 0,
        "official_instruction_identifier_count": official_with_identifiers,
        "clean_instruction_identifier_count": 0,
        "clean_instruction_digit_task_count": len(clean_with_digits),
        "clean_instruction_digit_task_keys": clean_with_digits,
        "entries_sha256": manifest_entries_sha256,
        "entries": entries,
    }
    prompt_manifest = {
        "schema_version": 2,
        "protocol": "clean",
        "source_instruction_manifest": str(args.instruction_manifest.resolve()),
        "source_entries_sha256": manifest_entries_sha256,
        "task_result_count": len(entries),
        "count": len(clean_prompts),
        "prompts": clean_prompts,
        "sha256": stable_json_sha256(clean_prompts),
    }

    atomic_write_json(args.instruction_manifest, instruction_manifest)
    atomic_write_json(args.prompt_manifest, prompt_manifest)
    if args.released_prompt_manifest is not None:
        released_prompts = sorted({str(item["official_instruction"]) for item in entries})
        released_manifest = {
            "schema_version": 2,
            "protocol": "official",
            "source_instruction_manifest": str(args.instruction_manifest.resolve()),
            "source_entries_sha256": manifest_entries_sha256,
            "task_result_count": len(entries),
            "count": len(released_prompts),
            "prompts": released_prompts,
            "sha256": stable_json_sha256(released_prompts),
        }
        atomic_write_json(args.released_prompt_manifest, released_manifest)
    print(
        json.dumps(
            {
                "task_count": len(entries),
                "unique_clean_instructions": len(clean_prompts),
                "language_perturbation_tasks": instruction_manifest["language_perturbation_task_count"],
                "official_with_identifiers": official_with_identifiers,
                "clean_with_identifiers": 0,
                "clean_with_digits": len(clean_with_digits),
                "non_language_exact_suffix_contract": len(non_language_entries),
                "language_exact_preservation": len(language_entries),
                "entries_sha256": manifest_entries_sha256,
                "instruction_manifest": str(args.instruction_manifest.resolve()),
                "prompt_manifest": str(args.prompt_manifest.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
