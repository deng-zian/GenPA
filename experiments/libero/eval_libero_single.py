import hashlib
import json
import inspect
import importlib
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_rollout_video,
)
from genpa.datasets.lerobot.processors.genpa_processor import GenPAProcessor
from genpa.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from genpa.utils.pytorch_utils import set_global_seed
from genpa.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_CLEAN_INSTRUCTION_MANIFEST_CACHE: dict[Path, tuple[str, str, dict[str, dict[str, Any]]]] = {}
_FORBIDDEN_PERTURBATION_IDENTIFIER = re.compile(
    r"(?ix)"
    r"(?:^|[\s_-])"
    r"(?:table|tb|view|initstate|light|noise|layout|add|level|sample|moved)"
    r"(?:[\s_-]*\d+)"
)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _load_clean_instruction_manifest(path: Path) -> tuple[str, str, dict[str, dict[str, Any]]]:
    resolved = path.expanduser().resolve()
    cached = _CLEAN_INSTRUCTION_MANIFEST_CACHE.get(resolved)
    if cached is not None:
        return cached
    if not resolved.is_file():
        raise FileNotFoundError(f"Clean-instruction manifest not found: {resolved}")

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    saved_label = str(payload.get("protocol", "")).strip().lower()
    schema = payload.get("schema_version", 1)
    format_name = {1: "bddl", 2: "clean"}.get(schema)
    # Earlier exports stored versioned labels. Their schema and per-task fields
    # identify the instruction rules independently of the display name.
    legacy_label = saved_label.startswith("clean_") and saved_label.endswith(f"_v{schema}")
    if format_name is None or (saved_label != format_name and not legacy_label):
        raise ValueError(f"Unsupported clean-instruction format in {resolved}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != int(payload.get("task_count", -1)):
        raise ValueError(f"Invalid clean-instruction entry contract in {resolved}")
    if payload.get("clean_instruction_identifier_count") != 0:
        raise ValueError(f"Clean-instruction manifest reports identifier contamination: {resolved}")
    if format_name == "bddl" and payload.get("clean_instruction_digit_task_count") != 0:
        raise ValueError(f"Clean-instruction manifest reports numeric contamination: {resolved}")
    if format_name == "clean" and payload.get("strict_contract_violation_count") != 0:
        raise ValueError(f"Clean instruction manifest contains invalid task descriptions: {resolved}")
    encoded = json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    entries_sha256 = hashlib.sha256(encoded).hexdigest()
    if entries_sha256 != payload.get("entries_sha256"):
        raise ValueError(
            "Clean-instruction manifest fingerprint mismatch: "
            f"{entries_sha256} != {payload.get('entries_sha256')}"
        )

    by_key: dict[str, dict[str, Any]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Invalid clean-instruction entry in {resolved}: {raw_entry!r}")
        key = str(raw_entry.get("key", ""))
        instruction = str(raw_entry.get("clean_instruction", "")).strip()
        if not key or not instruction:
            raise ValueError(f"Empty key or clean instruction in {resolved}: {raw_entry!r}")
        if key in by_key:
            raise ValueError(f"Duplicate clean-instruction key {key!r} in {resolved}")
        forbidden = _FORBIDDEN_PERTURBATION_IDENTIFIER.search(instruction)
        if forbidden is not None:
            raise ValueError(
                f"Clean instruction for {key} still contains perturbation identifier "
                f"{forbidden.group(0)!r}: {instruction!r}"
            )
        is_language = bool(raw_entry.get("is_language_perturbation"))
        if format_name == "clean":
            if is_language:
                if (
                    instruction != str(raw_entry.get("official_instruction", "")).strip()
                    or raw_entry.get("clean_instruction_source")
                    != "official_language_perturbation_instruction"
                ):
                    raise ValueError(
                        f"Clean instructions changed a language perturbation for {key}: {raw_entry!r}"
                    )
            elif (
                instruction
                != str(raw_entry.get("canonical_official_instruction", "")).strip()
                or str(raw_entry.get("official_instruction", "")).strip()
                != str(raw_entry.get("expected_official_instruction", "")).strip()
                or raw_entry.get("clean_instruction_source")
                != "canonical_benchmark_filename_instruction"
            ):
                raise ValueError(
                    f"Incorrect perturbation suffix removal for {key}: {raw_entry!r}"
                )
        by_key[key] = raw_entry

    result = (saved_label, format_name, by_key)
    _CLEAN_INSTRUCTION_MANIFEST_CACHE[resolved] = result
    logging.info("Loaded %d task instructions from %s", len(by_key), resolved)
    return result


def _resolve_task_description(
    task: Any,
    official_task_description: str,
    cfg: DictConfig,
) -> tuple[str, str]:
    protocol = str(cfg.EVALUATION.get("instruction_protocol", "official")).strip().lower()
    if protocol == "official":
        return official_task_description, protocol
    manifest_raw = cfg.EVALUATION.get("clean_instruction_manifest_path", None)
    if _is_none_like(manifest_raw):
        raise ValueError(
            f"EVALUATION.clean_instruction_manifest_path is required for {protocol}."
        )
    saved_label, format_name, manifest = _load_clean_instruction_manifest(Path(str(manifest_raw)))
    if protocol not in {saved_label, format_name}:
        raise ValueError(f"Requested instruction protocol {protocol!r}, manifest uses {format_name!r}")
    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    key = f"{suite_name}:{task_id}"
    entry = manifest.get(key)
    if entry is None:
        raise KeyError(f"Clean-instruction manifest has no entry for {key}")
    if str(entry.get("problem_folder")) != str(task.problem_folder):
        raise ValueError(
            f"Clean-instruction problem_folder mismatch for {key}: "
            f"{entry.get('problem_folder')!r} != {task.problem_folder!r}"
        )
    if str(entry.get("bddl_file")) != str(task.bddl_file):
        raise ValueError(
            f"Clean-instruction BDDL mismatch for {key}: "
            f"{entry.get('bddl_file')!r} != {task.bddl_file!r}"
        )
    if str(entry.get("official_instruction", "")).strip() != official_task_description.strip():
        raise ValueError(
            f"Clean-instruction official-text mismatch for {key}: "
            f"{entry.get('official_instruction')!r} != {official_task_description!r}"
        )
    entry_format = "clean" if "canonical_official_instruction" in entry else "bddl"
    if entry_format != format_name:
        raise ValueError(f"Instruction format differs from the manifest for {key}")
    task_description = str(entry["clean_instruction"]).strip()
    logging.info(
        "Resolved task instruction protocol=%s key=%s source=%s official=%r clean=%r",
        protocol,
        key,
        entry.get("clean_instruction_source"),
        official_task_description,
        task_description,
    )
    return task_description, protocol


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _resolve_tagged_output_dir(raw_output_dir: Path, ckpt_tag: str) -> Path:
    run_ts = raw_output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run timestamp): {raw_output_dir}")

    eval_root = (project_root / "evaluate_results").resolve()
    try:
        relative = raw_output_dir.resolve().relative_to(eval_root)
    except ValueError:
        if raw_output_dir.parent.name == ckpt_tag:
            return raw_output_dir.resolve()
        return (raw_output_dir.parent / ckpt_tag / run_ts).resolve()

    if len(relative.parts) >= 2 and relative.parts[1] == ckpt_tag:
        return raw_output_dir.resolve()

    family = relative.parts[0] if len(relative.parts) > 0 else "libero"
    return (eval_root / family / ckpt_tag / run_ts).resolve()


def _resolve_target(target: str):
    module_name, _, attr_name = target.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid Hydra target: {target}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _filter_model_cfg_for_target(model_cfg: DictConfig) -> DictConfig:
    model_dict = OmegaConf.to_container(model_cfg, resolve=True)
    if not isinstance(model_dict, dict):
        raise ValueError(f"`model` config must resolve to a dict, got {type(model_dict)}")

    target = model_dict.get("_target_")
    if _is_none_like(target):
        return OmegaConf.create(model_dict)

    target_fn = _resolve_target(str(target))
    signature = inspect.signature(target_fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return OmegaConf.create(model_dict)

    allowed_keys = {"_target_"}
    allowed_keys.update(
        name
        for name, param in signature.parameters.items()
        if param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    )
    dropped_keys = sorted(set(model_dict) - allowed_keys)
    if dropped_keys:
        logging.info("Ignoring unsupported model config keys for %s: %s", target, dropped_keys)

    return OmegaConf.create({key: value for key, value in model_dict.items() if key in allowed_keys})


def _load_model_overrides_path(path_value: Any) -> DictConfig | None:
    if _is_none_like(path_value):
        return None
    path = _resolve_path(str(path_value), base=project_root)
    if not path.exists():
        raise FileNotFoundError(f"model_overrides_path not found: {path}")
    loaded = OmegaConf.load(path)
    if not isinstance(OmegaConf.to_container(loaded, resolve=False), dict):
        raise ValueError(f"model_overrides_path must point to a mapping config, got: {path}")
    return loaded


def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    # Keep the full config context while applying overrides so model fields that
    # interpolate into data.* (for example proprio_dim) still resolve correctly.
    cfg_for_resolve = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False))
    model_cfg = cfg_for_resolve.model

    file_overrides = _load_model_overrides_path(cfg.get("model_overrides_path", None))
    if file_overrides is not None:
        model_cfg = OmegaConf.merge(model_cfg, file_overrides)

    inline_overrides = cfg.get("model_overrides", None)
    if not _is_none_like(inline_overrides):
        model_cfg = OmegaConf.merge(model_cfg, inline_overrides)

    cfg_for_resolve.model = model_cfg
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg_for_resolve.model, resolve=True))
    return _filter_model_cfg_for_target(model_cfg)


def _save_resolved_eval_config(cfg: DictConfig, model_cfg: DictConfig, output_file: Path) -> None:
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
    resolved_cfg.model_effective = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=resolved_cfg, f=str(output_file))


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        selected = Path(os.path.expanduser(os.path.expandvars(str(explicit)))).resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"Explicit dataset statistics do not exist: {selected}")
        return selected

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_task_choices(cfg: DictConfig) -> list[tuple[str, int]]:
    task_chunk_file = cfg.EVALUATION.get("task_chunk_file", None)
    if task_chunk_file is None or str(task_chunk_file).strip() == "":
        return [(str(cfg.EVALUATION.task_suite_name), int(cfg.EVALUATION.task_id))]

    chunk_path = _resolve_path(str(task_chunk_file), base=project_root)
    if not chunk_path.exists():
        raise FileNotFoundError(f"EVALUATION.task_chunk_file not found: {chunk_path}")

    choices: list[tuple[str, int]] = []
    with chunk_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if line == "" or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2 or parts[0] == "" or parts[1] == "":
                raise ValueError(
                    f"Invalid task chunk line {line_no} in {chunk_path}: {raw_line.rstrip()!r}. "
                    "Expected format: suite_name,task_id"
                )
            choices.append((parts[0], int(parts[1])))

    if not choices:
        raise ValueError(f"EVALUATION.task_chunk_file is empty: {chunk_path}")
    return choices


def _checkpoint_sha256(path: Path) -> str:
    """Hash once per evaluator process, outside the episode/task loops."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: GenPAProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: GenPAProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: GenPAProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_cached_qwen_context(
    prompt: str,
    model: torch.nn.Module,
    cfg: DictConfig,
    *,
    model_device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = getattr(model, "_libero_eval_qwen_context_cache", None)
    if cache is None:
        cache = {}
        setattr(model, "_libero_eval_qwen_context_cache", cache)
    if prompt in cache:
        return cache[prompt]

    cache_dir_raw = cfg.data.train.get("qwen_text_cache_dir")
    if _is_none_like(cache_dir_raw):
        raise ValueError(
            "EVALUATION.use_qwen_text_cache=true requires data.train.qwen_text_cache_dir."
        )
    context_len = int(cfg.data.train.get("qwen_context_len", model.qwen_context_len))
    cache_dir = _resolve_path(str(cache_dir_raw), base=project_root)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{prompt_hash}.qwen3_flux2_len{context_len}.pt"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing cached Qwen3 context for prompt {prompt!r}: {cache_path}. "
            "Precompute the exact evaluation prompt instead of falling back to an online encoder."
        )

    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    context = payload.get("text_hidden_states")
    context_mask = payload.get("text_attention_mask")
    if not isinstance(context, torch.Tensor) or not isinstance(context_mask, torch.Tensor):
        raise ValueError(
            f"Invalid Qwen3 cache payload in {cache_path}; expected text_hidden_states and "
            "text_attention_mask tensors."
        )
    if context.ndim != 2 or context.shape[0] != context_len:
        raise ValueError(
            f"Cached context must be [L,D] with L={context_len}, got {tuple(context.shape)} "
            f"in {cache_path}."
        )
    if context_mask.ndim != 1 or tuple(context_mask.shape) != (context_len,):
        raise ValueError(
            f"Cached context mask must be [{context_len}], got {tuple(context_mask.shape)} "
            f"in {cache_path}."
        )

    context = context.to(device=model_device, dtype=model.torch_dtype, non_blocking=True)
    context_mask = context_mask.to(device=model_device, dtype=torch.bool, non_blocking=True)
    cache[prompt] = (context, context_mask)
    logging.info(
        "Loaded cached Qwen3 context for LIBERO prompt from %s shape=%s",
        cache_path,
        tuple(context.shape),
    )
    return cache[prompt]


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: GenPAProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[np.ndarray, dict]:
    prompt_template = str(
        cfg.EVALUATION.get(
            "prompt_template",
            cfg.data.train.get("prompt_template", DEFAULT_PROMPT),
        )
    )
    if "{task}" not in prompt_template:
        raise ValueError("prompt_template must contain the '{task}' placeholder")
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "input_image": image,
        "action_horizon": action_horizon,
        "proprio": proprio,
    }
    if bool(cfg.EVALUATION.get("use_qwen_text_cache", False)):
        infer_signature = inspect.signature(model.infer_action).parameters
        if "context" not in infer_signature or "context_mask" not in infer_signature:
            raise TypeError(
                "EVALUATION.use_qwen_text_cache=true requires infer_action(context, context_mask)."
            )
        context, context_mask = _get_cached_qwen_context(
            prompt,
            model,
            cfg,
            model_device=model_device,
        )
        infer_kwargs.update(
            prompt=None,
            context=context,
            context_mask=context_mask,
        )
    else:
        infer_kwargs["prompt"] = prompt
    with torch.no_grad():
        pred = model.infer_action(**infer_kwargs)
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: GenPAProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[bool, list]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))

    env.reset()
    obs = env.set_init_state(initial_state)
    replay_images = []
    pending_actions: list[list[float]] = []

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if done:
            break
        t += 1
    pbar.close()

    return bool(done), replay_images


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: GenPAProcessor,
    cfg: DictConfig,
    video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    env, official_task_description = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        cfg.get("seed"),
    )
    try:
        task_description, instruction_protocol = _resolve_task_description(
            task,
            official_task_description,
            cfg,
        )
        results = {
            "successes": 0,
            "failure_episodes": [],
            "success_episodes": [],
            "task_description": task_description,
        }
        if instruction_protocol != "official":
            results["official_task_description"] = official_task_description
            results["instruction_protocol"] = instruction_protocol
    
        for trial_idx in range(int(cfg.EVALUATION.num_trials)):
            success, replay_images = run_single_episode(
                env=env,
                initial_state=initial_states[trial_idx],
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                episode_idx=trial_idx,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            if success:
                results["successes"] += 1
                results["success_episodes"].append(trial_idx)
            else:
                results["failure_episodes"].append(trial_idx)
    
            if bool(cfg.EVALUATION.get("save_rollout_video", True)):
                save_rollout_video(
                    video_dir,
                    replay_images,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    success=success,
                    task_description=task_description,
                )
    
        return results
    finally:
        env.close()


@hydra.main(version_base="1.3", config_path="../../configs/libero", config_name="eval_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=project_root)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint_sha256 = _checkpoint_sha256(ckpt_path)
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)
    output_dir = _resolve_tagged_output_dir(
        _resolve_path(str(cfg.EVALUATION.output_dir), base=project_root),
        ckpt_tag,
    )
    cfg.EVALUATION.output_dir = str(output_dir)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model_cfg = _resolve_model_cfg(cfg)
    model = instantiate(model_cfg, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(ckpt_path))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: GenPAProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    benchmark_dict = benchmark.get_benchmark_dict()
    task_choices = _load_task_choices(cfg)
    chunk_mode = len(task_choices) > 1
    all_results = []

    for suite_name, task_id in task_choices:
        task_start_time = time.time()
        cfg.EVALUATION.task_suite_name = suite_name
        cfg.EVALUATION.task_id = int(task_id)

        local_log_dir = Path(cfg.EVALUATION.output_dir)
        local_log_dir.mkdir(parents=True, exist_ok=True)
        video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        task_output_dir = local_log_dir / cfg.EVALUATION.task_suite_name
        _save_resolved_eval_config(
            cfg,
            model_cfg,
            task_output_dir / f"eval_config_gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}.yaml",
        )

        task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
        task = task_suite.get_task(cfg.EVALUATION.task_id)
        initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

        if len(initial_states) < int(cfg.EVALUATION.num_trials):
            raise ValueError(
                f"Requested {cfg.EVALUATION.num_trials} initial states, "
                f"but {suite_name}:{task_id} provides only {len(initial_states)}."
            )

        results = {
            "benchmark": str(cfg.EVALUATION.get("benchmark_kind", "libero")),
            "evaluation_seed": int(cfg.seed),
            "checkpoint_sha256": checkpoint_sha256,
            "initial_state_indices": list(range(int(cfg.EVALUATION.num_trials))),
            "instruction_protocol": str(cfg.EVALUATION.get("instruction_protocol", "official")),
            "task_suite": cfg.EVALUATION.task_suite_name,
            "task_id": cfg.EVALUATION.task_id,
            "task_description": None,
            "successes": 0,
            "total_episodes": int(cfg.EVALUATION.num_trials),
            "gpu_id": int(cfg.gpu_id),
            "success_episodes": [],
            "failure_episodes": [],
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": 0,
        }

        logging.info(
            "Running LIBERO evaluation with env_num=1 suite=%s task_id=%s chunk=%s/%s",
            cfg.EVALUATION.task_suite_name,
            cfg.EVALUATION.task_id,
            len(all_results) + 1,
            len(task_choices),
        )
        task_results = run_single_task(
            task=task,
            initial_states=initial_states,
            model=model,
            processor=processor,
            cfg=cfg,
            video_dir=video_dir,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        results.update(task_results)

        results["duration"] = time.time() - (task_start_time if chunk_mode else start_time)
        output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

        print(
            f"Task {cfg.EVALUATION.task_suite_name}/{cfg.EVALUATION.task_id} completed: "
            f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
        )
        print(f"Time taken: {results['duration']:.2f} seconds")
        all_results.append(results)

    return all_results[0] if len(all_results) == 1 else all_results


if __name__ == "__main__":
    eval_single_process()
