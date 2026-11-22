import json
import os
import re
import shutil
from math import ceil
from pathlib import Path
import time

import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from genpa.utils.fs import ensure_dir
from genpa.utils.logging_config import get_logger
from genpa.utils.pytorch_utils import set_global_seed
from genpa.utils.samplers import ResumableEpochSampler

logger = get_logger(__name__)


class PolicyTrainer:
    """Optimizer and checkpoint infrastructure used by RGBDTrainer."""

    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        # Optional dataloader knobs (default: PyTorch defaults)
        self.prefetch_factor = (
            int(cfg.prefetch_factor) if getattr(cfg, "prefetch_factor", None) is not None else None
        )
        self.persistent_workers = bool(getattr(cfg, "persistent_workers", False))
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_weights_start_step = max(
            int(getattr(cfg, "save_weights_start_step", 0) or 0), 0
        )
        self.save_weight_steps = frozenset(
            int(step)
            for step in (getattr(cfg, "save_weight_steps", None) or [])
            if int(step) >= 0
        )
        self.save_final_checkpoint = bool(getattr(cfg, "save_final_checkpoint", True))
        self.keep_latest_state_only = bool(getattr(cfg, "keep_latest_state_only", False))
        self.keep_latest_weights = max(int(getattr(cfg, "keep_latest_weights", 0) or 0), 0)
        self.keep_weight_steps = frozenset(
            int(step)
            for step in (getattr(cfg, "keep_weight_steps", None) or [])
            if int(step) >= 0
        )
        self.eval_every = int(cfg.eval_every)
        self.eval_num_samples = max(int(getattr(cfg, "eval_num_samples", 1)), 1)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self._weights_resume_loaded_before_prepare = False
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        if self.accelerator.is_main_process:
            logger.info(
                "[mem-init] main pid=%d GENPA_MEM_TRIM_EVERY=%s "
                "GENPA_M_TRIM_THRESHOLD=%s GENPA_M_MMAP_THRESHOLD=%s "
                "GENPA_M_TOP_PAD=%s MALLOC_ARENA_MAX=%s",
                os.getpid(),
                os.environ.get("GENPA_MEM_TRIM_EVERY", "<unset>"),
                os.environ.get("GENPA_M_TRIM_THRESHOLD", "<unset>"),
                os.environ.get("GENPA_M_MMAP_THRESHOLD", "<unset>"),
                os.environ.get("GENPA_M_TOP_PAD", "<unset>"),
                os.environ.get("MALLOC_ARENA_MAX", "<unset>"),
            )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        self._load_weights_checkpoint_before_prepare()
        optimizer_group_builder = getattr(self.model, "optimizer_param_groups", None)
        if callable(optimizer_group_builder):
            trainable_params = optimizer_group_builder(default_lr=self.learning_rate)
            if not trainable_params:
                raise ValueError("Model returned no optimizer parameter groups.")
            logger.info(
                "Using model-defined optimizer groups: %s",
                [
                    {
                        "name": group.get("name", f"group_{index}"),
                        "lr": group.get("lr", self.learning_rate),
                        "parameters": sum(parameter.numel() for parameter in group["params"]),
                    }
                    for index, group in enumerate(trainable_params)
                ],
            )
        else:
            trainable_params = [param for param in self.model.dit.parameters() if param.requires_grad]
            proprio_encoder = getattr(self.model, "proprio_encoder", None)
            if proprio_encoder is not None:
                trainable_params.extend(param for param in proprio_encoder.parameters() if param.requires_grad)
            if not trainable_params:
                raise ValueError("No trainable parameters found after applying trainable policy.")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps_cfg = getattr(cfg, "warmup_steps", None)
        warmup_steps = int(total_train_steps * 0.05) if warmup_steps_cfg is None else int(warmup_steps_cfg)
        logger.info(
            "Scheduler setup: type=%s total_train_steps=%d warmup_steps=%d",
            cfg.lr_scheduler_type,
            total_train_steps,
            warmup_steps,
        )
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        wandb_run_id = os.environ.get("WANDB_RUN_ID") or None
        wandb_resume = os.environ.get("WANDB_RESUME") if wandb_run_id is not None else None
        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
            id=wandb_run_id,
            resume=wandb_resume,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s id=%s resume=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
            wandb_run_id,
            wandb_resume,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )
        if self.num_workers > 0:
            if self.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
            # persistent_workers avoids re-forking 24 workers every epoch boundary
            # (which is both slow and visually looks like a memory leak as RSS
            # ramps from ~0 again on each restart).
            loader_kwargs["persistent_workers"] = self.persistent_workers
        if self.accelerator.is_main_process:
            logger.info(
                "[loader] num_workers=%d batch_size=%d pin_memory=%s "
                "prefetch_factor=%s persistent_workers=%s",
                self.num_workers,
                self.batch_size,
                loader_kwargs["pin_memory"],
                loader_kwargs.get("prefetch_factor", "<default=2>"),
                loader_kwargs.get("persistent_workers", False),
            )
        return DataLoader(**loader_kwargs)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    @staticmethod
    def _looks_like_weights_checkpoint(path: Path) -> bool:
        return path.suffix.lower() in {".pt", ".pth", ".bin"} or path.is_file()

    def _load_weights_checkpoint_before_prepare(self):
        resume = self.resume
        if not resume:
            return

        resume_path = Path(str(resume))
        if resume_path.is_dir():
            return
        if not resume_path.exists():
            if self._looks_like_weights_checkpoint(resume_path):
                raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
            return
        if not self._looks_like_weights_checkpoint(resume_path):
            return

        logger.info(
            "Loading weights checkpoint before accelerator.prepare so optimizer/master weights start from checkpoint: %s",
            resume,
        )
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        self._weights_resume_loaded_before_prepare = True
        logger.warning("Loaded weights only; optimizer/scheduler/global step will start fresh.")

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if self._weights_resume_loaded_before_prepare:
            logger.info("Weights checkpoint was loaded before accelerator.prepare; skipping post-prepare weights load.")
            return
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning(
            "Loaded weights after accelerator.prepare; optimizer/master weights may not be synchronized. "
            "Prefer loading weights-only checkpoints before prepare."
        )

    def _set_dit_only_train_mode(self):
        # Reapply the model policy after evaluation or checkpoint restoration.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        apply_policy = getattr(model, "apply_trainable_policy", None)
        if callable(apply_policy):
            apply_policy()
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _prune_old_state_checkpoints(self, keep_step_tag: str):
        if not self.keep_latest_state_only:
            return
        state_root = Path(self.state_dir)
        if not state_root.exists():
            return
        for path in state_root.iterdir():
            if not path.is_dir() or path.name == keep_step_tag:
                continue
            if not re.match(r"step[_-]\d+$", path.name):
                continue
            shutil.rmtree(path)
            logger.info("Removed old training state checkpoint: %s", path)

    def _prune_old_weight_checkpoints(self):
        if self.keep_latest_weights <= 0:
            return
        weights_root = Path(self.weights_dir)
        if not weights_root.exists():
            return
        candidates = []
        for path in weights_root.iterdir():
            match = re.match(r"step[_-](\d+)\.pt$", path.name)
            if path.is_file() and match:
                candidates.append((int(match.group(1)), path))
        candidates.sort(key=lambda item: item[0])
        for step, path in candidates[:-self.keep_latest_weights]:
            if step in self.keep_weight_steps:
                logger.info("Keeping protected weights checkpoint: %s", path)
                continue
            path.unlink()
            logger.info("Removed old weights checkpoint: %s", path)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if (
            self.accelerator.is_main_process
            and self.global_step >= self.save_weights_start_step
            and (
                not self.save_weight_steps
                or self.global_step in self.save_weight_steps
            )
        ):
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
            self._prune_old_weight_checkpoints()
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
            self._prune_old_state_checkpoints(keep_step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )
