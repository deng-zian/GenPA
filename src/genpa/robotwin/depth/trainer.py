"""Reuse RoboTwin optimizer/checkpoint infrastructure with explicit multi-objective updates."""
import json
import time
from pathlib import Path

import torch
from accelerate.utils import DistributedType, send_to_device

from genpa.robotwin.trainer import PolicyTrainer
from genpa.utils.logging_config import get_logger

logger = get_logger(__name__)


class RGBDTrainer(PolicyTrainer):
    def save_checkpoint(self):
        if self.accelerator.state.deepspeed_plugin.zero_stage == 3:
            # ZeRO-3 partitions weights; this exporter requires an unpartitioned ZeRO-1 state.
            raise RuntimeError("Policy export requires the supplied ZeRO-1 configuration")
        return super().save_checkpoint()

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None
        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        device = self.accelerator.device
        values = []
        # Preserve training RNG: periodic diagnostics must not perturb augmentations/flow noise.
        with torch.random.fork_rng(devices=[device.index]):
            generator = torch.Generator().manual_seed(42 + self.global_step + self.accelerator.process_index)
            indices = torch.randint(len(self.val_dataset), (self.eval_num_samples,), generator=generator).tolist()
            for index in indices:
                sample = self.val_dataset[index]
                batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
                         for key, value in sample.items()}
                batch = send_to_device(batch, device)
                with self.accelerator.autocast():
                    loss, _ = self.model(batch, objective="action")
                values.append(loss.detach().float())
        mean = self.accelerator.gather(torch.stack(values)).mean().item()
        self._set_dit_only_train_mode()
        return {"val_loss": mean, "num_samples": len(values) * self.accelerator.num_processes}

    def train(self):
        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            raise RuntimeError("Use the supplied DeepSpeed launcher")
        self._set_dit_only_train_mode()
        engine = self.model
        model = self.accelerator.unwrap_model(engine)
        objectives = ("action", "depth") if model.visual_supervision_enabled else ("action",)
        data_iter = iter(self.train_loader)
        self.run_start_step, self.run_start_time = self.global_step, time.perf_counter()
        micro = 0
        metrics_sum, total_loss, data_wait = {}, 0.0, 0.0
        update_start = time.perf_counter()
        while self.global_step < self.max_steps:
            start = time.perf_counter()
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue
            data_wait += time.perf_counter() - start
            boundary = (micro + 1) % self.gradient_accumulation_steps == 0
            before = engine.global_steps
            for i, objective in enumerate(objectives):
                # Accumulate both objectives before updating. Accelerate.backward would
                # invoke engine.step after each objective on this installed runtime.
                engine.set_gradient_accumulation_boundary(boundary and i == len(objectives) - 1)
                with self.accelerator.autocast():
                    loss, metrics = engine(sample, objective=objective)
                engine.backward(loss)
                total_loss += loss.detach().float().item()
                for key, value in metrics.items():
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value)
                del loss
            engine.step()
            if engine.global_steps != before + int(boundary):
                raise RuntimeError("Exactly one optimizer update per global batch is required")
            micro += 1
            if not boundary:
                continue
            self.global_step += 1
            if not self.accelerator.optimizer_step_was_skipped:
                self.scheduler.step()
            torch.cuda.synchronize(self.accelerator.device)
            elapsed = time.perf_counter() - update_start
            keys = sorted(metrics_sum)
            local = torch.tensor([total_loss, *[metrics_sum[k] for k in keys]], device=self.accelerator.device)
            averaged = self.accelerator.gather(local.unsqueeze(0)).mean(0) / self.gradient_accumulation_steps
            if self.log_every and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                eta, rate = self._estimate_eta()
                gb = self.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes
                payload = {"train/loss": averaged[0].item(), "performance/steps_per_sec": rate,
                           "performance/samples_per_sec": rate * gb, "performance/effective_global_batch": gb,
                           "performance/update_seconds": elapsed, "performance/data_wait_seconds": data_wait}
                payload.update({f"train/{k}": averaged[i + 1].item() for i, k in enumerate(keys)})
                payload.update({f"train/lr_{g.get('name', i)}": float(g["lr"])
                                for i, g in enumerate(self.optimizer.param_groups)})
                logger.info("[rgbd-train] step=%d/%d update_s=%.3f data_s=%.3f loss=%.5f eta=%s",
                            self.global_step, self.max_steps, elapsed, data_wait, averaged[0], eta)
                self._wandb_log(payload)
            if self.eval_every and self.global_step % self.eval_every == 0:
                metrics = self.evaluate()
                if metrics is not None and self.accelerator.is_main_process:
                    logger.info("[rgbd-validation] step=%d %s", self.global_step, metrics)
                    self._wandb_log({f"eval/{k}": v for k, v in metrics.items()})
            final = self.global_step == self.max_steps
            if (self.save_every and self.global_step % self.save_every == 0) or self.global_step in self.save_weight_steps or (final and self.save_final_checkpoint):
                info = self.save_checkpoint()
                if self.accelerator.is_main_process:
                    logger.info("[rgbd-checkpoint] step=%d %s", self.global_step, info)
            metrics_sum, total_loss, data_wait = {}, 0.0, 0.0
            update_start = time.perf_counter()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            marker = {"complete": True, "optimizer_steps": self.global_step, "mode": self.cfg.model.mode,
                      "dataset_revision": model.action_contract["revision"]}
            Path(self.output_dir, "TRAINING_COMPLETE.json").write_text(json.dumps(marker, indent=2) + "\n")
            logger.info("[rgbd-complete] %s", marker)
