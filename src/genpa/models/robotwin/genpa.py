from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from genpa.models.backbones.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .reference_policy import ReferencePolicy, create_reference_policy

logger = logging.getLogger(__name__)


class GenPAPolicy(ReferencePolicy):
    """Reference action path plus current-depth perception objective."""

    TASK_NAMES = ("depth",)
    CAMERA_NAMES = ("agent", "wrist")

    def __init__(
        self,
        *,
        visual_train_shift: float,
        visual_num_train_timesteps: int,
        lambda_visual: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if float(lambda_visual) < 0.0:
            raise ValueError(f"lambda_visual must be non-negative, got {lambda_visual}.")
        self.visual_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(visual_num_train_timesteps),
            shift=float(visual_train_shift),
        )
        self.lambda_visual = float(lambda_visual)
        self.visual_supervision_enabled = self.lambda_visual > 0.0

    def _pre_dit_with_reference_timestep(
        self,
        *,
        noisy_target: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        reference_tokens: torch.Tensor,
        target_ids: torch.Tensor,
        reference_ids: torch.Tensor,
    ) -> dict[str, Any]:
        video_pre = self.video_expert.pre_dit(
            x=noisy_target,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            ref_image_hidden_states=reference_tokens,
            target_img_ids=target_ids,
            ref_img_ids=reference_ids,
        )
        cond_len = int(video_pre["cond_len"])
        if cond_len <= 0:
            return video_pre

        from flux2.model import (
            _blend_double_mods,
            _blend_single_mods,
            timestep_embedding,
        )

        transformer = self.video_expert.transformer
        reference_timestep = torch.zeros_like(timestep)
        reference_vec = transformer.time_in(
            timestep_embedding(reference_timestep, 256)
        )
        reference_double_mod = transformer.double_stream_modulation_img(
            reference_vec
        )
        reference_single_mod, _ = transformer.single_stream_modulation(
            reference_vec
        )

        t_mod = dict(video_pre["t_mod"])
        image_sequence_length = cond_len + int(video_pre["target_len"])
        full_sequence_length = int(video_pre["txt_len"]) + image_sequence_length
        t_mod["double_img"] = _blend_double_mods(
            t_mod["double_img"],
            reference_double_mod,
            cond_len,
            image_sequence_length,
        )
        t_mod["single"] = _blend_single_mods(
            t_mod["single"],
            reference_single_mod,
            int(video_pre["txt_len"]),
            cond_len,
            full_sequence_length,
        )
        t_mod["ref_vec"] = reference_vec
        video_pre["t_mod"] = t_mod
        return video_pre

    @staticmethod
    def _per_sample_camera_losses(
        *,
        squared_error: torch.Tensor,
        target_ids: torch.Tensor,
        scheduler_weight: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(squared_error.shape[0])
        scheduler_weight = scheduler_weight.reshape(-1)
        if scheduler_weight.numel() == 1 and batch_size > 1:
            scheduler_weight = scheduler_weight.expand(batch_size)
        if scheduler_weight.numel() != batch_size:
            raise ValueError(
                "GenPA scheduler weight must be scalar or [B], got "
                f"{tuple(scheduler_weight.shape)}."
            )

        x_ids = target_ids[..., 2]
        x_min = x_ids.amin(dim=1, keepdim=True)
        x_span = x_ids.amax(dim=1, keepdim=True) - x_min + 1
        if torch.any(torch.remainder(x_span, 2) != 0):
            raise ValueError("GenPA target-token width must be divisible by two cameras.")
        camera_width = x_span / 2
        camera_losses: list[torch.Tensor] = []
        for camera_index in range(2):
            low = x_min + camera_index * camera_width
            high = low + camera_width
            token_mask = (x_ids >= low) & (x_ids < high)
            denominator = (
                token_mask.sum(dim=1).clamp_min(1).to(squared_error.dtype)
                * int(squared_error.shape[-1])
            )
            per_sample = (
                squared_error
                * token_mask[:, :, None].to(dtype=squared_error.dtype)
            ).sum(dim=(1, 2)) / denominator
            camera_losses.append(per_sample * scheduler_weight)
        return torch.stack(camera_losses, dim=1)

    def compute_visual_loss(
        self,
        sample: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        video = sample.get("video")
        if video is None or video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(
                "GenPA expects video [B,3,T,H,W], got "
                f"{None if video is None else tuple(video.shape)}."
            )
        if int(video.shape[2]) != 2:
            raise ValueError(
                "GenPA samples must contain current RGB and one selected visual target, "
                f"got T={int(video.shape[2])}."
            )

        batch_size = int(video.shape[0])
        current_image = video[:, :, 0]
        visual_target_image = video[:, :, 1]
        reference_tokens, reference_ids = self._encode_image_tokens(
            current_image,
            time_value=10.0,
        )
        target_tokens, target_ids = self._encode_image_tokens(
            visual_target_image,
            time_value=0.0,
        )
        noise = torch.randn_like(target_tokens)
        raw_timestep = self.visual_scheduler.sample_training_t(
            batch_size=batch_size,
            device=target_tokens.device,
            dtype=target_tokens.dtype,
        )
        noisy_target = self.visual_scheduler.add_noise(
            target_tokens,
            noise,
            raw_timestep,
        )
        target_velocity = self.visual_scheduler.training_target(
            target_tokens,
            noise,
            raw_timestep,
        )
        unit_timestep = raw_timestep / float(
            self.visual_scheduler.num_train_timesteps
        )

        context = sample.get("visual_text_hidden_states")
        context_mask = sample.get("visual_text_attention_mask")
        if context is None or context_mask is None:
            raise ValueError(
                "GenPA requires visual_text_hidden_states and visual_text_attention_mask."
            )
        context = context.to(
            device=target_tokens.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        context_mask = context_mask.to(
            device=target_tokens.device,
            dtype=torch.bool,
            non_blocking=True,
        )
        if tuple(context.shape[:2]) != (batch_size, self.qwen_context_len):
            raise ValueError(
                "GenPA visual text context must be [B,L,D], got "
                f"{tuple(context.shape)}."
            )

        video_pre = self._pre_dit_with_reference_timestep(
            noisy_target=noisy_target,
            timestep=unit_timestep,
            context=context,
            context_mask=context_mask,
            reference_tokens=reference_tokens,
            target_ids=target_ids,
            reference_ids=reference_ids,
        )
        attention_mask = self.dit.build_attention_mask(
            text_mask=video_pre["text_mask"],
            ref_len=int(video_pre["cond_len"]),
            target_len=int(video_pre["target_len"]),
        )
        tokens_out = self.dit.forward_flux(
            video_pre=video_pre,
            attention_mask=attention_mask,
        )
        prediction = self.video_expert.post_dit(tokens_out, video_pre)
        squared_error = F.mse_loss(
            prediction.float(),
            target_velocity.float(),
            reduction="none",
        )
        scheduler_weight = self.visual_scheduler.training_weight(
            raw_timestep
        ).to(device=squared_error.device, dtype=squared_error.dtype)
        camera_losses = self._per_sample_camera_losses(
            squared_error=squared_error,
            target_ids=target_ids,
            scheduler_weight=scheduler_weight,
        )

        camera_valid = torch.as_tensor(
            sample.get("visual_loss_valid"),
            device=camera_losses.device,
            dtype=camera_losses.dtype,
        )
        if tuple(camera_valid.shape) != (batch_size, 2):
            raise ValueError(
                f"GenPA visual_loss_valid must be [B,2], got {tuple(camera_valid.shape)}."
            )
        camera_valid = camera_valid.clamp(0.0, 1.0)
        valid_camera_count = camera_valid.sum(dim=1)
        per_sample_visual = torch.where(
            valid_camera_count > 0,
            (camera_losses * camera_valid).sum(dim=1)
            / valid_camera_count.clamp_min(1.0),
            camera_losses.sum(dim=1) * 0.0,
        )
        loss = per_sample_visual.mean()

        task_ids = torch.as_tensor(
            sample.get("visual_task_id"),
            device=loss.device,
            dtype=torch.long,
        ).reshape(-1)
        if tuple(task_ids.shape) != (batch_size,) or bool((task_ids != 0).any()):
            raise ValueError(f"GenPA visual_task_id must be [B] with depth id 0, got {task_ids}.")

        metrics: dict[str, float] = {
            "loss_visual": float(loss.detach().item()),
            "visual_valid_camera_fraction": float(camera_valid.detach().mean().item()),
            "visual_valid_sample_fraction": float(
                (valid_camera_count > 0).detach().float().mean().item()
            ),
            "visual_timestep_mean": float(unit_timestep.detach().float().mean().item()),
        }
        for camera_index, camera_name in enumerate(self.CAMERA_NAMES):
            valid = camera_valid[:, camera_index]
            component = torch.where(
                valid.sum() > 0,
                (camera_losses[:, camera_index] * valid).sum()
                / valid.sum().clamp_min(1.0),
                camera_losses[:, camera_index].sum() * 0.0,
            )
            metrics[f"loss_visual_{camera_name}"] = float(component.detach().item())
        for task_id, task_name in enumerate(self.TASK_NAMES):
            selected = (task_ids == task_id).to(per_sample_visual.dtype)
            selected_count = selected.sum()
            task_loss = torch.where(
                selected_count > 0,
                (per_sample_visual * selected).sum() / selected_count.clamp_min(1.0),
                per_sample_visual.sum() * 0.0,
            )
            valid_selected = torch.where(
                selected_count > 0,
                (((valid_camera_count > 0).to(selected.dtype)) * selected).sum()
                / selected_count.clamp_min(1.0),
                selected.sum() * 0.0,
            )
            metrics[f"loss_visual_{task_name}"] = float(task_loss.detach().item())
            metrics[f"visual_{task_name}_fraction"] = float(
                selected.detach().mean().item()
            )
            metrics[f"visual_{task_name}_valid_fraction"] = float(
                valid_selected.detach().item()
            )
        return loss, metrics

    def compute_action_loss(
        self,
        sample: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return super().training_loss(sample)

    def iter_training_losses(self, sample: dict[str, Any]):
        loss_action, action_metrics = self.compute_action_loss(sample)
        action_value = float(loss_action.detach().item())
        if not self.visual_supervision_enabled:
            yield loss_action, {
                **action_metrics,
                "loss_visual": 0.0,
                "loss_visual_weighted": 0.0,
                "loss_total": action_value,
                "visual_supervision_enabled": 0.0,
            }
            return

        yield loss_action, action_metrics
        del loss_action

        loss_visual, visual_metrics = self.compute_visual_loss(sample)
        weighted_visual = self.lambda_visual * loss_visual
        weighted_visual_value = float(weighted_visual.detach().item())
        yield weighted_visual, {
            **visual_metrics,
            "loss_visual_weighted": weighted_visual_value,
            "loss_total": action_value + weighted_visual_value,
        }

    def training_loss(self, sample: dict[str, Any], tiled: bool = False):
        del tiled
        loss_action, action_metrics = self.compute_action_loss(sample)
        if not self.visual_supervision_enabled:
            return loss_action, {
                **action_metrics,
                "loss_visual": 0.0,
                "loss_visual_weighted": 0.0,
                "loss_total": float(loss_action.detach().item()),
                "visual_supervision_enabled": 0.0,
            }

        loss_visual, visual_metrics = self.compute_visual_loss(sample)
        loss = loss_action + self.lambda_visual * loss_visual
        return loss, {
            **action_metrics,
            **visual_metrics,
            "loss_visual_weighted": self.lambda_visual
            * float(loss_visual.detach().item()),
            "loss_total": float(loss.detach().item()),
        }

    def save_checkpoint(self, path: str, optimizer=None, step=None) -> None:
        payload = {
            "genpa": self.dit.state_dict(),
            "step": step,
            "architecture": (
                "genpa_reference_policy_action_plus_sampled_depth_visual_flow"
                if self.visual_supervision_enabled
                else "genpa_reference_policy_action_only"
            ),
            "lambda_visual": self.lambda_visual,
            "visual_tasks": self.TASK_NAMES,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path: str, optimizer=None) -> None:
        payload = torch.load(path, map_location="cpu")
        if "genpa" not in payload:
            raise ValueError(f"GenPA checkpoint is missing the `genpa` state: {path}")
        result = self.dit.load_state_dict(payload["genpa"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Strict GenPA load produced missing={result.missing_keys[:20]} "
                f"unexpected={result.unexpected_keys[:20]}"
            )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        logger.info(
            "Loaded GenPA checkpoint %s at step=%s lambda_visual=%s",
            path,
            payload.get("step"),
            payload.get("lambda_visual"),
        )


def create_genpa(
    *,
    visual_train_shift: float = 5.0,
    visual_num_train_timesteps: int = 1000,
    lambda_visual: float = 0.1,
    **kwargs: Any,
) -> GenPAPolicy:
    base = create_reference_policy(**kwargs)
    policy = GenPAPolicy(
        core=base.dit,
        vae=base.vae,
        qwen3_model_spec=base.qwen3_model_spec,
        flux2_src_path=base.flux2_src_path,
        qwen_context_len=base.qwen_context_len,
        action_horizon=base.action_horizon,
        action_dim=base.action_dim,
        proprio_dim=base.proprio_dim,
        flux_learning_rate=base.flux_learning_rate,
        adapter_learning_rate=base.adapter_learning_rate,
        device=str(base.device),
        torch_dtype=base.torch_dtype,
        visual_train_shift=float(visual_train_shift),
        visual_num_train_timesteps=int(visual_num_train_timesteps),
        lambda_visual=float(lambda_visual),
    )
    logger.info(
        "Initialized GenPA from Reference: visual_shift=%.3f visual_steps=%d "
        "lambda_visual=%.6f tasks=%s",
        float(visual_train_shift),
        int(visual_num_train_timesteps),
        float(lambda_visual),
        GenPAPolicy.TASK_NAMES,
    )
    return policy
