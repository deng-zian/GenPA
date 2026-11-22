import torch
import torch.nn.functional as F

from genpa.models.robotwin.genpa import GenPAPolicy
from genpa.models.robotwin.reference_policy import create_reference_policy
from genpa.robotwin.depth.spatial import CAMERAS
from genpa.robotwin.depth.contract import checkpoint_contract


class DepthPolicy(GenPAPolicy):
    CAMERA_NAMES = CAMERAS
    TASK_NAMES = ("depth",)

    def _camera_token_masks(self, target_ids):
        x = target_ids[..., 2]
        x_min = x.amin(1, keepdim=True)
        x_span = x.amax(1, keepdim=True) - x_min + 1
        if torch.any(x_span.remainder(3) != 0):
            raise ValueError("Three camera strips must each be 16-pixel aligned")
        return [((x >= x_min + camera * x_span / 3) &
                 (x < x_min + (camera + 1) * x_span / 3)).float()
                for camera in range(3)]

    def save_checkpoint(self, path, optimizer=None, step=None):
        if not hasattr(self, "action_contract"):
            raise ValueError("Diagnostic models without frozen statistics cannot be saved as policies")
        payload = {"genpa": self.dit.state_dict(), "step": step,
                   "action_contract": self.action_contract, "lambda_visual": self.lambda_visual}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("action_contract") != self.action_contract:
            raise ValueError("Checkpoint action, dataset, normalization, mode or resolution differs")
        self.dit.load_state_dict(payload["genpa"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])

    def compute_visual_loss(self, sample):
        video = sample["video"]
        if video.ndim != 5 or video.shape[1:3] != (3, 2):
            raise ValueError(f"Expected RGB/Depth video [B,3,2,H,W], got {video.shape}")
        batch = video.shape[0]
        reference, reference_ids = self._encode_image_tokens(video[:, :, 0], time_value=10.0)
        target, target_ids = self._encode_image_tokens(video[:, :, 1], time_value=0.0)
        noise = torch.randn_like(target)
        raw_t = self.visual_scheduler.sample_training_t(batch_size=batch, device=target.device, dtype=target.dtype)
        t = raw_t / float(self.visual_scheduler.num_train_timesteps)
        noisy = self.visual_scheduler.add_noise(target, noise, raw_t)
        velocity = self.visual_scheduler.training_target(target, noise, raw_t)
        context = sample["visual_text_hidden_states"].to(device=target.device, dtype=self.torch_dtype)
        context_mask = sample["visual_text_attention_mask"].to(device=target.device, dtype=torch.bool)
        prepared = self._pre_dit_with_reference_timestep(
            noisy_target=noisy, timestep=t, context=context, context_mask=context_mask,
            reference_tokens=reference, reference_ids=reference_ids, target_ids=target_ids,
        )
        attention = self.dit.build_attention_mask(text_mask=prepared["text_mask"],
                                               ref_len=int(prepared["cond_len"]), target_len=int(prepared["target_len"]))
        hidden = self.dit.forward_flux(video_pre=prepared, attention_mask=attention)
        prediction = self.video_expert.post_dit(hidden, prepared)
        squared = (prediction.float() - velocity.float()).square().mean(dim=-1)
        valid = sample["visual_pixel_valid"].to(device=target.device, dtype=torch.float32)
        if tuple(valid.shape) != (batch, 1, video.shape[-2], video.shape[-1]):
            raise ValueError(f"Invalid visual pixel mask shape: {valid.shape}")
        token_valid = F.avg_pool2d(valid, kernel_size=16, stride=16).flatten(1)
        if token_valid.shape != squared.shape:
            raise ValueError("Pixel mask and FLUX token grid do not align")
        weight = self.visual_scheduler.training_weight(raw_t).float().reshape(-1)
        if weight.numel() == 1:
            weight = weight.expand(batch)
        losses, camera_valid = [], []
        for mask in self._camera_token_masks(target_ids):
            weights = mask * token_valid
            denominator = weights.sum(1)
            losses.append((squared * weights).sum(1) / denominator.clamp_min(1) * weight)
            camera_valid.append((denominator > 0).float())
        losses = torch.stack(losses, 1)
        camera_valid = torch.stack(camera_valid, 1)
        per_sample = (losses * camera_valid).sum(1) / camera_valid.sum(1).clamp_min(1)
        loss = per_sample.mean()
        metrics = {"loss_visual": loss.detach().item(), "visual_valid_camera_fraction": camera_valid.mean().item(),
                   "visual_timestep_mean": t.float().mean().item(), "visual_pixel_valid_fraction": valid.mean().item()}
        metrics.update({f"loss_visual_{camera}": losses[:, i].mean().detach().item() for i, camera in enumerate(CAMERAS)})
        return loss, metrics

    def forward(self, sample, objective="action"):
        if objective == "action":
            return self.compute_action_loss(sample)
        if objective == "depth":
            loss, metrics = self.compute_visual_loss(sample)
            return loss * self.lambda_visual, metrics
        raise ValueError(objective)


def create_policy(lambda_visual=0.1, visual_train_shift=5.0, visual_num_train_timesteps=1000,
                  stats_path=None, height=240, mode="relative_depth", _policy_class=DepthPolicy, **kwargs):
    if stats_path is not None and (float(lambda_visual) == 0.0) != (mode == "relative_action_only"):
        raise ValueError("Mode and visual loss enablement differ")
    contract = checkpoint_contract(stats_path, height, mode) if stats_path is not None else None
    base = create_reference_policy(**kwargs)
    policy = _policy_class(
        core=base.dit, vae=base.vae, qwen3_model_spec=base.qwen3_model_spec,
        flux2_src_path=base.flux2_src_path, qwen_context_len=base.qwen_context_len,
        action_horizon=base.action_horizon, action_dim=base.action_dim, proprio_dim=base.proprio_dim,
        flux_learning_rate=base.flux_learning_rate, adapter_learning_rate=base.adapter_learning_rate,
        device=str(base.device), torch_dtype=base.torch_dtype,
        visual_train_shift=visual_train_shift, visual_num_train_timesteps=visual_num_train_timesteps,
        lambda_visual=lambda_visual,
    )
    if contract is not None:
        policy.action_contract = contract
    return policy
