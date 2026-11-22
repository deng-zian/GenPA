from __future__ import annotations

import logging
from typing import Any

import torch
from safetensors.torch import load_file as load_sft

from genpa.models.backbones.flux2_imports import ensure_flux2_importable
from genpa.models.backbones.flux2_video_expert import Flux2VideoExpert
from .action_model import (
    ActionAdapter,
    BasePolicy,
    TrainableCore,
    _to_plain_dict,
)

logger = logging.getLogger(__name__)


class ReferencePolicy(BasePolicy):
    """Reference-only FLUX policy followed by the serial action adapter."""

    def _forward_action(
        self,
        *,
        current_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        action_horizon: int,
    ) -> torch.Tensor:
        reference_tokens, reference_ids = self._encode_image_tokens(
            current_image,
            time_value=10.0,
        )
        empty_target_tokens = reference_tokens[:, :0]
        empty_target_ids = reference_ids[:, :0]
        timestep = torch.zeros(
            int(reference_tokens.shape[0]),
            device=reference_tokens.device,
            dtype=reference_tokens.dtype,
        )
        video_pre = self.video_expert.pre_dit(
            x=empty_target_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            ref_image_hidden_states=reference_tokens,
            target_img_ids=empty_target_ids,
            ref_img_ids=reference_ids,
        )
        if int(video_pre["target_len"]) != 0:
            raise RuntimeError(
                f"Reference must not instantiate target tokens, got target_len={video_pre['target_len']}."
            )
        attention_mask = self.dit.build_attention_mask(
            text_mask=video_pre["text_mask"],
            ref_len=int(video_pre["cond_len"]),
            target_len=0,
        )
        tokens_out = self.dit.forward_flux(
            video_pre=video_pre,
            attention_mask=attention_mask,
        )
        reference_length = int(video_pre["cond_len"])
        reference_hidden = tokens_out["img"]
        if reference_hidden.shape[1] != reference_length:
            raise RuntimeError(
                "Reference hidden length mismatch: "
                f"hidden={reference_hidden.shape[1]} cond_len={reference_length}."
            )
        if not getattr(self, "_reference_policy_shape_logged", False):
            logger.info(
                "Reference-token action route: text=%d reference=%d target=%d hidden=%s",
                int(video_pre["txt_len"]),
                reference_length,
                int(video_pre["target_len"]),
                tuple(reference_hidden.shape),
            )
            self._reference_policy_shape_logged = True
        return self.dit.action_adapter(
            reference_hidden=reference_hidden,
            reference_ids=reference_ids,
            proprio=proprio,
            action_horizon=action_horizon,
        )

    def save_checkpoint(self, path: str, optimizer=None, step=None) -> None:
        payload = {
            "reference_policy": self.dit.state_dict(),
            "step": step,
            "architecture": "reference_policy_reference_only_hidden_transformer_adapter",
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path: str, optimizer=None) -> None:
        payload = torch.load(path, map_location="cpu")
        if "reference_policy" not in payload:
            raise ValueError(f"Reference checkpoint is missing the `reference_policy` state: {path}")
        result = self.dit.load_state_dict(payload["reference_policy"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Strict Reference load produced missing={result.missing_keys[:20]} "
                f"unexpected={result.unexpected_keys[:20]}"
            )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        logger.info("Loaded Reference checkpoint %s at step=%s", path, payload.get("step"))


def create_reference_policy(
    *,
    flux2_model_path: str,
    ae_model_path: str,
    qwen3_model_spec: str,
    flux2_src_path: str,
    variant: str = "klein-base-4b",
    qwen_context_len: int = 128,
    prefix_length: int = 16,
    action_adapter_config: Any = None,
    proprio_dim: int = 8,
    flux_learning_rate: float = 1.0e-5,
    adapter_learning_rate: float = 1.0e-4,
    checkpoint_blocks: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    **_: Any,
) -> ReferencePolicy:
    config = _to_plain_dict(action_adapter_config)
    action_horizon = int(config.get("action_horizon", 16))
    action_dim = int(config.get("action_dim", 7))
    hidden_dim = int(config.get("hidden_dim", 512))

    ensure_flux2_importable(flux2_src_path)
    video_expert = Flux2VideoExpert.from_pretrained(
        flux2_model_path=flux2_model_path,
        variant=variant,
        flux2_src_path=flux2_src_path,
        device=device,
        torch_dtype=model_dtype,
    )
    adapter = ActionAdapter(
        flux_hidden_dim=int(video_expert.hidden_dim),
        hidden_dim=hidden_dim,
        num_layers=int(config.get("num_layers", 4)),
        num_heads=int(config.get("num_heads", 8)),
        action_horizon=action_horizon,
        action_dim=action_dim,
        proprio_dim=int(proprio_dim),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device=device, dtype=model_dtype)
    core = TrainableCore(
        video_expert=video_expert,
        text_dim=7680,
        prefix_length=int(prefix_length),
        context_length=int(qwen_context_len),
        action_adapter=adapter,
        checkpoint_blocks=bool(checkpoint_blocks),
    )
    core.action_prefix.to(device=device, dtype=model_dtype)

    from flux2.autoencoder import AutoEncoder, AutoEncoderParams

    with torch.device("meta"):
        vae = AutoEncoder(AutoEncoderParams())
    vae.load_state_dict(load_sft(str(ae_model_path), device=str(device)), strict=True, assign=True)
    vae = vae.to(device=device, dtype=model_dtype).eval()
    vae.requires_grad_(False)

    policy = ReferencePolicy(
        core=core,
        vae=vae,
        qwen3_model_spec=qwen3_model_spec,
        flux2_src_path=flux2_src_path,
        qwen_context_len=int(qwen_context_len),
        action_horizon=action_horizon,
        action_dim=action_dim,
        proprio_dim=int(proprio_dim),
        flux_learning_rate=float(flux_learning_rate),
        adapter_learning_rate=float(adapter_learning_rate),
        device=device,
        torch_dtype=model_dtype,
    )
    flux_parameters = list(policy.video_expert.parameters())
    flux_parameter_ids = {id(parameter) for parameter in flux_parameters}
    flux_count = sum(parameter.numel() for parameter in flux_parameters)
    adapter_count = sum(
        parameter.numel()
        for parameter in policy.dit.parameters()
        if id(parameter) not in flux_parameter_ids
    )
    logger.info(
        "Initialized Reference: FLUX params=%.3fB adapter/prefix params=%.3fM "
        "context=%d prefix=%d horizon=%d target_tokens=0",
        flux_count / 1e9,
        adapter_count / 1e6,
        int(qwen_context_len),
        int(prefix_length),
        action_horizon,
    )
    return policy
