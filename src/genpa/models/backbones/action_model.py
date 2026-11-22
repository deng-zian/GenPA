from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch.utils.checkpoint import checkpoint

from .flux2_imports import ensure_flux2_importable
from .flux2_video_expert import Flux2VideoExpert
from .mot import MoT

logger = logging.getLogger(__name__)


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dict-like config, got {type(value)}")
    return dict(value)


def _sinusoidal_1d(position: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"Sinusoidal dimension must be even, got {dim}.")
    half = dim // 2
    omega = torch.arange(half, device=position.device, dtype=torch.float32)
    omega = torch.exp(-math.log(10000.0) * omega / max(half - 1, 1))
    phase = position.float().unsqueeze(-1) * omega
    return torch.cat([phase.sin(), phase.cos()], dim=-1)


def build_2d_sincos_position(reference_ids: torch.Tensor, dim: int) -> torch.Tensor:
    """Build explicit 2D position features from FLUX row/column token ids."""
    if reference_ids.ndim != 3 or reference_ids.shape[-1] != 4:
        raise ValueError(f"`reference_ids` must be [B,N,4], got {tuple(reference_ids.shape)}")
    if dim % 2 != 0:
        raise ValueError(f"2D positional dimension must be even, got {dim}.")
    half = dim // 2
    row = _sinusoidal_1d(reference_ids[..., 1], half)
    col = _sinusoidal_1d(reference_ids[..., 2], half)
    return torch.cat([row, col], dim=-1)


class ActionAdapter(nn.Module):
    """Lightweight decoder from final FLUX reference tokens to one action chunk."""

    def __init__(
        self,
        *,
        flux_hidden_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        action_horizon: int,
        action_dim: int,
        proprio_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.hidden_dim = int(hidden_dim)

        self.memory_norm = nn.LayerNorm(int(flux_hidden_dim))
        self.memory_projection = nn.Linear(int(flux_hidden_dim), self.hidden_dim)
        self.memory_post_norm = nn.LayerNorm(self.hidden_dim)
        self.action_queries = nn.Embedding(self.action_horizon, self.hidden_dim)
        self.proprio_mlp = nn.Sequential(
            nn.LayerNorm(self.proprio_dim),
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.decoder = nn.ModuleList(
            [
                ActionDecoderBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.decoder_norm = nn.LayerNorm(self.hidden_dim)
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        nn.init.normal_(self.action_queries.weight, mean=0.0, std=0.02)

    def forward(
        self,
        *,
        reference_hidden: torch.Tensor,
        reference_ids: torch.Tensor,
        proprio: torch.Tensor,
        action_horizon: Optional[int] = None,
    ) -> torch.Tensor:
        if reference_hidden.ndim != 3:
            raise ValueError(f"`reference_hidden` must be [B,N,D], got {tuple(reference_hidden.shape)}")
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` must be [B,{self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        horizon = self.action_horizon if action_horizon is None else int(action_horizon)
        if horizon <= 0 or horizon > self.action_horizon:
            raise ValueError(f"action_horizon must be in [1,{self.action_horizon}], got {horizon}.")

        memory = self.memory_projection(self.memory_norm(reference_hidden))
        position = build_2d_sincos_position(reference_ids, self.hidden_dim).to(
            device=memory.device,
            dtype=memory.dtype,
        )
        memory = self.memory_post_norm(memory + position)

        query_ids = torch.arange(horizon, device=reference_hidden.device)
        queries = self.action_queries(query_ids)
        queries = queries.unsqueeze(0).expand(reference_hidden.shape[0], -1, -1)
        queries = queries + self.proprio_mlp(proprio).unsqueeze(1)
        decoded = queries
        for block in self.decoder:
            decoded = block(decoded, memory)
        decoded = self.decoder_norm(decoded)
        return self.action_head(decoded)


class ActionAttention(nn.Module):
    """Explicit attention whose projections remain ZeRO-3 gather points."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_length, _ = query.shape
        key_length = key_value.shape[1]
        query_states = self.q_proj(query).view(
            batch_size, query_length, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(key_value).view(
            batch_size, key_length, self.num_heads, self.head_dim
        )
        value_states = self.v_proj(key_value).view(
            batch_size, key_length, self.num_heads, self.head_dim
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, query_length, self.hidden_dim)
        return self.out_proj(attended)


class ActionDecoderBlock(nn.Module):
    """Pre-normalized action self-attention, FLUX cross-attention, and MLP."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = ActionAttention(hidden_dim, num_heads, dropout)
        self.cross_attention = ActionAttention(hidden_dim, num_heads, dropout)
        mlp_dim = int(round(hidden_dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.residual_dropout(
            self.self_attention(self.self_norm(hidden_states), self.self_norm(hidden_states))
        )
        hidden_states = hidden_states + self.residual_dropout(
            self.cross_attention(self.cross_norm(hidden_states), self.memory_norm(memory))
        )
        hidden_states = hidden_states + self.residual_dropout(self.mlp(self.mlp_norm(hidden_states)))
        return hidden_states


class TrainableCore(nn.Module):
    """Trainable FLUX backbone, action prefix, and serial action adapter."""

    def __init__(
        self,
        *,
        video_expert: Flux2VideoExpert,
        text_dim: int,
        prefix_length: int,
        context_length: int,
        action_adapter: ActionAdapter,
        checkpoint_blocks: bool,
    ):
        super().__init__()
        self.mot = MoT(
            mixtures={"video": video_expert},
            mot_checkpoint_mixed_attn=False,
            gqa_implementation="repeat",
            force_flash_attention=False,
        )
        self.action_adapter = action_adapter
        self.prefix_length = int(prefix_length)
        self.context_length = int(context_length)
        self.text_dim = int(text_dim)
        self.checkpoint_blocks = bool(checkpoint_blocks)
        if self.prefix_length <= 0 or self.prefix_length >= self.context_length:
            raise ValueError(
                f"prefix_length must be in [1,{self.context_length - 1}], got {self.prefix_length}."
            )
        self.action_prefix = nn.Embedding(self.prefix_length, self.text_dim)
        nn.init.normal_(self.action_prefix.weight, mean=0.0, std=1.0)

    @property
    def video_expert(self) -> Flux2VideoExpert:
        return self.mot.mixtures["video"]

    def pack_action_prefix(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[:2] != context_mask.shape or context.shape[-1] != self.text_dim:
            raise ValueError(
                f"Context shape mismatch: context={tuple(context.shape)}, "
                f"mask={tuple(context_mask.shape)}, text_dim={self.text_dim}."
            )
        mask = context_mask.to(device=context.device, dtype=torch.bool)
        if bool((mask[:, 1:] & ~mask[:, :-1]).any().item()):
            raise ValueError("Action requires a right-padded contiguous text attention mask.")

        max_task_length = self.context_length - self.prefix_length
        valid_lengths = mask.sum(dim=1)
        if int(valid_lengths.max().item()) > max_task_length:
            raise ValueError(
                f"Task text uses {int(valid_lengths.max().item())} valid tokens, but only "
                f"{max_task_length} remain after the {self.prefix_length}-token action prefix."
            )

        batch_size = int(context.shape[0])
        packed = context.new_zeros(batch_size, self.context_length, self.text_dim)
        packed_mask = torch.zeros(
            batch_size,
            self.context_length,
            device=context.device,
            dtype=torch.bool,
        )
        prefix_ids = torch.arange(self.prefix_length, device=context.device)
        prefix = self.action_prefix(prefix_ids).to(dtype=context.dtype)
        packed[:, : self.prefix_length] = prefix.unsqueeze(0)
        packed_mask[:, : self.prefix_length] = True
        for batch_index, valid_length_tensor in enumerate(valid_lengths):
            valid_length = int(valid_length_tensor.item())
            if valid_length <= 0:
                continue
            start = self.prefix_length
            end = start + valid_length
            packed[batch_index, start:end] = context[batch_index, :valid_length]
            packed_mask[batch_index, start:end] = True
        return packed, packed_mask

    @staticmethod
    def build_attention_mask(
        *,
        text_mask: torch.Tensor,
        ref_len: int,
        target_len: int,
    ) -> dict[str, torch.Tensor]:
        batch_size, text_len = text_mask.shape
        prefix_end = int(text_len) + int(ref_len)
        total = prefix_end + int(target_len)
        mask = torch.zeros(
            batch_size,
            total,
            total,
            dtype=torch.bool,
            device=text_mask.device,
        )
        # Both streams use GenPA's asymmetric mask: text/reference tokens never
        # read target tokens, whereas target tokens read the entire sequence.
        # The action pass contains only text/reference tokens (target_len=0).
        mask[:, :prefix_end, :prefix_end] = True
        mask[:, prefix_end:total, :total] = True
        mask[:, :, :text_len] &= text_mask[:, None, :]
        return {"double_joint": mask, "single": mask.clone()}

    def _run_double_block(
        self,
        *,
        block: nn.Module,
        img: torch.Tensor,
        txt: torch.Tensor,
        img_pe: torch.Tensor,
        txt_pe: torch.Tensor,
        double_img: torch.Tensor,
        double_txt: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flux2.model import apply_rope

        q, k, v, pe_full, num_txt, mods = block._prepare_qkv(
            img,
            txt,
            img_pe,
            txt_pe,
            double_img,
            double_txt,
        )
        q, k = apply_rope(q, k, pe_full)
        mixed = self.mot._mixed_attention(
            self.mot._flux2_flatten_heads(q),
            self.mot._flux2_flatten_heads(k),
            self.mot._flux2_flatten_heads(v),
            attention_mask,
        )
        txt_attn, img_attn = torch.split(mixed, [num_txt, img.shape[1]], dim=1)
        return block._apply_residuals(img, txt, img_attn, txt_attn, mods)

    def _run_single_block(
        self,
        *,
        block: nn.Module,
        stream: torch.Tensor,
        stream_pe: torch.Tensor,
        single_mod: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        state = self.mot._flux2_video_single_io(block, stream, stream_pe, single_mod)
        mixed = self.mot._mixed_attention(
            state["q"],
            state["k"],
            state["v"],
            attention_mask,
        )
        return block._out(state["residual_x"], mixed, state["mlp"], state["gate"])

    def forward_flux(
        self,
        *,
        video_pre: dict[str, Any],
        attention_mask: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        expert = self.video_expert
        txt = video_pre["tokens"]["txt"]
        img = video_pre["tokens"]["img"]
        txt_pe = video_pre["freqs"]["txt"]
        img_pe = video_pre["freqs"]["img"]
        t_mod = video_pre["t_mod"]

        for layer_index in range(expert.double_layers):
            block = expert.double_blocks[layer_index]

            def double_forward(
                img_in: torch.Tensor,
                txt_in: torch.Tensor,
                _block: nn.Module = block,
            ):
                return self._run_double_block(
                    block=_block,
                    img=img_in,
                    txt=txt_in,
                    img_pe=img_pe,
                    txt_pe=txt_pe,
                    double_img=t_mod["double_img"],
                    double_txt=t_mod["double_txt"],
                    attention_mask=attention_mask["double_joint"],
                )

            if self.checkpoint_blocks and self.training:
                img, txt = checkpoint(double_forward, img, txt, use_reentrant=False)
            else:
                img, txt = double_forward(img, txt)

        stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        for layer_index in range(expert.single_layers):
            block = expert.single_blocks[layer_index]

            def single_forward(
                stream_in: torch.Tensor,
                _block: nn.Module = block,
            ):
                return self._run_single_block(
                    block=_block,
                    stream=stream_in,
                    stream_pe=stream_pe,
                    single_mod=t_mod["single"],
                    attention_mask=attention_mask["single"],
                )

            if self.checkpoint_blocks and self.training:
                stream = checkpoint(single_forward, stream, use_reentrant=False)
            else:
                stream = single_forward(stream)

        text_length = int(txt.shape[1])
        return {
            "txt": stream[:, :text_length],
            "img": stream[:, text_length:],
        }


class BasePolicy(nn.Module):
    """One FLUX forward followed by a lightweight serial action decoder."""

    def __init__(
        self,
        *,
        core: TrainableCore,
        vae: nn.Module,
        qwen3_model_spec: str,
        flux2_src_path: str,
        qwen_context_len: int,
        action_horizon: int,
        action_dim: int,
        proprio_dim: int,
        flux_learning_rate: float,
        adapter_learning_rate: float,
        device: str,
        torch_dtype: torch.dtype,
    ):
        super().__init__()
        self.dit = core
        self.vae = vae
        self.text_encoder = None
        self.qwen3_model_spec = str(qwen3_model_spec)
        self.flux2_src_path = str(flux2_src_path)
        self.qwen_context_len = int(qwen_context_len)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.flux_learning_rate = float(flux_learning_rate)
        self.adapter_learning_rate = float(adapter_learning_rate)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.vae.eval()
        self.vae.requires_grad_(False)

    @property
    def video_expert(self) -> Flux2VideoExpert:
        return self.dit.video_expert

    def apply_trainable_policy(self) -> None:
        self.dit.train()
        self.dit.requires_grad_(True)
        self.vae.eval()
        self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.model.eval()
            self.text_encoder.model.requires_grad_(False)

    def optimizer_param_groups(self, default_lr: float) -> list[dict[str, Any]]:
        del default_lr
        flux_parameters = [parameter for parameter in self.video_expert.parameters() if parameter.requires_grad]
        flux_parameter_ids = {id(parameter) for parameter in flux_parameters}
        adapter_parameters = [
            parameter
            for parameter in self.dit.parameters()
            if parameter.requires_grad and id(parameter) not in flux_parameter_ids
        ]
        if not flux_parameters or not adapter_parameters:
            raise ValueError(
                f"Action optimizer groups are empty: flux={len(flux_parameters)}, "
                f"adapter={len(adapter_parameters)}"
            )
        return [
            {
                "params": adapter_parameters,
                "lr": self.adapter_learning_rate,
                "name": "adapter_and_prefix",
            },
            {
                "params": flux_parameters,
                "lr": self.flux_learning_rate,
                "name": "flux",
            },
        ]

    @torch.no_grad()
    def _encode_image_tokens(
        self,
        image: torch.Tensor,
        *,
        time_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        if image.shape[-2] % 16 != 0 or image.shape[-1] % 16 != 0:
            raise ValueError(f"FLUX image size must be divisible by 16, got {tuple(image.shape[-2:])}")
        image = image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        latents = self.vae.encode(image).to(dtype=self.torch_dtype)
        tokens = Flux2VideoExpert.pack_latents(latents)
        ids = Flux2VideoExpert.build_img_ids(
            batch_size=int(latents.shape[0]),
            token_height=int(latents.shape[-2]),
            token_width=int(latents.shape[-1]),
            time_value=float(time_value),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens, ids

    def _prepare_proprio(self, proprio: torch.Tensor, batch_size: int) -> torch.Tensor:
        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        elif proprio.ndim == 2:
            if proprio.shape[0] != batch_size and batch_size == 1:
                proprio = proprio[:1]
        elif proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        else:
            raise ValueError(f"`proprio` must be [B,T,D], [B,D], or [D], got {tuple(proprio.shape)}")
        if tuple(proprio.shape) != (batch_size, self.proprio_dim):
            raise ValueError(
                f"`proprio` must resolve to {(batch_size, self.proprio_dim)}, got {tuple(proprio.shape)}"
            )
        return proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

    def _prepare_cached_text(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        context = sample.get("text_hidden_states")
        context_mask = sample.get("text_attention_mask")
        if context is None or context_mask is None:
            raise ValueError("Action training requires cached text_hidden_states and text_attention_mask.")
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return self.dit.pack_action_prefix(context, context_mask)

    def _forward_action(
        self,
        *,
        current_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        action_horizon: int,
    ) -> torch.Tensor:
        """Implemented by the reference-token action policy."""
        raise NotImplementedError

    @staticmethod
    def _masked_action_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_dim_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        squared_error = F.mse_loss(prediction.float(), target.float(), reduction="none")
        valid = torch.ones_like(squared_error)
        if action_is_pad is not None:
            valid = valid * (~action_is_pad).to(device=valid.device, dtype=valid.dtype).unsqueeze(-1)
        if action_dim_is_pad is not None:
            valid = valid * (~action_dim_is_pad).to(device=valid.device, dtype=valid.dtype).unsqueeze(1)
        valid_count = valid.flatten(1).sum(dim=1).clamp(min=1.0)
        return (squared_error * valid).flatten(1).sum(dim=1) / valid_count

    def training_loss(self, sample: dict[str, Any], tiled: bool = False):
        del tiled
        video = sample.get("video")
        action = sample.get("action")
        proprio = sample.get("proprio")
        if video is None or video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Action expects video [B,3,T,H,W], got {None if video is None else tuple(video.shape)}")
        if action is None or action.ndim != 3:
            raise ValueError(f"Action expects action [B,H,D], got {None if action is None else tuple(action.shape)}")
        if action.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"Action action shape must end in {(self.action_horizon, self.action_dim)}, "
                f"got {tuple(action.shape)}"
            )
        if proprio is None:
            raise ValueError("Action requires proprioception.")

        batch_size = int(video.shape[0])
        context, context_mask = self._prepare_cached_text(sample)
        current_image = video[:, :, 0].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        proprio_first = self._prepare_proprio(proprio, batch_size)
        prediction = self._forward_action(
            current_image=current_image,
            context=context,
            context_mask=context_mask,
            proprio=proprio_first,
            action_horizon=self.action_horizon,
        )
        target = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        per_sample = self._masked_action_loss(
            prediction,
            target,
            action_is_pad,
            action_dim_is_pad,
        )
        loss = per_sample.mean()
        l1 = (prediction.float() - target.float()).abs()
        return loss, {
            "loss_action": float(loss.detach().item()),
            "action_l1_unmasked": float(l1.mean().detach().item()),
            "prediction_abs_mean": float(prediction.float().abs().mean().detach().item()),
        }

    def _load_text_encoder(self) -> None:
        if self.text_encoder is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ensure_flux2_importable(self.flux2_src_path)
        logger.info("Lazy-loading Action Qwen3 encoder from %s", self.qwen3_model_spec)
        model = AutoModelForCausalLM.from_pretrained(
            self.qwen3_model_spec,
            torch_dtype=self.torch_dtype,
        ).to(self.device).eval()
        model.requires_grad_(False)
        self.text_encoder = SimpleNamespace(
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(self.qwen3_model_spec),
        )

    @torch.no_grad()
    def _encode_prompts(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        self._load_text_encoder()
        assert self.text_encoder is not None
        from flux2.text_encoder import OUTPUT_LAYERS_QWEN3

        rendered = []
        for prompt in prompts:
            rendered.append(
                self.text_encoder.tokenizer.apply_chat_template(
                    [{"role": "user", "content": str(prompt)}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        encoded = self.text_encoder.tokenizer(
            rendered,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.qwen_context_len,
        )
        input_ids = encoded.input_ids.to(self.device)
        mask = encoded.attention_mask.to(self.device)
        outputs = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = torch.stack([outputs.hidden_states[index] for index in OUTPUT_LAYERS_QWEN3], dim=1)
        hidden = rearrange(hidden, "b c l d -> b l (c d)").to(dtype=self.torch_dtype)
        return self.dit.pack_action_prefix(hidden, mask.to(dtype=torch.bool))

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        if context is None and context_mask is None:
            if prompt is None:
                raise ValueError("Action inference requires either prompt or context/context_mask.")
            packed_context, packed_mask = self._encode_prompts([prompt])
        elif context is not None and context_mask is not None and prompt is None:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            packed_context, packed_mask = self.dit.pack_action_prefix(
                context.to(device=self.device, dtype=self.torch_dtype),
                context_mask.to(device=self.device, dtype=torch.bool),
            )
        else:
            raise ValueError("Prompt and context/context_mask are mutually exclusive.")
        proprio_first = self._prepare_proprio(proprio, batch_size=1)
        prediction = self._forward_action(
            current_image=input_image.to(device=self.device, dtype=self.torch_dtype),
            context=packed_context,
            context_mask=packed_mask,
            proprio=proprio_first,
            action_horizon=int(action_horizon),
        )
        return {"action": prediction[0].detach().to(device="cpu", dtype=torch.float32)}
