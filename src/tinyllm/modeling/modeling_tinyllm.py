"""PyTorch implementation of the TinyLLM decoder-only model skeleton."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinyllm.modeling.configuration import TinyLLMConfig


@dataclass(slots=True)
class MoEAuxLoss:
    """Auxiliary routing losses emitted by a MoE block."""

    load_balancing: Tensor
    router_z: Tensor

    @property
    def total(self) -> Tensor:
        return self.load_balancing + self.router_z


@dataclass(slots=True)
class TinyLLMModelOutput:
    last_hidden_state: Tensor
    router_aux_loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None


@dataclass(slots=True)
class TinyLLMCausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None
    router_aux_loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None


class TinyLLMRMSNorm(nn.Module):
    """RMSNorm used by Qwen-style decoder blocks."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states.to(input_dtype) * self.weight.to(input_dtype)


class TinyLLMRotaryEmbedding(nn.Module):
    """Rotary position embedding with a Qwen-style high base theta."""

    def __init__(self, config: TinyLLMConfig) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        freqs = torch.einsum("d,bs->bsd", self.inv_freq.float(), position_ids.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class TinyLLMAttention(nn.Module):
    """Grouped-query self-attention backed by PyTorch SDPA."""

    def __init__(self, config: TinyLLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.use_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.use_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.use_bias,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.q_norm = (
            TinyLLMRMSNorm(self.head_dim, config.rms_norm_eps)
            if config.use_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            TinyLLMRMSNorm(self.head_dim, config.rms_norm_eps)
            if config.use_qk_norm
            else nn.Identity()
        )
        self.resid_dropout = nn.Dropout(config.resid_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        return self.resid_dropout(self.o_proj(attn_output))


class TinyLLMMLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, config: TinyLLMConfig, intermediate_size: int | None = None) -> None:
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=config.use_bias)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=config.use_bias)
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.gate_proj(hidden_states)
        hidden_states = F.silu(hidden_states) * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return self.dropout(hidden_states)


class TinyLLMMoE(nn.Module):
    """Simple token-choice Top-k MoE block for optional sparse TinyLLM variants."""

    def __init__(self, config: TinyLLMConfig) -> None:
        super().__init__()
        if config.num_experts <= 0:
            raise ValueError("TinyLLMMoE requires num_experts > 0")
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                TinyLLMMLP(
                    config,
                    intermediate_size=config.effective_moe_intermediate_size,
                )
                for _ in range(config.num_experts)
            ]
        )

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, MoEAuxLoss]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)
        router_logits = self.router(flat_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.config.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        topk_weights = topk_weights.to(flat_states.dtype)

        final_states = torch.zeros_like(flat_states)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, expert_position = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_output = expert(flat_states[token_idx])
            expert_weight = topk_weights[token_idx, expert_position].unsqueeze(-1)
            final_states.index_add_(0, token_idx, expert_output * expert_weight)

        router_probs = routing_weights.mean(dim=0)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.mean(dim=(0, 1))
        load_loss = self.num_experts * torch.sum(router_probs * tokens_per_expert)
        load_loss = load_loss * self.config.router_aux_loss_coef
        router_z_loss = torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean()
        router_z_loss = router_z_loss * self.config.router_z_loss_coef
        aux_loss = MoEAuxLoss(load_balancing=load_loss, router_z=router_z_loss)
        return final_states.reshape(batch_size, seq_len, hidden_size), aux_loss


class TinyLLMDecoderLayer(nn.Module):
    """Pre-norm decoder layer with attention and dense or sparse feed-forward block."""

    def __init__(self, config: TinyLLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = TinyLLMAttention(config, layer_idx)
        self.mlp = TinyLLMMoE(config) if config.uses_moe_at_layer(layer_idx) else TinyLLMMLP(config)
        self.input_layernorm = TinyLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TinyLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        router_aux_loss = None
        mlp_output = self.mlp(hidden_states)
        if isinstance(mlp_output, tuple):
            hidden_states, aux_loss = mlp_output
            router_aux_loss = aux_loss.total
        else:
            hidden_states = mlp_output
        hidden_states = residual + hidden_states
        return hidden_states, router_aux_loss


class TinyLLMPreTrainedModel(nn.Module):
    config_class = TinyLLMConfig
    base_model_prefix = "model"

    def __init__(self, config: TinyLLMConfig) -> None:
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def post_init(self) -> None:
        self.apply(self._init_weights)


class TinyLLMModel(TinyLLMPreTrainedModel):
    """Qwen3-like decoder backbone with RoPE, GQA, QK norm, and optional MoE."""

    def __init__(self, config: TinyLLMConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.layers = nn.ModuleList(
            [
                TinyLLMDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = TinyLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TinyLLMRotaryEmbedding(config)
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> TinyLLMModelOutput:
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(
                batch_size, -1
            )

        hidden_states = self.embedding_dropout(self.embed_tokens(input_ids))
        cos, sin = self.rotary_emb(position_ids)
        attention_mask = self._prepare_attention_mask(
            attention_mask=attention_mask,
            seq_len=seq_len,
            device=input_ids.device,
        )

        all_hidden_states: list[Tensor] | None = [] if output_hidden_states else None
        router_aux_losses: list[Tensor] = []

        for decoder_layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states, router_aux_loss = decoder_layer(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
            )
            if router_aux_loss is not None:
                router_aux_losses.append(router_aux_loss)

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        router_aux_loss = None
        if router_aux_losses:
            router_aux_loss = torch.stack(router_aux_losses).sum()

        return TinyLLMModelOutput(
            last_hidden_state=hidden_states,
            router_aux_loss=router_aux_loss,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )

    def _prepare_attention_mask(
        self,
        attention_mask: Tensor | None,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        )[None, None, :, :]
        if attention_mask is None:
            return causal_mask
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape [batch, seq_len]")
        key_padding_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        return causal_mask & key_padding_mask


class TinyLLMForCausalLM(TinyLLMPreTrainedModel):
    """TinyLLM decoder with a causal language-modeling head."""

    def __init__(self, config: TinyLLMConfig) -> None:
        super().__init__(config)
        self.model = TinyLLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self._init_weights(self.lm_head)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> TinyLLMCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if outputs.router_aux_loss is not None:
                loss = loss + outputs.router_aux_loss

        return TinyLLMCausalLMOutput(
            logits=logits,
            loss=loss,
            router_aux_loss=outputs.router_aux_loss,
            hidden_states=outputs.hidden_states,
        )
