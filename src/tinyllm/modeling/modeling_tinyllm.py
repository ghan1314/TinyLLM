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

    # load_balancing 约束专家使用率，避免路由器长期只选择少数专家。
    load_balancing: Tensor
    # router_z 是可选的 router logit 稳定项；默认系数为 0，保留接口便于后续打开。
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
        # RMSNorm 的均方根统计用 float32 计算，避免 fp16/bf16 下方差过小带来的数值抖动。
        # 输出再转回输入 dtype，便于混合精度训练。
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states.to(input_dtype) * self.weight.to(input_dtype)


class TinyLLMRotaryEmbedding(nn.Module):
    """Rotary position embedding with a Qwen-style high base theta."""

    def __init__(self, config: TinyLLMConfig) -> None:
        super().__init__()
        # inv_freq 形状为 [head_dim / 2]，不是可训练参数。RoPE 在 forward 中根据
        # position_ids 动态生成 cos/sin，因此可以支持任意 batch 的位置编号。
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        # position_ids: [batch, seq_len]；输出 cos/sin: [batch, seq_len, head_dim]。
        # einsum 将每个位置编号乘以每个频率，得到每个 token 的旋转角。
        freqs = torch.einsum("d,bs->bsd", self.inv_freq.float(), position_ids.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: Tensor) -> Tensor:
    # RoPE 的二维旋转写法：把最后一维拆成两半后做 [-x2, x1]，等价于复数旋转的虚实部交换。
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    # q/k: [batch, heads, seq_len, head_dim]；cos/sin: [batch, seq_len, head_dim]。
    # 在第 1 维补一个 heads 维度后，cos/sin 会广播到所有注意力头。
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    # GQA/MQA 中 K/V 头数少于 Q 头数。这里把每个 K/V 头复制 n_rep 次，
    # 让 K/V 的 heads 维度与 query heads 对齐，便于直接调用 SDPA。
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

        # 线性投影后仍是 [batch, seq_len, heads * head_dim] 的扁平形态。
        # 后面 reshape/transpose 会把 heads 维度显式拆出来。
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # SDPA 需要 [batch, heads, seq_len, head_dim]。Q 使用完整 attention heads；
        # K/V 使用较少的 num_key_value_heads，稍后通过 repeat_kv 扩展。
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # QK norm 先在每个 head 内做 RMSNorm，再应用 RoPE；这是 Qwen 风格骨架中
        # 提升训练稳定性的关键位置之一。
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attention_mask 是 bool mask，True 表示允许注意力访问。因果约束和 padding
        # 约束已经在 TinyLLMModel._prepare_attention_mask 中合并，所以这里关闭
        # is_causal，避免 PyTorch 再叠加一个内部 causal mask。
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        # 从 [batch, heads, seq_len, head_dim] 合并回 [batch, seq_len, hidden_size]，
        # 这样才能接输出投影和残差连接。
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
        # SwiGLU 必须让原始 hidden_states 同时经过 gate_proj 和 up_proj。
        # 不能先覆盖 hidden_states 再送入 up_proj，否则输入维度会变成 intermediate_size。
        gate_states = self.gate_proj(hidden_states)
        hidden_states = F.silu(gate_states) * self.up_proj(hidden_states)
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
        # 路由器按 token 独立选择专家，所以先把 [batch, seq_len, hidden] 展平为
        # [tokens, hidden]，其中 tokens = batch * seq_len。
        flat_states = hidden_states.reshape(-1, hidden_size)
        router_logits = self.router(flat_states)
        # router softmax 固定用 float32，避免混合精度下 Top-k 边界概率不稳定。
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.config.norm_topk_prob:
            # 归一化后，每个 token 被选中的 top-k 专家权重之和为 1。
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        topk_weights = topk_weights.to(flat_states.dtype)

        final_states = torch.zeros_like(flat_states)
        for expert_idx, expert in enumerate(self.experts):
            # selected_experts: [tokens, top_k]。where 返回 token 下标和该专家在 top-k
            # 列表中的位置，用于取出对应路由权重。
            token_idx, expert_position = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_output = expert(flat_states[token_idx])
            expert_weight = topk_weights[token_idx, expert_position].unsqueeze(-1)
            # 一个 token 可能路由到多个专家，用 index_add_ 把专家输出按权重累加回原 token。
            final_states.index_add_(0, token_idx, expert_output * expert_weight)

        # 负载均衡项参考 Switch/GShard 思路：同时考虑路由概率质量和实际分配频率。
        # 当前实现是清晰优先的 smoke 版本，后续大模型训练可替换为更高效的 grouped kernel。
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

        # 第二个 pre-norm 分支接 dense MLP 或 MoE。MoE 分支会额外返回 router aux loss，
        # dense 分支只返回 hidden states。
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
            # 保持 HF 风格的简单正态初始化，后续如需按层缩放残差可在这里集中调整。
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
            # 默认位置从 0 递增。显式 position_ids 预留给后续 KV cache、packing 或长上下文续训。
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(
                batch_size, -1
            )

        hidden_states = self.embedding_dropout(self.embed_tokens(input_ids))
        cos, sin = self.rotary_emb(position_ids)
        # mask 统一整理成 SDPA 可消费的 bool 形状 [batch, 1, query_len, key_len]。
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
            # 多个 MoE 层的 aux loss 直接求和；系数已经在每层 MoE 内部乘过。
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
        # causal_mask: [1, 1, seq_len, seq_len]，True 表示当前位置可以看见该 key。
        # 下三角保证 token 只能看见自己和历史 token。
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        )[None, None, :, :]
        if attention_mask is None:
            return causal_mask
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape [batch, seq_len]")
        # attention_mask: [batch, seq_len]，通常 1 表示有效 token，0 表示 padding。
        # 扩展成 key 维度 mask 后与 causal_mask 相与，得到最终可见区域。
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
            # 自回归语言模型预测下一个 token：位置 t 的 logits 对齐 labels[t + 1]。
            # labels 中的 -100 会被 cross_entropy 忽略，便于后续 padding/packing。
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if outputs.router_aux_loss is not None:
                # MoE 辅助损失加入主 LM loss，使路由器在普通训练 loop 中也能收到梯度。
                loss = loss + outputs.router_aux_loss

        return TinyLLMCausalLMOutput(
            logits=logits,
            loss=loss,
            router_aux_loss=outputs.router_aux_loss,
            hidden_states=outputs.hidden_states,
        )
