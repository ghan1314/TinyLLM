"""Configuration for the TinyLLM Qwen3-like model family."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TinyLLMConfig:
    """HF-style configuration object for TinyLLM decoder-only language models."""

    vocab_size: int = 151_936
    hidden_size: int = 768
    intermediate_size: int = 2_048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 4_096
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    resid_dropout: float = 0.0
    embedding_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    use_bias: bool = False
    use_qk_norm: bool = True
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int = 151_643
    eos_token_id: int = 151_645

    # MoE is disabled by default. Set num_experts > 0 to replace dense MLP blocks.
    num_experts: int = 0
    num_experts_per_tok: int = 2
    moe_intermediate_size: int | None = None
    router_aux_loss_coef: float = 0.001
    router_z_loss_coef: float = 0.0
    norm_topk_prob: bool = True
    moe_layers: tuple[int, ...] | None = None

    model_type: str = "tinyllm"

    def __post_init__(self) -> None:
        self.validate()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_head_dim(self) -> int:
        return self.head_dim

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def effective_moe_intermediate_size(self) -> int:
        return self.moe_intermediate_size or self.intermediate_size

    def uses_moe_at_layer(self, layer_idx: int) -> bool:
        if not self.is_moe:
            return False
        if self.moe_layers is None:
            return True
        return layer_idx in self.moe_layers

    def validate(self) -> None:
        positive_int_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_int_fields.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads cannot exceed num_attention_heads")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not 0 <= self.resid_dropout < 1:
            raise ValueError("resid_dropout must be in [0, 1)")
        if not 0 <= self.embedding_dropout < 1:
            raise ValueError("embedding_dropout must be in [0, 1)")
        if self.hidden_act != "silu":
            raise ValueError("TinyLLM currently supports only hidden_act='silu'")
        if self.num_experts < 0:
            raise ValueError("num_experts cannot be negative")
        if self.num_experts == 0 and self.moe_layers:
            raise ValueError("moe_layers requires num_experts > 0")
        if self.num_experts > 0:
            if self.num_experts_per_tok <= 0:
                raise ValueError("num_experts_per_tok must be positive when MoE is enabled")
            if self.num_experts_per_tok > self.num_experts:
                raise ValueError("num_experts_per_tok cannot exceed num_experts")
            if self.effective_moe_intermediate_size <= 0:
                raise ValueError("moe_intermediate_size must be positive")
            if self.moe_layers is not None:
                invalid = [
                    layer
                    for layer in self.moe_layers
                    if layer < 0 or layer >= self.num_hidden_layers
                ]
                if invalid:
                    raise ValueError(f"moe_layers contains invalid layer indexes: {invalid}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.moe_layers is not None:
            data["moe_layers"] = list(self.moe_layers)
        return data

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(self.to_json_string(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TinyLLMConfig:
        values = dict(data)
        if isinstance(values.get("moe_layers"), list):
            values["moe_layers"] = tuple(values["moe_layers"])
        return cls(**values)

    @classmethod
    def from_json_file(cls, path: str | Path) -> TinyLLMConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_pretrained(cls, path: str | Path) -> TinyLLMConfig:
        config_path = Path(path)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        return cls.from_json_file(config_path)
