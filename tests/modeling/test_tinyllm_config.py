from pathlib import Path

import pytest

from tinyllm.modeling import TinyLLMConfig


def test_config_round_trip(tmp_path: Path) -> None:
    config = TinyLLMConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        moe_layers=(1,),
    )

    config.save_pretrained(tmp_path)
    loaded = TinyLLMConfig.from_pretrained(tmp_path)

    assert loaded.to_dict() == config.to_dict()
    assert loaded.uses_moe_at_layer(0) is False
    assert loaded.uses_moe_at_layer(1) is True


def test_config_rejects_invalid_hidden_size() -> None:
    with pytest.raises(ValueError, match="hidden_size must be divisible"):
        TinyLLMConfig(
            hidden_size=65,
            num_attention_heads=6,
            num_key_value_heads=3,
        )


def test_config_rejects_invalid_kv_heads() -> None:
    with pytest.raises(ValueError, match="num_attention_heads must be divisible"):
        TinyLLMConfig(
            hidden_size=64,
            num_attention_heads=8,
            num_key_value_heads=3,
        )
