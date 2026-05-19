import pytest

from tinyllm.modeling import TinyLLMConfig

torch = pytest.importorskip("torch")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def tiny_config(**overrides: object) -> TinyLLMConfig:
    values = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    values.update(overrides)
    return TinyLLMConfig(**values)


def test_dense_forward_with_labels() -> None:
    from tinyllm.modeling import TinyLLMForCausalLM

    config = tiny_config()
    model = TinyLLMForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs.logits.shape == (2, 8, config.vocab_size)
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)
    assert outputs.router_aux_loss is None
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_moe_forward_emits_router_loss() -> None:
    from tinyllm.modeling import TinyLLMForCausalLM

    config = tiny_config(
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=96,
    )
    model = TinyLLMForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs.logits.shape == (2, 8, config.vocab_size)
    assert outputs.loss is not None
    assert outputs.router_aux_loss is not None
    assert torch.isfinite(outputs.router_aux_loss)
    assert count_parameters(model) > 0
