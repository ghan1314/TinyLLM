# TinyLLM Qwen3-like Architecture

TinyLLM uses a decoder-only Transformer layout intended to stay close to modern Qwen-style small language models while remaining simple enough for a from-scratch implementation.

## Core Blocks

- Token embedding plus tied LM head by default.
- Pre-norm decoder layers using RMSNorm.
- Grouped-query self-attention using PyTorch SDPA.
- RoPE with `rope_theta=1000000.0` for 4K context first and later 8K continuation.
- Optional QK norm on per-head query and key states.
- SwiGLU feed-forward layers.
- Optional token-choice Top-k MoE feed-forward layers.
- No linear bias by default.

## Initial Dense Configs

`configs/model/tinyllm_64m.json` is the first trial model for pipeline debugging:

- vocab size: 32768
- hidden size: 512
- layers: 10
- attention heads: 8
- KV heads: 4
- intermediate size: 1376
- context: 4096

`configs/model/tinyllm_128m.json` is the first MVP-size target:

- vocab size: 32768
- hidden size: 768
- layers: 12
- attention heads: 12
- KV heads: 4
- intermediate size: 2048
- context: 4096

The final parameter count depends on the tokenizer vocabulary size and whether embeddings are tied. Use the test helper in `tests/modeling/test_tinyllm_model.py` or instantiate the model and sum trainable parameters for exact counts.

## MoE Option

MoE is disabled when `num_experts` is `0`. To enable sparse feed-forward blocks:

- set `num_experts` to the number of experts;
- set `num_experts_per_tok` to the Top-k routing count;
- optionally set `moe_intermediate_size`;
- optionally set `moe_layers` to a list of layer indexes if only selected layers should use MoE.

The current implementation uses a simple token-choice router and emits a load-balancing auxiliary loss. It is intended as a correctness-first skeleton before adding expert parallelism, capacity factors, fused routing kernels, or distributed dispatch.

## Compatibility Notes

`TinyLLMConfig` can save and load a Hugging Face-style `config.json`. Checkpoint layout, tokenizer export, generation helpers, and direct `transformers.PreTrainedModel` integration are planned for later training and inference features.
