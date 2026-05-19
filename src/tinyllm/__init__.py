"""TinyLLM package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tinyllm.modeling.configuration import TinyLLMConfig

if TYPE_CHECKING:
    from tinyllm.modeling.modeling_tinyllm import TinyLLMForCausalLM, TinyLLMModel

__all__ = ["TinyLLMConfig", "TinyLLMForCausalLM", "TinyLLMModel"]


def __getattr__(name: str) -> object:
    if name in {"TinyLLMForCausalLM", "TinyLLMModel"}:
        from tinyllm.modeling.modeling_tinyllm import TinyLLMForCausalLM, TinyLLMModel

        return {"TinyLLMForCausalLM": TinyLLMForCausalLM, "TinyLLMModel": TinyLLMModel}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
