# Codex Progress

## 2026-05-18

- Implemented TinyLLM pretraining data mix v1 as a machine-readable JSON config.
- Added human-readable documentation for the token-based Chinese-priority bilingual data mix.
- Added a no-dependency validation script for the pretraining mix config.
- Recorded the completed data-mix feature and verification evidence in `feature_list.json`.
- Added canonical project agent instructions in `AGENTS.md`.
- Added `AGENT.md` as a compatibility pointer to `AGENTS.md`.
- Added `CODEX.md` with project status, commands, structure, conventions, and working style.
- Recorded the completed governance docs feature in `feature_list.json`.
- Moved pretraining data mix details out of global `AGENTS.md` and `CODEX.md`; the data-specific source of truth remains under `configs/data/` and `docs/data/`.
- Strengthened restart/continuation rules in `AGENTS.md` and `CODEX.md`.
- Upgraded `feature_list.json` to schema version 2 with priority, dependencies, owner module, docs, next steps, and last-updated fields.
- Added planned backlog entries for package scaffold, architecture spec, data pipeline spec, and pretraining loop spec.

## Current Resume Point

- Current highest-priority incomplete feature: `data.pipeline_spec`.
- Recommended next action: document the pretraining data pipeline spec under `docs/data/pipeline.md` and the related validation commands.
- Before starting any new feature, read `AGENTS.md`, `CODEX.md`, `feature_list.json`, and this file.

## 2026-05-19

- Added a Python package scaffold under `src/tinyllm` with `pyproject.toml`, minimal tests, and repository-level validation commands in `CODEX.md`.
- Implemented the TinyLLM model skeleton: RMSNorm, RoPE, grouped-query self-attention with SDPA, SwiGLU MLP, optional token-choice MoE, and a causal LM head.
- Added initial dense model configs for a 64M trial path and a 120M-130M MVP path, plus a small MoE smoke config under `configs/model/`.
- Added `docs/model/architecture.md` describing the Qwen3-like design and the optional MoE path.
- Added `docs/setup/server_environment.md` and conda/pip dependency files for Python 3.11 and CUDA 12.6 on the server, using the official PyTorch cu126 wheel index.
- Added `scripts/validate_model_configs.py` and validated the model configs.
- Verified with `python -m pytest`, `python -m ruff check .`, `python scripts\\validate_model_configs.py`, and `python scripts\\validate_pretrain_mix.py`.
- Recorded completion for `project.scaffold_python_package` and `model.architecture_spec` in `feature_list.json`.
