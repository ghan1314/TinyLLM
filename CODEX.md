# TinyLLM Codex Guide

## Project Status

TinyLLM is currently in project bootstrap. The repository has governance docs and early configuration, but no runnable model, data pipeline, training loop, or package code yet.

For a new conversation, resume from `feature_list.json` first and use `codex-progress.md` as the human-readable timeline. The current recommended next feature is the highest-priority incomplete item in `feature_list.json`.

## Target

The first implementation target is a Qwen3-like small LLM stack:

- 64M trial model to debug the full pipeline.
- 120M-130M MVP model as the first serious result.
- 4K context first, with later 8K continuation.
- Self-trained 64K BBPE tokenizer.
- Pretraining, SFT, DPO, evaluation, inference, and HF-compatible export.

## Expected Structure

Planned project layout:

```text
configs/
  data/
  model/
  train/
docs/
  data/
scripts/
src/
  tinyllm/
tests/
data/
outputs/
```

Current existing files:

```text
AGENTS.md
AGENT.md
CODEX.md
codex-progress.md
feature_list.json
```

`data/` and `outputs/` are intended for local datasets, tokenized shards, logs, checkpoints, and generated artifacts. Do not treat them as source-of-truth code.

## Commands

Repository-level validation commands:

```powershell
python scripts\validate_pretrain_mix.py
python scripts\validate_model_configs.py
python -m pytest
python -m ruff check .
```

Module-specific validation commands should live beside the relevant module documentation. Add global commands here only when they apply to the whole project, such as package tests, linting, formatting checks, or training smoke tests.

## Resume Workflow

Use this process at the start of each new session:

1. Read `AGENTS.md`, `CODEX.md`, `feature_list.json`, and `codex-progress.md`.
2. Select the next feature by status and priority:
   - continue `in_progress` first;
   - then unblock or report `blocked`;
   - then start the lowest-numbered `planned` priority whose dependencies are complete.
3. Read the feature's module docs and evidence files.
4. Implement only that feature unless the user asks otherwise.
5. Validate with the smallest relevant command.
6. Update `feature_list.json` and `codex-progress.md`.

Feature records are the source of truth for backlog and completion. Progress notes are the source of truth for recent decisions and evidence.

## Environment

Recommended baseline:

- Python 3.11.
- Conda + pip.
- PyTorch.
- Ubuntu server for real GPU training.
- Windows is acceptable for editing, config validation, and light smoke tests.

Do not write SSH passwords, API keys, W&B keys, Hugging Face tokens, dataset credentials, or other secrets into this repository.

## Coding Conventions

- Use `src/tinyllm` for importable code.
- Use `scripts` for thin CLI entrypoints.
- Keep model, data, training, evaluation, and inference code separated.
- Keep configs explicit and reproducible.
- Prefer Hugging Face-compatible config/checkpoint/tokenizer conventions.
- Add tests near the behavior being introduced.
- Update `feature_list.json` and `codex-progress.md` whenever a feature moves status.

## Work Style

Before implementing:

1. Read the project guidance files.
2. Inspect relevant existing files.
3. Keep edits narrow.
4. Validate with the smallest relevant command.
5. Record evidence before claiming completion.

If a task would require unsafe deletion, bulk cleanup, credential storage, or unverifiable training claims, stop and ask the user how to proceed.
