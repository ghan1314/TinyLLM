# TinyLLM Agent Instructions

These instructions apply to the whole repository.

## Read Before Work

Before making changes, read:

1. `AGENTS.md`
2. `CODEX.md`
3. `codex-progress.md`
4. `feature_list.json`
5. Any relevant config, docs, source, or test files for the task.

After reading, identify the current continuation point from `feature_list.json` and `codex-progress.md`. Prefer continuing the highest-priority `in_progress` feature. If none exists, start the highest-priority `planned` feature whose dependencies are complete.

## Hard Safety Rules

Do not batch-delete files or directories.

Do not use:

- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

When deleting is required, delete only one explicit file path at a time, for example:

```powershell
Remove-Item "C:\path\to\file.txt"
```

If a task appears to require bulk deletion, stop and ask the user to delete the files manually.

## Code Boundaries

- Keep changes scoped to the requested task.
- Do not rewrite unrelated files.
- Do not overwrite user changes.
- Do not commit secrets, server passwords, API keys, dataset credentials, W&B keys, or SSH details.
- Do not mark features complete unless there is evidence in `feature_list.json`.
- Do not invent benchmark scores, training results, model quality claims, or dataset statistics.
- Prefer ASCII in code and config unless existing file content or domain text requires Unicode.

## Continuation Protocol

When starting a fresh conversation or resuming work:

1. Read `feature_list.json`.
2. Find incomplete features in this order: `in_progress`, then `blocked`, then `planned`.
3. Respect `priority`, `dependencies`, `owner_module`, and `next_steps`.
4. Read the feature's `docs` and any referenced files before editing.
5. If multiple features are available, continue the lowest numeric `priority` value first.
6. If a feature is blocked, do not work around it silently; record the blocker or ask the user.

When updating feature state:

- `planned`: accepted backlog item, not started.
- `in_progress`: files or design work have started, but acceptance is not complete.
- `blocked`: cannot continue without a missing dependency, decision, credential, dataset, or environment.
- `complete`: all acceptance criteria are satisfied and evidence is recorded.

Every feature entry should include `owner_module`, `dependencies`, `docs`, `next_steps`, `acceptance`, `evidence`, and `last_updated`.

## Project Direction

TinyLLM is a from-scratch, Qwen3-like small language model project. The first practical target is not a fixed 0.6B reproduction. The current target is:

- A Qwen3-like architecture.
- A 64M trial model.
- A 120M-130M MVP model.
- A full path from pretraining to post-training, evaluation, and inference.

## Implementation Preferences

- Main language: Python 3.11.
- Core framework: PyTorch.
- Training loops: custom PyTorch loops.
- Distributed first step: DDP via `torchrun`.
- Larger-scale fallback: reserve FSDP or DeepSpeed for later memory pressure.
- Compatibility: Hugging Face-style config, tokenizer, `safetensors`, and load/save layout.
- Attention implementation: PyTorch SDPA first; FlashAttention optional later.
- Configs: YAML or JSON plus typed Python config objects.
- Tests and quality: `pytest` and `ruff`.

## Completion Requirements

Before saying a task is complete:

1. Run the relevant validation command or explain why it cannot be run.
2. Update `feature_list.json` when a feature status changes.
3. Update `codex-progress.md` with the date, what changed, and evidence.
4. Keep generated data, checkpoints, logs, caches, and local secrets out of tracked project docs unless explicitly requested.

For config-only changes, at minimum validate parseability and invariants such as weight totals.
