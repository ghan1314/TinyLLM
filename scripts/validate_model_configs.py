"""Validate TinyLLM model configuration files."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_config_class() -> type:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tinyllm.modeling.configuration import TinyLLMConfig

    return TinyLLMConfig


def validate_config(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        config_class = get_config_class()
        config = config_class.from_json_file(path)
    except Exception as exc:  # noqa: BLE001
        return [f"{path}: {exc}"]

    if config.model_type != "tinyllm":
        errors.append(f"{path}: model_type must be tinyllm")
    if config.max_position_embeddings < 4096 and "smoke" not in path.stem:
        errors.append(f"{path}: non-smoke configs must support at least 4096 positions")
    if config.tie_word_embeddings is not True:
        errors.append(f"{path}: initial configs should tie word embeddings")
    return errors


def main() -> int:
    paths = [Path(arg) for arg in sys.argv[1:]]
    if not paths:
        paths = sorted((REPO_ROOT / "configs" / "model").glob("*.json"))

    errors: list[str] = []
    for path in paths:
        errors.extend(validate_config(path))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: validated {len(paths)} model config(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
