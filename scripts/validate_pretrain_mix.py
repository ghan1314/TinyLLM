"""Validate TinyLLM pretraining data mix configuration files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_BUCKET_FIELDS = {
    "id",
    "name",
    "weight",
    "language",
    "category",
    "description",
    "examples",
    "required_filters",
}


def validate_mix(path: Path) -> list[str]:
    errors: list[str] = []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON: {exc}"]

    buckets = data.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        errors.append("buckets must be a non-empty list")
        return errors

    total_weight = 0
    seen_ids: set[str] = set()
    for index, bucket in enumerate(buckets):
        if not isinstance(bucket, dict):
            errors.append(f"bucket #{index} must be an object")
            continue

        missing = REQUIRED_BUCKET_FIELDS - set(bucket)
        if missing:
            errors.append(f"bucket #{index} is missing fields: {sorted(missing)}")

        bucket_id = bucket.get("id")
        if not isinstance(bucket_id, str) or not bucket_id:
            errors.append(f"bucket #{index} has invalid id")
        elif bucket_id in seen_ids:
            errors.append(f"duplicate bucket id: {bucket_id}")
        else:
            seen_ids.add(bucket_id)

        weight = bucket.get("weight")
        if not isinstance(weight, int) or weight <= 0:
            errors.append(f"bucket {bucket_id or index} weight must be a positive integer")
        else:
            total_weight += weight

        if not isinstance(bucket.get("examples"), list) or not bucket["examples"]:
            errors.append(f"bucket {bucket_id or index} examples must be a non-empty list")

        if not isinstance(bucket.get("required_filters"), list) or not bucket["required_filters"]:
            errors.append(f"bucket {bucket_id or index} required_filters must be a non-empty list")

    expected_total = data.get("total_weight", 100)
    if total_weight != expected_total:
        errors.append(f"bucket weights sum to {total_weight}, expected {expected_total}")

    unit = data.get("unit")
    if unit != "tokens_after_cleaning_dedup_filtering":
        errors.append("unit must be tokens_after_cleaning_dedup_filtering")

    return errors


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/data/pretrain_mix_v1.json")
    errors = validate_mix(path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: {path} is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
