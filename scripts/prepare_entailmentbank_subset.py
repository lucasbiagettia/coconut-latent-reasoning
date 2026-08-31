#!/usr/bin/env python3
"""Select EntailmentBank IDs reproducibly without copying dataset rows."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from coconut.curriculum import CurriculumEncoder  # noqa: E402
from coconut.data import EntailmentBankAdapter, EntailmentBankRecord  # noqa: E402
from coconut.huggingface_auth import load_huggingface_token  # noqa: E402

DATASET_ID = "sxiong/entailmentbank"
CONFIG_NAME = "task1"
DATASET_REVISION = "8c6c148f7a21c21ff037a42d9a22446c9d42debc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="EleutherAI/pythia-70m")
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-proof-depth", type=int, default=1)
    parser.add_argument("--max-proof-depth", type=int, default=4)
    parser.add_argument("--max-proof-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--c", type=int, default=1)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    token = load_huggingface_token()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token=token,
        local_files_only=args.local_files_only,
    )
    encoder = CurriculumEncoder(tokenizer, c=args.c, max_length=None)
    adapter = EntailmentBankAdapter(
        DATASET_ID,
        CONFIG_NAME,
        token=token,
        revision=args.revision,
    )

    split_sizes = {"train": args.train_size, "validation": args.validation_size}
    split_metadata: dict[str, object] = {}
    selected_id_sets: dict[str, set[str]] = {}
    for offset, (split, requested_size) in enumerate(split_sizes.items()):
        records = adapter.load_records(split)
        eligible, excluded = _eligible_records(
            records,
            encoder,
            min_depth=args.min_proof_depth,
            max_depth=args.max_proof_depth,
            max_steps=args.max_proof_steps,
            max_length=args.max_length,
        )
        if len(eligible) < requested_size:
            raise ValueError(
                f"Only {len(eligible)} {split} examples satisfy the filters; "
                f"{requested_size} requested"
            )
        selected = random.Random(args.seed + offset).sample(eligible, requested_size)
        selected_id_sets[split] = {item["id"] for item in selected}
        split_metadata[split] = {
            "original_count": len(records),
            "eligible_count": len(eligible),
            "excluded_count_by_reason": dict(sorted(excluded.items())),
            "selected_count": len(selected),
            "selected_depth_counts": _counts(selected, "proof_depth"),
            "selected_proof_length_counts": _counts(selected, "proof_length"),
            "selected_max_encoded_length": max(
                item["max_encoded_length"] for item in selected
            ),
            "selected": selected,
        }

    overlap = selected_id_sets["train"] & selected_id_sets["validation"]
    if overlap:
        raise RuntimeError("Train and validation selections overlap by ID")

    selected_max_steps = max(
        item["proof_length"]
        for split in split_metadata.values()
        for item in split["selected"]  # type: ignore[index]
    )
    metadata = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_id": DATASET_ID,
            "config_name": CONFIG_NAME,
            "revision": args.revision,
        },
        "selection": {
            "seed": args.seed,
            "sampling": "independent seeded sampling within each original split",
            "min_proof_depth": args.min_proof_depth,
            "max_proof_depth": args.max_proof_depth,
            "max_proof_steps": args.max_proof_steps,
            "tokenizer_model_id": args.model_id,
            "max_length": args.max_length,
            "c": args.c,
            "length_policy": "complete example at every curriculum stage; no truncation",
        },
        "curriculum": {
            "max_latent_stage": selected_max_steps,
            "derivation": "maximum parsed gold proof length among selected examples",
        },
        "splits": split_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)

    print(f"Wrote selected IDs and metadata to {args.output}")
    for split in ("train", "validation"):
        details = split_metadata[split]
        print(
            f"{split}: {details['selected_count']} selected from "
            f"{details['eligible_count']} eligible / {details['original_count']} original; "
            f"depths={details['selected_depth_counts']}; "
            f"max_tokens={details['selected_max_encoded_length']}"
        )
    print(f"Derived max_latent_stage={selected_max_steps}")


def _eligible_records(
    records: list[EntailmentBankRecord],
    encoder: CurriculumEncoder,
    *,
    min_depth: int,
    max_depth: int,
    max_steps: int,
    max_length: int,
) -> tuple[list[dict[str, int | str]], Counter[str]]:
    eligible: list[dict[str, int | str]] = []
    excluded: Counter[str] = Counter()
    for record in records:
        if not min_depth <= record.proof_depth <= max_depth:
            excluded["proof_depth"] += 1
            continue
        if not 1 <= record.proof_length <= max_steps:
            excluded["proof_length"] += 1
            continue
        stage_lengths = [
            len(encoder.encode(record.example, stage).input_ids)
            for stage in range(record.proof_length + 1)
        ]
        longest = max(stage_lengths)
        if longest > max_length:
            excluded["max_length"] += 1
            continue
        eligible.append(
            {
                "id": record.id,
                "source_index": record.source_index,
                "proof_depth": record.proof_depth,
                "proof_length": record.proof_length,
                "max_encoded_length": longest,
            }
        )
    return eligible, excluded


def _counts(records: list[dict[str, int | str]], field: str) -> dict[str, int]:
    values = Counter(int(record[field]) for record in records)
    return {str(value): count for value, count in sorted(values.items())}


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_size < 1 or args.validation_size < 1:
        raise ValueError("split sizes must be positive")
    if args.min_proof_depth < 1 or args.max_proof_depth < args.min_proof_depth:
        raise ValueError("invalid proof-depth range")
    if args.max_proof_steps < 1 or args.max_length < 2 or args.c < 1:
        raise ValueError("max-proof-steps, max-length, and c must be positive")


if __name__ == "__main__":
    main()
