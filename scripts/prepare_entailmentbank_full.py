#!/usr/bin/env python3
"""Record every complete, length-safe EntailmentBank Task 1 row by split."""

from __future__ import annotations

import argparse
import json
import statistics
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
from scripts.prepare_entailmentbank_subset import (  # noqa: E402
    CONFIG_NAME,
    DATASET_ID,
    DATASET_REVISION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="EleutherAI/pythia-410m")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--c", type=int, default=1)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_length < 2 or args.c < 1:
        raise ValueError("max-length must be >= 2 and c must be positive")
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

    splits: dict[str, dict] = {}
    for split in ("train", "validation", "test"):
        records = adapter.load_records(split)
        usable, excluded = _filter_by_complete_length(
            records, encoder, max_length=args.max_length
        )
        token_lengths = [int(item["max_encoded_length"]) for item in usable]
        splits[split] = {
            "original_count": len(records),
            "usable_count": len(usable),
            "excluded_count_by_reason": {"max_length": excluded},
            "usable_depth_counts": _counts(usable, "proof_depth"),
            "usable_proof_length_counts": _counts(usable, "proof_length"),
            "maximum_token_length": max(token_lengths),
            "mean_token_length": statistics.fmean(token_lengths),
            # ``selected`` is the generic adapter input: here it contains every
            # usable row in original order, not a random subset.
            "selected": usable,
        }

    max_stage = max(int(item["proof_length"]) for item in splits["train"]["selected"])
    for split in ("validation", "test"):
        split_max = max(
            int(item["proof_length"]) for item in splits[split]["selected"]
        )
        if split_max > max_stage:
            raise ValueError(
                f"{split} requires {split_max} stages but train only derives {max_stage}"
            )

    metadata = {
        "format_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_id": DATASET_ID,
            "config_name": CONFIG_NAME,
            "revision": args.revision,
        },
        "selection": {
            "sampling": "none; every valid row fitting max_length in its original split",
            "tokenizer_model_id": args.model_id,
            "max_length": args.max_length,
            "c": args.c,
            "length_policy": "complete example at every curriculum stage; no truncation",
        },
        "curriculum": {
            "max_latent_stage": max_stage,
            "derivation": "maximum parsed gold proof length in usable train rows",
        },
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Wrote full-split metadata to {args.output}")
    _print_statistics(splits, max_stage)


def _filter_by_complete_length(
    records: list[EntailmentBankRecord],
    encoder: CurriculumEncoder,
    *,
    max_length: int,
) -> tuple[list[dict[str, int | str]], int]:
    usable: list[dict[str, int | str]] = []
    excluded = 0
    for record in records:
        stage_lengths = [
            len(encoder.encode(record.example, stage).input_ids)
            for stage in range(record.proof_length + 1)
        ]
        longest = max(stage_lengths)
        if longest > max_length:
            excluded += 1
            continue
        usable.append(
            {
                "id": record.id,
                "source_index": record.source_index,
                "proof_depth": record.proof_depth,
                "proof_length": record.proof_length,
                "max_encoded_length": longest,
            }
        )
    if not usable:
        raise ValueError("No examples fit max_length")
    return usable, excluded


def _counts(records: list[dict[str, int | str]], field: str) -> dict[str, int]:
    values = Counter(int(record[field]) for record in records)
    return {str(value): count for value, count in sorted(values.items())}


def _print_statistics(splits: dict[str, dict], max_stage: int) -> None:
    train = splits["train"]
    print(f"original train examples: {train['original_count']}")
    print(f"usable train examples: {train['usable_count']}")
    print(f"validation examples: {splits['validation']['usable_count']}")
    print(f"test examples (held out): {splits['test']['usable_count']}")
    for split in ("train", "validation", "test"):
        details = splits[split]
        print(
            f"{split} distribution by reasoning depth: "
            f"{details['usable_depth_counts']}"
        )
        print(
            f"{split} token length: maximum={details['maximum_token_length']} "
            f"mean={details['mean_token_length']:.2f}"
        )
    print(f"derived max_latent_stage: {max_stage}")


if __name__ == "__main__":
    main()
