#!/usr/bin/env python3
"""Download pinned official ProsQA splits and build a reproducible small subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from coconut.curriculum import CurriculumEncoder  # noqa: E402
from coconut.data import JsonReasoningDatasetAdapter, ReasoningExample  # noqa: E402

OFFICIAL_COMMIT = "27273cb8cca4bb763c041a63b036d0c3b7cbbb48"
OFFICIAL_FILES = {
    "train": {
        "name": "prosqa_train.json",
        "sha256": "99e40ce7e9107fd02e35bcd78a7a4479bd57f415bab0a6efd37cc11a72e594f6",
    },
    "validation": {
        "name": "prosqa_valid.json",
        "sha256": "c74e0de24f1e90ec48b1a993e6458a7be4caeef3928f5104e3c2e45d689a5249",
    },
}
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/facebookresearch/coconut/"
    f"{OFFICIAL_COMMIT}/data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--validation-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="EleutherAI/pythia-70m")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-latent-stage", type=int, default=6)
    parser.add_argument("--c", type=int, default=1)
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "data/prosqa_official"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/experiments/prosqa300"
    )
    parser.add_argument("--redownload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_size < 1 or args.validation_size < 1:
        raise ValueError("Subset sizes must be positive")
    if args.max_length < 2 or args.max_latent_stage < 0 or args.c < 1:
        raise ValueError("Invalid curriculum/length arguments")

    source_paths = _prepare_official_files(args.source_dir, args.redownload)
    adapter = JsonReasoningDatasetAdapter(source_paths)
    official_train = list(adapter.load_split("train"))
    official_validation = list(adapter.load_split("validation"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    encoder = CurriculumEncoder(tokenizer, c=args.c, max_length=None)
    eligible_train = _eligible_examples(
        official_train, encoder, args.max_latent_stage, args.max_length
    )
    eligible_validation = _eligible_examples(
        official_validation, encoder, args.max_latent_stage, args.max_length
    )
    if len(eligible_train) < args.train_size:
        raise ValueError(
            f"Only {len(eligible_train)} official train examples fit max_length="
            f"{args.max_length}; requested {args.train_size}"
        )
    if len(eligible_validation) < args.validation_size:
        raise ValueError(
            f"Only {len(eligible_validation)} official validation examples fit "
            f"max_length={args.max_length}; requested {args.validation_size}"
        )

    train_selection = random.Random(args.seed).sample(
        eligible_train, args.train_size
    )
    validation_selection = random.Random(args.seed + 1).sample(
        eligible_validation, args.validation_size
    )
    train_examples = [example for _, example, _ in train_selection]
    validation_examples = [example for _, example, _ in validation_selection]
    _assert_disjoint(train_examples, validation_examples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "train.json", [asdict(x) for x in train_examples])
    _write_json(
        args.output_dir / "validation.json",
        [asdict(x) for x in validation_examples],
    )
    _write_json(
        args.output_dir / "subset_metadata.json",
        {
            "official_repository": "facebookresearch/coconut",
            "official_commit": OFFICIAL_COMMIT,
            "official_files": OFFICIAL_FILES,
            "seed": args.seed,
            "model_id": args.model_id,
            "max_length": args.max_length,
            "max_latent_stage": args.max_latent_stage,
            "c": args.c,
            "official_train_count": len(official_train),
            "official_validation_count": len(official_validation),
            "eligible_train_count": len(eligible_train),
            "eligible_validation_count": len(eligible_validation),
            "train_source_indices": [index for index, _, _ in train_selection],
            "validation_source_indices": [
                index for index, _, _ in validation_selection
            ],
            "train_encoded_lengths": [length for _, _, length in train_selection],
            "validation_encoded_lengths": [
                length for _, _, length in validation_selection
            ],
        },
    )
    print(
        f"Prepared {len(train_examples)} train and {len(validation_examples)} "
        f"validation examples in {args.output_dir}"
    )
    print(
        f"Maximum encoded lengths: train={max(x[2] for x in train_selection)}, "
        f"validation={max(x[2] for x in validation_selection)}"
    )


def _prepare_official_files(source_dir: Path, redownload: bool) -> dict[str, Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, metadata in OFFICIAL_FILES.items():
        path = source_dir / metadata["name"]
        expected_hash = metadata["sha256"]
        if redownload or not path.exists():
            _download(f"{RAW_BASE_URL}/{metadata['name']}", path)
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected_hash}, got "
                f"{actual_hash}. Use --redownload to replace it from the pinned source."
            )
        paths[split] = path
    return paths


def _download(url: str, destination: Path) -> None:
    print(f"Downloading official ProsQA: {url}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "coconut-experiment"})
    with urllib.request.urlopen(request) as response:
        temporary.write_bytes(response.read())
    temporary.replace(destination)


def _eligible_examples(
    examples: list[ReasoningExample],
    encoder: CurriculumEncoder,
    max_stage: int,
    max_length: int,
) -> list[tuple[int, ReasoningExample, int]]:
    eligible: list[tuple[int, ReasoningExample, int]] = []
    for index, example in enumerate(examples):
        encoded_length = max(
            len(encoder.encode(example, stage).input_ids)
            for stage in range(max_stage + 1)
        )
        if encoded_length <= max_length:
            eligible.append((index, example, encoded_length))
    return eligible


def _assert_disjoint(
    train_examples: list[ReasoningExample],
    validation_examples: list[ReasoningExample],
) -> None:
    def signature(example: ReasoningExample) -> tuple[str, tuple[str, ...], str]:
        return example.question, tuple(example.steps), example.answer

    overlap = {signature(x) for x in train_examples} & {
        signature(x) for x in validation_examples
    }
    if overlap:
        raise ValueError("Official train and validation selections unexpectedly overlap")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
