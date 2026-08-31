#!/usr/bin/env python3
"""Load a persisted Coconut experiment and answer one or more questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from coconut.curriculum import CurriculumEncoder, add_coconut_tokens  # noqa: E402
from coconut.evaluation import generate_answer  # noqa: E402
from coconut.model import CoconutModel  # noqa: E402
from coconut.training import resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--question")
    parser.add_argument(
        "--context",
        help="Optional factual context prepended to --question (for EntailmentBank)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--show-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.model_dir / "coconut_config.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {metadata_path}; training must finish before using ask_model.py"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir / "tokenizer")
    token_ids = add_coconut_tokens(tokenizer)
    precision = metadata.get("precision", "fp32")
    dtype = torch.float16 if device.type == "cuda" and precision == "fp16" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir / "model", dtype=dtype
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = CoconutModel(
        base_model,
        latent_token_id=token_ids.latent,
        implementation=metadata.get("implementation", "reference"),
    ).to(device)
    model.eval()
    encoder = CurriculumEncoder(tokenizer, c=int(metadata["c"]))
    num_latents = int(metadata["num_latent_thoughts"])
    max_new_tokens = args.max_new_tokens or int(metadata["max_new_tokens"])
    print(
        f"Loaded Coconut from {args.model_dir} on {device}; "
        f"latent_thoughts={num_latents}"
    )

    if args.question is not None:
        _answer(
            _with_context(args.question, args.context),
            model,
            tokenizer,
            encoder,
            num_latents,
            max_new_tokens,
            device,
            args.show_raw,
        )
        return

    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        _answer(
            _with_context(question, args.context),
            model,
            tokenizer,
            encoder,
            num_latents,
            max_new_tokens,
            device,
            args.show_raw,
        )


def _with_context(question: str, context: str | None) -> str:
    if context is None or not context.strip():
        return question
    return f"Context: {context.strip()}\nQuestion: {question.strip()}"


def _answer(
    question,
    model,
    tokenizer,
    encoder,
    num_latents,
    max_new_tokens,
    device,
    show_raw,
) -> None:
    generated = generate_answer(
        model,
        tokenizer,
        encoder,
        question,
        num_latents,
        device,
        max_new_tokens,
    )
    print(f"Answer> {generated.answer}")
    if show_raw and generated.raw_text != generated.answer:
        print(f"Raw> {generated.raw_text}")


if __name__ == "__main__":
    main()
