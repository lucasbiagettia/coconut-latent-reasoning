"""Small generation and exact-match helpers shared by training and inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from .curriculum import CurriculumEncoder
from .model import CoconutModel


@dataclass(frozen=True)
class GeneratedAnswer:
    raw_text: str
    answer: str


def extract_answer(text: str) -> str:
    """Extract the final answer from Coconut's generated language continuation."""
    matches = list(re.finditer(r"Answer\s*:\s*", text, flags=re.IGNORECASE))
    candidate = text[matches[-1].end() :] if matches else text
    return candidate.strip().splitlines()[0].strip() if candidate.strip() else ""


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().casefold().split())


@torch.no_grad()
def generate_answer(
    model: CoconutModel,
    tokenizer: Any,
    encoder: CurriculumEncoder,
    question: str,
    num_latent_thoughts: int,
    device: torch.device,
    max_new_tokens: int,
) -> GeneratedAnswer:
    prompt = encoder.encode_inference_prompt(question, num_latent_thoughts)
    prompt_ids = torch.tensor([prompt.input_ids], dtype=torch.long, device=device)
    generated_ids = model.generate(
        prompt_ids,
        eos_token_id=encoder.token_ids.eos,
        max_new_tokens=max_new_tokens,
    )
    raw_text = tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return GeneratedAnswer(raw_text=raw_text, answer=extract_answer(raw_text))
