"""Coconut's stage curriculum and token-level training representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .data import ReasoningExample

BOT_TOKEN = "<bot>"
EOT_TOKEN = "<eot>"
LATENT_TOKEN = "<latent>"  # Placeholder only; its embedding is never used as a thought.
IGNORE_INDEX = -100


@dataclass(frozen=True)
class CurriculumView:
    removed_steps: list[str]
    visible_steps: list[str]
    num_latent_thoughts: int


def apply_curriculum(example: ReasoningExample, stage: int, c: int) -> CurriculumView:
    """Replace the first ``stage`` reasoning steps with ``c`` thoughts each."""
    if stage < 0:
        raise ValueError("stage must be >= 0")
    if c < 1:
        raise ValueError("c must be >= 1")

    removed_count = min(stage, len(example.steps))
    return CurriculumView(
        removed_steps=list(example.steps[:removed_count]),
        visible_steps=list(example.steps[removed_count:]),
        num_latent_thoughts=removed_count * c,
    )


@dataclass(frozen=True)
class SpecialTokenIds:
    bot: int
    eot: int
    latent: int
    pad: int
    eos: int


@dataclass(frozen=True)
class EncodedReasoningExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    latent_positions: tuple[int, ...]
    removed_step_count: int
    visible_steps: tuple[str, ...]


def add_coconut_tokens(tokenizer: Any) -> SpecialTokenIds:
    """Register Coconut markers and ensure that right-padding is available."""
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [BOT_TOKEN, EOT_TOKEN, LATENT_TOKEN]}
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Coconut requires a tokenizer with an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ids = SpecialTokenIds(
        bot=int(tokenizer.convert_tokens_to_ids(BOT_TOKEN)),
        eot=int(tokenizer.convert_tokens_to_ids(EOT_TOKEN)),
        latent=int(tokenizer.convert_tokens_to_ids(LATENT_TOKEN)),
        pad=int(tokenizer.pad_token_id),
        eos=int(tokenizer.eos_token_id),
    )
    if len({ids.bot, ids.eot, ids.latent}) != 3:
        raise ValueError("Tokenizer did not assign distinct IDs to Coconut special tokens")
    return ids


class CurriculumEncoder:
    """Turn separated reasoning steps into one stage-specific causal LM example."""

    def __init__(self, tokenizer: Any, c: int, max_length: int | None = None) -> None:
        if c < 1:
            raise ValueError("c must be >= 1")
        self.tokenizer = tokenizer
        self.c = c
        self.max_length = max_length
        self.token_ids = add_coconut_tokens(tokenizer)

    def encode(self, example: ReasoningExample, stage: int) -> EncodedReasoningExample:
        view = apply_curriculum(example, stage=stage, c=self.c)
        question_ids = self._encode_text(f"Question: {example.question.strip()}\n", special=True)
        visible_step_ids = [
            token
            for step in view.visible_steps
            for token in self._encode_text(f"{step.strip()}\n", special=False)
        ]
        answer_ids = self._encode_text(
            f"Answer: {example.answer.strip()}", special=False
        ) + [self.token_ids.eos]

        prefix = question_ids + [self.token_ids.bot]
        latent = [self.token_ids.latent] * view.num_latent_thoughts
        # <eot> is supplied at a fixed location, as in the paper's fixed-length
        # inference strategy. Its label is masked together with the prompt/latents.
        target_text = visible_step_ids + answer_ids
        input_ids = prefix + latent + [self.token_ids.eot] + target_text
        labels = [IGNORE_INDEX] * (len(prefix) + len(latent) + 1) + target_text

        if self.max_length is not None and len(input_ids) > self.max_length:
            raise ValueError(
                f"Encoded example has {len(input_ids)} tokens, exceeding max_length="
                f"{self.max_length}. Increase max_length; truncating would corrupt step boundaries."
            )

        latent_start = len(prefix)
        return EncodedReasoningExample(
            input_ids=tuple(input_ids),
            labels=tuple(labels),
            latent_positions=tuple(range(latent_start, latent_start + len(latent))),
            removed_step_count=len(view.removed_steps),
            visible_steps=tuple(view.visible_steps),
        )

    def encode_inference_prompt(
        self, question: str, num_latent_thoughts: int
    ) -> EncodedReasoningExample:
        if num_latent_thoughts < 0:
            raise ValueError("num_latent_thoughts must be >= 0")
        question_ids = self._encode_text(f"Question: {question.strip()}\n", special=True)
        prefix = question_ids + [self.token_ids.bot]
        latent = [self.token_ids.latent] * num_latent_thoughts
        input_ids = prefix + latent + [self.token_ids.eot]
        latent_start = len(prefix)
        return EncodedReasoningExample(
            input_ids=tuple(input_ids),
            labels=tuple([IGNORE_INDEX] * len(input_ids)),
            latent_positions=tuple(range(latent_start, latent_start + len(latent))),
            removed_step_count=0,
            visible_steps=(),
        )

    def _encode_text(self, text: str, *, special: bool) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=special))


class StageDataset:
    """A lightweight torch-compatible dataset for one curriculum stage."""

    def __init__(
        self,
        examples: Sequence[ReasoningExample],
        encoder: CurriculumEncoder,
        stage: int,
    ) -> None:
        self.examples = examples
        self.encoder = encoder
        self.stage = stage

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedReasoningExample:
        return self.encoder.encode(self.examples[index], self.stage)


class CoconutCollator:
    """Right-pad examples while masking every padded label."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: Sequence[EncodedReasoningExample]) -> dict[str, Any]:
        import torch

        if not examples:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(len(example.input_ids) for example in examples)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for example in examples:
            padding = max_length - len(example.input_ids)
            input_ids.append(list(example.input_ids) + [self.pad_token_id] * padding)
            labels.append(list(example.labels) + [IGNORE_INDEX] * padding)
            attention_masks.append([1] * len(example.input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
