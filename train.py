"""Train Coconut from a YAML configuration."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut.config import TrainingConfig
from coconut.curriculum import CurriculumEncoder, add_coconut_tokens
from coconut.data import JsonReasoningDatasetAdapter
from coconut.model import CoconutModel
from coconut.training import CurriculumTrainer, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_yaml(args.config)
    seed_everything(config.seed)

    adapter = JsonReasoningDatasetAdapter(
        {"train": config.train_path, "validation": config.validation_path}
    )
    train_examples = adapter.load_split("train")
    validation_examples = adapter.load_split("validation")
    print(
        f"Loaded {len(train_examples)} training and "
        f"{len(validation_examples)} validation examples"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, local_files_only=config.local_files_only
    )
    token_ids = add_coconut_tokens(tokenizer)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_id, local_files_only=config.local_files_only
    )
    # Some small checkpoints are stored as fp16 and recent Transformers versions
    # preserve that dtype even on CPU. AdamW's default epsilon underflows in fp16,
    # turning the first update into NaNs. Correctness-first training stays fp32 on
    # both CPU and CUDA; mixed precision can be added later with proper scaling.
    base_model.float()
    base_model.resize_token_embeddings(len(tokenizer))
    model = CoconutModel(base_model, latent_token_id=token_ids.latent)

    trainer = CurriculumTrainer(
        model, tokenizer, train_examples, validation_examples, config
    )
    trainer.train()

    # The smoke config deliberately reaches this path after both Stage 0 and a
    # latent stage, proving that fixed-length latent inference also runs.
    encoder = CurriculumEncoder(tokenizer, c=config.c, max_length=config.max_length)
    num_latents = min(config.max_latent_stage, len(train_examples[0].steps)) * config.c
    prompt = encoder.encode_inference_prompt(
        train_examples[0].question, num_latent_thoughts=num_latents
    )
    prompt_ids = torch.tensor([prompt.input_ids], dtype=torch.long, device=trainer.device)
    generated_ids = model.generate(
        prompt_ids,
        eos_token_id=token_ids.eos,
        max_new_tokens=config.inference_max_new_tokens,
    )
    generated = tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(f"Smoke inference ({num_latents} latent thoughts): {generated!r}")


if __name__ == "__main__":
    main()
