"""Train Coconut from a YAML configuration."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut.config import TrainingConfig
from coconut.curriculum import CurriculumEncoder, add_coconut_tokens
from coconut.data import (
    EntailmentBankAdapter,
    HuggingFaceDatasetAdapter,
    JsonReasoningDatasetAdapter,
)
from coconut.huggingface_auth import load_huggingface_token
from coconut.model import CoconutModel
from coconut.training import CurriculumTrainer, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument(
        "--resume-from",
        help="Override resume_from_checkpoint with an epoch checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_yaml(args.config)
    if args.resume_from:
        config = replace(config, resume_from_checkpoint=args.resume_from)
    seed_everything(config.seed)
    hf_token = load_huggingface_token()
    runtime_device = resolve_device(config.device)
    if config.precision == "fp16" and runtime_device.type != "cuda":
        raise ValueError("precision=fp16 requires CUDA")

    data = config.data
    if data.type == "json":
        assert data.train_path is not None and data.validation_path is not None
        adapter = JsonReasoningDatasetAdapter(
            {
                data.train_split: data.train_path,
                data.validation_split: data.validation_path,
            },
            columns=data.columns,
        )
        train_examples = adapter.load_split(data.train_split)
        validation_examples = adapter.load_split(data.validation_split)
        validation_proof_depths = None
    elif data.type == "huggingface":
        assert data.dataset_id is not None
        adapter = HuggingFaceDatasetAdapter(
            data.dataset_id,
            config_name=data.config_name,
            columns=data.columns,
            token=hf_token,
            revision=data.revision,
        )
        train_examples = adapter.load_split(data.train_split)
        validation_examples = adapter.load_split(data.validation_split)
        validation_proof_depths = None
    else:
        assert data.dataset_id is not None
        assert data.config_name is not None
        assert data.selection_metadata_path is not None
        with open(data.selection_metadata_path, encoding="utf-8") as handle:
            selection = json.load(handle)
        derived_stage = int(selection["curriculum"]["max_latent_stage"])
        if config.max_latent_stage > derived_stage:
            raise ValueError(
                f"max_latent_stage={config.max_latent_stage} exceeds the selected "
                f"gold proofs ({derived_stage})"
            )
        if config.max_latent_stage < derived_stage:
            print(
                f"curriculum stage limit: configured={config.max_latent_stage} "
                f"dataset_max={derived_stage}"
            )
        selected_rows = {
            split: selection["splits"][split]["selected"]
            for split in (data.train_split, data.validation_split)
        }
        adapter = EntailmentBankAdapter(
            data.dataset_id,
            config_name=data.config_name,
            selected_rows=selected_rows,
            token=hf_token,
            revision=data.revision,
        )
        train_records = adapter.load_records(data.train_split)
        validation_records = adapter.load_records(data.validation_split)
        train_examples = [record.example for record in train_records]
        validation_examples = [record.example for record in validation_records]
        validation_proof_depths = [
            record.proof_depth for record in validation_records
        ]
        _print_entailmentbank_statistics(selection)
    print(
        f"Loaded {len(train_examples)} training and "
        f"{len(validation_examples)} validation examples"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        local_files_only=config.local_files_only,
        token=hf_token,
    )
    token_ids = add_coconut_tokens(tokenizer)
    model_kwargs = {
        "local_files_only": config.local_files_only,
        "token": hf_token,
    }
    if config.precision == "fp16":
        model_kwargs["dtype"] = torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if config.precision == "fp32":
        # CPU smoke checkpoints can arrive as fp16. Standard AdamW with fp16
        # parameters is unstable without scaling, so the fp32 path is explicit.
        base_model.float()
    if config.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    base_model.resize_token_embeddings(len(tokenizer))
    model = CoconutModel(
        base_model,
        latent_token_id=token_ids.latent,
        implementation=config.implementation,
    )

    trainer = CurriculumTrainer(
        model,
        tokenizer,
        train_examples,
        validation_examples,
        config,
        validation_proof_depths=validation_proof_depths,
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


def _print_entailmentbank_statistics(selection: dict) -> None:
    splits = selection["splits"]
    train = splits["train"]
    if "usable_count" not in train:
        return
    validation = splits["validation"]
    test = splits.get("test")
    print(f"original train examples: {train['original_count']}")
    print(f"usable train examples: {train['usable_count']}")
    print(f"validation examples: {validation['usable_count']}")
    if test is not None:
        print(f"test examples (held out): {test['usable_count']}")
    for name in ("train", "validation", "test"):
        if name not in splits:
            continue
        details = splits[name]
        print(
            f"{name} distribution by reasoning depth: "
            f"{details['usable_depth_counts']}"
        )
        print(
            f"{name} token length: maximum={details['maximum_token_length']} "
            f"mean={details['mean_token_length']:.2f}"
        )


if __name__ == "__main__":
    main()
