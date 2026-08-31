#!/usr/bin/env python3
"""Run one full training step at Stage 0 and the maximum latent stage."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from coconut.config import TrainingConfig  # noqa: E402
from coconut.curriculum import (  # noqa: E402
    CoconutCollator,
    CurriculumEncoder,
    add_coconut_tokens,
)
from coconut.data import EntailmentBankAdapter  # noqa: E402
from coconut.huggingface_auth import load_huggingface_token  # noqa: E402
from coconut.model import CoconutModel  # noqa: E402
from coconut.training import (  # noqa: E402
    DynamicLossScaler,
    build_optimizer,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainingConfig.from_yaml(args.config)
    try:
        device = resolve_device(config.device)
    except RuntimeError as error:
        print(f"Memory smoke test cannot start: {error}", file=sys.stderr)
        return 2
    if device.type != "cuda":
        print("Memory smoke test requires CUDA; no CUDA device is available.", file=sys.stderr)
        return 2
    if config.optimizer == "adamw8bit" and importlib.util.find_spec("bitsandbytes") is None:
        print(
            "bitsandbytes is required. Install it with: "
            f"{sys.executable} -m pip install -r requirements-gpu.txt",
            file=sys.stderr,
        )
        return 2
    capability = torch.cuda.get_device_capability(device)
    if config.optimizer == "adamw8bit" and capability < (6, 0):
        print(
            f"adamw8bit requires compute capability >= 6.0; found {capability}",
            file=sys.stderr,
        )
        return 2

    try:
        _run(config, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(
            f"MEMORY TEST FAILED: {config.model_id} ran out of CUDA memory. "
            "Training was not started.",
            file=sys.stderr,
        )
        return 1
    except Exception as error:
        print(
            f"MEMORY TEST FAILED for a non-OOM reason: {type(error).__name__}: "
            f"{error}",
            file=sys.stderr,
        )
        return 2
    return 0


def _run(config: TrainingConfig, device: torch.device) -> None:
    seed_everything(config.seed)
    token = load_huggingface_token()
    data = config.data
    if data.type != "entailmentbank" or data.selection_metadata_path is None:
        raise ValueError("memory_smoke_test.py requires EntailmentBank selection metadata")
    metadata = json.loads(
        Path(data.selection_metadata_path).read_text(encoding="utf-8")
    )
    max_stage = int(metadata["curriculum"]["max_latent_stage"])
    if config.max_latent_stage > max_stage:
        raise ValueError("Config max_latent_stage exceeds prepared metadata")
    max_stage = config.max_latent_stage
    adapter = EntailmentBankAdapter(
        dataset_id=data.dataset_id or "sxiong/entailmentbank",
        config_name=data.config_name or "task1",
        selected_rows={"train": metadata["splits"]["train"]["selected"]},
        token=token,
        revision=data.revision,
    )
    records = adapter.load_records("train")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        token=token,
        local_files_only=config.local_files_only,
    )
    token_ids = add_coconut_tokens(tokenizer)
    model_kwargs = {
        "token": token,
        "local_files_only": config.local_files_only,
    }
    if config.precision == "fp16":
        model_kwargs["dtype"] = torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if config.precision == "fp32":
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
    ).to(device)
    optimizer = build_optimizer(model.parameters(), config, device)
    scaler = DynamicLossScaler(
        enabled=config.precision == "fp16", device=device
    )
    encoder = CurriculumEncoder(tokenizer, c=config.c, max_length=config.max_length)
    collator = CoconutCollator(encoder.token_ids.pad)

    stage_zero_records = sorted(
        records,
        key=lambda record: len(encoder.encode(record.example, 0).input_ids),
        reverse=True,
    )[: config.batch_size]
    max_stage_candidates = [
        record for record in records if record.proof_length >= max_stage
    ]
    max_stage_records = sorted(
        max_stage_candidates,
        key=lambda record: len(
            encoder.encode(record.example, max_stage).input_ids
        ),
        reverse=True,
    )[: config.batch_size]
    if len(stage_zero_records) != config.batch_size or len(max_stage_records) != config.batch_size:
        raise RuntimeError("Not enough examples to exercise the configured microbatch")
    print(
        f"GPU: {torch.cuda.get_device_name(device)} | capability={capability_text(device)} "
        f"| total={torch.cuda.get_device_properties(device).total_memory / 1024**2:.0f} MiB"
    )
    print(
        f"model={config.model_id} precision={config.precision} "
        f"optimizer={config.optimizer} gradient_checkpointing="
        f"{'ON' if config.gradient_checkpointing else 'OFF'} "
        f"batch_size={config.batch_size} "
        f"gradient_accumulation_steps={config.gradient_accumulation_steps} "
        f"effective_batch_size={config.effective_batch_size}"
    )
    _training_step(
        model,
        optimizer,
        scaler,
        collator(
            [encoder.encode(record.example, 0) for record in stage_zero_records]
        ),
        device,
        label="Stage 0",
    )
    _training_step(
        model,
        optimizer,
        scaler,
        collator(
            [
                encoder.encode(record.example, max_stage)
                for record in max_stage_records
            ]
        ),
        device,
        label=f"Stage {max_stage}",
    )
    print("MEMORY TEST PASSED: forward, backward, and optimizer step completed.")


def _training_step(
    model: CoconutModel,
    optimizer,
    scaler,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    label: str,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    moved = {name: tensor.to(device) for name, tensor in batch.items()}
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=scaler.is_enabled(),
    ):
        output = model(**moved)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError(f"{label} produced a non-finite loss")
    scaler.scale(output.loss).backward()
    if not scaler.step(optimizer, model.parameters()):
        raise RuntimeError(f"{label} produced non-finite gradients; optimizer step skipped")
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    print(
        f"{label}: loss={float(output.loss.detach().cpu()):.4f} "
        f"allocated={torch.cuda.memory_allocated(device) / 1024**2:.0f} MiB "
        f"peak={torch.cuda.max_memory_allocated(device) / 1024**2:.0f} MiB"
    )


def capability_text(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"{major}.{minor}"


if __name__ == "__main__":
    raise SystemExit(main())
