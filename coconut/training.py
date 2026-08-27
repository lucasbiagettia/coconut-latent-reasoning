"""A small observable, resumable trainer for the Coconut curriculum."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .curriculum import CoconutCollator, CurriculumEncoder, StageDataset
from .data import ReasoningExample
from .evaluation import generate_answer, normalize_answer
from .model import CoconutModel


@dataclass(frozen=True)
class EpochMetrics:
    stage: int
    epoch: int
    train_loss: float
    validation_loss: float | None
    validation_answer_exact_match: float | None
    accuracy_examples: int
    learning_rate: float
    elapsed_seconds: float
    epoch_seconds: float
    gpu_memory_allocated_mb: float | None
    gpu_memory_peak_mb: float | None


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


class CurriculumTrainer:
    def __init__(
        self,
        model: CoconutModel,
        tokenizer: object,
        train_examples: Sequence[ReasoningExample],
        validation_examples: Sequence[ReasoningExample],
        config: TrainingConfig,
    ) -> None:
        if not train_examples or not validation_examples:
            raise ValueError("Training and validation splits must both be non-empty")
        self.model = model
        self.tokenizer = tokenizer
        self.train_examples = train_examples
        self.validation_examples = validation_examples
        self.config = config
        self.device = resolve_device(config.device)
        self.model.to(self.device)
        self.encoder = CurriculumEncoder(tokenizer, c=config.c, max_length=config.max_length)
        self.collator = CoconutCollator(self.encoder.token_ids.pad)
        self.output_dir = Path(config.output_dir or config.checkpoint_dir)
        self.checkpoint_dir = (
            self.output_dir / "checkpoints" if config.output_dir else self.output_dir
        )
        self.history_path = self.output_dir / "training_history.json"

    def train(self) -> list[EpochMetrics]:
        self._prepare_output_directory()
        resume = self._load_resume_checkpoint()
        history = self._history_from_checkpoint(resume)
        start_stage, start_epoch = self._resume_position(resume)
        elapsed_offset = history[-1].elapsed_seconds if history else 0.0
        run_started = time.perf_counter()

        if start_stage > self.config.max_latent_stage:
            print("Checkpoint already completed the configured curriculum.")
            self._save_final_artifacts(history)
            return history

        for stage in range(start_stage, self.config.max_latent_stage + 1):
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            first_epoch = start_epoch if stage == start_stage else 1
            if resume is not None and stage == int(resume["stage"]) and first_epoch > 1:
                optimizer.load_state_dict(resume["optimizer_state_dict"])

            train_loader = self._loader(self.train_examples, stage, shuffle=True)
            validation_loader = self._loader(
                self.validation_examples, stage, shuffle=False
            )
            max_latents = (
                min(stage, max(len(example.steps) for example in self.train_examples))
                * self.config.c
            )
            stage_started = time.perf_counter()
            print(
                f"\n=== Stage {stage}/{self.config.max_latent_stage} | "
                f"up to {max_latents} continuous thoughts | "
                f"implementation={self.config.implementation} ==="
            )

            last_metrics: EpochMetrics | None = None
            epochs_in_stage = self.config.epochs_for(stage)
            for epoch in range(first_epoch, epochs_in_stage + 1):
                epoch_started = time.perf_counter()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)

                train_loss = self._train_epoch(train_loader, optimizer)
                should_evaluate = (
                    epoch % self.config.eval_every_epochs == 0
                    or epoch == epochs_in_stage
                )
                validation_loss: float | None = None
                answer_accuracy: float | None = None
                predictions: list[dict[str, str]] = []
                if should_evaluate:
                    validation_loss = self.validate_loss(validation_loader)
                    answer_accuracy, predictions = self.evaluate_answers(stage)

                epoch_seconds = time.perf_counter() - epoch_started
                elapsed_seconds = elapsed_offset + time.perf_counter() - run_started
                allocated_mb, peak_mb = self._gpu_memory()
                metrics = EpochMetrics(
                    stage=stage,
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    validation_answer_exact_match=answer_accuracy,
                    accuracy_examples=len(predictions),
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    elapsed_seconds=elapsed_seconds,
                    epoch_seconds=epoch_seconds,
                    gpu_memory_allocated_mb=allocated_mb,
                    gpu_memory_peak_mb=peak_mb,
                )
                history.append(metrics)
                last_metrics = metrics
                self._write_history(history)
                self._print_epoch(metrics)

                if (
                    predictions
                    and self.config.qualitative_examples > 0
                    and epoch % self.config.qualitative_every_epochs == 0
                ):
                    self._print_qualitative(predictions)

                if (
                    epoch % self.config.checkpoint_every_epochs == 0
                    or epoch == epochs_in_stage
                ):
                    self._save_checkpoint(metrics, optimizer, history)

            if last_metrics is not None:
                stage_seconds = time.perf_counter() - stage_started
                validation = _format_metric(last_metrics.validation_loss)
                accuracy = _format_percent(
                    last_metrics.validation_answer_exact_match
                )
                print(
                    f"=== Stage {stage} complete in {_format_duration(stage_seconds)} | "
                    f"train_loss={last_metrics.train_loss:.4f} | "
                    f"validation_loss={validation} | exact_match={accuracy} ==="
                )
            resume = None
            start_epoch = 1

        self._save_final_artifacts(history)
        return history

    def _loader(
        self, examples: Sequence[ReasoningExample], stage: int, *, shuffle: bool
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + stage)
        return DataLoader(
            StageDataset(examples, self.encoder, stage),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
            generator=generator,
        )

    def _train_epoch(self, loader: DataLoader, optimizer: AdamW) -> float:
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        accumulation = self.config.gradient_accumulation_steps
        batches_to_process = len(loader)
        if self.config.max_train_batches is not None:
            batches_to_process = min(batches_to_process, self.config.max_train_batches)

        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > batches_to_process:
                break
            output = self.model(**self._to_device(batch))
            assert output.loss is not None
            group_start = ((batch_index - 1) // accumulation) * accumulation + 1
            group_size = min(accumulation, batches_to_process - group_start + 1)
            (output.loss / group_size).backward()
            losses.append(float(output.loss.detach().cpu()))

            if batch_index % accumulation == 0 or batch_index == batches_to_process:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if not losses:
            raise RuntimeError("Training loader produced no batches")
        return sum(losses) / len(losses)

    @torch.no_grad()
    def validate_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        weighted_loss = 0.0
        target_count = 0
        for batch_index, batch in enumerate(loader, start=1):
            if (
                self.config.max_validation_batches is not None
                and batch_index > self.config.max_validation_batches
            ):
                break
            output = self.model(**self._to_device(batch))
            assert output.loss is not None
            weighted_loss += float(output.loss.cpu()) * output.target_token_count
            target_count += output.target_token_count
        if target_count == 0:
            raise RuntimeError("Validation loader produced no target tokens")
        return weighted_loss / target_count

    def evaluate_answers(
        self, stage: int
    ) -> tuple[float | None, list[dict[str, str]]]:
        limit = min(self.config.accuracy_max_examples, len(self.validation_examples))
        if limit == 0:
            return None, []

        self.model.eval()
        predictions: list[dict[str, str]] = []
        correct = 0
        for example in self.validation_examples[:limit]:
            num_latents = min(stage, len(example.steps)) * self.config.c
            generated = generate_answer(
                self.model,
                self.tokenizer,
                self.encoder,
                example.question,
                num_latents,
                self.device,
                self.config.inference_max_new_tokens,
            )
            is_correct = normalize_answer(generated.answer) == normalize_answer(
                example.answer
            )
            correct += int(is_correct)
            predictions.append(
                {
                    "question": example.question,
                    "expected": example.answer,
                    "predicted": generated.answer,
                    "raw_generation": generated.raw_text,
                }
            )
        return correct / limit, predictions

    def _prepare_output_directory(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_dir = self.output_dir / "tokenizer"
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(tokenizer_dir)
        _write_json(self.output_dir / "training_config.json", self.config.to_dict())

    def _load_resume_checkpoint(self) -> dict[str, Any] | None:
        if self.config.resume_from_checkpoint is None:
            return None
        path = Path(self.config.resume_from_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Resuming from {path} after stage={checkpoint['stage']} "
            f"epoch={checkpoint['epoch']}"
        )
        return checkpoint

    @staticmethod
    def _history_from_checkpoint(
        checkpoint: dict[str, Any] | None,
    ) -> list[EpochMetrics]:
        if checkpoint is None:
            return []
        return [EpochMetrics(**item) for item in checkpoint.get("history", [])]

    def _resume_position(
        self, checkpoint: dict[str, Any] | None
    ) -> tuple[int, int]:
        if checkpoint is None:
            return 0, 1
        stage = int(checkpoint["stage"])
        epoch = int(checkpoint["epoch"]) + 1
        if epoch > self.config.epochs_for(stage):
            return stage + 1, 1
        return stage, epoch

    def _save_checkpoint(
        self,
        metrics: EpochMetrics,
        optimizer: AdamW,
        history: Sequence[EpochMetrics],
    ) -> None:
        if self.config.output_dir:
            path = self.checkpoint_dir / "latest.pt"
        else:
            path = self.checkpoint_dir / f"stage_{metrics.stage}_epoch_{metrics.epoch}.pt"
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": self.config.to_dict(),
                "stage": metrics.stage,
                "epoch": metrics.epoch,
                "history": [asdict(item) for item in history],
            },
            temporary,
        )
        temporary.replace(path)

    def _save_final_artifacts(self, history: Sequence[EpochMetrics]) -> None:
        if self.config.output_dir is None:
            return
        model_dir = self.output_dir / "model"
        tokenizer_dir = self.output_dir / "tokenizer"
        self.model.base_causal_lm.save_pretrained(model_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(tokenizer_dir)
        _write_json(
            self.output_dir / "coconut_config.json",
            {
                "format_version": 1,
                "model_id": self.config.model_id,
                "implementation": self.config.implementation,
                "c": self.config.c,
                "final_stage": self.config.max_latent_stage,
                "num_latent_thoughts": self.config.max_latent_stage * self.config.c,
                "max_new_tokens": self.config.inference_max_new_tokens,
                "special_token_ids": asdict(self.encoder.token_ids),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        self._write_history(history)
        print(f"\nFinal self-contained model saved to: {self.output_dir}")

    def _write_history(self, history: Sequence[EpochMetrics]) -> None:
        _write_json(self.history_path, [asdict(item) for item in history])

    def _to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {name: tensor.to(self.device) for name, tensor in batch.items()}

    def _gpu_memory(self) -> tuple[float | None, float | None]:
        if self.device.type != "cuda":
            return None, None
        divisor = 1024**2
        allocated = torch.cuda.memory_allocated(self.device) / divisor
        peak = torch.cuda.max_memory_allocated(self.device) / divisor
        return allocated, peak

    @staticmethod
    def _print_epoch(metrics: EpochMetrics) -> None:
        memory = ""
        if metrics.gpu_memory_allocated_mb is not None:
            memory = (
                f" | gpu_allocated={metrics.gpu_memory_allocated_mb:.0f} MiB"
                f" | gpu_peak={metrics.gpu_memory_peak_mb:.0f} MiB"
            )
        print(
            f"stage={metrics.stage} epoch={metrics.epoch} "
            f"train_loss={metrics.train_loss:.4f} "
            f"validation_loss={_format_metric(metrics.validation_loss)} "
            f"validation_exact_match={_format_percent(metrics.validation_answer_exact_match)} "
            f"accuracy_examples={metrics.accuracy_examples} "
            f"lr={metrics.learning_rate:.3e} "
            f"epoch_time={_format_duration(metrics.epoch_seconds)} "
            f"elapsed={_format_duration(metrics.elapsed_seconds)}{memory}"
        )

    def _print_qualitative(self, predictions: Sequence[dict[str, str]]) -> None:
        count = min(self.config.qualitative_examples, len(predictions))
        print(f"\n--- Fixed validation examples ({count}) ---")
        for prediction in predictions[:count]:
            print(f"QUESTION:\n{prediction['question']}")
            print(f"EXPECTED:\n{prediction['expected']}")
            print(f"PREDICTED:\n{prediction['raw_generation']}")
            if prediction["raw_generation"] != prediction["predicted"]:
                print(f"EXTRACTED ANSWER:\n{prediction['predicted']}")
            print()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
