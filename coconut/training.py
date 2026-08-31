"""A small observable, resumable trainer for the Coconut curriculum."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer
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
    validation_exact_match_by_proof_depth: dict[str, float] | None = None


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
        validation_proof_depths: Sequence[int] | None = None,
    ) -> None:
        if not train_examples or not validation_examples:
            raise ValueError("Training and validation splits must both be non-empty")
        self.model = model
        self.tokenizer = tokenizer
        self.train_examples = train_examples
        self.validation_examples = validation_examples
        if validation_proof_depths is not None and len(validation_proof_depths) != len(
            validation_examples
        ):
            raise ValueError(
                "validation_proof_depths must align with validation_examples"
            )
        self.validation_proof_depths = (
            list(validation_proof_depths)
            if validation_proof_depths is not None
            else None
        )
        self.config = config
        self.device = resolve_device(config.device)
        self.model.to(self.device)
        self.amp_enabled = config.precision == "fp16"
        if self.amp_enabled and self.device.type != "cuda":
            raise ValueError("FP16 training requires a CUDA device")
        self.scaler = DynamicLossScaler(enabled=self.amp_enabled, device=self.device)
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
        best_validation_loss = _best_validation_loss(history)
        start_stage, start_epoch = self._resume_position(resume)
        elapsed_offset = history[-1].elapsed_seconds if history else 0.0
        run_started = time.perf_counter()

        if start_stage > self.config.max_latent_stage:
            print("Checkpoint already completed the configured curriculum.")
            self._save_final_artifacts(history)
            return history

        print(
            "training_runtime: "
            f"gradient_checkpointing={'ON' if self.config.gradient_checkpointing else 'OFF'} "
            f"batch_size={self.config.batch_size} "
            f"gradient_accumulation_steps={self.config.gradient_accumulation_steps} "
            f"effective_batch_size={self.config.effective_batch_size} "
            f"precision={self.config.precision} optimizer={self.config.optimizer}"
        )

        for stage in range(start_stage, self.config.max_latent_stage + 1):
            optimizer = build_optimizer(
                self.model.parameters(), self.config, self.device
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
            stage_history = [item for item in history if item.stage == stage]
            stage_best_loss = min(
                (
                    item.validation_loss
                    for item in stage_history
                    if item.validation_loss is not None
                ),
                default=None,
            )
            stage_best_epoch = next(
                (
                    item.epoch
                    for item in stage_history
                    if item.validation_loss == stage_best_loss
                ),
                None,
            )
            non_improving_epochs = _non_improving_tail(
                stage_history, self.config.early_stopping_min_delta
            )
            stage_best_path = self.checkpoint_dir / f"stage_{stage}_best.pt"
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
                accuracy_by_depth: dict[str, float] | None = None
                predictions: list[dict[str, str]] = []
                if should_evaluate:
                    validation_loss = self.validate_loss(validation_loader)
                    answer_accuracy, accuracy_by_depth, predictions = (
                        self.evaluate_answers(stage)
                    )

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
                    validation_exact_match_by_proof_depth=accuracy_by_depth,
                )
                history.append(metrics)
                last_metrics = metrics
                self._write_history(history)
                self._print_epoch(metrics)

                is_stage_best = (
                    validation_loss is not None
                    and (
                        stage_best_loss is None
                        or validation_loss
                        < stage_best_loss - self.config.early_stopping_min_delta
                    )
                )
                if is_stage_best:
                    stage_best_loss = validation_loss
                    stage_best_epoch = epoch
                    non_improving_epochs = 0
                    self._save_stage_best_checkpoint(
                        stage_best_path,
                        metrics,
                        optimizer,
                        history,
                        best_validation_loss=best_validation_loss,
                    )
                elif validation_loss is not None:
                    non_improving_epochs += 1

                is_global_best = (
                    validation_loss is not None
                    and (
                        best_validation_loss is None
                        or validation_loss < best_validation_loss
                    )
                )
                if is_global_best:
                    best_validation_loss = validation_loss
                    self._save_best_checkpoint(metrics, optimizer, history)

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
                    self._save_checkpoint(
                        metrics,
                        optimizer,
                        history,
                        best_validation_loss=best_validation_loss,
                        stage_complete=False,
                    )

                patience = self.config.early_stopping_patience
                if patience is not None and non_improving_epochs >= patience:
                    print(
                        f"early_stopping: stage={stage} epoch={epoch} "
                        f"patience={patience} best_stage_validation_loss="
                        f"{_format_metric(stage_best_loss)}"
                    )
                    break

            if self.config.restore_best_stage_checkpoint:
                if stage_best_loss is None or not stage_best_path.is_file():
                    raise RuntimeError(
                        f"Stage {stage} has no validation-loss checkpoint to restore"
                    )
                self._restore_stage_checkpoint(stage_best_path, optimizer)
                print(
                    f"best_stage_validation_loss={stage_best_loss:.4f} "
                    f"best_stage_epoch={stage_best_epoch} "
                    f"restored_checkpoint={stage_best_path}"
                )

            if last_metrics is not None:
                self._save_checkpoint(
                    last_metrics,
                    optimizer,
                    history,
                    best_validation_loss=best_validation_loss,
                    stage_complete=True,
                    restored_stage_best_epoch=(
                        stage_best_epoch
                        if self.config.restore_best_stage_checkpoint
                        else None
                    ),
                )

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

    def _train_epoch(self, loader: DataLoader, optimizer: Optimizer) -> float:
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
            with self._autocast():
                output = self.model(**self._to_device(batch))
            assert output.loss is not None
            group_start = ((batch_index - 1) // accumulation) * accumulation + 1
            group_size = min(accumulation, batches_to_process - group_start + 1)
            self.scaler.scale(output.loss / group_size).backward()
            losses.append(float(output.loss.detach().cpu()))

            if batch_index % accumulation == 0 or batch_index == batches_to_process:
                stepped = self.scaler.step(optimizer, self.model.parameters())
                if not stepped:
                    print(
                        "Skipped optimizer step after non-finite fp16 gradients; "
                        f"new loss_scale={self.scaler.scale_value:.0f}"
                    )
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
            with self._autocast():
                output = self.model(**self._to_device(batch))
            assert output.loss is not None
            weighted_loss += float(output.loss.cpu()) * output.target_token_count
            target_count += output.target_token_count
        if target_count == 0:
            raise RuntimeError("Validation loader produced no target tokens")
        return weighted_loss / target_count

    def evaluate_answers(
        self, stage: int
    ) -> tuple[float | None, dict[str, float] | None, list[dict[str, str]]]:
        limit = min(self.config.accuracy_max_examples, len(self.validation_examples))
        if limit == 0:
            return None, None, []

        self.model.eval()
        predictions: list[dict[str, str]] = []
        correct = 0
        depth_totals: dict[int, int] = {}
        depth_correct: dict[int, int] = {}
        for index, example in enumerate(self.validation_examples[:limit]):
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
            proof_depth = (
                self.validation_proof_depths[index]
                if self.validation_proof_depths is not None
                else None
            )
            if proof_depth is not None:
                depth_totals[proof_depth] = depth_totals.get(proof_depth, 0) + 1
                depth_correct[proof_depth] = depth_correct.get(proof_depth, 0) + int(
                    is_correct
                )
            predictions.append(
                {
                    "question": example.question,
                    "expected": example.answer,
                    "predicted": generated.answer,
                    "raw_generation": generated.raw_text,
                    "proof_depth": str(proof_depth) if proof_depth is not None else "",
                }
            )
        by_depth = (
            {
                str(depth): depth_correct.get(depth, 0) / total
                for depth, total in sorted(depth_totals.items())
            }
            if depth_totals
            else None
        )
        return correct / limit, by_depth, predictions

    def _prepare_output_directory(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_dir = self.output_dir / "tokenizer"
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(tokenizer_dir)
        _write_json(self.output_dir / "training_config.json", self.config.to_dict())
        metadata_path = self.config.data.selection_metadata_path
        if self.config.output_dir and metadata_path:
            source = Path(metadata_path)
            if not source.is_file():
                raise FileNotFoundError(
                    f"Experiment selection metadata does not exist: {source}"
                )
            _write_json(
                self.output_dir / "experiment_metadata.json",
                json.loads(source.read_text(encoding="utf-8")),
            )

    def _load_resume_checkpoint(self) -> dict[str, Any] | None:
        if self.config.resume_from_checkpoint is None:
            return None
        path = Path(self.config.resume_from_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
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
        if checkpoint.get("stage_complete", False):
            return stage + 1, 1
        epoch = int(checkpoint["epoch"]) + 1
        if epoch > self.config.epochs_for(stage):
            return stage + 1, 1
        return stage, epoch

    def _save_checkpoint(
        self,
        metrics: EpochMetrics,
        optimizer: Optimizer,
        history: Sequence[EpochMetrics],
        *,
        best_validation_loss: float | None,
        stage_complete: bool,
        restored_stage_best_epoch: int | None = None,
    ) -> None:
        if self.config.output_dir:
            path = self.checkpoint_dir / "latest.pt"
        else:
            path = self.checkpoint_dir / f"stage_{metrics.stage}_epoch_{metrics.epoch}.pt"
        payload = self._checkpoint_payload(
            metrics,
            optimizer,
            history,
            best_validation_loss=best_validation_loss,
            stage_complete=stage_complete,
            restored_stage_best_epoch=restored_stage_best_epoch,
        )
        _write_torch_checkpoint(path, payload)
        if stage_complete and metrics.stage == self.config.max_latent_stage:
            _write_torch_checkpoint(self.checkpoint_dir / "final.pt", payload)

    def _save_stage_best_checkpoint(
        self,
        path: Path,
        metrics: EpochMetrics,
        optimizer: Optimizer,
        history: Sequence[EpochMetrics],
        *,
        best_validation_loss: float | None,
    ) -> None:
        payload = self._checkpoint_payload(
            metrics,
            optimizer,
            history,
            best_validation_loss=best_validation_loss,
            stage_complete=False,
            restored_stage_best_epoch=metrics.epoch,
        )
        _write_torch_checkpoint(path, payload)

    def _restore_stage_checkpoint(
        self, path: Path, optimizer: Optimizer
    ) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    def _save_best_checkpoint(
        self,
        metrics: EpochMetrics,
        optimizer: Optimizer,
        history: Sequence[EpochMetrics],
    ) -> None:
        assert metrics.validation_loss is not None
        if self.config.output_dir is None:
            return
        payload = self._checkpoint_payload(
            metrics,
            optimizer,
            history,
            best_validation_loss=metrics.validation_loss,
            stage_complete=False,
            restored_stage_best_epoch=metrics.epoch,
        )
        _write_torch_checkpoint(self.checkpoint_dir / "best.pt", payload)
        self._save_inference_artifacts(
            self.output_dir / "best",
            stage=metrics.stage,
            extra_metadata={
                "selection_metric": "validation_loss",
                "selection_metric_value": metrics.validation_loss,
                "selected_stage": metrics.stage,
                "selected_epoch": metrics.epoch,
            },
        )
        print(
            f"New global best validation loss: {metrics.validation_loss:.4f} "
            f"(stage={metrics.stage}, epoch={metrics.epoch})"
        )

    def _checkpoint_payload(
        self,
        metrics: EpochMetrics,
        optimizer: Optimizer,
        history: Sequence[EpochMetrics],
        *,
        best_validation_loss: float | None,
        stage_complete: bool,
        restored_stage_best_epoch: int | None,
    ) -> dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config.to_dict(),
            "stage": metrics.stage,
            "epoch": metrics.epoch,
            "stage_complete": stage_complete,
            "restored_stage_best_epoch": restored_stage_best_epoch,
            "best_validation_loss": best_validation_loss,
            "history": [asdict(item) for item in history],
        }

    def _save_final_artifacts(self, history: Sequence[EpochMetrics]) -> None:
        if self.config.output_dir is None:
            return
        self._save_inference_artifacts(
            self.output_dir, stage=self.config.max_latent_stage
        )
        self._write_history(history)
        print(f"\nFinal self-contained model saved to: {self.output_dir}")

    def _write_history(self, history: Sequence[EpochMetrics]) -> None:
        _write_json(self.history_path, [asdict(item) for item in history])

    def _to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {name: tensor.to(self.device) for name, tensor in batch.items()}

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        )

    def _save_inference_artifacts(
        self,
        directory: Path,
        *,
        stage: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model.base_causal_lm.save_pretrained(
            directory / "model", safe_serialization=True
        )
        self.tokenizer.save_pretrained(directory / "tokenizer")
        metadata: dict[str, Any] = {
            "format_version": 1,
            "model_id": self.config.model_id,
            "implementation": self.config.implementation,
            "precision": self.config.precision,
            "c": self.config.c,
            "final_stage": stage,
            "num_latent_thoughts": stage * self.config.c,
            "max_new_tokens": self.config.inference_max_new_tokens,
            "special_token_ids": asdict(self.encoder.token_ids),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        _write_json(directory / "coconut_config.json", metadata)

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
        depth_metric = ""
        if metrics.validation_exact_match_by_proof_depth:
            values = ",".join(
                f"d{depth}:{100 * accuracy:.2f}%"
                for depth, accuracy in metrics.validation_exact_match_by_proof_depth.items()
            )
            depth_metric = f" exact_match_by_depth=[{values}]"
        print(
            f"stage={metrics.stage} epoch={metrics.epoch} "
            f"train_loss={metrics.train_loss:.4f} "
            f"validation_loss={_format_metric(metrics.validation_loss)} "
            f"validation_exact_match={_format_percent(metrics.validation_answer_exact_match)} "
            f"accuracy_examples={metrics.accuracy_examples} "
            f"lr={metrics.learning_rate:.3e} "
            f"epoch_time={_format_duration(metrics.epoch_seconds)} "
            f"elapsed={_format_duration(metrics.elapsed_seconds)}{depth_metric}{memory}"
        )

    def _print_qualitative(self, predictions: Sequence[dict[str, str]]) -> None:
        count = min(self.config.qualitative_examples, len(predictions))
        print(f"\n--- Fixed validation examples ({count}) ---")
        for prediction in predictions[:count]:
            print(f"QUESTION:\n{prediction['question']}")
            print(f"EXPECTED:\n{prediction['expected']}")
            print(f"PREDICTED:\n{prediction['predicted']}")
            if prediction["raw_generation"] != prediction["predicted"]:
                print(f"RAW GENERATION:\n{prediction['raw_generation']}")
            print()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_torch_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: TrainingConfig,
    device: torch.device,
) -> Optimizer:
    """Construct the configured full-finetuning optimizer."""

    kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "eps": config.optimizer_eps,
    }
    if config.optimizer == "adamw":
        return AdamW(parameters, **kwargs)
    if device.type != "cuda":
        raise RuntimeError("adamw8bit requires CUDA")
    try:
        import bitsandbytes as bnb
    except ImportError as error:
        raise ImportError(
            "optimizer=adamw8bit requires bitsandbytes; install "
            "requirements-gpu.txt in the experiment environment"
        ) from error
    return bnb.optim.AdamW8bit(parameters, **kwargs)


class DynamicLossScaler:
    """Dynamic loss scaling that supports memory-saving fp16 parameters.

    PyTorch's public GradScaler assumes fp32 master parameters and rejects fp16
    gradients. Coconut's 3 GiB profile keeps the actual parameters in fp16, so
    this small scaler uses PyTorch's fused finite-check/unscale CUDA primitive
    before handing unscaled gradients to the optimizer.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        device: torch.device,
        initial_scale: float = 256.0,
        growth_interval: int = 2000,
    ) -> None:
        self.enabled = enabled
        self.device = device
        self.scale_value = initial_scale
        self.growth_interval = growth_interval
        self.successful_steps = 0

    def is_enabled(self) -> bool:
        return self.enabled

    def scale(self, loss: Tensor) -> Tensor:
        return loss * self.scale_value if self.enabled else loss

    def step(
        self,
        optimizer: Optimizer,
        parameters: Iterable[torch.nn.Parameter],
    ) -> bool:
        if not self.enabled:
            optimizer.step()
            return True
        gradients = [
            parameter.grad
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("No gradients were produced")
        found_inf = torch.zeros((), dtype=torch.float32, device=self.device)
        inverse_scale = torch.full(
            (), 1.0 / self.scale_value, dtype=torch.float32, device=self.device
        )
        torch._amp_foreach_non_finite_check_and_unscale_(
            gradients, found_inf, inverse_scale
        )
        if float(found_inf.cpu()) != 0.0:
            self.scale_value = max(self.scale_value / 2.0, 1.0)
            self.successful_steps = 0
            return False
        optimizer.step()
        self.successful_steps += 1
        if self.successful_steps >= self.growth_interval:
            self.scale_value *= 2.0
            self.successful_steps = 0
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "scale_value": self.scale_value,
            "growth_interval": self.growth_interval,
            "successful_steps": self.successful_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.scale_value = float(state.get("scale_value", self.scale_value))
        self.growth_interval = int(
            state.get("growth_interval", self.growth_interval)
        )
        self.successful_steps = int(state.get("successful_steps", 0))


def _best_validation_loss(history: Sequence[EpochMetrics]) -> float | None:
    values = [
        item.validation_loss
        for item in history
        if item.validation_loss is not None
    ]
    return min(values) if values else None


def _non_improving_tail(
    history: Sequence[EpochMetrics], min_delta: float
) -> int:
    best: float | None = None
    tail = 0
    for item in history:
        value = item.validation_loss
        if value is None:
            continue
        if best is None or value < best - min_delta:
            best = value
            tail = 0
        else:
            tail += 1
    return tail


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
