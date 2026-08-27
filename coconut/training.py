"""A small single-process trainer for the Coconut curriculum."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .curriculum import CoconutCollator, CurriculumEncoder, StageDataset
from .data import ReasoningExample
from .model import CoconutModel


@dataclass(frozen=True)
class EpochMetrics:
    stage: int
    epoch: int
    train_loss: float
    validation_loss: float


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
        self.checkpoint_dir = Path(config.checkpoint_dir)

    def train(self) -> list[EpochMetrics]:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(self.checkpoint_dir / "tokenizer")

        history: list[EpochMetrics] = []
        for stage in range(self.config.max_latent_stage + 1):
            # The paper resets optimizer state at every stage transition.
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            train_loader = self._loader(self.train_examples, stage, shuffle=True)
            validation_loader = self._loader(
                self.validation_examples, stage, shuffle=False
            )
            latent_count = min(stage, max(len(x.steps) for x in self.train_examples)) * self.config.c
            print(
                f"Stage {stage}/{self.config.max_latent_stage}: "
                f"up to {latent_count} continuous thoughts per example"
            )

            for epoch in range(1, self.config.epochs_for(stage) + 1):
                train_loss = self._train_epoch(train_loader, optimizer)
                validation_loss = self.validate(validation_loader)
                metrics = EpochMetrics(stage, epoch, train_loss, validation_loss)
                history.append(metrics)
                print(
                    f"  epoch {epoch}: train_loss={train_loss:.4f} "
                    f"validation_loss={validation_loss:.4f}"
                )
                self._save_checkpoint(metrics, optimizer)
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
        processed_batches = 0
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
            processed_batches += 1

            if batch_index % accumulation == 0 or batch_index == batches_to_process:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if not losses:
            raise RuntimeError("Training loader produced no batches")
        return sum(losses) / len(losses)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
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

    def _to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {name: tensor.to(self.device) for name, tensor in batch.items()}

    def _save_checkpoint(self, metrics: EpochMetrics, optimizer: AdamW) -> None:
        path = self.checkpoint_dir / f"stage_{metrics.stage}_epoch_{metrics.epoch}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": self.config.to_dict(),
                "stage": metrics.stage,
                "epoch": metrics.epoch,
                "train_loss": metrics.train_loss,
                "validation_loss": metrics.validation_loss,
            },
            path,
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
