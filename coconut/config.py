"""Minimal YAML configuration for Coconut experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    model_id: str
    train_path: str
    validation_path: str
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    epochs_per_stage: int | tuple[int, ...] = 1
    max_latent_stage: int = 1
    c: int = 1
    gradient_accumulation_steps: int = 1
    max_length: int | None = None
    max_train_batches: int | None = None
    max_validation_batches: int | None = None
    inference_max_new_tokens: int = 16
    seed: int = 42
    local_files_only: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
        values: dict[str, Any] = dict(raw)
        if isinstance(values.get("epochs_per_stage"), list):
            values["epochs_per_stage"] = tuple(values["epochs_per_stage"])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        positive = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "c": self.c,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "inference_max_new_tokens": self.inference_max_new_tokens,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.max_latent_stage < 0:
            raise ValueError("max_latent_stage must be >= 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.max_length is not None and self.max_length < 2:
            raise ValueError("max_length must be >= 2")
        if isinstance(self.epochs_per_stage, int):
            if self.epochs_per_stage < 1:
                raise ValueError("epochs_per_stage must be >= 1")
        else:
            expected = self.max_latent_stage + 1
            if len(self.epochs_per_stage) != expected:
                raise ValueError(
                    f"epochs_per_stage must contain {expected} values (stages 0.."
                    f"{self.max_latent_stage})"
                )
            if any(epoch < 1 for epoch in self.epochs_per_stage):
                raise ValueError("Every epochs_per_stage value must be >= 1")

    def epochs_for(self, stage: int) -> int:
        if isinstance(self.epochs_per_stage, int):
            return self.epochs_per_stage
        return self.epochs_per_stage[stage]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
