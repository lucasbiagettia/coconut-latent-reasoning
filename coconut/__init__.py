"""Small, correctness-first implementation of Coconut latent reasoning."""

from .curriculum import (
    BOT_TOKEN,
    EOT_TOKEN,
    LATENT_TOKEN,
    CurriculumEncoder,
    EncodedReasoningExample,
    apply_curriculum,
)
from .data import (
    ColumnMapping,
    HuggingFaceDatasetAdapter,
    JsonReasoningDatasetAdapter,
    ReasoningDatasetAdapter,
    ReasoningExample,
)

__all__ = [
    "BOT_TOKEN",
    "EOT_TOKEN",
    "LATENT_TOKEN",
    "CurriculumEncoder",
    "EncodedReasoningExample",
    "ColumnMapping",
    "HuggingFaceDatasetAdapter",
    "JsonReasoningDatasetAdapter",
    "ReasoningDatasetAdapter",
    "ReasoningExample",
    "apply_curriculum",
]
