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
    EntailmentBankAdapter,
    EntailmentBankRecord,
    HuggingFaceDatasetAdapter,
    JsonReasoningDatasetAdapter,
    ReasoningDatasetAdapter,
    ReasoningExample,
    parse_entailmentbank_record,
)

__all__ = [
    "BOT_TOKEN",
    "EOT_TOKEN",
    "LATENT_TOKEN",
    "CurriculumEncoder",
    "EncodedReasoningExample",
    "ColumnMapping",
    "EntailmentBankAdapter",
    "EntailmentBankRecord",
    "HuggingFaceDatasetAdapter",
    "JsonReasoningDatasetAdapter",
    "ReasoningDatasetAdapter",
    "ReasoningExample",
    "parse_entailmentbank_record",
    "apply_curriculum",
]
