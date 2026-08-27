"""Dataset-independent reasoning examples and a JSON/JSONL adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class ReasoningExample:
    question: str
    steps: list[str]
    answer: str


@runtime_checkable
class ReasoningDatasetAdapter(Protocol):
    def load_split(self, split: str) -> Sequence[ReasoningExample]:
        """Load one named split without applying Coconut-specific processing."""


class JsonReasoningDatasetAdapter:
    """Load reasoning examples from JSON arrays or one-object-per-line JSONL."""

    def __init__(self, split_paths: Mapping[str, str | Path]) -> None:
        if not split_paths:
            raise ValueError("split_paths cannot be empty")
        self._split_paths = {name: Path(path) for name, path in split_paths.items()}

    def load_split(self, split: str) -> list[ReasoningExample]:
        try:
            path = self._split_paths[split]
        except KeyError as error:
            available = ", ".join(sorted(self._split_paths))
            raise KeyError(f"Unknown split {split!r}; available splits: {available}") from error

        if not path.is_file():
            raise FileNotFoundError(f"Dataset split does not exist: {path}")

        if path.suffix.lower() == ".jsonl":
            records = self._read_jsonl(path)
        elif path.suffix.lower() == ".json":
            records = self._read_json(path)
        else:
            raise ValueError(f"Expected a .json or .jsonl dataset, got: {path}")

        return [self._parse_record(record, path, index) for index, record in enumerate(records)]

    @staticmethod
    def _read_json(path: Path) -> list[Any]:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise ValueError(f"Top-level JSON value must be an array: {path}")
        return value

    @staticmethod
    def _read_jsonl(path: Path) -> list[Any]:
        records: list[Any] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}: {error}") from error
        return records

    @staticmethod
    def _parse_record(record: Any, path: Path, index: int) -> ReasoningExample:
        location = f"{path} record {index}"
        if not isinstance(record, dict):
            raise ValueError(f"{location} must be an object")

        missing = {"question", "steps", "answer"} - record.keys()
        if missing:
            raise ValueError(f"{location} is missing fields: {', '.join(sorted(missing))}")

        question = record["question"]
        steps = record["steps"]
        answer = record["answer"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{location}.question must be a non-empty string")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{location}.answer must be a non-empty string")
        if not isinstance(steps, list) or not all(isinstance(step, str) for step in steps):
            raise ValueError(f"{location}.steps must be a list of strings")
        if any(not step.strip() for step in steps):
            raise ValueError(f"{location}.steps cannot contain empty strings")

        # Copy the list: the adapter owns no mutable state shared with the JSON parser.
        return ReasoningExample(question=question, steps=list(steps), answer=answer)
