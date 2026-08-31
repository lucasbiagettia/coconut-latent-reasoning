"""Dataset-independent examples plus JSON, Hub, and EntailmentBank adapters."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .huggingface_auth import load_huggingface_token


@dataclass(frozen=True)
class ReasoningExample:
    question: str
    steps: list[str]
    answer: str


@dataclass(frozen=True)
class ColumnMapping:
    question: str = "question"
    steps: str = "steps"
    answer: str = "answer"


@runtime_checkable
class ReasoningDatasetAdapter(Protocol):
    def load_split(self, split: str) -> Sequence[ReasoningExample]:
        """Load one named split without applying Coconut-specific processing."""


class JsonReasoningDatasetAdapter:
    """Load reasoning examples from JSON arrays or one-object-per-line JSONL."""

    def __init__(
        self,
        split_paths: Mapping[str, str | Path],
        columns: ColumnMapping = ColumnMapping(),
    ) -> None:
        if not split_paths:
            raise ValueError("split_paths cannot be empty")
        self._split_paths = {name: Path(path) for name, path in split_paths.items()}
        self._columns = columns

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

        return [
            _parse_record(record, self._columns, f"{path} record {index}")
            for index, record in enumerate(records)
        ]

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



class HuggingFaceDatasetAdapter:
    """Load any Hub/local dataset whose columns can map to Coconut's schema."""

    def __init__(
        self,
        dataset_id: str,
        config_name: str | None = None,
        columns: ColumnMapping = ColumnMapping(),
        token: str | None = None,
        revision: str | None = None,
    ) -> None:
        if not dataset_id:
            raise ValueError("dataset_id cannot be empty")
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.columns = columns
        self.token = token if token is not None else load_huggingface_token()
        self.revision = revision

    def load_split(self, split: str) -> list[ReasoningExample]:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise ImportError(
                "HuggingFaceDatasetAdapter requires the 'datasets' package"
            ) from error

        args = [self.dataset_id]
        if self.config_name is not None:
            args.append(self.config_name)
        kwargs: dict[str, Any] = {"split": split}
        if self.token:
            kwargs["token"] = self.token
        if self.revision:
            kwargs["revision"] = self.revision
        dataset = load_dataset(*args, **kwargs)
        return [
            _parse_record(
                record,
                self.columns,
                f"{self.dataset_id}[{split}] record {index}",
            )
            for index, record in enumerate(dataset)
        ]


@dataclass(frozen=True)
class EntailmentBankRecord:
    """One converted EntailmentBank row with metadata used by experiments."""

    id: str
    example: ReasoningExample
    proof_depth: int
    proof_length: int
    source_index: int


class EntailmentBankAdapter:
    """Convert gold EntailmentBank Task 1 proofs without generating any text."""

    def __init__(
        self,
        dataset_id: str = "sxiong/entailmentbank",
        config_name: str = "task1",
        selected_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        token: str | None = None,
        revision: str | None = None,
    ) -> None:
        if not dataset_id:
            raise ValueError("dataset_id cannot be empty")
        if not config_name:
            raise ValueError("config_name cannot be empty")
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.selected_rows = (
            {split: [dict(row) for row in rows] for split, rows in selected_rows.items()}
            if selected_rows is not None
            else None
        )
        self.token = token if token is not None else load_huggingface_token()
        self.revision = revision

    def load_split(self, split: str) -> list[ReasoningExample]:
        return [record.example for record in self.load_records(split)]

    def load_records(self, split: str) -> list[EntailmentBankRecord]:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise ImportError(
                "EntailmentBankAdapter requires the 'datasets' package"
            ) from error

        kwargs: dict[str, Any] = {"split": split}
        if self.token:
            kwargs["token"] = self.token
        if self.revision:
            kwargs["revision"] = self.revision
        dataset = load_dataset(self.dataset_id, self.config_name, **kwargs)
        records = [
            parse_entailmentbank_record(row, source_index=index)
            for index, row in enumerate(dataset)
        ]

        if self.selected_rows is None:
            return records
        if split not in self.selected_rows:
            available = ", ".join(sorted(self.selected_rows))
            raise KeyError(
                f"No selected EntailmentBank rows for split {split!r}; "
                f"available splits: {available}"
            )
        selected: list[EntailmentBankRecord] = []
        seen_indices: set[int] = set()
        for item in self.selected_rows[split]:
            source_index = item.get("source_index")
            expected_id = item.get("id")
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or not isinstance(expected_id, str)
            ):
                raise ValueError(
                    f"Selected rows for split {split!r} require integer "
                    "source_index and string id"
                )
            if source_index in seen_indices:
                raise ValueError(
                    f"Selected rows for split {split!r} repeat source_index="
                    f"{source_index}"
                )
            seen_indices.add(source_index)
            if source_index < 0 or source_index >= len(records):
                raise ValueError(
                    f"Selected source_index={source_index} is outside split {split!r}"
                )
            record = records[source_index]
            if record.id != expected_id:
                raise ValueError(
                    f"Selected row {split}[{source_index}] expected id={expected_id!r}, "
                    f"found {record.id!r}"
                )
            selected.append(record)
        return selected


_PROOF_TARGET = re.compile(r"^(int\d+)\s*:\s*(.+)$", flags=re.DOTALL)


def parse_entailmentbank_record(
    row: Mapping[str, Any], *, source_index: int = -1
) -> EntailmentBankRecord:
    """Parse proof conclusions and cross-check all three gold proof fields.

    EntailmentBank's structured ``proof`` gives inference order.  Explicit
    ``intN: text`` targets are checked against ``meta.intermediate_conclusions``.
    The terminal ``hypothesis`` target is resolved through ``meta.hypothesis_id``;
    no conclusion is generated or paraphrased.
    """

    required = {
        "id",
        "context",
        "question",
        "answer",
        "hypothesis",
        "proof",
        "full_text_proof",
        "depth_of_proof",
        "length_of_proof",
        "meta",
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(
            f"EntailmentBank row is missing fields: {', '.join(sorted(missing))}"
        )
    record_id = _nonempty_string(row["id"], "id")
    context = _nonempty_string(row["context"], f"{record_id}.context")
    question = _nonempty_string(row["question"], f"{record_id}.question")
    answer = _nonempty_string(row["answer"], f"{record_id}.answer")
    hypothesis = _nonempty_string(row["hypothesis"], f"{record_id}.hypothesis")
    proof = _nonempty_string(row["proof"], f"{record_id}.proof")
    full_text_proof = _nonempty_string(
        row["full_text_proof"], f"{record_id}.full_text_proof"
    )
    meta = row["meta"]
    if not isinstance(meta, Mapping):
        raise ValueError(f"{record_id}.meta must be an object")
    conclusions = meta.get("intermediate_conclusions")
    if isinstance(conclusions, str):
        try:
            conclusions = json.loads(conclusions)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{record_id}.meta.intermediate_conclusions is invalid JSON"
            ) from error
    if not isinstance(conclusions, Mapping):
        raise ValueError(
            f"{record_id}.meta.intermediate_conclusions must be an object"
        )
    hypothesis_id = _nonempty_string(
        meta.get("hypothesis_id"), f"{record_id}.meta.hypothesis_id"
    )

    steps: list[str] = []
    clauses = [clause.strip() for clause in proof.split(";") if clause.strip()]
    for clause_index, clause in enumerate(clauses, start=1):
        premises, arrow, target = clause.partition("->")
        if not arrow or not premises.strip() or not target.strip():
            raise ValueError(
                f"{record_id}.proof clause {clause_index} is malformed: {clause!r}"
            )
        target = target.strip()
        if target == "hypothesis":
            conclusion_id = hypothesis_id
            conclusion = _conclusion_for(conclusions, conclusion_id, record_id)
            if _normalized(conclusion) != _normalized(hypothesis):
                raise ValueError(
                    f"{record_id}: terminal proof conclusion does not match hypothesis"
                )
        else:
            match = _PROOF_TARGET.fullmatch(target)
            if match is None:
                raise ValueError(
                    f"{record_id}.proof clause {clause_index} has unsupported target: "
                    f"{target!r}"
                )
            conclusion_id, conclusion = match.groups()
            conclusion = conclusion.strip()
            metadata_conclusion = _conclusion_for(
                conclusions, conclusion_id, record_id
            )
            if _normalized(conclusion) != _normalized(metadata_conclusion):
                raise ValueError(
                    f"{record_id}: {conclusion_id} differs between proof and meta"
                )
        if _normalized(conclusion) not in _normalized(full_text_proof):
            raise ValueError(
                f"{record_id}: {conclusion_id} is absent from full_text_proof"
            )
        steps.append(conclusion.strip())

    proof_depth = _nonnegative_int(
        row["depth_of_proof"], f"{record_id}.depth_of_proof"
    )
    proof_length = _positive_int(
        row["length_of_proof"], f"{record_id}.length_of_proof"
    )
    if len(steps) != proof_length:
        raise ValueError(
            f"{record_id}: parsed {len(steps)} proof steps, expected {proof_length}"
        )
    combined_input = f"Context: {context}\nQuestion: {question}"
    return EntailmentBankRecord(
        id=record_id,
        example=ReasoningExample(
            question=combined_input,
            steps=steps,
            answer=answer,
        ),
        proof_depth=proof_depth,
        proof_length=proof_length,
        source_index=source_index,
    )


def _conclusion_for(
    conclusions: Mapping[str, Any], conclusion_id: str, record_id: str
) -> str:
    if conclusion_id not in conclusions:
        raise ValueError(
            f"{record_id}.meta.intermediate_conclusions lacks {conclusion_id!r}"
        )
    return _nonempty_string(
        conclusions[conclusion_id],
        f"{record_id}.meta.intermediate_conclusions.{conclusion_id}",
    )


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _nonnegative_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return value


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def _parse_record(
    record: Any, columns: ColumnMapping, location: str
) -> ReasoningExample:
    if not isinstance(record, Mapping):
        raise ValueError(f"{location} must be an object")

    names = {columns.question, columns.steps, columns.answer}
    missing = names - record.keys()
    if missing:
        raise ValueError(f"{location} is missing columns: {', '.join(sorted(missing))}")

    question = record[columns.question]
    steps = record[columns.steps]
    answer = record[columns.answer]
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{location}.{columns.question} must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{location}.{columns.answer} must be a non-empty string")
    if not isinstance(steps, list) or not all(isinstance(step, str) for step in steps):
        raise ValueError(f"{location}.{columns.steps} must be a list of strings")
    if any(not step.strip() for step in steps):
        raise ValueError(f"{location}.{columns.steps} cannot contain empty strings")

    return ReasoningExample(question=question, steps=list(steps), answer=answer)
