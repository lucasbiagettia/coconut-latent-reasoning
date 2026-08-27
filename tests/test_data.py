import json
import sys
from types import SimpleNamespace

import pytest

from coconut.data import (
    ColumnMapping,
    HuggingFaceDatasetAdapter,
    JsonReasoningDatasetAdapter,
    ReasoningExample,
)


def test_jsonl_adapter_keeps_reasoning_steps_separate(tmp_path):
    path = tmp_path / "train.jsonl"
    record = {"question": "Q", "steps": ["R1", "R2", "R3"], "answer": "A"}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    examples = JsonReasoningDatasetAdapter({"train": path}).load_split("train")

    assert examples == [ReasoningExample("Q", ["R1", "R2", "R3"], "A")]
    assert isinstance(examples[0].steps, list)


def test_json_adapter_rejects_a_joined_reasoning_string(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps([{"question": "Q", "steps": "R1 R2", "answer": "A"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="list of strings"):
        JsonReasoningDatasetAdapter({"train": path}).load_split("train")


def test_huggingface_adapter_loads_a_split_and_maps_columns(monkeypatch):
    calls = []

    def fake_load_dataset(dataset_id, config_name, *, split):
        calls.append((dataset_id, config_name, split))
        return [
            {"prompt": "Q", "reasoning_steps": ["R1", "R2"], "target": "A"}
        ]

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
    )
    adapter = HuggingFaceDatasetAdapter(
        "owner/reasoning-data",
        config_name="default",
        columns=ColumnMapping(
            question="prompt", steps="reasoning_steps", answer="target"
        ),
    )

    examples = adapter.load_split("train")

    assert calls == [("owner/reasoning-data", "default", "train")]
    assert examples == [ReasoningExample("Q", ["R1", "R2"], "A")]
