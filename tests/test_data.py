import json

import pytest

from coconut.data import JsonReasoningDatasetAdapter, ReasoningExample


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
