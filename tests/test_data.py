import json
import sys
from types import SimpleNamespace

import pytest

from coconut.data import (
    ColumnMapping,
    EntailmentBankAdapter,
    HuggingFaceDatasetAdapter,
    JsonReasoningDatasetAdapter,
    ReasoningExample,
    parse_entailmentbank_record,
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
        token="",
    )

    examples = adapter.load_split("train")

    assert calls == [("owner/reasoning-data", "default", "train")]
    assert examples == [ReasoningExample("Q", ["R1", "R2"], "A")]


def _entailmentbank_row():
    return {
        "id": "science-1",
        "context": "sent1: mammals are warm-blooded sent2: whales are mammals",
        "question": "What can be concluded about whales?",
        "answer": "They are warm-blooded.",
        "hypothesis": "whales are warm-blooded",
        "proof": (
            "sent2 -> int1: whales are mammals; "
            "int1 & sent1 -> hypothesis; "
        ),
        "full_text_proof": (
            "[BECAUSE] whales are mammals [INFER] int1: whales are mammals "
            "[BECAUSE] int1 [AND] mammals are warm-blooded "
            "[INFER] int2: whales are warm-blooded"
        ),
        "depth_of_proof": 2,
        "length_of_proof": 2,
        "meta": {
            "hypothesis_id": "int2",
            "intermediate_conclusions": {
                "int1": "whales are mammals",
                "int2": "whales are warm-blooded",
            },
        },
    }


def test_entailmentbank_parser_uses_gold_conclusions_and_combines_context():
    record = parse_entailmentbank_record(_entailmentbank_row(), source_index=7)

    assert record.id == "science-1"
    assert record.source_index == 7
    assert record.proof_depth == 2
    assert record.example == ReasoningExample(
        question=(
            "Context: sent1: mammals are warm-blooded sent2: whales are mammals\n"
            "Question: What can be concluded about whales?"
        ),
        steps=["whales are mammals", "whales are warm-blooded"],
        answer="They are warm-blooded.",
    )


def test_entailmentbank_parser_rejects_a_rewritten_proof_conclusion():
    row = _entailmentbank_row()
    row["proof"] = (
        "sent2 -> int1: whales might be mammals; int1 & sent1 -> hypothesis;"
    )

    with pytest.raises(ValueError, match="differs between proof and meta"):
        parse_entailmentbank_record(row)


def test_entailmentbank_adapter_passes_auth_and_keeps_selected_id_order(monkeypatch):
    first = _entailmentbank_row()
    second = _entailmentbank_row()
    second["id"] = "science-2"
    calls = []

    def fake_load_dataset(dataset_id, config_name, **kwargs):
        calls.append((dataset_id, config_name, kwargs))
        return [first, second]

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
    )
    adapter = EntailmentBankAdapter(
        selected_rows={
            "train": [
                {"id": "science-2", "source_index": 1},
                {"id": "science-1", "source_index": 0},
            ]
        },
        token="secret",
        revision="commit",
    )

    records = adapter.load_records("train")

    assert [record.id for record in records] == ["science-2", "science-1"]
    assert calls == [
        (
            "sxiong/entailmentbank",
            "task1",
            {"split": "train", "token": "secret", "revision": "commit"},
        )
    ]
