from coconut.config import TrainingConfig


def test_nested_huggingface_data_configuration(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model_id: test/model
implementation: batched
data:
  type: huggingface
  dataset_id: owner/data
  config_name: subset
  train_split: training
  validation_split: dev
  columns:
    question: prompt
    steps: rationale
    answer: target
""",
        encoding="utf-8",
    )

    config = TrainingConfig.from_yaml(path)

    assert config.implementation == "batched"
    assert config.data.dataset_id == "owner/data"
    assert config.data.train_split == "training"
    assert config.data.columns.steps == "rationale"
