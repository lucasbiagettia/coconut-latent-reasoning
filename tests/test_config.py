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


def test_entailmentbank_data_configuration(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model_id: EleutherAI/pythia-70m
max_latent_stage: 4
epochs_per_stage: [1, 1, 1, 1, 1]
data:
  type: entailmentbank
  dataset_id: sxiong/entailmentbank
  config_name: task1
  revision: commit
  selection_metadata_path: selection.json
""",
        encoding="utf-8",
    )

    config = TrainingConfig.from_yaml(path)

    assert config.data.type == "entailmentbank"
    assert config.data.config_name == "task1"
    assert config.data.selection_metadata_path == "selection.json"


def test_fp16_8bit_stage_specific_training_configuration(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model_id: EleutherAI/pythia-410m
precision: fp16
optimizer: adamw8bit
optimizer_eps: 1.0e-8
max_latent_stage: 2
epochs_per_stage: [5, 3, 3]
data:
  type: json
  train_path: train.json
  validation_path: validation.json
""",
        encoding="utf-8",
    )

    config = TrainingConfig.from_yaml(path)

    assert config.precision == "fp16"
    assert config.optimizer == "adamw8bit"
    assert [config.epochs_for(stage) for stage in range(3)] == [5, 3, 3]


def test_controlled_160m_config_has_five_stages_and_effective_batch_16():
    config = TrainingConfig.from_yaml(
        "configs/local_pythia160m_entailmentbank_controlled.yaml"
    )

    assert list(range(config.max_latent_stage + 1)) == [0, 1, 2, 3, 4]
    assert [config.epochs_for(stage) for stage in range(5)] == [5, 3, 3, 3, 3]
    assert config.gradient_checkpointing is False
    assert config.batch_size == 2
    assert config.gradient_accumulation_steps == 8
    assert config.effective_batch_size == 16
    assert config.early_stopping_patience == 1
    assert config.restore_best_stage_checkpoint is True
