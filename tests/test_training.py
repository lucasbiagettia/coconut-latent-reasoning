from pathlib import Path

import torch

from coconut.config import DataConfig, TrainingConfig
from coconut.curriculum import CurriculumEncoder
from coconut.data import ReasoningExample
from coconut.evaluation import GeneratedAnswer
from coconut.model import CoconutModel
from coconut.training import CurriculumTrainer
from tests.fakes import CharacterTokenizer, TinyCausalLM


def test_trainer_persists_best_and_final_checkpoints(tmp_path, monkeypatch):
    tokenizer = CharacterTokenizer()
    encoder = CurriculumEncoder(tokenizer, c=1)
    base_model = TinyCausalLM()

    def save_model(path, **kwargs):
        del kwargs
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.bin").write_bytes(b"model")

    def save_tokenizer(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    base_model.save_pretrained = save_model
    tokenizer.save_pretrained = save_tokenizer
    model = CoconutModel(base_model, encoder.token_ids.latent)
    example = ReasoningExample("Q", ["gold conclusion"], "correct")
    config = TrainingConfig(
        model_id="tiny",
        data=DataConfig(
            type="json", train_path="train.json", validation_path="validation.json"
        ),
        output_dir=str(tmp_path / "output"),
        device="cpu",
        max_latent_stage=0,
        epochs_per_stage=(1,),
        max_train_batches=1,
        max_validation_batches=1,
        accuracy_max_examples=1,
    )
    monkeypatch.setattr(
        "coconut.training.generate_answer",
        lambda *args, **kwargs: GeneratedAnswer("Answer: correct", "correct"),
    )

    trainer = CurriculumTrainer(
        model,
        tokenizer,
        [example],
        [example],
        config,
        validation_proof_depths=[1],
    )
    history = trainer.train()

    output = tmp_path / "output"
    assert history[0].validation_answer_exact_match == 1.0
    assert history[0].validation_exact_match_by_proof_depth == {"1": 1.0}
    assert (output / "checkpoints" / "latest.pt").is_file()
    assert (output / "checkpoints" / "best.pt").is_file()
    assert (output / "checkpoints" / "final.pt").is_file()
    assert (output / "model" / "weights.bin").is_file()
    assert (output / "best" / "model" / "weights.bin").is_file()
    best = torch.load(
        output / "checkpoints" / "best.pt", weights_only=False
    )
    assert best["best_validation_loss"] == history[0].validation_loss


def test_stage_early_stopping_restores_best_weights_before_next_stage(
    tmp_path, monkeypatch, capsys
):
    tokenizer = CharacterTokenizer()
    encoder = CurriculumEncoder(tokenizer, c=1)
    base_model = TinyCausalLM()

    def save_model(path, **kwargs):
        del kwargs
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.bin").write_bytes(b"model")

    def save_tokenizer(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    base_model.save_pretrained = save_model
    tokenizer.save_pretrained = save_tokenizer
    model = CoconutModel(base_model, encoder.token_ids.latent)
    example = ReasoningExample("Q", ["gold conclusion"], "answer")
    config = TrainingConfig(
        model_id="tiny",
        data=DataConfig(
            type="json", train_path="train.json", validation_path="validation.json"
        ),
        output_dir=str(tmp_path / "controlled"),
        device="cpu",
        max_latent_stage=1,
        epochs_per_stage=(5, 1),
        early_stopping_patience=1,
        early_stopping_min_delta=0.001,
        restore_best_stage_checkpoint=True,
    )
    trainer = CurriculumTrainer(
        model, tokenizer, [example], [example], config
    )
    parameter = next(model.parameters())
    stage_calls = {0: 0, 1: 0}
    stage_one_initial_values = []

    def fake_train_epoch(loader, optimizer):
        del optimizer
        stage = loader.dataset.stage
        if stage == 1:
            stage_one_initial_values.append(
                float(parameter.detach().flatten()[0])
            )
        stage_calls[stage] += 1
        with torch.no_grad():
            parameter.fill_(10 * (stage + 1) + stage_calls[stage])
        return float(stage_calls[stage])

    validation_losses = {0: iter([1.0, 1.2]), 1: iter([0.8])}
    monkeypatch.setattr(trainer, "_train_epoch", fake_train_epoch)
    monkeypatch.setattr(
        trainer,
        "validate_loss",
        lambda loader: next(validation_losses[loader.dataset.stage]),
    )
    monkeypatch.setattr(
        trainer, "evaluate_answers", lambda stage: (None, None, [])
    )

    history = trainer.train()

    assert [(item.stage, item.epoch) for item in history] == [(0, 1), (0, 2), (1, 1)]
    assert stage_one_initial_values == [11.0]
    stage_zero_best = torch.load(
        tmp_path / "controlled" / "checkpoints" / "stage_0_best.pt",
        weights_only=False,
    )
    saved_parameter = next(iter(stage_zero_best["model_state_dict"].values()))
    assert float(saved_parameter.flatten()[0]) == 11.0
    output = capsys.readouterr().out
    assert "early_stopping: stage=0 epoch=2" in output
    assert "best_stage_validation_loss=1.0000" in output
    assert "restored_checkpoint=" in output
