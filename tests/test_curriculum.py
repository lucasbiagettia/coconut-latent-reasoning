from coconut.curriculum import (
    IGNORE_INDEX,
    CoconutCollator,
    CurriculumEncoder,
    apply_curriculum,
)
from coconut.data import ReasoningExample
from tests.fakes import CharacterTokenizer


def example() -> ReasoningExample:
    return ReasoningExample("Question", ["R1", "R2", "R3"], "Answer")


def test_curriculum_removes_reasoning_steps_progressively():
    expected = [
        ["R1", "R2", "R3"],
        ["R2", "R3"],
        ["R3"],
        [],
    ]
    for stage, visible in enumerate(expected):
        view = apply_curriculum(example(), stage=stage, c=1)
        assert view.visible_steps == visible
        assert view.removed_steps == example().steps[:stage]


def test_c_controls_number_of_continuous_thoughts():
    assert apply_curriculum(example(), stage=0, c=2).num_latent_thoughts == 0
    assert apply_curriculum(example(), stage=1, c=2).num_latent_thoughts == 2
    assert apply_curriculum(example(), stage=2, c=2).num_latent_thoughts == 4
    assert apply_curriculum(example(), stage=99, c=2).num_latent_thoughts == 6


def test_stage_zero_is_full_language_cot_and_masks_only_the_prompt_markers():
    encoder = CurriculumEncoder(CharacterTokenizer(), c=2)
    encoded = encoder.encode(example(), stage=0)

    assert encoded.latent_positions == ()
    assert encoded.removed_step_count == 0
    assert list(encoded.visible_steps) == example().steps
    first_target = next(i for i, label in enumerate(encoded.labels) if label != IGNORE_INDEX)
    assert all(label == IGNORE_INDEX for label in encoded.labels[:first_target])
    assert tuple(encoded.labels[first_target:]) == encoded.input_ids[first_target:]


def test_collator_masks_padding():
    encoder = CurriculumEncoder(CharacterTokenizer(), c=1)
    short = encoder.encode(ReasoningExample("Q", ["R1"], "A"), stage=1)
    long = encoder.encode(example(), stage=1)
    batch = CoconutCollator(encoder.token_ids.pad)([short, long])

    short_padding = batch["attention_mask"][0] == 0
    assert short_padding.any()
    assert (batch["labels"][0][short_padding] == IGNORE_INDEX).all()
