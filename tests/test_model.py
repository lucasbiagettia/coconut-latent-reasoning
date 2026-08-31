from copy import deepcopy

import torch
import torch.nn.functional as F

from coconut.curriculum import CoconutCollator, CurriculumEncoder
from coconut.data import ReasoningExample
from coconut.model import CoconutModel
from tests.fakes import CharacterTokenizer, TinyCausalLM


def make_encoded(stage: int = 2, c: int = 1):
    tokenizer = CharacterTokenizer()
    encoder = CurriculumEncoder(tokenizer, c=c)
    item = encoder.encode(
        ReasoningExample("Q", ["R1", "R2", "R3"], "A"), stage=stage
    )
    batch = CoconutCollator(encoder.token_ids.pad)([item])
    return encoder, item, batch


def test_continuous_thoughts_are_hidden_states_reused_as_embeddings():
    encoder, item, batch = make_encoded(stage=2)
    model = CoconutModel(TinyCausalLM(), encoder.token_ids.latent)

    output = model(**batch)

    thoughts = output.latent_hidden_states[0]
    assert len(thoughts) == 2
    for position, thought in zip(item.latent_positions, thoughts):
        torch.testing.assert_close(output.inputs_embeds[:, position, :], thought)


def test_stage_zero_is_one_normal_causal_lm_forward():
    encoder, _, batch = make_encoded(stage=0)
    base_model = TinyCausalLM()
    model = CoconutModel(base_model, encoder.token_ids.latent)

    output = model(**batch)

    assert output.latent_hidden_states == ((),)
    assert base_model.forward_calls == 1
    assert output.loss is not None


def test_second_thought_differentiably_depends_on_first():
    encoder, _, batch = make_encoded(stage=2)
    model = CoconutModel(TinyCausalLM(), encoder.token_ids.latent)
    output = model(**batch)
    h1, h2 = output.latent_hidden_states[0]

    dependency = torch.autograd.grad(h2.sum(), h1, retain_graph=True)[0]

    assert dependency.abs().sum().item() > 0


def test_loss_is_shifted_causal_lm_loss_with_masking():
    encoder, _, batch = make_encoded(stage=1)
    model = CoconutModel(TinyCausalLM(), encoder.token_ids.latent)
    output = model(**batch)

    expected = F.cross_entropy(
        output.logits[:, :-1, :].reshape(-1, output.logits.shape[-1]),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(output.loss, expected)


def test_padding_contributes_no_loss():
    tokenizer = CharacterTokenizer()
    encoder = CurriculumEncoder(tokenizer, c=1)
    items = [
        encoder.encode(ReasoningExample("Q", ["R1"], "A"), stage=1),
        encoder.encode(
            ReasoningExample("A longer Q", ["R1", "R2"], "A longer answer"),
            stage=1,
        ),
    ]
    collator = CoconutCollator(encoder.token_ids.pad)
    model = CoconutModel(TinyCausalLM(), encoder.token_ids.latent)

    batch_output = model(**collator(items))
    individual_outputs = [model(**collator([item])) for item in items]
    expected = sum(
        output.loss * output.target_token_count for output in individual_outputs
    ) / sum(output.target_token_count for output in individual_outputs)

    torch.testing.assert_close(batch_output.loss, expected)


def test_backward_crosses_the_latent_reasoning_chain():
    encoder, item, batch = make_encoded(stage=2)
    base_model = TinyCausalLM()
    model = CoconutModel(base_model, encoder.token_ids.latent)
    prefix_token_id = int(batch["input_ids"][0, item.latent_positions[0] - 1])

    output = model(**batch)
    output.loss.backward()

    gradient = base_model.embedding.weight.grad[prefix_token_id]
    assert gradient.abs().sum().item() > 0


def test_batched_matches_reference_logits_loss_and_gradients():
    torch.manual_seed(0)
    tokenizer = CharacterTokenizer()
    encoder = CurriculumEncoder(tokenizer, c=1)
    examples = [
        ReasoningExample("Short?", ["R1", "R2", "R3"], "A"),
        ReasoningExample("A much longer question?", ["Only one step"], "B"),
        ReasoningExample("Medium question?", ["First", "Second"], "C"),
    ]
    items = [encoder.encode(example, stage=2) for example in examples]
    batch = CoconutCollator(encoder.token_ids.pad)(items)
    reference_base = TinyCausalLM()
    batched_base = deepcopy(reference_base)
    reference = CoconutModel(
        reference_base, encoder.token_ids.latent, implementation="reference"
    )
    batched = CoconutModel(
        batched_base, encoder.token_ids.latent, implementation="batched"
    )
    reference.eval()
    batched.eval()

    reference_output = reference(**batch)
    batched_output = batched(**batch)

    relevant = batch["attention_mask"].bool()
    torch.testing.assert_close(
        batched_output.logits[relevant],
        reference_output.logits[relevant],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        batched_output.loss, reference_output.loss, rtol=1e-5, atol=1e-6
    )

    reference_output.loss.backward()
    batched_output.loss.backward()
    reference_gradients = dict(reference.named_parameters())
    batched_gradients = dict(batched.named_parameters())
    assert reference_gradients.keys() == batched_gradients.keys()
    for name in reference_gradients:
        torch.testing.assert_close(
            batched_gradients[name].grad,
            reference_gradients[name].grad,
            rtol=2e-5,
            atol=2e-6,
            msg=lambda message, name=name: f"Gradient mismatch for {name}: {message}",
        )

    # 3+2+3 reference forwards versus (2 thoughts + final) and
    # (1 thought + final) for the two compatible batched groups.
    assert reference_base.forward_calls == 8
    assert batched_base.forward_calls == 5


def test_batched_stage_zero_parallelizes_normal_cot():
    tokenizer = CharacterTokenizer()
    encoder = CurriculumEncoder(tokenizer, c=1)
    items = [
        encoder.encode(ReasoningExample("Q", ["R1"], "A"), stage=0),
        encoder.encode(
            ReasoningExample("A longer question", ["R1", "R2"], "B"), stage=0
        ),
    ]
    batch = CoconutCollator(encoder.token_ids.pad)(items)
    reference_base = TinyCausalLM()
    batched_base = deepcopy(reference_base)
    reference = CoconutModel(
        reference_base, encoder.token_ids.latent, implementation="reference"
    )
    batched = CoconutModel(
        batched_base, encoder.token_ids.latent, implementation="batched"
    )

    reference_output = reference(**batch)
    batched_output = batched(**batch)

    relevant = batch["attention_mask"].bool()
    torch.testing.assert_close(
        batched_output.logits[relevant], reference_output.logits[relevant]
    )
    torch.testing.assert_close(batched_output.loss, reference_output.loss)
    assert reference_base.forward_calls == 2
    assert batched_base.forward_calls == 1
