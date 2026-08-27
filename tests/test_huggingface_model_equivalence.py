from copy import deepcopy

import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from coconut.curriculum import CoconutCollator, CurriculumEncoder
from coconut.data import ReasoningExample
from coconut.model import CoconutModel
from tests.fakes import CharacterTokenizer


def test_batched_matches_reference_with_a_real_transformer():
    tokenizer = CharacterTokenizer()
    encoder = CurriculumEncoder(tokenizer, c=1)
    examples = [
        ReasoningExample("Short?", ["R1", "R2", "R3"], "A"),
        ReasoningExample("A longer question?", ["Only one"], "B"),
        ReasoningExample("Medium?", ["First", "Second"], "C"),
    ]
    batch = CoconutCollator(encoder.token_ids.pad)(
        [encoder.encode(example, stage=2) for example in examples]
    )
    config = GPTNeoXConfig(
        vocab_size=512,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=256,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        rotary_pct=0.5,
        use_cache=False,
    )
    reference_base = GPTNeoXForCausalLM(config)
    reference = CoconutModel(
        reference_base, encoder.token_ids.latent, implementation="reference"
    ).eval()
    batched = CoconutModel(
        deepcopy(reference_base),
        encoder.token_ids.latent,
        implementation="batched",
    ).eval()

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
        batched_output.loss, reference_output.loss, rtol=1e-6, atol=1e-7
    )

    reference_output.loss.backward()
    batched_output.loss.backward()
    for (reference_name, reference_parameter), (batched_name, batched_parameter) in zip(
        reference.named_parameters(), batched.named_parameters(), strict=True
    ):
        assert reference_name == batched_name
        torch.testing.assert_close(
            batched_parameter.grad,
            reference_parameter.grad,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda message, name=reference_name: (
                f"Gradient mismatch for {name}: {message}"
            ),
        )
