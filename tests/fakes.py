from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class CharacterTokenizer:
    """Tiny deterministic tokenizer implementing the methods used by the encoder."""

    def __init__(self) -> None:
        self._tokens = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
        self._next_id = 3
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.padding_side = "right"

    def __len__(self) -> int:
        return self._next_id

    def add_special_tokens(self, values: dict[str, object]) -> int:
        added = 0
        for token in values.get("additional_special_tokens", []):
            added += self._add(str(token))
        if "pad_token" in values:
            token = str(values["pad_token"])
            added += self._add(token)
            self.pad_token = token
            self.pad_token_id = self._tokens[token]
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._tokens[token]

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        ids = [self.bos_token_id] if add_special_tokens else []
        for character in text:
            token = f"char:{character}"
            self._add(token)
            ids.append(self._tokens[token])
        return ids

    def _add(self, token: str) -> int:
        if token in self._tokens:
            return 0
        self._tokens[token] = self._next_id
        self._next_id += 1
        return 1


class TinyCausalLM(nn.Module):
    """A differentiable causal stand-in with the Hugging Face LM interface."""

    def __init__(self, vocab_size: int = 512, hidden_size: int = 12) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.forward_calls = 0

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask, output_hidden_states, use_cache, return_dict
        self.forward_calls += 1
        hidden = torch.cumsum(inputs_embeds, dim=1)
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(hidden,))
