"""Generic causal-LM wrapper implementing sequential continuous thoughts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .curriculum import IGNORE_INDEX


@dataclass
class CoconutOutput:
    loss: Tensor | None
    logits: Tensor
    inputs_embeds: Tensor
    latent_hidden_states: tuple[tuple[Tensor, ...], ...]
    target_token_count: int


class CoconutModel(nn.Module):
    """Use a causal LM's final hidden state as the next input embedding.

    This intentionally performs a complete prefix forward for every continuous
    thought, followed by a final complete forward for the language loss. It is
    slower than a KV-cache implementation, but directly matches the algorithm.
    """

    def __init__(self, base_causal_lm: nn.Module, latent_token_id: int) -> None:
        super().__init__()
        self.base_causal_lm = base_causal_lm
        self.latent_token_id = latent_token_id

    def get_input_embeddings(self) -> nn.Module:
        embeddings = self.base_causal_lm.get_input_embeddings()
        if embeddings is None:
            raise TypeError("The causal LM does not expose input embeddings")
        return embeddings

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> CoconutOutput:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")

        row_logits: list[Tensor] = []
        row_embeds: list[Tensor] = []
        all_thoughts: list[tuple[Tensor, ...]] = []
        total_loss = input_ids.new_zeros((), dtype=torch.float32)
        total_targets = 0

        for row in range(input_ids.shape[0]):
            length = int(attention_mask[row].sum().item())
            if length < 2:
                raise ValueError("Each sequence must contain at least two non-padding tokens")
            if not torch.all(attention_mask[row, :length] == 1) or torch.any(
                attention_mask[row, length:] != 0
            ):
                raise ValueError("CoconutModel expects right-padded attention masks")

            row_ids = input_ids[row : row + 1, :length]
            embeds, thoughts = self._replace_latent_placeholders(row_ids)
            outputs = self._base_forward(embeds)
            logits = outputs.logits
            row_logits.append(logits)
            row_embeds.append(embeds)
            all_thoughts.append(thoughts)

            if labels is not None:
                row_labels = labels[row : row + 1, :length]
                shift_labels = row_labels[:, 1:].contiguous()
                target_count = int((shift_labels != IGNORE_INDEX).sum().item())
                if target_count:
                    total_loss = total_loss + F.cross_entropy(
                        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                        shift_labels.view(-1),
                        ignore_index=IGNORE_INDEX,
                        reduction="sum",
                    )
                    total_targets += target_count

        max_length = max(tensor.shape[1] for tensor in row_logits)
        padded_logits = torch.cat(
            [F.pad(tensor, (0, 0, 0, max_length - tensor.shape[1])) for tensor in row_logits],
            dim=0,
        )
        padded_embeds = torch.cat(
            [F.pad(tensor, (0, 0, 0, max_length - tensor.shape[1])) for tensor in row_embeds],
            dim=0,
        )
        loss = None
        if labels is not None:
            if total_targets == 0:
                raise ValueError("Batch has no unmasked target tokens")
            loss = total_loss / total_targets

        return CoconutOutput(
            loss=loss,
            logits=padded_logits,
            inputs_embeds=padded_embeds,
            latent_hidden_states=tuple(all_thoughts),
            target_token_count=total_targets,
        )

    def _replace_latent_placeholders(
        self, input_ids: Tensor
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        latent_positions = (input_ids[0] == self.latent_token_id).nonzero(as_tuple=False).flatten()
        if latent_positions.numel() > 1 and not torch.all(
            latent_positions[1:] == latent_positions[:-1] + 1
        ):
            raise ValueError("Latent placeholders must form one contiguous block")

        inputs_embeds = self.get_input_embeddings()(input_ids)
        thoughts: list[Tensor] = []
        for position_tensor in latent_positions:
            position = int(position_tensor.item())
            if position == 0:
                raise ValueError("A latent thought needs a preceding token")
            prefix_outputs = self._base_forward(inputs_embeds[:, :position, :])
            thought = prefix_outputs.hidden_states[-1][:, -1, :]
            thoughts.append(thought)
            # Rebuild instead of assigning in-place, preserving the complete graph.
            inputs_embeds = torch.cat(
                (
                    inputs_embeds[:, :position, :],
                    thought.unsqueeze(1),
                    inputs_embeds[:, position + 1 :, :],
                ),
                dim=1,
            )
        return inputs_embeds, tuple(thoughts)

    def _base_forward(self, inputs_embeds: Tensor) -> Any:
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )
        return self.base_causal_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        eos_token_id: int,
        max_new_tokens: int = 32,
    ) -> Tensor:
        """Greedy decoding after replacing the prompt's latent placeholders."""
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("generate currently accepts one prompt at a time")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        self.eval()
        inputs_embeds, _ = self._replace_latent_placeholders(prompt_ids)
        generated: list[Tensor] = []
        for _ in range(max_new_tokens):
            logits = self._base_forward(inputs_embeds).logits
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated.append(next_token)
            if int(next_token.item()) == eos_token_id:
                break
            next_embed = self.get_input_embeddings()(next_token).unsqueeze(1)
            inputs_embeds = torch.cat((inputs_embeds, next_embed), dim=1)
        return torch.stack(generated, dim=1)
