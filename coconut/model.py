"""Generic causal-LM wrapper implementing sequential continuous thoughts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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

    ``reference`` evaluates rows independently as a debugging oracle.
    ``batched`` groups compatible rows and evaluates each latent step in
    parallel. Both intentionally recompute prefixes instead of using a KV cache.
    """

    def __init__(
        self,
        base_causal_lm: nn.Module,
        latent_token_id: int,
        implementation: Literal["reference", "batched"] = "reference",
    ) -> None:
        super().__init__()
        if implementation not in {"reference", "batched"}:
            raise ValueError("implementation must be 'reference' or 'batched'")
        self.base_causal_lm = base_causal_lm
        self.latent_token_id = latent_token_id
        self.implementation = implementation

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
        if self.implementation == "reference":
            return self.forward_reference(input_ids, attention_mask, labels)
        return self.forward_batched(input_ids, attention_mask, labels)

    def forward_reference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> CoconutOutput:
        """Original per-example implementation, kept as the debugging oracle."""
        lengths = self._validate_batch(input_ids, attention_mask, labels)

        row_logits: list[Tensor] = []
        row_embeds: list[Tensor] = []
        all_thoughts: list[tuple[Tensor, ...]] = []
        total_loss = input_ids.new_zeros((), dtype=torch.float32)
        total_targets = 0

        for row, length in enumerate(lengths):
            row_ids = input_ids[row : row + 1, :length]
            embeds, thoughts = self._replace_latent_placeholders(row_ids)
            outputs = self._base_forward(embeds)
            logits = outputs.logits
            row_logits.append(logits)
            row_embeds.append(embeds)
            all_thoughts.append(thoughts)

            if labels is not None:
                row_labels = labels[row : row + 1, :length]
                row_loss, target_count = self._causal_loss_sum(logits, row_labels)
                total_loss = total_loss + row_loss
                total_targets += target_count

        return self._assemble_output(
            row_logits,
            row_embeds,
            all_thoughts,
            total_loss,
            total_targets,
            labels is not None,
        )

    def forward_batched(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> CoconutOutput:
        """Process compatible examples together at every sequential latent step.

        Rows are grouped only by latent count. At a given latent step, each row
        may have a different prefix length: the group runs up to the longest
        prefix, masks future positions, and gathers the final hidden state at
        the correct position for every row.
        """
        lengths = self._validate_batch(input_ids, attention_mask, labels)
        layouts: list[tuple[int, int]] = []
        groups: dict[int, list[int]] = {}
        for row, length in enumerate(lengths):
            positions = self._latent_positions(input_ids[row, :length])
            latent_start = int(positions[0].item()) if positions.numel() else 0
            latent_count = int(positions.numel())
            layouts.append((latent_start, latent_count))
            groups.setdefault(latent_count, []).append(row)

        row_logits: list[Tensor | None] = [None] * input_ids.shape[0]
        row_embeds: list[Tensor | None] = [None] * input_ids.shape[0]
        all_thoughts: list[tuple[Tensor, ...] | None] = [None] * input_ids.shape[0]
        total_loss = input_ids.new_zeros((), dtype=torch.float32)
        total_targets = 0

        for latent_count, row_indices in groups.items():
            group_rows = torch.tensor(row_indices, dtype=torch.long, device=input_ids.device)
            group_width = max(lengths[row] for row in row_indices)
            group_ids = input_ids.index_select(0, group_rows)[:, :group_width]
            group_attention_mask = attention_mask.index_select(0, group_rows)[
                :, :group_width
            ]
            inputs_embeds = self.get_input_embeddings()(group_ids)
            latent_starts = torch.tensor(
                [layouts[row][0] for row in row_indices],
                dtype=torch.long,
                device=input_ids.device,
            )
            thoughts_by_row: list[list[Tensor]] = [[] for _ in row_indices]

            for latent_step in range(latent_count):
                current_positions = latent_starts + latent_step
                prefix_width = int(current_positions.max().item())
                prefix_columns = torch.arange(prefix_width, device=input_ids.device)
                prefix_mask = prefix_columns.unsqueeze(0) < current_positions.unsqueeze(1)
                prefix_outputs = self._base_forward(
                    inputs_embeds[:, :prefix_width, :],
                    attention_mask=prefix_mask.long(),
                )
                batch_indices = torch.arange(len(row_indices), device=input_ids.device)
                thought_batch = prefix_outputs.hidden_states[-1][
                    batch_indices, current_positions - 1, :
                ]
                for local_row, thought in enumerate(thought_batch):
                    thoughts_by_row[local_row].append(thought.unsqueeze(0))
                replacement_mask = (
                    torch.arange(group_width, device=input_ids.device).unsqueeze(0)
                    == current_positions.unsqueeze(1)
                )
                inputs_embeds = torch.where(
                    replacement_mask.unsqueeze(-1),
                    thought_batch.unsqueeze(1),
                    inputs_embeds,
                )

            outputs = self._base_forward(
                inputs_embeds,
                attention_mask=group_attention_mask,
            )
            if labels is not None:
                aligned_labels = labels.index_select(0, group_rows)[:, :group_width]
                group_loss, group_targets = self._causal_loss_sum(
                    outputs.logits, aligned_labels
                )
                total_loss = total_loss + group_loss
                total_targets += group_targets

            for local_row, row in enumerate(row_indices):
                length = lengths[row]
                row_logits[row] = outputs.logits[
                    local_row : local_row + 1, :length, :
                ]
                row_embeds[row] = inputs_embeds[
                    local_row : local_row + 1, :length, :
                ]
                all_thoughts[row] = tuple(thoughts_by_row[local_row])

        return self._assemble_output(
            self._complete_rows(row_logits),
            self._complete_rows(row_embeds),
            self._complete_thoughts(all_thoughts),
            total_loss,
            total_targets,
            labels is not None,
        )

    def _validate_batch(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor | None
    ) -> list[int]:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        lengths: list[int] = []
        for row in range(input_ids.shape[0]):
            length = int(attention_mask[row].sum().item())
            if length < 2:
                raise ValueError("Each sequence must contain at least two non-padding tokens")
            if not torch.all(attention_mask[row, :length] == 1) or torch.any(
                attention_mask[row, length:] != 0
            ):
                raise ValueError("CoconutModel expects right-padded attention masks")
            self._latent_positions(input_ids[row, :length])
            lengths.append(length)
        return lengths

    def _assemble_output(
        self,
        row_logits: list[Tensor],
        row_embeds: list[Tensor],
        all_thoughts: list[tuple[Tensor, ...]],
        total_loss: Tensor,
        total_targets: int,
        has_labels: bool,
    ) -> CoconutOutput:
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
        if has_labels:
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

    @staticmethod
    def _complete_rows(rows: list[Tensor | None]) -> list[Tensor]:
        if any(row is None for row in rows):
            raise RuntimeError("Internal batching error: an output row was not populated")
        return [row for row in rows if row is not None]

    @staticmethod
    def _complete_thoughts(
        rows: list[tuple[Tensor, ...] | None],
    ) -> list[tuple[Tensor, ...]]:
        if any(row is None for row in rows):
            raise RuntimeError("Internal batching error: latent states were not populated")
        return [row for row in rows if row is not None]

    @staticmethod
    def _causal_loss_sum(logits: Tensor, labels: Tensor) -> tuple[Tensor, int]:
        shift_labels = labels[:, 1:].contiguous()
        target_count = int((shift_labels != IGNORE_INDEX).sum().item())
        if not target_count:
            return logits.new_zeros((), dtype=torch.float32), 0
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        return loss, target_count

    def _latent_positions(self, input_ids: Tensor) -> Tensor:
        positions = (input_ids == self.latent_token_id).nonzero(as_tuple=False).flatten()
        if positions.numel() > 1 and not torch.all(positions[1:] == positions[:-1] + 1):
            raise ValueError("Latent placeholders must form one contiguous block")
        if positions.numel() and int(positions[0].item()) == 0:
            raise ValueError("A latent thought needs a preceding token")
        return positions

    def _replace_latent_placeholders(
        self, input_ids: Tensor
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        latent_positions = self._latent_positions(input_ids[0])

        inputs_embeds = self.get_input_embeddings()(input_ids)
        thoughts: list[Tensor] = []
        for position_tensor in latent_positions:
            position = int(position_tensor.item())
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

    def _base_forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Any:
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
            )
        kwargs: dict[str, Any] = {}
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        return self.base_causal_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            **kwargs,
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
