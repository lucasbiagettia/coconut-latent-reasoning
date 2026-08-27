# Coconut latent reasoning

This repository is a small, correctness-first implementation of **Coconut
(Chain of Continuous Thought)** from *Training Large Language Models to Reason
in a Continuous Latent Space*. The included [paper](coconutPaper.md), not the
previous prototype, is the source of truth.

## How it works

For a training example with a question, reasoning steps `R1..Rn`, and an
answer, Stage 0 trains ordinary language chain-of-thought. At Stage `k`, the
first `min(k, n)` text steps disappear and each removed step is replaced by
`c` continuous thoughts:

```text
Stage 0: Question <bot> <eot> R1 R2 R3 Answer
Stage 1: Question <bot> H1 <eot> R2 R3 Answer
Stage 2: Question <bot> H1 H2 <eot> R3 Answer
Stage 3: Question <bot> H1 H2 H3 <eot> Answer
```

`H1`, `H2`, ... are not vocabulary tokens. For every latent position, the
model runs on the prefix and its final-layer hidden state is inserted directly
as the next input embedding. It never passes through the LM head. The next
latent forward consumes that inserted state, and the graph is kept intact so
the final causal language loss backpropagates through the complete latent
chain.

The implementation inserts `<bot>` and `<eot>` in every stage, including the
zero-latent Stage 0, matching the paper authors' released training
representation. Question, boundary, latent, and padding labels are `-100`;
loss starts with the first remaining reasoning/answer token and uses the
standard one-token causal shift. The optimizer is recreated at each curriculum
stage.

The core code is intentionally small:

- [data.py](coconut/data.py): dataset-neutral examples and JSON adapter.
- [curriculum.py](coconut/curriculum.py): stages, tokenization, masking, padding.
- [model.py](coconut/model.py): hidden-state feedback and greedy inference.
- [training.py](coconut/training.py): single-process CPU/CUDA trainer.

## Dataset format

Both JSON arrays and JSONL are supported. Every record must use separated
reasoning steps:

```json
{
  "question": "What is 2 plus 3?",
  "steps": ["Start with 2.", "Add 3 to get 5."],
  "answer": "5"
}
```

`ReasoningExample` stores `steps` as `list[str]`. Dataset integrations only
need to implement:

```python
class ReasoningDatasetAdapter(Protocol):
    def load_split(self, split: str) -> Sequence[ReasoningExample]: ...
```

No generated data in this repository is presented as GSM8K, ProntoQA, or
ProsQA. `data/toy/` is explicitly a tiny arithmetic smoke dataset.

## Install and test

Use a supported Python version (3.10-3.12 is recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
```

The unit tests use a tiny in-memory causal model, so they need no model download
or GPU. They verify step preservation, stages, `c`, direct hidden-state reuse,
the dependency of `H2` on `H1`, causal shifting, prompt/latent and padding
masking, backward through latent reasoning, and Stage 0.

## CPU smoke test

```bash
python train.py --config configs/cpu_smoke.yaml
```

This uses `EleutherAI/pythia-14m`, the smallest Pythia checkpoint, and a
12-example toy dataset. It deliberately runs only one train and validation
batch at Stage 0 and Stage 1. The command exercises model/tokenizer loading,
JSONL loading, tokenization, normal CoT, latent reasoning, forward, backward,
an optimizer step, checkpoints, validation loss, and fixed-length latent
inference. It tests plumbing, not task accuracy. Remove the two
`max_*_batches` limits and raise the epoch count to try overfitting the toy set.

## Change the model

The model is loaded through `AutoModelForCausalLM` and embeddings through
`get_input_embeddings()`. Change only `model_id` in YAML, for example:

```yaml
model_id: EleutherAI/pythia-1.4b
```

or:

```yaml
model_id: Qwen/Qwen3-1.7B-Base
```

Then choose `device: cuda` and tune batch size/gradient accumulation. See
`configs/example_gpu.yaml`. A compatible decoder-only model must accept
`inputs_embeds` and expose its final hidden states.

To run a saved checkpoint:

```bash
python infer.py \
  --checkpoint checkpoints/cpu_smoke/stage_1_epoch_1.pt \
  --question "What is 2 plus 3?" \
  --latent-thoughts 1
```

## Deliberate scope choices

- Inference uses the paper's simple fixed number of latent thoughts and inserts
  `<eot>` at that known position. The optional learned termination classifier
  is not implemented.
- Every latent thought recomputes its full prefix. There is no KV cache, DDP,
  FSDP, DeepSpeed, or experiment tracking.
- Training uses fp32 on CPU and CUDA. Mixed precision is intentionally deferred
  until it can include the necessary gradient scaling and stability checks.
- Batches are padded for data loading, but examples are evaluated independently
  inside the wrapper. This keeps variable latent locations unambiguous at the
  cost of throughput.
- The repository implements the method and pipeline, not the paper's full
  benchmark datasets, training budget, or reported accuracy reproduction.

These choices simplify computation without changing the continuous-thought
semantics or gradient path.
