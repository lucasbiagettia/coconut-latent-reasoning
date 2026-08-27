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

- [data.py](coconut/data.py): dataset-neutral JSON/JSONL and Hugging Face adapters.
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

For JSON/JSONL—including the official ProsQA JSON files distributed by the
Coconut repository—configure paths directly:

```yaml
data:
  type: json
  train_path: path/to/train.json
  validation_path: path/to/validation.json
  train_split: train
  validation_split: validation
  columns:
    question: question
    steps: steps
    answer: answer
```

For a dataset loadable by `datasets.load_dataset`, use:

```yaml
data:
  type: huggingface
  dataset_id: owner/dataset-name
  config_name: null
  train_split: train
  validation_split: validation
  columns:
    question: question_column
    steps: reasoning_steps_column
    answer: answer_column
```

The adapter only maps and validates columns. Dataset-specific transformations
belong outside Coconut so the model never depends on a particular benchmark.

## Install and test

Use a supported Python version (3.10-3.12 is recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
```

The unit tests use tiny randomly initialized models, including a real GPTNeoX
transformer, so they need no model download or GPU. They verify step
preservation, stages, `c`, direct hidden-state reuse,
the dependency of `H2` on `H1`, causal shifting, prompt/latent and padding
masking, backward through latent reasoning, and Stage 0. They also compare the
reference and batched implementations with identical weights and heterogeneous
examples, including relevant logits, loss, and every parameter gradient.

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

## Local ProsQA experiment on a 3 GiB GPU

The first local experiment uses official ProsQA data and
`EleutherAI/pythia-70m`. Run:

```bash
bash scripts/run_local_experiment.sh
```

The script downloads `prosqa_train.json` and `prosqa_valid.json` from the
[official Coconut repository](https://github.com/facebookresearch/coconut/tree/main/data),
pins commit `27273cb8cca4bb763c041a63b036d0c3b7cbbb48`, verifies both SHA-256
hashes, and samples train and validation independently. It selects 300 train
and 50 validation examples with seed 42. To fit the memory target without
truncating reasoning, only official examples whose complete curriculum fits in
384 Pythia tokens are eligible. Source indices, lengths, hashes, and sampling
parameters are recorded in `data/experiments/prosqa300/subset_metadata.json`.

The experiment uses:

```yaml
implementation: batched
batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
max_length: 384
```

`batch_size` is the actual GPU microbatch; accumulation gives an effective
batch size of 16. Each epoch prints stage, losses, exact match on a fixed
20-example validation subset, learning rate, elapsed time, and allocated/peak
CUDA memory. Three fixed validation generations are also printed. Metrics are
written after every epoch.

Do not confuse the ignored legacy `data/raw/prosqa/` artifact with the official
dataset. The preparation script uses its own verified `data/prosqa_official/`
directory.

After successful completion, the self-contained result is:

```text
outputs/pythia70m_prosqa300/
├── model/                    # Hugging Face config + trained safetensors
├── tokenizer/                # tokenizer and Coconut special tokens
├── checkpoints/latest.pt     # model + optimizer + history for resume
├── coconut_config.json       # final stage and latent-thought count
├── training_config.json
└── training_history.json
```

Resume from the last completed epoch with:

```bash
python train.py \
  --config configs/local_pythia70m_prosqa300.yaml \
  --resume-from outputs/pythia70m_prosqa300/checkpoints/latest.pt
```

Ask one question after training:

```bash
python scripts/ask_model.py \
  --model-dir outputs/pythia70m_prosqa300 \
  --question "Every ... Is Tom a lempus or scrompus?"
```

Or omit `--question` for an interactive loop. The script loads the model only
from that output directory and automatically applies the saved number of
continuous thoughts.

Three GiB remains a tight budget because Coconut retains a differentiable graph
through several sequential prefix forwards. If CUDA reports OOM, `batch_size`
is already at its minimum; first lower `max_length` to 352 in both the
experiment config and `run_local_experiment.sh`, and lower the script's
`--validation-size` to 40. If necessary, use length 320 with validation size 16.
Rerunning preparation selects a shorter official pool rather than truncating
examples. Do not lower `gradient_accumulation_steps` expecting lower peak
VRAM—it changes effective batch size, not microbatch memory.

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

## Reference and batched forwards

Select the implementation in YAML:

```yaml
implementation: reference  # per-example oracle used by the CPU smoke test
```

or:

```yaml
implementation: batched    # used by configs/example_gpu.yaml
```

The batched path groups examples by latent count. At every step it masks each
row after its current prefix and gathers the hidden state from that row's own
latent position, so question lengths and latent positions may differ. It then
computes `H1` for the compatible group, followed by `H2`, and so on. The latent
chain remains sequential, but each step uses real batch parallelism. Examples
with incompatible latent counts are evaluated in separate groups.

`batch_size` is the actual microbatch passed into this grouped forward;
`gradient_accumulation_steps` controls how many microbatches contribute to one
optimizer update:

```yaml
batch_size: 4
gradient_accumulation_steps: 8
```

With one process this corresponds to an effective batch size of 32 examples.
Increase `batch_size` until GPU memory is comfortably utilized, then use
gradient accumulation to reach the desired effective batch size.

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
- `reference` remains intentionally per-example. `batched` groups compatible
  layouts without changing latent-thought semantics.
- The repository implements the method and pipeline, not the paper's full
  benchmark datasets, training budget, or reported accuracy reproduction.

These choices simplify computation without changing the continuous-thought
semantics or gradient path.
