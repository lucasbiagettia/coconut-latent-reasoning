#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COCONUT_PYTHON="${COCONUT_PYTHON:-python}"
COCONUT_MODEL_SIZE="${COCONUT_MODEL_SIZE:-70m}"

case "$COCONUT_MODEL_SIZE" in
  70m)
    MODEL_ID="EleutherAI/pythia-70m"
    CONFIG_PATH="configs/local_pythia70m_entailmentbank500.yaml"
    ;;
  160m)
    MODEL_ID="EleutherAI/pythia-160m"
    CONFIG_PATH="configs/local_pythia160m_entailmentbank500.yaml"
    ;;
  *)
    echo "COCONUT_MODEL_SIZE must be 70m or 160m" >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"

"$COCONUT_PYTHON" scripts/prepare_entailmentbank_subset.py \
  --output data/experiments/entailmentbank500/selection_metadata.json \
  --model-id "$MODEL_ID" \
  --train-size 500 \
  --validation-size 100 \
  --seed 42 \
  --min-proof-depth 1 \
  --max-proof-depth 4 \
  --max-proof-steps 4 \
  --max-length 256 \
  --c 1

"$COCONUT_PYTHON" train.py --config "$CONFIG_PATH"
