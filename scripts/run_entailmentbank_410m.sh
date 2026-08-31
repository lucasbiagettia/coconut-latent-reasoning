#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COCONUT_PYTHON="${COCONUT_PYTHON:-python}"
COCONUT_MODEL_SIZE="${COCONUT_MODEL_SIZE:-410m}"
MEMORY_TEST_ONLY=false

if [[ "${1:-}" == "--memory-test-only" ]]; then
  MEMORY_TEST_ONLY=true
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--memory-test-only]" >&2
  exit 2
fi

case "$COCONUT_MODEL_SIZE" in
  410m)
    MODEL_ID="EleutherAI/pythia-410m"
    CONFIG_PATH="configs/local_pythia410m_entailmentbank_full.yaml"
    ;;
  160m)
    MODEL_ID="EleutherAI/pythia-160m"
    CONFIG_PATH="configs/local_pythia160m_entailmentbank_full.yaml"
    ;;
  *)
    echo "COCONUT_MODEL_SIZE must be 410m or 160m" >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"

"$COCONUT_PYTHON" scripts/prepare_entailmentbank_full.py \
  --output data/experiments/entailmentbank_full/selection_metadata.json \
  --model-id "$MODEL_ID" \
  --max-length 256 \
  --c 1

"$COCONUT_PYTHON" scripts/memory_smoke_test.py --config "$CONFIG_PATH"

if [[ "$MEMORY_TEST_ONLY" == true ]]; then
  exit 0
fi

"$COCONUT_PYTHON" train.py --config "$CONFIG_PATH"
