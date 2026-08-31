#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COCONUT_PYTHON="${COCONUT_PYTHON:-python}"
PRIMARY_CONFIG="configs/local_pythia160m_entailmentbank_controlled.yaml"
SAFE_CONFIG="configs/local_pythia160m_entailmentbank_controlled_safe.yaml"
MEMORY_TEST_ONLY=false

if [[ "${1:-}" == "--memory-test-only" ]]; then
  MEMORY_TEST_ONLY=true
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--memory-test-only]" >&2
  exit 2
fi

cd "$REPO_ROOT"

"$COCONUT_PYTHON" scripts/prepare_entailmentbank_full.py \
  --output data/experiments/entailmentbank_full/selection_metadata.json \
  --model-id EleutherAI/pythia-160m \
  --max-length 256 \
  --c 1

set +e
"$COCONUT_PYTHON" scripts/memory_smoke_test.py --config "$PRIMARY_CONFIG"
PRIMARY_STATUS=$?
set -e

if [[ "$PRIMARY_STATUS" -eq 0 ]]; then
  SELECTED_CONFIG="$PRIMARY_CONFIG"
elif [[ "$PRIMARY_STATUS" -eq 1 ]]; then
  echo "batch_size=2 did not fit; testing batch_size=1 with accumulation=16"
  "$COCONUT_PYTHON" scripts/memory_smoke_test.py --config "$SAFE_CONFIG"
  SELECTED_CONFIG="$SAFE_CONFIG"
else
  exit "$PRIMARY_STATUS"
fi

echo "selected_training_config=$SELECTED_CONFIG"

if [[ "$MEMORY_TEST_ONLY" == true ]]; then
  exit 0
fi

"$COCONUT_PYTHON" train.py --config "$SELECTED_CONFIG"
