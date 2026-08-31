#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COCONUT_PYTHON="${COCONUT_PYTHON:-python}"

cd "$REPO_ROOT"

"$COCONUT_PYTHON" scripts/prepare_prosqa_subset.py \
  --train-size 300 \
  --validation-size 50 \
  --seed 42 \
  --model-id EleutherAI/pythia-70m \
  --max-length 384

"$COCONUT_PYTHON" train.py --config configs/local_pythia70m_prosqa300.yaml
