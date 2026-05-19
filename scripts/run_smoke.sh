#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ARGS=()
if [[ "${1:-}" == "--gpu" ]]; then
  GPU_ARGS=(--gpu "${2:?missing GPU value after --gpu}")
  shift 2
fi

BASE_CONFIG="${1:-configs/base.yaml}"
EXPERIMENT_CONFIG="${2:-configs/experiments/pc_lora_unweighted.yaml}"

eval "$(conda run -n torch2.7 python scripts/select_gpu.py \
  --base-config "$BASE_CONFIG" \
  --config "$EXPERIMENT_CONFIG" \
  "${GPU_ARGS[@]}" \
  --format shell)"

conda run -n torch2.7 python scripts/train.py \
  --base-config "$BASE_CONFIG" \
  --config "$EXPERIMENT_CONFIG" \
  "${GPU_ARGS[@]}" \
  --override training.max_steps=5 \
  --override training.synthetic_train_samples=32 \
  --override training.synthetic_eval_samples=8 \
  --override training.logging_steps=1 \
  --override training.eval_steps=5 \
  --override training.save_steps=5
