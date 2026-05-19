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
MODEL_CONFIG="${2:-configs/presets/models/llama32_1b_local.yaml}"
DATA_CONFIG="${3:-configs/presets/datasets/math500.yaml}"
RUN_CONFIG="${4:-configs/presets/runs/math500_full.yaml}"

METHOD_CONFIGS=(
  "configs/experiments/lora.yaml"
  "configs/experiments/oplora.yaml"
  "configs/experiments/lora_nsc.yaml"
  "configs/experiments/pc_lora_unweighted.yaml"
  "configs/experiments/pc_lora_loss_weighted.yaml"
)

for METHOD_CONFIG in "${METHOD_CONFIGS[@]}"; do
  eval "$(conda run -n torch2.7 python scripts/select_gpu.py \
    --base-config "$BASE_CONFIG" \
    --config "$MODEL_CONFIG" \
    --config "$DATA_CONFIG" \
    --config "$RUN_CONFIG" \
    --config "$METHOD_CONFIG" \
    "${GPU_ARGS[@]}" \
    --format shell)"

  conda run -n torch2.7 python scripts/train.py \
    --base-config "$BASE_CONFIG" \
    --config "$MODEL_CONFIG" \
    --config "$DATA_CONFIG" \
    --config "$RUN_CONFIG" \
    --config "$METHOD_CONFIG" \
    "${GPU_ARGS[@]}"
done
