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
METHOD_CONFIG="${2:-configs/experiments/pc_lora_unweighted.yaml}"
MODEL_CONFIG="${3:-configs/presets/models/llama32_1b_local.yaml}"
DATA_CONFIG="${4:-configs/presets/datasets/commonsense_170k.yaml}"
RUN_CONFIG="${5:-configs/presets/runs/commonsense_170k_full.yaml}"

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
