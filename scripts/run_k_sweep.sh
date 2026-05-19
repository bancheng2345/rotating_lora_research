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
shift || true

KS=(8 16 32 64 128)
for K in "${KS[@]}"; do
  eval "$(conda run -n torch2.7 python scripts/select_gpu.py \
    --base-config "$BASE_CONFIG" \
    --config configs/experiments/k_sweep.yaml \
    "${GPU_ARGS[@]}" \
    --format shell)"
  conda run -n torch2.7 python scripts/train.py \
    --base-config "$BASE_CONFIG" \
    --config configs/experiments/k_sweep.yaml \
    "${GPU_ARGS[@]}" \
    --override lora.projection_rank_k="$K" \
    --override experiment.notes="k sweep k=${K}" \
    "$@"
done
