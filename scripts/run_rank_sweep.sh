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

RANKS=(4 8 16 32)
for RANK in "${RANKS[@]}"; do
  eval "$(conda run -n torch2.7 python scripts/select_gpu.py \
    --base-config "$BASE_CONFIG" \
    --config configs/experiments/rank_sweep.yaml \
    "${GPU_ARGS[@]}" \
    --format shell)"
  conda run -n torch2.7 python scripts/train.py \
    --base-config "$BASE_CONFIG" \
    --config configs/experiments/rank_sweep.yaml \
    "${GPU_ARGS[@]}" \
    --override lora.rank="$RANK" \
    --override lora.alpha="$((RANK * 2))" \
    --override experiment.notes="rank sweep r=${RANK}" \
    "$@"
done
