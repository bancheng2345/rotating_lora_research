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

eval "$(conda run -n torch2.7 python scripts/select_gpu.py \
  --base-config "$BASE_CONFIG" \
  --config configs/experiments/ablation_random_projection.yaml \
  "${GPU_ARGS[@]}" \
  --format shell)"
conda run -n torch2.7 python scripts/train.py \
  --base-config "$BASE_CONFIG" \
  --config configs/experiments/ablation_random_projection.yaml \
  "${GPU_ARGS[@]}" \
  "$@"

eval "$(conda run -n torch2.7 python scripts/select_gpu.py \
  --base-config "$BASE_CONFIG" \
  --config configs/experiments/ablation_no_orth.yaml \
  "${GPU_ARGS[@]}" \
  --format shell)"
conda run -n torch2.7 python scripts/train.py \
  --base-config "$BASE_CONFIG" \
  --config configs/experiments/ablation_no_orth.yaml \
  "${GPU_ARGS[@]}" \
  "$@"

eval "$(conda run -n torch2.7 python scripts/select_gpu.py \
  --base-config "$BASE_CONFIG" \
  --config configs/experiments/pc_lora_unweighted.yaml \
  "${GPU_ARGS[@]}" \
  --format shell)"
conda run -n torch2.7 python scripts/train.py \
  --base-config "$BASE_CONFIG" \
  --config configs/experiments/pc_lora_unweighted.yaml \
  "${GPU_ARGS[@]}" \
  --override lora.capture_space=full \
  --override experiment.notes="ablation full activation capture" \
  "$@"

conda run -n torch2.7 python scripts/train.py \
  --base-config "$BASE_CONFIG" \
  --config configs/experiments/pc_lora_unweighted.yaml \
  "${GPU_ARGS[@]}" \
  --override lora.left_projection_source=random_orthogonal \
  --override lora.right_projection_source=random_orthogonal \
  --override experiment.notes="ablation random U/V projections" \
  "$@"
