#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ARGS=()
RUN_CONFIG="configs/presets/runs/commonsense_170k_dry_run_8b.yaml"
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ARGS=(--gpu "${2:?missing GPU value after --gpu}")
      shift 2
      ;;
    --full)
      RUN_CONFIG="configs/presets/runs/commonsense_170k_full_8b.yaml"
      shift
      ;;
    --run-config)
      RUN_CONFIG="${2:?missing value after --run-config}"
      shift 2
      ;;
    --k)
      EXTRA_OVERRIDES+=(--override "lora.projection_rank_k=${2:?missing value after --k}")
      shift 2
      ;;
    --lambda-cap)
      EXTRA_OVERRIDES+=(--override "lora.lambda_cap=${2:?missing value after --lambda-cap}")
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

conda run -n torch2.7 python scripts/train.py \
  --base-config configs/base.yaml \
  --config configs/presets/models/llama31_8b_local.yaml \
  --config configs/presets/datasets/commonsense_170k.yaml \
  --config "$RUN_CONFIG" \
  --config configs/experiments/actcov_guided_lora.yaml \
  --override metrics.compute_rho_k=false \
  --override experiment.notes=actcov_guided_lora \
  "${EXTRA_OVERRIDES[@]}" \
  "${GPU_ARGS[@]}"
