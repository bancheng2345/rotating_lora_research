#!/usr/bin/env bash
set -euo pipefail

GPU="auto"
SEEDS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-/home/zck/miniconda3/envs/torch2.7/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

IFS=',' read -ra SEED_LIST <<< "${SEEDS}"
for SEED in "${SEED_LIST[@]}"; do
  echo "=== ActCov-rank PC-LoRA seed=${SEED} ==="
  "${PYTHON_BIN}" scripts/train.py \
    --base-config configs/base.yaml \
    --config configs/presets/models/llama31_8b_local.yaml \
    --config configs/presets/datasets/commonsense_170k.yaml \
    --config configs/presets/runs/commonsense_170k_full_8b.yaml \
    --config configs/experiments/actcov_rank_selected_pc_lora.yaml \
    --gpu "${GPU}" \
    --override "experiment.seed=${SEED}" \
    --override metrics.compute_rho_k=false
done
