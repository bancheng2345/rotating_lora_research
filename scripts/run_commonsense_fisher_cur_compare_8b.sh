#!/usr/bin/env bash
set -euo pipefail

GPU="auto"
SEEDS="0"
SKIP_BUILD="0"

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
    --skip-build)
      SKIP_BUILD="1"
      shift
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

COMMON_CONFIGS=(
  --base-config configs/base.yaml
  --config configs/presets/models/llama31_8b_local.yaml
  --config configs/presets/datasets/commonsense_170k.yaml
  --config configs/presets/runs/commonsense_170k_full_8b.yaml
)

if [[ "${SKIP_BUILD}" != "1" ]]; then
  echo "=== Build Fisher directions ==="
  "${PYTHON_BIN}" scripts/compute_fisher_directions.py \
    "${COMMON_CONFIGS[@]}" \
    --config configs/experiments/fisher_oplora_probe_layers.yaml \
    --gpu "${GPU}"

  echo "=== Build CUR directions ==="
  "${PYTHON_BIN}" scripts/compute_cur_directions.py \
    "${COMMON_CONFIGS[@]}" \
    --config configs/experiments/cur_oplora_probe_layers.yaml \
    --gpu "${GPU}"
fi

IFS=',' read -ra SEED_LIST <<< "${SEEDS}"
for SEED in "${SEED_LIST[@]}"; do
  echo "=== Fisher OPLoRA seed=${SEED} ==="
  "${PYTHON_BIN}" scripts/train.py \
    "${COMMON_CONFIGS[@]}" \
    --config configs/experiments/fisher_oplora_probe_layers.yaml \
    --gpu "${GPU}" \
    --override "experiment.seed=${SEED}" \
    --override metrics.compute_rho_k=false

  echo "=== CUR OPLoRA seed=${SEED} ==="
  "${PYTHON_BIN}" scripts/train.py \
    "${COMMON_CONFIGS[@]}" \
    --config configs/experiments/cur_oplora_probe_layers.yaml \
    --gpu "${GPU}" \
    --override "experiment.seed=${SEED}" \
    --override metrics.compute_rho_k=false
done
