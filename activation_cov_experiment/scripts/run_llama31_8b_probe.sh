#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/activation_cov_mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/activation_cov_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GPU_ARGS=()
if [[ "${1:-}" == "--gpu" ]]; then
  GPU_ARGS=(--gpu "${2:?missing GPU value after --gpu}")
  shift 2
fi

EXTRA_ARGS=("$@")

BASE_CONFIG="activation_cov_experiment/configs/activation_cov_base.yaml"
PROBE_CONFIG="activation_cov_experiment/configs/llama31_8b_commonsense_probe.yaml"

eval "$(conda run -n torch2.7 python activation_cov_experiment/scripts/select_gpu.py \
  --base-config "$BASE_CONFIG" \
  --config "$PROBE_CONFIG" \
  "${GPU_ARGS[@]}" \
  --format shell)"

conda run -n torch2.7 python activation_cov_experiment/scripts/collect_activations.py \
  --base-config "$BASE_CONFIG" \
  --config "$PROBE_CONFIG" \
  "${GPU_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

conda run -n torch2.7 python activation_cov_experiment/scripts/compute_activation_cov.py \
  --base-config "$BASE_CONFIG" \
  --config "$PROBE_CONFIG" \
  "${EXTRA_ARGS[@]}"

conda run -n torch2.7 python activation_cov_experiment/scripts/compute_svd_directions.py \
  --base-config "$BASE_CONFIG" \
  --config "$PROBE_CONFIG" \
  "${EXTRA_ARGS[@]}"

conda run -n torch2.7 python activation_cov_experiment/scripts/compare_directions.py \
  --base-config "$BASE_CONFIG" \
  --config "$PROBE_CONFIG" \
  "${EXTRA_ARGS[@]}"

conda run -n torch2.7 python activation_cov_experiment/scripts/plot_results.py \
  --base-config "$BASE_CONFIG" \
  --config "$PROBE_CONFIG" \
  "${EXTRA_ARGS[@]}"
