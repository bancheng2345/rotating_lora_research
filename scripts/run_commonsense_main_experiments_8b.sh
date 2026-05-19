#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ARGS=()
if [[ "${1:-}" == "--gpu" ]]; then
  GPU_ARGS=(--gpu "${2:?missing GPU value after --gpu}")
  shift 2
fi

bash scripts/run_commonsense_main_experiments.sh \
  "${GPU_ARGS[@]}" \
  configs/base.yaml \
  configs/presets/models/llama31_8b_local.yaml \
  configs/presets/datasets/commonsense_170k.yaml \
  configs/presets/runs/commonsense_170k_full_8b.yaml
