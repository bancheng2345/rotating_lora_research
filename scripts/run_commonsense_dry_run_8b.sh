#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ARGS=()
if [[ "${1:-}" == "--gpu" ]]; then
  GPU_ARGS=(--gpu "${2:?missing GPU value after --gpu}")
  shift 2
fi

METHOD_CONFIG="${1:-configs/experiments/pc_lora_unweighted.yaml}"

bash scripts/run_commonsense_dry_run.sh \
  "${GPU_ARGS[@]}" \
  configs/base.yaml \
  "$METHOD_CONFIG" \
  configs/presets/models/llama31_8b_local.yaml \
  configs/presets/datasets/commonsense_170k.yaml \
  configs/presets/runs/commonsense_170k_dry_run_8b.yaml
