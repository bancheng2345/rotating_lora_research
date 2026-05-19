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

bash scripts/run_math500_dry_run.sh \
  "${GPU_ARGS[@]}" \
  configs/base.yaml \
  "$METHOD_CONFIG" \
  configs/presets/models/llama32_3b_local.yaml \
  configs/presets/datasets/math500.yaml \
  configs/presets/runs/math500_dry_run.yaml
