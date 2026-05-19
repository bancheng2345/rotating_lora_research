#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/zck/miniconda3/envs/torch2.7/bin/python}"
# Use the torch2.7 env python directly. On this server, wrapping long 8B
# training with `conda run` has intermittently segfaulted after completion.
BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/presets/models/llama31_8b_local_sdpa.yaml"
DATA_CONFIG="configs/presets/datasets/metamathqa.yaml"
RUN_CONFIG="configs/presets/runs/metamathqa_full_8b_bsz8.yaml"
GPU_ARGS=()
WITH_RHO=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ARGS=(--gpu "${2:?missing GPU value after --gpu}")
      shift 2
      ;;
    --dry-run)
      RUN_CONFIG="configs/presets/runs/metamathqa_dry_run_8b.yaml"
      shift
      ;;
    --run-config)
      RUN_CONFIG="${2:?missing value after --run-config}"
      shift 2
      ;;
    --no-rho)
      WITH_RHO=false
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

METHOD_CONFIGS=(
  "configs/experiments/lora_actcov_probe_layers.yaml"
  "configs/experiments/oplora_svd_right_probe_layers.yaml"
  "configs/experiments/pc_lora_svd_right_probe_layers.yaml"
  "configs/experiments/residual_mwa_pc_lora.yaml"
)

METHOD_IDS=(
  "lora"
  "oplora"
  "pc_lora"
  "residual_mwa_pc_lora"
)

EXTRA_OVERRIDES=(
  --override "experiment.notes=metamathqa_8b_residual_joint_capture_bsz8"
)
if [[ "$WITH_RHO" == false ]]; then
  EXTRA_OVERRIDES+=(--override "metrics.compute_rho_k=false")
fi

latest_completed_run_after() {
  local method_id="$1"
  local start_epoch="$2"
  "$PYTHON_BIN" - "$method_id" "$start_epoch" <<'PY'
import json
import pathlib
import sys

method = sys.argv[1]
start_epoch = float(sys.argv[2])
root = pathlib.Path("outputs/runs/meta_llama_llama_3_1_8b")
matches = []
for path in root.glob(f"{method}_*metamathqa_train_100000*r8_k16_seed0_*"):
    summary_path = path / "final_summary.json"
    if not summary_path.exists() or summary_path.stat().st_mtime < start_epoch:
        continue
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if summary.get("method") != method:
        continue
    last_eval = summary.get("last_eval") or {}
    max_steps = int(summary.get("max_steps") or 0)
    if max_steps <= 0 or int(last_eval.get("step") or -1) != max_steps:
        continue
    matches.append(path)

matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
if matches:
    print(matches[0])
PY
}

for idx in "${!METHOD_CONFIGS[@]}"; do
  METHOD_CONFIG="${METHOD_CONFIGS[$idx]}"
  METHOD_ID="${METHOD_IDS[$idx]}"
  echo "=== MetaMathQA M_WA compare: ${METHOD_CONFIG} ==="
  START_EPOCH="$("$PYTHON_BIN" -c 'import time; print(time.time())')"
  set +e
  "$PYTHON_BIN" scripts/train.py \
    --base-config "$BASE_CONFIG" \
    --config "$MODEL_CONFIG" \
    --config "$DATA_CONFIG" \
    --config "$RUN_CONFIG" \
    --config "$METHOD_CONFIG" \
    "${EXTRA_OVERRIDES[@]}" \
    "${GPU_ARGS[@]}"
  STATUS=$?
  set -e

  if [[ "$STATUS" -ne 0 ]]; then
    COMPLETED_RUN="$(latest_completed_run_after "$METHOD_ID" "$START_EPOCH")"
    if [[ -n "$COMPLETED_RUN" ]]; then
      echo "warning: ${METHOD_CONFIG} exited with status ${STATUS}, but final_summary.json is complete."
      echo "run_dir=${COMPLETED_RUN}"
      continue
    fi
    echo "error: ${METHOD_CONFIG} failed before producing a complete final_summary.json" >&2
    exit "$STATUS"
  fi
done
