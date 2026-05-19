#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ARGS=()
RUN_CONFIG="configs/presets/runs/commonsense_170k_dry_run_8b.yaml"
WITH_RHO=false

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
    --with-rho)
      WITH_RHO=true
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/presets/models/llama31_8b_local.yaml"
DATA_CONFIG="configs/presets/datasets/commonsense_170k.yaml"

METHOD_CONFIGS=(
  "configs/experiments/lora_actcov_probe_layers.yaml"
  "configs/experiments/oplora_svd_right_probe_layers.yaml"
  "configs/experiments/actcov_oplora.yaml"
  "configs/experiments/actcov_pc_lora.yaml"
)

METHOD_IDS=(
  "lora"
  "oplora"
  "actcov_oplora"
  "actcov_pc_lora"
)

EXTRA_OVERRIDES=(
  --override "experiment.notes=commonsense_8b_actcov_compare"
)
if [[ "$WITH_RHO" == false ]]; then
  # 8B rho_k computes fresh CPU SVDs for metric layers and can dominate short dry runs.
  EXTRA_OVERRIDES+=(--override "metrics.compute_rho_k=false")
fi

latest_completed_run_after() {
  local method_id="$1"
  local start_epoch="$2"
  python3 - "$method_id" "$start_epoch" <<'PY'
import json
import pathlib
import sys

method = sys.argv[1]
start_epoch = float(sys.argv[2])
root = pathlib.Path("outputs/runs/meta_llama_llama_3_1_8b")
matches = []
for path in root.glob(f"{method}_*commonsense_170k_train_165420*r8_k16_seed0_*"):
    summary_path = path / "final_summary.json"
    if not summary_path.exists():
        continue
    if summary_path.stat().st_mtime < start_epoch:
        continue
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if summary.get("method") != method:
        continue
    last_eval = summary.get("last_eval") or {}
    if int(summary.get("max_steps") or 0) <= 0:
        continue
    if int(last_eval.get("step") or -1) != int(summary.get("max_steps") or -2):
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
  echo "=== ActivationCov compare: ${METHOD_CONFIG} ==="
  eval "$(conda run -n torch2.7 python scripts/select_gpu.py \
    --base-config "$BASE_CONFIG" \
    --config "$MODEL_CONFIG" \
    --config "$DATA_CONFIG" \
    --config "$RUN_CONFIG" \
    --config "$METHOD_CONFIG" \
    "${GPU_ARGS[@]}" \
    --format shell)"

  START_EPOCH="$(python3 - <<'PY'
import time
print(time.time())
PY
)"
  set +e
  conda run -n torch2.7 python scripts/train.py \
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
