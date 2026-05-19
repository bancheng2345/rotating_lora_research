#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ARGS=()
if [[ "${1:-}" == "--gpu" ]]; then
  GPU_ARGS=(--gpu "${2:?missing GPU value after --gpu}")
  shift 2
fi

BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/presets/models/llama31_8b_local.yaml"
DATA_CONFIG="configs/presets/datasets/commonsense_170k.yaml"
RUN_CONFIG="configs/presets/runs/commonsense_170k_full_8b.yaml"

METHODS=(
  "configs/experiments/lora.yaml"
  "configs/experiments/oplora.yaml"
  "configs/experiments/pc_lora_unweighted.yaml"
)

METHOD_IDS=(
  "lora"
  "oplora"
  "pc_lora"
)

SEEDS=(0 1 2)

latest_run_dir() {
  local method_id="$1"
  local seed="$2"
  python3 - "$method_id" "$seed" <<'PY'
import json
import pathlib
import sys

method = sys.argv[1]
seed = int(sys.argv[2])
root = pathlib.Path("outputs/runs/meta_llama_llama_3_1_8b")
matches = []
for path in root.glob(f"{method}_*commonsense_170k_train_165420*r8_k16_seed{seed}_*"):
    if method == "pc_lora" and path.name.startswith("pc_lora_weighted_"):
        continue
    summary_path = path / "final_summary.json"
    if not summary_path.exists():
        continue
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if summary.get("method") != method:
        continue
    if summary.get("last_eval", {}).get("eval_num_samples") != 5000:
        continue
    matches.append(path)

matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
if matches:
    print(matches[0])
PY
}

is_complete_run() {
  local method_id="$1"
  local seed="$2"
  local run_dir
  run_dir="$(latest_run_dir "$method_id" "$seed")"
  [[ -n "$run_dir" && -f "$run_dir/final_summary.json" ]]
}

for idx in "${!METHODS[@]}"; do
  method="${METHODS[$idx]}"
  method_id="${METHOD_IDS[$idx]}"
  for seed in "${SEEDS[@]}"; do
    if is_complete_run "$method_id" "$seed"; then
      echo "=== method=${method} seed=${seed} already complete, skipping ==="
      echo "run_dir=$(latest_run_dir "$method_id" "$seed")"
      continue
    fi

    echo "=== method=${method} seed=${seed} ==="
    set +e
    conda run -n torch2.7 python scripts/train.py \
      --base-config "$BASE_CONFIG" \
      --config "$MODEL_CONFIG" \
      --config "$DATA_CONFIG" \
      --config "$RUN_CONFIG" \
      --config "$method" \
      "${GPU_ARGS[@]}" \
      --override experiment.seed="$seed" \
      --override experiment.notes="commonsense_170k_8b_seed_sweep"
    status=$?
    set -e

    if [[ "$status" -ne 0 ]]; then
      if is_complete_run "$method_id" "$seed"; then
        echo "warning: training command exited with status ${status}, but run completed successfully."
        echo "run_dir=$(latest_run_dir "$method_id" "$seed")"
        continue
      fi
      echo "error: training failed before writing final_summary.json for method=${method} seed=${seed}" >&2
      exit "$status"
    fi
  done
done
