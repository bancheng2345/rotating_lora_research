#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/zck/miniconda3/envs/torch2.7/bin/python}"
GPU="0"
SEEDS=("1" "2")
GEN_BATCH_SIZE="2"
MAX_EXAMPLES="5000"
MAX_NEW_TOKENS="128"
NOTES="metamathqa_8b_residual_mwa_lambda01_seed_sweep"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="${2:?missing GPU value after --gpu}"
      shift 2
      ;;
    --seeds)
      IFS=',' read -r -a SEEDS <<< "${2:?missing comma-separated seeds after --seeds}"
      shift 2
      ;;
    --gen-batch-size)
      GEN_BATCH_SIZE="${2:?missing value after --gen-batch-size}"
      shift 2
      ;;
    --max-examples)
      MAX_EXAMPLES="${2:?missing value after --max-examples}"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="${2:?missing value after --max-new-tokens}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

latest_completed_run() {
  local seed="$1"
  "$PYTHON_BIN" - "$seed" "$NOTES" <<'PY'
import json
import pathlib
import sys

seed = int(sys.argv[1])
notes = sys.argv[2]
root = pathlib.Path("outputs/runs/meta_llama_llama_3_1_8b")
matches = []
for path in root.glob(f"residual_mwa_pc_lora_*metamathqa_train_100000*r8_k16_seed{seed}_*"):
    summary_path = path / "final_summary.json"
    config_path = path / "config_resolved.yaml"
    if not summary_path.exists() or not config_path.exists():
        continue
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if summary.get("method") != "residual_mwa_pc_lora":
        continue
    config_text = config_path.read_text(encoding="utf-8", errors="ignore")
    if notes not in config_text:
        continue
    matches.append(path)
matches.sort(key=lambda item: (item / "final_summary.json").stat().st_mtime, reverse=True)
if not matches:
    raise SystemExit(f"no completed residual_mwa_pc_lora run found for seed={seed}")
print(matches[0])
PY
}

for SEED in "${SEEDS[@]}"; do
  echo "=== train residual_mwa_pc_lora lambda_align=0.1 seed=${SEED} gpu=${GPU} ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" scripts/train.py \
    --base-config configs/base.yaml \
    --config configs/presets/models/llama31_8b_local.yaml \
    --config configs/presets/datasets/metamathqa.yaml \
    --config configs/presets/runs/metamathqa_full_8b_bsz8.yaml \
    --config configs/experiments/residual_mwa_pc_lora.yaml \
    --gpu "$GPU" \
    --override training.eval_batch_size=1 \
    --override training.eval_steps=300 \
    --override training.save_steps=300 \
    --override metrics.eval_steps=300 \
    --override metrics.mechanism_steps=300 \
    --override lora.lambda_align=0.1 \
    --override experiment.seed="$SEED" \
    --override experiment.notes="$NOTES" \
    --override metrics.compute_rho_k=false

  RUN_DIR="$(latest_completed_run "$SEED")"
  echo "=== generation EM residual_mwa_pc_lora lambda_align=0.1 seed=${SEED} ==="
  "$PYTHON_BIN" scripts/eval_math_generation.py \
    --run-dir "$RUN_DIR" \
    --gpu "$GPU" \
    --checkpoint-mode best \
    --max-examples "$MAX_EXAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$GEN_BATCH_SIZE"
  echo "=== complete seed=${SEED}: ${RUN_DIR} ==="
done

