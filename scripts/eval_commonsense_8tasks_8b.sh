#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/zck/miniconda3/envs/torch2.7/bin/python}"
GPU_ARGS=()
DATA_ROOT="data/commonsense_eval"
SPLIT="validation"
BATCH_SIZE=8
MAX_EXAMPLES_PER_TASK=""
CHECKPOINT_MODE="best"
NOTES_CONTAINS=""
PROGRESS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ARGS=(--gpu "${2:?missing GPU value after --gpu}")
      shift 2
      ;;
    --data-root)
      DATA_ROOT="${2:?missing value after --data-root}"
      shift 2
      ;;
    --split)
      SPLIT="${2:?missing value after --split}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:?missing value after --batch-size}"
      shift 2
      ;;
    --max-examples-per-task)
      MAX_EXAMPLES_PER_TASK="${2:?missing value after --max-examples-per-task}"
      shift 2
      ;;
    --checkpoint-mode)
      CHECKPOINT_MODE="${2:?missing value after --checkpoint-mode}"
      shift 2
      ;;
    --notes-contains)
      if [[ $# -lt 2 ]]; then
        echo "missing value after --notes-contains" >&2
        exit 2
      fi
      NOTES_CONTAINS="$2"
      shift 2
      ;;
    --no-progress)
      PROGRESS_ARGS=(--no-progress)
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mapfile -t RUN_DIRS < <("$PYTHON_BIN" - "$NOTES_CONTAINS" <<'PY'
import json
import pathlib
import sys

notes = sys.argv[1]
methods = ["lora", "oplora", "lora_nsc", "svd_preserve_only", "pc_lora", "residual_mwa_pc_lora"]
root = pathlib.Path("outputs/runs/meta_llama_llama_3_1_8b")
for method in methods:
    matches = []
    for path in root.glob(f"{method}_*commonsense_170k_train_165420*r8_k16_seed0_*"):
        summary_path = path / "final_summary.json"
        config_path = path / "config_resolved.yaml"
        if not summary_path.exists() or not config_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if summary.get("method") != method:
            continue
        if notes and notes not in config_path.read_text(encoding="utf-8"):
            continue
        matches.append(path)
    matches.sort(key=lambda item: (item / "final_summary.json").stat().st_mtime, reverse=True)
    if matches:
        print(matches[0])
PY
)

if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
  echo "no matched Commonsense170k 8B runs found" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" scripts/eval_commonsense_8tasks.py
  "${GPU_ARGS[@]}"
  --data-root "$DATA_ROOT"
  --split "$SPLIT"
  --checkpoint-mode "$CHECKPOINT_MODE"
  --batch-size "$BATCH_SIZE"
  "${PROGRESS_ARGS[@]}"
)
if [[ -n "$MAX_EXAMPLES_PER_TASK" ]]; then
  CMD+=(--max-examples-per-task "$MAX_EXAMPLES_PER_TASK")
fi
for RUN_DIR in "${RUN_DIRS[@]}"; do
  CMD+=(--run-dir "$RUN_DIR")
done

printf 'Evaluating Commonsense 8-task accuracy for %d runs:\n' "${#RUN_DIRS[@]}"
printf '  %s\n' "${RUN_DIRS[@]}"
"${CMD[@]}"
