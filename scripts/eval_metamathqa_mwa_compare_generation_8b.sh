#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/zck/miniconda3/envs/torch2.7/bin/python}"
GPU_ARGS=()
MAX_EXAMPLES=5000
MAX_PROMPT_LENGTH=384
MAX_NEW_TOKENS=128
BATCH_SIZE=1
CHECKPOINT_MODE="best"
NOTES_CONTAINS="metamathqa_8b_residual_joint_capture_bsz8"
PROGRESS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ARGS=(--gpu "${2:?missing GPU value after --gpu}")
      shift 2
      ;;
    --max-examples)
      MAX_EXAMPLES="${2:?missing value after --max-examples}"
      shift 2
      ;;
    --max-prompt-length)
      MAX_PROMPT_LENGTH="${2:?missing value after --max-prompt-length}"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="${2:?missing value after --max-new-tokens}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:?missing value after --batch-size}"
      shift 2
      ;;
    --no-progress)
      PROGRESS_ARGS=(--no-progress)
      shift
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
methods = ["lora", "svd_preserve_only", "pc_lora", "residual_mwa_pc_lora"]
root = pathlib.Path("outputs/runs/meta_llama_llama_3_1_8b")
for method in methods:
    matches = []
    for path in root.glob(f"{method}_*metamathqa_train_100000*r8_k16_seed0_*"):
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
  echo "no matched MetaMathQA M_WA comparison runs found" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" scripts/eval_math_generation.py
  "${GPU_ARGS[@]}"
  --checkpoint-mode "$CHECKPOINT_MODE"
  --max-examples "$MAX_EXAMPLES"
  --max-prompt-length "$MAX_PROMPT_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --batch-size "$BATCH_SIZE"
  "${PROGRESS_ARGS[@]}"
)
for RUN_DIR in "${RUN_DIRS[@]}"; do
  CMD+=(--run-dir "$RUN_DIR")
done

printf 'Evaluating generation EM for %d runs:\n' "${#RUN_DIRS[@]}"
printf '  %s\n' "${RUN_DIRS[@]}"
"${CMD[@]}"
