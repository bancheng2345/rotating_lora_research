#!/usr/bin/env python
"""Builds final summaries from run-level metric jsonl files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.run_summary import save_final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Single run directory to summarize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = save_final_summary(Path(args.run_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
