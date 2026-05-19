#!/usr/bin/env python
"""Prepares deterministic JSONL splits from the local MetaMathQA JSON file."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-file",
        default="data/MetaMathQA/MetaMathQA-395K.json",
        help="Local MetaMathQA JSON file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--train-size", type=int, default=100000, help="Number of train rows to write.")
    parser.add_argument("--val-size", type=int, default=5000, help="Number of validation rows to write.")
    parser.add_argument("--out-dir", default="data/splits", help="Output directory for JSONL splits.")
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Writes rows as JSONL and returns the number of rows written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_file)
    out_dir = Path(args.out_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"MetaMathQA source file not found: {source_path}")

    rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {source_path}, got {type(rows).__name__}")
    required_columns = {"query", "response"}
    missing = required_columns.difference(rows[0].keys()) if rows else required_columns
    if missing:
        raise ValueError(f"MetaMathQA rows are missing required columns: {sorted(missing)}")

    rng = random.Random(int(args.seed))
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    val_size = min(int(args.val_size), len(indices))
    train_size = min(int(args.train_size), max(0, len(indices) - val_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]

    train_path = out_dir / f"metamathqa_train_{train_size}.jsonl"
    val_path = out_dir / f"metamathqa_val_{val_size}.jsonl"
    metadata_path = out_dir / "metamathqa_split_metadata.json"

    written_train = write_jsonl(train_path, (rows[i] for i in train_indices))
    written_val = write_jsonl(val_path, (rows[i] for i in val_indices))
    metadata = {
        "seed": int(args.seed),
        "source_file": str(source_path.resolve()),
        "total_rows": len(rows),
        "train_path": str(train_path.resolve()),
        "train_rows": written_train,
        "val_path": str(val_path.resolve()),
        "val_rows": written_val,
        "columns": sorted(rows[0].keys()) if rows else [],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
