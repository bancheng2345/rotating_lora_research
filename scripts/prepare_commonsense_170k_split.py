#!/usr/bin/env python
"""Prepares a deterministic local train/validation split for commonsense_170k."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-file",
        default="data/commonsense_170k/data/train-00000-of-00001.parquet",
        help="Local parquet file downloaded from Hugging Face.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--val-size", type=int, default=5000, help="Validation set size.")
    parser.add_argument("--out-dir", default="data/splits", help="Output directory for split jsonl files.")
    return parser.parse_args()


def write_jsonl(path: Path, rows) -> None:
    """Writes a dataset split to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_file)
    out_dir = Path(args.out_dir)
    dataset = load_dataset("parquet", data_files=str(source_path))["train"]
    dataset = dataset.shuffle(seed=int(args.seed))

    total_rows = int(dataset.num_rows)
    val_size = min(int(args.val_size), total_rows)
    train_size = total_rows - val_size

    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, total_rows))

    train_path = out_dir / f"commonsense_170k_train_{train_size}.jsonl"
    val_path = out_dir / f"commonsense_170k_val_{val_size}.jsonl"
    metadata_path = out_dir / "commonsense_170k_split_metadata.json"

    write_jsonl(train_path, train_dataset)
    write_jsonl(val_path, val_dataset)
    metadata_path.write_text(
        json.dumps(
            {
                "seed": int(args.seed),
                "source_file": str(source_path.resolve()),
                "total_rows": total_rows,
                "train_path": str(train_path.resolve()),
                "train_rows": train_size,
                "val_path": str(val_path.resolve()),
                "val_rows": val_size,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "train_path": str(train_path),
                "train_rows": train_size,
                "val_path": str(val_path),
                "val_rows": val_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
