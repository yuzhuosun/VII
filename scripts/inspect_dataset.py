#!/usr/bin/env python3
"""Inspect processed dataset JSONL files for integrity only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset key or path to a processed JSONL file.")
    parser.add_argument("--processed-dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.dataset)
    if path.suffix != ".jsonl":
        path = Path(args.processed_dir) / f"{args.dataset}.jsonl"
    inspect_jsonl(path)


def inspect_jsonl(path: Path) -> None:
    category_counts: Counter[str] = Counter()
    missing_images: list[str] = []
    total = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            category_counts[record.get("category") or "uncategorized"] += 1
            image_path = record.get("image_path")
            if not image_path or not Path(image_path).exists():
                missing_images.append(str(record.get("sample_id", total)))

    print(f"Dataset file: {path}")
    print(f"Sample count: {total}")
    print("Category distribution:")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count}")
    print(f"Missing images: {len(missing_images)}")
    if missing_images:
        preview = ", ".join(missing_images[:20])
        suffix = " ..." if len(missing_images) > 20 else ""
        print(f"Missing image sample_ids: {preview}{suffix}")


if __name__ == "__main__":
    main()
