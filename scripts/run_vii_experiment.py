#!/usr/bin/env python3
"""Run one VII experiment from dataset loading through JSONL metadata export."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vii.data.hf_datasets import DATASET_CONFIGS
from vii.experiment import DATASET_SOURCE_CHOICES, MODEL_CHOICES, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--config", default="configs/vii.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", default=None, help="Dataset split override; defaults to dataset config.")
    parser.add_argument(
        "--dataset-source",
        choices=DATASET_SOURCE_CHOICES,
        default="auto",
        help="Use processed JSONL, HuggingFace, or a synthetic smoke-test sample.",
    )
    parser.add_argument(
        "--append-metadata",
        action="store_true",
        help="Append to an existing metadata.jsonl instead of starting a fresh run file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        dataset=args.dataset,
        model=args.model,
        config_path=args.config,
        output_dir=args.output_dir,
        limit=args.limit,
        seed=args.seed,
        dry_run=args.dry_run,
        split=args.split,
        dataset_source=args.dataset_source,
        reset_output=not args.append_metadata,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
