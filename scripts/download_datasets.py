#!/usr/bin/env python3
"""Download supported HuggingFace datasets into reproducible local artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image
from tqdm import tqdm

from vii.data.hf_datasets import DATASET_CONFIGS, iter_dataset_samples

DATASET_CHOICES = [*DATASET_CONFIGS.keys(), "all"]
SPLIT_CHOICES = ["train", "validation", "test", "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--output-dir", default="data/raw", help="Directory for raw images and annotations.")
    parser.add_argument("--split", choices=SPLIT_CHOICES, default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = DATASET_CONFIGS.keys() if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        export_dataset(dataset, Path(args.output_dir), args.split)


def export_dataset(dataset: str, output_dir: Path, split: str) -> Path:
    raw_dir = output_dir / dataset
    image_dir = raw_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = processed_dir / f"{dataset}.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for sample in tqdm(iter_dataset_samples(dataset, split=split), desc=f"Downloading {dataset}"):
            local_image = persist_image(sample.image_path, sample.image, image_dir, sample.sample_id)
            sample.image_path = str(local_image) if local_image else str(sample.image_path or "")
            sample.image = None
            record = asdict(sample)
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": dataset,
        "hf_id": DATASET_CONFIGS[dataset].hf_id,
        "split": split,
        "processed_jsonl": str(jsonl_path),
        "raw_dir": str(raw_dir),
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonl_path


def persist_image(image_path: str | Path, image: Any, image_dir: Path, sample_id: str) -> Path | None:
    safe_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in sample_id)
    if image is not None:
        suffix = ".png"
        filename = getattr(image, "filename", None)
        if filename:
            suffix = Path(filename).suffix or suffix
        output_path = image_dir / f"{safe_id}{suffix}"
        if isinstance(image, Image.Image):
            image.save(output_path)
            return output_path
        if isinstance(image, dict):
            nested_path = image.get("path") or image.get("filename")
            if nested_path and Path(nested_path).exists():
                return copy_image(Path(nested_path), image_dir, safe_id)
            if image.get("bytes"):
                output_path.write_bytes(image["bytes"])
                return output_path
        return None

    if image_path:
        source = Path(image_path)
        if source.exists():
            return copy_image(source, image_dir, safe_id)
    return None


def copy_image(source: Path, image_dir: Path, safe_id: str) -> Path:
    output_path = image_dir / f"{safe_id}{source.suffix or '.png'}"
    if source.resolve() != output_path.resolve():
        shutil.copy2(source, output_path)
    return output_path


if __name__ == "__main__":
    main()
