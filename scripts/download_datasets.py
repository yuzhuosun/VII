#!/usr/bin/env python3
"""Download supported HuggingFace datasets into reproducible local artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image
from datasets import DownloadConfig
from tqdm import tqdm

from vii.data.hf_datasets import DATASET_CONFIGS, iter_dataset_samples

DATASET_CHOICES = [*DATASET_CONFIGS.keys(), "all"]
SPLIT_CHOICES = ["train", "validation", "test", "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--output-dir", default="data/raw", help="Directory for raw images and annotations.")
    parser.add_argument("--split", choices=SPLIT_CHOICES, default="train")
    parser.add_argument("--cache-dir", default=None, help="Override HuggingFace cache directory.")
    parser.add_argument("--streaming", action="store_true", help="Stream rows instead of preparing the full dataset first.")
    parser.add_argument("--resume", action="store_true", help="Append to existing processed JSONL and skip completed sample IDs.")
    parser.add_argument("--max-retries", type=int, default=5, help="HuggingFace download retry count and per-sample image save retries.")
    parser.add_argument("--num-proc", type=int, default=1, help="HuggingFace download worker count; use 1 on unstable networks.")
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=120,
        help="Seconds for HF Hub HTTP connect/download timeouts; exported to HF_HUB_* timeout env vars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_timeouts(args.download_timeout)
    datasets = DATASET_CONFIGS.keys() if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        export_dataset(
            dataset,
            Path(args.output_dir),
            args.split,
            cache_dir=args.cache_dir,
            streaming=args.streaming,
            resume=args.resume,
            max_retries=args.max_retries,
            num_proc=args.num_proc,
        )


def export_dataset(
    dataset: str,
    output_dir: Path,
    split: str,
    *,
    cache_dir: str | None = None,
    streaming: bool = False,
    resume: bool = False,
    max_retries: int = 5,
    num_proc: int = 1,
) -> Path:
    raw_dir = output_dir / dataset
    image_dir = raw_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = processed_dir / f"{dataset}.jsonl"
    completed_ids = read_completed_ids(jsonl_path) if resume else set()
    mode = "a" if resume and jsonl_path.exists() else "w"
    download_config = DownloadConfig(
        cache_dir=cache_dir,
        resume_download=True,
        max_retries=max_retries,
        num_proc=num_proc,
    )

    with jsonl_path.open(mode, encoding="utf-8") as jsonl:
        iterator = iter_dataset_samples(
            dataset,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
            download_config=download_config,
        )
        for sample in tqdm(iterator, desc=f"Downloading {dataset}"):
            if sample.sample_id in completed_ids:
                continue
            local_image = persist_image_with_retries(
                sample.image_path,
                sample.image,
                image_dir,
                sample.sample_id,
                max_retries=max_retries,
            )
            sample.image_path = str(local_image) if local_image else str(sample.image_path or "")
            sample.image = None
            record = asdict(sample)
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl.flush()

    manifest = {
        "dataset": dataset,
        "hf_id": DATASET_CONFIGS[dataset].hf_id,
        "split": split,
        "streaming": streaming,
        "resume": resume,
        "max_retries": max_retries,
        "num_proc": num_proc,
        "processed_jsonl": str(jsonl_path),
        "raw_dir": str(raw_dir),
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonl_path


def configure_hf_timeouts(seconds: int) -> None:
    """Set HF Hub timeout environment variables unless the caller already did."""

    value = str(seconds)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", value)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", value)


def read_completed_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    completed = set()
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = record.get("sample_id")
            if sample_id:
                completed.add(str(sample_id))
    return completed


def persist_image_with_retries(
    image_path: str | Path,
    image: Any,
    image_dir: Path,
    sample_id: str,
    *,
    max_retries: int,
) -> Path | None:
    last_error: Exception | None = None
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            return persist_image(image_path, image, image_dir, sample_id)
        except Exception as exc:  # noqa: BLE001 - keep downloader resumable across transient network/image errors.
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Failed to persist image for sample {sample_id} after {max_retries} attempts") from last_error


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
