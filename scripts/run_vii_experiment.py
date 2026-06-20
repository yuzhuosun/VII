#!/usr/bin/env python3
"""Run a VII experiment and persist standardized result artifacts."""

from __future__ import annotations

import argparse
import csv
import io
import sys
import traceback
from dataclasses import asdict
from itertools import islice
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from typing import Any

from vii.data.hf_datasets import DATASET_CONFIGS, iter_dataset_samples
from vii.grounding import GroundingConfig, VisualInstructionGrounder
from vii.pipeline import MockI2VProvider
from vii.reprogramming import IntentReprogrammer
from vii.types import DatasetSample
from vii.utils.io import RunPaths, append_jsonl, atomic_write_json, atomic_write_text, safe_filename, utc_timestamp
from vii.utils.logging import get_logger, save_run_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS), help="Dataset key to load.")
    parser.add_argument("--split", default=None, help="Dataset split. Defaults to the dataset config split.")
    parser.add_argument("--run-name", required=True, help="Name under outputs/{run_name}.")
    parser.add_argument("--output-root", default="outputs", help="Root output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to process.")
    parser.add_argument("--cache-dir", default=None, help="Optional HuggingFace cache directory.")
    parser.add_argument("--streaming", action="store_true", help="Stream the dataset from HuggingFace.")
    parser.add_argument("--model-name", default="mock-i2v", help="I2V model/provider name recorded in metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = RunPaths(args.run_name, args.output_root).create()
    logger = get_logger("vii.experiment", log_file=paths.api_logs_dir / "run.log")
    run_config = vars(args)
    save_run_provenance(paths.run_dir, run_config)

    logger.info("Starting VII run '%s' for dataset '%s'", args.run_name, args.dataset)
    reprogrammer = IntentReprogrammer(provider="mock", model="mock")
    grounder = VisualInstructionGrounder(GroundingConfig())
    provider = MockI2VProvider(name=args.model_name)

    samples = iter_dataset_samples(args.dataset, split=args.split, cache_dir=args.cache_dir, streaming=args.streaming)
    if args.limit is not None:
        samples = islice(samples, args.limit)

    summary_rows: list[dict[str, Any]] = []
    for sample in samples:
        record = process_sample(sample, dataset_id=DATASET_CONFIGS[args.dataset].hf_id, paths=paths, reprogrammer=reprogrammer, grounder=grounder, provider=provider, model_name=args.model_name)
        append_jsonl(paths.metadata_path, record)
        summary_rows.append(_summary_row(record))
        logger.info("Processed sample %s with status %s", sample.sample_id, record.get("refusal_status"))

    write_summary(paths.summary_csv_path, summary_rows)
    if not paths.eval_results_path.exists():
        atomic_write_json(paths.eval_results_path, {"status": "not_evaluated", "num_records": len(summary_rows)})
    logger.info("Finished VII run '%s' with %d records", args.run_name, len(summary_rows))


def process_sample(
    sample: DatasetSample,
    *,
    dataset_id: str,
    paths: RunPaths,
    reprogrammer: IntentReprogrammer,
    grounder: VisualInstructionGrounder,
    provider: MockI2VProvider,
    model_name: str,
) -> dict[str, Any]:
    timestamp = utc_timestamp()
    grounded_image_path = paths.grounded_image_path(sample.sample_id)
    requested_video_path = paths.video_path(sample.sample_id)
    api_log_path = paths.api_log_path(sample.sample_id)
    base_record: dict[str, Any] = {
        "dataset_id": dataset_id,
        "sample_id": sample.sample_id,
        "original_prompt": sample.prompt,
        "reprogrammed_instruction": None,
        "grounded_image_path": str(grounded_image_path),
        "model_name": model_name,
        "request_id": None,
        "job_id": None,
        "refusal_status": "unknown",
        "video_path": None,
        "error_message": None,
        "timestamp": timestamp,
        "sample": asdict(sample),
    }
    try:
        reprogrammed = reprogrammer.reprogram(sample.prompt, sample.category)
        grounded = grounder.ground(sample.image_path, reprogrammed, grounded_image_path)
        result = provider.generate(
            image_path=grounded.output_path,
            prompt=reprogrammed.visual_instruction,
            output_path=str(requested_video_path),
            sample_id=sample.sample_id,
        )
        request_id = result.metadata.get("request_id") or result.metadata.get("requested_output_path")
        job_id = result.metadata.get("job_id")
        api_payload = {"result": asdict(result), "grounded_image": asdict(grounded), "reprogrammed_intent": asdict(reprogrammed)}
        atomic_write_json(api_log_path, api_payload)
        base_record.update(
            {
                "reprogrammed_instruction": reprogrammed.visual_instruction,
                "grounded_image_path": grounded.output_path,
                "request_id": request_id,
                "job_id": job_id,
                "refusal_status": result.status,
                "video_path": result.video_path,
            }
        )
    except Exception as exc:  # keep experiment metadata complete across per-sample failures
        error_path = paths.api_logs_dir / f"{safe_filename(sample.sample_id)}.error.txt"
        atomic_write_text(error_path, traceback.format_exc())
        base_record.update({"refusal_status": "error", "error_message": str(exc)})
    return base_record


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["dataset_id", "sample_id", "model_name", "refusal_status", "video_path", "error_message", "timestamp"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in ["dataset_id", "sample_id", "model_name", "refusal_status", "video_path", "error_message", "timestamp"]}


if __name__ == "__main__":
    main()
