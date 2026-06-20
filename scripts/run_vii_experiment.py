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
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vii.data.hf_datasets import DATASET_CONFIGS, iter_dataset_samples  # noqa: E402
from vii.grounding import GroundingConfig, VisualInstructionGrounder  # noqa: E402
from vii.pipeline import MockI2VProvider  # noqa: E402
from vii.reprogramming import IntentReprogrammer  # noqa: E402
from vii.types import DatasetSample, GenerationResult, GroundedImage, ReprogrammedIntent  # noqa: E402
from vii.utils.io import (  # noqa: E402
    RunPaths,
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    safe_filename,
    utc_timestamp,
)
from vii.utils.logging import get_logger, save_run_provenance  # noqa: E402

SUMMARY_FIELDS = [
    "dataset_id",
    "source_dataset_id",
    "sample_id",
    "model_name",
    "refusal_status",
    "video_path",
    "error_message",
    "timestamp",
]


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
    dataset_config = DATASET_CONFIGS[args.dataset]
    paths = RunPaths(args.run_name, args.output_root).create()
    logger = get_logger("vii.experiment", log_file=paths.api_logs_dir / "run.log")

    save_run_provenance(paths.run_dir, vars(args))
    logger.info("Starting VII run '%s' for dataset '%s'", args.run_name, args.dataset)

    samples = iter_dataset_samples(
        args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )
    if args.limit is not None:
        samples = islice(samples, args.limit)

    summary_rows = run_samples(
        samples,
        dataset_id=args.dataset,
        source_dataset_id=dataset_config.hf_id,
        paths=paths,
        model_name=args.model_name,
    )
    write_summary(paths.summary_csv_path, summary_rows)
    if not paths.eval_results_path.exists():
        atomic_write_json(paths.eval_results_path, {"status": "not_evaluated", "num_records": len(summary_rows)})

    logger.info("Finished VII run '%s' with %d records", args.run_name, len(summary_rows))


def run_samples(
    samples: Iterable[DatasetSample],
    *,
    dataset_id: str,
    source_dataset_id: str,
    paths: RunPaths,
    model_name: str,
) -> list[dict[str, Any]]:
    """Process samples and persist per-sample metadata records."""

    logger = get_logger("vii.experiment")
    reprogrammer = IntentReprogrammer(provider="mock", model="mock")
    grounder = VisualInstructionGrounder(GroundingConfig())
    provider = MockI2VProvider(name=model_name)
    summary_rows: list[dict[str, Any]] = []

    for sample in samples:
        record = process_sample(
            sample,
            dataset_id=dataset_id,
            source_dataset_id=source_dataset_id,
            paths=paths,
            reprogrammer=reprogrammer,
            grounder=grounder,
            provider=provider,
            model_name=model_name,
        )
        append_jsonl(paths.metadata_path, record)
        summary_rows.append(summary_row(record))
        logger.info("Processed sample %s with status %s", sample.sample_id, record["refusal_status"])

    return summary_rows


def process_sample(
    sample: DatasetSample,
    *,
    dataset_id: str,
    source_dataset_id: str,
    paths: RunPaths,
    reprogrammer: IntentReprogrammer,
    grounder: VisualInstructionGrounder,
    provider: MockI2VProvider,
    model_name: str,
) -> dict[str, Any]:
    """Run one sample and return the flattened metadata record requested by the benchmark."""

    record = base_metadata_record(
        sample,
        dataset_id=dataset_id,
        source_dataset_id=source_dataset_id,
        model_name=model_name,
        grounded_image_path=paths.grounded_image_path(sample.sample_id),
    )
    try:
        reprogrammed = reprogrammer.reprogram(sample.prompt, sample.category)
        grounded = grounder.ground(sample.image_path, reprogrammed, record["grounded_image_path"])
        result = provider.generate(
            image_path=grounded.output_path,
            prompt=reprogrammed.visual_instruction,
            output_path=str(paths.video_path(sample.sample_id)),
            sample_id=sample.sample_id,
        )
        write_api_log(paths.api_log_path(sample.sample_id), result, grounded, reprogrammed)
        record.update(success_fields(result, grounded, reprogrammed))
    except Exception as exc:  # keep experiment metadata complete across per-sample failures
        error_path = paths.api_logs_dir / f"{safe_filename(sample.sample_id)}.error.txt"
        atomic_write_text(error_path, traceback.format_exc())
        record.update({"refusal_status": "error", "error_message": str(exc)})
    return record


def base_metadata_record(
    sample: DatasetSample,
    *,
    dataset_id: str,
    source_dataset_id: str,
    model_name: str,
    grounded_image_path: Path,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "source_dataset_id": source_dataset_id,
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
        "timestamp": utc_timestamp(),
        "sample": asdict(sample),
    }


def success_fields(
    result: GenerationResult,
    grounded: GroundedImage,
    reprogrammed: ReprogrammedIntent,
) -> dict[str, Any]:
    return {
        "reprogrammed_instruction": reprogrammed.visual_instruction,
        "grounded_image_path": grounded.output_path,
        "request_id": result.metadata.get("request_id"),
        "job_id": result.metadata.get("job_id"),
        "refusal_status": result.status,
        "video_path": result.video_path,
    }


def write_api_log(
    path: Path,
    result: GenerationResult,
    grounded: GroundedImage,
    reprogrammed: ReprogrammedIntent,
) -> None:
    atomic_write_json(
        path,
        {
            "result": asdict(result),
            "grounded_image": asdict(grounded),
            "reprogrammed_intent": asdict(reprogrammed),
        },
    )


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def summary_row(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in SUMMARY_FIELDS}


if __name__ == "__main__":
    main()
