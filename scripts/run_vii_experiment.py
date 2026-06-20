#!/usr/bin/env python3
"""Run a VII experiment and persist standardized result artifacts."""

from __future__ import annotations

import argparse
import csv
import io
import sys
import traceback
from dataclasses import asdict
import json
import random
import shutil
import sys
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
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vii.data.hf_datasets import DATASET_CONFIGS, iter_dataset_samples
from vii.grounding import GroundingConfig, VisualInstructionGrounder
from vii.pipeline import I2VProvider, MockI2VProvider, VIIPipeline
from vii.reprogramming import IntentReprogrammer, MockReprogrammingProvider
from vii.types import DatasetSample, GenerationResult

MODEL_CHOICES = ("kling", "veo", "seedance", "pixverse", "mock")


class TemplateReprogrammingProvider(MockReprogrammingProvider):
    """Deterministic provider that applies the configured instruction template."""

    def __init__(self, template: str, name: str = "mock", max_chars: int = 220):
        super().__init__(name=name, max_chars=max_chars)
        self.template = template

    def generate(self, prompt: str, category: str) -> str:
        compact_prompt = " ".join(prompt.split())[: self.max_chars]
        return self.template.format(prompt=compact_prompt, category=category).strip()


class PlaceholderI2VProvider:
    """Offline stand-in for named commercial I2V backends.

    The runner intentionally does not embed vendor API calls. It preserves the
    experiment contract by writing a request JSON artifact for later dispatch.
    """

    def __init__(self, name: str, dry_run: bool = False):
        self.name = name
        self.dry_run = dry_run

    def generate(self, image_path: str, prompt: str, output_path: str, **kwargs: Any) -> GenerationResult:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        request_path = output.with_suffix(".request.json")
        request_path.write_text(
            json.dumps(
                {
                    "model": self.name,
                    "dry_run": self.dry_run,
                    "image_path": image_path,
                    "prompt": prompt,
                    "requested_output_path": str(output),
                    "kwargs": kwargs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        status = "dry_run" if self.dry_run else "request_recorded"
        return GenerationResult(
            sample_id=str(kwargs.get("sample_id", "unknown")),
            grounded_image_path=image_path,
            video_path=None if self.dry_run else str(request_path),
            provider=self.name,
            status=status,
            metadata={"request_path": str(request_path), "requested_output_path": str(output)},
        )


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
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--config", default="configs/vii.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", default=None, help="Dataset split override; defaults to dataset config.")
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
    config = load_config(Path(args.config))
    seed = args.seed if args.seed is not None else int(config.get("attack", {}).get("seed", 42))
    random.seed(seed)

    output_dir = Path(args.output_dir)
    samples = list(load_samples(args.dataset, output_dir, args.limit or config.get("attack", {}).get("max_samples"), args.split))
    provider = build_i2v_provider(args.model, dry_run=args.dry_run)
    pipeline = build_pipeline(config, output_dir, provider, seed)

    results = pipeline.run_many(samples)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "model": args.model,
                "config": str(Path(args.config)),
                "output_dir": str(output_dir),
                "limit": args.limit,
                "seed": seed,
                "dry_run": args.dry_run,
                "num_samples": len(samples),
                "status_counts": count_statuses(results),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote metadata to {pipeline.metadata_path}")
    print(f"Wrote summary to {summary_path}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_pipeline(config: dict[str, Any], output_dir: Path, provider: I2VProvider, seed: int) -> VIIPipeline:
    attack = config.get("attack", {})
    grounding_cfg = dict(attack.get("grounding", {}))
    grounding_cfg["seed"] = seed
    for key in ("fill", "border_fill"):
        if key in grounding_cfg:
            grounding_cfg[key] = tuple(grounding_cfg[key])
    template = attack.get("instruction_template") or "{category}: {prompt}"
    reprogramming = attack.get("reprogramming", {})
    reprogram_provider = TemplateReprogrammingProvider(template=template, name=reprogramming.get("provider", "mock"))
    return VIIPipeline(
        output_dir=output_dir,
        reprogrammer=IntentReprogrammer(provider=reprogram_provider, model=reprogramming.get("model")),
        grounder=VisualInstructionGrounder(GroundingConfig(**grounding_cfg)),
        i2v_provider=provider,
    )


def build_i2v_provider(model: str, dry_run: bool) -> I2VProvider:
    if model == "mock":
        return MockI2VProvider(name="mock-i2v-dry-run" if dry_run else "mock-i2v")
    return PlaceholderI2VProvider(name=model, dry_run=dry_run)


def load_samples(dataset: str, output_dir: Path, limit: int | None, split: str | None) -> Iterable[DatasetSample]:
    image_dir = output_dir / "source_images"
    iterator = iter_local_or_hf_samples(dataset, split=split)
    for sample in islice(iterator, limit):
        sample.image_path = str(persist_sample_image(sample, image_dir) or sample.image_path)
        sample.image = None
        if not sample.image_path or not Path(sample.image_path).exists():
            raise FileNotFoundError(f"Sample {sample.sample_id} does not have a local image. Run scripts/download_datasets.py first.")
        yield sample


def iter_local_or_hf_samples(dataset: str, split: str | None) -> Iterable[DatasetSample]:
    """Prefer downloaded processed JSONL, then fall back to HuggingFace loading."""

    processed_path = ROOT / "data" / "processed" / f"{dataset}.jsonl"
    if processed_path.exists():
        with processed_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                yield DatasetSample(**record)
        return

    yield from iter_dataset_samples(dataset, split=split)


def persist_sample_image(sample: DatasetSample, image_dir: Path) -> Path | None:
    image_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in sample.sample_id)
    if sample.image is not None:
        output = image_dir / f"{safe_id}.png"
        if isinstance(sample.image, Image.Image):
            sample.image.save(output)
            return output
        if isinstance(sample.image, dict) and sample.image.get("bytes"):
            output.write_bytes(sample.image["bytes"])
            return output
    if sample.image_path and Path(sample.image_path).exists():
        source = Path(sample.image_path)
        output = image_dir / f"{safe_id}{source.suffix or '.png'}"
        if source.resolve() != output.resolve():
            shutil.copy2(source, output)
        return output
    return None


def count_statuses(results: list[GenerationResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


if __name__ == "__main__":
    main()
