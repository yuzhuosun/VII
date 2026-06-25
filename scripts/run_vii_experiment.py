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
import os
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vii.data.hf_datasets import DATASET_CONFIGS, iter_dataset_samples
from vii.grounding import GroundingConfig, VisualInstructionGrounder
from vii.models import GenericI2VClient, KlingI2VClient, MockI2VClient, PixVerseI2VClient, SeedanceI2VClient, VeoI2VClient
from vii.pipeline import I2VProvider, VIIPipeline
from vii.reprogramming import IntentReprogrammer, MockReprogrammingProvider
from vii.types import DatasetSample, GenerationResult

MODEL_CHOICES = ("kling", "veo", "seedance", "pixverse", "generic_i2v", "mock")


class TemplateReprogrammingProvider(MockReprogrammingProvider):
    """Deterministic provider that applies the configured instruction template."""

    def __init__(self, template: str, name: str = "mock", max_chars: int = 220):
        super().__init__(name=name, max_chars=max_chars)
        self.template = template

    def generate(self, prompt: str, category: str) -> str:
        compact_prompt = " ".join(prompt.split())[: self.max_chars]
        return self.template.format(prompt=compact_prompt, category=category).strip()


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
    parser.add_argument("--wait", action="store_true", help="Poll provider APIs and download completed videos.")
    parser.add_argument("--split", default=None, help="Dataset split override; defaults to dataset config.")
    parser.add_argument(
        "--provider-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override generation provider keyword arguments. Values are parsed as JSON when possible.",
    )
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
    apply_api_config(config)
    seed = args.seed if args.seed is not None else int(config.get("attack", {}).get("seed", 42))
    random.seed(seed)

    output_dir = Path(args.output_dir)
    samples = list(load_samples(args.dataset, output_dir, args.limit or config.get("attack", {}).get("max_samples"), args.split))
    provider = build_i2v_provider(args.model, dry_run=args.dry_run)
    provider_kwargs = build_provider_kwargs(config, args.model, wait=args.wait, overrides=args.provider_kwarg)
    pipeline = build_pipeline(config, output_dir, provider, seed, provider_kwargs)

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
                "wait": args.wait,
                "provider_kwargs": provider_kwargs,
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


def apply_api_config(config: dict[str, Any]) -> None:
    """Apply API settings from YAML config to environment variables.

    Secret values can be provided directly with ``api.generic_i2v.api_key`` or,
    preferably, by naming an environment variable such as ``DEEPSEEK_API_KEY``.
    The generic I2V client still receives the standard ``I2V_*`` variables.
    """

    api_config = config.get("api", {})
    generic_config = api_config.get("generic_i2v", {})
    if not isinstance(generic_config, dict):
        return

    mappings = {
        "I2V_API_KEY": ("api_key", "api_key_env"),
        "I2V_BASE_URL": ("base_url", "base_url_env"),
        "I2V_MODEL": ("model", "model_env"),
        "I2V_ENDPOINT_PATH": ("endpoint_path", "endpoint_path_env"),
        "I2V_STATUS_PATH_TEMPLATE": ("status_path_template", "status_path_template_env"),
    }
    for env_name, (value_key, env_key) in mappings.items():
        if os.getenv(env_name):
            continue
        value = generic_config.get(value_key)
        if value is None and generic_config.get(env_key):
            value = os.getenv(str(generic_config[env_key]))
        if value is not None:
            os.environ[env_name] = str(value)


def build_pipeline(
    config: dict[str, Any],
    output_dir: Path,
    provider: I2VProvider,
    seed: int,
    provider_kwargs: dict[str, Any] | None = None,
) -> VIIPipeline:
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
        provider_kwargs=provider_kwargs,
    )


def build_i2v_provider(model: str, dry_run: bool) -> I2VProvider:
    if model == "mock" or dry_run:
        return MockI2VClient()
    clients = {
        "kling": KlingI2VClient,
        "veo": VeoI2VClient,
        "seedance": SeedanceI2VClient,
        "pixverse": PixVerseI2VClient,
        "generic_i2v": GenericI2VClient,
    }
    return clients[model]()


def build_provider_kwargs(config: dict[str, Any], model: str, wait: bool, overrides: list[str]) -> dict[str, Any]:
    """Build provider kwargs from shared generation config, model config, and CLI overrides."""

    model_config_path = ROOT / "configs" / "models.yaml"
    model_config = load_config(model_config_path).get("models", {}).get(model, {}) if model_config_path.exists() else {}
    model_config.update(config.get("models", {}).get(model, {}))
    generation = config.get("generation", {})
    provider_kwargs = dict(generation.get("provider_kwargs", {}))
    provider_kwargs.update(
        {
            "resolution": model_config.get("default_resolution"),
            "duration": model_config.get("video_duration_seconds"),
            "model": model_config.get("api_model"),
            "poll_interval": model_config.get("poll_interval_seconds"),
            "max_retries": model_config.get("max_retries"),
            "wait": wait,
        }
    )
    provider_kwargs = {key: value for key, value in provider_kwargs.items() if value is not None}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--provider-kwarg must be KEY=VALUE, got: {override}")
        key, value = override.split("=", 1)
        provider_kwargs[key] = parse_provider_value(value)
    return provider_kwargs


def parse_provider_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


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
        Image = import_pillow_image()
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


def import_pillow_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to handle dataset images, but `from PIL import Image` failed. "
            "Install it in the same Python environment with `python -m pip install --upgrade --force-reinstall pillow`, "
            "and make sure your working directory does not contain a file or folder named `PIL`."
        ) from exc
    return Image


def count_statuses(results: list[GenerationResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


if __name__ == "__main__":
    main()
