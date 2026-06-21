#!/usr/bin/env python3
"""Run one VII experiment from dataset loading through JSONL metadata export."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vii.data.hf_datasets import DATASET_CONFIGS, iter_dataset_samples
from vii.grounding import GroundingConfig, VisualInstructionGrounder
from vii.models import KlingI2VClient, MockI2VClient, PixVerseI2VClient, SeedanceI2VClient, VeoI2VClient
from vii.pipeline import I2VProvider, VIIPipeline
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    config = load_config(Path(args.config))
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
    }
    return clients[model]()


def build_provider_kwargs(config: dict[str, Any], model: str, wait: bool, overrides: list[str]) -> dict[str, Any]:
    """Build provider kwargs from shared generation config, model config, and CLI overrides."""

    model_config_path = ROOT / "configs" / "models.yaml"
    model_config = load_config(model_config_path).get("models", {}).get(model, {}) if model_config_path.exists() else {}
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
