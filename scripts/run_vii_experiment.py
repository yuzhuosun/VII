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
