"""Experiment-level orchestration for VII benchmark runs."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from PIL import Image

from .data.hf_datasets import DATASET_CONFIGS, iter_dataset_samples
from .grounding import GroundingConfig, VisualInstructionGrounder
from .pipeline import I2VProvider, MockI2VProvider, VIIPipeline
from .reprogramming import IntentReprogrammer, MockReprogrammingProvider
from .types import DatasetSample, GenerationResult

DatasetSource = Literal["auto", "processed", "hf", "synthetic"]
ModelName = Literal["kling", "veo", "seedance", "pixverse", "mock"]

MODEL_CHOICES: tuple[str, ...] = ("kling", "veo", "seedance", "pixverse", "mock")
DATASET_SOURCE_CHOICES: tuple[str, ...] = ("auto", "processed", "hf", "synthetic")


@dataclass(slots=True)
class ExperimentConfig:
    """Validated subset of the VII YAML config used by the runner."""

    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    instruction_template: str = "{category}: {prompt}"
    reprogramming_provider: str = "mock"
    reprogramming_model: str | None = None
    max_samples: int | None = 100
    seed: int = 42
    provider_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentSummary:
    """Serializable summary emitted after one experiment run."""

    dataset: str
    model: str
    config: str
    output_dir: str
    limit: int | None
    seed: int
    dry_run: bool
    dataset_source: str
    num_samples: int
    metadata_path: str
    status_counts: dict[str, int]


class TemplateReprogrammingProvider(MockReprogrammingProvider):
    """Deterministic reprogramming provider backed by the YAML template."""

    def __init__(self, template: str, name: str = "mock", max_chars: int = 220):
        super().__init__(name=name, max_chars=max_chars)
        self.template = template

    def generate(self, prompt: str, category: str) -> str:
        compact_prompt = " ".join(prompt.split())[: self.max_chars]
        return self.template.format(prompt=compact_prompt, category=category).strip()


class RequestRecordingI2VProvider:
    """Offline request recorder for model integrations managed outside this repo."""

    def __init__(self, name: str, dry_run: bool = False, provider_kwargs: dict[str, Any] | None = None):
        self.name = name
        self.dry_run = dry_run
        self.provider_kwargs = provider_kwargs or {}

    def generate(self, image_path: str, prompt: str, output_path: str, **kwargs: Any) -> GenerationResult:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        request_path = output.with_suffix(".request.json")
        request = {
            "model": self.name,
            "dry_run": self.dry_run,
            "image_path": image_path,
            "prompt": prompt,
            "requested_output_path": str(output),
            "provider_kwargs": self.provider_kwargs,
            "sample_kwargs": kwargs,
        }
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        return GenerationResult(
            sample_id=str(kwargs.get("sample_id", "unknown")),
            grounded_image_path=image_path,
            video_path=None if self.dry_run else str(request_path),
            provider=self.name,
            status="dry_run" if self.dry_run else "request_recorded",
            metadata={"request_path": str(request_path), "requested_output_path": str(output)},
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and normalize ``configs/vii.yaml`` style settings."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    attack = raw.get("attack", {})
    grounding_values = dict(attack.get("grounding", {}))
    for key in ("fill", "border_fill"):
        if key in grounding_values:
            grounding_values[key] = tuple(grounding_values[key])
    reprogramming = attack.get("reprogramming", {})
    return ExperimentConfig(
        grounding=GroundingConfig(**grounding_values),
        instruction_template=attack.get("instruction_template") or "{category}: {prompt}",
        reprogramming_provider=reprogramming.get("provider", "mock"),
        reprogramming_model=reprogramming.get("model"),
        max_samples=attack.get("max_samples"),
        seed=int(attack.get("seed", 42)),
        provider_kwargs=dict(raw.get("generation", {}).get("provider_kwargs", {})),
    )


def run_experiment(
    *,
    dataset: str,
    model: str,
    config_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    seed: int | None = None,
    dry_run: bool = False,
    split: str | None = None,
    dataset_source: DatasetSource = "auto",
    reset_output: bool = True,
) -> ExperimentSummary:
    """Run one VII experiment and write metadata/summary artifacts."""

    config = load_experiment_config(config_path)
    effective_seed = seed if seed is not None else config.seed
    random.seed(effective_seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if reset_output:
        metadata_path = output_path / "metadata.jsonl"
        if metadata_path.exists():
            metadata_path.unlink()

    sample_limit = limit if limit is not None else config.max_samples
    samples = list(load_samples(dataset, output_path, sample_limit, split, dataset_source, effective_seed))
    pipeline = build_pipeline(config, output_path, build_i2v_provider(model, dry_run, config.provider_kwargs), effective_seed)
    results = pipeline.run_many(samples)

    summary = ExperimentSummary(
        dataset=dataset,
        model=model,
        config=str(config_path),
        output_dir=str(output_path),
        limit=limit,
        seed=effective_seed,
        dry_run=dry_run,
        dataset_source=dataset_source,
        num_samples=len(samples),
        metadata_path=str(pipeline.metadata_path),
        status_counts=count_statuses(results),
    )
    (output_path / "summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_pipeline(config: ExperimentConfig, output_dir: Path, provider: I2VProvider, seed: int) -> VIIPipeline:
    grounding = asdict(config.grounding)
    grounding["seed"] = seed
    reprogram_provider = TemplateReprogrammingProvider(
        template=config.instruction_template,
        name=config.reprogramming_provider,
    )
    return VIIPipeline(
        output_dir=output_dir,
        reprogrammer=IntentReprogrammer(provider=reprogram_provider, model=config.reprogramming_model),
        grounder=VisualInstructionGrounder(GroundingConfig(**grounding)),
        i2v_provider=provider,
    )


def build_i2v_provider(model: str, dry_run: bool, provider_kwargs: dict[str, Any] | None = None) -> I2VProvider:
    if model == "mock":
        return MockI2VProvider(name="mock-i2v-dry-run" if dry_run else "mock-i2v")
    return RequestRecordingI2VProvider(name=model, dry_run=dry_run, provider_kwargs=provider_kwargs)


def load_samples(
    dataset: str,
    output_dir: Path,
    limit: int | None,
    split: str | None,
    dataset_source: DatasetSource,
    seed: int,
) -> Iterable[DatasetSample]:
    image_dir = output_dir / "source_images"
    iterator = iter_samples(dataset, split=split, source=dataset_source, seed=seed)
    for sample in islice(iterator, limit):
        sample.image_path = str(persist_sample_image(sample, image_dir) or sample.image_path)
        sample.image = None
        if not sample.image_path or not Path(sample.image_path).exists():
            raise FileNotFoundError(
                f"Sample {sample.sample_id} does not have a local image. "
                "Run scripts/download_datasets.py first or use --dataset-source synthetic for smoke tests."
            )
        yield sample


def iter_samples(dataset: str, split: str | None, source: DatasetSource, seed: int) -> Iterable[DatasetSample]:
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: {sorted(DATASET_CONFIGS)}")

    if source in ("auto", "processed"):
        processed_path = Path("data") / "processed" / f"{dataset}.jsonl"
        if processed_path.exists():
            yield from iter_processed_jsonl(processed_path)
            return
        if source == "processed":
            raise FileNotFoundError(f"Processed dataset not found: {processed_path}")

    if source == "synthetic":
        yield synthetic_sample(dataset, seed)
        return

    yield from iter_dataset_samples(dataset, split=split)


def iter_processed_jsonl(path: Path) -> Iterable[DatasetSample]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            yield DatasetSample(**record)


def synthetic_sample(dataset: str, seed: int) -> DatasetSample:
    image_dir = Path("outputs") / ".synthetic_inputs"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{dataset}_{seed}.png"
    color = ((seed * 37) % 255, (seed * 73) % 255, (seed * 109) % 255)
    Image.new("RGB", (384, 256), color).save(image_path)
    return DatasetSample(
        sample_id=f"{dataset}-synthetic-{seed}",
        prompt="Synthetic smoke-test prompt for validating the VII experiment pipeline.",
        category="ci_smoke",
        image_path=str(image_path),
        source_dataset=dataset,
        metadata={"synthetic": True},
    )


def persist_sample_image(sample: DatasetSample, image_dir: Path) -> Path | None:
    image_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in sample.sample_id)
    if sample.image is not None:
        output = image_dir / f"{safe_id}.png"
        if isinstance(sample.image, Image.Image):
            sample.image.save(output)
            return output
        if isinstance(sample.image, dict):
            nested_path = sample.image.get("path") or sample.image.get("filename")
            if nested_path and Path(nested_path).exists():
                return copy_image(Path(nested_path), image_dir, safe_id)
            if sample.image.get("bytes"):
                output.write_bytes(sample.image["bytes"])
                return output
    if sample.image_path and Path(sample.image_path).exists():
        return copy_image(Path(sample.image_path), image_dir, safe_id)
    return None


def copy_image(source: Path, image_dir: Path, safe_id: str) -> Path:
    output = image_dir / f"{safe_id}{source.suffix or '.png'}"
    if source.resolve() != output.resolve():
        shutil.copy2(source, output)
    return output


def count_statuses(results: Iterable[GenerationResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts
