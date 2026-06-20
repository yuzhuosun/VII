"""End-to-end VII pipeline orchestration."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from .grounding import GroundingConfig, VisualInstructionGrounder
from .reprogramming import IntentReprogrammer
from .models.base import SafetyAcknowledgementRequired
from .types import DatasetSample, GenerationResult


class I2VProvider(Protocol):
    """Interface for image-to-video generation backends."""

    name: str

    def generate(self, image_path: str, prompt: str, output_path: str, **kwargs: Any) -> GenerationResult:
        """Generate a video conditioned on an image and text prompt."""


SAFETY_NOTICE = """# Safety Notice

Outputs in this directory are generated only for controlled AI safety red-teaming, vulnerability assessment, and defensive research. They must not be used to enable, promote, or distribute harmful, abusive, illegal, or malicious content. Keep generated artifacts access-controlled and follow applicable laws, platform policies, and institutional review requirements.
"""


@dataclass(slots=True)
class MockI2VProvider:
    """Offline backend that records inputs instead of calling a model API."""

    name: str = "mock-i2v"

    def generate(self, image_path: str, prompt: str, output_path: str, **kwargs: Any) -> GenerationResult:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        placeholder = Path(output_path).with_suffix(".json")
        placeholder.write_text(
            json.dumps({"image_path": image_path, "prompt": prompt, "kwargs": kwargs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return GenerationResult(
            sample_id=str(kwargs.get("sample_id", "unknown")),
            grounded_image_path=image_path,
            video_path=str(placeholder),
            provider=self.name,
            status="mocked",
            metadata={"requested_output_path": output_path},
        )


class VIIPipeline:
    """Pipeline for loading samples, reprogramming intent, grounding, and I2V calls."""

    def __init__(
        self,
        output_dir: str | Path = "outputs",
        reprogrammer: IntentReprogrammer | None = None,
        grounder: VisualInstructionGrounder | None = None,
        i2v_provider: I2VProvider | None = None,
        acknowledge_safety_research_use: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.videos_dir = self.output_dir / "videos"
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.reprogrammer = reprogrammer or IntentReprogrammer(provider="mock")
        self.grounder = grounder or VisualInstructionGrounder(GroundingConfig())
        self.i2v_provider = i2v_provider or MockI2VProvider()
        self.acknowledge_safety_research_use = acknowledge_safety_research_use
        if not isinstance(self.i2v_provider, MockI2VProvider) and not acknowledge_safety_research_use:
            raise SafetyAcknowledgementRequired(
                "Real I2V providers require acknowledge_safety_research_use=True. "
                "Use the mock provider/dry-run for local examples, or explicitly acknowledge "
                "controlled AI safety red-teaming use before calling a real API."
            )
        self._write_safety_notice()

    def run(self, sample: DatasetSample | dict[str, Any]) -> GenerationResult:
        """Run one sample through the VII workflow and append JSONL metadata."""

        sample_obj = self._coerce_sample(sample)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

        reprogrammed = self.reprogrammer.reprogram(sample_obj.prompt, sample_obj.category)
        grounded_path = self.images_dir / f"{sample_obj.sample_id}.png"
        grounded = self.grounder.ground(sample_obj.image_path, reprogrammed, grounded_path)
        requested_video = self.videos_dir / f"{sample_obj.sample_id}.mp4"
        result = self.i2v_provider.generate(
            image_path=grounded.output_path,
            prompt=reprogrammed.visual_instruction,
            output_path=str(requested_video),
            sample_id=sample_obj.sample_id,
        )

        record = {
            "sample": asdict(sample_obj),
            "reprogrammed_intent": asdict(reprogrammed),
            "grounded_image": asdict(grounded),
            "generation_result": asdict(result),
        }
        self._append_metadata(record)
        return result

    def run_many(self, samples: list[DatasetSample] | list[dict[str, Any]]) -> list[GenerationResult]:
        """Run multiple samples with a progress bar."""

        return [self.run(sample) for sample in tqdm(samples, desc="VII pipeline")]

    def _write_safety_notice(self) -> None:
        """Write the responsible-use notice into the experiment output directory."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        notice_path = self.output_dir / "SAFETY_NOTICE.md"
        if not notice_path.exists():
            notice_path.write_text(SAFETY_NOTICE, encoding="utf-8")

    def _append_metadata(self, record: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _coerce_sample(sample: DatasetSample | dict[str, Any]) -> DatasetSample:
        if isinstance(sample, DatasetSample):
            return sample
        required = {"sample_id", "prompt", "category", "image_path"}
        missing = required.difference(sample)
        if missing:
            raise ValueError(f"sample is missing required keys: {sorted(missing)}")
        return DatasetSample(**sample)

    @staticmethod
    def copy_reference_image(image_path: str | Path, output_path: str | Path) -> str:
        """Utility for providers that need a copied reference artifact."""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output)
        return str(output)
