"""Typed data models for the VII research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class DatasetSample:
    """One input example for Visual Instruction Injection experiments."""

    sample_id: str
    prompt: str
    category: str
    image_path: str | Path = ""
    image: Any | None = None
    source_dataset: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReprogrammedIntent:
    """A visually grounded instruction derived from an unsafe text prompt."""

    original_prompt: str
    category: str
    visual_instruction: str
    provider: str = "mock"
    model: str | None = None
    safety_notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GroundedImage:
    """Metadata for an image with an overlaid visual instruction."""

    source_path: str
    output_path: str
    instruction: str
    position: tuple[int, int]
    size: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    """Result returned by an image-to-video backend or an offline mock."""

    sample_id: str
    grounded_image_path: str
    video_path: str | None
    provider: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    """Evaluation output for a generated video."""

    sample_id: str
    passed: bool
    scores: Mapping[str, float] = field(default_factory=dict)
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
