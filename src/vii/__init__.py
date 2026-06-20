"""VII: Visual Instruction Injection research utilities."""

from __future__ import annotations

from .models.base import SafetyAcknowledgementRequired
from .types import DatasetSample, EvaluationResult, GenerationResult, GroundedImage, ReprogrammedIntent

__all__ = [
    "DatasetSample",
    "EvaluationResult",
    "GenerationResult",
    "GroundedImage",
    "GroundingConfig",
    "IntentReprogrammer",
    "ReprogrammedIntent",
    "SafetyAcknowledgementRequired",
    "VIIPipeline",
    "VisualInstructionGrounder",
]


def __getattr__(name: str):
    """Lazily import optional pipeline helpers to keep data utilities lightweight."""

    if name in {"GroundingConfig", "VisualInstructionGrounder"}:
        from .grounding import GroundingConfig, VisualInstructionGrounder

        return {"GroundingConfig": GroundingConfig, "VisualInstructionGrounder": VisualInstructionGrounder}[name]
    if name == "VIIPipeline":
        from .pipeline import VIIPipeline

        return VIIPipeline
    if name == "IntentReprogrammer":
        from .reprogramming import IntentReprogrammer

        return IntentReprogrammer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
