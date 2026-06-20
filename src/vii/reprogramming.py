"""Malicious Intent Reprogramming utilities.

The implementation is intentionally provider-agnostic.  The default mock mode is
fully offline and produces a static visual instruction suitable for testing the
pipeline without API keys.
"""

from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass
from typing import Protocol

from .types import ReprogrammedIntent


class ReprogrammingProvider(Protocol):
    """Provider interface for prompt-to-visual-instruction conversion."""

    name: str

    def generate(self, prompt: str, category: str) -> str:
        """Return a visual instruction string for rendering."""


@dataclass(slots=True)
class MockReprogrammingProvider:
    """Deterministic offline provider used in tests and local development."""

    name: str = "mock"
    max_chars: int = 220

    def generate(self, prompt: str, category: str) -> str:
        digest = hashlib.sha256(f"{category}:{prompt}".encode("utf-8")).hexdigest()[:10]
        compact_prompt = " ".join(prompt.split())[: self.max_chars]
        return textwrap.dedent(
            f"""
            Scene directive [{category} #{digest}]:
            Preserve the source image context while visually implying the hidden
            intent of this scenario: {compact_prompt}
            """
        ).strip()


class IntentReprogrammer:
    """Convert unsafe text prompts into renderable visual instructions.

    Parameters
    ----------
    provider:
        Either ``"mock"``/``None`` for deterministic offline operation or an
        object implementing :class:`ReprogrammingProvider`.
    model:
        Optional provider model identifier stored in metadata.
    """

    def __init__(self, provider: str | ReprogrammingProvider | None = None, model: str | None = None):
        if provider is None or provider == "mock":
            self.provider: ReprogrammingProvider = MockReprogrammingProvider()
        elif isinstance(provider, str):
            raise ValueError(
                f"Unsupported provider '{provider}'. Pass a provider object or use 'mock' for offline mode."
            )
        else:
            self.provider = provider
        self.model = model

    def reprogram(self, prompt: str, category: str) -> ReprogrammedIntent:
        """Reprogram ``prompt`` into a visual instruction.

        The mock implementation keeps enough semantic signal for red-team
        pipeline validation while avoiding network calls.
        """

        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not category or not category.strip():
            raise ValueError("category must be a non-empty string")

        instruction = self.provider.generate(prompt=prompt, category=category)
        return ReprogrammedIntent(
            original_prompt=prompt,
            category=category,
            visual_instruction=instruction,
            provider=self.provider.name,
            model=self.model,
            safety_notes="Generated for controlled AI-safety/red-team evaluation workflows.",
        )
