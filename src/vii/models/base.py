"""Common abstractions for image-to-video model clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vii.types import GenerationResult


class I2VModelClient(ABC):
    """Abstract image-to-video generation client interface."""

    name: str

    @abstractmethod
    def generate(self, image_path: str, prompt: str, output_path: str, **kwargs: Any) -> GenerationResult:
        """Start or perform image-to-video generation."""

    @abstractmethod
    def check_status(self, job_id: str) -> dict[str, Any]:
        """Return provider-specific status for a generation job."""

    @abstractmethod
    def download(self, job_id: str, output_path: str) -> str:
        """Download a completed job artifact and return the local path."""
