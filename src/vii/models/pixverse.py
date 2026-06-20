"""PixVerse image-to-video API client."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from vii.models._http import LoggedHTTPClientMixin, image_data_uri, require_env
from vii.models.base import I2VModelClient
from vii.types import GenerationResult


class PixVerseI2VClient(LoggedHTTPClientMixin, I2VModelClient):
    """Thin wrapper around a PixVerse-compatible image-to-video API."""

    name = "pixverse"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or require_env("PIXVERSE_API_KEY")
        self.base_url = (base_url or os.getenv("PIXVERSE_BASE_URL") or "https://app-api.pixverse.ai").rstrip("/")

    def generate(self, image_path: str, prompt: str, output_path: str, **kwargs: Any) -> GenerationResult:
        response = self._json_request(self.name, "POST", f"{self.base_url}/v1/videos/image-to-video", self._headers(), {
            "prompt": prompt, "image": image_data_uri(image_path), "quality": kwargs.get("resolution", "720p"), "duration": kwargs.get("duration", 5)
        })
        job_id = str(response.get("job_id") or response.get("id"))
        return GenerationResult(str(kwargs.get("sample_id", "unknown")), image_path, None, self.name, str(response.get("status", "submitted")), {"job_id": job_id, "response": response})

    def check_status(self, job_id: str) -> dict[str, Any]:
        return self._json_request(self.name, "GET", f"{self.base_url}/v1/videos/{job_id}", self._headers())

    def download(self, job_id: str, output_path: str) -> str:
        status = self.check_status(job_id)
        url = status.get("video_url") or status.get("url") or status.get("output_url")
        if not url:
            raise ValueError(f"PixVerse job {job_id} has no downloadable video URL")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(str(url), output_path)
        self._write_log(self.name, "download", {"job_id": job_id, "output_path": output_path, "status": status})
        return output_path

    def _headers(self) -> dict[str, str]:
        return {"API-KEY": self.api_key, "Content-Type": "application/json"}
