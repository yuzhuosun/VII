"""Kling image-to-video API client."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from vii.models._http import LoggedHTTPClientMixin, image_data_uri, require_env, sleep_for_poll
from vii.models.base import I2VModelClient
from vii.types import GenerationResult


class KlingI2VClient(LoggedHTTPClientMixin, I2VModelClient):
    """Thin wrapper around a Kling-compatible image-to-video API."""

    name = "kling"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or require_env("KLING_API_KEY")
        self.base_url = (base_url or os.getenv("KLING_BASE_URL") or "https://api.klingai.com").rstrip("/")

    def generate(self, image_path: str, prompt: str, output_path: str, **kwargs: Any) -> GenerationResult:
        body = {
            "model": kwargs.get("model", "kling-v1"),
            "prompt": prompt,
            "image": image_data_uri(image_path),
            "resolution": kwargs.get("resolution", "1280x720"),
            "duration": kwargs.get("duration", 5),
        }
        response = self._json_request(self.name, "POST", f"{self.base_url}/v1/videos/image-to-video", self._headers(), body)
        job_id = str(response.get("job_id") or response.get("id"))
        status = str(response.get("status", "submitted"))
        video_path = None
        if kwargs.get("wait", False):
            video_path = self._wait_and_download(job_id, output_path, kwargs)
            status = "succeeded"
        return GenerationResult(str(kwargs.get("sample_id", "unknown")), image_path, video_path, self.name, status, {"job_id": job_id, "response": response})

    def check_status(self, job_id: str) -> dict[str, Any]:
        return self._json_request(self.name, "GET", f"{self.base_url}/v1/videos/{job_id}", self._headers())

    def download(self, job_id: str, output_path: str) -> str:
        status = self.check_status(job_id)
        url = status.get("video_url") or status.get("url") or status.get("output_url")
        if not url:
            raise ValueError(f"Kling job {job_id} has no downloadable video URL")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(str(url), output_path)
        self._write_log(self.name, "download", {"job_id": job_id, "output_path": output_path, "status": status})
        return output_path

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _wait_and_download(self, job_id: str, output_path: str, kwargs: dict[str, Any]) -> str:
        for _ in range(int(kwargs.get("max_retries", 60))):
            status = self.check_status(job_id)
            if status.get("status") in {"succeeded", "completed", "success"}:
                return self.download(job_id, output_path)
            sleep_for_poll(float(kwargs.get("poll_interval", 5)))
        raise TimeoutError(f"Timed out waiting for Kling job {job_id}")
