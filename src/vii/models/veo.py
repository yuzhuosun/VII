"""Gemini Veo image-to-video client."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from vii.models._http import LoggedHTTPClientMixin, image_data_uri, require_env, sleep_for_poll
from vii.models.base import I2VModelClient
from vii.types import GenerationResult


class VeoI2VClient(LoggedHTTPClientMixin, I2VModelClient):
    """Wrapper for Gemini/Vertex Veo image-to-video generation."""

    name = "veo"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, use_vertex: bool | None = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.use_vertex = bool(use_vertex if use_vertex is not None else os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))
        self.base_url = (base_url or os.getenv("VEO_BASE_URL") or self._default_base_url()).rstrip("/")
        if not self.api_key and not self.use_vertex:
            require_env("GOOGLE_API_KEY")

    def generate(self, image_path: str, prompt: str, output_path: str, **kwargs: Any) -> GenerationResult:
        model = kwargs.get("model", "veo-2.0-generate-001")
        body = {
            "instances": [{"prompt": prompt, "image": image_data_uri(image_path)}],
            "parameters": {"durationSeconds": kwargs.get("duration", 5), "resolution": kwargs.get("resolution", "720p")},
        }
        response = self._json_request(self.name, "POST", f"{self.base_url}/models/{model}:predictLongRunning", self._headers(), body)
        job_id = str(response.get("name") or response.get("job_id") or response.get("id"))
        status = str(response.get("status", "submitted"))
        video_path = None
        if kwargs.get("wait", False):
            video_path = self._wait_and_download(job_id, output_path, kwargs)
            status = "succeeded"
        return GenerationResult(str(kwargs.get("sample_id", "unknown")), image_path, video_path, self.name, status, {"job_id": job_id, "response": response, "vertex": self.use_vertex})

    def check_status(self, job_id: str) -> dict[str, Any]:
        if job_id.startswith("http"):
            url = job_id
        else:
            url = f"{self.base_url}/{job_id.lstrip('/')}"
        return self._json_request(self.name, "GET", url, self._headers())

    def download(self, job_id: str, output_path: str) -> str:
        status = self.check_status(job_id)
        url = status.get("video_url") or status.get("url") or status.get("outputUri")
        if not url:
            predictions = status.get("response", {}).get("predictions", []) if isinstance(status.get("response"), dict) else []
            url = predictions[0].get("videoUri") if predictions else None
        if not url:
            raise ValueError(f"Veo job {job_id} has no downloadable video URL")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(str(url), output_path)
        self._write_log(self.name, "download", {"job_id": job_id, "output_path": output_path, "status": status})
        return output_path

    def _default_base_url(self) -> str:
        if self.use_vertex:
            project = require_env("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google"
        return "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and not self.use_vertex:
            headers["x-goog-api-key"] = self.api_key
        token = os.getenv("GOOGLE_OAUTH_ACCESS_TOKEN")
        if self.use_vertex and token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _wait_and_download(self, job_id: str, output_path: str, kwargs: dict[str, Any]) -> str:
        for _ in range(int(kwargs.get("max_retries", 90))):
            status = self.check_status(job_id)
            done = status.get("done") is True or status.get("status") in {"succeeded", "completed", "success"}
            if done:
                return self.download(job_id, output_path)
            sleep_for_poll(float(kwargs.get("poll_interval", 10)))
        raise TimeoutError(f"Timed out waiting for Veo job {job_id}")
