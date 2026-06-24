"""Configurable image-to-video client for API gateways and compatible providers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from vii.models._http import LoggedHTTPClientMixin, image_data_uri, require_env, sleep_for_poll
from vii.models.base import I2VModelClient
from vii.types import GenerationResult


class GenericI2VClient(LoggedHTTPClientMixin, I2VModelClient):
    """Generic JSON-over-HTTP image-to-video client.

    This client is intended for API gateways that expose image-to-video models
    with a simple JSON payload containing ``model``, ``prompt`` and ``image``.
    It supports both synchronous responses with a video URL and asynchronous
    responses with a job ID that can be polled.
    """

    name = "generic_i2v"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        endpoint_path: str | None = None,
        status_path_template: str | None = None,
    ):
        self.api_key = api_key or require_env("I2V_API_KEY")
        self.base_url = (base_url or os.getenv("I2V_BASE_URL") or "").rstrip("/")
        if not self.base_url:
            raise ValueError("Missing required environment variable: I2V_BASE_URL")
        self.endpoint_path = endpoint_path or os.getenv("I2V_ENDPOINT_PATH") or "/v1/videos/image-to-video"
        self.status_path_template = status_path_template or os.getenv("I2V_STATUS_PATH_TEMPLATE") or "/v1/videos/{job_id}"

    def generate(self, image_path: str, prompt: str, output_path: str, **kwargs: Any) -> GenerationResult:
        model = kwargs.get("model") or os.getenv("I2V_MODEL") or "MiniMax-I2V-01"
        body = {
            "model": model,
            "prompt": prompt,
            "image": image_data_uri(image_path),
            "resolution": kwargs.get("resolution", "720p"),
            "duration": kwargs.get("duration", 5),
        }
        body.update(kwargs.get("extra_body", {}))
        response = self._json_request(self.name, "POST", self._url(self.endpoint_path), self._headers(), body)
        job_id = str(response.get("job_id") or response.get("task_id") or response.get("id") or response.get("request_id") or "")
        status = str(response.get("status", "submitted" if job_id else "succeeded"))
        video_path = None
        direct_url = self._extract_video_url(response)
        if direct_url:
            video_path = self._download_url(direct_url, output_path)
            status = "succeeded"
        elif kwargs.get("wait", False) and job_id:
            video_path = self._wait_and_download(job_id, output_path, kwargs)
            status = "succeeded"
        return GenerationResult(
            sample_id=str(kwargs.get("sample_id", "unknown")),
            grounded_image_path=image_path,
            video_path=video_path,
            provider=self.name,
            status=status,
            metadata={"job_id": job_id, "model": model, "response": response},
        )

    def check_status(self, job_id: str) -> dict[str, Any]:
        path = self.status_path_template.format(job_id=job_id)
        return self._json_request(self.name, "GET", self._url(path), self._headers())

    def download(self, job_id: str, output_path: str) -> str:
        status = self.check_status(job_id)
        url = self._extract_video_url(status)
        if not url:
            raise ValueError(f"Generic I2V job {job_id} has no downloadable video URL")
        return self._download_url(url, output_path)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def _wait_and_download(self, job_id: str, output_path: str, kwargs: dict[str, Any]) -> str:
        for _ in range(int(kwargs.get("max_retries", 60))):
            status = self.check_status(job_id)
            state = str(status.get("status") or status.get("state") or "").lower()
            url = self._extract_video_url(status)
            if url or state in {"succeeded", "completed", "success", "done"}:
                return self._download_url(url, output_path) if url else self.download(job_id, output_path)
            sleep_for_poll(float(kwargs.get("poll_interval", 5)))
        raise TimeoutError(f"Timed out waiting for Generic I2V job {job_id}")

    @staticmethod
    def _extract_video_url(payload: dict[str, Any]) -> str | None:
        candidates = (
            payload.get("video_url"),
            payload.get("videoUrl"),
            payload.get("url"),
            payload.get("output_url"),
            payload.get("outputUrl"),
            payload.get("download_url"),
        )
        for candidate in candidates:
            if candidate:
                return str(candidate)
        data = payload.get("data") if isinstance(payload.get("data"), dict) else None
        output = payload.get("output") if isinstance(payload.get("output"), dict) else None
        for nested in (data, output):
            if nested:
                nested_url = GenericI2VClient._extract_video_url(nested)
                if nested_url:
                    return nested_url
        videos = payload.get("videos")
        if videos is None and isinstance(data, dict):
            videos = data.get("videos")
        if isinstance(videos, list) and videos:
            first = videos[0]
            if isinstance(first, dict):
                return GenericI2VClient._extract_video_url(first)
            return str(first)
        return None

    def _download_url(self, url: str, output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(str(url), output_path)
        self._write_log(self.name, "download", {"url": url, "output_path": output_path})
        return output_path
