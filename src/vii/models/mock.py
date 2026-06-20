"""Offline image-to-video client for dry-run environments."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vii.models.base import I2VModelClient
from vii.types import GenerationResult


class MockI2VClient(I2VModelClient):
    """Dry-run client that records metadata without calling external APIs."""

    name = "mock"

    def generate(self, image_path: str, prompt: str, output_path: str, **kwargs: Any) -> GenerationResult:
        job_id = str(kwargs.get("job_id") or uuid.uuid4())
        metadata_path = Path(output_path).with_suffix(".metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "job_id": job_id,
            "provider": self.name,
            "image_path": image_path,
            "prompt": prompt,
            "requested_output_path": output_path,
            "kwargs": kwargs,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": True,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return GenerationResult(
            sample_id=str(kwargs.get("sample_id", "unknown")),
            grounded_image_path=image_path,
            video_path=str(metadata_path),
            provider=self.name,
            status="mocked",
            metadata=metadata,
        )

    def check_status(self, job_id: str) -> dict[str, Any]:
        return {"job_id": job_id, "provider": self.name, "status": "succeeded", "dry_run": True}

    def download(self, job_id: str, output_path: str) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"job_id": job_id, "provider": self.name, "dry_run": True}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(output)
