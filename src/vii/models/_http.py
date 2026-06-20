"""Small HTTP and logging helpers shared by model clients."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class APIRequestError(RuntimeError):
    """Raised when an upstream model API request fails."""


class LoggedHTTPClientMixin:
    """Mixin providing JSON requests and request/response audit logs."""

    log_root = Path("outputs/api_logs")
    timeout = 120

    def _write_log(self, provider: str, action: str, payload: dict[str, Any]) -> None:
        log_dir = self.log_root / provider
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (log_dir / f"{stamp}_{action}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _json_request(
        self,
        provider: str,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_headers = {k: ("***" if "key" in k.lower() or "authorization" in k.lower() else v) for k, v in (headers or {}).items()}
        self._write_log(provider, "request", {"method": method, "url": url, "headers": safe_headers, "body": body})
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw) if raw else {}
                self._write_log(provider, "response", {"status": response.status, "body": parsed})
                return parsed
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            self._write_log(provider, "response_error", {"status": exc.code, "body": raw})
            raise APIRequestError(f"{provider} API request failed with HTTP {exc.code}: {raw}") from exc


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def image_data_uri(image_path: str) -> str:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def sleep_for_poll(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)
