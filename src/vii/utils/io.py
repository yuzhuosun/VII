"""Result artifact and JSONL I/O helpers for VII experiment runs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


RUN_SUBDIRS = ("grounded_images", "videos", "api_logs")


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles common pipeline objects."""

    def default(self, obj: Any) -> Any:  # noqa: D401
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for metadata records."""

    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write text by replacing the destination only after data is flushed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, data: Any, *, indent: int | None = 2) -> None:
    """Atomically write JSON with UTF-8 encoding."""

    atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=indent, cls=EnhancedJSONEncoder) + "\n",
    )


def read_jsonl(path: str | Path, *, skip_blank: bool = True) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""

    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if skip_blank and not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {jsonl_path}:{line_number}: {exc}") from exc


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomically replace a JSONL file with the provided records."""

    lines = [json.dumps(record, ensure_ascii=False, cls=EnhancedJSONEncoder) for record in records]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSONL record using an atomic read-modify-replace cycle."""

    jsonl_path = Path(path)
    records = list(read_jsonl(jsonl_path)) if jsonl_path.exists() else []
    records.append(record)
    write_jsonl(jsonl_path, records)


class RunPaths:
    """Path manager for the standard ``outputs/{run_name}`` layout."""

    def __init__(self, run_name: str, root: str | Path = "outputs"):
        if not run_name or not str(run_name).strip():
            raise ValueError("run_name must be a non-empty string")
        self.run_name = str(run_name)
        self.root = Path(root)
        self.run_dir = self.root / self.run_name
        self.grounded_images_dir = self.run_dir / "grounded_images"
        self.videos_dir = self.run_dir / "videos"
        self.metadata_path = self.run_dir / "metadata.jsonl"
        self.api_logs_dir = self.run_dir / "api_logs"
        self.eval_results_path = self.run_dir / "eval_results.json"
        self.summary_csv_path = self.run_dir / "summary.csv"

    def create(self) -> "RunPaths":
        """Create the run directory and standard subdirectories."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        for subdir in RUN_SUBDIRS:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)
        return self

    def grounded_image_path(self, sample_id: str, suffix: str = ".png") -> Path:
        return self.grounded_images_dir / f"{safe_filename(sample_id)}{suffix}"

    def video_path(self, sample_id: str, suffix: str = ".mp4") -> Path:
        return self.videos_dir / f"{safe_filename(sample_id)}{suffix}"

    def api_log_path(self, sample_id: str, suffix: str = ".json") -> Path:
        return self.api_logs_dir / f"{safe_filename(sample_id)}{suffix}"


def safe_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""

    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    return safe.strip("._") or "sample"
