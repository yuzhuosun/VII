"""Result artifact and JSONL I/O helpers for VII experiment runs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
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


def dumps_json(data: Any, *, indent: int | None = None) -> str:
    """Serialize JSON consistently across result-management helpers."""

    return json.dumps(data, ensure_ascii=False, indent=indent, cls=EnhancedJSONEncoder)


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write text by replacing the destination only after data is flushed.

    The previous file remains intact until ``os.replace`` succeeds, so a crash or
    interruption during serialization does not leave a partially-written result
    file at the destination path.
    """

    data = text.encode(encoding)
    atomic_write_bytes(path, data)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Atomically write binary data to ``path``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
        _fsync_directory(destination.parent)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, data: Any, *, indent: int | None = 2) -> None:
    """Atomically write JSON with UTF-8 encoding."""

    atomic_write_text(path, dumps_json(data, indent=indent) + "\n")


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

    payload = "".join(f"{dumps_json(record)}\n" for record in records)
    atomic_write_text(path, payload)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSONL record while preserving the last valid file on failure.

    This intentionally rewrites the complete JSONL file through a temporary file
    and atomic rename instead of writing directly to the destination. It is aimed
    at experiment metadata durability in single-run workflows: if the process is
    interrupted mid-write, the existing ``metadata.jsonl`` is not truncated or
    left with a partial final record.
    """

    jsonl_path = Path(path)
    records = list(read_jsonl(jsonl_path)) if jsonl_path.exists() else []
    records.append(record)
    write_jsonl(jsonl_path, records)


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Path manager for the standard ``outputs/{run_name}`` layout."""

    run_name: str
    root: str | Path = "outputs"

    def __post_init__(self) -> None:
        if not str(self.run_name).strip():
            raise ValueError("run_name must be a non-empty string")
        object.__setattr__(self, "root", Path(self.root))

    @property
    def run_dir(self) -> Path:
        return Path(self.root) / self.run_name

    @property
    def grounded_images_dir(self) -> Path:
        return self.run_dir / "grounded_images"

    @property
    def videos_dir(self) -> Path:
        return self.run_dir / "videos"

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.jsonl"

    @property
    def api_logs_dir(self) -> Path:
        return self.run_dir / "api_logs"

    @property
    def eval_results_path(self) -> Path:
        return self.run_dir / "eval_results.json"

    @property
    def summary_csv_path(self) -> Path:
        return self.run_dir / "summary.csv"

    def create(self) -> "RunPaths":
        """Create the run directory and standard subdirectories."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        for subdir in RUN_SUBDIRS:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)
        return self

    def grounded_image_path(self, sample_id: str, suffix: str = ".png") -> Path:
        """Return the grounded image path for a sample."""

        return self.grounded_images_dir / f"{safe_filename(sample_id)}{suffix}"

    def video_path(self, sample_id: str, suffix: str = ".mp4") -> Path:
        """Return the generated video path for a sample."""

        return self.videos_dir / f"{safe_filename(sample_id)}{suffix}"

    def api_log_path(self, sample_id: str, suffix: str = ".json") -> Path:
        """Return the API log path for a sample."""

        return self.api_logs_dir / f"{safe_filename(sample_id)}{suffix}"


def safe_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""

    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    return safe.strip("._") or "sample"


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync so atomic renames survive power loss."""

    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
