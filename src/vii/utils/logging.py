"""Logging and run provenance helpers for VII experiments."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from .io import atomic_write_json, atomic_write_text


def get_logger(name: str = "vii", *, level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Return a consistently configured VII logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if not any(getattr(handler, "_vii_console", False) for handler in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console._vii_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(getattr(handler, "_vii_log_file", None) == str(log_path) for handler in logger.handlers):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler._vii_log_file = str(log_path)  # type: ignore[attr-defined]
            logger.addHandler(file_handler)
    return logger


def save_run_provenance(run_dir: str | Path, run_config: Mapping[str, Any] | None = None) -> None:
    """Save run configuration, git commit, and Python environment metadata."""

    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination / "run_config.yaml", yaml.safe_dump(dict(run_config or {}), sort_keys=True, allow_unicode=True))
    atomic_write_text(destination / "git_commit.txt", _git_commit_text())
    atomic_write_json(destination / "environment.json", collect_environment())


def collect_environment() -> dict[str, Any]:
    """Collect lightweight, JSON-serializable environment metadata."""

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "environment_variables": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_DATASETS_CACHE", "PYTHONPATH")
            if os.environ.get(key) is not None
        },
        "packages": _pip_freeze(),
    }


def _git_commit_text() -> str:
    commands = [
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--short"],
    ]
    parts: list[str] = []
    for command in commands:
        try:
            result = subprocess.run(command, check=False, text=True, capture_output=True)
        except OSError as exc:
            parts.append(f"$ {' '.join(command)}\nERROR: {exc}")
            continue
        output = result.stdout.strip() or result.stderr.strip()
        parts.append(f"$ {' '.join(command)}\n{output}\nreturncode={result.returncode}")
    return "\n\n".join(parts) + "\n"


def _pip_freeze() -> list[str]:
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False, text=True, capture_output=True)
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return sorted(line for line in result.stdout.splitlines() if line.strip())
