"""Video frame extraction and pluggable visual safety classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class FrameSafetyResult:
    frame_path: str
    unsafe: bool
    label: str = "unknown"
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VideoSafetyResult:
    video_path: str
    unsafe: bool
    frames: list[FrameSafetyResult]
    classifier: str = "mock"
    metadata: dict[str, Any] = field(default_factory=dict)


class SafetyClassifier(Protocol):
    name: str

    def classify(self, frame_path: str) -> FrameSafetyResult:
        """Classify a single frame."""


class MockSafetyClassifier:
    """Filename/metadata based classifier for tests and dry runs."""

    name = "mock"

    def classify(self, frame_path: str) -> FrameSafetyResult:
        lowered = Path(frame_path).name.lower()
        unsafe = any(token in lowered for token in ("unsafe", "violence", "weapon", "nudity", "blood"))
        return FrameSafetyResult(frame_path=frame_path, unsafe=unsafe, label="unsafe" if unsafe else "safe", score=1.0 if unsafe else 0.0)


class PluginSafetyClassifier:
    """Adapter for optional classifiers such as NudeNet, Q16, or in-house models."""

    def __init__(self, name: str, plugin: Any):
        self.name = name
        self.plugin = plugin

    def classify(self, frame_path: str) -> FrameSafetyResult:
        result = self.plugin.classify(frame_path)
        if isinstance(result, FrameSafetyResult):
            return result
        if isinstance(result, dict):
            return FrameSafetyResult(
                frame_path=frame_path,
                unsafe=bool(result.get("unsafe", result.get("flagged", False))),
                label=str(result.get("label", "unsafe" if result.get("unsafe") else "safe")),
                score=float(result.get("score", 0.0)),
                metadata={k: v for k, v in result.items() if k not in {"unsafe", "flagged", "label", "score"}},
            )
        raise TypeError("safety classifier plugins must return FrameSafetyResult or dict")


def extract_frames(video_path: str | Path, output_dir: str | Path, *, every_n_seconds: float = 1.0, max_frames: int = 16) -> list[str]:
    """Extract frames with OpenCV. Returns an empty list if the video cannot be read."""

    import cv2

    video = Path(video_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return []
    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    step = max(1, int(fps * every_n_seconds))
    frame_paths: list[str] = []
    index = 0
    while len(frame_paths) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            frame_path = output / f"{video.stem}_frame_{len(frame_paths):04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
        index += 1
    capture.release()
    return frame_paths


def evaluate_video_safety(
    video_path: str | Path,
    *,
    frames_dir: str | Path | None = None,
    classifier: SafetyClassifier | None = None,
    preextracted_frames: list[str] | None = None,
) -> VideoSafetyResult:
    """Run a safety classifier over extracted or supplied frames."""

    classifier = classifier or MockSafetyClassifier()
    frames = preextracted_frames
    if frames is None:
        frames = extract_frames(video_path, frames_dir or Path(video_path).with_suffix(".frames"))
    results = [classifier.classify(frame) for frame in frames]
    return VideoSafetyResult(
        video_path=str(video_path),
        unsafe=any(result.unsafe for result in results),
        frames=results,
        classifier=classifier.name,
        metadata={"num_frames": len(results)},
    )
