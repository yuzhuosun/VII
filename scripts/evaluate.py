#!/usr/bin/env python3
"""Evaluate VII experiment metadata and write aggregate metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vii.eval.metrics import compute_metrics
from vii.eval.refusal import detect_refusal
from vii.eval.semantic import get_semantic_evaluator
from vii.eval.vision_safety import MockSafetyClassifier, evaluate_video_safety


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="outputs/metadata.jsonl", help="Pipeline metadata JSONL")
    parser.add_argument("--videos", default="outputs/videos", help="Directory containing generated videos")
    parser.add_argument("--output", default="outputs/eval_results.json", help="Output JSON path")
    parser.add_argument("--semantic-evaluator", choices=["mock", "gpt", "openai"], default="mock")
    parser.add_argument("--skip-vision", action="store_true", help="Skip frame extraction and visual safety classification")
    args = parser.parse_args()

    evaluator = get_semantic_evaluator(args.semantic_evaluator)
    eval_records = []
    for raw in _read_jsonl(Path(args.results)):
        record = _normalize_record(raw, Path(args.videos))
        refusal = detect_refusal(record.get("response"), status_code=record.get("status_code"), error=record.get("error"), moderation=record.get("moderation"))
        semantic = evaluator.verify(record["prompt"], record.get("video_description", ""), record["category"])
        if args.skip_vision or not record.get("video_path"):
            unsafe_generation = False
            vision = None
        else:
            vision_result = evaluate_video_safety(record["video_path"], classifier=MockSafetyClassifier())
            unsafe_generation = vision_result.unsafe
            vision = asdict(vision_result)

        eval_records.append(
            {
                **record,
                "refused": refusal.refused,
                "refusal_reason": refusal.refusal_reason,
                "semantic_achieved": semantic.achieved,
                "semantic_score": semantic.score,
                "semantic_rationale": semantic.rationale,
                "unsafe_generation": unsafe_generation,
                "vision_safety": vision,
            }
        )

    output = {"metrics": compute_metrics(eval_records), "records": eval_records}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_record(raw: dict[str, Any], videos_dir: Path) -> dict[str, Any]:
    sample = raw.get("sample", raw)
    generation = raw.get("generation_result", raw.get("generation", {}))
    reprogrammed = raw.get("reprogrammed_intent", {})
    metadata = generation.get("metadata", {}) if isinstance(generation, dict) else {}
    sample_id = sample.get("sample_id", raw.get("sample_id", generation.get("sample_id", "unknown")))
    video_path = generation.get("video_path") or raw.get("video_path")
    if not video_path:
        candidate = videos_dir / f"{sample_id}.mp4"
        video_path = str(candidate) if candidate.exists() else None
    return {
        "sample_id": sample_id,
        "prompt": sample.get("prompt", raw.get("prompt", "")),
        "category": sample.get("category", raw.get("category", "unknown")),
        "dataset": sample.get("source_dataset", raw.get("dataset", "unknown")),
        "model": generation.get("provider", raw.get("model", raw.get("provider", "unknown"))),
        "video_path": video_path,
        "video_description": raw.get("video_description") or metadata.get("video_description") or reprogrammed.get("visual_instruction", ""),
        "response": raw.get("response", generation),
        "status_code": raw.get("status_code", metadata.get("status_code")),
        "error": raw.get("error", metadata.get("error")),
        "moderation": raw.get("moderation", metadata.get("moderation")),
    }


if __name__ == "__main__":
    main()
