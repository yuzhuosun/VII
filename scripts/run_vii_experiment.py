#!/usr/bin/env python3
"""Run a small VII experiment with safe defaults.

The command defaults to dry-run/mock behavior so examples never call a real
commercial image-to-video API unless the caller explicitly selects that mode and
acknowledges controlled AI safety research use.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vii.models.base import SAFETY_NOTICE_FILENAME, SafetyAcknowledgementRequired
from vii.pipeline import MockI2VProvider, VIIPipeline
from vii.types import GenerationResult


class RealI2VProviderPlaceholder:
    """Placeholder guard for future commercial I2V integrations."""

    name = "real-i2v-api"

    def generate(self, image_path: str, prompt: str, output_path: str, **kwargs: Any) -> GenerationResult:
        raise NotImplementedError(
            "Real commercial I2V API integrations are intentionally not wired into the example runner. "
            "Add an audited provider implementation in controlled research infrastructure only."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VII with safe dry-run/mock defaults.")
    parser.add_argument("--input-jsonl", type=Path, help="Optional JSONL samples with sample_id, prompt, category, image_path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vii_experiment"))
    parser.add_argument(
        "--provider",
        choices=("mock", "real-api"),
        default="mock",
        help="Generation backend. Defaults to mock and does not call commercial APIs.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate configuration and write SAFETY_NOTICE.md without generating videos. Enabled by default.",
    )
    parser.add_argument(
        "--acknowledge-safety-research-use",
        action="store_true",
        help="Required for real-api mode; confirms controlled AI safety red-teaming use only.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Maximum samples to process in mock mode.")
    return parser.parse_args()


def load_samples(path: Path, limit: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


def main() -> int:
    args = parse_args()
    if args.provider == "real-api" and not args.acknowledge_safety_research_use:
        raise SafetyAcknowledgementRequired(
            "--provider real-api requires --acknowledge-safety-research-use. "
            "Use the default --provider mock/--dry-run path for examples."
        )

    provider = MockI2VProvider() if args.provider == "mock" else RealI2VProviderPlaceholder()
    pipeline = VIIPipeline(
        output_dir=args.output_dir,
        i2v_provider=provider,
        acknowledge_safety_research_use=args.acknowledge_safety_research_use,
    )

    if args.dry_run:
        print(f"Dry run complete. Safety notice written to {args.output_dir / SAFETY_NOTICE_FILENAME}")
        return 0

    if args.provider != "mock":
        raise SafetyAcknowledgementRequired("Non-dry-run execution is only enabled for the mock provider in this runner.")
    if not args.input_jsonl:
        raise ValueError("--input-jsonl is required when running mock generation with --no-dry-run.")

    if args.limit < 1:
        raise ValueError("--limit must be a positive integer.")

    results = pipeline.run_many(load_samples(args.input_jsonl, args.limit))
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
