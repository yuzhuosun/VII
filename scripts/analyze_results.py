#!/usr/bin/env python3
"""Create CSV tables and optional PNG plots from ``scripts/evaluate.py`` output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-results", default="outputs/eval_results.json")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    data = json.loads(Path(args.eval_results).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = _metric_tables(data.get("metrics", {}))
    for name, rows in tables.items():
        _write_csv(output_dir / f"{name}.csv", rows)
        _plot(rows, output_dir / f"{name}_asr.png", title=f"{name} ASR")

    records = data.get("records", [])
    if records:
        _write_csv(output_dir / "records.csv", [{"name": str(i), **row} for i, row in enumerate(records)])
    print(f"Wrote analysis artifacts to {output_dir}")


def _metric_tables(metrics: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    overall = metrics.get("overall")
    if overall:
        tables["overall"] = [{"name": "overall", **overall}]
    for key in ("by_model", "by_category", "by_dataset"):
        if metrics.get(key):
            tables[key] = [{"name": name, **values} for name, values in sorted(metrics[key].items())]
    return tables


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()}, key=lambda key: (key != "name", key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows or "asr" not in rows[0]:
        return
    ordered = sorted(rows, key=lambda row: float(row.get("asr", 0.0)), reverse=True)
    labels = [str(row["name"]) for row in ordered]
    values = [float(row.get("asr", 0.0)) for row in ordered]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("ASR")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
