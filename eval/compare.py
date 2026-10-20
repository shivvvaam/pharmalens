"""Print a Markdown table comparing every eval/results_<variant>.json side by side.

Usage: uv run python eval/compare.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EVAL_DIR = Path(__file__).parent
ORDER = ["baseline", "small_chunks", "score_filter", "combined"]


def summarise(path: Path) -> dict[str, tuple[float, int, int]]:
    scores: dict[str, list[float]] = defaultdict(list)
    passed: dict[str, int] = defaultdict(int)
    for case in json.loads(path.read_text(encoding="utf-8")):
        for m in case["metrics"]:
            scores[m["name"]].append(m["score"] or 0.0)
            passed[m["name"]] += bool(m["passed"])
    return {name: (sum(v) / len(v), passed[name], len(v)) for name, v in scores.items()}


def main() -> None:
    files = {p.stem.removeprefix("results_"): p for p in EVAL_DIR.glob("results_*.json")}
    if not files:
        sys.exit("No eval/results_*.json files yet. Run: uv run python evaluate.py --variant baseline")
    variants = sorted(files, key=lambda v: ORDER.index(v) if v in ORDER else len(ORDER))
    summaries = {v: summarise(files[v]) for v in variants}
    metrics = sorted({m for s in summaries.values() for m in s})

    print("| Metric | " + " | ".join(variants) + " |")
    print("|---|" + "---|" * len(variants))
    for metric in metrics:
        cells = []
        for v in variants:
            if metric in summaries[v]:
                mean, ok, n = summaries[v][metric]
                cells.append(f"{mean:.2f} ({ok}/{n} pass)")
            else:
                cells.append("–")
        print(f"| {metric} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
