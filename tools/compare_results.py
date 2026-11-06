#!/usr/bin/env python3
"""One Markdown table over several results.json files of the same machine.

    python tools/compare_results.py results-*.json >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def row(path: Path) -> str:
    d = json.loads(path.read_text(encoding="utf-8"))
    t = d.get("timing", {})

    def med(key: str) -> str:
        return f"{t[key]['median']:.0f}" if key in t else "–"

    n = d["n"]
    return (f"| {d.get('label') or path.stem} | {d['mode']} | {n} | {d['intent_correct']}/{n} "
            f"({d['intent_correct'] / n:.1%}) | {d['within_timeout']}/{n} | {d['median_ms']} "
            f"| {med('prompt_ms')} | {med('gen_ms')} |")


def main(paths: list[str]) -> int:
    files = [Path(p) for p in paths]
    cpus = {json.loads(f.read_text(encoding="utf-8")).get("cpu") for f in files}
    lines = ["## Comparison on one machine", "",
             f"CPU: {', '.join(sorted(c or 'unknown' for c in cpus))}"
             + (" · **runs used different CPUs, milliseconds are not comparable**" if len(cpus) > 1 else ""),
             "",
             "| Model | Mode | N | Accuracy (intent) | Within timeout | Median ms | Prompt ms | After prompt ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    lines += [row(f) for f in files]
    print("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
