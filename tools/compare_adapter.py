#!/usr/bin/env python3
"""Accept or reject an adapter by comparing it with its base model, question by question.

Both results.json files must come from the same job (same machine, same prompt,
same mode). A regression is a question the base model got right and the adapter
gets wrong. The adapter is accepted only with zero regressions and a higher
intent accuracy than the base model.

    python tools/compare_adapter.py base.json adapter.json --verdict verdict.json

Exit code 0 when accepted, 1 when rejected, 2 when the files are not comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def decide(base: dict, lora: dict) -> dict:
    for key in ("mode", "prompt", "n"):
        if base.get(key) != lora.get(key):
            raise ValueError(f"not comparable: {key} differs ({base.get(key)} vs {lora.get(key)})")
    b = {i["id"]: i for i in base["items"]}
    a = {i["id"]: i for i in lora["items"]}
    if set(b) != set(a):
        raise ValueError("not comparable: different questions")
    right_b = {k for k, i in b.items() if i["got"] == i["expected"]}
    right_a = {k for k, i in a.items() if i["got"] == i["expected"]}
    regressions = sorted(right_b - right_a)
    fixed = sorted(right_a - right_b)
    accepted = not regressions and len(right_a) > len(right_b)
    return {
        "accepted": accepted,
        "base_correct": len(right_b), "adapter_correct": len(right_a), "n": len(b),
        "base_within_timeout": base["within_timeout"], "adapter_within_timeout": lora["within_timeout"],
        "base_median_ms": base["median_ms"], "adapter_median_ms": lora["median_ms"],
        "regressions": [{"id": k, "expected": b[k]["expected"], "adapter": a[k]["got"]} for k in regressions],
        "fixed": [{"id": k, "expected": b[k]["expected"], "base": b[k]["got"]} for k in fixed],
        "reason": ("no regression and higher accuracy" if accepted else
                   f"{len(regressions)} regression(s)" if regressions else
                   "accuracy not higher than the base model"),
        "cpu": lora.get("cpu"), "prompt": lora.get("prompt"), "mode": lora.get("mode"),
    }


def report(v: dict, adapter: str) -> str:
    n = v["n"]
    lines = [
        f"## Adapter `{adapter}` against its base model",
        "",
        f"**{'Accepted' if v['accepted'] else 'Rejected'}:** {v['reason']}. "
        f"Same job, CPU {v['cpu']}, prompt `{v['prompt']}`, mode `{v['mode']}`.",
        "",
        "| | Accuracy (intent) | Within timeout | Median ms |",
        "|---|---:|---:|---:|",
        f"| Base model | {v['base_correct']}/{n} | {v['base_within_timeout']}/{n} | {v['base_median_ms']} |",
        f"| With adapter | {v['adapter_correct']}/{n} | {v['adapter_within_timeout']}/{n} | {v['adapter_median_ms']} |",
    ]
    if v["regressions"]:
        lines += ["", "### Regressions (right before, wrong with the adapter)", "",
                  "| ID | Expected | With adapter |", "|---|---|---|"]
        lines += [f"| {r['id']} | {r['expected']} | {r['adapter']} |" for r in v["regressions"]]
    if v["fixed"]:
        lines += ["", "### Fixed by the adapter", "", "| ID | Expected | Base model |", "|---|---|---|"]
        lines += [f"| {r['id']} | {r['expected']} | {r['base']} |" for r in v["fixed"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Accept or reject an adapter against its base model")
    p.add_argument("base", type=Path)
    p.add_argument("adapter", type=Path)
    p.add_argument("--name", default="adapter")
    p.add_argument("--verdict", type=Path, help="write the decision as JSON")
    args = p.parse_args(argv)
    try:
        v = decide(json.loads(args.base.read_text(encoding="utf-8")),
                   json.loads(args.adapter.read_text(encoding="utf-8")))
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    print(report(v, args.name))
    if args.verdict:
        args.verdict.write_text(json.dumps(v, indent=1) + "\n", encoding="utf-8")
    return 0 if v["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
