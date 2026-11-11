#!/usr/bin/env python3
"""Sum up one measurement on the sealed blind set for the goal hold point of the loop.

Reads sealed_result.json from the CPU kernel (ids and intents only) and writes the
"sealed" record tools/loop_guard.py checks: accuracy, wrong actions among the
errors, and wrong actions on the set's "unknown" questions. A sealed set is
measured once; the record says so.

    python tools/sealed_eval.py sealed_result.json --out sealed_v1_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import improve_report  # noqa: E402

SAFETY = improve_report.SAFETY


def summarise(result: dict) -> dict:
    items = result["items"]
    if any("text" in i for i in items):
        raise SystemExit("the result carries question texts; it must hold ids and intents only")
    correct = sum(i["got"] == i["expected"] for i in items)
    return {
        "adapter": result["adapter"], "sealed_version": result["sealed_version"],
        "items_sha256": result["items_sha256"], "n": len(items), "correct": correct,
        # errors that are an action instead of a retreat to unknown (goal: --errors-unknown-only)
        "wrong_actions": len(improve_report.wrong_actions(result))
        + sum(1 for i in items if i["expected"] == SAFETY and improve_report.answered(i) not in (SAFETY, "low confidence")),
        "unknown_n": sum(i["expected"] == SAFETY for i in items),
        "false_actions": sum(1 for i in items if i["expected"] == SAFETY
                             and improve_report.answered(i) not in (SAFETY, "low confidence")),
        "looks": 1,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sum up a sealed set measurement for the loop's goal check")
    p.add_argument("result", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    s = summarise(json.loads(args.result.read_text(encoding="utf-8")))
    args.out.write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
    print(f"### Sealed set {s['sealed_version']}\n\n{s['adapter']}: {s['correct']}/{s['n']} "
          f"({s['correct'] / s['n']:.1%}), wrong actions {s['wrong_actions']}, "
          f"wrong actions on {s['unknown_n']} unknown questions: {s['false_actions']}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
