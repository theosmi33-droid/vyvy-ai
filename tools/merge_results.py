#!/usr/bin/env python3
"""Join the parts of one benchmark run (runner/bench.py --part i/m) into one results file.

bench-lora measures a large set in parts on parallel runners, baseline and
candidate of a part always on the same runner. The decision needs one results
file per side, as if the whole set had run at once. This joins the parts and
refuses anything that does not belong together:

- every part 1..m exactly once, all with the same m;
- the same data set, contract, grammar, mode, prompt, label and adapter;
- no question in two parts.

Per-question records are kept as they were measured; the totals (n, correct,
within timeout, timeouts, median ms, per intent) are recomputed from them.
Timing is per runner: the median over parts mixes runners and is reported as
such, never compared across runs.

    python tools/merge_results.py results/lora_decision_*.json --out lora_decision.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SAME = ("data", "contract", "grammar_sha256", "mode", "prompt", "label", "adapter", "timeout_s")


def merge(parts: list[dict]) -> dict:
    if not parts:
        raise ValueError("no parts")
    first = parts[0]
    for key in SAME:
        values = {json.dumps(p.get(key)) for p in parts}
        if len(values) > 1:
            raise ValueError(f"parts differ in {key}: {sorted(values)}")
    numbers = sorted(tuple(int(x) for x in str(p.get("part", "1/1")).split("/")) for p in parts)
    m = numbers[0][1]
    if numbers != [(i, m) for i in range(1, m + 1)]:
        raise ValueError(f"parts incomplete or twice: {['/'.join(map(str, n)) for n in numbers]}")
    items = [it for p in sorted(parts, key=lambda p: int(str(p["part"]).split("/")[0])) for it in p["items"]]
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError("a question is in more than one part")
    per_intent: dict[str, dict] = {}
    for it in items:
        row = per_intent.setdefault(it["expected"], {"n": 0, "intent_correct": 0, "within_timeout": 0,
                                                     "timeouts": 0, "ms": []})
        ok = it["got"] == it["expected"]
        row["n"] += 1
        row["intent_correct"] += ok
        row["within_timeout"] += ok and not it["timeout"]
        row["timeouts"] += it["timeout"]
        row["ms"].append(it["ms"])
    out = {k: first.get(k) for k in SAME}
    out.update({
        "n": len(items),
        "intent_correct": sum(r["intent_correct"] for r in per_intent.values()),
        "within_timeout": sum(r["within_timeout"] for r in per_intent.values()),
        "timeouts": sum(r["timeouts"] for r in per_intent.values()),
        "median_ms": round(statistics.median([it["ms"] for it in items])) if items else 0,
        "median_note": f"over {m} parts on {m} runners",
        "part": "1/1", "parts": m,
        "cpu": sorted({str(p.get("cpu")) for p in parts}),
        "per_intent": {k: {"n": r["n"], "intent_correct": r["intent_correct"], "within_timeout": r["within_timeout"],
                           "timeouts": r["timeouts"], "accuracy": r["intent_correct"] / r["n"]}
                       for k, r in sorted(per_intent.items())},
        "items": items,
    })
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Join the parts of one benchmark run")
    p.add_argument("parts", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        merged = merge([json.loads(f.read_text(encoding="utf-8")) for f in args.parts])
    except ValueError as e:
        print(f"{args.out.name}: {e}", file=sys.stderr)
        return 2
    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {merged['n']} questions from {merged['parts']} parts, "
          f"{merged['intent_correct']} correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
