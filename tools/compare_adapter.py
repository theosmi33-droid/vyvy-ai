#!/usr/bin/env python3
"""Accept or reject an adapter by comparing it with its baseline, question by question.

Both results.json files must come from the same job (same machine, same prompt,
same mode). The rule is statistical rather than zero tolerance, because on a few
questions per intent a single borderline sentence would decide: see decide().

    python tools/compare_adapter.py base.json adapter.json --verdict verdict.json

Exit code 0 when accepted, 1 when rejected, 2 when the files are not comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


SAFETY_INTENTS = ("unknown",)  # a wrong action instead of "I don't know" is never acceptable


def mcnemar_p(b: int, c: int) -> float:
    """One-sided exact McNemar test: P(X >= b) for X ~ Binomial(b + c, 0.5).

    b = questions the adapter fixed, c = questions it broke. Only these discordant
    pairs carry information about which model is better.
    """
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n


def intent_tolerance(n: int) -> int:
    """Net losses one intent may show: one question, or 5 % of its questions if more."""
    return max(1, math.floor(0.05 * n))


def decide(base: dict, lora: dict, alpha: float = 0.05) -> dict:
    """Accept only a statistically clear improvement that does not hurt any intent noticeably.

    1. exact McNemar p < alpha on the discordant pairs
    2. at least twice as many fixed as broken questions (b >= 2c)
    3. no intent loses more than intent_tolerance(n) questions net
    4. no question of a safety intent ("unknown") gets worse
    """
    for key in ("mode", "prompt", "n"):
        if base.get(key) != lora.get(key):
            raise ValueError(f"not comparable: {key} differs ({base.get(key)} vs {lora.get(key)})")
    bi = {i["id"]: i for i in base["items"]}
    ai = {i["id"]: i for i in lora["items"]}
    if set(bi) != set(ai):
        raise ValueError("not comparable: different questions")
    right_b = {k for k, i in bi.items() if i["got"] == i["expected"]}
    right_a = {k for k, i in ai.items() if i["got"] == i["expected"]}
    broken, fixed = sorted(right_b - right_a), sorted(right_a - right_b)
    b, c = len(fixed), len(broken)
    p = mcnemar_p(b, c)
    per_intent: dict[str, dict] = {}
    for k, item in bi.items():
        row = per_intent.setdefault(item["expected"], {"n": 0, "fixed": 0, "broken": 0})
        row["n"] += 1
        row["fixed"] += k in fixed
        row["broken"] += k in broken
    reasons = []
    if p >= alpha:
        reasons.append(f"not significant (b={b}, c={c}, p={p:.4f} >= {alpha})")
    if b < 2 * c:
        reasons.append(f"net gain too small (b={b} < 2*c={2 * c})")
    for intent, r in sorted(per_intent.items()):
        net = r["fixed"] - r["broken"]
        if intent in SAFETY_INTENTS and r["broken"]:
            reasons.append(f"safety intent {intent} got worse on {r['broken']} question(s)")
        elif net < -intent_tolerance(r["n"]):
            reasons.append(f"{intent} lost {-net} net (tolerance {intent_tolerance(r['n'])} of {r['n']})")
    accepted = not reasons
    return {
        "accepted": accepted,
        "b_fixed": b, "c_broken": c, "p_value": round(p, 6), "alpha": alpha,
        "base_correct": len(right_b), "adapter_correct": len(right_a), "n": len(bi),
        "base_within_timeout": base["within_timeout"], "adapter_within_timeout": lora["within_timeout"],
        "base_median_ms": base["median_ms"], "adapter_median_ms": lora["median_ms"],
        "per_intent": per_intent,
        "regressions": [{"id": k, "expected": bi[k]["expected"], "adapter": ai[k]["got"]} for k in broken],
        "fixed": [{"id": k, "expected": bi[k]["expected"], "base": bi[k]["got"]} for k in fixed],
        "reason": (f"significant improvement (b={b}, c={c}, p={p:.4f}), no intent noticeably worse"
                   if accepted else "; ".join(reasons)),
        "cpu": lora.get("cpu"), "prompt": lora.get("prompt"), "mode": lora.get("mode"),
    }


def report(v: dict, adapter: str, baseline: str = "base model") -> str:
    n = v["n"]
    lines = [
        f"## Adapter `{adapter}` against the baseline ({baseline})",
        "",
        f"**{'Accepted' if v['accepted'] else 'Rejected'}:** {v['reason']}. "
        f"Same job, CPU {v['cpu']}, prompt `{v['prompt']}`, mode `{v['mode']}`.",
        "",
        f"Fixed b = {v['b_fixed']}, broken c = {v['c_broken']}, exact McNemar p = {v['p_value']:.4f}.",
        "",
        "| | Accuracy (intent) | Within timeout | Median ms |",
        "|---|---:|---:|---:|",
        f"| Baseline: {baseline} | {v['base_correct']}/{n} | {v['base_within_timeout']}/{n} | {v['base_median_ms']} |",
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
    p.add_argument("--baseline-name", default="base model")
    p.add_argument("--verdict", type=Path, help="write the decision as JSON")
    args = p.parse_args(argv)
    try:
        v = decide(json.loads(args.base.read_text(encoding="utf-8")),
                   json.loads(args.adapter.read_text(encoding="utf-8")))
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    v["baseline"] = args.baseline_name
    print(report(v, args.name, args.baseline_name))
    if args.verdict:
        args.verdict.write_text(json.dumps(v, indent=1) + "\n", encoding="utf-8")
    return 0 if v["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
