#!/usr/bin/env python3
"""What the next round should practise, from the bench results of rejected adapters.

Reads pairs of bench results (baseline, candidate) from one job each, lists the
fixed (b) and broken (c) questions by id and intent, and sums up the confusions
(expected -> answered) of every wrong answer. From those counts it proposes the
intents to double and the near-miss pairs to ask the teacher for next round.

It never reads or prints a question text: the blind holdouts decide acceptance
and must not leak into training, so only ids, intents and counts leave this
script (the same rule as WEAK in training/synthesize.py).

    python tools/improve_report.py RUN_DIR [RUN_DIR ...] --json report.json > report.md

RUN_DIR holds base_blind.json and lora_blind.json as attached to a release.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_adapter as ca  # noqa: E402

SAFETY = ca.SAFETY_INTENTS[0]
READS = {"appointments_today", "appointments_tomorrow", "tasks_open", "tasks_overdue",
         "emails_unread", "daily_agenda"}
WRITES = {"create_task", "create_appointment"}
# Confusion kinds, in the order the report lists them. The first is the safety edge.
KINDS = {
    "false_action": "unknown answered with an action",
    "refused": "a request answered with unknown",
    "low_confidence": "below the confidence threshold",
    "day": "today and tomorrow mixed up",
    "overview": "day overview and a single area mixed up",
    "tasks": "open and overdue tasks mixed up",
    "write": "create a task and an appointment mixed up",
    "read_write": "a read answered with a write or the other way round",
    "other": "other confusions",
}
DAY = {"appointments_today", "appointments_tomorrow"}
TASKS = {"tasks_open", "tasks_overdue"}
# A near-miss pair is proposed when it occurs in at least half of the runs (and in two):
# one run alone is noise. Only the WEAK_TOP intents with the most wrong answers are
# doubled, so the round stays inside the kernel limit (see WEAK in training/synthesize.py).
WEAK_TOP = 3


def answered(item: dict) -> str:
    """The intent the model gave; low confidence counts as its own answer."""
    got = item["got"]
    return "low confidence" if got.startswith("low confidence") else got


def kind(expected: str, got: str) -> str:
    if expected == SAFETY:
        return "false_action"
    if got == SAFETY:
        return "refused"
    if got == "low confidence":
        return "low_confidence"
    pair = {expected, got}
    if pair <= DAY:
        return "day"
    if "daily_agenda" in pair and pair <= READS:
        return "overview"
    if pair <= TASKS:
        return "tasks"
    if pair <= WRITES:
        return "write"
    if (expected in READS and got in WRITES) or (expected in WRITES and got in READS):
        return "read_write"
    return "other"


def wrong_actions(result: dict) -> list[str]:
    """Ids of requests answered with another action: neither right nor a retreat (unknown, low
    confidence). Questions that expect unknown are counted apart as false_actions."""
    return sorted(i["id"] for i in result["items"] if i["expected"] != SAFETY
                  and answered(i) not in (i["expected"], SAFETY, "low confidence"))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_report(base: dict, lora: dict) -> dict:
    v = ca.decide(base, lora)
    wrong = [i for i in lora["items"] if i["got"] != i["expected"]]
    confusions = collections.Counter((i["expected"], answered(i)) for i in wrong)
    return {
        "adapter": lora.get("adapter"),
        "baseline": base.get("adapter") or "base model",
        "n": v["n"], "base_correct": v["base_correct"], "adapter_correct": v["adapter_correct"],
        "b_fixed": v["b_fixed"], "c_broken": v["c_broken"], "p_value": round(v["p_value"], 6),
        "accepted": v["accepted"],
        "fixed": [{"id": r["id"], "expected": r["expected"]} for r in v["fixed"]],
        "broken": [{"id": r["id"], "expected": r["expected"], "got": answered({"got": r["adapter"]})}
                   for r in v["regressions"]],
        "kinds": dict(collections.Counter(kind(i["expected"], answered(i)) for i in wrong)),
        "confusions": [{"expected": e, "got": g, "count": k} for (e, g), k in confusions.most_common()],
        "false_actions": sum(k for (e, _), k in confusions.items() if e == SAFETY),
        # Side condition proposed with the acceptance rule: wrong actions must not rise
        # against the baseline. The goal (tools/loop_guard.py --errors-unknown-only) is zero.
        "wrong_actions": len(wrong_actions(lora)), "base_wrong_actions": len(wrong_actions(base)),
        "wrong_actions_not_up": len(wrong_actions(lora)) <= len(wrong_actions(base)),
        "wrong_ids": sorted(i["id"] for i in wrong),
    }


def propose(runs: list[dict]) -> dict:
    """Intents to double and near-miss pairs for the next teacher round.

    A pair (meant, mistaken for) is proposed when it occurs in at least half of
    the runs; the WEAK_TOP intents with the most wrong answers over all runs are
    doubled. Ids that are wrong in most runs are listed as stubborn (ids only),
    for the person who writes the next fresh holdout, not for training.
    """
    pair_runs: collections.Counter = collections.Counter()
    intent_wrong: collections.Counter = collections.Counter()
    id_runs: collections.Counter = collections.Counter()
    for r in runs:
        pair_runs.update({(c["expected"], c["got"]) for c in r["confusions"] if c["got"] != "low confidence"})
        for c in r["confusions"]:
            intent_wrong[c["expected"]] += c["count"]
        id_runs.update(r["wrong_ids"])
    need = max(2, -(-len(runs) // 2))
    near = sorted(([e, g, k] for (e, g), k in pair_runs.items() if k >= need), key=lambda x: (-x[2], x[0], x[1]))
    return {
        "runs": len(runs),
        "weak": sorted(i for i, _ in intent_wrong.most_common(WEAK_TOP)),
        "near": [{"meant": e, "mistaken_for": g, "runs": k} for e, g, k in near],
        "stubborn_ids": sorted(i for i, k in id_runs.items() if k * 2 > len(runs)),
    }


def markdown(runs: list[dict], plan: dict) -> str:
    out = ["## Improvement report", "",
           "| Adapter | Baseline | right | b | c | p | wrong actions on unknown | action instead of another (baseline) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        out.append(f"| {r['adapter']} | {r['baseline']} | {r['adapter_correct']}/{r['n']} | {r['b_fixed']} | "
                   f"{r['c_broken']} | {r['p_value']:.4f} | {r['false_actions']} | {r['wrong_actions']} "
                   f"({r['base_wrong_actions']}{'' if r['wrong_actions_not_up'] else ', more'}) |")
    out += ["", "### Wrong answers by kind", "", "| Kind | " + " | ".join(r["adapter"] or "?" for r in runs) + " |",
            "|---|" + "---|" * len(runs)]
    for k, label in KINDS.items():
        if any(r["kinds"].get(k) for r in runs):
            out.append(f"| {label} | " + " | ".join(str(r["kinds"].get(k, 0)) for r in runs) + " |")
    for r in runs:
        out += ["", f"### {r['adapter']} against {r['baseline']}", "",
                "b: " + (", ".join(f"{x['id']} ({x['expected']})" for x in r["fixed"]) or "none"),
                "", "c: " + (", ".join(f"{x['id']} ({x['expected']} -> {x['got']})" for x in r["broken"]) or "none")]
    out += ["", "### Proposal for the next teacher round", "",
            f"Double: {', '.join(plan['weak']) or 'nothing'}.", "",
            "Near misses (meant -> mistaken for, in how many runs):", ""]
    out += [f"- {p['meant']} -> {p['mistaken_for']}: {p['runs']}" for p in plan["near"]] or ["- none"]
    out += ["", f"Wrong in most runs (ids only, not for training): {', '.join(plan['stubborn_ids']) or 'none'}", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sum up what rejected adapters got wrong, by ids and intents only")
    p.add_argument("runs", type=Path, nargs="+", help="directories with base_blind.json and lora_blind.json")
    p.add_argument("--set", default="blind", help="result files to read: base_<set>.json, lora_<set>.json")
    p.add_argument("--json", type=Path, help="also write the report as JSON here")
    args = p.parse_args(argv)
    runs = [run_report(load(d / f"base_{args.set}.json"), load(d / f"lora_{args.set}.json")) for d in args.runs]
    plan = propose(runs)
    if args.json:
        args.json.write_text(json.dumps({"runs": runs, "plan": plan}, indent=1, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(markdown(runs, plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
