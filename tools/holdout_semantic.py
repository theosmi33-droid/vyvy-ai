#!/usr/bin/env python3
"""The semantic half of tools/holdout_build.py, in the cloud: check the sets, replace from the reserve.

holdout_build.py runs on a computer where no sentence model may run, so the sets it writes
say "semantic filter NOT RUN (local ...)". This runs the check of tools/check_semantic_leak.py
(same model and revision, cosine --max-cosine, default 0.85) on every sentence of the
decision set and the shadow set, in their order, against the training data, every holdout
ever used, the probes, the prompt and teacher examples, and every sentence of both sets
accepted before it. A sentence at or above the threshold is replaced by the first reserve
sentence of the same intent and round that passes the same check (then of the same intent
from another round, which the report names). The sets are rewritten in place with the
result, marked "run (cloud ...)" only if nothing is short; the report holds numbers only.
Exit 0 when both sets are complete, 3 when the reserve ran out for an intent (the sets
stay marked not run), 2 on bad input.

    python tools/holdout_semantic.py data/holdout_decision_c2.json data/holdout_shadow_c2.json \
        data/holdout_reserve_c2.json --report semantic_report.json --run "$GITHUB_RUN_ID"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import holdout_build as hb  # noqa: E402


def check(sets: dict[str, dict], reserve: dict, model, max_cosine: float, known: dict[str, list[str]]) -> dict:
    """sets: {"decision": ..., "shadow": ...} in that order; model: an object with add(kind, texts) and near(text)."""
    for kind, texts in known.items():
        model.add(f"semantic_{kind}", texts)
    pool: dict[tuple[str, str], list[dict]] = {}
    for r in reserve["items"]:
        pool.setdefault((r["expected_intent"], r["origin"]), []).append(r)
    hits: Counter[str] = Counter()
    replaced: Counter[str] = Counter()
    other_round: Counter[str] = Counter()
    short: Counter[str] = Counter()
    used_reserve: set[str] = set()

    def take(intent: str, origin: str) -> dict | None:
        rounds = [origin] + sorted({o for (i, o) in pool if i == intent and o != origin})
        for o in rounds:
            for cand in pool.get((intent, o), []):
                if cand["id"] in used_reserve:
                    continue
                used_reserve.add(cand["id"])
                reason = model.near(cand["text"])
                if reason:
                    hits[f"reserve {reason}"] += 1
                    continue
                if o != origin:
                    other_round[f"{intent}: {o} for {origin}"] += 1
                return cand
        return None

    out = {}
    for name in ("decision", "shadow"):
        items = []
        for item in sets[name]["items"]:
            reason = model.near(item["text"])
            if reason:
                hits[reason] += 1
                new = take(item["expected_intent"], item["origin"])
                if new is None:
                    short[item["expected_intent"]] += 1
                    continue
                replaced[item["expected_intent"]] += 1
                item = {**new, "id": item["id"], "set": name}
            items.append(item)
            model.add("semantic_candidate", [item["text"]])
        out[name] = items
    complete = not short
    return {"items": out, "complete": complete,
            "report": {"max_cosine": max_cosine, "hits": dict(hits), "replaced_per_intent": dict(replaced),
                       "filled_from_other_round": dict(other_round), "short_per_intent": dict(short),
                       "reserve_used": len(used_reserve), "reserve_left": len(reserve["items"]) - len(used_reserve),
                       "complete": complete}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Semantic check of a built holdout pair, with replacement from the reserve")
    p.add_argument("decision", type=Path)
    p.add_argument("shadow", type=Path)
    p.add_argument("reserve", type=Path)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--max-cosine", type=float, default=0.85)
    p.add_argument("--run", default="", help="the run that checked, recorded in the sets")
    args = p.parse_args(argv)
    try:
        sets = {"decision": json.loads(args.decision.read_text(encoding="utf-8")),
                "shadow": json.loads(args.shadow.read_text(encoding="utf-8"))}
        reserve = json.loads(args.reserve.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 2
    import check_semantic_leak as sem

    result = check(sets, reserve, hb.Semantic(args.max_cosine), args.max_cosine, hb.known_sentences())
    mark = (f"run (cloud, run {args.run}, {sem.MODEL.split('/')[-1]} at {sem.REVISION[:12]}, cosine "
            f"{args.max_cosine})" if result["complete"] else sets["decision"].get("semantic_filter", hb.LOCAL_ONLY))
    for name, path in (("decision", args.decision), ("shadow", args.shadow)):
        data = {**sets[name], "semantic_filter": mark, "items": result["items"][name]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps({**result["report"], "semantic_filter": mark}, indent=1) + "\n",
                           encoding="utf-8")
    r = result["report"]
    print(f"semantic check: {sum(r['hits'].values())} hits, {sum(r['replaced_per_intent'].values())} replaced, "
          f"short {r['short_per_intent'] or 'none'}, reserve left {r['reserve_left']}")
    return 0 if result["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
