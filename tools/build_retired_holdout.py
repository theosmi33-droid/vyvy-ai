#!/usr/bin/env python3
"""Release the retired blind holdout v2 into the training data.

Blind v2 decided 8 adapters until 11.11.2026 and was then rotated out (issue 8:
"v2 may go into the training data"); since 20.11.2026 it is no longer measured or
reported. Its sentences are human-written office questions, the most realistic
data the project has, so they go into training - except where they come close to
a holdout that still measures (tools/check_holdout_leak.GUARDED): a sentence at or
above --max-similarity (0.8, stricter than the lock's 0.85) to a v3, v4, v1 or
development sentence is left out and listed. The semantic check (Data check
workflow) may drop more; those are left out too, never reworded.

Training rows carry the expected intent; entity stays null. The timeframe follows
the intent (today/tomorrow for the read intents); for create_appointment and
create_task it is read from the time words of the sentence, the last one wins
("heute - nee, morgen" is tomorrow), dialect forms included.

    python tools/build_retired_holdout.py
    python tools/build_retired_holdout.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402
import check_holdout_leak as lock  # noqa: E402

SOURCE = lock.BLIND_V2
OUT = ROOT / "data" / "train" / "vyvy_retired_blind_v2.jsonl"
FIXED = {"appointments_today": "today", "appointments_tomorrow": "tomorrow", "daily_agenda": "today"}
TIME_WORDS = [
    ("tomorrow", r"\b(morgen|morgn|morn|morge|morga|morsche|morschen|morje|morjen|moorn)\b"),
    ("today", r"\b(heute|heut|hüt|hüüt|hüte|heit|heid|hück|hücks|hut)\b"),
    ("week", r"\b(woche|wuche|woch)\b"),
]


def timeframe(text: str, intent: str) -> str | None:
    if intent in FIXED:
        return FIXED[intent]
    if intent not in ("create_appointment", "create_task"):
        return None
    low = text.casefold()
    hits = [(m.start(), name) for name, pattern in TIME_WORDS for m in re.finditer(pattern, low)]
    return max(hits)[1] if hits else None


def build(max_similarity: float) -> tuple[list[dict], list[tuple[str, str, str]]]:
    items = json.loads(SOURCE.read_text(encoding="utf-8"))["items"]
    guarded = [i for h in lock.GUARDED if h.exists() for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    near = {}
    for leak in lock.find_leaks([(i["id"], i["text"]) for i in items], guarded, max_similarity):
        near.setdefault(leak["train"], leak["holdout"])
    rows, dropped = [], []
    for item in items:
        if item["id"] in near:
            dropped.append((item["id"], near[item["id"]], item["text"]))
            continue
        rows.append({"text": item["text"], "intent": item["expected_intent"],
                     "timeframe": timeframe(item["text"], item["expected_intent"]), "entity": None})
    return rows, dropped


def render(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Release the retired blind holdout v2 into the training data")
    p.add_argument("--max-similarity", type=float, default=0.8)
    p.add_argument("--check", action="store_true", help="only compare with the committed file")
    args = p.parse_args(argv)
    intents = bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))
    rows, dropped = build(args.max_similarity)
    for r in rows:
        answer = json.dumps({**{k: r[k] for k in ("intent", "timeframe", "entity")}, "confidence": 0.9})
        if bench.parse_output(answer, intents) is None:
            raise SystemExit(f"not in the output contract: {r}")
    text = render(rows)
    if args.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print(f"{OUT.relative_to(ROOT)} {'matches' if same else 'differs from'} the generator")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["intent"]] = counts.get(r["intent"], 0) + 1
    print(f"{len(rows)} sentences -> {OUT.relative_to(ROOT)}")
    print(json.dumps(counts, indent=1, sort_keys=True))
    print(f"left out {len(dropped)} (near a guarded holdout at or above {args.max_similarity})")
    for sid, hid, t in dropped:
        print(f"  {sid} ~ {hid}: {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
