"""Scoring of the slots a create request carries: title, when, person (draft, not wired in yet).

Kept apart from bench.py on purpose: the intent score there stays exactly as it is,
and this is a second, separate figure. Nothing here runs a model.

The contract the scoring enforces is the one people rely on: a slot is either a
piece of the sentence the person typed, or empty. A slot whose words are not in
the sentence is "invented" and always counts as an error, whatever the reference
says; the application applies the same check (is_span_of) before it shows a draft,
and leaves such a slot empty. Dates and times are never computed here: "when" is
the time expression as written, the application turns it into a date.

Per slot and sentence one of:
- right:   filled and matching the reference (title: token F1 >= TITLE_F1,
           when and person: equal after normalisation)
- wrong:   filled, a piece of the sentence, but not matching the reference
- invented: filled with words that are not in the sentence
- missed:  empty although the reference has a value
- filled:  filled although the reference is empty (the sentence does not say it)
- empty:   empty and the reference is empty too (right)
"""

from __future__ import annotations

import re

SLOTS = ("title", "when", "person")
TITLE_F1 = 0.8


def normalise(text: str) -> str:
    """Same normalisation as the holdout lock (tools/check_holdout_leak.py)."""
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return " ".join(text.split())


def is_span_of(sentence: str, value: str) -> bool:
    """True if value, normalised, is a run of whole words of the normalised sentence."""
    v, s = normalise(value), normalise(sentence)
    return bool(v) and f" {v} " in f" {s} "


def token_f1(got: str, want: str) -> float:
    g, w = normalise(got).split(), normalise(want).split()
    if not g or not w:
        return float(g == w)
    common = sum(min(g.count(t), w.count(t)) for t in set(g))
    if not common:
        return 0.0
    precision, recall = common / len(g), common / len(w)
    return 2 * precision * recall / (precision + recall)


def judge(sentence: str, slot: str, got: str | None, want: str | list[str] | None) -> str:
    """One of right, wrong, invented, missed, filled, empty (see the module doc)."""
    wants = [w for w in ([want] if isinstance(want, str) else (want or [])) if normalise(w)]
    if got is None or not normalise(got):
        return "missed" if wants else "empty"
    if not is_span_of(sentence, got):
        return "invented"
    if not wants:
        return "filled"
    if slot == "title":
        return "right" if max(token_f1(got, w) for w in wants) >= TITLE_F1 else "wrong"
    return "right" if any(normalise(got) == normalise(w) for w in wants) else "wrong"


def score(items: list[dict], outputs: dict[str, dict]) -> dict:
    """items: {"id", "text", "slots": {slot: reference or list of references or None}};
    outputs: id -> parsed model output. Only items with "slots" are scored."""
    kinds = ("right", "wrong", "invented", "missed", "filled", "empty")
    per_slot = {s: dict.fromkeys(kinds, 0) for s in SLOTS}
    records = []
    for item in items:
        ref = item.get("slots")
        if ref is None:
            continue
        out = outputs.get(item["id"]) or {}
        verdict = {s: judge(item["text"], s, out.get(s), ref.get(s)) for s in SLOTS}
        for s, v in verdict.items():
            per_slot[s][v] += 1
        records.append({"id": item["id"], **verdict})
    n = len(records)
    good = sum(1 for r in records if all(r[s] in ("right", "empty") for s in SLOTS))
    return {"n": n, "all_slots_right": good, "invented": sum(p["invented"] for p in per_slot.values()),
            "per_slot": per_slot, "items": records}
