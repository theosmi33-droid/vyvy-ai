#!/usr/bin/env python3
"""Check privately written sealed sets (create-v<N>, inject-v<N>) without showing anyone's text.

The writer of a sealed set runs this over their private folder. It reads the
training data, every holdout and the teacher's example sentences by machine and
prints only the writer's own ids with numbers - never a training or holdout
sentence, never one of the writer's sentences.

Checks per file sealed_<version>.json in --dir:
- form: frame fields, items_sha256 (sha256 of json.dumps(items, ensure_ascii=False,
  sort_keys=True)), unique ids and texts, known intents; create: "slots" only and
  always for create_appointment/create_task, every span a literal piece of the
  sentence (runner/slots.is_span_of), null instead of empty; inject: "accept" without
  a writing intent, "kind" set
- counts per intent, per tag (create) or kind (inject)
- lexical lock (tools/check_holdout_leak.py: characters and word overlap) against
  data/train/*.jsonl, every holdout, the probe sets and the teacher examples; ids at
  or above --threshold (0.8) are listed with their score and the kind of the closer
  sentence (train, holdout, probe, teacher)
- --terms FILE: a JSON file of terms that must not appear (any list of strings
  in it counts; matched case-insensitively as words)

    python tools/audit_check.py --dir ~/.vyvy-audit-sealed [--terms FILE]

Exit code 0 when every check passes, 1 otherwise. Nothing is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402
import check_holdout_leak as lock  # noqa: E402
import sealed_score  # noqa: E402
import slots  # noqa: E402

WRITES = sealed_score.WRITES
PROBES = sorted((ROOT / "data").glob("probe_*.json"))
FRAME = ("name", "version", "n", "items_sha256", "created", "items")


def items_sha(items: list[dict]) -> str:
    return hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def form(data: dict, kind: str, intents: tuple[str, ...]) -> list[str]:
    """Problems with the file's form, by id; no text."""
    bad = [f"frame: missing {k}" for k in FRAME if k not in data]
    items = data.get("items", [])
    if data.get("n") != len(items):
        bad.append(f"frame: n {data.get('n')} != {len(items)} items")
    if data.get("items_sha256") != items_sha(items):
        bad.append("frame: items_sha256 does not match the items")
    ids = Counter(i.get("id") for i in items)
    bad += [f"{i}: id twice" for i, c in ids.items() if c > 1]
    texts = Counter(lock.normalise(i.get("text", "")) for i in items)
    bad += [f"{i['id']}: same text as another entry" for i in items if texts[lock.normalise(i.get("text", ""))] > 1]
    for i in items:
        iid, text = i.get("id", "?"), i.get("text")
        if not isinstance(text, str) or not text.strip():
            bad.append(f"{iid}: no text")
            continue
        if kind == "create":
            intent = i.get("expected_intent")
            if intent not in intents:
                bad.append(f"{iid}: unknown intent {intent!r}")
            has = "slots" in i and i["slots"] is not None
            if has != (intent in WRITES):
                bad.append(f"{iid}: slots {'missing' if intent in WRITES else 'only for create intents'}")
            for name, spans in (i.get("slots") or {}).items():
                if name not in slots.SLOTS:
                    bad.append(f"{iid}: unknown slot {name}")
                elif spans is not None and (not isinstance(spans, list) or not spans):
                    bad.append(f"{iid}: {name} must be null or a non-empty list of spans")
                else:
                    bad += [f"{iid}: {name} span not a literal piece of the sentence"
                            for s in spans or [] if not (isinstance(s, str) and slots.is_span_of(text, s))]
        else:
            accept = i.get("accept")
            if not isinstance(accept, list) or not accept or any(a not in intents for a in accept):
                bad.append(f"{iid}: accept must list known intents")
            elif set(accept) & WRITES:
                bad.append(f"{iid}: accept holds a writing intent")
            if not i.get("kind"):
                bad.append(f"{iid}: kind missing")
    return bad


def counts(items: list[dict], kind: str) -> dict:
    if kind == "create":
        return {"intent": dict(Counter(i.get("expected_intent") for i in items)),
                "tag": dict(Counter(t for i in items for t in i.get("tags", []))),
                "missing": dict(Counter(s for i in items for s, v in (i.get("slots") or {}).items() if v is None))}
    return {"kind": dict(Counter(i.get("kind") for i in items)),
            "accept": dict(Counter(a for i in items for a in i.get("accept", [])))}


def known_sentences() -> list[tuple[str, str]]:
    """(kind, text) of everything a sealed sentence must stay away from."""
    out = [("train", t) for _, t in lock.load_train(sorted(lock.TRAIN_DIR.glob("*.jsonl")))]
    out += [("holdout", i["text"]) for h in lock.HOLDOUTS if h.exists()
            for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    out += [("probe", i["text"]) for p in PROBES for i in json.loads(p.read_text(encoding="utf-8")).get("items", [])]
    out += [("teacher", t) for p in (lock.SYNTH, lock.PATTERNS) for _, t in lock.load_prompt_examples(p)]
    return out


def near(items: list[dict], known: list[tuple[str, str]], threshold: float) -> list[tuple[str, float, str]]:
    """(own id, best score, kind of the closer sentence) at or above threshold; no text."""
    import difflib

    norm = [(k, s, set(s.split())) for k, t in known for s in [lock.normalise(t)]]
    out = []
    for i in items:
        t = lock.normalise(i["text"])
        tw = set(t.split())
        best, where = 0.0, ""
        for k, s, sw in norm:
            words = len(tw & sw) / len(tw | sw) if tw | sw else 1.0
            if words > best:
                best, where = words, k
            # Same score as lock.similarity; 2*min/(la+lb) bounds the ratio, so most pairs skip it.
            floor = max(best, threshold * 0.9)
            if 2 * min(len(t), len(s)) / (len(t) + len(s) or 1) < floor:
                continue
            m = difflib.SequenceMatcher(None, t, s, autojunk=False)
            if m.real_quick_ratio() < floor or m.quick_ratio() < floor:
                continue
            chars = m.ratio()
            if chars > best:
                best, where = chars, k
        if best >= threshold:
            out.append((i["id"], round(best, 3), where))
    return out


def terms_from(path: Path) -> list[str]:
    def strings(v):
        if isinstance(v, str):
            return []
        if isinstance(v, list):
            return [x for x in v if isinstance(x, str)]
        if isinstance(v, dict):
            return [x for k, w in v.items() if not k.startswith("_") for x in strings(w)]
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted({t for t in strings(data) if t.strip()})


def term_hits(items: list[dict], terms: list[str]) -> list[str]:
    pats = [re.compile(r"(?<!\w)" + re.escape(t.casefold()) + r"(?!\w)") for t in terms]
    return [i["id"] for i in items if any(p.search(i["text"].casefold()) for p in pats)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check private sealed sets; prints ids and numbers only")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--terms", type=Path, help="JSON file with terms that must not appear")
    p.add_argument("--threshold", type=float, default=0.8)
    args = p.parse_args(argv)
    folder = args.dir.expanduser().resolve()
    if folder == ROOT or ROOT in folder.parents:
        raise SystemExit("the sealed folder must stay outside this repository")
    intents = bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))
    files = sorted(folder.glob("sealed_*-v*.json"))
    if not files:
        raise SystemExit(f"no sealed_create-v<N>.json or sealed_inject-v<N>.json in {folder}")
    known = known_sentences()
    terms = terms_from(args.terms) if args.terms else []
    ok = True
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        version = path.stem.removeprefix("sealed_")
        kind = sealed_score.kind_of(version)
        items = data.get("items", [])
        print(f"## {path.name}: {kind}, {len(items)} entries")
        bad = form(data, kind, intents)
        close = near([i for i in items if isinstance(i.get("text"), str)], known, args.threshold)
        hits = term_hits([i for i in items if isinstance(i.get("text"), str)], terms)
        print(json.dumps(counts(items, kind), indent=1, sort_keys=True, ensure_ascii=False))
        for line in bad:
            print(f"form: {line}")
        for iid, score, where in close:
            print(f"near: {iid} {score} ({where})")
        for iid in hits:
            print(f"term: {iid}")
        print(f"form problems {len(bad)}, near {len(close)} (>= {args.threshold}), term hits {len(hits)}\n")
        ok = ok and not (bad or close or hits)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
