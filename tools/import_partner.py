#!/usr/bin/env python3
"""Take a partner's training sentences into data/train/ after a strict check.

Partner data is synthetic and anonymised before it is handed over; this check
does not trust that. A sentence is dropped if it

- contains an e-mail address, a phone number, an IBAN, a web address or a
  street with a house number,
- does not fit the output contract (intent, timeframe, entity),
- is at or above --max-similarity (default 0.8, as the generators; the lock is
  0.85) to any holdout sentence, or is already in the training data.

Writes data/train/vyvy_partner_<first 9 hex of the input's SHA-256>.jsonl and a
.json report with the counts per reason, and prints the path. The files name no
origin. --check only re-runs the personal-data check on files already in
data/train/ (Data check workflow) and fails on any hit.

    python tools/import_partner.py path/to/partner.jsonl --sha256 <expected>
    python tools/import_partner.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402
import check_holdout_leak as lock  # noqa: E402

PERSONAL = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "url": re.compile(r"https?://|\bwww\.|\b[\w-]+\.(?:de|com|net|org|eu|at|ch|io|info)\b", re.I),
    "phone": re.compile(r"(?:\+\d{2}|\b00\d{2}|\b0)[\s/()-]*\d(?:[\s/()-]*\d){5,}"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b", re.I),
    "street": re.compile(r"\b[\wäöüß-]*(?:straße|strasse|str\.|weg|allee|platz|gasse|ring|damm|ufer)\s*\d+\s?[a-z]?\b",
                         re.I),
}


def personal_hits(text: str) -> list[str]:
    return [name for name, pattern in PERSONAL.items() if pattern.search(text)]


def check_existing() -> int:
    bad = 0
    for path in sorted(lock.TRAIN_DIR.glob("vyvy_partner_*.jsonl")):
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            hits = personal_hits(json.loads(line)["text"]) if line.strip() else []
            if hits:
                bad += 1
                print(f"{path.name}:{no}: {', '.join(hits)}")
    print("OK: no personal data in partner files" if not bad else f"{bad} partner sentences with personal data")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check and import partner training sentences")
    p.add_argument("source", type=Path, nargs="?")
    p.add_argument("--sha256", help="expected SHA-256 of the source file")
    p.add_argument("--train-dir", type=Path, default=lock.TRAIN_DIR)
    p.add_argument("--max-similarity", type=float, default=0.8)
    p.add_argument("--check", action="store_true", help="only check partner files already in data/train/")
    args = p.parse_args(argv)
    if args.check:
        return check_existing()
    if not args.source:
        p.error("source file required")

    raw = args.source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if args.sha256 and digest != args.sha256:
        raise SystemExit(f"SHA-256 mismatch: {digest} != {args.sha256}")
    out = args.train_dir / f"vyvy_partner_{digest[:9]}.jsonl"
    if out.exists():
        print(f"{out} exists; nothing to import", file=sys.stderr)
        return 3

    intents = bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))
    holdout = [i for h in lock.HOLDOUTS if h.exists() for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    hold_norm = [lock.normalise(h["text"]) for h in holdout]
    seen = {lock.normalise(t) for _, t in lock.load_train(sorted(args.train_dir.glob("*.jsonl")))}

    rows, dropped = [], {"personal": 0, "contract": 0, "near_holdout": 0, "duplicate": 0}
    hits_by_kind: dict[str, int] = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        row = {"text": str(r.get("text", "")).strip(), "intent": r.get("intent"),
               "timeframe": r.get("timeframe"), "entity": r.get("entity")}
        hits = personal_hits(row["text"]) + (personal_hits(row["entity"]) if isinstance(row["entity"], str) else [])
        if hits:
            dropped["personal"] += 1
            for h in hits:
                hits_by_kind[h] = hits_by_kind.get(h, 0) + 1
            continue
        answer = json.dumps({**{k: row[k] for k in ("intent", "timeframe", "entity")}, "confidence": 0.9})
        if not row["text"] or bench.parse_output(answer, intents) is None:
            dropped["contract"] += 1
            continue
        key = lock.normalise(row["text"])
        if key in seen:
            dropped["duplicate"] += 1
            continue
        if any(max(lock.similarity(key, h)) >= args.max_similarity for h in hold_norm):
            dropped["near_holdout"] += 1
            continue
        seen.add(key)
        rows.append(row)

    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report = {"source_sha256": digest, "kind": "partner data, synthetic, anonymised", "imported": len(rows),
              "dropped": dropped, "personal_hits": hits_by_kind, "max_similarity": args.max_similarity,
              "per_intent": {i: sum(r["intent"] == i for r in rows) for i in intents}}
    out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(out)
    print(json.dumps(report, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
