#!/usr/bin/env python3
"""Sealed blind set from a grammar: frames x word lists x a secret seed.

Every question is built from a frame whose intent is fixed, so the label is
known by construction and cannot be wrong. The same grammar and seed always
give the same set; a new seed gives a new sealed set when the old one is spent.

The grammar file and the seed stay outside this public repository: with both
public, training data could be written to fit them. The output is refused
inside the repository for the same reason.

Each generated question is checked against every training, holdout and probe
sentence (character similarity and word overlap, as tools/check_holdout_leak.py)
and against the questions generated before it; near copies are dropped.

    python tools/sealed_gen.py GRAMMAR.json --seed "$SEALED_SEED" --n 1000 --out ~/sealed/sealed_v1.json

Grammar: {"schema": 1, "words": {"TAG_HEUTE": ["heute", "heid", ...], ...},
          "frames": [{"intent": "appointments_today", "text": "Was hab ich {TAG_HEUTE} an Terminen?"}, ...]}
"""

from __future__ import annotations

import argparse
import collections
import difflib
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_holdout_leak as leak  # noqa: E402

SLOT = re.compile(r"\{([A-Z_]+)\}")
THRESHOLD = 0.8


def reference_sentences(root: Path = ROOT) -> list[str]:
    texts = [json.loads(line)["text"] for f in sorted((root / "data" / "train").glob("*.jsonl"))
             for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    for f in sorted(glob.glob(str(root / "data" / "holdout_*.json")) + glob.glob(str(root / "data" / "probe_*.json"))):
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        texts += [i["text"] for i in (data["items"] if isinstance(data, dict) else data)]
    return texts


class Near:
    """Near-copy test against a growing list; cheap upper bounds first, difflib last."""

    def __init__(self, texts: list[str], threshold: float = THRESHOLD) -> None:
        self.threshold = threshold
        self.items: list[tuple[str, set[str]]] = []
        self.by_word: dict[str, list[int]] = collections.defaultdict(list)
        for t in texts:
            self.add(t)

    def add(self, text: str) -> None:
        n = leak.normalise(text)
        words = set(n.split())
        for w in words:
            self.by_word[w].append(len(self.items))
        self.items.append((n, words))

    def hit(self, text: str) -> bool:
        n = leak.normalise(text)
        words = set(n.split())
        for k in {k for w in words for k in self.by_word.get(w, ())}:
            other, owords = self.items[k]
            if len(words & owords) / len(words | owords) >= self.threshold:
                return True
            m = difflib.SequenceMatcher(None, n, other, autojunk=False)
            if m.real_quick_ratio() >= self.threshold and m.quick_ratio() >= self.threshold \
                    and m.ratio() >= self.threshold:
                return True
        return False


def check_grammar(grammar: dict) -> None:
    if grammar.get("schema") != 1:
        raise ValueError("schema must be 1")
    for k, f in enumerate(grammar["frames"], 1):
        missing = set(SLOT.findall(f["text"])) - set(grammar["words"])
        if missing:
            raise ValueError(f"frame {k}: no word list for {sorted(missing)}")


def generate(grammar: dict, seed: str, n: int, references: list[str], per_intent_even: bool = True) -> list[dict]:
    """n questions, as evenly over the intents as the grammar allows, deterministic for (grammar, seed)."""
    check_grammar(grammar)
    rng = random.Random(hashlib.sha256(seed.encode()).digest())
    frames: dict[str, list[dict]] = collections.defaultdict(list)
    for f in grammar["frames"]:
        frames[f["intent"]].append(f)
    intents = sorted(frames)
    near = Near(references)
    out: list[dict] = []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        intent = intents[len(out) % len(intents)] if per_intent_even else rng.choice(intents)
        frame = rng.choice(frames[intent])
        text = SLOT.sub(lambda m: rng.choice(grammar["words"][m.group(1)]), frame["text"])
        text = text[0].upper() + text[1:]
        if near.hit(text):
            continue
        near.add(text)
        out.append({"id": f"sealed-{len(out) + 1:04d}", "text": text, "expected": intent})
    if len(out) < n:
        raise SystemExit(f"only {len(out)} of {n} questions without a near copy; the grammar is too small")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a sealed blind set from a private grammar and seed")
    p.add_argument("grammar", type=Path)
    p.add_argument("--seed", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if ROOT in args.out.resolve().parents or ROOT in args.grammar.resolve().parents:
        raise SystemExit("grammar and sealed set stay outside the public repository")
    items = generate(json.loads(args.grammar.read_text(encoding="utf-8")), args.seed, args.n, reference_sentences())
    args.out.write_text(json.dumps({"name": "sealed blind set", "n": len(items),
                                    "grammar_sha256": hashlib.sha256(args.grammar.read_bytes()).hexdigest(),
                                    "items": items}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(items)} questions, " + ", ".join(f"{i} {k}" for i, k in
                                                  sorted(collections.Counter(x['expected'] for x in items).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
