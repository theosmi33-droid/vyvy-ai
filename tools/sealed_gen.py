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
and the teacher's example sentences; near copies are dropped, and a frame whose
fillings are often near copies refuses the whole grammar (frame_check).

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
import math
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
    """Near-copy test, exact against tools/check_holdout_leak.py at the same threshold.

    Characters: difflib's ratio is at most 2*min(la, lb)/(la + lb), so only texts in a
    length window around the query can reach the threshold; those are filtered by
    quick_ratio (another upper bound) before the full ratio. Words: Jaccard over the
    texts that share a word (no shared word gives 0).
    """

    def __init__(self, texts: list[str], threshold: float = THRESHOLD) -> None:
        self.threshold = threshold
        self.items: list[tuple[str, set[str]]] = []
        self.by_len: dict[int, list[int]] = collections.defaultdict(list)
        self.by_word: dict[str, list[int]] = collections.defaultdict(list)
        for t in texts:
            self.add(t)

    def add(self, text: str) -> None:
        n = leak.normalise(text)
        words = set(n.split())
        k = len(self.items)
        self.items.append((n, words))
        self.by_len[len(n)].append(k)
        for w in words:
            self.by_word[w].append(k)

    def hit(self, text: str) -> bool:
        n = leak.normalise(text)
        words = set(n.split())
        t = self.threshold
        for k in {k for w in words for k in self.by_word.get(w, ())}:
            owords = self.items[k][1]
            if len(words & owords) / len(words | owords) >= t:
                return True
        la = len(n)
        lo, hi = math.ceil(la * t / (2 - t)), math.floor(la * (2 - t) / t)
        m = difflib.SequenceMatcher(None, autojunk=False)
        m.set_seq2(n)
        for length in range(lo, hi + 1):
            for k in self.by_len.get(length, ()):
                m.set_seq1(self.items[k][0])
                if m.quick_ratio() >= t and m.ratio() >= t:
                    return True
        return False


def check_grammar(grammar: dict) -> None:
    if grammar.get("schema") != 1:
        raise ValueError("schema must be 1")
    for k, f in enumerate(grammar["frames"], 1):
        missing = set(SLOT.findall(f["text"])) - set(grammar["words"])
        if missing:
            raise ValueError(f"frame {k}: no word list for {sorted(missing)}")


def pattern_sentences(root: Path = ROOT) -> list[str]:
    """Example sentences the teacher sees: RULES of the synthesis and training/patterns.py."""
    texts = [t for _, t in leak.load_prompt_examples(root / "training" / "synthesize.py")]
    sys.path.insert(0, str(root / "training"))
    import patterns  # noqa: E402
    return texts + [e for p in patterns.PATTERNS.values() for e in p["examples"]]


def fill(frame: dict, words: dict[str, list[str]], rng: random.Random) -> str:
    text = SLOT.sub(lambda m: rng.choice(words[m.group(1)]), frame["text"])
    return text[0].upper() + text[1:]


def frame_check(grammar: dict, near: Near, samples: int = 20, limit: float = 0.2) -> list[str]:
    """Frames whose filled sentences are near copies of training, holdout or teacher sentences.

    A seed hides which sentences are drawn, not the frames: a frame that the training
    data already covers would test what was trained. More than `limit` of `samples`
    fillings near a known sentence refuses the frame.
    """
    rng = random.Random(0)
    bad = []
    for k, f in enumerate(grammar["frames"], 1):
        share = sum(near.hit(fill(f, grammar["words"], rng)) for _ in range(samples)) / samples
        if share > limit:
            bad.append(f"frame {k} ({f['intent']}): {share:.0%} of {samples} fillings are near known sentences")
    return bad


def generate(grammar: dict, seed: str, n: int, references: list[str], frame_limit: float = 0.2) -> list[dict]:
    """n questions, evenly over the intents, deterministic for (grammar, seed).

    Sentences of one frame are alike by construction, so within the set only exact
    repeats are dropped; against references the near-copy threshold applies. The
    questions of one frame are not independent: confidence bounds should count frames
    (see "frame" in each item), not questions.
    """
    check_grammar(grammar)
    near = Near(references)
    bad = frame_check(grammar, near, limit=frame_limit)
    if bad:
        raise SystemExit("frames too close to known sentences: " + "; ".join(bad))
    rng = random.Random(hashlib.sha256(seed.encode()).digest())
    frames: dict[str, list[tuple[int, dict]]] = collections.defaultdict(list)
    for k, f in enumerate(grammar["frames"], 1):
        frames[f["intent"]].append((k, f))
    intents = sorted(frames)
    seen: set[str] = set()
    out: list[dict] = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        intent = intents[len(out) % len(intents)]
        k, frame = rng.choice(frames[intent])
        text = fill(frame, grammar["words"], rng)
        key = leak.normalise(text)
        if key in seen or near.hit(text):
            continue
        seen.add(key)
        out.append({"id": f"sealed-{len(out) + 1:04d}", "text": text, "expected": intent, "frame": k})
    if len(out) < n:
        got = collections.Counter(x["expected"] for x in out)
        raise SystemExit(f"only {len(out)} of {n} questions; too few combinations for "
                         + ", ".join(f"{i} ({got[i]})" for i in intents if got[i] == min(got.values())))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a sealed blind set from a private grammar and seed")
    p.add_argument("grammar", type=Path)
    p.add_argument("--seed", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frame-limit", type=float, default=0.2, help="share of near fillings that refuses a frame")
    args = p.parse_args(argv)
    if ROOT in args.out.resolve().parents or ROOT in args.grammar.resolve().parents:
        raise SystemExit("grammar and sealed set stay outside the public repository")
    items = generate(json.loads(args.grammar.read_text(encoding="utf-8")), args.seed, args.n,
                     reference_sentences() + pattern_sentences(), args.frame_limit)
    args.out.write_text(json.dumps({"name": "sealed blind set", "n": len(items),
                                    "grammar_sha256": hashlib.sha256(args.grammar.read_bytes()).hexdigest(),
                                    "items": items}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(items)} questions, " + ", ".join(f"{i} {k}" for i, k in
                                                  sorted(collections.Counter(x['expected'] for x in items).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
