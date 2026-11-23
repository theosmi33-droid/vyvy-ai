#!/usr/bin/env python3
"""Holdout lock: training data must not contain holdout questions or near copies.

Protects the development holdout and both blind holdouts.

Compares every training sentence with every holdout sentence after
normalisation (case, punctuation, whitespace) by two measures:

- character similarity (difflib ratio, close to a normalised Levenshtein distance)
- word overlap (Jaccard of the word sets)

If either reaches --threshold (default MAX_SIMILARITY, 0.8), the pair is reported and the
script exits with code 1. Training data: data/train/*.jsonl, one JSON object per
line with "text" and "intent", plus the example sentences of RULES in
training/synthesize.py (the teacher would copy a holdout sentence from there
into many training sentences) and the example questions of the benchmark
prompts in runner/bench.py (every training line and every measurement carries
them).

    python tools/check_holdout_leak.py
    python tools/check_holdout_leak.py --train path/to/file.jsonl
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "data" / "holdout_benchmarks.json"
BLIND = ROOT / "data" / "holdout_blind_v1.json"
BLIND_V2 = ROOT / "data" / "holdout_blind_v2.json"
BLIND_V3 = ROOT / "data" / "holdout_blind_v3.json"
BLIND_V4 = ROOT / "data" / "holdout_blind_v4.json"
#: Every holdout ever used; generators that were built against all of them keep filtering
#: against this list, so their committed output does not change.
HOLDOUTS = [HOLDOUT, BLIND, BLIND_V2, BLIND_V3, BLIND_V4]
#: Retired holdouts: no longer measured, released into the training data
#: (data/train/vyvy_retired_blind_v2.jsonl, tools/build_retired_holdout.py). Issue 8, 20.11.2026.
RETIRED = [BLIND_V2]
#: What the lock protects today: training data must stay away from these.
GUARDED = [h for h in HOLDOUTS if h not in RETIRED]
TRAIN_DIR = ROOT / "data" / "train"
# The teacher prompt carries example sentences (RULES); a holdout sentence there
# would be copied into training data in many variants, so they are checked too.
SYNTH = ROOT / "training" / "synthesize.py"
PATTERNS = ROOT / "training" / "patterns.py"
BENCH = ROOT / "runner" / "bench.py"


def normalise(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return " ".join(text.split())


# One threshold for this lock and for tools/holdout_build.py.
MAX_SIMILARITY = 0.8
# A sentence that differs only by these words (or punctuation and case) is the same sentence.
FILLERS = frozenset({"eigentlich", "denn", "mal", "bitte", "doch"})


def without_fillers(normalised: str) -> str:
    return " ".join(w for w in normalised.split() if w not in FILLERS)


def char_similarity(a: str, b: str) -> float:
    """difflib's ratio depends on the order (0.87 one way, 0.783 the other); take the larger."""
    return max(difflib.SequenceMatcher(None, x, y, autojunk=False).ratio() for x, y in ((a, b), (b, a)))


def similarity(a: str, b: str) -> tuple[float, float]:
    chars = char_similarity(a, b)
    wa, wb = set(a.split()), set(b.split())
    words = len(wa & wb) / len(wa | wb) if wa | wb else 1.0
    return chars, words


def load_train(paths: list[Path]) -> list[tuple[str, str]]:
    rows = []
    for path in paths:
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                rows.append((f"{path.name}:{no}", json.loads(line)["text"]))
    return rows


def load_prompt_examples(path: Path) -> list[tuple[str, str]]:
    """Example sentences the teacher sees, read without importing the script: RULES of the
    synthesis, or PATTERNS of the error-category templates (training/patterns.py)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        names = {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set()
        if "RULES" in names:
            rules = ast.literal_eval(node.value)
            return [(f"{path.name}:RULES[{intent}][{i}]", text)
                    for intent, (_rule, examples) in rules.items() for i, text in enumerate(examples)]
        if "PATTERNS" in names:
            pats = ast.literal_eval(node.value)
            return [(f"{path.name}:PATTERNS[{key}][{i}]", text)
                    for key, p in pats.items() for i, text in enumerate(p["examples"])]
    raise SystemExit(f"{path}: no RULES or PATTERNS found; the prompt examples cannot be checked")


def load_bench_examples(path: Path = BENCH) -> list[tuple[str, str]]:
    """The example questions of the benchmark prompts of both contracts, read from the prompt texts.

    Contract 1 is frozen and measured; its two known near copies of holdout questions are KNOWN_OLD_PAIRS."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("bench_for_lock", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return [(f"{path.name}:{label}[{name}][{i}]", q)
            for label, prompts in (("PROMPTS", bench.PROMPTS), ("PROMPTS_V1", bench.PROMPTS_V1))
            for name, text in prompts.items() for i, q in enumerate(re.findall(r'Question: "([^"]+)"', text))]


# Known old pairs, contract 1 only (frozen and measured): exactly this holdout id, this place and both
# normalised texts. Anything else stays a leak; contract 2 has none.
_V1_WHY = "contract 1 is measured with this prompt example and this holdout sentence; changing either makes old results incomparable"
KNOWN_OLD_PAIRS = {
    ("blind-022", "bench.py:PROMPTS_V1[detailed][2]"): {
        "train_sha256": "6c7d262bc1797bc19e0cc1c7eba9bb1fc21e0c78bbfd61641ab4733f06ab510f",
        "holdout_sha256": "e65e2c8f9260cec171643514fbc4e8b8bfa07e0748e8a71059e1cf30bb5d2df4",
        "date": "2026-11-23", "why": "chars 0.833, hidden below the old 0.85; " + _V1_WHY,
    },
    ("blind3-124", "bench.py:PROMPTS_V1[detailed][4]"): {
        "train_sha256": "a71d520944090b7a835f3730a1e4a42bf54b606b28be55a6af4e27bfd1c8157b",
        "holdout_sha256": "6743a1b7e114725dc8d17bc52fdf715be229f3c5c30fee02e1a2a53980e89290",
        "date": "2026-11-23", "why": "chars 0.902, found when the lock first read the prompts; " + _V1_WHY,
    },
}


def known_old_pair(leak: dict) -> bool:
    entry = KNOWN_OLD_PAIRS.get((leak["holdout"], leak["train"]))
    sha = lambda t: hashlib.sha256(normalise(t).encode("utf-8")).hexdigest()  # noqa: E731
    return bool(entry) and sha(leak["text"]) == entry["train_sha256"] and sha(leak["holdout_text"]) == entry["holdout_sha256"]


def find_leaks(train: list[tuple[str, str]], holdout: list[dict], threshold: float) -> list[dict]:
    hold = [(h["id"], h["text"], normalise(h["text"])) for h in holdout]
    leaks = []
    for where, text in train:
        t = normalise(text)
        for hid, htext, h in hold:
            chars, words = similarity(t, h)
            if max(chars, words) >= threshold or without_fillers(t) == without_fillers(h):
                leaks.append({"train": where, "text": text, "holdout": hid, "holdout_text": htext,
                              "chars": round(chars, 3), "words": round(words, 3)})
    return leaks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fail if training data overlaps the holdout set")
    p.add_argument("--train", type=Path, nargs="*", help="training files (default: data/train/*.jsonl)")
    p.add_argument("--holdout", type=Path, nargs="*", default=GUARDED,
                   help="holdout sets to protect (default: development holdout and blind holdout)")
    p.add_argument("--threshold", type=float, default=MAX_SIMILARITY)
    p.add_argument("--prompts", type=Path, nargs="*",
                   help="scripts whose RULES or PATTERNS examples are checked (default: training/synthesize.py and "
                        "training/patterns.py; "
                        "with --train only when given)")
    args = p.parse_args(argv)

    paths = args.train if args.train else sorted(TRAIN_DIR.glob("*.jsonl"))
    prompts = args.prompts if args.prompts is not None else ([] if args.train else [SYNTH, PATTERNS])
    if not paths and not prompts:
        print("No training data found; nothing to check.")
        return 0
    holdout = [i for h in args.holdout if h.exists() for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    train = load_train(paths) + [row for path in prompts for row in load_prompt_examples(path)]
    if args.prompts is None and not args.train:
        train += load_bench_examples()
    leaks = find_leaks(train, holdout, args.threshold)
    for leak in (k for k in leaks if known_old_pair(k)):
        print(f"known old pair: {leak['train']} ~ {leak['holdout']} (KNOWN_OLD_PAIRS)")
    leaks = [k for k in leaks if not known_old_pair(k)]
    if leaks:
        print(f"Holdout leak: {len(leaks)} training sentence(s) at or above {args.threshold}:", file=sys.stderr)
        for leak in leaks:
            print(f"  {leak['train']} ~ {leak['holdout']} (chars {leak['chars']}, words {leak['words']})\n"
                  f"    train:   {leak['text']}\n    holdout: {leak['holdout_text']}", file=sys.stderr)
        return 1
    print(f"OK: {len(train)} training sentences, {len(holdout)} holdout sentences, no pair at or above "
          f"{args.threshold}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
