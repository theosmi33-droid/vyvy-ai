#!/usr/bin/env python3
"""Holdout lock: training data must not contain holdout questions or near copies.

Protects the development holdout and both blind holdouts.

Compares every training sentence with every holdout sentence after
normalisation (case, punctuation, whitespace) by two measures:

- character similarity (difflib ratio, close to a normalised Levenshtein distance)
- word overlap (Jaccard of the word sets)

If either reaches --threshold (default 0.85), the pair is reported and the
script exits with code 1. Training data: data/train/*.jsonl, one JSON object per
line with "text" and "intent", plus the example sentences of RULES in
training/synthesize.py (the teacher would copy a holdout sentence from there
into many training sentences).

    python tools/check_holdout_leak.py
    python tools/check_holdout_leak.py --train path/to/file.jsonl
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "data" / "holdout_benchmarks.json"
BLIND = ROOT / "data" / "holdout_blind_v1.json"
BLIND_V2 = ROOT / "data" / "holdout_blind_v2.json"
BLIND_V3 = ROOT / "data" / "holdout_blind_v3.json"
HOLDOUTS = [HOLDOUT, BLIND, BLIND_V2, BLIND_V3]
TRAIN_DIR = ROOT / "data" / "train"
# The teacher prompt carries example sentences (RULES); a holdout sentence there
# would be copied into training data in many variants, so they are checked too.
SYNTH = ROOT / "training" / "synthesize.py"
PATTERNS = ROOT / "training" / "patterns.py"


def normalise(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return " ".join(text.split())


def similarity(a: str, b: str) -> tuple[float, float]:
    chars = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
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


def find_leaks(train: list[tuple[str, str]], holdout: list[dict], threshold: float) -> list[dict]:
    hold = [(h["id"], h["text"], normalise(h["text"])) for h in holdout]
    leaks = []
    for where, text in train:
        t = normalise(text)
        for hid, htext, h in hold:
            chars, words = similarity(t, h)
            if max(chars, words) >= threshold:
                leaks.append({"train": where, "text": text, "holdout": hid, "holdout_text": htext,
                              "chars": round(chars, 3), "words": round(words, 3)})
    return leaks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fail if training data overlaps the holdout set")
    p.add_argument("--train", type=Path, nargs="*", help="training files (default: data/train/*.jsonl)")
    p.add_argument("--holdout", type=Path, nargs="*", default=HOLDOUTS,
                   help="holdout sets to protect (default: development holdout and blind holdout)")
    p.add_argument("--threshold", type=float, default=0.85)
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
    leaks = find_leaks(train, holdout, args.threshold)
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
