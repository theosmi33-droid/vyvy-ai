#!/usr/bin/env python3
"""Semantic holdout check: training sentences that mean the same as a holdout sentence.

The holdout lock (check_holdout_leak.py) compares characters and words, so a
paraphrase with other words passes it. This check embeds every training and
holdout sentence with a small multilingual sentence model and lists pairs with
a cosine similarity at or above --report (default 0.90). It fails only at or
above --fail (default 0.97, near-identical meaning) until the distribution on
real data shows where paraphrase ends and the same question begins.

Same-intent paraphrases are expected: the training data teaches exactly those
intents. The list is for review, and the fail threshold keeps out copies that
only change a word or two.

Runs on the GitHub runner (Data check workflow), never on a developer machine.

    pip install -r tools/semantic-requirements.txt
    python tools/check_semantic_leak.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_holdout_leak as lock  # noqa: E402

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Apache 2.0
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="List training sentences that paraphrase a holdout sentence")
    p.add_argument("--report", type=float, default=0.90)
    p.add_argument("--fail", type=float, default=0.97)
    p.add_argument("--summary", type=Path, help="append the report as Markdown")
    args = p.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    train = lock.load_train(sorted(lock.TRAIN_DIR.glob("*.jsonl")))
    holdout = [i for h in lock.HOLDOUTS if h.exists() for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    t = model.encode([text for _, text in train], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    h = model.encode([i["text"] for i in holdout], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    sim = t @ h.T
    best = sim.max(axis=1)
    which = sim.argmax(axis=1)

    hits = sorted(((float(best[k]), k, int(which[k])) for k in range(len(train)) if best[k] >= args.report),
                  reverse=True)
    failing = [x for x in hits if x[0] >= args.fail]
    quantiles = {q: round(float(sorted(best)[int(q * (len(best) - 1))]), 3) for q in (0.5, 0.9, 0.99)}
    lines = [f"## Semantic holdout check ({MODEL.split('/')[-1]})", "",
             f"{len(train)} training sentences against {len(holdout)} holdout sentences. "
             f"Closest holdout match per training sentence: median {quantiles[0.5]}, "
             f"90 % {quantiles[0.9]}, 99 % {quantiles[0.99]}.", "",
             f"{len(hits)} at or above {args.report}; {len(failing)} at or above {args.fail} (fails).", ""]
    if hits:
        lines += ["| Cosine | Training | Holdout |", "|---:|---|---|"]
        lines += [f"| {s:.3f} | {train[k][1]} | {holdout[j]['id']}: {holdout[j]['text']} |" for s, k, j in hits[:60]]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write(text)
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
