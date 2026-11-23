#!/usr/bin/env python3
"""Build the contract 2 decision set and shadow set from written candidates, blind.

Input is a folder OUTSIDE this repository (the sentences must never pass through it
before they are a holdout):

    candidates.jsonl   {"text": ..., "expected_intent": ...}  one per line, from the writers
    blind_a.jsonl      {"text": ..., "intent": ...}           first blind assignment
    blind_b.jsonl      {"text": ..., "intent": ...}           second blind assignment

A blind line belongs to the candidate with the same text (whitespace collapsed). A
candidate is kept only if

1. its intent is one of the 25 of contract 2 and it is no duplicate of an earlier one,
2. both blind assignments exist and both equal the writer's intent,
3. it is below --max-similarity (0.8) in characters and in words (tools/check_holdout_leak.py
   measures, after the same normalisation) to every training sentence, every holdout
   sentence ever used, every sentence of the sealed fingerprint and every candidate kept
   before it,
4. its meaning is below --max-cosine (0.85, the sentence model of tools/check_semantic_leak.py)
   to the same sentences, unless --semantic skip, which the report and the sets then say.

The kept candidates are split per intent, deterministically (order of
SHA-256(--seed + normalised text)): the first --per-intent (40; --unknown 160 for unknown)
go to the decision set, the next --shadow-per-intent (20; --shadow-unknown 80) to the
shadow set; the rest are counted and not written. Written to --out (also outside the
repository): decision.json and shadow.json in the holdout format (items with id, text,
expected_intent), report.json (counts per intent and stage, drops per reason, input
SHA-256s, thresholds; no sentence text) and fingerprint.json (per sentence of both sets
only hashes: the normalised text, its words and its character trigrams, so a later set
can be checked against these without the text). No sentence is printed. Exit 0 when
every intent is complete, 3 when one is short (the sets are still written and marked
incomplete), 2 on bad input.

Landing a set in data/ and recording its fingerprint needs the user's word; this tool
never writes into the repository.

    python tools/holdout_build.py --candidates /path/outside/repo --out /path/outside/repo/built \
        --sealed-fingerprint /path/to/sealed_fingerprint.json
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402
import check_holdout_leak as lock  # noqa: E402

FINGERPRINT_SCHEMA = 1
REASONS = ("intent_not_in_contract", "duplicate", "blind_missing", "blind_disagrees", "near_training",
           "near_holdout", "near_sealed", "near_candidate", "semantic_training", "semantic_holdout",
           "semantic_sealed", "semantic_candidate")


def norm(text: str) -> str:
    return lock.normalise(text)


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def trigrams(n: str) -> set[str]:
    padded = f"  {n}  "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def fingerprint_item(text: str) -> dict:
    n = norm(text)
    return {"sha256": hashlib.sha256(n.encode("utf-8")).hexdigest(), "words": sorted({_h(w) for w in n.split()}),
            "trigrams": sorted({_h(t) for t in trigrams(n)})}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


class Lexical:
    """Character similarity (difflib ratio) and word overlap against known texts, with cheap bounds first,
    plus fingerprint items known only by hashes (exact text, word and trigram overlap)."""

    def __init__(self, threshold: float):
        self.t = threshold
        self.texts: list[tuple[str, str, set[str]]] = []  # (kind, normalised, words)
        self.prints: list[tuple[str, str, set[str], set[str]]] = []  # (kind, sha, word hashes, trigram hashes)

    def add_text(self, kind: str, text: str) -> None:
        n = norm(text)
        self.texts.append((kind, n, set(n.split())))

    def add_fingerprint(self, kind: str, item: dict) -> None:
        self.prints.append((kind, item["sha256"], set(item["words"]), set(item["trigrams"])))

    def near(self, text: str) -> str | None:
        """Kind of the first known sentence at or above the threshold, or None."""
        n = norm(text)
        words = set(n.split())
        for kind, other, other_words in self.texts:
            if jaccard(words, other_words) >= self.t:
                return kind
            longest = max(len(n), len(other))
            if not longest or 2 * min(len(n), len(other)) / (len(n) + len(other)) < self.t:
                continue
            m = difflib.SequenceMatcher(None, n, other, autojunk=False)
            if m.quick_ratio() >= self.t and m.ratio() >= self.t:
                return kind
        if self.prints:
            fp = fingerprint_item(text)
            w, t = set(fp["words"]), set(fp["trigrams"])
            for kind, sha, pw, pt in self.prints:
                if sha == fp["sha256"] or jaccard(w, pw) >= self.t or jaccard(t, pt) >= self.t:
                    return kind
        return None


class Semantic:
    """Cosine of sentence embeddings (same model and revision as tools/check_semantic_leak.py)."""

    def __init__(self, threshold: float):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        import check_semantic_leak as sem

        self.np, self.t = np, threshold
        self.model = SentenceTransformer(sem.MODEL, revision=sem.REVISION, device="cpu")
        self.kinds: list[str] = []
        self.vectors = None

    def encode(self, texts: list[str]):
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)

    def add(self, kind: str, texts: list[str]) -> None:
        if not texts:
            return
        v = self.encode(texts)
        self.vectors = v if self.vectors is None else self.np.vstack([self.vectors, v])
        self.kinds += [kind] * len(texts)

    def near(self, text: str) -> str | None:
        if self.vectors is None:
            return None
        scores = self.vectors @ self.encode([text])[0]
        best = int(scores.argmax())
        return self.kinds[best] if scores[best] >= self.t else None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(text: str) -> str:
    return " ".join(str(text).split())


def outside_repo(path: Path) -> bool:
    p, root = path.resolve(), ROOT.resolve()
    return p != root and root not in p.parents


def known_sentences() -> tuple[list[str], list[str]]:
    train = [t for _, t in lock.load_train(sorted(lock.TRAIN_DIR.glob("*.jsonl")))]
    holdout = [i["text"] for h in lock.HOLDOUTS if h.exists() for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    return train, holdout


def rank(seed: str, text: str) -> str:
    return hashlib.sha256(f"{seed}\n{norm(text)}".encode("utf-8")).hexdigest()


def build(folder: Path, sealed: dict | None, *, max_similarity: float = 0.8, max_cosine: float = 0.85,
          semantic: bool = True, seed: str = "contract-2", per_intent: int = 40, unknown: int = 160,
          shadow_per_intent: int = 20, shadow_unknown: int = 80,
          known: tuple[list[str], list[str]] | None = None, semantic_model=None) -> dict:
    intents = bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))
    files = {name: folder / f"{name}.jsonl" for name in ("candidates", "blind_a", "blind_b")}
    for name, path in files.items():
        if not path.is_file():
            raise ValueError(f"{name}.jsonl missing in the candidate folder")
    cands = read_jsonl(files["candidates"])
    blind = {name: {key(r["text"]): r.get("intent") for r in read_jsonl(files[name])} for name in ("blind_a", "blind_b")}
    train, holdout = known or known_sentences()

    lex = Lexical(max_similarity)
    for t in train:
        lex.add_text("near_training", t)
    for t in holdout:
        lex.add_text("near_holdout", t)
    for item in (sealed or {}).get("items", []):
        lex.add_fingerprint("near_sealed", item)
    sem = None
    if semantic:
        sem = semantic_model or Semantic(max_cosine)
        sem.add("semantic_training", train)
        sem.add("semantic_holdout", holdout)
        # A sealed fingerprint holds no text; its meaning can only be compared if it carries vectors of the same model.
        vecs = [i["embedding"] for i in (sealed or {}).get("items", []) if i.get("embedding")]
        if vecs:
            v = sem.np.asarray(vecs, dtype="float32")
            sem.vectors = v if sem.vectors is None else sem.np.vstack([sem.vectors, v])
            sem.kinds += ["semantic_sealed"] * len(vecs)

    dropped: Counter[str] = Counter()
    per_intent_stage: dict[str, Counter] = {i: Counter() for i in intents}
    kept: list[dict] = []
    seen: set[str] = set()
    for c in cands:
        text, intent = key(c.get("text", "")), c.get("expected_intent")
        if intent not in intents or not text:
            dropped["intent_not_in_contract"] += 1
            continue
        stage = per_intent_stage[intent]
        stage["candidates"] += 1
        n = norm(text)
        if n in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(n)
        a, b = blind["blind_a"].get(text), blind["blind_b"].get(text)
        if a is None or b is None:
            dropped["blind_missing"] += 1
            continue
        if not a == b == intent:
            dropped["blind_disagrees"] += 1
            continue
        stage["blind_agree"] += 1
        reason = lex.near(text)
        if reason:
            dropped[reason] += 1
            continue
        stage["lexical_ok"] += 1
        if sem is not None:
            reason = sem.near(text)
            if reason:
                dropped[reason] += 1
                continue
            stage["semantic_ok"] += 1
        kept.append({"text": text, "expected_intent": intent})
        # Later candidates must also stay away from this one (and from each other across both sets).
        lex.add_text("near_candidate", text)
        if sem is not None:
            sem.add("semantic_candidate", [text])

    decision, shadow, counts, complete = [], [], {}, True
    for intent in intents:
        group = sorted((k for k in kept if k["expected_intent"] == intent), key=lambda k: rank(seed, k["text"]))
        want_d, want_s = (unknown, shadow_unknown) if intent == "unknown" else (per_intent, shadow_per_intent)
        d, s = group[:want_d], group[want_d:want_d + want_s]
        decision += d
        shadow += s
        short = (want_d - len(d)) + (want_s - len(s))
        complete &= short == 0
        counts[intent] = {**dict(per_intent_stage[intent]), "kept": len(group), "decision": len(d), "shadow": len(s),
                          "unused": max(0, len(group) - want_d - want_s), "short": short}

    def with_ids(rows: list[dict], prefix: str) -> list[dict]:
        return [{"id": f"{prefix}-{n:04d}", **r} for n, r in enumerate(rows, 1)]

    meta = {"contract": bench.CONTRACT_VERSION, "complete": complete,
            "semantic_filter": "run" if sem is not None else "NOT RUN (--semantic skip)"}
    return {
        "decision": {**meta, "kind": "decision set", "items": with_ids(decision, "c2d")},
        "shadow": {**meta, "kind": "shadow set", "items": with_ids(shadow, "c2s")},
        "report": {**meta, "inputs_sha256": {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files.items()},
                   "sealed_fingerprint_items": len((sealed or {}).get("items", [])),
                   "known_sentences": {"training": len(train), "holdout": len(holdout)},
                   "max_similarity": max_similarity, "max_cosine": max_cosine if sem is not None else None,
                   "seed": seed, "wanted": {"per_intent": per_intent, "unknown": unknown,
                                            "shadow_per_intent": shadow_per_intent, "shadow_unknown": shadow_unknown},
                   "dropped": {r: dropped.get(r, 0) for r in REASONS}, "per_intent": counts,
                   "decision_items": len(decision), "shadow_items": len(shadow)},
        "fingerprint": {"schema": FINGERPRINT_SCHEMA, "kind": "vyvy holdout fingerprint",
                        "normalise": "tools/check_holdout_leak.py normalise; hashes are SHA-256, the first 16 hex "
                                     "digits for words and character trigrams",
                        "items": [fingerprint_item(r["text"]) for r in decision + shadow]},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the contract 2 decision and shadow sets from blind candidates")
    p.add_argument("--candidates", type=Path, required=True, help="folder outside the repository")
    p.add_argument("--out", type=Path, required=True, help="folder outside the repository")
    p.add_argument("--sealed-fingerprint", type=Path, help="fingerprint of the sealed sets (hashes only)")
    p.add_argument("--max-similarity", type=float, default=0.8)
    p.add_argument("--max-cosine", type=float, default=0.85)
    p.add_argument("--semantic", choices=("model", "skip"), default="model",
                   help="skip only to try the tool; a set built without it is marked so")
    p.add_argument("--seed", default="contract-2")
    p.add_argument("--per-intent", type=int, default=40)
    p.add_argument("--unknown", type=int, default=160)
    p.add_argument("--shadow-per-intent", type=int, default=20)
    p.add_argument("--shadow-unknown", type=int, default=80)
    args = p.parse_args(argv)
    for name in ("candidates", "out"):
        if not outside_repo(getattr(args, name)):
            print(f"--{name} must be outside the repository", file=sys.stderr)
            return 2
    sealed = json.loads(args.sealed_fingerprint.read_text(encoding="utf-8")) if args.sealed_fingerprint else None
    if sealed is not None and sealed.get("schema") != FINGERPRINT_SCHEMA:
        print(f"sealed fingerprint: schema {sealed.get('schema')} is not {FINGERPRINT_SCHEMA}", file=sys.stderr)
        return 2
    try:
        result = build(args.candidates, sealed, max_similarity=args.max_similarity, max_cosine=args.max_cosine,
                       semantic=args.semantic == "model", seed=args.seed, per_intent=args.per_intent,
                       unknown=args.unknown, shadow_per_intent=args.shadow_per_intent,
                       shadow_unknown=args.shadow_unknown)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("decision", "shadow", "report", "fingerprint"):
        (args.out / f"{name}.json").write_text(json.dumps(result[name], ensure_ascii=False, indent=1) + "\n",
                                               encoding="utf-8")
    r = result["report"]
    print(f"decision set {r['decision_items']}, shadow set {r['shadow_items']}, complete: {r['complete']}, "
          f"semantic filter: {r['semantic_filter']}")
    print("dropped: " + ", ".join(f"{k} {v}" for k, v in r["dropped"].items() if v))
    for intent, c in r["per_intent"].items():
        if c["short"]:
            print(f"  {intent}: short by {c['short']} (kept {c['kept']})")
    return 0 if r["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
