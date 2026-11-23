#!/usr/bin/env python3
"""Build the contract 2 decision set and shadow set from written candidates, blind.

Input is a folder OUTSIDE this repository (the sentences must never pass through it
before they are a holdout). It holds one or more rounds; each round is a candidate file
and two blind assignments (--round NAME:CANDIDATES:LABELS_A:LABELS_B, default: round 1
kandidaten.jsonl with labels_1/labels_2, round 2 kandidaten_hart.jsonl with
labels_hart_1/labels_hart_2):

    candidates  {"id", "writer", "text", "expected_intent"}
    labels      {"id", "intent", "sicher"}     (by id; without id, by text)

A candidate is kept only if

1. its intent is one of the 25 of contract 2 and it is no duplicate of an earlier one,
2. both assignments exist, both equal the writer's intent and both are marked sure,
3. it is below --max-similarity (0.8) in characters and in words (tools/check_holdout_leak.py
   measures, after the same normalisation) to every training sentence, every holdout ever
   used, the probes, the prompt and teacher examples, the sealed fingerprint and every
   candidate kept before it (near copies across writers and rounds),
4. its meaning is below --max-cosine (0.85, the sentence model of tools/check_semantic_leak.py)
   to the same sentences, unless --semantic skip, which the report and the sets then say.

The kept candidates are split per intent and round, deterministically (order of
SHA-256(--seed + normalised text), then in turn over the writers so each writer of a round
contributes evenly). Quotas per set and round (default: decision 14 from round 2 and 26
from round 1 per intent, unknown 60 and 100; shadow 7 and 13, unknown 30 and 50). A round
that falls short is filled from the next one in the list, and the report says how many. No
sentence is in both sets. After both sets, the next --reserve (6) per intent and round
(--reserve-unknown 20) go to a reserve: the semantic check runs in the cloud
(tools/holdout_semantic.py, Holdout semantic workflow) and replaces a sentence it rejects
by the next reserve sentence of the same intent and round, so a hit there needs no new
writing round. Written to --out (also outside the repository): decision.json, shadow.json
and reserve.json (items with id, set, text, expected_intent, origin), report.json (numbers only:
available per intent and round, drops per reason, taken per round, fills; no sentence) and
fingerprint.json (per sentence only hashes of the normalised text, its words and its
character trigrams). No sentence is printed. Exit 0 when every quota is met, 3 when one is
short (the sets are written and marked incomplete), 2 on bad input.

Landing a set in data/ and recording its fingerprint need the user's word; this tool
never writes into the repository.

The fingerprint subcommand writes a fingerprint in the format this tool reads, from any
sentence files (JSON with items[].text, or JSONL with text), for sets whose text must not
be shown to the one who builds the next set:

    python tools/holdout_build.py --candidates /path/outside --out /path/outside/built \
        --sealed-fingerprint /path/to/fingerprint.json
    python tools/holdout_build.py fingerprint sealed_a.json sealed_b.json --out fingerprint.json
"""

from __future__ import annotations

import argparse
import difflib
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

FINGERPRINT_SCHEMA = 1
DEFAULT_ROUNDS = ["1:kandidaten.jsonl:labels_1.jsonl:labels_2.jsonl",
                  "2:kandidaten_hart.jsonl:labels_hart_1.jsonl:labels_hart_2.jsonl"]
# Per set: (quota per intent, quota for unknown), each a list of (round, n) in fill order.
DEFAULT_QUOTAS = {"decision": ("2:14,1:26", "2:60,1:100"), "shadow": ("2:7,1:13", "2:30,1:50")}
DEFAULT_RESERVE = (6, 20)  # per intent and round; per round for unknown
LOCAL_ONLY = "NOT RUN (local; the semantic check runs in the cloud, tools/holdout_semantic.py)"
REASONS = ("intent_not_in_contract", "duplicate", "blind_missing", "blind_disagrees", "blind_unsure",
           "near_training", "near_holdout", "near_probe", "near_template", "near_sealed", "near_candidate",
           "semantic_training", "semantic_holdout", "semantic_probe", "semantic_template", "semantic_sealed",
           "semantic_candidate")
CAVEAT = ("Writers and blind assigners are the same language model; their agreement shows that the labels are "
          "consistent, not that the sentences are hard. Separation shows only in the first measurement of the "
          "base models: a base model without adapter above 95 % means the set is too easy.")


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
    """Character similarity (difflib ratio) and word overlap against known texts, cheap bounds first,
    plus fingerprint items known only by hashes (exact text, word and trigram overlap)."""

    def __init__(self, threshold: float):
        self.t = threshold
        self.texts: list[tuple[str, str, set[str]]] = []
        self.prints: list[tuple[str, str, set[str], set[str]]] = []

    def add_text(self, kind: str, text: str) -> None:
        n = norm(text)
        self.texts.append((kind, n, set(n.split())))

    def add_fingerprint(self, kind: str, item: dict) -> None:
        self.prints.append((kind, item["sha256"], set(item["words"]), set(item["trigrams"])))

    def near(self, text: str) -> str | None:
        n = norm(text)
        words = set(n.split())
        for kind, other, other_words in self.texts:
            if jaccard(words, other_words) >= self.t:
                return kind
            if not n or not other or 2 * min(len(n), len(other)) / (len(n) + len(other)) < self.t:
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

    def add_vectors(self, kind: str, vectors: list[list[float]]) -> None:
        if not vectors:
            return
        v = self.np.asarray(vectors, dtype="float32")
        self.vectors = v if self.vectors is None else self.np.vstack([self.vectors, v])
        self.kinds += [kind] * len(vectors)

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


def known_sentences() -> dict[str, list[str]]:
    """Everything a new set must stay away from, by kind."""
    train = [t for _, t in lock.load_train(sorted(lock.TRAIN_DIR.glob("*.jsonl")))]
    holdout = [i["text"] for h in lock.HOLDOUTS if h.exists() for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    probes = [i["text"] for p in sorted((ROOT / "data").glob("probe_*.json"))
              for i in json.loads(p.read_text(encoding="utf-8"))["items"]]
    templates = [t for path in (lock.SYNTH, lock.PATTERNS) for _, t in lock.load_prompt_examples(path)]
    templates += re.findall(r'Question: "([^"]+)"', "\n".join([*bench.PROMPTS.values(), *bench.PROMPTS_V1.values()]))
    return {"training": train, "holdout": holdout, "probe": probes, "template": templates}


def rank(seed: str, text: str) -> str:
    return hashlib.sha256(f"{seed}\n{norm(text)}".encode("utf-8")).hexdigest()


def parse_quota(spec: str) -> list[tuple[str, int]]:
    out = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, n = part.partition(":")
        out.append((name.strip(), int(n)))
    return out


def writer_order(rows: list[dict], seed: str) -> list[dict]:
    """Deterministic order that takes the writers in turn, each writer's sentences by rank."""
    by_writer: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: rank(seed, r["text"])):
        by_writer.setdefault(str(r.get("writer", "")), []).append(r)
    order, queues = [], [by_writer[w] for w in sorted(by_writer)]
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    return order


def load_round(folder: Path, spec: str) -> tuple[str, list[dict], dict, dict, dict]:
    name, cand, la, lb = spec.split(":")
    paths = {"candidates": folder / cand, "labels_a": folder / la, "labels_b": folder / lb}
    for what, path in paths.items():
        if not path.is_file():
            raise ValueError(f"round {name}: {path.name} missing in the candidate folder")
    rows = read_jsonl(paths["candidates"])

    def labels(path: Path) -> dict:
        out = {}
        for r in read_jsonl(path):
            out[("id", r["id"]) if "id" in r else ("text", key(r["text"]))] = (r.get("intent"), r.get("sicher", True))
        return out

    hashes = {f"round {name} {k}": hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    return name, rows, labels(paths["labels_a"]), labels(paths["labels_b"]), hashes


def build(folder: Path, sealed: dict | None, *, rounds: list[str] = DEFAULT_ROUNDS,
          quotas: dict[str, tuple[str, str]] = DEFAULT_QUOTAS, reserve: tuple[int, int] = DEFAULT_RESERVE,
          max_similarity: float = 0.8,
          max_cosine: float = 0.85, semantic: bool = True, seed: str = "contract-2",
          known: dict[str, list[str]] | None = None, semantic_model=None) -> dict:
    intents = bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))
    loaded = [load_round(folder, spec) for spec in rounds]
    known = known if known is not None else known_sentences()

    lex = Lexical(max_similarity)
    for kind, texts in known.items():
        for t in texts:
            lex.add_text(f"near_{kind}", t)
    for item in (sealed or {}).get("items", []):
        lex.add_fingerprint("near_sealed", item)
    sem = None
    if semantic:
        sem = semantic_model or Semantic(max_cosine)
        for kind, texts in known.items():
            sem.add(f"semantic_{kind}", texts)
        # A fingerprint holds no text; its meaning can only be compared if it carries vectors of the same model.
        sem.add_vectors("semantic_sealed", [i["embedding"] for i in (sealed or {}).get("items", []) if i.get("embedding")])

    dropped: Counter[str] = Counter()
    stages: dict[tuple[str, str], Counter] = {}
    kept: dict[tuple[str, str], list[dict]] = {}
    seen: set[str] = set()
    hashes: dict[str, str] = {}
    # Round by round in the order given, each in rank order, so "kept before it" is deterministic.
    for name, rows, la, lb, h in loaded:
        hashes.update(h)
        for c in sorted(rows, key=lambda r: rank(seed, key(r.get("text", "")))):
            text, intent = key(c.get("text", "")), c.get("expected_intent")
            if intent not in intents or not text:
                dropped["intent_not_in_contract"] += 1
                continue
            stage = stages.setdefault((intent, name), Counter())
            stage["candidates"] += 1
            n = norm(text)
            if n in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(n)
            ref = ("id", c["id"]) if "id" in c else ("text", text)
            a, b = la.get(ref), lb.get(ref)
            if a is None or b is None:
                dropped["blind_missing"] += 1
                continue
            if not a[0] == b[0] == intent:
                dropped["blind_disagrees"] += 1
                continue
            if a[1] is not True or b[1] is not True:
                dropped["blind_unsure"] += 1
                continue
            stage["blind_agree_sure"] += 1
            reason = lex.near(text)
            if not reason and sem is not None:
                reason = sem.near(text)
            if reason:
                dropped[reason] += 1
                continue
            stage["available"] += 1
            kept.setdefault((intent, name), []).append({"text": text, "expected_intent": intent,
                                                        "writer": c.get("writer", ""), "origin": f"round {name}"})
            lex.add_text("near_candidate", text)
            if sem is not None:
                sem.add("semantic_candidate", [text])

    order = {k: writer_order(v, seed) for k, v in kept.items()}
    taken = {k: 0 for k in order}
    sets: dict[str, list[dict]] = {"decision": [], "shadow": []}
    per_intent: dict[str, dict] = {i: {"available": {}, "taken": {"decision": {}, "shadow": {}}, "filled": {},
                                       "short": 0} for i in intents}
    for intent in intents:
        for (i, name), st in stages.items():
            if i == intent:
                per_intent[intent]["available"][f"round {name}"] = st.get("available", 0)
    complete = True
    for set_name in ("decision", "shadow"):
        regular, unknown_spec = quotas[set_name]
        for intent in intents:
            plan = parse_quota(unknown_spec if intent == "unknown" else regular)
            missing = 0
            for pos, (round_name, want) in enumerate(plan):
                pool = order.get((intent, round_name), [])
                start = taken.get((intent, round_name), 0)
                want_now = want + missing
                got = pool[start:start + want_now]
                taken[(intent, round_name)] = start + len(got)
                missing = want_now - len(got)
                if pos and len(got) > want:
                    per_intent[intent]["filled"][f"{set_name}: round {round_name} for earlier rounds"] = len(got) - want
                sets[set_name] += [{**r, "set": set_name} for r in got]
                per_intent[intent]["taken"][set_name][f"round {round_name}"] = len(got)
            per_intent[intent]["short"] += missing
            complete &= missing == 0
    sets["reserve"] = []
    for intent in intents:
        want = reserve[1] if intent == "unknown" else reserve[0]
        for (i, round_name), pool in sorted(order.items()):
            if i != intent:
                continue
            start = taken.get((i, round_name), 0)
            got = pool[start:start + want]
            taken[(i, round_name)] = start + len(got)
            sets["reserve"] += [{**r, "set": "reserve"} for r in got]
            per_intent[intent]["taken"].setdefault("reserve", {})[f"round {round_name}"] = len(got)

    def finish(rows: list[dict], prefix: str) -> list[dict]:
        return [{"id": f"{prefix}-{n:04d}", "set": r["set"], "text": r["text"], "expected_intent": r["expected_intent"],
                 "origin": r["origin"]} for n, r in enumerate(rows, 1)]

    meta = {"contract": bench.CONTRACT_VERSION, "complete": complete, "description": CAVEAT,
            "semantic_filter": "run" if sem is not None else LOCAL_ONLY,
            "checked_against": {"training": "data/train/*.jsonl", "holdouts": "every holdout ever used",
                                "probes": "data/probe_*.json", "templates": "prompt and teacher examples",
                                "sealed": (f"fingerprint, {len((sealed or {}).get('items', []))} sentences, by hashes"
                                           if sealed else "NOT CHECKED: no fingerprint given")}}
    decision, shadow = finish(sets["decision"], "c2d"), finish(sets["shadow"], "c2s")
    spare = finish(sets["reserve"], "c2r")
    return {
        "decision": {**meta, "kind": "decision set", "items": decision},
        "shadow": {**meta, "kind": "shadow set", "items": shadow},
        "reserve": {**meta, "kind": "reserve for the cloud semantic check; never measured", "items": spare},
        "report": {**meta, "inputs_sha256": hashes,
                   "sealed_fingerprint_items": len((sealed or {}).get("items", [])),
                   "known_sentences": {k: len(v) for k, v in known.items()},
                   "max_similarity": max_similarity, "max_cosine": max_cosine if sem is not None else None,
                   "seed": seed, "quotas": {k: {"per_intent": v[0], "unknown": v[1]} for k, v in quotas.items()},
                   "dropped": {r: dropped.get(r, 0) for r in REASONS}, "per_intent": per_intent,
                   "decision_items": len(decision), "shadow_items": len(shadow), "reserve_items": len(spare),
                   "taken_per_round": {s: dict(Counter(r["origin"] for r in rows))
                                       for s, rows in (("decision", decision), ("shadow", shadow))}},
        "fingerprint": {"schema": FINGERPRINT_SCHEMA, "kind": "vyvy holdout fingerprint",
                        "normalise": "tools/check_holdout_leak.py normalise; hashes are SHA-256, the first 16 hex "
                                     "digits for words and character trigrams",
                        "items": [fingerprint_item(r["text"]) for r in decision + shadow + spare]},
    }


def sentences_of(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        return [r["text"] for r in read_jsonl(path)]
    data = json.loads(path.read_text(encoding="utf-8"))
    return [i["text"] for i in (data["items"] if isinstance(data, dict) else data)]


def fingerprint_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="holdout_build.py fingerprint",
                                description="Write a fingerprint (hashes only) of sentence files")
    p.add_argument("files", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    texts = [t for f in args.files for t in sentences_of(f)]
    fp = {"schema": FINGERPRINT_SCHEMA, "kind": "vyvy holdout fingerprint",
          "normalise": "tools/check_holdout_leak.py normalise; hashes are SHA-256, the first 16 hex digits for "
                       "words and character trigrams",
          "sources_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in args.files},
          "items": [fingerprint_item(t) for t in texts]}
    args.out.write_text(json.dumps(fp, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(texts)} sentences, hashes only")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "fingerprint":
        return fingerprint_main(argv[1:])
    p = argparse.ArgumentParser(description="Build the contract 2 decision and shadow sets from blind candidates")
    p.add_argument("--candidates", type=Path, required=True, help="folder outside the repository")
    p.add_argument("--out", type=Path, required=True, help="folder outside the repository")
    p.add_argument("--round", action="append", dest="rounds",
                   help="NAME:CANDIDATES:LABELS_A:LABELS_B (repeatable; default: the two rounds of the v5 folder)")
    p.add_argument("--decision", default=DEFAULT_QUOTAS["decision"][0], help="per intent, e.g. 2:14,1:26")
    p.add_argument("--decision-unknown", default=DEFAULT_QUOTAS["decision"][1])
    p.add_argument("--shadow", default=DEFAULT_QUOTAS["shadow"][0])
    p.add_argument("--shadow-unknown", default=DEFAULT_QUOTAS["shadow"][1])
    p.add_argument("--reserve", type=int, default=DEFAULT_RESERVE[0], help="reserve per intent and round")
    p.add_argument("--reserve-unknown", type=int, default=DEFAULT_RESERVE[1])
    p.add_argument("--sealed-fingerprint", type=Path, help="fingerprint of the sealed sets (hashes only)")
    p.add_argument("--max-similarity", type=float, default=0.8)
    p.add_argument("--max-cosine", type=float, default=0.85)
    p.add_argument("--semantic", choices=("model", "skip"), default="model",
                   help="skip only to try the tool; a set built without it is marked so")
    p.add_argument("--seed", default="contract-2")
    args = p.parse_args(argv)
    for name in ("candidates", "out"):
        if not outside_repo(getattr(args, name)):
            print(f"--{name} must be outside the repository", file=sys.stderr)
            return 2
    sealed = json.loads(args.sealed_fingerprint.read_text(encoding="utf-8")) if args.sealed_fingerprint else None
    if sealed is not None and sealed.get("schema") != FINGERPRINT_SCHEMA:
        print(f"sealed fingerprint: schema {sealed.get('schema')} is not {FINGERPRINT_SCHEMA}", file=sys.stderr)
        return 2
    quotas = {"decision": (args.decision, args.decision_unknown), "shadow": (args.shadow, args.shadow_unknown)}
    try:
        result = build(args.candidates, sealed, rounds=args.rounds or DEFAULT_ROUNDS, quotas=quotas,
                       reserve=(args.reserve, args.reserve_unknown),
                       max_similarity=args.max_similarity, max_cosine=args.max_cosine,
                       semantic=args.semantic == "model", seed=args.seed)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("decision", "shadow", "reserve", "report", "fingerprint"):
        (args.out / f"{name}.json").write_text(json.dumps(result[name], ensure_ascii=False, indent=1) + "\n",
                                               encoding="utf-8")
    r = result["report"]
    print(f"decision set {r['decision_items']}, shadow set {r['shadow_items']}, reserve {r['reserve_items']}, "
          f"complete: {r['complete']}, semantic filter: {r['semantic_filter']}")
    print(f"sealed: {r['checked_against']['sealed']}")
    print("taken per round: " + json.dumps(r["taken_per_round"]))
    print("dropped: " + ", ".join(f"{k} {v}" for k, v in r["dropped"].items() if v))
    for intent, c in r["per_intent"].items():
        if c["short"] or c["filled"]:
            print(f"  {intent}: short {c['short']}, filled {c['filled']}")
    return 0 if r["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
