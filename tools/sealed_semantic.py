#!/usr/bin/env python3
"""Semantic check of private sealed sets against everything public; runs in a Kaggle CPU kernel.

The sealed files are only attached to the kernel (private Kaggle datasets); the
output holds ids, scores and the kind of the closer sentence (train, holdout,
probe, teacher), never a text. Same sentence model and revision as
tools/check_semantic_leak.py; an id at or above --fail fails the set.

    python tools/sealed_semantic.py /kaggle/input/.../sealed_create-v1.json ... --out semantic_result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import audit_check  # noqa: E402
import check_semantic_leak as sem  # noqa: E402


def recorded(path: Path, data: dict) -> None:
    """The file must be the recorded, unchanged set (data/sealed_sets.json), as in the measuring kernel."""
    version = path.stem.removeprefix("sealed_")
    digest = hashlib.sha256(json.dumps(data["items"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    rows = [r for r in json.loads((ROOT / "data" / "sealed_sets.json").read_text(encoding="utf-8"))
            if r["version"] == version]
    if not rows or rows[-1]["items_sha256"] != digest or data.get("items_sha256") != digest:
        raise SystemExit(f"{version}: the sealed set differs from its record in data/sealed_sets.json")


def versions() -> dict:
    import platform

    import sentence_transformers
    import torch
    import transformers

    return {"python": platform.python_version(), "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__, "transformers": transformers.__version__,
            "model": sem.MODEL, "revision": sem.REVISION}


def check(files: list[Path], report: float, fail: float) -> dict:
    from sentence_transformers import SentenceTransformer

    data = {path: json.loads(path.read_text(encoding="utf-8")) for path in files}
    for path, d in data.items():
        recorded(path, d)
    known = audit_check.known_sentences()
    model = SentenceTransformer(sem.MODEL, revision=sem.REVISION, device="cpu")
    k = model.encode([t for _, t, _ in known], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    out = {"model": sem.MODEL, "report": report, "fail": fail, "versions": versions(), "sets": {}}
    for path in files:
        items = data[path]["items"]
        e = model.encode([i["text"] for i in items], batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True)
        sim = e @ k.T
        best, which = sim.max(axis=1), sim.argmax(axis=1)
        # "where": file:line of a training sentence (or a holdout id) - on a failing hit the
        # training line is dropped, never the sealed sentence. No text.
        hits = sorted(({"id": items[n]["id"], "score": round(float(best[n]), 3), "kind": known[int(which[n])][0],
                        "where": known[int(which[n])][2]}
                       for n in range(len(items)) if best[n] >= report), key=lambda h: -h["score"])
        out["sets"][path.stem.removeprefix("sealed_")] = {
            "n": len(items), "hits": hits, "failing": sum(h["score"] >= fail for h in hits),
            "median": round(float(sorted(best)[len(best) // 2]), 3)}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Semantic check of sealed sets; ids and scores out")
    p.add_argument("files", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=float, default=0.90)
    p.add_argument("--fail", type=float, default=0.97)
    args = p.parse_args(argv)
    result = check(args.files, args.report, args.fail)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 1 if any(s["failing"] for s in result["sets"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
