#!/usr/bin/env python3
"""Take the teacher's sentences (training/synthesize.py output) into data/train/.

Runs on the GitHub runner after the synthesis kernel finished. Drops every
sentence at or above --max-similarity (default 0.8, as the generator; the lock
is 0.85) to any holdout sentence, and every sentence already in the training
data, then checks the output contract. Writes

    data/train/vyvy_synth_<script hash>_<round>.jsonl   the sentences
    data/train/vyvy_synth_<script hash>_<round>.json    teacher, counts, what was dropped

and prints the path of the .jsonl. An existing file for the same script is
left alone (exit code 3): one synthesis per round (ISO week) is taken in.

    python tools/import_synth.py collect/synth
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402
import check_holdout_leak as lock  # noqa: E402


def target(manifest: dict, train_dir: Path, round_: str = "") -> Path:
    round_ = round_ or manifest.get("round") or "manual"
    if not re.match(r"^[0-9A-Za-z-]{1,20}$", round_):
        raise SystemExit(f"invalid round {round_!r}")
    return train_dir / f"vyvy_synth_{manifest['script_sha256'][:9]}_{round_}.jsonl"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Import teacher sentences into data/train/")
    p.add_argument("folder", type=Path, help="kernel output with synth.jsonl and synth_manifest.json")
    p.add_argument("--train-dir", type=Path, default=lock.TRAIN_DIR)
    p.add_argument("--max-similarity", type=float, default=0.8)
    p.add_argument("--round", default="", help="round name if the run did not record one")
    args = p.parse_args(argv)

    manifest = json.loads((args.folder / "synth_manifest.json").read_text(encoding="utf-8"))
    out = target(manifest, args.train_dir, args.round)
    manifest["round"] = out.stem.rsplit("_", 1)[-1]
    if list(args.train_dir.glob(f"vyvy_synth_*_{manifest['round']}.jsonl")):
        print(f"{out} exists; nothing to import", file=sys.stderr)
        return 3
    intents = bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))
    holdout = [i for h in lock.HOLDOUTS if h.exists() for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    hold_norm = [(h["id"], lock.normalise(h["text"])) for h in holdout]
    seen = {lock.normalise(t) for _, t in lock.load_train(sorted(args.train_dir.glob("*.jsonl")))}

    rows, near_holdout, duplicates, invalid = [], 0, 0, 0
    for line in (args.folder / "synth.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        row = {"text": r["text"], "intent": r["intent"], "timeframe": r.get("timeframe"), "entity": r.get("entity")}
        answer = json.dumps({**{k: row[k] for k in ("intent", "timeframe", "entity")}, "confidence": 0.9})
        if bench.parse_output(answer, intents) is None:
            invalid += 1
            continue
        key = lock.normalise(row["text"])
        if not key or key in seen:
            duplicates += 1
            continue
        if max(max(lock.similarity(key, h)) for _, h in hold_norm) >= args.max_similarity:
            near_holdout += 1
            continue
        seen.add(key)
        rows.append(row)

    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    record = {**manifest, "imported": len(rows), "dropped_near_holdout": near_holdout,
              "dropped_duplicate": duplicates, "dropped_invalid": invalid, "max_similarity": args.max_similarity,
              "per_intent_imported": {i: sum(r["intent"] == i for r in rows) for i in intents}}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(out)
    print(json.dumps(record, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
