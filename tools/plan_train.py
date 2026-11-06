#!/usr/bin/env python3
"""Plan a training run: size, sweep, time budget and the adapter name.

The name is a hash of everything that decides the result (training data, the
prompt for this model size, the trainer, sweep, budget, epochs), so the same
input is never trained twice and any change trains anew. Used by train.yml and
by the Collect scheduler, so both agree on what is missing.

    python tools/plan_train.py Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402

PLANS = {
    "Qwen/Qwen2.5-1.5B-Instruct": {"size": "1.5b", "budget": 240, "sweep": "rank=16,8,32;lr=2e-4,1e-4,5e-4"},
    "Qwen/Qwen2.5-7B-Instruct": {"size": "7b", "budget": 360, "sweep": "rank=16,32;lr=2e-4,1e-4"},
}
ORDER = list(PLANS)


def plan(base: str, epochs: float = 4) -> dict:
    if base not in PLANS:
        raise SystemExit(f"unknown base model: {base}")
    p = dict(PLANS[base], base=base, epochs=epochs)
    h = hashlib.sha256()
    for f in sorted((ROOT / "data" / "train").glob("*.jsonl")):
        h.update(f.read_bytes())
    h.update(bench.PROMPTS[bench.prompt_for(base)].encode("utf-8"))
    h.update((ROOT / "training" / "train_lora.py").read_bytes())
    h.update(f"{p['sweep']} {p['budget']} {epochs:g}".encode())
    p["name"] = f"vyvy-{p['size']}-d{h.hexdigest()[:8]}"
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plan a training run")
    ap.add_argument("base")
    ap.add_argument("--epochs", type=float, default=4)
    args = ap.parse_args(argv)
    print(json.dumps(plan(args.base, args.epochs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
