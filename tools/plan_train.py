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

# Same effective batch (16) for both; 7B uses small device batches so the 4-bit
# model, activations and optimizer fit a 16 GB T4.
#
# "where" decides the GPU: 1.5B runs on one free Colab T4 (train-colab.yml), so
# the weekly Kaggle quota stays with 7B and the teacher synthesis. On Colab it
# trains one config only — rank 16, lr 2e-4 won every 1.5B sweep so far — within
# 60 minutes, because a free Colab session is not held for long.
PLANS = {
    "Qwen/Qwen2.5-1.5B-Instruct": {"size": "1.5b", "budget": 60, "sweep": "rank=16;lr=2e-4",
                                   "batch": 8, "grad_accum": 2, "gpus": 1, "gpus_after_7b": False,
                                   "where": "colab"},
    "Qwen/Qwen2.5-7B-Instruct": {"size": "7b", "budget": 600, "sweep": "rank=16,32;lr=2e-4,1e-4,3e-4",
                                 "batch": 2, "grad_accum": 8, "gpus": 1, "gpus_after_7b": True,
                                 "where": "kaggle"},
}
#: Workflow that trains a plan, by where it runs.
WORKFLOW = {"kaggle": "train.yml", "colab": "train-colab.yml"}
# 7B first: it needs the longer budget and decides whether two GPUs are safe
# (see seven_b_passed); 1.5B runs on Colab and does not wait for it.
ORDER = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]


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


def inputs(base: str) -> tuple[dict, str]:
    """What decides whether a base model needs training: training files and prompt."""
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted((ROOT / "data" / "train").glob("*.jsonl"))}
    prompt = hashlib.sha256(bench.PROMPTS[bench.prompt_for(base)].encode("utf-8")).hexdigest()
    return files, prompt


def trained_release(repo: str, base: str) -> str | None:
    """Tag of a release (accepted or candidate) trained the way this plan trains today.

    Decided by the release manifest, not the name: same base, same training files,
    same prompt and the same sweep budget. Trainer bug fixes therefore do not
    retrain a model, while new data, a new prompt or a new sweep plan do.
    """
    import subprocess

    files, prompt = inputs(base)
    budget = PLANS[base]["budget"]
    out = subprocess.run(["gh", "release", "list", "--repo", repo, "--limit", "100", "--json", "tagName"],
                         capture_output=True, text=True)
    for tag in [r["tagName"] for r in json.loads(out.stdout or "[]")]:
        got = subprocess.run(["gh", "release", "download", tag, "--repo", repo, "--pattern", "manifest.json",
                              "--output", "-"], capture_output=True, text=True)
        try:
            m = json.loads(got.stdout)
        except json.JSONDecodeError:
            continue
        if (m.get("base") == base and m.get("train_files") == files and m.get("prompt_sha256") == prompt
                and m.get("time_budget_min") == budget):
            return tag
    return None


def seven_b_passed(repo: str) -> bool:
    """True once a 7B run trained an adapter (any release vyvy-7b-…, candidate or not).

    Two GPUs per run (one sweep shard each) stay off until then: the 7B runs of
    07.11. failed on one GPU (bitsandbytes), and a failure on two GPUs would not
    tell whether the library or the split is to blame.
    """
    import subprocess

    out = subprocess.run(["gh", "release", "list", "--repo", repo, "--limit", "100", "--json", "tagName"],
                         capture_output=True, text=True)
    return any(r["tagName"].startswith("vyvy-7b-") for r in json.loads(out.stdout or "[]"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plan a training run")
    ap.add_argument("base")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--repo", help="also report a release already trained on today's data and prompt")
    args = ap.parse_args(argv)
    p = plan(args.base, args.epochs)
    if args.repo:
        p["trained"] = trained_release(args.repo, args.base)
        if p["gpus"] == 1 and PLANS[args.base].get("gpus_after_7b") and seven_b_passed(args.repo):
            p["gpus"] = 2
    print(json.dumps(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
