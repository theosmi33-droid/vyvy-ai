#!/usr/bin/env python3
"""Write adapter_manifest.json: one fixed record per measured adapter release.

Joins what training wrote (manifest.json), what bench-lora decided
(verdict.json) and what the benchmark ran on (the pinned base GGUF from
models.json), so a loader can check an adapter before it applies it: the
adapter file's checksum, the exact base weights it was measured with, the
training commit and data, and the statistical decision. bench-lora attests the
file with GitHub's build provenance, which ties it to this repository and run.

    python tools/release_manifest.py adapter/manifest.json verdict.json \
        --gguf-key qwen2.5-1.5b-instruct-q8_0 --run-id 123 --out adapter_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = 1


DECISION_SET = "data/holdout_decision_c2.json"
RULE = ("exact McNemar p < 0.05, b >= 2c, an intent fails above its tolerance if one-sided binomial "
        "p < 0.05 or its net loss is at least 10 %, no loss on unknown")


def build(manifest: dict, verdict: dict, gguf_key: str, run_id: str, models: dict,
          remeasure: dict | None = None, holdout: str = DECISION_SET) -> dict:
    gguf = models["gguf"][gguf_key]
    record = {
        "schema": SCHEMA,
        "adapter": manifest["adapter"],
        "adapter_sha256": manifest["adapter_sha256"],
        "status": "accepted" if verdict["accepted"] else "rejected",
        "base_model": manifest["base"],
        "base_revision": manifest["base_revision"],
        "measured_with": {"gguf": gguf_key, "repo": gguf["repo"], "revision": gguf["revision"],
                          "files_sha256": gguf["files"]},
        # The adapter is applied at full strength (scale 1.0); that is the only scale measured.
        "lora_scale": 1.0,
        "training_git_commit": manifest.get("training_git_commit"),
        "train_files_sha256": manifest["train_files"],
        "prompt": manifest["prompt"],
        "prompt_sha256": manifest["prompt_sha256"],
        "llama_cpp_commit": manifest.get("llama_cpp_commit"),
        "decision": {
            "holdout": holdout,
            "rule": RULE,
            "n": verdict["n"], "b_fixed": verdict["b_fixed"], "c_broken": verdict["c_broken"],
            "p_value": verdict["p_value"], "baseline_correct": verdict["base_correct"],
            "adapter_correct": verdict["adapter_correct"], "reason": verdict["reason"],
            "bench_lora_run": run_id,
        },
    }
    # Optional within schema 1: the output contract the adapter was trained for (contract 2 on).
    # A manifest without it is contract 1; an application loads only the contract it speaks,
    # with exactly this grammar.
    if manifest.get("contract_version"):
        record["contract_version"] = manifest["contract_version"]
        record["grammar_sha256"] = manifest["grammar_sha256"]
    # Optional, so readers of schema 1 (tools/verify_adapter.py) are not affected: a decided
    # release measured again under today's rule says so, next to its first decision.
    if remeasure:
        record["remeasure"] = remeasure
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write the release manifest of a measured adapter")
    p.add_argument("manifest", type=Path)
    p.add_argument("verdict", type=Path)
    p.add_argument("--gguf-key", required=True, help="key in models.json the benchmark ran on")
    p.add_argument("--run-id", required=True)
    p.add_argument("--models", type=Path, default=ROOT / "models.json")
    p.add_argument("--out", type=Path, default=Path("adapter_manifest.json"))
    p.add_argument("--holdout", default=DECISION_SET, help="the set the decision was made on")
    p.add_argument("--remeasured-on", help="date of a remeasurement of an already decided release")
    p.add_argument("--first-decision", help="with --remeasured-on: the release's first decision, e.g. its date and holdout")
    args = p.parse_args(argv)
    remeasure = None
    if args.remeasured_on:
        remeasure = {"remeasured_on": args.remeasured_on,
                     "rule": f"{args.holdout} decides, the contract 2 shadow set can veto",
                     "first_decision": args.first_decision}
    record = build(json.loads(args.manifest.read_text(encoding="utf-8")),
                   json.loads(args.verdict.read_text(encoding="utf-8")),
                   args.gguf_key, args.run_id, json.loads(args.models.read_text(encoding="utf-8")), remeasure,
                   args.holdout)
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(args.out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
