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


def build(manifest: dict, verdict: dict, gguf_key: str, run_id: str, models: dict) -> dict:
    gguf = models["gguf"][gguf_key]
    return {
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
            "holdout": "data/holdout_blind_v2.json",
            "rule": "exact McNemar p < 0.05, b >= 2c, per-intent tolerance, no loss on unknown",
            "n": verdict["n"], "b_fixed": verdict["b_fixed"], "c_broken": verdict["c_broken"],
            "p_value": verdict["p_value"], "baseline_correct": verdict["base_correct"],
            "adapter_correct": verdict["adapter_correct"], "reason": verdict["reason"],
            "bench_lora_run": run_id,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write the release manifest of a measured adapter")
    p.add_argument("manifest", type=Path)
    p.add_argument("verdict", type=Path)
    p.add_argument("--gguf-key", required=True, help="key in models.json the benchmark ran on")
    p.add_argument("--run-id", required=True)
    p.add_argument("--models", type=Path, default=ROOT / "models.json")
    p.add_argument("--out", type=Path, default=Path("adapter_manifest.json"))
    args = p.parse_args(argv)
    record = build(json.loads(args.manifest.read_text(encoding="utf-8")),
                   json.loads(args.verdict.read_text(encoding="utf-8")),
                   args.gguf_key, args.run_id, json.loads(args.models.read_text(encoding="utf-8")))
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(args.out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
