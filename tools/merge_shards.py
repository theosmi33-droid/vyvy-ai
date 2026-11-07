#!/usr/bin/env python3
"""Join the outputs of a sweep that ran as several processes (one per GPU).

Each shard directory holds what training/train_lora.py --shard i/n wrote:
manifest.json, sweep.json and adapter/. The shard whose chosen config is best
(highest validation accuracy, then lower validation loss, as pick_best) wins:
its adapter and manifest go to --out, the manifest's sweep table lists the
configs of all shards, and "shards" records each shard's choice. Collect then
sees the same layout as from a single process.

    python tools/merge_shards.py /kaggle/working/shard0 /kaggle/working/shard1 --out /kaggle/working
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def rank_key(manifest: dict) -> tuple:
    acc = manifest.get("val_accuracy")
    losses = [h.get("eval_loss") for h in manifest.get("validation", []) if h.get("eval_loss") is not None]
    return (-(acc if acc is not None else -1), min(losses) if losses else float("inf"))


def merge(shards: list[Path], out: Path) -> dict:
    found = [(d, json.loads((d / "manifest.json").read_text(encoding="utf-8")))
             for d in shards if (d / "manifest.json").exists()]
    if not found:
        raise SystemExit("no shard finished with a manifest")
    best_dir, best = min(found, key=lambda x: rank_key(x[1]))
    sweep = [dict(row, shard=m.get("shard")) for _, m in found for row in m.get("sweep", [])]
    manifest = {**best, "sweep": sweep,
                "shards": [{"shard": m.get("shard"), "dir": d.name, "val_accuracy": m.get("val_accuracy"),
                            "rank": m.get("rank"), "lr": m.get("lr"), "seed": m.get("chosen_seed")}
                           for d, m in found],
                "shards_missing": [d.name for d in shards if not (d / "manifest.json").exists()]}
    out.mkdir(parents=True, exist_ok=True)
    target = out / best.get("adapter_dir", "adapter")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(best_dir / best.get("adapter_dir", "adapter"), target)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    (out / "sweep.json").write_text(json.dumps(sweep, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Join sweep shards into one adapter output")
    p.add_argument("shards", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    m = merge(args.shards, args.out)
    print(json.dumps({k: m[k] for k in ("name", "rank", "lr", "val_accuracy", "shards", "shards_missing")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
