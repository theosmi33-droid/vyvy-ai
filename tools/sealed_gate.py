#!/usr/bin/env python3
"""Sealed check before training: today's training data must have passed it.

Training data can come close to a sentence of a privately written sealed set
(create-v<N>, inject-v<N>), and Data check cannot see those sets. So every data
state is checked once by the Sealed check workflow (semantic check in a Kaggle CPU
kernel); its result is kept as the artifact sealed-check-<data hash>. Collect and
Train start a GPU run only when that artifact exists and says the check passed; when
it is missing they start the check (--start) and wait. A failed check holds
training: the hit names the training line (file:line), which is dropped, never the
sealed sentence.

Without a recorded create/inject set (data/sealed_sets.json) there is nothing to check.

    python tools/sealed_gate.py --repo OWNER/REPO [--start]   # exit 0 = train, 3 = wait or held
    python tools/sealed_gate.py --hash                        # name part of today's data state
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train"
RECORD = ROOT / "data" / "sealed_sets.json"
WORKFLOW = "sealed-check.yml"


def data_hash(train: Path = TRAIN) -> str:
    h = hashlib.sha256()
    for f in sorted(train.glob("*.jsonl")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def versions(record: Path = RECORD) -> list[str]:
    rows = json.loads(record.read_text(encoding="utf-8")) if record.exists() else []
    return [r["version"] for r in rows if r["version"].startswith(("create-", "inject-"))]


def artifact(digest: str) -> str:
    return f"sealed-check-{digest}"


def verdict(result: dict) -> tuple[bool, str]:
    if result.get("error"):
        return False, f"check failed in the kernel: {result['error'].get('kind')}: {result['error'].get('line')}"
    failing = {v: s["failing"] for v, s in result.get("sets", {}).items() if s.get("failing")}
    if failing:
        where = [h["where"] for s in result["sets"].values() for h in s.get("hits", [])
                 if h.get("score", 0) >= result.get("fail", 0.97) and h.get("kind") == "train"]
        return False, f"training data too close to a sealed set {failing}; drop the training lines {where}"
    return True, "passed"


def gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def status(repo: str, digest: str, run=gh) -> tuple[str, str]:
    """("ok" | "held" | "running" | "missing", detail)."""
    name = artifact(digest)
    found = run("api", f"repos/{repo}/actions/artifacts?name={name}", "--jq",
                "[.artifacts[] | select(.expired == false) | .workflow_run.id][0] // empty")
    run_id = found.stdout.strip()
    if run_id:
        with tempfile.TemporaryDirectory() as tmp:
            got = run("run", "download", run_id, "--repo", repo, "--name", name, "--dir", tmp)
            path = Path(tmp) / "semantic_result.json"
            if got.returncode != 0 or not path.exists():
                return "held", f"artifact {name} of run {run_id} could not be read"
            ok, why = verdict(json.loads(path.read_text(encoding="utf-8")))
        return ("ok" if ok else "held"), f"{why} (run {run_id})"
    busy = run("run", "list", "--repo", repo, "--workflow", WORKFLOW, "--json", "status",
               "--jq", '[.[] | select(.status != "completed")] | length')
    if busy.stdout.strip() not in ("", "0"):
        return "running", "a Sealed check is running"
    return "missing", f"no Sealed check for data state {digest} yet"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Has today's training data passed the Sealed check?")
    p.add_argument("--repo")
    p.add_argument("--start", action="store_true", help="start the Sealed check when it is missing")
    p.add_argument("--hash", action="store_true", help="print the data state's hash and exit")
    args = p.parse_args(argv)
    digest = data_hash()
    if args.hash:
        print(digest)
        return 0
    if not args.repo:
        p.error("--repo is required")
    sets = versions()
    if not sets:
        print(json.dumps({"state": "ok", "detail": "no create/inject sealed set recorded"}))
        return 0
    state, detail = status(args.repo, digest)
    if state == "missing" and args.start:
        started = gh("workflow", "run", WORKFLOW, "--repo", args.repo, "--ref", "main", "-f", f"versions={' '.join(sets)}")
        detail += "; started" if started.returncode == 0 else f"; could not start: {started.stderr.strip()[-200:]}"
    print(json.dumps({"state": state, "detail": detail, "data": digest}))
    return 0 if state == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
