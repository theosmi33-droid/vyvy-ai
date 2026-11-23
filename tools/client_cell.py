#!/usr/bin/env python3
"""One cell of bench-client: check an adapter's contract, and turn a run into numbers.

    python tools/client_cell.py check-adapter adapter/manifest.json
    python tools/client_cell.py summary --platform "Linux container" --model M --cpus 2 --limit-gb 5.5 \
        --limited --result client.json --time time.txt --exit exit.txt --out cell.json

summary prints one Markdown table row and writes the cell as JSON: median and p95 ms per
question, share over 4 s, intent accuracy, peak memory, and whether the run stayed within
the class limit. In a limited Linux container the limit is enforced, so "within" means it
finished and was not killed for lack of memory; peak memory comes from /usr/bin/time. On
macOS and Windows nothing is limited: peak memory is what the benchmark read for its own
process (bench.py peak_memory_mb, read differently per platform), compared with the limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402

SLOW_MS = 4000


def check_adapter(manifest: dict) -> str | None:
    """Why the adapter does not fit this benchmark, or None."""
    grammar = hashlib.sha256(bench.DEFAULT_GRAMMAR.read_bytes()).hexdigest()
    version = manifest.get("contract_version", 1)
    if version != bench.CONTRACT_VERSION or manifest.get("grammar_sha256") != grammar:
        return (f"adapter is contract {version} (grammar {str(manifest.get('grammar_sha256'))[:12]}), the benchmark "
                f"measures contract {bench.CONTRACT_VERSION} ({grammar[:12]})")
    return None


def read_state(text: str) -> dict:
    out = dict(re.findall(r"(\w+)=(\S+)", text))
    return {"exit": int(out["exit"]) if out.get("exit", "").lstrip("-").isdigit() else None,
            "oom": out.get("oom") == "true"}


def peak_from_time(text: str) -> float | None:
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    return round(int(m.group(1)) / 1024 / 1024, 2) if m else None


def cell(result: dict | None, *, platform: str, model: str, cpus: int, limit_gb: float, limited: bool,
         state: dict, time_peak_gb: float | None) -> dict:
    out = {"platform": platform, "model": model, "cpus": cpus, "limit_gb": limit_gb, "limited": limited,
           "exit": state.get("exit"), "oom_killed": state.get("oom", False)}
    if not result:
        out.update({"measured": False, "within_class": False if (limited and state.get("oom")) else None})
        return out
    ms = sorted(i["ms"] for i in result["items"])
    peak = time_peak_gb if limited else (None if (result.get("resources") or {}).get("peak_memory_mb") is None
                                          else round(result["resources"]["peak_memory_mb"] / 1024, 2))
    finished = state.get("exit") == 0 and not state.get("oom")
    out.update({
        "measured": True, "n": result["n"], "contract": result.get("contract"),
        "intent_correct": result["intent_correct"], "accuracy": round(result["intent_correct"] / result["n"], 4),
        "median_ms": round(statistics.median(ms)), "p95_ms": ms[min(len(ms) - 1, int(0.95 * len(ms)))],
        "over_4s": round(sum(1 for i in result["items"] if i.get("timeout") or i["ms"] > SLOW_MS) / len(ms), 4),
        "peak_gb": peak,
        "within_class": finished if limited else (None if peak is None else finished and peak <= limit_gb),
        "cpu": result.get("cpu"),
    })
    return out


def row(c: dict) -> str:
    head = f"| {c['platform']} | {c['model']} | {c['cpus']} | {c['limit_gb']} GB{' (enforced)' if c['limited'] else ''}"
    if not c["measured"]:
        why = "killed for lack of memory" if c["oom_killed"] else f"no result (exit {c['exit']})"
        return f"{head} | {why} | | | | | |"
    within = {True: "yes", False: "no", None: "unknown"}[c["within_class"]]
    return (f"{head} | {c['accuracy']:.1%} | {c['median_ms']} ms | {c['p95_ms']} ms | {c['over_4s']:.1%} "
            f"| {c['peak_gb'] if c['peak_gb'] is not None else 'unknown'} GB | {within} |")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="bench-client: adapter check and the numbers of one cell")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("check-adapter")
    a.add_argument("manifest", type=Path)
    s = sub.add_parser("summary")
    s.add_argument("--platform", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--cpus", type=int, required=True)
    s.add_argument("--limit-gb", type=float, required=True)
    s.add_argument("--limited", action="store_true", help="the limit was enforced (Linux container)")
    s.add_argument("--result", type=Path, required=True)
    s.add_argument("--time", type=Path, help="output of /usr/bin/time -v (limited runs)")
    s.add_argument("--exit", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "check-adapter":
        why = check_adapter(json.loads(args.manifest.read_text(encoding="utf-8")))
        if why:
            print(why, file=sys.stderr)
            return 1
        return 0
    result = json.loads(args.result.read_text(encoding="utf-8")) if args.result.exists() else None
    state = read_state(args.exit.read_text(encoding="utf-8")) if args.exit.exists() else {"exit": None, "oom": False}
    peak = peak_from_time(args.time.read_text(encoding="utf-8")) if args.time and args.time.exists() else None
    c = cell(result, platform=args.platform, model=args.model, cpus=args.cpus, limit_gb=args.limit_gb,
             limited=args.limited, state=state, time_peak_gb=peak)
    args.out.write_text(json.dumps(c, indent=1) + "\n", encoding="utf-8")
    print("| Platform | Model | CPUs | Class limit | Accuracy | Median | p95 | Over 4 s | Peak | Within class |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---|")
    print(row(c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
