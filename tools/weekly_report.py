#!/usr/bin/env python3
"""Weekly report of the improvement loop: one issue per ISO week.

Lists the adapters decided in the week (release with verdict.json, base_blind.json
and lora_blind.json) with right, b, c, p, wrong actions against the baseline and
on unknown (tools/improve_report.py, ids and intents only, never a question
text), the shadow holdout, the Kaggle quota line, the loop lock and the open
"Loop health" issues. The issue "Weekly report <week>" is written once.

    python tools/weekly_report.py --repo OWNER/REPO [--week 2026-W46] [--quota "…"] [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import improve_report  # noqa: E402
import loop_guard  # noqa: E402


def previous_week(now: dt.datetime) -> str:
    return loop_guard.iso_week((now - dt.timedelta(days=7)).isoformat())


def body(week: str, decided: list[dict], runs: list[dict], quota: str, lock: str, health: list[str]) -> str:
    out = [f"## Weekly report {week}", ""]
    if decided:
        out += [f"Decided: {len(decided)} adapter(s), accepted: "
                + (", ".join(d["adapter"] for d in decided if d["accepted"]) or "none") + ".", ""]
    else:
        out += ["No adapter was decided this week.", ""]
    if runs:
        out += improve_report.table(runs) + ["", "Wrong answers by kind:", ""] + improve_report.kinds_table(runs)
        shadow = [f"- {r['adapter']}: {r['shadow']}" for r in runs if r.get("shadow")]
        if shadow:
            out += ["", "Shadow holdout:", ""] + shadow
        plan = improve_report.propose(runs)
        out += ["", "Proposal for the next teacher round: double " + (", ".join(plan["weak"]) or "nothing") + "; "
                + (", ".join(f"{p['meant']} -> {p['mistaken_for']}" for p in plan["near"][:6]) or "no near misses")
                + "."]
    out += ["", f"Kaggle quota: {quota or 'not read'}", "", f"Loop lock: {lock}", "",
            "Open loop health issues: " + (", ".join(health) or "none"), "",
            "Written by loop-watch.yml (tools/weekly_report.py); adapters only by ids and intents."]
    return "\n".join(out)


def gh(*args: str, check: bool = True) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check).stdout


def runs_of(repo: str, decided: list[dict]) -> list[dict]:
    runs = []
    for d in decided:
        with tempfile.TemporaryDirectory() as tmp:
            got = subprocess.run(["gh", "release", "download", d["adapter"], "--repo", repo, "--dir", tmp,
                                  "--pattern", "base_blind.json", "--pattern", "lora_blind.json",
                                  "--pattern", "verdict.json"], capture_output=True, text=True)
            folder = Path(tmp)
            if got.returncode or not (folder / "lora_blind.json").exists():
                continue
            r = improve_report.run_report(improve_report.load(folder / "base_blind.json"),
                                          improve_report.load(folder / "lora_blind.json"))
            if (folder / "verdict.json").exists():
                r["shadow"] = (improve_report.load(folder / "verdict.json").get("shadow") or {}).get("reason")
            runs.append(r)
    return runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write the weekly report issue of the improvement loop")
    p.add_argument("--repo", required=True)
    p.add_argument("--week", help="ISO week, default: the week before today")
    p.add_argument("--quota", default="", help="the line of tools/quota_guard.py")
    p.add_argument("--dry-run", action="store_true", help="print the report, open no issue")
    args = p.parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc)
    week = args.week or previous_week(now)
    title = f"Weekly report {week}"
    existing = json.loads(gh("issue", "list", "--repo", args.repo, "--state", "all", "--search",
                             f'"{title}" in:title', "--json", "title") or "[]")
    if any(i["title"] == title for i in existing) and not args.dry_run:
        print(f"{title} exists already.")
        return 0
    history = loop_guard.release_history(args.repo)
    decided = [d for d in history if loop_guard.iso_week(d["created"]) == week]
    state = loop_guard.load_state()
    _, lock = loop_guard.round_allowed(state, history, "Qwen/Qwen2.5-1.5B-Instruct", now.isoformat())
    health = [i["title"].removeprefix("Loop health: ") for i in json.loads(
        gh("issue", "list", "--repo", args.repo, "--state", "open", "--search", '"Loop health: " in:title',
           "--json", "title") or "[]") if i["title"].startswith("Loop health: ")]
    text = body(week, decided, runs_of(args.repo, decided), args.quota, lock, health)
    if args.dry_run:
        print(text)
    else:
        gh("issue", "create", "--repo", args.repo, "--title", title, "--body", text)
        print(f"opened {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
