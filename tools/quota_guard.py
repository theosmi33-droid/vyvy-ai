#!/usr/bin/env python3
"""Stop new Kaggle runs before the weekly GPU quota runs out.

Kaggle grants 30 GPU hours per week (reset Monday 00:00 UTC) and has no API for
the remaining quota. This guard counts what this pipeline used: the duration of
the Kaggle steps of every Train run in the current ISO week, read from the
GitHub API. GPU time used outside this pipeline (notebooks started by hand on
the same account) is not visible here; pass it with --external (minutes) or the
repository variable KAGGLE_EXTERNAL_GPU_MIN.

With less than --reserve minutes left (default 120, i.e. pause at 28 hours) it
prints "Quota paused until Monday ..." and writes paused=true to $GITHUB_OUTPUT.
Exit code is 0 in both cases; the workflow decides what to skip.

    python tools/quota_guard.py --repo OWNER/REPO
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections.abc import Callable

KAGGLE_STEPS = ("Push the training job to Kaggle", "Watch the Kaggle run", "Wait for Kaggle")


def week_start(now: dt.datetime) -> dt.datetime:
    monday = now - dt.timedelta(days=now.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def parse(ts: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None


def used_minutes(runs: list[dict], jobs_of: Callable[[int], list[dict]], since: dt.datetime,
                 now: dt.datetime) -> float:
    total = 0.0
    for run in runs:
        created = parse(run.get("created_at"))
        if not created or created < since:
            continue
        for job in jobs_of(run["id"]):
            for step in job.get("steps", []):
                if step.get("name") not in KAGGLE_STEPS:
                    continue
                start = parse(step.get("started_at"))
                end = parse(step.get("completed_at")) or now
                if start:
                    total += max(0.0, (end - start).total_seconds() / 60)
    return total


def gh_json(path: str) -> dict:
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pause Kaggle runs before the weekly quota runs out")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    p.add_argument("--workflow", default="train.yml")
    p.add_argument("--limit", type=float, default=30 * 60, help="weekly GPU minutes")
    p.add_argument("--reserve", type=float, default=120, help="minutes that must remain")
    p.add_argument("--external", type=float, default=float(os.environ.get("KAGGLE_EXTERNAL_GPU_MIN") or 0),
                   help="GPU minutes used this week outside this pipeline")
    args = p.parse_args(argv)

    now = dt.datetime.now(dt.UTC)
    since = week_start(now)
    runs = gh_json(f"repos/{args.repo}/actions/workflows/{args.workflow}/runs"
                   f"?per_page=100&created=>={since.date().isoformat()}")["workflow_runs"]
    used = used_minutes(runs, lambda rid: gh_json(f"repos/{args.repo}/actions/runs/{rid}/jobs")["jobs"],
                        since, now) + args.external
    left = args.limit - used
    paused = left < args.reserve
    next_monday = (since + dt.timedelta(days=7)).strftime("%Y-%m-%d")
    msg = (f"Kaggle GPU this week: {used:.0f} of {args.limit:.0f} min used (incl. {args.external:.0f} external), "
           f"{left:.0f} left, reserve {args.reserve:.0f}.")
    if paused:
        msg += f" Quota paused until Monday {next_monday} 00:00 UTC."
    print(msg)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(msg + "\n\n")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"paused={'true' if paused else 'false'}\nused={used:.0f}\nleft={left:.0f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
