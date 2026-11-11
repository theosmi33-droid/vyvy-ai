#!/usr/bin/env python3
"""Stop new Kaggle runs before the weekly GPU quota runs out.

Primary source: Kaggle's own account quota (`kaggle quota`, the same figure as
Settings -> Quotas), which includes every GPU session of the account, also
notebooks started by hand. If that call fails, the guard falls back to counting
this pipeline's Kaggle steps of the current ISO week from the GitHub API; GPU
time used outside the pipeline is then added with --external or the repository
variable KAGGLE_EXTERNAL_GPU_MIN.

With less than --reserve minutes left (default 120, i.e. pause at 28 hours) it
prints "Quota paused until <refresh time>" and writes paused=true to $GITHUB_OUTPUT.
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
    """Start of the current Kaggle quota week: Saturday 00:00 UTC (Kaggle's refreshAt)."""
    saturday = now - dt.timedelta(days=(now.weekday() - 5) % 7)
    return saturday.replace(hour=0, minute=0, second=0, microsecond=0)


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


def kaggle_quota() -> tuple[float, float, str] | None:
    """(used minutes, total minutes, refresh time) from the Kaggle API, or None."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        r = api.quota_view()
        q = r.gpu_quota
        return (q.time_used.total_seconds() / 60, q.total_time_allowed.total_seconds() / 60,
                r.quota_refresh_time.isoformat() if r.quota_refresh_time else "")
    except Exception as exc:  # noqa: BLE001  (any failure means: fallback count, and no run starts)
        print(f"Kaggle quota not available ({type(exc).__name__}: {exc}); counting from GitHub for display only.")
        return None


def gh_json(path: str) -> dict:
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pause Kaggle runs before the weekly quota runs out")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    p.add_argument("--workflow", default="train.yml")
    p.add_argument("--limit", type=float, default=30 * 60, help="weekly GPU minutes")
    p.add_argument("--reserve", type=float, default=120, help="minutes that must remain after the run")
    p.add_argument("--need", type=float, default=0, help="minutes the planned run will use")
    p.add_argument("--external", type=float, default=float(os.environ.get("KAGGLE_EXTERNAL_GPU_MIN") or 0),
                   help="GPU minutes used this week outside this pipeline")
    args = p.parse_args(argv)

    now = dt.datetime.now(dt.UTC)
    since = week_start(now)
    live = kaggle_quota()
    if live:
        used, limit, refresh = live
        source = "Kaggle account"
    else:
        runs = gh_json(f"repos/{args.repo}/actions/workflows/{args.workflow}/runs"
                       f"?per_page=100&created=>={since.date().isoformat()}")["workflow_runs"]
        used = used_minutes(runs, lambda rid: gh_json(f"repos/{args.repo}/actions/runs/{rid}/jobs")["jobs"],
                            since, now) + args.external
        limit, refresh = args.limit, (since + dt.timedelta(days=7)).isoformat()
        source = f"this pipeline, incl. {args.external:.0f} min external"
    left = limit - used
    paused = left < args.reserve + args.need
    msg = (f"Kaggle GPU this week ({source}): {used:.0f} of {limit:.0f} min used, "
           f"{left:.0f} left; this run needs {args.need:.0f} plus a reserve of {args.reserve:.0f}.")
    # The fallback only sees GitHub step times (the kernel runs on its own), so it counts
    # seconds for hours of GPU: good enough for a status line, never to start a run.
    if not live and args.need > 0:
        paused = True
        msg += " Quota paused: Kaggle's own figure could not be read, so no GPU run starts (fail closed)."
    elif paused:
        msg += f" Quota paused until {refresh or 'the weekly reset'}."
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
