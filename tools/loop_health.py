#!/usr/bin/env python3
"""Health of the improvement loop, as issues: one per finding, closed when it is gone.

Runs from loop-watch.yml on its own schedule, not from Collect: a check that
hangs on the chain would stop together with it. Findings:

- chain_collect / chain_heartbeat: the last Collect or Heartbeat run is older than
  --stale hours (default 4): the chain stopped (or GitHub disabled the schedules).
- paused_long: data/loop_state.json says "paused" for more than --idle days.
- no_decision: released, but no adapter decided for more than --idle days.
- token_kaggle / token_colab: the weekly sign-in check failed (--kaggle/--colab fail).

Each finding has one issue titled "Loop health: <key>"; an open one is not
opened twice, and it is closed with a comment when the check passes again. Keys
not checked in a run (tokens outside the weekly run) are left alone.

    python tools/loop_health.py --repo OWNER/REPO [--kaggle ok|fail --colab ok|fail] [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import loop_guard  # noqa: E402

PREFIX = "Loop health: "
TEXT = {
    "chain_collect": "Collect has not run for {age:.1f} h (limit {limit} h). The chain Heartbeat -> Collect stopped "
                     "or GitHub disabled the schedules; a run of heartbeat.yml by hand restarts it.",
    "chain_heartbeat": "Heartbeat has not run for {age:.1f} h (limit {limit} h). The chain is broken; a run of "
                       "heartbeat.yml by hand restarts it.",
    "paused_long": "The loop has been paused for {age:.0f} days (data/loop_state.json). Nothing trains until the "
                   "user releases a round.",
    "no_decision": "No adapter was decided for {age:.0f} days although a round is released.",
    "token_kaggle": "The weekly Kaggle sign-in failed: the token (secret KAGGLE_API_TOKEN) is missing or expired.",
    "token_colab": "The weekly Colab sign-in failed: the credentials (secret COLAB_ADC, environment colab) are "
                   "missing or expired.",
}


def hours(since: str, now: dt.datetime) -> float:
    return (now - dt.datetime.fromisoformat(since.replace("Z", "+00:00"))).total_seconds() / 3600


def findings(now: dt.datetime, last_run: dict[str, str | None], state: dict | None, state_since: str | None,
             last_decision: str | None, tokens: dict[str, str], stale: float = 4, idle: float = 7) -> tuple[dict, set]:
    """(findings {key: text}, keys checked). last_run: workflow -> createdAt of its newest run."""
    found: dict[str, str] = {}
    checked = {"chain_collect", "chain_heartbeat", "paused_long", "no_decision"}
    for key, wf in (("chain_collect", "Collect"), ("chain_heartbeat", "Heartbeat")):
        age = hours(last_run[wf], now) if last_run.get(wf) else float("inf")
        if age > stale:
            found[key] = TEXT[key].format(age=age, limit=stale)
    mode = (state or {}).get("mode", "paused")
    if mode == "paused" and state_since and hours(state_since, now) > idle * 24:
        found["paused_long"] = TEXT["paused_long"].format(age=hours(state_since, now) / 24)
    if mode != "paused" and state:
        since = max(filter(None, [last_decision, state.get("approved_on")]), default=None)
        if since and hours(since, now) > idle * 24:
            found["no_decision"] = TEXT["no_decision"].format(age=hours(since, now) / 24)
    for name, result in tokens.items():
        checked.add(f"token_{name}")
        if result != "ok":
            found[f"token_{name}"] = TEXT[f"token_{name}"]
    return found, checked


def sync(found: dict[str, str], checked: set, open_titles: dict[str, int],
         create: Callable[[str, str], None], close: Callable[[int, str], None]) -> list[str]:
    """Open an issue per new finding, close those whose check passed; returns what was done."""
    done = []
    for key, text in sorted(found.items()):
        title = PREFIX + key
        if title not in open_titles:
            create(title, text + "\n\nOpened by loop-watch.yml (tools/loop_health.py); it closes this issue "
                                 "when the check passes again.")
            done.append(f"opened {title}")
    for title, number in sorted(open_titles.items()):
        key = title.removeprefix(PREFIX)
        if title.startswith(PREFIX) and key in checked and key not in found:
            close(number, "The check passes again (loop-watch.yml).")
            done.append(f"closed {title}")
    return done


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def last_runs(repo: str) -> dict[str, str | None]:
    out = {}
    for wf, file in (("Collect", "collect.yml"), ("Heartbeat", "heartbeat.yml")):
        rows = json.loads(gh("run", "list", "--repo", repo, "--workflow", file, "--limit", "1", "--json", "createdAt")
                          or "[]")
        out[wf] = rows[0]["createdAt"] if rows else None
    return out


def state_since(path: Path = loop_guard.STATE) -> str | None:
    """When data/loop_state.json last changed (its last commit)."""
    out = subprocess.run(["git", "log", "-1", "--format=%cI", "--", str(path)], capture_output=True, text=True,
                         cwd=ROOT).stdout.strip()
    return out or None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Open or close loop health issues")
    p.add_argument("--repo", required=True)
    p.add_argument("--kaggle", choices=["ok", "fail"], help="result of the weekly Kaggle sign-in")
    p.add_argument("--colab", choices=["ok", "fail"], help="result of the weekly Colab sign-in")
    p.add_argument("--stale", type=float, default=4)
    p.add_argument("--idle", type=float, default=7)
    p.add_argument("--dry-run", action="store_true", help="only print what would be opened or closed")
    args = p.parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc)
    history = loop_guard.release_history(args.repo)
    tokens = {k: v for k, v in (("kaggle", args.kaggle), ("colab", args.colab)) if v}
    found, checked = findings(now, last_runs(args.repo), loop_guard.load_state(), state_since(),
                              history[-1]["created"] if history else None, tokens, args.stale, args.idle)
    open_titles = {i["title"]: i["number"] for i in json.loads(
        gh("issue", "list", "--repo", args.repo, "--state", "open", "--search", f'"{PREFIX}" in:title',
           "--json", "number,title") or "[]")}
    if args.dry_run:
        create = lambda t, b: None  # noqa: E731
        close = lambda n, c: None  # noqa: E731
    else:
        create = lambda t, b: gh("issue", "create", "--repo", args.repo, "--title", t, "--body", b)  # noqa: E731
        close = lambda n, c: gh("issue", "close", str(n), "--repo", args.repo, "--comment", c)  # noqa: E731
    done = sync(found, checked, open_titles, create, close)
    print("### Loop health\n")
    print("\n".join(f"- {k}: {v}" for k, v in sorted(found.items())) or "- all checks pass")
    print("\n" + ("\n".join(f"- {d}" for d in done) if done else "- no issue changed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
