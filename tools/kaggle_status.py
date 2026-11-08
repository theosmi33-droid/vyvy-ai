#!/usr/bin/env python3
"""Show the latest kernels of this Kaggle account with their run status.

Read-only: lists and asks, never pushes, starts or cancels anything. Prints a
Markdown table (also to $GITHUB_STEP_SUMMARY), newest kernels first.

    python tools/kaggle_status.py --limit 10
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
from collections.abc import Callable

from kaggle_busy import kaggle


def status_word(raw: str) -> str:
    """'x has status "KernelWorkerStatus.RUNNING"' -> 'running'."""
    match = re.search(r'status "(?:KernelWorkerStatus\.)?([A-Za-z_]+)"', raw)
    return match.group(1).lower() if match else (raw.strip().splitlines() or ["unknown"])[-1]


def table(listing: str, status: Callable[[str], str], limit: int) -> str:
    rows = [row for row in csv.DictReader(io.StringIO(listing)) if row.get("ref")]
    rows.sort(key=lambda row: row.get("lastRunTime") or "", reverse=True)
    lines = ["| Kernel | Last run (UTC) | Status |", "|---|---|---|"]
    for row in rows[:limit]:
        lines.append(f"| {row['ref']} | {row.get('lastRunTime') or '-'} | {status_word(status(row['ref']))} |")
    if len(lines) == 2:
        lines.append("| (no kernels) | - | - |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Read-only status of the account's Kaggle kernels")
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args(argv)
    listing = kaggle("kernels", "list", "--mine", "--page-size", "50", "-v")
    out = table(listing, lambda ref: kaggle("kernels", "status", ref), args.limit)
    print(out)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(out + "\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
