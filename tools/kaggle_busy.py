#!/usr/bin/env python3
"""List kernels of this Kaggle account that are running or queued right now.

`kaggle kernels list --mine` has no status column, so every kernel's latest run
is asked with `kaggle kernels status`. Covers all projects on the account, not
only Vyvy's kernels. Prints one kernel ref per line; nothing means the GPU is free.

    python tools/kaggle_busy.py
"""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from collections.abc import Callable


def kaggle(*args: str) -> str:
    done = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    return done.stdout + done.stderr


def busy(listing: str, status: Callable[[str], str]) -> list[str]:
    refs = [row["ref"] for row in csv.DictReader(io.StringIO(listing)) if row.get("ref")]
    out = []
    for ref in refs:
        s = status(ref).lower()
        if "running" in s or "queued" in s:
            out.append(ref)
    return out


def main() -> int:
    listing = kaggle("kernels", "list", "--mine", "--page-size", "50", "-v")
    for ref in busy(listing, lambda r: kaggle("kernels", "status", r)):
        print(ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
