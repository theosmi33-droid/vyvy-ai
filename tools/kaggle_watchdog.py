#!/usr/bin/env python3
"""Watch one Kaggle kernel run until it ends, then fetch its output or diagnose it.

Polls the kernel status every --interval seconds (default 60) up to --timeout
minutes (default 180):

- running / queued: logs a timestamped line and keeps waiting
- complete: downloads the files matching --pattern into --out
- error / cancelled: downloads the run log, keeps its last 100 lines, names the
  likely cause (out of memory, CUDA, missing module, wrong argument, disk,
  network) and writes both to --summary

Writes minutes=<wall-clock minutes of the run> and state=<complete|error|timeout>
to $GITHUB_OUTPUT when set. Exit code 0 only for complete.

    python tools/kaggle_watchdog.py --kernel owner/slug --out out --summary "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

CAUSES = (
    ("out of GPU memory", r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED"),
    ("CUDA or 4-bit kernel error", r"CUDA error|CUBLAS_STATUS|cuDNN|bitsandbytes"),
    ("missing Python module", r"ModuleNotFoundError|ImportError"),
    ("API mismatch in a library", r"unexpected keyword argument|has no attribute|TypeError"),
    ("disk full", r"No space left on device"),
    ("network or download", r"ConnectionError|HTTPError|Read timed out|Temporary failure in name resolution"),
)


def kaggle(*args: str) -> str:
    done = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    return (done.stdout + done.stderr).strip()


def state_of(text: str) -> str:
    t = text.lower()
    if "complete" in t:
        return "complete"
    if "error" in t or "cancel" in t or "fail" in t:
        return "error"
    return "running"


def diagnose(log: str) -> tuple[list[str], list[str]]:
    lines = log.splitlines()[-100:]
    tail = "\n".join(lines)
    causes = [name for name, pattern in CAUSES if re.search(pattern, tail)]
    return causes or ["unknown (see the log lines below)"], lines


def watch(kernel: str, interval: float, timeout_min: float,
          status: Callable[[str], str], sleep: Callable[[float], None] = time.sleep,
          clock: Callable[[], float] = time.monotonic, say: Callable[[str], None] = print) -> tuple[str, float]:
    start = clock()
    while (clock() - start) / 60 < timeout_min:
        text = status(kernel)
        state = state_of(text)
        say(f"{time.strftime('%H:%M:%S', time.gmtime())} {state}: {text}")
        if state != "running":
            return state, (clock() - start) / 60
        sleep(interval)
    return "timeout", (clock() - start) / 60


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Watch a Kaggle kernel run")
    p.add_argument("--kernel", required=True, help="owner/slug")
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--timeout", type=float, default=180, help="minutes")
    p.add_argument("--first-wait", type=float, default=90, help="seconds before the first poll")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--pattern", default=r"(^|/)([a-z0-9.-]+\.gguf|manifest\.json)$")
    p.add_argument("--summary", type=Path)
    args = p.parse_args(argv)

    time.sleep(args.first_wait)
    state, minutes = watch(args.kernel, args.interval, args.timeout,
                           lambda k: kaggle("kernels", "status", k))
    report = [f"### Kaggle run `{args.kernel}`", "", f"**{state}** after {minutes:.0f} min."]
    if state == "complete":
        print(kaggle("kernels", "output", args.kernel, "-p", str(args.out), "--file-pattern", args.pattern))
    else:
        logdir = args.out / "log"
        kaggle("kernels", "output", args.kernel, "-p", str(logdir), "--file-pattern", r"\.log$")
        log = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in sorted(logdir.glob("*.log")))
        causes, tail = diagnose(log) if log else (["no log available"], [])
        report += ["", "Likely cause: " + ", ".join(causes), "", "```", *tail, "```"]
    text = "\n".join(report) + "\n"
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write(text)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"minutes={minutes:.0f}\nstate={state}\n")
    return 0 if state == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
