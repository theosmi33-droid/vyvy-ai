"""Weekly report: one body per ISO week, adapters by ids and intents only.

    python -m pytest tests/test_weekly_report.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import improve_report  # noqa: E402
import weekly_report  # noqa: E402


def result(got_q2: str, adapter: str) -> dict:
    items = [{"id": "q1", "expected": "unknown", "got": "unknown", "text": "SECRET"},
             {"id": "q2", "expected": "appointments_today", "got": got_q2, "text": "SECRET"}]
    right = sum(i["got"] == i["expected"] for i in items)
    return {"n": 2, "mode": "route", "prompt": "compact", "adapter": adapter, "items": items,
            "intent_correct": right, "within_timeout": right, "median_ms": 600}


def test_previous_week() -> None:
    assert weekly_report.previous_week(dt.datetime(2026, 11, 16, 6, tzinfo=dt.timezone.utc)) == "2026-W46"


def test_body_lists_decisions_quota_lock_and_no_text() -> None:
    run = improve_report.run_report(result("appointments_tomorrow", "old"), result("appointments_today", "new"))
    run["shadow"] = "shadow holdout: b=1, c=0; not worse"
    text = weekly_report.body("2026-W46", [{"adapter": "new", "accepted": True}], [run, run],
                              "600 left", "paused", ["chain_collect"])
    assert "accepted: new" in text and "| new | old | 2/2 | 1 | 0 |" in text
    assert "600 left" in text and "Loop lock: paused" in text and "chain_collect" in text
    assert "SECRET" not in text


def test_an_empty_week_says_so() -> None:
    text = weekly_report.body("2026-W47", [], [], "", "paused", [])
    assert "No adapter was decided" in text and "not read" in text
