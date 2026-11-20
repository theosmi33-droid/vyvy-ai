"""Continuous mode: rounds follow decisions and free GPU time, not the calendar.

User's decision of 20.11.2026: no deadline dependency. The next round starts once the
previous one is decided and quota is left; the hold points (changed rules, goal,
accepted adapter, plateau of rejections) still stop it, and "paused" stops everything.

    python -m pytest tests/test_continuous.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import collect  # noqa: E402
import kaggle_job  # noqa: E402
import loop_guard  # noqa: E402

SMALL = "Qwen/Qwen2.5-1.5B-Instruct"
LARGE = loop_guard.SEVEN_B
STATE = {"schema": 1, "mode": "continuous", "approved_on": "2026-11-20T22:00:00Z", "bases": [SMALL, LARGE],
         "fingerprint": "x", "seven_b_after_small": False}


def decided(adapter: str, created: str, accepted: bool = False) -> dict:
    return {"adapter": adapter, "created": created, "accepted": accepted}


def test_no_weekly_limit():
    # Two decisions per base in one week do not stop the next round.
    hist = [decided("vyvy-1.5b-d1", "2026-11-21T08:00:00Z"), decided("vyvy-7b-d1", "2026-11-21T18:00:00Z")]
    now = "2026-11-22T09:00:00Z"
    assert loop_guard.round_allowed(STATE, hist, SMALL, now)[0] is True
    assert loop_guard.gate(STATE, hist, SMALL, now, "x")[0] is True
    assert loop_guard.round_allowed(dict(STATE, mode="weekly"), hist, SMALL, now)[0] is False


def test_hold_points_still_stop_it():
    now = "2026-11-25T00:00:00Z"
    plateau = [decided(f"vyvy-1.5b-d{i}", f"2026-11-2{i}T08:00:00Z") for i in (1, 2, 3)]
    go, why = loop_guard.gate(STATE, plateau, SMALL, now, "x")
    assert go is False and "plateau" in why
    accepted = [decided("vyvy-1.5b-d1", "2026-11-21T08:00:00Z", True)]
    go, why = loop_guard.gate(STATE, accepted, SMALL, now, "x")
    assert go is False and "rotate" in why
    assert "rules" in loop_guard.gate(STATE, [], SMALL, now, "y")[1]
    assert loop_guard.gate(dict(STATE, mode="paused"), [], SMALL, now, "x")[0] is False
    # Decisions before the user's release do not count towards the plateau.
    old = [decided(f"vyvy-1.5b-d{i}", f"2026-11-1{i}T08:00:00Z") for i in (1, 2, 3)]
    assert loop_guard.gate(STATE, old, SMALL, now, "x")[0] is True


def test_the_round_moves_on_when_every_base_has_decided():
    assert collect.continuous_round(STATE, []) == "2026-11-20-c00"
    one = [decided("vyvy-1.5b-d1", "2026-11-21T08:00:00Z")]
    assert collect.continuous_round(STATE, one) == "2026-11-20-c00"  # 7B still training on the same sentences
    both = one + [decided("vyvy-7b-d1", "2026-11-21T20:00:00Z")]
    assert collect.continuous_round(STATE, both) == "2026-11-20-c01"
    before = [decided("vyvy-7b-d0", "2026-11-19T20:00:00Z"), decided("vyvy-1.5b-d0", "2026-11-19T08:00:00Z")]
    assert collect.continuous_round(STATE, before) == "2026-11-20-c00"
    assert collect.synth_round(state=STATE, history=both) == "2026-11-20-c01"


def test_weekly_and_one_round_keep_the_saturday_label():
    from datetime import datetime, timezone

    now = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)
    for mode in ("one_round", "weekly", "paused"):
        assert collect.synth_round(now, state=dict(STATE, mode=mode)) == "2026-11-14"


def test_the_teacher_kernel_takes_a_continuous_label(tmp_path):
    kaggle_job.write_synth_job(tmp_path, "user", "owner/repo", "a" * 40, "2026-11-20-c03")
    with pytest.raises(SystemExit):
        kaggle_job.write_synth_job(tmp_path, "user", "owner/repo", "a" * 40, "2026-11-20-c")
