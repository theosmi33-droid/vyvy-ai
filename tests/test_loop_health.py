"""Loop health: findings by age and state, one issue per finding, closed when it passes.

    python -m pytest tests/test_loop_health.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import loop_health as lh  # noqa: E402

NOW = dt.datetime(2026, 11, 12, 12, tzinfo=dt.timezone.utc)
FRESH = {"Collect": "2026-11-12T11:30:00Z", "Heartbeat": "2026-11-12T10:00:00Z"}


def test_a_running_chain_has_no_finding() -> None:
    found, checked = lh.findings(NOW, FRESH, {"mode": "paused"}, "2026-11-10T22:00:00Z", None, {})
    assert found == {} and "token_kaggle" not in checked


def test_a_stopped_chain_is_found() -> None:
    found, _ = lh.findings(NOW, {"Collect": "2026-11-12T07:00:00Z", "Heartbeat": None}, None, None, None, {})
    assert set(found) == {"chain_collect", "chain_heartbeat"} and "5.0 h" in found["chain_collect"]


def test_paused_or_idle_for_a_week() -> None:
    found, _ = lh.findings(NOW, FRESH, {"mode": "paused"}, "2026-11-01T00:00:00Z", None, {})
    assert set(found) == {"paused_long"}
    released = {"mode": "one_round", "approved_on": "2026-11-02T00:00:00Z"}
    assert set(lh.findings(NOW, FRESH, released, None, "2026-11-03T00:00:00Z", {})[0]) == {"no_decision"}
    assert lh.findings(NOW, FRESH, released, None, "2026-11-11T00:00:00Z", {})[0] == {}


def test_tokens_only_when_checked() -> None:
    found, checked = lh.findings(NOW, FRESH, None, None, None, {"kaggle": "fail", "colab": "ok"})
    assert set(found) == {"token_kaggle"} and {"token_kaggle", "token_colab"} <= checked


def test_sync_opens_once_and_closes_when_passed() -> None:
    made, shut = [], []
    create = lambda t, b: made.append(t)  # noqa: E731
    close = lambda n, c: shut.append(n)  # noqa: E731
    open_titles = {"Loop health: chain_collect": 4, "Loop health: token_colab": 9, "Workflow failed: X": 2}
    found = {"chain_collect": "x", "paused_long": "y"}
    done = lh.sync(found, {"chain_collect", "paused_long", "chain_heartbeat"}, open_titles, create, close)
    assert made == ["Loop health: paused_long"]
    assert shut == []  # token_colab was not checked in this run; the other issue is not ours
    lh.sync({}, {"chain_collect", "token_colab"}, open_titles, create, close)
    assert sorted(shut) == [4, 9] and done == ["opened Loop health: paused_long"]
