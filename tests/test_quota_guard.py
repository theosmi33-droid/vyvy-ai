"""Quota guard: without Kaggle's own figure no GPU run starts; with it, the reserve decides.

    python -m pytest tests/test_quota_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import quota_guard  # noqa: E402


def run(monkeypatch, capsys, live, need: float) -> str:
    monkeypatch.setattr(quota_guard, "kaggle_quota", lambda: live)
    monkeypatch.setattr(quota_guard, "gh_json", lambda path: {"workflow_runs": [], "jobs": []})
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    quota_guard.main(["--repo", "o/r", "--need", str(need)])
    return capsys.readouterr().out


def test_unreadable_quota_pauses_every_gpu_run(monkeypatch, capsys) -> None:
    out = run(monkeypatch, capsys, None, 240)
    assert "Quota paused" in out and "fail closed" in out  # the workflows grep for "Quota paused"


def test_unreadable_quota_is_only_shown_for_a_status_line(monkeypatch, capsys) -> None:
    assert "Quota paused" not in run(monkeypatch, capsys, None, 0)


def test_kaggle_figure_decides_with_the_reserve(monkeypatch, capsys) -> None:
    live = (1200.0, 1800.0, "2026-11-14T00:00:00")
    assert "Quota paused until 2026-11-14" in run(monkeypatch, capsys, live, 630)
    assert "Quota paused" not in run(monkeypatch, capsys, live, 240)
