"""Synthesis rounds follow the Kaggle quota week (Saturday 00:00 UTC).

    python tests/test_synth_round.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import collect  # noqa: E402


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def test_week_change() -> None:
    assert collect.synth_round(at("2026-11-13T23:59:59")) == "2026-11-07"
    assert collect.synth_round(at("2026-11-14T00:00:00")) == "2026-11-14"
    assert collect.synth_round(at("2026-11-07T00:00:00")) == "2026-11-07"
    assert collect.synth_round(at("2026-11-09T12:00:00")) == "2026-11-07"  # Monday: no new round


def test_legacy_round_counts_for_its_week() -> None:
    # The 2026-W45 sentences were taken in on Saturday 07.11; that quota week is done.
    assert collect.synth_done(at("2026-11-10T08:00:00"))
    assert not collect.synth_done(at("2026-11-14T00:00:00"))


if __name__ == "__main__":
    test_week_change()
    test_legacy_round_counts_for_its_week()
    print("ok")
