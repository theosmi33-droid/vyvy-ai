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


def test_an_extra_round_keeps_the_regular_one() -> None:
    import tempfile

    import import_synth
    import kaggle_job
    with tempfile.TemporaryDirectory() as tmp:
        sha = "a" * 40
        kaggle_job.write_synth_job(Path(tmp), "user", "owner/repo", sha, "2026-11-07-b")
        for bad in ("2026-11-07-c", "2026-11-07b", "manual"):
            try:
                kaggle_job.write_synth_job(Path(tmp), "user", "owner/repo", sha, bad)
                raise AssertionError(bad)
            except SystemExit:
                pass
    target = import_synth.target({"script_sha256": "f" * 64, "round": "2026-11-07-b"}, Path("data/train"))
    assert target.name == "vyvy_synth_fffffffff_2026-11-07-b.jsonl"
    # The extra file does not count as the week's regular round on the next Saturday,
    # and the regular glob does not match it.
    assert not target.match("vyvy_synth_*_2026-11-07.jsonl")


if __name__ == "__main__":
    test_week_change()
    test_legacy_round_counts_for_its_week()
    print("ok")
