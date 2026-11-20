"""The retired blind holdout v2 is training data; the lock protects the holdouts that still measure.

    python -m pytest tests/test_retired_holdout.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_others_calendar  # noqa: E402
import build_retired_holdout as retired  # noqa: E402
import check_holdout_leak as lock  # noqa: E402


def test_v2_is_retired_and_the_others_stay_guarded():
    assert lock.BLIND_V2 in lock.RETIRED and lock.BLIND_V2 not in lock.GUARDED
    assert {lock.BLIND_V3, lock.BLIND_V4, lock.BLIND, lock.HOLDOUT} <= set(lock.GUARDED)
    # The holdout file itself stays (the approval fingerprint reads data/holdout_*.json).
    assert lock.BLIND_V2.exists()


def test_files_match_their_generators():
    assert retired.main(["--check"]) == 0
    assert build_others_calendar.main(["--check"]) == 0


def test_nothing_near_a_guarded_holdout():
    rows = [json.loads(line) for line in retired.OUT.read_text(encoding="utf-8").splitlines()]
    guarded = [i for h in lock.GUARDED for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    assert not lock.find_leaks([(str(n), r["text"]) for n, r in enumerate(rows)], guarded, 0.85)


def test_timeframe_follows_the_last_time_word():
    assert retired.timeframe("Termin heute – nee, morgen um neun", "create_appointment") == "tomorrow"
    assert retired.timeframe("Trog ma morgn um zehne an Termin ei", "create_appointment") == "tomorrow"
    assert retired.timeframe("Termin nächste Woch mit dem Steuerberater", "create_appointment") == "week"
    assert retired.timeframe("Termin am Freitag um drei", "create_appointment") is None
    assert retired.timeframe("Was steht morgen an?", "appointments_today") == "today"  # the intent decides
    assert retired.timeframe("Welche Mails sind neu?", "emails_unread") is None
