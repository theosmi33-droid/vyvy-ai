"""Hard negatives for "unknown": contract, holdout lock, no contradicting labels, no personal data.

    python -m pytest tests/test_unknown_hard.py
    python tests/test_unknown_hard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runner"))
sys.path.insert(0, str(ROOT / "training"))

import bench  # noqa: E402
import build_unknown_hard as hard  # noqa: E402
import check_holdout_leak as lock  # noqa: E402
import import_partner  # noqa: E402
import train_lora  # noqa: E402


def rows() -> list[dict]:
    return [json.loads(line) for line in hard.OUT.read_text(encoding="utf-8").splitlines() if line.strip()]


def intents() -> tuple[str, ...]:
    return bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))


def test_file_matches_generator() -> None:
    assert hard.main(["--check"]) == 0


def test_count_and_contract() -> None:
    data = rows()
    assert len(data) == sum(hard.QUOTAS.values()) == 500
    for r in data:
        assert set(r) == {"text", "intent", "timeframe", "entity"}, r
        assert r["intent"] == "unknown" and r["timeframe"] is None and r["entity"] is None, r
        assert isinstance(r["text"], str) and r["text"] == r["text"].strip() and r["text"], r
    keys = [lock.normalise(r["text"]) for r in data]
    assert len(set(keys)) == len(keys)


def test_training_pipeline_reads_every_line() -> None:
    loaded = train_lora.load_rows([hard.OUT], intents())
    assert len(loaded) == 500
    assert all(json.loads(r["answer"])["intent"] == "unknown" for r in loaded)


def test_holdout_lock() -> None:
    holdout = [i for h in lock.HOLDOUTS if h.exists() for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    train = [(f"{hard.OUT.name}:{no}", r["text"]) for no, r in enumerate(rows(), 1)]
    assert lock.find_leaks(train, holdout, 0.85) == []


def test_no_sentence_of_another_intent() -> None:
    other = set()
    for path in (ROOT / "data" / "train").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line)["intent"] != "unknown":
                other.add(lock.normalise(json.loads(line)["text"]))
    assert not [r["text"] for r in rows() if lock.normalise(r["text"]) in other]


def test_no_personal_data() -> None:
    for r in rows():
        assert not import_partner.personal_hits(r["text"]), r


def test_time_words() -> None:
    # The rejected adapter read heute or morgen as a calendar request; the negatives must contain them.
    words = ("heute", "morgen", "gestern", "wochenende")
    assert sum(any(w in r["text"].casefold() for w in words) for r in rows()) >= 150


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
