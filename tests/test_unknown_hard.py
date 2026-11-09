"""Hard negatives for "unknown" and their contrast pairs: contract, holdout lock, no contradicting labels.

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


def rows(path: Path = hard.OUT) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def both() -> list[dict]:
    return rows(hard.OUT) + rows(hard.CONTRAST_OUT)


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


def test_contrast_pairs() -> None:
    data = rows(hard.CONTRAST_OUT)
    assert len(data) == sum(hard.CONTRAST_QUOTAS.values()) == 80
    for r in data:
        assert r["intent"] in hard.CONTRAST_QUOTAS and r["timeframe"] == hard.TIMEFRAME[r["intent"]], r
        assert r["entity"] and r["entity"].casefold() in r["text"].casefold(), r
    # Every person of the unknown side also appears in the user's own calendar.
    named = {e.split()[-1] for e in (r["entity"] for r in data)}
    assert {n.split()[-1] for n, _, _ in hard.PERSONS} <= named


def test_training_pipeline_reads_every_line() -> None:
    loaded = train_lora.load_rows([hard.OUT, hard.CONTRAST_OUT], intents())
    assert len(loaded) == 580
    assert sum(json.loads(r["answer"])["intent"] == "unknown" for r in loaded) == 500


def test_holdout_lock() -> None:
    holdout = [i for h in lock.HOLDOUTS if h.exists() for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    train = [(f"{path.name}:{no}", r["text"]) for path in (hard.OUT, hard.CONTRAST_OUT)
             for no, r in enumerate(rows(path), 1)]
    assert lock.find_leaks(train, holdout, 0.85) == []


def test_one_label_per_sentence() -> None:
    labels: dict[str, set[str]] = {}
    for path in (ROOT / "data" / "train").glob("*.jsonl"):
        for r in rows(path):
            labels.setdefault(lock.normalise(r["text"]), set()).add(r["intent"])
    mine = {lock.normalise(r["text"]) for r in both()}
    assert not [k for k in mine if len(labels[k]) > 1]


def test_known_paraphrases_stay_out() -> None:
    assert not [r["text"] for r in rows() if lock.normalise(r["text"]) in hard.PARAPHRASES]


def test_no_personal_data() -> None:
    for r in both():
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
