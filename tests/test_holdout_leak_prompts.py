"""The holdout lock also covers the example sentences in the teacher prompt (RULES)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import check_holdout_leak as lock  # noqa: E402

SYNTH = '''
RULES = {
    "create_task": (
        "Regel "
        "in zwei Teilen.",
        ["Notier mir den Ölwechsel am Transporter.", "Schreib auf: Gerüst abbestellen."]),
}
WEAK = set(RULES)
'''


def _holdout(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps({"items": [{"id": "h-1", "text": text}]}), encoding="utf-8")
    return path


def _synth(tmp_path: Path) -> Path:
    path = tmp_path / "synthesize.py"
    path.write_text(SYNTH, encoding="utf-8")
    return path


def test_reads_rules_examples_without_importing(tmp_path):
    rows = lock.load_prompt_examples(_synth(tmp_path))
    assert rows == [
        ("synthesize.py:RULES[create_task][0]", "Notier mir den Ölwechsel am Transporter."),
        ("synthesize.py:RULES[create_task][1]", "Schreib auf: Gerüst abbestellen."),
    ]


def test_prompt_example_matching_a_holdout_fails(tmp_path, capsys):
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"text": "Wann kommt der Kran?", "intent": "unknown"}) + "\n", encoding="utf-8")
    code = lock.main(["--train", str(train), "--prompts", str(_synth(tmp_path)),
                      "--holdout", str(_holdout(tmp_path, "Schreib auf, Gerüst abbestellen!"))])
    assert code == 1
    assert "RULES[create_task][1]" in capsys.readouterr().err


def test_clean_prompt_examples_pass(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"text": "Wann kommt der Kran?", "intent": "unknown"}) + "\n", encoding="utf-8")
    code = lock.main(["--train", str(train), "--prompts", str(_synth(tmp_path)),
                      "--holdout", str(_holdout(tmp_path, "Wie spät ist es in Tokio?"))])
    assert code == 0


def test_missing_rules_is_an_error(tmp_path):
    path = tmp_path / "synthesize.py"
    path.write_text("INTENTS = {}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        lock.load_prompt_examples(path)


def test_real_synthesis_script_has_examples():
    assert len(lock.load_prompt_examples(lock.SYNTH)) >= 6
