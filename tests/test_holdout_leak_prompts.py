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


def test_the_benchmark_prompt_examples_are_checked_too(tmp_path, capsys):
    rows = lock.load_bench_examples()
    texts = [t for _, t in rows]
    assert "Wer hat die Glühbirne erfunden?" in texts and not any("v1" in w for w, _ in rows)
    assert any("PROMPTS[detailed]" in w for w, _ in rows)
    leaks = lock.find_leaks(rows, [{"id": "h-1", "text": "Wer hat eigentlich die Glühbirne erfunden?"}], lock.MAX_SIMILARITY)
    assert leaks and leaks[0]["train"].startswith("bench.py:PROMPTS[")


def test_lock_and_build_read_one_threshold(tmp_path, capsys):
    import inspect

    import holdout_build as build
    assert lock.MAX_SIMILARITY == 0.8
    assert inspect.signature(build.build).parameters["max_similarity"].default == lock.MAX_SIMILARITY
    assert "default=lock.MAX_SIMILARITY" in Path(build.__file__).read_text(encoding="utf-8")
    train = tmp_path / "t.jsonl"
    train.write_text(json.dumps({"text": "Wer hat die Glühbirne erfunden?"}) + "\n", encoding="utf-8")
    held = tmp_path / "h.json"
    held.write_text(json.dumps({"items": [{"id": "h-1", "text": "Wer hat eigentlich die Glühbirne erfunden?"}]}),
                    encoding="utf-8")
    assert lock.main(["--train", str(train), "--holdout", str(held)]) == 1
    assert "at or above 0.8" in capsys.readouterr().err


@pytest.mark.parametrize("held", [
    "Wer hat denn eigentlich bitte mal die Glühbirne erfunden?",
    "wer HAT doch die glühbirne erfunden!!",
])
def test_filler_words_alone_are_always_a_leak(held):
    rows = [("bench.py:PROMPTS[compact][3]", "Wer hat die Glühbirne erfunden?")]
    assert lock.find_leaks(rows, [{"id": "h-1", "text": held}], 0.99)


def test_other_words_are_not_fillers():
    rows = [("x:1", "Wer hat die Glühbirne erfunden?")]
    assert not lock.find_leaks(rows, [{"id": "h-1", "text": "Wann wurde die erste Glühbirne in Berlin verkauft?"}], 0.99)


ORDER = ("wann ist das naechste treffen mit dem team", "watns iut kasnaechste treffen mit dem tram")


def test_character_similarity_does_not_depend_on_the_order():
    """difflib gives 0.881 one way and 0.786 the other; lock and build missed the second (shadow c2s-0551)."""
    import holdout_build as build
    a, b = ORDER
    for train, held in ((a, b), (b, a)):
        assert lock.find_leaks([("x:1", train)], [{"id": "h-1", "text": held}], lock.MAX_SIMILARITY)
        lex = build.Lexical(lock.MAX_SIMILARITY)
        lex.add_text("training", train)
        assert lex.near(held) == "training"
