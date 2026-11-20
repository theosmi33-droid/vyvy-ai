"""Slot scoring for create requests: a slot is a piece of the sentence or empty.

    python -m pytest tests/test_slots.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))

import slots  # noqa: E402

SENTENCE = "Leg mir für morgen um drei 'Angebot Kofler prüfen' an, mit Frau Brandauer"


def test_a_slot_must_be_a_piece_of_the_sentence():
    assert slots.is_span_of(SENTENCE, "Angebot Kofler prüfen")
    assert slots.is_span_of(SENTENCE, "morgen um drei")
    assert not slots.is_span_of(SENTENCE, "Angebot prüfen")  # words out of order: not a piece
    assert not slots.is_span_of(SENTENCE, "15:00")  # computed, not written
    assert not slots.is_span_of(SENTENCE, "")


def test_invented_beats_the_reference():
    assert slots.judge(SENTENCE, "when", "15:00", "morgen um drei") == "invented"
    assert slots.judge(SENTENCE, "title", "Angebot für Kofler prüfen", "Angebot Kofler prüfen") == "invented"


def test_kinds():
    assert slots.judge(SENTENCE, "title", "Angebot Kofler prüfen", "Angebot Kofler prüfen") == "right"
    assert slots.judge(SENTENCE, "title", "Angebot Kofler prüfen an", "Angebot Kofler prüfen") == "right"  # F1 0.86
    assert slots.judge(SENTENCE, "title", "prüfen", "Angebot Kofler prüfen") == "wrong"
    assert slots.judge(SENTENCE, "person", "Frau Brandauer", ["Brandauer", "Frau Brandauer"]) == "right"
    assert slots.judge(SENTENCE, "person", None, "Frau Brandauer") == "missed"
    assert slots.judge(SENTENCE, "when", "morgen", None) == "filled"
    assert slots.judge(SENTENCE, "when", None, None) == "empty"


def test_score_counts_only_items_with_slots():
    items = [{"id": "a", "text": SENTENCE,
              "slots": {"title": "Angebot Kofler prüfen", "when": "morgen um drei", "person": "Frau Brandauer"}},
             {"id": "b", "text": "Was steht heute an?"}]
    outputs = {"a": {"title": "Angebot Kofler prüfen", "when": "15 Uhr", "person": None}}
    result = slots.score(items, outputs)
    assert result["n"] == 1 and result["invented"] == 1 and result["all_slots_right"] == 0
    assert result["items"][0] == {"id": "a", "title": "right", "when": "invented", "person": "missed"}
