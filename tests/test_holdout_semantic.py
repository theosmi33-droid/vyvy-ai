"""The cloud half of the holdout build: semantic hits are replaced from the reserve of the same intent and round."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import holdout_semantic as hs  # noqa: E402


class Fake:
    """Calls a sentence a paraphrase of training data when it contains "alt"."""

    def __init__(self):
        self.seen: list[tuple[str, str]] = []

    def add(self, kind, texts):
        self.seen += [(kind, t) for t in texts]

    def near(self, text):
        return "semantic_training" if "alt" in text else None


def item(i: int, text: str, intent: str = "create_goal", origin: str = "round 2") -> dict:
    return {"id": f"x{i}", "set": "decision", "text": text, "expected_intent": intent, "origin": origin}


def test_a_hit_is_replaced_by_the_next_clean_reserve_sentence_of_its_round():
    sets = {"decision": {"items": [item(1, "neu eins"), item(2, "alt zwei")]}, "shadow": {"items": [item(3, "neu drei")]}}
    reserve = {"items": [item(10, "alt reserve", origin="round 2"), item(11, "neu reserve", origin="round 2"),
                         item(12, "neu runde eins", origin="round 1")]}
    r = hs.check(sets, reserve, Fake(), 0.85, {"training": ["irgendwas"]})
    texts = [i["text"] for i in r["items"]["decision"]]
    assert texts == ["neu eins", "neu reserve"] and r["items"]["decision"][1]["id"] == "x2"
    assert r["complete"] and r["report"]["replaced_per_intent"] == {"create_goal": 1}
    assert r["report"]["hits"] == {"semantic_training": 1, "reserve semantic_training": 1}


def test_other_round_fills_when_the_own_round_is_used_up_and_it_is_reported():
    sets = {"decision": {"items": [item(2, "alt zwei")]}, "shadow": {"items": []}}
    reserve = {"items": [item(12, "neu runde eins", origin="round 1")]}
    r = hs.check(sets, reserve, Fake(), 0.85, {})
    assert r["items"]["decision"][0]["origin"] == "round 1"
    assert r["report"]["filled_from_other_round"] == {"create_goal: round 1 for round 2": 1}


def test_an_empty_reserve_leaves_the_set_short_and_incomplete():
    sets = {"decision": {"items": [item(2, "alt zwei")]}, "shadow": {"items": []}}
    r = hs.check(sets, {"items": []}, Fake(), 0.85, {})
    assert not r["complete"] and r["report"]["short_per_intent"] == {"create_goal": 1}
    assert r["items"]["decision"] == []
