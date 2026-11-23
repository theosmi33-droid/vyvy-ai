"""The acceptance rule for contract 2 (tools/compare_adapter.py), one test per change."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import compare_adapter as ca  # noqa: E402


def result(rows: list[tuple[str, str, str]], **meta) -> dict:
    """rows: (id, expected, got)."""
    return {"mode": "route", "prompt": "compact", "n": len(rows), "contract": 2, "grammar_sha256": "a" * 64,
            "within_timeout": 0, "median_ms": 0, "items": [{"id": i, "expected": e, "got": g} for i, e, g in rows],
            **meta}


def test_runs_of_different_contracts_are_not_comparable():
    rows = [("q1", "unknown", "unknown")]
    with pytest.raises(ValueError, match="contract differs"):
        ca.decide(result(rows, contract=1), result(rows, contract=2))
    with pytest.raises(ValueError, match="grammar_sha256 differs"):
        ca.decide(result(rows), result(rows, grammar_sha256="b" * 64))
    assert ca.decide(result(rows), result(rows))["contract"] == 2


def _intent_rows(intent: str, n: int, fixed: int, broken: int, start: int = 0) -> tuple[list, list]:
    """n questions of one intent: `fixed` wrong before and right after, `broken` the other way round."""
    base, lora = [], []
    for i in range(n):
        qid = f"{intent}-{start + i}"
        if i < fixed:
            base.append((qid, intent, "unknown_wrong"))
            lora.append((qid, intent, intent))
        elif i < fixed + broken:
            base.append((qid, intent, intent))
            lora.append((qid, intent, "unknown_wrong"))
        else:
            base.append((qid, intent, intent))
            lora.append((qid, intent, intent))
    return base, lora


def _with_overall_gain(intent_base: list, intent_lora: list) -> tuple[dict, dict]:
    """Add 30 questions of another intent the adapter fixed, so McNemar and b >= 2c pass on their own."""
    gb, gl = _intent_rows("tasks_open", 40, 30, 0)
    return result(intent_base + gb), result(intent_lora + gl)


@pytest.mark.parametrize("fixed, broken, fails", [
    (0, 2, False),  # net 2 = tolerance at n=40: never failed
    (0, 3, False),  # net 3 > 2, p = 0.125: noise, no longer fails
    (0, 4, False),  # net 4, p = 0.0625
    (0, 5, True),   # net 5, p = 0.03125 < 0.05: fails
    (1, 6, False),  # net 5, p = 0.0625
    (1, 7, True),   # net 6, p = 0.0352
])
def test_an_intent_fails_only_above_tolerance_and_binomially_noticeable(fixed, broken, fails):
    ib, il = _intent_rows("create_goal", 40, fixed, broken)
    v = ca.decide(*_with_overall_gain(ib, il))
    assert any(r.startswith("create_goal lost") for r in v["reason"].split("; ")) == fails
    assert v["per_intent"]["create_goal"]["loss_p"] == round(ca.intent_loss_p(fixed, broken), 6)


def test_unknown_stays_hard():
    ib, il = _intent_rows("unknown", 160, 0, 1)
    v = ca.decide(*_with_overall_gain(ib, il))
    assert not v["accepted"] and "safety intent unknown got worse on 1" in v["reason"]


def test_mcnemar_and_two_to_one_are_unchanged():
    ib, il = _intent_rows("create_task", 40, 4, 2)
    v = ca.decide(result(ib), result(il))
    assert not v["accepted"] and "not significant" in v["reason"]
    ib, il = _intent_rows("create_task", 40, 0, 0)
    gb, gl = _intent_rows("tasks_open", 80, 30, 16)
    v = ca.decide(result(ib + gb), result(il + gl))
    assert "net gain too small (b=30 < 2*c=32)" in v["reason"]
