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
