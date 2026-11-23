"""bench-lora measures the decision set in parts; the parts join into one results file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))
sys.path.insert(0, str(ROOT / "tools"))

import bench  # noqa: E402
import compare_adapter  # noqa: E402
import merge_results  # noqa: E402

INTENTS = bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))


def _set(tmp_path: Path, n: int = 50) -> Path:
    data = tmp_path / "decision.json"
    data.write_text(json.dumps({"items": [{"id": f"q{i}", "text": f"Satz Nummer {i}",
                                           "expected_intent": INTENTS[i % len(INTENTS)]} for i in range(n)]}),
                    encoding="utf-8")
    return data


def test_parts_are_disjoint_and_cover_the_set():
    items = [{"id": i} for i in range(11)]
    parts = [bench.part_items(items, f"{i}/4") for i in range(1, 5)]
    ids = [x["id"] for p in parts for x in p]
    assert sorted(ids) == list(range(11)) and len(ids) == len(set(ids))
    with pytest.raises(ValueError):
        bench.part_items(items, "5/4")


def _run_parts(tmp_path: Path, data: Path, m: int, prefix: str) -> list[Path]:
    out = []
    for i in range(1, m + 1):
        path = tmp_path / f"{prefix}_p{i}.json"
        assert bench.main(["--dry-run", "--data", str(data), "--part", f"{i}/{m}", "--json", str(path)]) == 0
        out.append(path)
    return out


def test_joined_parts_equal_one_run_over_the_whole_set(tmp_path):
    data = _set(tmp_path)
    whole = tmp_path / "whole.json"
    assert bench.main(["--dry-run", "--data", str(data), "--json", str(whole)]) == 0
    merged = merge_results.merge([json.loads(p.read_text()) for p in _run_parts(tmp_path, data, 4, "base")])
    one = json.loads(whole.read_text())
    assert merged["n"] == one["n"] == 50 and merged["parts"] == 4
    assert {i["id"] for i in merged["items"]} == {i["id"] for i in one["items"]}
    assert merged["contract"] == 2 and merged["grammar_sha256"] == one["grammar_sha256"]
    # The rule reads the joined files like a single run.
    v = compare_adapter.decide(merged, merged)
    assert v["n"] == 50 and v["b_fixed"] == v["c_broken"] == 0


def test_a_missing_or_doubled_part_is_refused(tmp_path):
    data = _set(tmp_path)
    parts = [json.loads(p.read_text()) for p in _run_parts(tmp_path, data, 4, "base")]
    with pytest.raises(ValueError, match="incomplete or twice"):
        merge_results.merge(parts[:3])
    with pytest.raises(ValueError, match="incomplete or twice"):
        merge_results.merge(parts + [parts[0]])


def test_parts_of_different_runs_are_refused(tmp_path):
    data = _set(tmp_path)
    parts = [json.loads(p.read_text()) for p in _run_parts(tmp_path, data, 2, "base")]
    parts[1]["adapter"] = "other-baseline.gguf"
    with pytest.raises(ValueError, match="differ in adapter"):
        merge_results.merge(parts)
    parts[1]["adapter"] = parts[0]["adapter"]
    parts[1]["grammar_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differ in grammar_sha256"):
        merge_results.merge(parts)


def test_main_exits_2_on_a_bad_join(tmp_path):
    data = _set(tmp_path)
    paths = _run_parts(tmp_path, data, 3, "lora")
    assert merge_results.main([str(paths[0]), str(paths[1]), "--out", str(tmp_path / "x.json")]) == 2
    assert merge_results.main([*map(str, paths), "--out", str(tmp_path / "x.json")]) == 0
