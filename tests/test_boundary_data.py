"""The boundary of creating and instructions in foreign text: files match their generators,
stay away from the holdouts, and keep their contrast pairs.

    python -m pytest tests/test_boundary_data.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_create_boundary  # noqa: E402
import build_foreign_text  # noqa: E402
import check_holdout_leak as lock  # noqa: E402


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_files_match_their_generators():
    assert build_create_boundary.main(["--check"]) == 0
    assert build_foreign_text.main(["--check"]) == 0


def test_nothing_near_a_guarded_holdout():
    guarded = [i for h in lock.GUARDED for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    for path in (build_create_boundary.OUT, build_foreign_text.OUT):
        train = [(f"{path.name}:{n}", r["text"]) for n, r in enumerate(rows(path), 1)]
        assert not lock.find_leaks(train, guarded, 0.85), path.name


def test_the_boundary_has_both_sides():
    got = Counter(r["intent"] for r in rows(build_create_boundary.OUT))
    assert got["unknown"] >= 100 and got["create_appointment"] >= 50 and got["create_task"] >= 30
    assert sum(v for k, v in got.items() if k not in ("unknown", "create_appointment", "create_task")) >= 15


def test_foreign_text_comes_as_contrast_and_keeps_its_kind():
    data = rows(build_foreign_text.OUT)
    assert all(r["kind"] in ("foreign_text", "fake_json", "obfuscated", "role_format") for r in data)
    by = Counter((r["variant"], r["intent"] == "unknown", r["intent"].startswith("create_")) for r in data)
    assert by[("plain", True, False)] >= 100       # foreign text alone: do nothing
    assert by[("create", False, True)] >= 40        # the person's own request to create from it
    assert sum(v for (variant, unk, cr), v in by.items() if variant == "read" and not unk and not cr) >= 40
    assert not any(r["intent"].startswith("create_") for r in data if r["variant"] != "create")
