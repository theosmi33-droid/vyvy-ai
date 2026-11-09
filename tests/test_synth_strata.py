"""Teacher rounds in strata, and the import's semantic guard against holdout paraphrases.

    python -m pytest tests/test_synth_strata.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "training"))

import import_synth  # noqa: E402
import synthesize  # noqa: E402


def test_every_style_has_a_stratum() -> None:
    assert set(synthesize.STYLE_STRATUM) == set(synthesize.STYLES)
    assert set(synthesize.STYLE_STRATUM.values()) == {1, 2, 3}
    assert synthesize.STYLE_STRATUM["kurzes, sauberes Hochdeutsch"] == 1
    assert synthesize.STYLE_STRATUM["Schwäbisch, so wie man es spricht"] == 3


def test_hard_cases_are_stratum_4() -> None:
    plain = synthesize.STYLES[0]
    assert synthesize.stratum("create_task", plain, None) == 1
    assert synthesize.stratum("create_task", plain, "create_appointment") == 4
    assert synthesize.stratum("unknown", plain, None) == 4


def test_dialect_prompts_ask_for_common_spellings() -> None:
    for style, layer in synthesize.STYLE_STRATUM.items():
        prompt = synthesize.gen_prompt("tasks_open", style, "Schreiner", 4, None)
        assert (synthesize.DIALECT_SPELLING in prompt) == (layer == 3), style


def run_import(tmp_path: Path, rows: list[dict], scores: dict[str, float]) -> tuple[list[dict], dict]:
    folder, train = tmp_path / "synth", tmp_path / "train"
    folder.mkdir()
    train.mkdir()
    (folder / "synth.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                        encoding="utf-8")
    (folder / "synth_manifest.json").write_text(json.dumps({"script_sha256": "f" * 64, "round": "2026-11-14"}),
                                                encoding="utf-8")
    seen: list[list[str]] = []

    def cosine(texts: list[str], holdout: list[str]) -> list[float]:
        seen.append(holdout)
        return [scores[t] for t in texts]

    assert import_synth.main([str(folder), "--train-dir", str(train)], cosine=cosine) == 0
    assert seen and len(seen[0]) > 600  # compared with every holdout set
    out = train / "vyvy_synth_fffffffff_2026-11-14.jsonl"
    record = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()], record


def test_semantic_guard_drops_close_paraphrases(tmp_path: Path) -> None:
    rows = [{"text": "Leg mir für Donnerstag die Inventur der Werkstatt als Aufgabe an", "intent": "create_task",
             "timeframe": None, "entity": None, "stratum": 1},
            {"text": "Hob i no wos offn beim Gerüstbau Brandl", "intent": "tasks_open", "timeframe": None,
             "entity": "Gerüstbau Brandl", "stratum": 3},
            {"text": "Wie heißt die Hauptstadt von Island", "intent": "unknown", "timeframe": None,
             "entity": None, "stratum": 9}]
    scores = {rows[0]["text"]: 0.86, rows[1]["text"]: 0.5, rows[2]["text"]: 0.849}
    kept, record = run_import(tmp_path, rows, scores)
    assert [r["text"] for r in kept] == [rows[1]["text"], rows[2]["text"]]
    assert record["dropped_semantic"] == 1 and record["max_cosine"] == 0.85
    assert kept[0]["stratum"] == 3 and "stratum" not in kept[1]  # unknown strata are not carried over
    assert record["per_stratum_imported"] == {"1": 0, "2": 0, "3": 1, "4": 0}
