"""Partner import checks output contract 2: 25 intents, entity and content as pieces of the sentence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import import_partner as ip  # noqa: E402


def _run(tmp_path: Path, monkeypatch, rows: list[dict]) -> tuple[int, Path]:
    monkeypatch.setattr(ip.lock, "HOLDOUTS", [])
    src = tmp_path / "partner.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    train = tmp_path / "train"
    train.mkdir()
    return ip.main([str(src), "--train-dir", str(train)]), train


def _written(train: Path) -> tuple[list[dict], dict]:
    out = next(train.glob("vyvy_partner_*.jsonl"))
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return rows, json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))


def test_contract_2_rows_are_imported_with_content(tmp_path, monkeypatch):
    rows = [
        {"text": "Hak die Aufgabe Kran bestellen ab", "intent": "complete_task", "timeframe": None,
         "entity": None, "content": "Kran bestellen"},
        {"text": "Leg das Projekt Dachsanierung West an", "intent": "create_project", "timeframe": None,
         "entity": "Dachsanierung West", "content": None},
        {"text": "Wie viel ist neun mal sieben", "intent": "unknown", "timeframe": None, "entity": None,
         "content": None},
    ]
    code, train = _run(tmp_path, monkeypatch, rows)
    assert code == 0
    got, report = _written(train)
    assert [r["intent"] for r in got] == ["complete_task", "create_project", "unknown"]
    assert got[0]["content"] == "Kran bestellen"
    assert report["contract"] == 2 and report["per_intent"]["complete_task"] == 1
    assert len(report["per_intent"]) == 25


def test_slots_not_written_in_the_sentence_are_dropped(tmp_path, monkeypatch):
    rows = [
        {"text": "Schreib eine Mail an die Spedition", "intent": "write_email", "timeframe": None,
         "entity": "Spedition Huber", "content": None},
        {"text": "Such die Mails zum Gerüst", "intent": "emails_search", "timeframe": None, "entity": None,
         "content": "Gerüstbau"},
        {"text": "Such die Mails zum Estrich", "intent": "emails_search", "timeframe": None, "entity": None,
         "content": "Estrich"},
    ]
    code, train = _run(tmp_path, monkeypatch, rows)
    assert code == 0
    got, report = _written(train)
    assert [r["content"] for r in got] == ["Estrich"]
    assert report["dropped"]["slot_not_in_text"] == 2


def test_contract_violations_are_dropped(tmp_path, monkeypatch):
    long = "x" * 121
    rows = [
        {"text": "Schick die Mail ab", "intent": "send_email", "timeframe": None, "entity": None, "content": None},
        {"text": f"Notier {long}", "intent": "create_task", "timeframe": None, "entity": None, "content": long},
        {"text": "Starte die Zeit", "intent": "start_timer", "timeframe": "yesterday", "entity": None,
         "content": None},
    ]
    code, train = _run(tmp_path, monkeypatch, rows)
    assert code == 0
    got, report = _written(train)
    assert got == [] and report["dropped"]["contract"] == 3


def test_a_contract_1_export_is_refused_as_a_whole(tmp_path, monkeypatch):
    rows = [{"text": "Hak die Aufgabe ab", "intent": "unknown", "timeframe": None, "entity": None}]
    code, train = _run(tmp_path, monkeypatch, rows)
    assert code == 4
    assert not list(train.iterdir())


def test_personal_data_in_content_is_dropped(tmp_path, monkeypatch):
    rows = [{"text": "Schreib an info@beispiel.de wegen Termin", "intent": "write_email", "timeframe": None,
             "entity": None, "content": "info@beispiel.de"}]
    code, train = _run(tmp_path, monkeypatch, rows)
    assert code == 0
    _got, report = _written(train)
    assert report["dropped"]["personal"] == 1
