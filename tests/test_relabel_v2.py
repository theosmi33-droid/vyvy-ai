"""Contract 1 lines are relabelled for contract 2 by fixed rules (tools/relabel_v2.py), never by hand."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402
import relabel_v2 as rl  # noqa: E402


def test_committed_relabelling_matches_rules_and_data():
    assert rl.main(["--check"]) == 0


@pytest.mark.parametrize("text, want", [
    ("Hak die Aufgabe Gerüst abbauen ab.", "complete_task"),
    ("Lösch die Aufgabe Zaun streichen.", "delete_task"),
    ("Verschieb die Aufgabe Angebot auf Freitag.", "reschedule_task"),
    ("Schreib eine neue Mail an die Spedition wegen der Lieferung.", "write_email"),
    ("Entwirf eine Antwort an die Gärtnerei zu dem Angebot.", "draft_reply"),
    ("Starte die Zeit für den Badumbau.", "start_timer"),
    ("Stemple mich aus.", "stop_timer"),
    ("Leg ein neues Projekt Kellerausbau an.", "create_project"),
    ("Neues Ziel: jeden Monat drei Angebote mehr.", "create_goal"),
    ("Brauche ich morgen einen Schirm?", "web_search"),
    ("Wo stelle ich im Programm die Erinnerung ein?", "help_lookup"),
    # Stays unknown: sending, paying, moving an appointment, fixed knowledge, statements, other companies.
    ("Schick die Mail an Herrn Kraus ab.", None),
    ("Bitte überweise das Geld an die Firma Lenz.", None),
    ("Verschieb die Besprechung mit Frau Albers auf morgen.", None),
    ("Was bedeutet Skonto?", None),
    ("Morgen fahr ich mit dem Zug.", None),
    ("Starte einen Timer für zehn Minuten.", None),
    ("Wie formatiere ich in Word eine Tabelle?", None),
])
def test_unknown_lines(text, want):
    new, _rule, _competing = rl.classify({"text": text, "intent": "unknown"})
    assert new == want


@pytest.mark.parametrize("text, want", [
    ("Was steht heute an?", "daily_agenda"),
    ("Was is heut so los bei mir?", "daily_agenda"),
    ("Heutiges Programm?", "daily_agenda"),
    ("Welche Termine habe ich heute?", None),        # narrows to appointments
    ("Bin ich heute noch bei Herrn Demirci?", None),  # a person: an appointment
    ("Was steht heute in meinem Kalender an?", None),
    ("Was hab ich heute alles zu tun?", None),        # asks for work: listed for review, not moved
])
def test_appointments_today_lines(text, want):
    new, _rule, _competing = rl.classify({"text": text, "intent": "appointments_today"})
    assert new == want


def test_only_contract_1_lines_and_only_one_rule(tmp_path):
    lines = [{"text": "Lösch die Aufgabe Zaun streichen.", "intent": "unknown"},
             {"text": "Lösch die Aufgabe Zaun streichen.", "intent": "unknown", "content": None},
             {"text": "Was steht heute an?", "intent": "appointments_today", "timeframe": "today"}]
    (tmp_path / "a.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines),
                                      encoding="utf-8")
    result = rl.build(tmp_path)
    assert [(c["line"], c["to"]) for c in result["changes"]] == [(1, "delete_task"), (3, "daily_agenda")]
    for c in result["changes"]:
        assert c["to"] in bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))


def test_apply_stops_on_a_changed_line(tmp_path):
    row = {"text": "Lösch die Aufgabe Zaun streichen.", "intent": "unknown"}
    overlay = {("a.jsonl", 1): {"file": "a.jsonl", "line": 1, "text_sha256": rl.text_sha(row["text"]),
                                "from": "unknown", "to": "delete_task"}}
    assert rl.apply(overlay, "a.jsonl", 1, row)["intent"] == "delete_task"
    assert rl.apply(overlay, "a.jsonl", 2, row) == row
    with pytest.raises(SystemExit, match="out of date"):
        rl.apply(overlay, "a.jsonl", 1, {**row, "text": "Lösch die Aufgabe Dach decken."})


def test_every_moved_line_is_still_in_the_committed_data():
    for (file, no), c in rl.load().items():
        line = (rl.TRAIN_DIR / file).read_text(encoding="utf-8").splitlines()[no - 1]
        row = json.loads(line)
        assert rl.text_sha(row["text"]) == c["text_sha256"] and row["intent"] == c["from"]
