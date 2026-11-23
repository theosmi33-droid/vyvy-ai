"""Output contract 2: 25 intents, five fields; contract 1 stays frozen."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))
import bench  # noqa: E402

V2 = bench.GRAMMARS[2].read_text(encoding="utf-8")
V1 = bench.GRAMMARS[1].read_text(encoding="utf-8")
INTENTS = bench.grammar_intents(V2)

# The order is the product mapping 1..25; changing it is a new contract.
ORDER = ("appointments_today", "appointments_tomorrow", "appointments_search", "daily_agenda", "tasks_open",
         "tasks_overdue", "tasks_search", "emails_unread", "emails_search", "email_thread_summary", "web_search",
         "help_lookup", "create_appointment", "create_task", "complete_task", "reschedule_task", "delete_task",
         "write_email", "draft_reply", "start_timer", "stop_timer", "create_project", "create_goal",
         "complete_goal", "unknown")


def test_v2_grammar_is_valid_and_lists_the_25_intents_in_order():
    assert bench.check_grammar(V2) == []
    assert INTENTS == ORDER
    assert bench.grammar_fields(V2) == ("intent", "timeframe", "entity", "content", "confidence")
    assert bench.grammar_contract(V2) == 2 and bench.DEFAULT_GRAMMAR == bench.GRAMMARS[2]


def test_every_intent_has_a_definition_and_unknown_stays_mandatory():
    assert tuple(bench.INTENT_HELP) == ORDER
    assert "unknown" in INTENTS


def test_contract_1_is_frozen():
    assert hashlib.sha256(bench.GRAMMARS[1].read_bytes()).hexdigest() == \
        "4246375388ee2ff9973df7af814f99763e22b0a0665972ffc44389a1332d21dd"
    assert bench.grammar_contract(V1) == 1
    # The SHA-256 the catalog recorded for the contract 1 prompts.
    assert {k: hashlib.sha256(v.encode("utf-8")).hexdigest()[:12] for k, v in bench.PROMPTS_V1.items()} == \
        {"compact": "8e462da85b42", "detailed": "478793e00add"}


def test_v2_prompts_differ_from_v1_and_name_every_intent():
    for name, text in bench.PROMPTS.items():
        assert text != bench.PROMPTS_V1[name]
        for intent in ORDER:
            assert intent in text, (name, intent)


def test_parse_accepts_five_fields_in_order_only():
    ok = bench.answer_json("emails_search", 0.9, entity="Frau Albrecht", content="Gerüst")
    assert bench.parse_output(ok, INTENTS) == json.loads(ok)
    four = json.dumps({"intent": "unknown", "timeframe": None, "entity": None, "confidence": 0.9})
    assert bench.parse_output(four, INTENTS) is None
    shuffled = json.dumps({"intent": "unknown", "timeframe": None, "content": None, "entity": None,
                           "confidence": 0.9})
    assert bench.parse_output(shuffled, INTENTS) is None


def test_lengths_are_checked_by_the_validator():
    assert bench.parse_output(bench.answer_json("create_task", 0.9, content="x" * 120), INTENTS)
    assert bench.parse_output(bench.answer_json("create_task", 0.9, content="x" * 121), INTENTS) is None
    assert bench.parse_output(bench.answer_json("create_task", 0.9, entity="y" * 80), INTENTS)
    assert bench.parse_output(bench.answer_json("create_task", 0.9, entity="y" * 81), INTENTS) is None


def test_contract_1_parse_still_works_with_its_fields():
    v1 = json.dumps({"intent": "unknown", "timeframe": None, "entity": "z" * 200, "confidence": 0.9})
    assert bench.parse_output(v1, bench.grammar_intents(V1), bench.FIELDS_BY_CONTRACT[1])


def test_row_fits_defaults_missing_slots_to_null():
    assert bench.row_fits({"intent": "unknown", "timeframe": None, "entity": None}, INTENTS)
    assert bench.row_fits({"intent": "write_email", "content": "Rückruf zum Angebot"}, INTENTS)
    assert not bench.row_fits({"intent": "send_email"}, INTENTS)
    assert not bench.row_fits({"intent": "create_task", "content": 7}, INTENTS)


def test_classify_letters_cover_25_intents():
    _prompt, letters = bench.classify_prompt(INTENTS)
    assert letters["appointments_today"] == "A" and letters["unknown"] == "Y"


def test_dry_run_records_the_contract(tmp_path):
    data = tmp_path / "set.json"
    data.write_text(json.dumps({"items": [{"id": f"q{i}", "text": f"Satz {i}", "expected_intent": intent}
                                          for i, intent in enumerate(ORDER)]}), encoding="utf-8")
    out = tmp_path / "r.json"
    assert bench.main(["--dry-run", "--data", str(data), "--json", str(out)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["contract"] == 2
    assert r["grammar_sha256"] == hashlib.sha256(bench.GRAMMARS[2].read_bytes()).hexdigest()


def test_benchmark_refuses_the_frozen_grammar(tmp_path):
    data = tmp_path / "set.json"
    data.write_text(json.dumps({"items": [{"id": "q", "text": "x", "expected_intent": "unknown"}]}),
                    encoding="utf-8")
    assert bench.main(["--dry-run", "--data", str(data), "--grammar", str(bench.GRAMMARS[1])]) == 1
