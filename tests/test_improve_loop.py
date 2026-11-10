"""Improvement loop: the report reads ids and intents only, the guard stops at every hold point.

    python -m pytest tests/test_improve_loop.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "training"))

import improve_report  # noqa: E402
import loop_guard  # noqa: E402
import patterns  # noqa: E402


def result(answers: dict[str, tuple[str, str]], adapter: str) -> dict:
    items = [{"id": k, "expected": e, "got": g} for k, (e, g) in sorted(answers.items())]
    right = sum(i["got"] == i["expected"] for i in items)
    return {"n": len(items), "mode": "route", "prompt": "compact", "adapter": adapter, "items": items,
            "intent_correct": right, "within_timeout": right, "median_ms": 600}


BASE = result({"q1": ("unknown", "appointments_tomorrow"), "q2": ("appointments_today", "appointments_today"),
               "q3": ("daily_agenda", "daily_agenda"), "q4": ("tasks_open", "tasks_overdue")}, "old")
LORA = result({"q1": ("unknown", "unknown"), "q2": ("appointments_today", "appointments_tomorrow"),
               "q3": ("daily_agenda", "low confidence (0.40, tasks_open)"), "q4": ("tasks_open", "tasks_overdue")},
              "new")


def test_report_lists_b_and_c_and_kinds() -> None:
    r = improve_report.run_report(BASE, LORA)
    assert [x["id"] for x in r["fixed"]] == ["q1"]
    assert [(x["id"], x["got"]) for x in r["broken"]] == [("q2", "appointments_tomorrow"), ("q3", "low confidence")]
    assert r["kinds"] == {"day": 1, "low_confidence": 1, "tasks": 1}
    assert r["false_actions"] == 0


def test_report_never_carries_text() -> None:
    base = json.loads(json.dumps(BASE))
    lora = json.loads(json.dumps(LORA))
    for i in base["items"] + lora["items"]:
        i["text"] = "SECRET SENTENCE"
    r = improve_report.run_report(base, lora)
    plan = improve_report.propose([r, r])
    out = json.dumps(r) + json.dumps(plan) + improve_report.markdown([r], plan)
    assert "SECRET" not in out


def test_proposal_needs_half_of_the_runs() -> None:
    once = improve_report.run_report(BASE, LORA)
    other = improve_report.run_report(BASE, BASE)
    plan = improve_report.propose([once, other, other, other])
    pairs = {(p["meant"], p["mistaken_for"]) for p in plan["near"]}
    assert ("appointments_today", "appointments_tomorrow") not in pairs
    assert ("tasks_open", "tasks_overdue") in pairs
    assert len(plan["weak"]) <= improve_report.WEAK_TOP


def decisions(*accepted: bool, week_start: str = "2026-11-09") -> list[dict]:
    return [{"adapter": f"a{k}", "created": f"{week_start}T{k:02d}:00:00Z", "accepted": a}
            for k, a in enumerate(accepted)]


def guard(history: list[dict], **kw) -> tuple[bool, str]:
    kw.setdefault("now", "2026-11-10T12:00:00Z")
    kw.setdefault("approved", "x")
    kw.setdefault("current", "x")
    return loop_guard.check(history, **kw)


def test_guard_allows_a_round() -> None:
    assert guard(decisions(True, False))[0] is True
    assert guard([])[0] is True


def test_guard_stops_on_changed_rules() -> None:
    go, why = guard([], approved="x", current="y")
    assert not go and "user" in why


def test_guard_goal_needs_both_thresholds_and_one_look() -> None:
    sealed = {"adapter": "a", "n": 270, "correct": 265, "wrong_actions": 2, "looks": 1, "unknown_n": 300,
              "false_actions": 0}
    go, why = guard([], sealed=sealed)
    assert not go and "goal" in why
    assert guard([], sealed=dict(sealed, false_actions=1))[0] is True
    assert guard([], sealed=dict(sealed, correct=264))[0] is True
    assert guard([], sealed=sealed, min_sealed=1000)[0] is True
    assert guard([], sealed=sealed, errors_unknown_only=True)[0] is True
    assert guard([], sealed=dict(sealed, wrong_actions=0), errors_unknown_only=True)[0] is False
    assert "more than once" in guard([], sealed=dict(sealed, looks=2))[1]


def test_guard_rotation_plateau_and_week() -> None:
    assert "rotate" in guard(decisions(False, True))[1]
    assert "plateau" in guard(decisions(True, False, False, False))[1]
    assert "rounds" in guard(decisions(False, False, False, False, False), plateau=99)[1]
    assert guard(decisions(False, False, False, False, False), plateau=99, now="2026-11-16T12:00:00Z")[0] is True


def test_fingerprint_follows_rule_and_holdouts(tmp_path: Path) -> None:
    for rel in loop_guard.RULE_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("rule")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "holdout_blind_v9.json").write_text("[]")
    before = loop_guard.fingerprint(tmp_path)
    (tmp_path / "data" / "holdout_blind_v10.json").write_text("[]")
    assert loop_guard.fingerprint(tmp_path) != before


def test_patterns_render_and_keep_the_holdout_word_out() -> None:
    for key in patterns.PATTERNS:
        prompt = patterns.pattern_prompt(key, "Kölsch", "Schreiner", 8)
        assert "{" not in prompt.split("JSON-Array")[0]
    words = " ".join(patterns.MORNING_COMPOUNDS) + json.dumps(patterns.PATTERNS, ensure_ascii=False)
    assert "Morgenüberblick" not in words


def test_user_forms_must_be_abstract(tmp_path: Path) -> None:
    good = {"category": "others_calendar", "form": "Hat {PERSON} {TAG} noch {TERMINWORT}?", "intent": "unknown"}
    assert patterns.check_form(good) == []
    assert patterns.check_form(dict(good, form="Hat Frau Beispiel morgen Zeit?"))
    assert patterns.check_form(dict(good, form="Ruf {PERSON} unter 0171 an"))
    assert patterns.check_form(dict(good, form="{NAME} morgen?"))
    assert patterns.check_form(dict(good, intent="book_flight"))
    path = tmp_path / "forms.json"
    path.write_text(json.dumps({"schema": 1, "forms": [good, dict(good, form="ohne Platzhalter")]}))
    try:
        patterns.load_forms(path)
        raise AssertionError("a form without placeholder must refuse the file")
    except ValueError as e:
        assert "form 2" in str(e)
    prompt = patterns.form_prompt([good], "Schreiner", 6)
    assert "{PERSON}" in prompt and "erfundene" in prompt


def test_region_priorities() -> None:
    w = patterns.region_weights()
    assert w["Bairisch/Österreichisch"] == w["Schwäbisch"] == w["Schweizerdeutsch"] == 3
    assert w["Kölsch/Rheinisch"] == 2 and w["Norddeutsch/Platt"] == 1
    words = {x for r in patterns.REGIONS.values() for k in ("today", "tomorrow", "other") for x in r[k]}
    assert patterns.GAPS <= words


def test_sealed_set_is_deterministic_labelled_and_clean() -> None:
    import sealed_gen
    grammar = {"schema": 1,
               "words": {"T": ["heute", "heid", "hück"], "W": ["Termine", "Besprechungen", "Einträge"],
                         "X": ["Wetter", "Bus", "Bäcker"], "Z": ["früh", "mittags", "abends"]},
               "frames": [{"intent": "appointments_today", "text": "welche {W} hab ich {T} {Z}?"},
                          {"intent": "unknown", "text": "wie ist das {X} {T} {Z}?"}]}
    refs = ["Welche Termine hab ich heute früh?"]
    a = sealed_gen.generate(grammar, "s1", 8, refs)
    assert a == sealed_gen.generate(grammar, "s1", 8, refs)
    assert a != sealed_gen.generate(grammar, "s2", 8, refs)
    assert all(("{" not in x["text"]) for x in a)
    assert {x["expected"] for x in a} == {"appointments_today", "unknown"}
    assert sealed_gen.Near(refs).hit(refs[0]) and not any(x["text"] == refs[0] for x in a)


def test_loop_stops_after_every_round() -> None:
    small = "Qwen/Qwen2.5-1.5B-Instruct"
    hist = [{"adapter": "vyvy-1.5b-d1", "created": "2026-11-12T10:00:00Z", "accepted": False}]
    now = "2026-11-12T12:00:00Z"
    assert loop_guard.round_allowed(None, hist, small, now)[0] is False
    once = {"mode": "one_round", "approved_on": "2026-11-11T00:00:00Z", "bases": [small]}
    assert "round done" in loop_guard.round_allowed(once, hist, small, now)[1]
    assert loop_guard.round_allowed(dict(once, approved_on="2026-11-12T11:00:00Z"), hist, small, now)[0] is True
    assert loop_guard.round_allowed(once, hist, loop_guard.SEVEN_B, now)[0] is False
    weekly = dict(once, mode="weekly")
    assert "next week" in loop_guard.round_allowed(weekly, hist, small, now)[1]
    assert loop_guard.round_allowed(weekly, hist, small, "2026-11-16T08:00:00Z")[0] is True


def test_seven_b_needs_a_promising_small_run() -> None:
    assert loop_guard.seven_b_ready(None)[0] is False
    assert loop_guard.seven_b_ready({"p_value": 0.0835, "per_intent": {"unknown": {"broken": 0}}})[0] is True
    assert loop_guard.seven_b_ready({"p_value": 0.0038, "per_intent": {"unknown": {"broken": 1}}})[0] is False
    assert loop_guard.seven_b_ready({"p_value": 0.32, "per_intent": {}})[0] is False


def test_gate_blocks_without_release_and_on_changed_rules() -> None:
    small = "Qwen/Qwen2.5-1.5B-Instruct"
    now = "2026-11-12T12:00:00Z"
    paused = json.loads((ROOT / "data" / "loop_state.json").read_text(encoding="utf-8"))
    assert paused["mode"] == "paused"
    assert loop_guard.gate(paused, [], small, now, "x")[0] is False
    once = {"mode": "one_round", "approved_on": "2026-11-11T00:00:00Z", "bases": [small, loop_guard.SEVEN_B],
            "fingerprint": "x"}
    assert loop_guard.gate(once, [], small, now, "x")[0] is True
    assert "rules" in loop_guard.gate(once, [], small, now, "y")[1]
    assert loop_guard.gate(once, [], loop_guard.SEVEN_B, now, "x")[0] is False
    good = {"p_value": 0.08, "per_intent": {"unknown": {"broken": 0}}}
    assert loop_guard.gate(once, [], loop_guard.SEVEN_B, now, "x", good)[0] is True


def test_gate_in_weekly_mode_counts_only_decisions_since_the_release() -> None:
    small = "Qwen/Qwen2.5-1.5B-Instruct"
    old = [{"adapter": f"vyvy-7b-d{k}", "created": f"2026-11-0{k}T10:00:00Z", "accepted": False} for k in range(3, 7)]
    weekly = {"mode": "weekly", "approved_on": "2026-11-09T00:00:00Z", "bases": [small], "fingerprint": "x"}
    assert loop_guard.gate(weekly, old, small, "2026-11-12T12:00:00Z", "x")[0] is True
    new = [{"adapter": f"vyvy-7b-e{k}", "created": f"2026-11-1{k}T10:00:00Z", "accepted": False} for k in range(0, 3)]
    assert "plateau" in loop_guard.gate(weekly, old + new, small, "2026-11-16T12:00:00Z", "x")[1]


def test_plan_train_runs_small_first() -> None:
    import plan_train
    assert plan_train.ORDER[0] == "Qwen/Qwen2.5-1.5B-Instruct"
