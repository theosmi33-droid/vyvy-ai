#!/usr/bin/env python3
"""Build the boundary of creating: requests that sound like creating but are not, with contrast pairs.

The baseline on the sealed set create-v1 (21.11.2026, adapter vyvy-1.5b-d1e100e8d,
totals only) answered 23 of 40 "unknown" traps with create_appointment or
create_task, took 10 of 80 appointment requests for a task and 16 of 160 create
requests for a read question. The sealed set's labelling rule says what counts as
unknown: questions ABOUT creating, creating for someone else, statements and wishes
without a request, and changing, moving, cancelling, deleting, ticking off or asking
about something that already exists. This file builds those four groups and, for
each, contrast pairs that use the same words but are a real request to create -
so the model learns the boundary, not a keyword - plus pairs appointment against
task and request against question.

Built from the categories only; no sentence of a sealed set is known here, and no
single result of one is used. Every sentence goes through the filter of
tools/build_unknown_hard.py (lexical lock against every holdout at 0.8, the
training data, the unknown boundary at 0.85); the semantic check against the
sealed sets runs in the Sealed check workflow, which names the training line of a
hit so that the line, not the sealed sentence, is dropped. Settings are mixed
(office, practice, shop, club, workshop, law firm); persons and companies are
invented.

    python tools/build_create_boundary.py
    python tools/build_create_boundary.py --check
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402
import build_unknown_hard as hard  # noqa: E402

OUT = ROOT / "data" / "train" / "vyvy_create_boundary_v1.jsonl"
QUOTAS = {"unknown": 130, "create_appointment": 70, "create_task": 45, "appointments_tomorrow": 8,
          "appointments_today": 8, "tasks_open": 7}

# Things one makes an appointment for or writes down as a task, in several trades.
MEETINGS = ["die Besprechung mit Frau Albers", "den Kundentermin bei der Firma Rohleder", "die Vorstandssitzung",
            "den Termin mit Herrn Okafor", "die Abnahme in der Werkstatt", "das Gespräch mit der Steuerberaterin",
            "die Probe vom Chor", "die Anprobe mit Frau Szabo"]
TASKS = ["die Rechnung an Bäckerei Lindner schreiben", "das Angebot für den Zaun rausschicken",
         "die Kasse zählen", "den Mitgliedsbeitrag einziehen", "die Akte Brenner kopieren",
         "Druckerpapier bestellen", "die Waage eichen lassen", "das Rezept für Herrn Uysal vorbereiten",
         "die Schlüssel für die Halle nachmachen lassen", "den Wartungsvertrag kündigen",
         "die Fotos fürs Schaufenster aussuchen", "den Getränkelieferanten anrufen",
         "die Vollmacht von Frau Haug einscannen", "den Ölwechsel beim Transporter eintragen lassen",
         "die Spendenquittungen verschicken", "das Wartezimmer aufräumen"]
TIMES = [("morgen um zehn", "tomorrow"), ("heute um drei", "today"), ("nächste Woche am Dienstag", "week"),
         ("am Freitag um neun", None), ("morgen früh", "tomorrow"), ("heute Nachmittag", "today")]
OTHERS = [("Frau Wendt", "von Frau Wendt"), ("Herrn Kaya", "von Herrn Kaya"), ("dem Azubi", "vom Azubi"),
          ("der Kollegin am Empfang", "der Kollegin am Empfang"), ("dem Vereinskassierer", "vom Vereinskassierer"),
          ("dem ganzen Team", "vom Team")]


def cap(text: str) -> str:
    return text[0].upper() + text[1:]


def about() -> list[str]:
    """(1) Questions ABOUT creating: how, where, whether, who may."""
    out = ["Wie lege ich hier einen Termin an?", "Wo kann ich eine neue Aufgabe eintragen?",
           "Kann man Aufgaben auch mit Erinnerung anlegen?", "Darf die Aushilfe selbst Termine eintragen?",
           "Geht es, einen Termin für mehrere Leute anzulegen?", "Wie trage ich einen Termin für jede Woche ein?",
           "Kann ich Aufgaben einer Farbe zuordnen?", "Wer darf bei uns Termine im gemeinsamen Kalender anlegen?",
           "Muss ich für jede Aufgabe ein Fälligkeitsdatum angeben?", "Kann man einen Termin auch ohne Uhrzeit eintragen?",
           "Wie viele Aufgaben kann man höchstens anlegen?", "Gibt es eine Vorlage für Termine mit Kunden?",
           "Wie leg i a Aufgob o, des hob i nia kapiert", "Wo trog i an Termin ei, im Handy oder am PC?",
           "Wie jeht dat, en Termin anzulegen?", "Chan mer do au Ufgabe mit Erinnerig mache?"]
    for m in MEETINGS[:4]:
        out.append(f"Wie trage ich {m} am besten ein?")
    for t in TASKS[:4]:
        out.append(f"Soll ich {t} als Aufgabe oder als Termin anlegen?")
    return out


def others() -> list[str]:
    """(2) Creating in someone else's calendar or list, or for a team."""
    out = []
    for who, of in OTHERS:
        for when, _ in TIMES[:3]:
            out.append(f"Trag im Kalender {of} {when} {MEETINGS[len(out) % len(MEETINGS)]} ein")
        out.append(f"Leg {who} die Aufgabe an, {TASKS[len(out) % len(TASKS)]}")
        out.append(f"Schreib {who} auf die Liste, {TASKS[(len(out) + 3) % len(TASKS)]}")
    out += ["Trog bei da Chefin morgn den Termin mitm Elektriker ei", "Schreib dem Mario auf, dass er d Kasse zählt",
            "Mach em Kassierer e Ufgab, de Beitrag iizieh"]
    return out


def statements() -> list[str]:
    """(3) Statements and wishes without a request."""
    out = []
    for m in MEETINGS:
        out.append(f"{cap(m)} ist übrigens {TIMES[len(out) % len(TIMES)][0]}.")
    for t in TASKS:
        out.append(f"Irgendwann müsste man mal {t}.")
        out.append(f"Eigentlich wollte ich heute {t}.")
    out += ["Ich hätte gern mal einen Tag ohne Termine.", "Die Aufgabenliste wird immer länger.",
            "Morgen wird wieder ein voller Tag.", "Mei, so vui Termine heid.", "Dat Wochenende is endlich frei."]
    return out


def existing() -> list[str]:
    """(4) Changing, moving, cancelling, deleting, ticking off, asking about something existing."""
    out = []
    for m in MEETINGS:
        out += [f"Verschieb {m} auf {TIMES[len(out) % len(TIMES)][0]}", f"Sag {m} ab",
                f"Ist {m} schon eingetragen?"]
    for t in TASKS:
        out += [f"Hak die Aufgabe „{t}“ ab", f"Lösch die Aufgabe „{t}“", f"Markier „{t}“ als erledigt"]
    out += ["Kannst du den Termin mit dem Elektriker eine Stunde später legen?",
            "Streich die Aufgabe mit dem Druckerpapier wieder", "Verleg die Probe vom Chor auf nächste Woche",
            "Hok de Kassn ob, de hob i scho zählt", "Mach de Termin mit em Okafor weg"]
    return out


def creates() -> dict[str, dict[str, list[tuple[str, str | None]]]]:
    """Real requests to create, with the same words as the groups above: (text, timeframe)."""
    appt, task = [], []
    for i, m in enumerate(MEETINGS):
        for when, tf in TIMES:
            appt.append((f"Trag mir {when} {m} ein", tf))
            appt.append((f"Leg einen Termin an: {m}, {when}", tf))
        appt.append((f"Mach mir für {m} einen Termin, {TIMES[i % len(TIMES)][0]}", TIMES[i % len(TIMES)][1]))
    for t in TASKS:
        task.append((f"Leg mir die Aufgabe an, {t}", None))
        task.append((f"Schreib mir auf, dass ich {t} muss", None))
        task.append((f"Neue Aufgabe: {t}", None))
        task.append((f"Erinner mich bis {TIMES[2][0].replace('nächste Woche am ', '')}, {t}", None))
    appt += [("Trog ma morgn um zehne an Termin mitm Elektriker ei", "tomorrow"),
             ("Mach ma heid um drei an Termin mit da Steuerberaterin", "today")]
    task += [("Schreib ma auf, i muass d Kassn zählen", None), ("Mach mer e Ufgab: Papier bstelle", None)]
    return {"create_appointment": {"appt": appt}, "create_task": {"task": task}}


def questions() -> dict[str, list[tuple[str, str]]]:
    """Read questions with the same words as the create requests: (text, intent)."""
    out = {"appointments_tomorrow": [], "appointments_today": [], "tasks_open": []}
    for m in MEETINGS:
        out["appointments_tomorrow"].append((f"Hab ich morgen {m}?", "tomorrow"))
        out["appointments_today"].append((f"Wann ist heute {m}?", "today"))
    # Questions about the whole list, not about one existing task (that is unknown by the rule).
    for place in ("in der Praxis", "im Laden", "im Verein", "in der Werkstatt", "in der Kanzlei", "im Büro"):
        out["tasks_open"] += [(f"Was hab ich {place} noch alles zu erledigen?", None),
                              (f"Welche Aufgaben sind {place} noch offen?", None)]
    out["appointments_tomorrow"] += [("Hob i morgn an Termin mitm Elektriker?", "tomorrow")]
    return out


def build(seed: int, max_similarity: float) -> tuple[list[dict], dict[str, int], list[tuple[str, str]]]:
    rng = random.Random(f"{seed}-create-boundary")
    keep = hard.Filter(max_similarity, own=(OUT,))
    rows, counts = [], {}
    groups = {"about": [(t, None) for t in about()], "others": [(t, None) for t in others()],
              "statements": [(t, None) for t in statements()], "existing": [(t, None) for t in existing()]}
    picked = hard.draw(groups, QUOTAS["unknown"], rng, keep, "unknown")
    counts["unknown"] = len(picked)
    rows += [{"text": t, "intent": "unknown", "timeframe": None, "entity": None} for t, _ in picked]
    for intent, g in creates().items():
        got = hard.draw(g, QUOTAS[intent], rng, keep, intent)
        counts[intent] = len(got)
        rows += [{"text": t, "intent": intent, "timeframe": tf, "entity": None} for t, tf in got]
    for intent, pairs in questions().items():
        got = hard.draw({"q": pairs}, QUOTAS[intent], rng, keep, intent)
        counts[intent] = len(got)
        rows += [{"text": t, "intent": intent, "timeframe": tf, "entity": None} for t, tf in got]
    short = {k: v for k, v in counts.items() if v < QUOTAS[k]}
    if short:
        raise SystemExit(f"too few sentences left after filtering: {short} of {QUOTAS}")
    rng.shuffle(rows)
    return rows, counts, keep.dropped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the boundary of creating with contrast pairs")
    p.add_argument("--seed", type=int, default=48)
    p.add_argument("--max-similarity", type=float, default=0.8)
    p.add_argument("--check", action="store_true", help="only compare with the committed file")
    args = p.parse_args(argv)
    intents = bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))
    rows, counts, dropped = build(args.seed, args.max_similarity)
    for r in rows:
        answer = json.dumps({**{k: r[k] for k in ("intent", "timeframe", "entity")}, "confidence": 0.9})
        if bench.parse_output(answer, intents) is None:
            raise SystemExit(f"not in the output contract: {r}")
    text = hard.render(rows)
    if args.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print(f"{OUT.relative_to(ROOT)} {'matches' if same else 'differs from'} the generator")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"{len(rows)} sentences -> {OUT.relative_to(ROOT)}")
    print(json.dumps(counts, indent=1))
    print(f"dropped {len(dropped)}")
    for t, why in dropped:
        print(f"  dropped ({why}): {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
