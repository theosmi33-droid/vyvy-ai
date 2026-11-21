#!/usr/bin/env python3
"""Build questions about another person's or company's appointments ("unknown") and
their contrast pairs about the user's own calendar, with dialect time words.

The 1.5B candidates of 09.11. and 11.11. were rejected for wrong actions on
"unknown": by intent, questions about the appointments of someone else with a time
word came back as appointments_today or appointments_tomorrow, several of them in
dialect, and a dialect "morgen" was read as today (per-intent result and the
pattern only; no holdout sentence is used here). tools/build_unknown_hard.py
already covers other people's calendars, but on purpose without the word
"Termine", because "Chef plus Termin" always landed next to a holdout sentence.
That leaves exactly this form untrained.

This file takes the form in a different build, not by rewording: genitive and
possessive ("im Kalender der Teamleiterin"), other verbs (verplant, belegt,
Sprechzeiten, ausgebucht), roles and companies that appear nowhere else, and
dialect blocks written for this file (Bavarian, Alemannic, Rhineland, Hessian and
Palatine, Berlin, Low German). "der Chef" is not used. Each unknown sentence has a
contrast pair about the user's own calendar with the same person and the same
time word, so the model learns whose calendar it is, not "person means unknown".

Every sentence goes through the same filter as tools/build_unknown_hard.py: the
lexical holdout lock (--max-similarity 0.8 against every holdout), the known
paraphrases, the training data, and the unknown boundary. The semantic check
(cosine >= 0.90) runs in the Data check workflow on the push.

    python tools/build_others_calendar.py
    python tools/build_others_calendar.py --check
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

OUT = ROOT / "data" / "train" / "vyvy_others_calendar_v1.jsonl"
#: Built later (tools/build_retired_holdout.py); left out of the comparison so --check stays stable.
LATER = (ROOT / "data" / "train" / "vyvy_retired_blind_v2.jsonl",
         ROOT / "data" / "train" / "vyvy_create_boundary_v1.jsonl",
         ROOT / "data" / "train" / "vyvy_foreign_text_v1.jsonl")
QUOTAS = {"unknown": 120, "appointments_today": 30, "appointments_tomorrow": 30}
TIMEFRAME = {"appointments_today": "today", "appointments_tomorrow": "tomorrow"}

# Invented roles and names (nominative, dative); none of them is in build_unknown_hard.PERSONS.
PERSONS = [("die Teamleiterin", "der Teamleiterin"), ("der Bauleiter", "dem Bauleiter"),
           ("Frau Kowalczyk", "Frau Kowalczyk"), ("Herr Albrecht-Nies", "Herrn Albrecht-Nies"),
           ("die Kollegin aus dem Einkauf", "der Kollegin aus dem Einkauf"),
           ("Herr Demirci", "Herrn Demirci"), ("die Geschäftsführerin", "der Geschäftsführerin"),
           ("Frau Brandauer", "Frau Brandauer")]
COMPANIES = [("die Praxis Dr. Vogelsang", "der Praxis Dr. Vogelsang"),
             ("das Steuerbüro Rauscher", "dem Steuerbüro Rauscher"),
             ("die Schreinerei Kofler", "der Schreinerei Kofler"),
             ("das Ingenieurbüro Hellwig", "dem Ingenieurbüro Hellwig")]
TODAY = ["heute", "heute Nachmittag", "heute noch"]
TOMORROW = ["morgen", "morgen Vormittag", "morgen früh"]

OTHERS = ["Welche Termine stehen {t} im Kalender von {d}?", "Wie viele Termine hat {n} {t}?",
          "Ist der Kalender von {d} {t} schon voll?", "Wann hat {n} {t} den ersten Termin?",
          "Bis wann ist {n} {t} mit Terminen belegt?", "Was steht {t} bei {d} an Terminen drin?",
          "Ist {n} {t} komplett verplant?", "Zeig mir die Termine von {d} für {t}."]
COMPANY = ["Welche Termine vergibt {n} {t} noch?", "Hat {n} {t} Sprechzeiten?",
           "Ist {n} {t} mit Terminen ausgebucht?", "Wann ist {t} der nächste freie Termin bei {d}?"]
# Dialect, written for this file: someone else's appointments with a time word.
DIALECT_OTHERS = [
    "Wos hod de Teamleiterin morsche fia Termine?", "Hod da Bauleiter heit an Haufa Termine?",
    "Hät d Gschäftsfüererin morn vill Termin?", "Isch d Frau Brandauer hüt verplant?",
    "Wat hätt dä Herr Demirci morje för Termine?", "Is dr Kalenner vun dr Kollejin hück voll?",
    "Was hat'n die Frau Kowalczyk morsche fer Termine?", "Is die Praxis Vogelsang heit ausgebucht?",
    "Wat hat'n der Bauleita morjen so für Termine?", "Hett de Teamleitersch morgen veel Termine?",
    "Hat's Steuerbüro Rauscher morsche no en Termin frei?", "Wia schaut da Kalenda vo da Gschäftsführerin morgn aus?",
]
CONTRAST = {
    "appointments_today": ["Habe ich {t} einen Termin mit {d}?", "Wann sehe ich {t} {d}?",
                           "Steht {t} bei mir ein Termin mit {d} drin?", "Bin ich {t} bei {d}?"],
    "appointments_tomorrow": ["Habe ich {t} einen Termin mit {d}?", "Wann sehe ich {t} {d}?",
                              "Steht {t} bei mir ein Termin mit {d} drin?", "Bin ich {t} bei {d}?"],
}
# Dialect own-calendar questions with the dialect time word; (text, intent, entity).
DIALECT_CONTRAST = [
    ("Hob i morsche an Termin mit da Teamleiterin?", "appointments_tomorrow", "da Teamleiterin"),
    ("Han ich morje e Treffe met dem Herr Demirci?", "appointments_tomorrow", "dem Herr Demirci"),
    ("Hani morn öppis mit dr Frau Brandauer?", "appointments_tomorrow", "dr Frau Brandauer"),
    ("Hob i morgn wos beim Bauleiter ausgmacht?", "appointments_tomorrow", "Bauleiter"),
    ("Hab ich morsche was mit der Frau Kowalczyk?", "appointments_tomorrow", "der Frau Kowalczyk"),
    ("Hob i heit no an Termin mit da Gschäftsführerin?", "appointments_today", "da Gschäftsführerin"),
    ("Han ich hück noch jet met dä Kollejin usm Einkauf?", "appointments_today", "dä Kollejin usm Einkauf"),
    ("Hani hüt en Termin bim Herr Albrecht-Nies?", "appointments_today", "Herr Albrecht-Nies"),
]


def unknown_groups() -> dict[str, list[tuple[str, None]]]:
    days = TODAY + TOMORROW
    return {
        "persons": [(hard.cap(s.format(n=n, d=d, t=t)), None) for s in OTHERS for n, d in PERSONS for t in days],
        "companies": [(hard.cap(s.format(n=n, d=d, t=t)), None) for s in COMPANY for n, d in COMPANIES for t in days],
        "dialect": [(s, None) for s in DIALECT_OTHERS],
    }


def contrast_groups(intent: str) -> dict[str, list[tuple[str, str]]]:
    days = TODAY if intent == "appointments_today" else TOMORROW
    groups = {"plain": [(hard.cap(s.format(d=d, t=t)), d) for s in CONTRAST[intent] for _, d in PERSONS for t in days]}
    groups["dialect"] = [(t, e) for t, i, e in DIALECT_CONTRAST if i == intent]
    return groups


def build(seed: int, max_similarity: float) -> tuple[list[dict], dict[str, int], list[tuple[str, str]]]:
    rng = random.Random(f"{seed}-others-calendar")
    # Compared against every training file except this one and build_unknown_hard's own output.
    keep = hard.Filter(max_similarity, own=(OUT, *LATER))
    rows, counts = [], {}
    picked = hard.draw(unknown_groups(), QUOTAS["unknown"], rng, keep, "unknown")
    counts["unknown"] = len(picked)
    rows += [{"text": t, "intent": "unknown", "timeframe": None, "entity": None} for t, _ in picked]
    for intent in TIMEFRAME:
        # No typos: the entity has to stay as written in the sentence.
        got = hard.draw(contrast_groups(intent), QUOTAS[intent], rng, keep, intent, typos=False)
        counts[intent] = len(got)
        rows += [{"text": t, "intent": intent, "timeframe": TIMEFRAME[intent], "entity": e} for t, e in got]
    short = {k: v for k, v in counts.items() if v < QUOTAS[k]}
    if short:
        raise SystemExit(f"too few sentences left after filtering: {short} of {QUOTAS}")
    rng.shuffle(rows)
    return rows, counts, keep.dropped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build questions about other people's appointments and contrast pairs")
    p.add_argument("--seed", type=int, default=47)
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
