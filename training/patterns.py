"""Teacher prompt templates for the error categories of the improvement loop.

Each category is an abstract sentence pattern: what is said, which intent it
belongs to, and fixed seed words the teacher must vary. The teacher writes new,
fully invented sentences from it. No template is derived from a holdout
sentence or from a real request: the categories come from intent-level counts
(tools/improve_report.py), the example sentences were written fresh and pass the
holdout lock (tools/check_holdout_leak.py --threshold 0.8).

Dialect here means standard German with regional time and state words, not a
full dialect text: the model learns the words for the day, the sentence stays
readable. Clock phrases (viertel acht, dreiviertel neun) are not taught: the
application that uses the adapter normalises them before the request.

User patterns: the user may hand in abstract sentence forms (load_forms), never
an original request; names, firms and times are placeholders the teacher fills
with invented ones.

Draft: training/synthesize.py does not read this file yet.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

INTENTS = ("appointments_today", "appointments_tomorrow", "tasks_open", "tasks_overdue", "emails_unread",
           "daily_agenda", "create_appointment", "create_task", "unknown")
# Regional words as people type them, by priority (1 = three times as many requests
# as 3). Only common spellings. "gap" marks words with fewer than five training
# sentences on 10.11.2026 (data/train at d7be13a); existing data stays as it is.
REGIONS = {
    "Bairisch/Österreichisch": {"prio": 1, "today": ["heid", "heit"], "tomorrow": ["morgn", "moang"],
                                "other": ["übermorgn", "nacha", "glei", "z'mittag", "auf d'Nacht", "feieromd",
                                          "vormittog", "nächste Wocha", "heuer"]},
    "Schwäbisch": {"prio": 1, "today": ["heit", "hait"], "tomorrow": ["morga"],
                   "other": ["iebermorga", "nohher", "z'mittag", "z'obed", "Feierobed"]},
    "Schweizerdeutsch": {"prio": 1, "today": ["hüt", "hüüt"], "tomorrow": ["morn"],
                         "other": ["übermorn", "zmittag", "zabig", "znüni", "nächschti Wuche", "geschter"]},
    "Kölsch/Rheinisch": {"prio": 2, "today": ["hück"], "tomorrow": ["morje"],
                         "other": ["övvermorje", "hück ovend", "noh'm Meddach"]},
    "Hessisch/Pfälzisch": {"prio": 2, "today": ["heit", "heut"], "tomorrow": ["morsche", "morje"],
                           "other": ["iwwermorsche", "nachher"]},
    "Berlinerisch": {"prio": 2, "today": ["heut"], "tomorrow": ["morjen"], "other": ["nachher", "jleich"]},
    "Norddeutsch/Platt": {"prio": 3, "today": ["vundaag"], "tomorrow": ["mörgen"],
                          "other": ["nahmiddag", "Fieravend", "glieks"]},
}
GAPS = {"moang", "übermorgn", "heuer", "nacha", "glei", "auf d'Nacht", "feieromd", "vormittog", "nächste Wocha",
        "hait", "morga", "iebermorga", "nohher", "z'obed", "hüüt", "übermorn", "zmittag", "zabig", "znüni",
        "geschter", "Feierobed", "övvermorje", "hück ovend", "noh'm Meddach", "vundaag", "mörgen", "nahmiddag",
        "Fieravend", "glieks"}
TODAY = {r: w["today"] for r, w in REGIONS.items()}
TOMORROW = {r: w["tomorrow"] for r, w in REGIONS.items()}


def region_weights() -> dict[str, int]:
    """Requests per region: priority 1 three times, 2 twice, 3 once."""
    return {r: 4 - w["prio"] for r, w in REGIONS.items()}
# "Morgen" as a part of the day or of a compound, not the next day. "Morgenüberblick"
# itself is left out on purpose: it is the wording of one blind holdout question.
MORNING_COMPOUNDS = ["Morgenrunde", "Morgenbesprechung", "Morgenpost", "Morgenschicht", "heute Morgen",
                     "am Morgen", "Frühbesprechung"]
OTHERS = ["die Kollegin", "der Nachbar", "der Steuerberater", "die Werkstatt", "das Autohaus",
          "die Zahnarztpraxis", "der Vermieter", "die Schwiegermutter", "der Lieferant", "die Hausverwaltung"]
APPOINTMENT_WORDS = ["Termine", "Zeit", "frei", "Sprechstunde", "offen", "erreichbar", "Lücke"]
EVERYDAY = ["Wetter", "Stau", "Zug", "Müllabfuhr", "Bäcker", "Fußballspiel", "Freibad", "Apotheke",
            "Baustelle auf der Bundesstraße"]
SMALLTALK = ["Gruß", "Wie geht's", "Danke", "Witz", "Lob", "Feierabendwunsch"]

PATTERNS = {
    "others_calendar": {
        "intent": "unknown",
        "near": "appointments_tomorrow",
        "rule": ("Die Person fragt nach Terminen, freier Zeit oder Öffnungszeiten einer ANDEREN Person oder "
                 "Firma, nicht nach dem eigenen Kalender. Der Assistent sieht nur den eigenen Kalender, also ist "
                 "es unknown. Nutze jeweils ein Zeitwort und eines dieser Wörter: {appointment_words}. "
                 "Wer gemeint ist, wechselt: {others}."),
        "examples": ["Ist beim Autohaus Okonkwo übermorgen jemand für die Probefahrt da?",
                     "Weißt du, ob die Hausverwaltung am Freitag eine Sprechstunde hat?"],
    },
    "everyday_time": {
        "intent": "unknown",
        "near": "appointments_today",
        "rule": ("Die Person fragt nach etwas aus dem Alltag mit einem Zeitwort ({everyday}), nicht nach "
                 "eigenen Terminen, Aufgaben oder Mails. Das ist unknown, auch wenn heute oder morgen vorkommt."),
        "examples": ["Fährt die S-Bahn übermorgen früh wieder nach Plan?",
                     "Wird's am Samstag warm genug fürs Freibad?"],
    },
    "smalltalk": {
        "intent": "unknown",
        "near": "daily_agenda",
        "rule": ("Die Person plaudert nur ({smalltalk}), ohne Auftrag und ohne Frage nach Kalender, Aufgaben "
                 "oder Mails. Das ist unknown, auch mit Tageszeit oder Zeitwort."),
        "examples": ["Na, alles fit bei dir heut Nachmittag?",
                     "Danke dir, schönen Feierabend nachher!"],
    },
    "dialect_time": {
        "intent": ["appointments_today", "appointments_tomorrow", "tasks_overdue", "daily_agenda"],
        "rule": ("Die Person fragt in Standarddeutsch, aber mit den Wörtern aus {region}, nach dem EIGENEN "
                 "Kalender, den eigenen Aufgaben oder dem Tagesüberblick. Den Tag sagt sie mit einem dieser "
                 "Wörter: heute = {today}, morgen = {tomorrow}; dazu passt eines von: {other}. Der Zeitrahmen "
                 "folgt genau dem Tageswort. Kein Mundart-Volltext."),
        "examples": ["Hab ich moang nach der Brotzeit noch irgendwas im Kalender stehen?",
                     "Steht bei mir hück nach der Mittagspause noch was an Terminen an?"],
    },
    "morning_word": {
        "intent": ["appointments_today", "daily_agenda", "appointments_tomorrow"],
        "rule": ("Die Person nutzt 'Morgen' als Tageszeit oder in einem zusammengesetzten Wort ({compounds}). "
                 "Das ist heute, nicht der nächste Tag. Schreibe zu jedem Satz einen zweiten, in dem 'morgen' "
                 "wirklich den nächsten Tag meint, damit beide Fälle nebeneinander stehen."),
        "examples": ["Was liegt für die Frühbesprechung gleich bei mir an Terminen an?",
                     "Bevor die Morgenschicht losgeht: was hab ich heute im Kalender?"],
    },
    "own_vs_others": {
        "intent": "pair",
        "rule": ("Schreibe Satzpaare mit gleichem Aufbau: der erste fragt nach dem EIGENEN Kalender "
                 "(appointments_today oder appointments_tomorrow), der zweite nach dem Kalender von {others} "
                 "(unknown). Nur das Wort für die Person unterscheidet die beiden."),
        "examples": ["Bin ich am Nachmittag noch irgendwo eingeplant?",
                     "Ist unser Lieferant am Nachmittag noch irgendwo eingeplant?"],
    },
}


def pattern_prompt(key: str, region: str, role: str, n: int) -> str:
    """The teacher prompt for one category, region and role."""
    p = PATTERNS[key]
    rule = p["rule"].format(
        appointment_words=", ".join(APPOINTMENT_WORDS), others=", ".join(OTHERS),
        everyday=", ".join(EVERYDAY), smalltalk=", ".join(SMALLTALK), region=region,
        today=" oder ".join(TODAY.get(region, ["heute"])), tomorrow=" oder ".join(TOMORROW.get(region, ["morgen"])),
        other=", ".join(REGIONS.get(region, {}).get("other", [])) or "nachher", compounds=", ".join(MORNING_COMPOUNDS))
    return (f"Schreibe {n} verschiedene Sätze, die eine Person aus dem Beruf {role} zu ihrem digitalen "
            f"Büro-Assistenten sagt, Sprachstil {region}. Regel: {rule} Beispiele für die Art (nicht wiederholen): "
            + " | ".join(p["examples"]) + " Erfinde Namen von Personen und Firmen frei, keine echten. "
            "Keine Nummerierung, keine Erklärung. Antworte nur mit einem JSON-Array von Objekten mit den Feldern "
            "\"text\" und \"intent\".")


# User sentence forms: abstract shapes only. Every name, firm, day, time or place is a
# placeholder; the teacher invents the filling.
PLACEHOLDERS = {"PERSON", "FIRMA", "TAG", "ZEIT", "ORT", "TERMINWORT", "ALLTAG", "ROLLE"}
FORM_FIELDS = {"category", "form", "intent"}
# A literal name after a title, digits, mail or web addresses: the form was not abstracted.
NOT_ABSTRACT = re.compile(r"\d|@|https?:|www\.|\b(Herr|Herrn|Frau|Dr\.|Firma|Familie)\s+[A-ZÄÖÜ]")
FORM_WORDS = 14


def check_form(f: dict) -> list[str]:
    """Why a user form is refused; empty when it may go to the teacher."""
    problems = []
    if set(f) - FORM_FIELDS - {"region", "note"} or not FORM_FIELDS <= set(f):
        problems.append(f"fields must be {sorted(FORM_FIELDS)} plus optional region and note")
        return problems
    if f["intent"] not in INTENTS:
        problems.append(f"unknown intent {f['intent']!r}")
    holes = set(re.findall(r"\{([A-Z]+)\}", f["form"]))
    if not holes:
        problems.append("no placeholder: write the form with {PERSON}, {TAG} and so on")
    if holes - PLACEHOLDERS:
        problems.append(f"unknown placeholder(s) {sorted(holes - PLACEHOLDERS)}")
    if NOT_ABSTRACT.search(re.sub(r"\{[A-Z]+\}", "", f["form"])):
        problems.append("looks like a real request (name, number or address in the text)")
    if len(re.sub(r"\{[A-Z]+\}", "", f["form"]).split()) > FORM_WORDS:
        problems.append(f"longer than {FORM_WORDS} words besides the placeholders")
    if f.get("region") and f["region"] not in REGIONS:
        problems.append(f"unknown region {f['region']!r}")
    return problems


def load_forms(path: Path) -> list[dict]:
    """Read {"schema": 1, "forms": [...]}; refuse the whole file if one form is not abstract."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise ValueError("schema must be 1")
    bad = {k: why for k, f in enumerate(data["forms"]) if (why := check_form(f))}
    if bad:
        raise ValueError("; ".join(f"form {k + 1}: {', '.join(why)}" for k, why in bad.items()))
    return data["forms"]


def form_prompt(forms: list[dict], role: str, n: int) -> str:
    """Teacher prompt from user forms of one intent: vary the build, fill every placeholder with invented words."""
    intents = {f["intent"] for f in forms}
    if len(intents) != 1:
        raise ValueError("one prompt per intent")
    shapes = " | ".join(f["form"] for f in forms)
    return (f"Schreibe {n} verschiedene Sätze, die eine Person aus dem Beruf {role} zu ihrem digitalen "
            f"Büro-Assistenten sagt. Alle gehören zur Absicht {intents.pop()}. Satzformen als grobe Vorlage, "
            f"Satzbau frei variieren: {shapes}. Ersetze jeden Platzhalter in geschweiften Klammern durch frei "
            "erfundene Namen, Firmen, Orte und Zeitwörter, keine echten. Keine Nummerierung, keine Erklärung. "
            "Antworte nur mit einem JSON-Array von Objekten mit den Feldern \"text\" und \"intent\".")


if __name__ == "__main__":
    for k, f in enumerate(load_forms(Path(sys.argv[1])), 1):
        print(k, f["intent"], f["form"])
