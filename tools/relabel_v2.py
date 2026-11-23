#!/usr/bin/env python3
"""Relabel the contract 1 training lines for contract 2, by fixed rules, never by hand.

Contract 1 had 9 intents; every request it had no intent for was "unknown". Contract 2
names 16 more. Two kinds of contract 1 lines therefore carry a label that is wrong now:

1. "unknown" lines that ask for an action contract 2 names (tick off, move or delete a
   task, write an email, draft a reply, start or stop the timer, create a project or a
   goal, mark a goal reached, look something up on the internet, how this software
   works, find own appointments, tasks or email, summarize a thread). Paying, sending
   and everything else stay unknown.
2. "appointments_today" lines that ask what is on today in general, without narrowing
   it to appointments: they are daily_agenda now; those that ask only for work (tasks,
   to-dos, what is to do today) and name no appointment are tasks_open, unless they ask
   for the day as a whole (overview, schedule, what is on), which stays daily_agenda.

Each rule is a pair of patterns (one must match, none of the exclusions may). A line is
moved only when exactly one rule fires; a line two rules claim stays as it is and is
listed for review. The committed files in data/train/ are never
changed: the result is data/train/relabel_v2.json, one entry per moved line (file, line,
SHA-256 of the text, old and new label, rule), plus counts and the review lists.
training/train_lora.py applies it and stops if a listed line no longer matches.
Lines that already carry the content field were written for contract 2 and are skipped.

    python tools/relabel_v2.py            # write data/train/relabel_v2.json, print the report
    python tools/relabel_v2.py --check    # committed file matches the rules and the data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "train"
OUT = TRAIN_DIR / "relabel_v2.json"

IGNORE = re.IGNORECASE
TASK = r"(?:aufgabe|aufgab|to-?do|todo|task)"
MAIL = r"(?:e-?mails?|mails?|nachricht(?:en)?)"
TIMER = r"(?:zeit(?:erfassung)?|arbeitszeit|stoppuhr|timer|zeitmessung|erfassung|stempeluhr)"
PROJECT = r"projekt"
GOAL = r"\bziele?\b"
# Sending and paying have no intent in contract 2; a line that asks for them stays unknown.
SEND_PAY = r"\b(?:schick|send|versend|verschick|übersend|abschick|überweis|bezahl|zahl)\w*"

#: intent -> (patterns of which one must match, exclusions of which none may match)
RULES: dict[str, tuple[list[str], list[str]]] = {
    "complete_task": (
        [rf"\bhak\w*\b.*\b{TASK}", rf"{TASK}\w*\b.*\b(?:abhaken|hak\w* .*ab\b|als erledigt|auf erledigt|erledigt markier)",
         rf"\b(?:markier|setz)\w*\b.*{TASK}\w*.*\berledigt"],
        [r"\bziel", SEND_PAY, r"\?$"]),
    "reschedule_task": (
        [rf"\b(?:verschieb|verleg)\w*\b.*\b{TASK}", rf"{TASK}\w*\b.*\b(?:verschieb|verleg)\w*"],
        [r"termin|besprech|meeting|treffen|anprobe|sitzung|gespräch|\btermine?\b", SEND_PAY, r"\?$"]),
    "delete_task": (
        [rf"\b(?:lösch|entfern|streich)\w*\b.*\b{TASK}", rf"{TASK}\w*\b.*\b(?:lösch|entfern|streich)\w*"],
        [r"\bziel", r"termin", SEND_PAY, r"\?$"]),
    "draft_reply": (
        [r"\b(?:entw[iu]rf|entwerf|formulier|schreib|verfass|setz)\w*\b.*\b(?:antwort|rückantwort)",
         r"\bantwort\w*\b.*\b(?:entw[iu]rf|entwerf|formulier|vorschlag)"],
        [SEND_PAY, r"\bhab\w*\b.*\bbeantwortet", r"\b(?:wo|wu)\b"]),
    "write_email": (
        [rf"\b(?:schreib|verfass|formulier|entw[iu]rf|entwerf)\w*\b.*\b{MAIL}\b",
         rf"\bneue\w* {MAIL}\b.*\b(?:an|für)\b"],
        [SEND_PAY, r"antwort", r"\bhab\w*\b|\bhast\b|\bist\b.*\bgeschrieben"]),
    "start_timer": (
        [rf"\b{TIMER}\w*\b.*\b(?:start|beginn|anwerf|los|anschalt|einschalt)\w*",
         rf"\b(?:start|beginn)\w*\b.*\b{TIMER}", r"\b(?:stempel|stemple)\w* (?:mich )?ein\b|\beinstempeln"],
        [r"\bwann\b|\bwie\b.*\?$", r"\bminuten\b|\bwecker\b|eieruhr"]),
    "stop_timer": (
        [rf"\b{TIMER}\w*\b.*\b(?:stopp|stop|beend|anhalt|ausschalt|abschalt)\w*",
         rf"\b(?:stopp|stop|beend|anhalt)\w*\b.*\b{TIMER}", r"\b(?:stempel|stemple)\w* (?:mich )?aus\b|\bausstempeln"],
        [r"\bwann\b|\bwie\b.*\?$"]),
    "create_project": (
        [rf"\b(?:leg|lege)\w*\b.*\b{PROJECT}\w*\b.*\b(?:an|o|oa|on)\b", rf"\b{PROJECT}\w*\b.*\b(?:anleg|erstell|eröffn|aufmach)\w*",
         rf"\b(?:erstell|eröffn|mach)\w*\b.*\bneue\w* {PROJECT}"],
        [r"\?$", r"\bwie weit\b|\bstand\b"]),
    "create_goal": (
        [rf"\b(?:setz|leg|lege|erstell)\w*\b.*{GOAL}.*\b(?:an|fest|o)?\b", r"\bneue\w* ziel\w*\b"],
        [r"\?$", r"\breiseziel|\bzielgrupp|\bziel ist\b|\blösch|\berreicht"]),
    "complete_goal": (
        [rf"{GOAL}.*\b(?:als )?(?:erreicht|geschafft)\b.*\b(?:markier|setz|hak|eintrag)\w*",
         rf"\b(?:markier|setz|hak)\w*\b.*{GOAL}.*\b(?:erreicht|geschafft|ab)\b"],
        [r"\?$", r"\blösch"]),
    "web_search": (
        [r"\b(?:wetter|regnet|regen|schirm|schnee|schneit|gewitter|temperatur|\d+ grad)\b",
         r"\b(?:öffnungszeit\w*|geöffnet|aufhat|offen hat|hat .* offen|macht .* zu|zu macht)\b",
         r"\b(?:fahrplan|zug|züge|bahn|bus|verspätung|stau|verkehr)\b",
         r"\b(?:im internet|im netz|googl\w*|online nach)\b", r"\b(?:preis|kostet|teuer)\w*\b.*\b(?:aktuell|gerade|momentan|heute)\b"],
        [r"\bwas ist\b|\bwas bedeutet\b|\bwer (?:hat|war|ist)\b|\berklär", r"\bkalender\b|\btermin", SEND_PAY,
         # A statement is no request, a past year is fixed knowledge, another company's affairs stay unknown.
         r"^(?:ich|i|morgen fahr|wusstest|übrigens)\b", r"\bletzt\w* jahr", r"\b(?:firma|gmbh|ag|& co)\b",
         r"^(?!.*(?:\?|\b(?:schau|such|guck|sag|kannst|könntest|brauch)\w*\b)).*$"]),
    "help_lookup": (
        [r"\b(?:wie|wo|kann ich|geht das|kann das)\b.*\b(?:im programm|in dem programm|in diesem programm|"
         r"in der app|in dieser software|das programm|die software|diese software)\b",
         r"\b(?:wie|wo)\b.*\b(?:einstell|vorlage|menü|funktion|knopf|passwort)\w*\b.*\bhier\b",
         r"\b(?:wie|wo)\b.*\bhier\b.*\b(?:einstell|vorlage|menü|funktion|knopf|passwort)\w*"],
        [r"\bword\b|\bexcel\b|\boutlook\b|\bwindows\b|\bdrucker\b|\bpraxissoftware\b|\bbuchhaltung", r"\bsoll\b",
         r"\bupdate\b|\bbekommen\b|\bwie heißt\b"]),
    "emails_search": (
        [rf"\b(?:such|find|zeig)\w*\b.*\b{MAIL}\b.*\b(?:von|zu|zum|zur|wegen|über|mit|betreff)\b"],
        [r"zusammen", SEND_PAY, r"\bungelesen|\bneue"]),
    "email_thread_summary": (
        [rf"\b(?:fass|zusammenfass)\w*\b.*\b(?:{MAIL}|verlauf|thread|konversation|mailwechsel)",
         rf"\b(?:{MAIL}|verlauf|mailwechsel)\w*\b.*\bzusammen"],
        [SEND_PAY]),
    "tasks_search": (
        [rf"\b(?:such|find)\w*\b.*\b{TASK}\w*\b.*\b(?:von|zu|zum|zur|wegen|über|mit)\b"],
        [r"termin", SEND_PAY]),
    "appointments_search": (
        [r"\b(?:such|find)\w*\b.*\btermin\w*\b.*\b(?:mit|von|zu|zum|wegen|im|nächste)\b"],
        [r"\bheute\b|\bmorgen\b", r"\bkalender von\b|\bvon (?:herrn?|frau)\b.*\bkalender", SEND_PAY]),
}

#: appointments_today -> daily_agenda: a general question about today, nothing that narrows it to appointments.
AGENDA_GENERAL = re.compile(
    r"(?:\b(?:an)?steh\w*\b.*\ban\b|\bansteht\b|\bsteiht\b.*\ban\b|\bliegt\b.*\ban\b|\banliegt\b|\blos\b|"
    r"\bgeplant\b|\bplaant\b|\bplan\b|\bplaner\b|\bprogramm\b|\bagenda\b|\btagesordnung\b|\bvorhämmt\b|"
    r"\bangesagt\b|\bmein tag\b|\bmeinen tag\b|\bder tag\b|\bwas hab\w* (?:ich|i|isch)\b|\bwat haste\b|"
    r"\bist bei mir\b.*\bwas\b|\bwas isch\b.*\ban\b)", IGNORE)
AGENDA_NARROW = re.compile(
    r"termin|tremin|kalend|kalenner|besprech|bsprechung|afsprook|meeting|treff|besuch|sitzung|gespräch|"
    r"eingetragen|eitraga|eitrogn|ausgmacht|verabred|\buhr\b|\bum \d|\bwo\b|\bwohin\b|persönlich|"
    r"\bbei (?:herrn?|frau|dem|der|den)\b|\bsehe ich\b|\bmit (?:herrn?|frau)\b|stundenplan", IGNORE)
TASK_WORDS = re.compile(r"aufgab|to-?do|zu tun|\bjobs?\b|erledig|zu schaffe|liste", IGNORE)
#: Asking for work, but for the day as a whole: daily_agenda, not tasks_open.
AGENDA_WHOLE = re.compile(r"überblick|\bansteht\b|\bsteht\b.*\ban\b|zeitplan|tagesplan|\bganze[nr]? tag", IGNORE)


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def rules_sha() -> str:
    return hashlib.sha256(json.dumps([RULES, AGENDA_GENERAL.pattern, AGENDA_NARROW.pattern, TASK_WORDS.pattern,
                                      AGENDA_WHOLE.pattern],
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def unknown_rule(text: str) -> list[str]:
    """Every contract 2 intent whose rule fires on this unknown line."""
    t = " ".join(text.casefold().split())
    return [intent for intent, (must, never) in RULES.items()
            if any(re.search(p, t, IGNORE) for p in must) and not any(re.search(p, t, IGNORE) for p in never)]


def classify(row: dict) -> tuple[str | None, str, list[str]]:
    """(new intent or None, rule, competing intents) for one contract 1 line."""
    if row["intent"] == "unknown":
        hits = unknown_rule(row["text"])
        if len(hits) == 1:
            return hits[0], f"unknown->{hits[0]}", []
        return None, "", hits
    if row["intent"] == "appointments_today":
        t = row["text"]
        if AGENDA_NARROW.search(t):
            return None, "", []
        if TASK_WORDS.search(t):
            if AGENDA_WHOLE.search(t):
                return "daily_agenda", "appointments_today->daily_agenda (whole day with work)", []
            return "tasks_open", "appointments_today->tasks_open", []
        if AGENDA_GENERAL.search(t):
            return "daily_agenda", "appointments_today->daily_agenda", []
    return None, "", []


def build(train_dir: Path = TRAIN_DIR) -> dict:
    changes, ambiguous = [], []
    seen_unknown = 0
    per_rule: Counter[str] = Counter()
    per_file: Counter[str] = Counter()
    for path in sorted(train_dir.glob("*.jsonl")):
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "content" in row:
                continue  # written for contract 2
            seen_unknown += row["intent"] == "unknown"
            new, rule, competing = classify(row)
            where = {"file": path.name, "line": no, "text_sha256": text_sha(row["text"]), "text": row["text"]}
            if new:
                changes.append({**where, "from": row["intent"], "to": new, "rule": rule})
                per_rule[rule] += 1
                per_file[path.name] += 1
            elif competing:
                ambiguous.append({**where, "rules": competing})
    return {
        "schema": 1,
        "about": "contract 1 training lines relabelled for contract 2 by tools/relabel_v2.py; applied by "
                 "training/train_lora.py; the files in data/train/ stay as committed",
        "rules_sha256": rules_sha(),
        "counts": {"unknown_lines_seen": seen_unknown, "moved": len(changes), "per_rule": dict(sorted(per_rule.items())),
                   "per_file": dict(sorted(per_file.items())), "ambiguous_kept": len(ambiguous)},
        "changes": changes,
        "review": {"ambiguous_kept": ambiguous},
    }


def render(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=1) + "\n"


def load(path: Path = OUT) -> dict[tuple[str, int], dict]:
    """The committed relabelling, by (file, line); empty if there is none."""
    if not path.exists():
        return {}
    return {(c["file"], c["line"]): c for c in json.loads(path.read_text(encoding="utf-8"))["changes"]}


def apply(overlay: dict[tuple[str, int], dict], file: str, no: int, row: dict) -> dict:
    """The line as contract 2 reads it. Stops if the overlay names this line but the text changed."""
    c = overlay.get((file, no))
    if c is None:
        return row
    if c["text_sha256"] != text_sha(row["text"]) or c["from"] != row.get("intent"):
        raise SystemExit(f"{file}:{no}: data/train/relabel_v2.json is out of date for this line; "
                         "run python tools/relabel_v2.py")
    return {**row, "intent": c["to"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Relabel contract 1 training lines for contract 2")
    p.add_argument("--check", action="store_true", help="only compare with the committed relabel_v2.json")
    p.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    args = p.parse_args(argv)
    result = build(args.train_dir)
    text = render(result)
    out = args.train_dir / OUT.name
    if args.check:
        same = out.exists() and out.read_text(encoding="utf-8") == text
        print(f"relabel_v2.json {'matches' if same else 'differs from'} the rules and the data")
        return 0 if same else 1
    out.write_text(text, encoding="utf-8")
    c = result["counts"]
    print(f"{out}: {c['moved']} lines moved, {c['ambiguous_kept']} ambiguous kept")
    for rule, n in c["per_rule"].items():
        print(f"  {rule}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
