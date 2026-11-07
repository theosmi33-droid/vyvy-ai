#!/usr/bin/env python3
"""Build data/train/vyvy_train_v1.jsonl from hand-written core sentences.

Every core sentence below was written for this file in standard German or a
regional variant. Variants add fillers, lowercase, missing punctuation and
typos. Each output sentence is unique, and every sentence at or above
--max-similarity (default 0.8, stricter than the holdout lock's 0.85) to any
holdout sentence (development and blind holdout) is dropped. The result is deterministic for a given --seed.

    python tools/build_train_data.py
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
import check_holdout_leak as lock  # noqa: E402

OUT = ROOT / "data" / "train" / "vyvy_train_v1.jsonl"
NEGATIVES = ROOT / "data" / "train" / "unknown_negatives.txt"

# (text, timeframe, entity) per intent. Persons and companies are invented.
CORE: dict[str, list[tuple[str, str | None, str | None]]] = {
    "appointments_today": [
        ("Habe ich heute noch einen Termin?", "today", None),
        ("Wann ist heute mein nächster Termin?", "today", None),
        ("Zeig mir meinen Kalender für heute.", "today", None),
        ("Was ist heute im Kalender eingetragen?", "today", None),
        ("Bin ich heute Nachmittag verplant?", "today", None),
        ("Muss ich heute noch irgendwo hin?", "today", None),
        ("Steht heute ein Kundentermin an?", "today", None),
        ("Treffe ich heute noch Frau Lindqvist?", "today", "Frau Lindqvist"),
        ("Wann sehe ich heute Herrn Okafor?", "today", "Herrn Okafor"),
        ("Was ist heute so angesetzt?", "today", None),
        ("Wie voll ist mein Tag heute im Kalender?", "today", None),
        ("Hab i heit no an Termin?", "today", None),
        ("Wann muas i heit wo sei?", "today", None),
        ("Bin i heit vormittag scho verplant?", "today", None),
        ("Han i hit no en Termin?", "today", None),
        ("Wenn isch hit mi nächschte Termin?", "today", None),
        ("Wann is hüt mien nächste Termin?", "today", None),
        ("Hab isch heit noch en Dermin?", "today", None),
        ("Hab ick heute noch'n Termin?", "today", None),
        ("Hob i heite noch an Termin?", "today", None),
        ("Kalender heute?", "today", None),
        ("termine heut", "today", None),
    ],
    "appointments_tomorrow": [
        ("Was steht morgen im Kalender?", "tomorrow", None),
        ("Habe ich morgen früh schon etwas?", "tomorrow", None),
        ("Wann ist morgen mein erster Termin?", "tomorrow", None),
        ("Bin ich morgen Nachmittag frei?", "tomorrow", None),
        ("Treffe ich morgen die Firma Beispiel Logistik?", "tomorrow", "Firma Beispiel Logistik"),
        ("Was hab ich morgen mit Herrn Okafor?", "tomorrow", "Herrn Okafor"),
        ("Zeig mir den Kalender von morgen.", "tomorrow", None),
        ("Hob i morgn scho wos ausgmacht?", "tomorrow", None),
        ("Wann muas i morgn in da Friah weg?", "tomorrow", None),
        ("Han i morge en Termin?", "tomorrow", None),
        ("Was isch morn im Kalender?", "tomorrow", None),
        ("Hab ich morje ene Termin?", "tomorrow", None),
        ("Wat hev ik morgen för Termine?", "tomorrow", None),
        ("Hab ick morjen wat vor?", "tomorrow", None),
        ("Kalender morgen?", "tomorrow", None),
    ],
    "tasks_open": [
        ("Was steht auf meiner To-do-Liste?", None, None),
        ("Was muss ich noch abarbeiten?", None, None),
        ("Was ist noch nicht erledigt?", None, None),
        ("Welche Aufgaben habe ich diese Woche?", "week", None),
        ("Was ist bis Freitag noch zu tun?", "week", None),
        ("Welche Aufgaben laufen noch für die Firma Beispiel Logistik?", None, "Firma Beispiel Logistik"),
        ("Was ist für Frau Lindqvist noch zu machen?", None, "Frau Lindqvist"),
        ("Zeig mir meine offenen Punkte.", None, None),
        ("Was liegt bei mir noch auf dem Tisch?", None, None),
        ("Was is no ned fertig?", None, None),
        ("Wos steht no auf mei'm Zettl?", None, None),
        ("Wos is no ausständig bei mir?", None, None),
        ("Was isch no uf minere Lischte?", None, None),
        ("Was isch no pendent?", None, None),
        ("Was muss i no schaffa?", None, None),
        ("Was muss isch noch schaffe?", None, None),
        ("To-dos?", None, None),
        ("offene aufgaben", None, None),
    ],
    "tasks_overdue": [
        ("Was hätte ich schon längst erledigen müssen?", None, None),
        ("Wo habe ich eine Frist verpasst?", None, None),
        ("Was ist über dem Fälligkeitsdatum?", None, None),
        ("Welche Aufgaben waren bis gestern fällig?", None, None),
        ("Habe ich eine Deadline gerissen?", None, None),
        ("Was habe ich verschlampt?", None, None),
        ("Was ist bei der Firma Beispiel Logistik schon zu spät?", None, "Firma Beispiel Logistik"),
        ("Welche Sachen für Herrn Okafor sind schon drüber?", None, "Herrn Okafor"),
        ("Was ist liegen geblieben, obwohl es schon fällig war?", None, None),
        ("Wos is scho üba da Zeit?", None, None),
        ("Wos hätt i scho längst firti hom miass'n?", None, None),
        ("Wo bin i scho z'spat dran?", None, None),
        ("Was isch scho lang fällig?", None, None),
        ("Wo han i e Frischt verpasst?", None, None),
        ("Was isch scho drüber, vom Termin her?", None, None),
        ("Wat is ald över de Tied?", None, None),
        ("Wat hätt ich ald fädisch han müsse?", None, None),
        ("Wo hab isch e Frist verpennt?", None, None),
        ("Wat is schon drüber, wat ick vergeigt hab?", None, None),
        ("Überfällig?", None, None),
        ("ueberfaellige aufgaben", None, None),
    ],
    "emails_unread": [
        ("Habe ich neue Mails?", None, None),
        ("Wie viele ungelesene E-Mails habe ich?", None, None),
        ("Ist eine Mail von Frau Lindqvist gekommen?", None, "Frau Lindqvist"),
        ("Hat die Firma Beispiel Logistik geschrieben?", None, "Firma Beispiel Logistik"),
        ("Was ist heute im Posteingang gelandet?", "today", None),
        ("Gibt es etwas Neues im Postfach?", None, None),
        ("Welche Mails habe ich noch nicht gelesen?", None, None),
        ("Hob i neiche Mails?", None, None),
        ("Is wos im Postfach eikemma?", None, None),
        ("Hän i neui Mails?", None, None),
        ("Isch öppis Neus im Poschtfach?", None, None),
        ("Hab ich neue Mailsche?", None, None),
        ("Hett een mi schreven?", None, None),
        ("Hab ick neue Mails jekriegt?", None, None),
        ("Hat mer jemand jeschrebbe?", None, None),
        ("mails?", None, None),
        ("ungelesene nachrichten im postfach", None, None),
    ],
    "daily_agenda": [
        ("Gib mir eine Übersicht über meinen Tag.", "today", None),
        ("Fass mir den heutigen Tag zusammen: Termine, Aufgaben und Mails.", "today", None),
        ("Was erwartet mich heute insgesamt?", "today", None),
        ("Mach mir einen Tagesüberblick.", "today", None),
        ("Briefing für heute bitte.", "today", None),
        ("Wie sieht mein ganzer Tag aus, alles zusammen?", "today", None),
        ("Tagesübersicht.", "today", None),
        ("Gib ma an Überblick über'n ganzen Tag.", "today", None),
        ("Wia schaut mei Tag insgesamt aus?", "today", None),
        ("Gib mer en Überblick über de ganz Tag.", "today", None),
        ("Wie gseht min ganze Tag uus?", "today", None),
        ("Gev mi mol 'n Överblick över den Dag.", "today", None),
        ("Wie sieht mei janzer Daach us?", "today", None),
    ],
    "create_appointment": [
        ("Trag mir morgen um zehn einen Termin mit Herrn Okafor ein.", "tomorrow", "Herrn Okafor"),
        ("Leg einen Termin mit Frau Lindqvist für heute Nachmittag an.", "today", "Frau Lindqvist"),
        ("Plane nächste Woche ein Treffen mit der Firma Beispiel Logistik ein.", "week", "Firma Beispiel Logistik"),
        ("Neuer Termin: Zahnarzt morgen um acht.", "tomorrow", None),
        ("Mach mir einen Termin für die Teambesprechung.", None, None),
        ("Blockier mir heute von zwei bis drei.", "today", None),
        ("Setz für Donnerstag ein Gespräch mit Herrn Okafor an.", "week", "Herrn Okafor"),
        ("Trog ma morgn an Termin mit da Frau Lindqvist ei.", "tomorrow", "Frau Lindqvist"),
        ("Mach ma an Termin für nächste Woch.", "week", None),
        ("Trag mer morn en Termin ii.", "tomorrow", None),
        ("Mach mer en Termin mit em Herr Okafor.", None, "Herr Okafor"),
        ("Dräg mir morje ene Termin en.", "tomorrow", None),
        ("Mok mi 'n Termin för morgen.", "tomorrow", None),
        ("Trag ma ma morjen 'n Termin ein.", "tomorrow", None),
    ],
    "create_task": [
        ("Leg eine Aufgabe an: Angebot für die Firma Beispiel Logistik schreiben.", None, "Firma Beispiel Logistik"),
        ("Erinnere mich daran, Frau Lindqvist zurückzurufen.", None, "Frau Lindqvist"),
        ("Neue Aufgabe: Rechnung prüfen bis Freitag.", "week", None),
        ("Setz auf meine Liste, dass ich die Unterlagen scannen muss.", None, None),
        ("Notier mir als Aufgabe: Lager aufräumen.", None, None),
        ("Mach mir eine Aufgabe für morgen: Bericht abschicken.", "tomorrow", None),
        ("Füg zur To-do-Liste hinzu: Herrn Okafor das Muster schicken.", None, "Herrn Okafor"),
        ("Schreib ma auf: Angebot firti macha.", None, None),
        ("Moch ma a Aufgab: d'Rechnung zahln.", None, None),
        ("Schrib mer uf: Offerte schicke.", None, None),
        ("Mach mer e Uffgab: Ordner sortiere.", None, None),
        ("Schriev mi op: Paket wegbringen.", None, None),
        ("Schreib ma uff: Kunden anrufen.", None, None),
    ],
    "unknown": [
        ("Wie wird das Wetter morgen?", None, None),
        ("Wer hat den Buchdruck erfunden?", None, None),
        ("Erzähl mir einen Witz.", None, None),
        ("Wie viel ist siebzehn mal dreiundzwanzig?", None, None),
        ("Was koche ich heute Abend?", None, None),
        ("Wie schreibt man eine höfliche Absage an einen Bewerber?", None, None),
        ("Was bedeutet Skonto?", None, None),
        ("Wie formuliere ich eine Mahnung freundlich?", None, None),
        ("Welche Termine hat die Firma Beispiel Logistik bei ihren Kunden?", None, None),
        ("Was ist der Unterschied zwischen Brutto und Netto?", None, None),
        ("Wie spät ist es in Tokio?", None, None),
        ("Kannst du mir eine Mail an den Steuerberater formulieren?", None, None),
        ("Wie funktioniert eine Pivot-Tabelle?", None, None),
        ("Was heißt Termingeschäft an der Börse?", None, None),
        ("Wia werd's Weda am Wochnend?", None, None),
        ("Wer hot de Glühbirn erfundn?", None, None),
        ("Wie wird's Wätter morn?", None, None),
        ("Wat jitt et hück en dr Kantin?", None, None),
        ("Wo is de nächste Tankstell?", None, None),
        ("Hallo, wie geht's?", None, None),
        ("Danke dir!", None, None),
    ],
}

PREFIXES = ["", "", "", "Sag mal, ", "Kurze Frage: ", "Du, ", "Hey, ", "Schau mal: ", "Bitte: ", "Ähm, "]
SUFFIXES = ["", "", "", " bitte", " danke", ", gell", ", oder?", " schnell"]


def typo(text: str, rng: random.Random) -> str:
    letters = [i for i, c in enumerate(text) if c.isalpha()]
    if len(letters) < 6:
        return text
    i = rng.choice(letters[1:-1])
    kind = rng.randrange(3)
    if kind == 0:
        return text[:i] + text[i + 1:]
    if kind == 1 and i + 1 < len(text):
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    return text[:i] + text[i] + text[i:]


def variants(text: str, rng: random.Random, n: int) -> list[str]:
    out = [text]
    for _ in range(n * 4):
        if len(out) > n:
            break
        core = text.rstrip(".?!")
        punct = text[len(core):]
        v = rng.choice(PREFIXES) + (core[0].lower() + core[1:] if rng.random() < 0.3 else core)
        suffix = rng.choice(SUFFIXES)
        v += suffix + (punct if rng.random() < 0.7 and not suffix.endswith("?") else "")
        if rng.random() < 0.25:
            v = v.lower()
        if rng.random() < 0.3:
            v = typo(v, rng)
        v = v[0].upper() + v[1:] if v and rng.random() < 0.7 else v
        if v not in out:
            out.append(v)
    return out


def load_negatives(path: Path = NEGATIVES) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def build(seed: int, per_core: int, max_similarity: float) -> tuple[list[dict], list[tuple[str, str, float]]]:
    rng = random.Random(seed)
    holdout = [i for h in lock.HOLDOUTS if h.exists() for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
    hold_norm = [(h["id"], lock.normalise(h["text"])) for h in holdout]
    rows, seen, dropped = [], set(), []
    # Core sentences get generated variants; the negatives list is already varied and
    # long, so each of its sentences is used once to keep "unknown" from dominating.
    sources = [(intent, core, per_core) for intent, cores in CORE.items() for core in cores]
    sources += [("unknown", (text, None, None), 0) for text in load_negatives()]
    for intent, (text, timeframe, entity), n in sources:
        for v in variants(text, rng, n):
            key = lock.normalise(v)
            if not key or key in seen:
                continue
            worst = max((max(lock.similarity(key, h)), hid) for hid, h in hold_norm)
            if worst[0] >= max_similarity:
                dropped.append((v, worst[1], round(worst[0], 3)))
                continue
            seen.add(key)
            rows.append({"text": v, "intent": intent, "timeframe": timeframe, "entity": entity})
    rng.shuffle(rows)
    return rows, dropped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the Vyvy AI training data")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--per-core", type=int, default=7, help="variants per core sentence besides the original")
    p.add_argument("--max-similarity", type=float, default=0.8)
    args = p.parse_args(argv)

    intents = bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))
    missing = set(intents) - set(CORE)
    if missing:
        raise SystemExit(f"no core sentences for {sorted(missing)}")
    rows, dropped = build(args.seed, args.per_core, args.max_similarity)
    for r in rows:
        answer = json.dumps({**{k: r[k] for k in ("intent", "timeframe", "entity")}, "confidence": 0.9})
        if bench.parse_output(answer, intents) is None:
            raise SystemExit(f"not in the output contract: {r}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts = {i: sum(r["intent"] == i for r in rows) for i in CORE}
    print(f"{len(rows)} sentences -> {args.out.relative_to(ROOT)}; dropped {len(dropped)} near holdout")
    print(json.dumps(counts, indent=1))
    for v, hid, sim in dropped:
        print(f"  dropped {sim} ~ {hid}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
