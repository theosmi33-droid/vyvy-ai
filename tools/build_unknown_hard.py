#!/usr/bin/env python3
"""Build 480 hard negatives for "unknown" and 80 contrast pairs for the read intents.

The 1.5B adapter of 08.11. was rejected because "unknown" got worse: questions
the model should have declined came back as appointments_tomorrow or
daily_agenda (per-intent result only, no holdout sentence used here). A time
word like heute or morgen, or a calendar word, is read as a request for the
user's own calendar. The sentences here carry such words on purpose but ask for
nothing Vyvy can do:

- phrases: office and everyday phrases (Mahlzeit, Danke für die Info, Bis morgen)
- fragments: incomplete or ambiguous pieces and statements without a request
  (das Dokument von gestern, später vielleicht)
- private: private requests without a task or calendar (weather, recipes, leisure)
- others: other people's or companies' calendars and mailboxes

Every building block was written for this file; persons and companies are
invented. Blocks are combined, a share gets lowercase, no final punctuation or
a typo, and the groups of a category are drawn in turn so no template
dominates. A sentence is dropped if it is at or above --max-similarity (0.8) to
a holdout sentence, a known paraphrase of one (PARAPHRASES), already in the
training data, or close to a training sentence on the other side of the
unknown boundary (that would teach two labels for one question).

The same persons appear in data/train/vyvy_person_calendar_v1.jsonl: 80 read
questions about the user's own calendar that name them (appointments_today,
appointments_tomorrow, daily_agenda), entity as written in the sentence. Without
them the model could learn "person plus time word means unknown". A pair
differs only in whose calendar it is: "Ist Herr Sommerfeld morgen im Büro?" is
unknown, "Habe ich morgen was mit Herrn Sommerfeld?" is appointments_tomorrow.

Deterministic for a given --seed.

    python tools/build_unknown_hard.py
    python tools/build_unknown_hard.py --check   # committed files match the generator
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402
import check_holdout_leak as lock  # noqa: E402
from build_train_data import load_negatives, typo  # noqa: E402

OUT = ROOT / "data" / "train" / "vyvy_unknown_hard_v1.jsonl"
QUOTAS = {"phrases": 130, "fragments": 120, "private": 150, "others": 80}
CONTRAST_OUT = ROOT / "data" / "train" / "vyvy_person_calendar_v1.jsonl"
CONTRAST_QUOTAS = {"appointments_today": 32, "appointments_tomorrow": 32, "daily_agenda": 16}
TIMEFRAME = {"appointments_today": "today", "appointments_tomorrow": "tomorrow", "daily_agenda": "today"}
# Closer than this to a training sentence on the other side of the unknown boundary
# counts as a contradiction. Between two read intents a close pair is fine: heute or
# morgen alone decides between appointments_today and appointments_tomorrow.
CONFLICT_SIMILARITY = 0.85
# Paraphrases of holdout sentences that only the semantic check (Data check workflow,
# cosine >= 0.90) finds; the lexical lock above does not see them. Normalised text.
PARAPHRASES = {
    "passt danke dir", "geht klar super danke", "ist angekommen danke dir",  # run 42582121327
    # run 42585109153
    "wann ist morgen mein termin mit dem chef", "wann hat der chef übermorgen feierabend",
    "wann treffe ich morgen den chef", "danke passt mach s gut", "nachher mal",
    # run 42586166857; since then "der Chef" is no longer one of PERSONS (Chef plus Termin
    # always lands next to a holdout sentence)
    "wann ist heute mein termin mit dem chef", "habe ich morgen einen termin mit dem chef",
    # run 42587201953 (the embedding reads Vorarbeiter like Chef); the other two as with Chef
    "wann ist morgen mein termin mit dem vorarbeiter", "wann ist heute mein termin mit dem vorarbeiter",
    "habe ich morgen einen termin mit dem vorarbeiter",
}

TIMES = ["heute", "morgen", "heute Abend", "morgen früh", "am Wochenende", "übermorgen",
         "heute Nachmittag", "am Samstag"]
DAYS = ["heute", "morgen", "übermorgen", "am Samstag", "am Sonntag", "am Wochenende"]


def lower_first(text: str) -> str:
    """Lowercase the first word inside a sentence, unless it is a noun."""
    return text if text.split()[0].strip(",") in NOUNS else text[0].lower() + text[1:]


NOUNS = {"Kaffee", "Feierabend", "Montag"}


def phrases() -> dict[str, list[str]]:
    greet = ["Mahlzeit", "Guten Morgen zusammen", "Moin", "Servus", "Grüezi mitenand", "Hallo zusammen",
             "Guten Tag", "Grüß Gott", "Tach auch", "Ei gude", "Morsche", "Hallo, da bin ich wieder",
             "Na, alles fit", "Hallihallo", "Guten Abend", "Hi"]
    thanks = ["Danke für die Info", "Danke dir", "Vielen Dank", "Merci", "Danke, passt", "Super, danke",
              "Danke fürs Bescheidgeben", "Besten Dank für die schnelle Antwort", "Danke, hat sich erledigt",
              "Vergelt's Gott", "Dankeschön", "Danke für gestern", "Danke für den Hinweis",
              "Merci vielmal", "Danke, das hilft"]
    bye = ["Bis morgen", "Bis später", "Bis heute Nachmittag", "Bis nachher", "Schönen Feierabend",
           "Schönes Wochenende", "Bis Montag", "Gute Nacht", "Bis morgen früh", "Ciao, bis morgen",
           "Pfiat di", "Ade, bis morn", "Schönen Abend noch", "Mach's gut", "Bis die Tage",
           "Schönen Tag noch", "Bis nächste Woche", "Schönen Urlaub", "Tschö, bis morgen", "Adieu zäme",
           "Bis übermorgen", "Erhol dich gut"]
    ack = ["Alles klar", "Passt", "Geht klar", "Okay, verstanden", "Gut zu wissen", "In Ordnung",
           "Jo, passt scho", "Wird gemacht", "Läuft", "Hab ich gesehen", "Ist angekommen", "Na dann",
           "Klingt gut", "Ich meld mich", "Super", "Prima", "Joa", "Genau so", "Stimmt", "Logisch"]
    talk = ["Kaffee ist fertig", "Ich bin mal kurz in der Pause", "Bin gleich wieder da",
            "Bin heute im Homeoffice", "Bin ab morgen im Urlaub", "Frohe Weihnachten", "Gutes neues Jahr",
            "Herzlichen Glückwunsch", "Gute Besserung", "Viel Erfolg heute", "Prost Mahlzeit",
            "Na, auch schon da", "Puh, war das ein Tag", "Endlich Freitag", "Heute ist wieder viel los",
            "Montag halt", "Feierabend", "Schon wieder Montag", "Heute zieht sich's", "Ich geh dann mal",
            "Morgen wird's ruhiger", "Wie die Zeit vergeht"]
    end = ["!", ".", "", "!"]

    def mark(i: int, text: str) -> str:
        return text + ("?" if "Na, " in text or "na, " in text else end[i % len(end)])

    def pairs(first: list[str], second: list[str]) -> list[str]:
        # "Super" and "Super, danke" would repeat a word.
        return [mark(i, f"{a}, {lower_first(b)}") for i, (a, b) in enumerate(
            (a, b) for a in first for b in second if a.split(",")[0].lower() not in b.lower())]

    return {
        "greet": [mark(i, t) for i, t in enumerate(greet)] + pairs(greet, talk),
        "thanks": [mark(i, t) for i, t in enumerate(thanks)] + pairs(thanks, bye),
        "bye": [mark(i, t) for i, t in enumerate(bye)] + pairs(ack, bye),
        # Not ack + thanks: "passt, danke dir" and the like paraphrase a holdout sentence.
        "ack": [mark(i, t) for i, t in enumerate(ack)] + pairs(ack, talk),
        "talk": [mark(i, t) for i, t in enumerate(talk)],
    }


def fragments() -> dict[str, list[str]]:
    things = [("das Dokument", "dem Dokument"), ("die Datei", "der Datei"), ("der Ordner", "dem Ordner"),
              ("das Angebot", "dem Angebot"), ("die Liste", "der Liste"), ("der Zettel", "dem Zettel"),
              ("das Ding", "dem Ding"), ("die Unterlagen", "den Unterlagen"), ("das Formular", "dem Formular"),
              ("die Rechnung", "der Rechnung"), ("der Plan", "dem Plan"), ("die Präsentation", "der Präsentation"),
              ("der Brief", "dem Brief"), ("das Foto", "dem Foto"), ("der Schlüssel", "dem Schlüssel")]
    which = ["von gestern", "von letzter Woche", "von vorhin", "von heute früh", "vom Chef", "mit den Zahlen",
             "für den Kunden", "aus der Besprechung", "von Montag", "von der Kollegin", "für morgen"]
    frames = ["{N} {w}", "Und {n} {w}?", "Was ist mit {d} {w}?", "Wegen {d} {w} …", "Ach, {n} {w}",
              "{N} {w}, du weißt schon"]
    vague = ["Später vielleicht", "Morgen dann", "Heute nicht mehr", "Nachher mal", "Irgendwann diese Woche",
             "Vielleicht morgen", "Eher nicht heute", "Mal sehen", "Wenn's passt", "Kommt drauf an",
             "Erst mal nicht", "Nächste Woche eventuell", "So gegen drei", "Nach der Mittagspause",
             "Morgen früh, oder?", "Heute noch?", "Bis wann eigentlich?", "Und dann?", "Ab wann?",
             "Lieber übermorgen", "Heute oder morgen, egal", "Wie gestern", "Gleich, Moment", "Nicht jetzt"]
    half = ["Ich wollte noch", "Kannst du mal", "Wo war jetzt noch mal", "Das mit dem", "Weil ich doch",
            "Also wegen gestern", "Hast du schon", "Wie war das noch mit", "Ich glaub, das war", "Mach mal",
            "Zeig mal das", "Schick das bitte an", "Wer hatte noch mal", "Ach so, und", "Moment, ich such noch",
            "Wenn morgen der", "Heute wollte ich eigentlich", "Und dann noch das von"]
    statements = ["Ich ruf den Kunden morgen zurück.", "Das Angebot hab ich heute früh rausgeschickt.",
                  "Morgen bin ich erst ab zehn da.", "Der Termin gestern war anstrengend.",
                  "Die Mail an den Steuerberater ist raus.", "Die Sache mit dem Lager hab ich erledigt.",
                  "Ich glaub, morgen wird's knapp.", "Gestern hab ich die Rechnungen sortiert.",
                  "Heute Nachmittag wird's ruhiger.", "Das Meeting heute war unnötig.",
                  "Übermorgen kommt der Handwerker zu mir nach Hause.", "Die Kollegin ist heute krank.",
                  "Morgen fahr ich mit dem Zug.", "Heute bin ich echt müde.",
                  "Der Kunde hat sich gestern nicht mehr gemeldet.", "Ich hab heute schon drei Kaffee getrunken."]
    dialect = ["Des Ding vo geschtern", "Wos is jetz mit dem Zeig?", "Des Bapier vom Chef", "Wat is denn nu mit dem Kram?",
               "Dat Dingens von vorhin", "Später vlt", "morge vielleicht", "Des mach'mer später",
               "Joa, schau ma moi", "Gugge mer mal", "Weiß nit", "Muss mol luege", "Dat kriejen wa schon",
               "Heit nimma", "Morn villicht", "Na, wart mal", "Dat Dokument vun jestern", "Des vo heit friah",
               "Öhm, des do", "Wie gseit, später"]
    return {
        "things": [f.format(n=n, N=n[0].upper() + n[1:], d=d, w=w) for f in frames for n, d in things for w in which],
        "vague": vague,
        "half": [h + " …" for h in half],
        "statements": statements,
        "dialect": dialect,
    }


def private() -> dict[str, list[str]]:
    weather = ["Wie wird das Wetter {t}?", "Regnet es {t}?", "Wird es {t} warm?", "Brauche ich {t} einen Schirm?",
               "Gibt es {t} Schnee?", "Wie kalt wird es {t}?", "Scheint {t} die Sonne?", "Soll es {t} gewittern?",
               "Wie windig wird es {t}?"]
    food = ["Rezept für Pasta", "Wie mache ich Kartoffelsalat?", "Gib mir ein Rezept für Linsensuppe.",
            "Was kann ich mit Zucchini kochen?", "Wie backe ich ein Brot ohne Hefe?", "Welcher Wein passt zu Fisch?",
            "Wie viel Reis rechnet man pro Person?", "Rezept für Kaiserschmarrn", "Wie mach ich Spätzle selber?",
            "Wie lange muss ein Schnitzel in die Pfanne?", "Was ist ein schnelles vegetarisches Gericht?",
            "Wie lange hält gekochter Reis im Kühlschrank?", "Rezept für einen einfachen Apfelkuchen",
            "Wie würze ich Gulasch richtig?", "Was kann man statt Sahne nehmen?"]
    food_day = ["Was koche ich {t}?", "Was gibt's {t} zum Abendessen?", "Was backe ich {t} für die Familie?"]
    leisure = ["Wer spielt {t} in der Bundesliga?", "Was kommt {t} im Fernsehen?", "Welcher Film läuft {t} im Kino?",
               "Wann beginnt {t} das Spiel?", "Gibt es {t} irgendwo einen Flohmarkt?",
               "Was kann man {t} mit Kindern unternehmen?", "Hat das Freibad {t} offen?",
               "Wann macht der Supermarkt {t} zu?", "Welche Apotheke hat {t} Notdienst?",
               "Wie voll ist {t} die Autobahn?", "Fährt {t} die S-Bahn?", "Wann geht {t} die Sonne auf?",
               "Ist {t} Markt in der Stadt?"]
    personal = ["Wie lange sollte man schlafen?", "Was hilft gegen Erkältung?",
                "Wie viele Schritte sollte man am Tag gehen?", "Was schenke ich meiner Mutter zum Geburtstag?",
                "Wo kann man in der Nähe schön wandern?", "Welche Serie soll ich als Nächstes schauen?",
                "Erzähl mir einen Witz.", "Sing mir was vor.", "Wie spät ist es gerade?",
                "Wie werde ich Schluckauf los?", "Welche Zimmerpflanze braucht wenig Licht?",
                "Wie putze ich Fenster ohne Streifen?", "Wie bekomme ich Rotwein aus dem Teppich?",
                "Was ziehe ich zur Hochzeit an?", "Welcher Hund passt in eine kleine Wohnung?",
                "Wie lerne ich am schnellsten Spanisch?", "Wann muss ich die Uhr umstellen?",
                "Wie pflege ich Lederschuhe?"]
    dialect = ["Wia werd's Weda morgn?", "Rengt's heit no?", "Wos kocht ma heit z'Mittag?",
               "Wo gibt's heit an Christkindlmarkt?", "Regnet's morge im Ländle?", "Was koch i hit z'Nacht?",
               "Wird's hüt warm?", "Wat jitt et hück zo esse?", "Rägnet's morn?",
               "Wie wird dat Wedder morgen an de Küst?", "Wat kommt heute in de Glotze?",
               "Wann spielt der Club heit?", "Is morgen Markt?", "Hod da Biergartn heit offa?",
               "Wo chan i morn guet z'Mittag ässe?", "Schneit's übermorge?", "Wat kost dat Kino hück?",
               "Wia lang braucht a Schweinsbratn?", "Wo gibt et hier lecker Döner?", "Schiint hüt d'Sunne?"]
    return {
        "weather": [w.format(t=t) for w in weather for t in TIMES],
        "food": food + [f.format(t=t) for f in food_day for t in DAYS],
        "leisure": [s.format(t=t) for s in leisure for t in DAYS],
        "personal": personal,
        "dialect": dialect,
    }


# The same persons on both sides: their own calendar is unknown, the user's
# calendar with them is a read intent. (nominative, dative, accusative)
PERSONS = [("Frau Lehnert", "Frau Lehnert", "Frau Lehnert"), ("Herr Sommerfeld", "Herrn Sommerfeld", "Herrn Sommerfeld"),
           ("Frau Wegener", "Frau Wegener", "Frau Wegener"), ("Herr Pfeiffer", "Herrn Pfeiffer", "Herrn Pfeiffer"),
           ("Frau Dr. Seidl", "Frau Dr. Seidl", "Frau Dr. Seidl"), ("Herr Haberl", "Herrn Haberl", "Herrn Haberl"),
           ("meine Kollegin", "meiner Kollegin", "meine Kollegin"), ("der Vorarbeiter", "dem Vorarbeiter", "den Vorarbeiter"),
           ("mein Azubi", "meinem Azubi", "meinen Azubi"), ("die Buchhaltung", "der Buchhaltung", "die Buchhaltung"),
           ("unser Meister", "unserem Meister", "unseren Meister"),
           ("die neue Kollegin", "der neuen Kollegin", "die neue Kollegin")]


def cap(text: str) -> str:
    return text[0].upper() + text[1:]


def others() -> dict[str, list[str]]:
    timed = ["Ist {n} {t} im Büro?", "Wann hat {n} {t} Feierabend?", "Wie sieht der Kalender von {d} {t} aus?",
             "Ist {n} {t} erreichbar?", "Hat {n} {t} noch Luft?", "Ist {n} {t} ausgebucht?",
             "Hat {n} {t} Urlaub?", "Ist {n} {t} auf der Baustelle?", "Wann kommt {n} {t} ins Büro?",
             "Wo ist {n} {t} unterwegs?"]
    mail = ["Hat {n} meine Mail schon gelesen?",
            "Hat {n} die Rechnung schon verschickt?", "Ist das Postfach von {d} voll?",
            "Wem hat {n} heute geschrieben?", "Was hat {n} heute schon erledigt?"]
    companies = [("die Firma Holzbau Reisinger", "der Firma Holzbau Reisinger"),
                 ("die Spedition Wendler", "der Spedition Wendler"), ("das Autohaus Brückl", "dem Autohaus Brückl"),
                 ("unser Lieferant", "unserem Lieferanten"), ("die Kanzlei Obermaier", "der Kanzlei Obermaier")]
    company = ["Was hat {n} {t} geplant?", "Hat {n} {t} geöffnet?", "Ist bei {d} {t} jemand erreichbar?",
               "Wann liefert {n} {t}?", "Hat {n} {t} Betriebsferien?"]
    dialect = ["Hod da Chef morgn Zeit?", "Isch d'Frau Lehnert hit im Gschäft?", "Is der Meister morgen uff da Baustell?",
               "Hät dr Chef hück vill ze dun?", "Hett de Chef morgen Tied?", "Is de Buchhaltung hüt besett?",
               "Wann hat'n die Kollegin heut Schluss?", "Is der Herr Pfeiffer heid do?",
               "Hot da Azubi heit Berufsschul?", "Isch dr Herr Sommerfeld morn im Büro?"]

    return {
        "persons": [cap(s.format(n=n, d=d, t=t)) for s in timed for n, d, _ in PERSONS for t in DAYS],
        "mail": [cap(s.format(n=n, d=d)) for s in mail for n, d, _ in PERSONS],
        "companies": [cap(s.format(n=n, d=d, t=t)) for s in company for n, d in companies for t in DAYS],
        "dialect": dialect,
    }



def contrasts() -> dict[str, dict[str, list[tuple[str, str]]]]:
    """Read questions about the user's own calendar that name a person: (text, entity) per person."""
    today = ["Habe ich heute einen Termin mit {d}?", "Habe ich heute noch was mit {d}?", "Wann treffe ich heute {a}?",
             "Sehe ich heute noch {a}?", "Bin ich heute mit {d} verabredet?", "Wann ist heute mein Termin mit {d}?",
             "Steht heute in meinem Kalender was mit {d}?", "Um wie viel Uhr bin ich heute bei {d}?",
             "Hab ich heute ein Gespräch mit {d}?", "Hab i heit no an Termin mit {d}?"]
    tomorrow = ["Habe ich morgen einen Termin mit {d}?", "Habe ich morgen was mit {d}?", "Wann treffe ich morgen {a}?",
                "Sehe ich morgen {a}?", "Bin ich morgen mit {d} verabredet?", "Wann ist morgen mein Termin mit {d}?",
                "Steht morgen in meinem Kalender was mit {d}?", "Um wie viel Uhr bin ich morgen bei {d}?",
                "Hab ich morgen ein Gespräch mit {d}?", "Hob i morgn wos mit {d} ausgmacht?"]
    agenda = ["Gib mir den Überblick für heute: Termine, Aufgaben und Mails, auch das Treffen mit {d}.",
              "Fass mir den ganzen Tag zusammen, Kalender, To-dos und neue Mails, ich sehe heute ja auch {a}.",
              "Alles für heute auf einmal bitte: Termine, offene Aufgaben, Mails und wann ich {a} treffe.",
              "Was steht heute alles an, Termine wie Aufgaben wie Mails? Ich hab auch was mit {d}.",
              "Kompletter Tagesüberblick bitte, mit dem Termin bei {d}, meinen Aufgaben und der Post."]

    def by_person(templates: list[str]) -> dict[str, list[tuple[str, str]]]:
        return {n: [(cap(s.format(d=d, a=a)), d if "{d}" in s else a) for s in templates] for n, d, a in PERSONS}

    return {"appointments_today": by_person(today), "appointments_tomorrow": by_person(tomorrow),
            "daily_agenda": by_person(agenda)}


CATEGORIES = {"phrases": phrases, "fragments": fragments, "private": private, "others": others}


def noisy(text: str, rng: random.Random, typos: bool = True) -> str:
    """Typed as people type: some lowercase, without final punctuation or with a typo."""
    r = rng.random()
    if r < 0.12:
        return text.lower()
    if r < 0.22:
        return text.rstrip(".!?… ")
    if r < 0.30 and typos:
        return typo(text, rng)
    return text


class Filter:
    def __init__(self, max_similarity: float) -> None:
        self.max_similarity = max_similarity
        holdout = [i for h in lock.HOLDOUTS if h.exists() for i in json.loads(h.read_text(encoding="utf-8"))["items"]]
        self.holdout = [(h["id"], lock.normalise(h["text"])) for h in holdout]
        self.seen = {lock.normalise(t) for t in load_negatives()}
        self.by_intent: dict[str, list[str]] = {}
        for path in sorted((ROOT / "data" / "train").glob("*.jsonl")):
            if path.resolve() in (OUT.resolve(), CONTRAST_OUT.resolve()):
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    key = lock.normalise(row["text"])
                    self.seen.add(key)
                    self.by_intent.setdefault(row["intent"], []).append(key)
        self.dropped: list[tuple[str, str]] = []

    def close(self, a: str, b: str, threshold: float) -> bool:
        # 2*min/(la+lb) bounds the difflib ratio, so most pairs skip the full comparison.
        if 2 * min(len(a), len(b)) / (len(a) + len(b) or 1) < threshold:
            return False
        m = difflib.SequenceMatcher(None, a, b, autojunk=False)
        return m.real_quick_ratio() >= threshold and m.quick_ratio() >= threshold and m.ratio() >= threshold

    def accept(self, text: str, intent: str) -> bool:
        key = lock.normalise(text)
        if not key or key in self.seen:
            return False
        if key in PARAPHRASES:
            self.dropped.append((text, "paraphrase of a holdout sentence (semantic check)"))
            return False
        words = set(key.split())
        for hid, h in self.holdout:
            hw = set(h.split())
            if len(words & hw) / len(words | hw) >= self.max_similarity or self.close(key, h, self.max_similarity):
                self.dropped.append((text, f"near holdout {hid}"))
                return False
        if any(self.close(key, k, CONFLICT_SIMILARITY) for other, keys in self.by_intent.items()
               if (other == "unknown") != (intent == "unknown") for k in keys):
            self.dropped.append((text, "close to a training sentence across the unknown boundary"))
            return False
        self.seen.add(key)
        self.by_intent.setdefault(intent, []).append(key)
        return True


def draw(groups: dict[str, list[tuple[str, str | None]]], quota: int, rng: random.Random, keep: Filter,
         intent: str, typos: bool = True) -> list[tuple[str, str | None]]:
    pools = {name: rng.sample(items, len(items)) for name, items in groups.items()}
    out: list[tuple[str, str | None]] = []
    while len(out) < quota and any(pools.values()):
        for name in groups:
            if pools[name] and len(out) < quota:
                text, entity = pools[name].pop()
                text = noisy(text, rng, typos)
                if keep.accept(text, intent):
                    out.append((text, entity))
    return out


def build(seed: int, max_similarity: float) -> tuple[list[dict], list[dict], dict[str, int], list[tuple[str, str]]]:
    rng = random.Random(seed)
    keep = Filter(max_similarity)
    negatives, contrast, counts = [], [], {}
    for name, make in CATEGORIES.items():
        groups = {g: [(t, None) for t in texts] for g, texts in make().items()}
        picked = draw(groups, QUOTAS[name], rng, keep, "unknown")
        if len(picked) < QUOTAS[name]:
            raise SystemExit(f"{name}: only {len(picked)} of {QUOTAS[name]} sentences left after filtering")
        counts[name] = len(picked)
        negatives += [{"text": t, "intent": "unknown", "timeframe": None, "entity": None} for t, _ in picked]
    for intent, groups in contrasts().items():
        # No typos here: the entity has to stay as written in the sentence.
        picked = draw(groups, CONTRAST_QUOTAS[intent], rng, keep, intent, typos=False)
        if len(picked) < CONTRAST_QUOTAS[intent]:
            raise SystemExit(f"{intent}: only {len(picked)} of {CONTRAST_QUOTAS[intent]} sentences left after filtering")
        counts[intent] = len(picked)
        contrast += [{"text": t, "intent": intent, "timeframe": TIMEFRAME[intent], "entity": e} for t, e in picked]
    rng.shuffle(negatives)
    rng.shuffle(contrast)
    return negatives, contrast, counts, keep.dropped


def render(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the hard negatives for intent unknown and their contrast pairs")
    p.add_argument("--seed", type=int, default=46)
    p.add_argument("--max-similarity", type=float, default=0.8)
    p.add_argument("--check", action="store_true", help="only compare with the committed files")
    args = p.parse_args(argv)

    intents = bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))
    negatives, contrast, counts, dropped = build(args.seed, args.max_similarity)
    for r in negatives + contrast:
        answer = json.dumps({**{k: r[k] for k in ("intent", "timeframe", "entity")}, "confidence": 0.9})
        if bench.parse_output(answer, intents) is None:
            raise SystemExit(f"not in the output contract: {r}")
    outputs = [(OUT, negatives), (CONTRAST_OUT, contrast)]
    if args.check:
        stale = [path for path, rows in outputs
                 if not path.exists() or path.read_text(encoding="utf-8") != render(rows)]
        for path in stale:
            print(f"{path.relative_to(ROOT)} differs from the generator; run tools/build_unknown_hard.py")
        if not stale:
            print(f"both files match the generator ({len(negatives)} + {len(contrast)} sentences)")
        return 1 if stale else 0
    for path, rows in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(rows), encoding="utf-8")
        print(f"{len(rows)} sentences -> {path.relative_to(ROOT)}")
    print(json.dumps(counts, indent=1))
    print(f"dropped {len(dropped)}")
    for text, why in dropped:
        print(f"  dropped ({why}): {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
