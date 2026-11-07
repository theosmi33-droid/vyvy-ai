#!/usr/bin/env python3
"""Generate training sentences with a large open teacher model (runs on a Kaggle GPU).

Teacher: Qwen2.5-14B-Instruct (Apache License 2.0), pinned, 4-bit on one T4.
(72B is not Apache-licensed; 32B needs 65 GB of downloads for 4-bit loading.)

1. Generate: for every intent and speaking style, the teacher writes short
   requests an office or trade worker would say, with timeframe and named
   person or company. Extra rounds ask for near misses between intents that are
   easy to confuse, labelled with the intent they really belong to.
2. Verify: the teacher classifies every generated sentence again, without
   knowing what it was asked for (one letter per intent, scored by logits).
   Only sentences where that matches the requested intent with probability
   >= --min-agree are kept.

The teacher never sees a holdout sentence. The GitHub runner filters the result
against all holdouts (tools/import_synth.py) before it enters data/train/.

    python training/synthesize.py --out /kaggle/working
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

TEACHER = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

# What each intent means, in the words the teacher gets. Mirrors runner/bench.py INTENT_HELP.
INTENTS = {
    "appointments_today": "fragt nach den eigenen Terminen von heute (was heute ansteht, geplant oder eingetragen ist)",
    "appointments_tomorrow": "fragt nach den eigenen Terminen von morgen",
    "tasks_open": "fragt nach eigenen Aufgaben, die noch nicht erledigt sind, egal wann sie fällig sind",
    "tasks_overdue": "fragt nach eigenen Aufgaben, deren Frist schon vorbei ist, oder nach Vergessenem und Liegengebliebenem",
    "emails_unread": "fragt nach neuen oder ungelesenen E-Mails im eigenen Postfach",
    "daily_agenda": "will ausdrücklich einen Überblick über den ganzen Tag: Termine, Aufgaben und Mails zusammen",
    "create_appointment": "will einen neuen Termin im eigenen Kalender anlegen",
    "create_task": "will eine neue Aufgabe oder To-do anlegen",
    "unknown": "alles andere: Allgemeinwissen, Smalltalk, Rechnen, IT-Probleme, Fragen zu anderen Personen oder "
               "Firmen, und Aktionen ohne eigene Funktion wie Termin löschen oder verschieben, Mail senden oder "
               "beantworten, Aufgabe abhaken, Überweisen",
}
STYLES = [
    "kurzes, sauberes Hochdeutsch",
    "lockere Umgangssprache mit Füllwörtern wie halt, mal, eigentlich",
    "Bairisch, so wie man es spricht",
    "Schwäbisch, so wie man es spricht",
    "Berlinerisch oder Ruhrpott-Deutsch",
    "Norddeutsch oder Plattdeutsch gefärbt",
    "Schweizerdeutsch",
    "hastig getippt: klein geschrieben, ohne Satzzeichen, mit ein, zwei Tippfehlern",
    "sehr knapp, nur Stichworte wie bei einer Sprachsuche",
    "lang und umständlich, mit einem Nebensatz über die eigene Arbeitssituation",
]
ROLES = ["Elektriker", "Malermeisterin", "Bauleiter", "Praxismanagerin", "Steuerfachangestellter",
         "Außendienstlerin", "Lagerleiter", "Büroleiterin", "Schreiner", "Einkäuferin", "Hausverwalter",
         "Projektleiterin", "Dachdecker", "Kanzleiassistentin", "Gärtnerin", "Installateur"]
# Pairs that are easy to confuse: (really meant, could be mistaken for).
NEAR = [
    ("daily_agenda", "appointments_today"), ("appointments_today", "daily_agenda"),
    ("tasks_overdue", "tasks_open"), ("tasks_open", "tasks_overdue"),
    ("create_task", "create_appointment"), ("create_appointment", "create_task"),
    ("unknown", "appointments_today"), ("unknown", "emails_unread"), ("unknown", "tasks_open"),
    ("unknown", "create_appointment"), ("appointments_tomorrow", "appointments_today"),
    ("emails_unread", "unknown"),
]
TIMEFRAMES = ("today", "tomorrow", "week", None)
LETTERS = "ABCDEFGHI"


def gen_prompt(intent: str, style: str, role: str, n: int, near: str | None) -> str:
    task = (f"Schreibe {n} verschiedene Sätze, die eine Person aus dem Beruf {role} zu ihrem digitalen Büro-Assistenten sagt. "
            f"Jeder Satz {INTENTS[intent]}. Sprachstil: {style}.")
    if near:
        task += (f" Wichtig: Jeder Satz soll auf den ersten Blick so klingen, als ob er {INTENTS[near]}, "
                 f"gehört aber eindeutig zur ersten Beschreibung.")
    return (task + " Erfinde Namen von Personen und Firmen frei, keine echten. Keine Nummerierung, keine "
            "Erklärung. Antworte nur mit einem JSON-Array von Objekten mit den Feldern \"text\" (der Satz), "
            "\"timeframe\" (\"today\", \"tomorrow\", \"week\" oder null) und \"entity\" (genannte Person oder "
            "Firma als Text, sonst null).")


def classify_prompt(text: str) -> str:
    options = "\n".join(f"{LETTERS[k]}: {desc}" for k, desc in enumerate(INTENTS.values()))
    return (f"Ein Mitarbeiter sagt zu seinem Büro-Assistenten: \"{text}\"\n\nWas will er? Wähle genau eine "
            f"Möglichkeit.\n{options}\n\nAntworte nur mit dem Buchstaben.")


def parse_items(reply: str) -> list[dict]:
    m = re.search(r"\[.*\]", reply, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not isinstance(it.get("text"), str):
            continue
        text = " ".join(it["text"].split())
        if not 3 <= len(text) <= 200:
            continue
        tf = it.get("timeframe")
        ent = it.get("entity")
        out.append({"text": text, "timeframe": tf if tf in TIMEFRAMES else None,
                    "entity": ent.strip() if isinstance(ent, str) and ent.strip() and ent.strip() in text else None})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate Vyvy training sentences with a teacher model")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-request", type=int, default=12)
    p.add_argument("--rounds", type=int, default=2, help="requests per intent and style")
    p.add_argument("--near-styles", type=int, default=4, help="styles per near-miss pair")
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--min-agree", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--round", default="", help="synthesis round, e.g. 2026-W45; also sets the seed")
    p.add_argument("--limit", type=int, default=0, help="only this many requests (smoke test)")
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Collect names failures after the round; each weekly round samples with its own seed.
    print(f"run vyvy-synth-{args.round or 'manual'}", flush=True)
    if args.round:
        args.seed = int(hashlib.sha256(args.round.encode()).hexdigest()[:8], 16)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    tok = AutoTokenizer.from_pretrained(TEACHER, revision=REVISION)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER, revision=REVISION, device_map={"": 0}, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16))
    model.eval()
    print(f"Teacher loaded in {time.time() - started:.0f} s", flush=True)

    def chat(text: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True)

    requests = []
    for intent in INTENTS:
        for style in STYLES:
            for _ in range(args.rounds):
                requests.append((intent, None, gen_prompt(intent, style, rng.choice(ROLES), args.per_request, None)))
    for intent, near in NEAR:
        for style in rng.sample(STYLES, args.near_styles):
            requests.append((intent, near, gen_prompt(intent, style, rng.choice(ROLES), args.per_request, near)))
    rng.shuffle(requests)
    if args.limit:
        requests = requests[:args.limit]

    candidates = []
    for start in range(0, len(requests), args.batch):
        chunk = requests[start:start + args.batch]
        enc = tok([chat(r[2]) for r in chunk], return_tensors="pt", padding=True).to(0)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=60 * args.per_request, do_sample=True, temperature=0.9,
                                 top_p=0.95, pad_token_id=tok.eos_token_id)
        for (intent, near, _), ids in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
            for it in parse_items(tok.decode(ids, skip_special_tokens=True)):
                candidates.append({**it, "intent": intent, "near": near})
        print(f"generated {start + len(chunk)}/{len(requests)} requests, {len(candidates)} sentences, "
              f"{time.time() - started:.0f} s", flush=True)

    seen, unique = set(), []
    for c in candidates:
        key = re.sub(r"[^\w]+", " ", c["text"].lower()).strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)

    letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    names = list(INTENTS)
    kept, dropped = [], []
    for start in range(0, len(unique), 16):
        chunk = unique[start:start + 16]
        enc = tok([chat(classify_prompt(c["text"])) for c in chunk], return_tensors="pt", padding=True).to(0)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :][:, letter_ids].float()
        probs = torch.softmax(logits, dim=-1)
        for c, pr in zip(chunk, probs):
            k = int(pr.argmax())
            agree = float(pr[names.index(c["intent"])])
            row = {"text": c["text"], "intent": c["intent"], "timeframe": c["timeframe"], "entity": c["entity"]}
            if names[k] == c["intent"] and agree >= args.min_agree:
                kept.append(row)
            else:
                dropped.append({**row, "teacher": names[k], "agree": round(agree, 3), "near": c["near"]})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "synth.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                                         encoding="utf-8")
    (args.out / "synth_dropped.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dropped), encoding="utf-8")
    counts = {i: sum(r["intent"] == i for r in kept) for i in INTENTS}
    manifest = {"teacher": TEACHER, "teacher_revision": REVISION, "license": "Apache-2.0",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "round": args.round, "seed": args.seed, "requests": len(requests), "generated": len(candidates),
                "unique": len(unique), "kept": len(kept), "dropped_by_verification": len(dropped),
                "min_agree": args.min_agree, "per_intent": counts, "minutes": round((time.time() - started) / 60, 1)}
    (args.out / "synth_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
