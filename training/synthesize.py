#!/usr/bin/env python3
"""Generate training sentences with two open teacher models (runs on a Kaggle GPU).

Teachers, both pinned and in 4 bits on one T4, one after the other:
- Qwen2.5-14B-Instruct (Apache License 2.0): writes sentences for every intent
  and style, and judges every sentence of both teachers.
- Phi-4 (14B, MIT): a different model family that writes only the hard cases,
  near misses between intents and "unknown" requests, so the data does not
  follow one teacher's phrasing.
(Qwen 72B is not Apache-licensed: its licence requires "Built with Qwen" on models
trained with its output. 32B needs 65 GB of downloads for 4-bit loading.)

1. Generate: requests an office or trade worker would say, with timeframe and
   named person or company; near misses are labelled with the intent they really
   belong to.
2. Verify: Qwen classifies every sentence without knowing what it was asked
   for (one letter per intent, scored by logits). Only sentences where that
   matches the requested intent with probability >= --min-agree are kept;
   disagreements are recorded in synth_dropped.jsonl and never trained on.

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
SECOND = "microsoft/phi-4"  # MIT
SECOND_REVISION = "2db69c1c3e91a05d2c64a3185acfbaf36f744e25"

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


def load(name: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, device_map={"": 0}, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16))
    model.eval()
    return model, tok


def unload(model, name: str) -> None:
    """Free the GPU and the downloaded weights before the next teacher is loaded."""
    import gc
    import os
    import shutil

    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    shutil.rmtree(hub / ("models--" + name.replace("/", "--")), ignore_errors=True)
    print(f"unloaded {name}: GPU memory allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
          f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)


def chat(tok, text: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def generate(model, tok, requests: list[tuple], teacher: str, per_request: int, batch: int, started: float) -> list[dict]:
    import torch

    out = []
    for start in range(0, len(requests), batch):
        chunk = requests[start:start + batch]
        enc = tok([chat(tok, r[2]) for r in chunk], return_tensors="pt", padding=True).to(0)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=60 * per_request, do_sample=True, temperature=0.9,
                                 top_p=0.95, pad_token_id=tok.pad_token_id)
        for (intent, near, _), ids in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
            for it in parse_items(tok.decode(ids, skip_special_tokens=True)):
                out.append({**it, "intent": intent, "near": near, "teacher": teacher})
        print(f"{teacher}: {start + len(chunk)}/{len(requests)} requests, {len(out)} sentences, "
              f"{time.time() - started:.0f} s", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate Vyvy training sentences with two teacher models")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-request", type=int, default=12)
    p.add_argument("--rounds", type=int, default=2, help="Qwen requests per intent and style")
    p.add_argument("--near-styles", type=int, default=4, help="styles per near-miss pair, per teacher")
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--min-agree", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--round", default="", help="synthesis round, e.g. 2026-W45; also sets the seed")
    p.add_argument("--no-phi", action="store_true", help="Qwen only")
    p.add_argument("--limit", type=int, default=0, help="only this many requests per teacher (smoke test)")
    args = p.parse_args(argv)

    import torch

    # Collect names failures after the round; each weekly round samples with its own seed.
    print(f"run vyvy-synth-{args.round or 'manual'}", flush=True)
    if args.round:
        args.seed = int(hashlib.sha256(args.round.encode()).hexdigest()[:8], 16)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()

    qwen_requests = [(intent, None, gen_prompt(intent, style, rng.choice(ROLES), args.per_request, None))
                     for intent in INTENTS for style in STYLES for _ in range(args.rounds)]
    qwen_requests += [(intent, near, gen_prompt(intent, style, rng.choice(ROLES), args.per_request, near))
                      for intent, near in NEAR for style in rng.sample(STYLES, args.near_styles)]
    # Phi-4 only writes the hard cases: near misses and "unknown" in every style.
    phi_requests = [(intent, near, gen_prompt(intent, style, rng.choice(ROLES), args.per_request, near))
                    for intent, near in NEAR for style in rng.sample(STYLES, args.near_styles)]
    phi_requests += [("unknown", None, gen_prompt("unknown", style, rng.choice(ROLES), args.per_request, None))
                     for style in STYLES]
    rng.shuffle(qwen_requests)
    rng.shuffle(phi_requests)
    if args.limit:
        qwen_requests, phi_requests = qwen_requests[:args.limit], phi_requests[:args.limit]
    if args.no_phi:
        phi_requests = []

    candidates = []
    if phi_requests:
        model, tok = load(SECOND, SECOND_REVISION)
        print(f"{SECOND} loaded after {time.time() - started:.0f} s", flush=True)
        candidates += generate(model, tok, phi_requests, "phi-4", args.per_request, args.batch, started)
        unload(model, SECOND)
    model, tok = load(TEACHER, REVISION)
    print(f"{TEACHER} loaded after {time.time() - started:.0f} s", flush=True)
    candidates += generate(model, tok, qwen_requests, "qwen2.5-14b", args.per_request, args.batch, started)

    seen, unique = set(), []
    for c in candidates:
        key = re.sub(r"[^\w]+", " ", c["text"].lower()).strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)

    # Blind check by Qwen for every sentence, its own and Phi-4's: kept only on agreement.
    letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    names = list(INTENTS)
    kept, dropped = [], []
    for start in range(0, len(unique), 16):
        chunk = unique[start:start + 16]
        enc = tok([chat(tok, classify_prompt(c["text"])) for c in chunk], return_tensors="pt", padding=True).to(0)
        with torch.no_grad():
            # Only the last position is needed; full logits for 16 prompts need ~1.7 GB.
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :][:, letter_ids].float()
        probs = torch.softmax(logits, dim=-1)
        for c, pr in zip(chunk, probs):
            k = int(pr.argmax())
            agree = float(pr[names.index(c["intent"])])
            row = {"text": c["text"], "intent": c["intent"], "timeframe": c["timeframe"], "entity": c["entity"]}
            if names[k] == c["intent"] and agree >= args.min_agree:
                kept.append({**row, "teacher": c["teacher"]})
            else:
                dropped.append({**row, "teacher": c["teacher"], "judge": names[k], "agree": round(agree, 3),
                                "near": c["near"]})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "synth.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                                         encoding="utf-8")
    # Disagreements stay on record (hard cases for a later look), but never enter training.
    (args.out / "synth_dropped.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dropped), encoding="utf-8")
    by_teacher = {t: {"generated": sum(c["teacher"] == t for c in candidates),
                      "kept": sum(r["teacher"] == t for r in kept),
                      "dropped": sum(r["teacher"] == t for r in dropped)} for t in ("qwen2.5-14b", "phi-4")}
    manifest = {"teacher": TEACHER, "teacher_revision": REVISION, "license": "Apache-2.0",
                "second_teacher": None if args.no_phi else SECOND,
                "second_teacher_revision": None if args.no_phi else SECOND_REVISION,
                "second_license": None if args.no_phi else "MIT",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "round": args.round, "seed": args.seed, "requests": len(qwen_requests) + len(phi_requests),
                "generated": len(candidates), "unique": len(unique), "kept": len(kept),
                "dropped_by_verification": len(dropped), "by_teacher": by_teacher, "min_agree": args.min_agree,
                "per_intent": {i: sum(r["intent"] == i for r in kept) for i in INTENTS},
                "minutes": round((time.time() - started) / 60, 1)}
    (args.out / "synth_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
