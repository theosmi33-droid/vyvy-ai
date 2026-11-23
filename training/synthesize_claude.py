#!/usr/bin/env python3
"""Generate training sentences with Claude as a third, bottleneck-focused teacher.

Unlike training/synthesize.py (Qwen2.5-14B + Phi-4 on a free Kaggle GPU, covering
all 9 intents every round), this script needs no GPU and is not scheduled
automatically. It targets only the areas the manifests of past rounds show as
hardest: the WEAK intents (create_task, tasks_overdue, unknown, daily_agenda)
and every near-miss pair (NEAR, imported from training/synthesize.py so both
scripts share one definition). Claude is not free API usage, so this script is
started by hand (synth-claude.yml, workflow_dispatch) and stops at a hard dollar
budget; it does not replace the Qwen/Phi-4 round.

Verification mirrors synthesize.py's philosophy (a teacher must not grade its
own homework blind to the label): Claude classifies every one of its own
candidate sentences in a second, separate call that does not see which intent
was requested. Only sentences where that blind classification agrees with the
requested intent are kept; disagreements are recorded in synth_dropped.jsonl
and never trained on.

The output (synth.jsonl, synth_manifest.json) has the exact shape
training/synthesize.py produces, so it runs through tools/import_synth.py
unchanged: the same holdout-leak lock, the same duplicate and semantic checks.
Claude never sees a holdout sentence; this script only ever writes requests
built from the fixed INTENTS/RULES text, never from data/train/ or any
holdout file.

    ANTHROPIC_API_KEY=... python training/synthesize_claude.py --out out --budget-usd 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthesize import (  # noqa: E402
    INTENTS,
    NEAR,
    ROLES,
    RULES,
    STRATA,
    STYLE_STRATUM,
    STYLES,
    TIMEFRAMES,
    WEAK,
    span,
)

TEACHER = "claude-sonnet-5-5"
# Published per-token prices (USD / million tokens) for the cost estimate in the
# manifest; the run stops by its own running total, not by this estimate alone.
PRICE_IN_PER_M = 3.0
PRICE_OUT_PER_M = 15.0


def gen_prompt(intent: str, style: str, role: str, n: int, near: str | None) -> str:
    task = (f"Schreibe {n} verschiedene Sätze, die eine Person aus dem Beruf {role} zu ihrem digitalen "
            f"Büro-Assistenten sagt. Jeder Satz {INTENTS[intent]}. Sprachstil: {style}.")
    if intent in RULES:
        rule, examples = RULES[intent]
        task += f" Regel: {rule} Beispiele für die Art (nicht wiederholen): " + " | ".join(examples)
    if near:
        task += (f" Wichtig: Jeder Satz soll auf den ersten Blick so klingen, als ob er {INTENTS[near]}, "
                 f"gehört aber eindeutig zur ersten Beschreibung.")
    return (task + " Erfinde Namen von Personen und Firmen frei, keine echten. Keine Nummerierung, keine "
            "Erklärung. Antworte nur mit einem JSON-Array von Objekten mit den Feldern \"text\" (der Satz), "
            "\"timeframe\" (\"today\", \"tomorrow\", \"week\" oder null), \"entity\" (genannte Person, Firma "
            "oder genanntes Projekt als Text, sonst null) und \"content\" (Titel, Betreff oder Suchbegriff genau in "
            "den Worten des Satzes, sonst null).")


def classify_prompt(text: str) -> str:
    options = "\n".join(f"- {name}: {desc}" for name, desc in INTENTS.items())
    return (f"Ein Mitarbeiter sagt zu seinem Büro-Assistenten: \"{text}\"\n\nWelche der folgenden Absichten "
            f"trifft am besten zu? Antworte nur mit dem Namen der Absicht, ohne weitere Worte.\n\n{options}")


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
        out.append({"text": text, "timeframe": tf if tf in TIMEFRAMES else None,
                    "entity": span(it.get("entity"), text, 80), "content": span(it.get("content"), text, 120)})
    return out


class Budget:
    """Tracks spend from the API's own reported token usage; stops new calls once over."""

    def __init__(self, usd: float):
        self.limit = usd
        self.in_tokens = 0
        self.out_tokens = 0

    def add(self, usage) -> None:
        self.in_tokens += getattr(usage, "input_tokens", 0)
        self.out_tokens += getattr(usage, "output_tokens", 0)

    @property
    def spent(self) -> float:
        return self.in_tokens / 1e6 * PRICE_IN_PER_M + self.out_tokens / 1e6 * PRICE_OUT_PER_M

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.limit


def call(client, model: str, prompt: str, budget: Budget, max_tokens: int = 1024) -> str | None:
    if budget.exhausted:
        return None
    resp = client.messages.create(model=model, max_tokens=max_tokens,
                                  messages=[{"role": "user", "content": prompt}])
    budget.add(resp.usage)
    # Extended thinking (or any future block type) can put a non-text block first;
    # take the first block that actually is text, not blindly content[0].
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate bottleneck training sentences with Claude")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-request", type=int, default=10)
    p.add_argument("--near-styles", type=int, default=3, help="styles per near-miss pair")
    p.add_argument("--weak-styles", type=int, default=4, help="styles per weak intent")
    p.add_argument("--budget-usd", type=float, default=5.0, help="hard stop; the run never spends more")
    p.add_argument("--verify-fraction", type=float, default=0.3,
                    help="share of --budget-usd reserved for the blind verification pass, so generation "
                         "can never spend the whole budget before a single candidate gets verified")
    p.add_argument("--seed", type=int, default=0, help="0 = random (manual runs vary each time)")
    p.add_argument("--round", default="", help="label for this round, e.g. a date; blank = 'manual-<timestamp>'")
    p.add_argument("--model", default=TEACHER)
    args = p.parse_args(argv)

    try:
        import anthropic
    except ImportError:
        print("pip install anthropic", file=sys.stderr)
        return 2
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set; refusing to run (never embed a key in code)", file=sys.stderr)
        return 2
    client = anthropic.Anthropic(api_key=api_key)

    round_ = args.round or f"manual-{int(time.time())}"
    rng = random.Random(args.seed or None)
    started = time.time()
    gen_budget = Budget(args.budget_usd * (1 - args.verify_fraction))
    verify_budget = Budget(args.budget_usd * args.verify_fraction)

    requests = [(intent, style, None) for intent in WEAK | {"daily_agenda"}
                for style in rng.sample(STYLES, args.weak_styles)]
    requests += [(intent, style, near) for intent, near in NEAR for style in rng.sample(STYLES, args.near_styles)]
    rng.shuffle(requests)

    candidates = []
    for intent, style, near in requests:
        if gen_budget.exhausted:
            print(f"generation budget of ${gen_budget.limit:.2f} reached after {len(candidates)} candidates; "
                  f"stopping to leave ${verify_budget.limit:.2f} for verification", flush=True)
            break
        role = rng.choice(ROLES)
        reply = call(client, args.model, gen_prompt(intent, style, role, args.per_request, near), gen_budget)
        if reply is None:
            continue
        s = 4 if near or intent == "unknown" else STYLE_STRATUM.get(style, 2)
        for it in parse_items(reply):
            candidates.append({**it, "intent": intent, "near": near, "stratum": s, "teacher": "claude"})
        print(f"{intent}{'/' + near if near else ''} {style}: {len(candidates)} candidates so far, "
              f"${gen_budget.spent:.2f} spent, {time.time() - started:.0f} s", flush=True)

    seen, unique = set(), []
    for c in candidates:
        key = re.sub(r"[^\w]+", " ", c["text"].lower()).strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)

    # Blind check: a fresh call per sentence, no mention of the intent it was written for.
    names = set(INTENTS)
    kept, dropped = [], []
    for c in unique:
        if verify_budget.exhausted:
            dropped.append({**{k: c[k] for k in ("text", "intent", "timeframe", "entity", "stratum")},
                            "teacher": "claude", "judge": None, "near": c["near"]})
            continue
        reply = call(client, args.model, classify_prompt(c["text"]), verify_budget, max_tokens=32)
        judged = (reply or "").strip().lower()
        judge = next((n for n in names if n in judged), None)
        row = {"text": c["text"], "intent": c["intent"], "timeframe": c["timeframe"], "entity": c["entity"],
               "content": c.get("content"),
               "stratum": c["stratum"]}
        if judge == c["intent"]:
            kept.append({**row, "teacher": "claude"})
        else:
            dropped.append({**row, "teacher": "claude", "judge": judge, "near": c["near"]})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "synth.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                                         encoding="utf-8")
    (args.out / "synth_dropped.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dropped), encoding="utf-8")
    manifest = {"teacher": f"anthropic/{args.model}", "license": None,
                "second_teacher": None, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "round": round_, "seed": args.seed, "requests": len(requests),
                "generated": len(candidates), "unique": len(unique), "kept": len(kept),
                "dropped_by_verification": len(dropped),
                "per_intent": {i: sum(r["intent"] == i for r in kept) for i in INTENTS},
                "per_stratum": {f"{s} {name}": sum(r["stratum"] == s for r in kept) for s, name in STRATA.items()},
                "budget_usd": args.budget_usd, "spent_usd": round(gen_budget.spent + verify_budget.spent, 4),
                "spent_generation_usd": round(gen_budget.spent, 4),
                "spent_verification_usd": round(verify_budget.spent, 4),
                "input_tokens": gen_budget.in_tokens + verify_budget.in_tokens,
                "output_tokens": gen_budget.out_tokens + verify_budget.out_tokens,
                "minutes": round((time.time() - started) / 60, 1)}
    (args.out / "synth_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
