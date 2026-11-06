#!/usr/bin/env python3
"""Vyvy AI benchmark: German intent & slot extraction with a local GGUF model.

Every question in the dataset is sent once to the model, constrained by a GBNF
grammar, at temperature 0. A prediction counts as correct only if the output
parses, the intent matches and the model's own confidence is at least
--min-confidence. Latency is wall-clock time per question on the machine that
runs the benchmark; it does not transfer to other hardware.

    python runner/bench.py --model models/model.gguf
    VYVY_MODEL_PATH=models/model.gguf python runner/bench.py
    python runner/bench.py --dry-run          # no model: grammar check + mock inference

Environment: VYVY_MODEL_PATH (default for --model), VYVY_SLOT_TIMEOUT_S (default
for --timeout: answers slower than this count as a miss, default 4 s; an
assistant that answers after the user gave up has not answered).

Stateless: reads the dataset and grammar, writes a Markdown report to stdout
(and optionally to --summary / --json). Only dependency for real runs:
llama-cpp-python.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "holdout_benchmarks.json"
DEFAULT_GRAMMAR = ROOT / "grammars" / "slots.gbnf"
FIELDS = ("intent", "timeframe", "entity", "confidence")
TIMEFRAMES = ("today", "tomorrow", "week")

SYSTEM_PROMPT = (
    "You translate one short German question or request from an office worker "
    "(often regional dialect, colloquial wording or typos) into a JSON object. "
    "You never answer the question.\n"
    "intent: appointments_today (own appointments today), appointments_tomorrow, "
    "tasks_open (what is still to do), tasks_overdue (overdue, forgotten or urgent work), "
    "emails_unread (new or unread email), daily_agenda (what is on today in general), "
    "create_appointment, create_task, unknown (anything else, including general "
    "knowledge and questions about other companies).\n"
    "timeframe: today, tomorrow, week or null. entity: a named person or company, or null. "
    "confidence: 0 to 1, how sure you are.\n"
    "Examples:\n"
    'Question: "Was liegt heut im Kalender, gell?" -> {"intent": "appointments_today", '
    '"timeframe": "today", "entity": null, "confidence": 0.9}\n'
    'Question: "Hab ich irgendwas Dringendes verpennt?" -> {"intent": "tasks_overdue", '
    '"timeframe": null, "entity": null, "confidence": 0.8}\n'
    'Question: "Wer hat die Glühbirne erfunden?" -> {"intent": "unknown", "timeframe": null, '
    '"entity": null, "confidence": 0.9}'
)

Model = Callable[[str, str, str], str]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


# ── Grammar ────────────────────────────────────────────────────────────────

_RULE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\s*::=\s*(.*)$")
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def _strip_terminals(body: str) -> str | None:
    """Replace "..." literals and [...] classes by spaces; None if one is unterminated."""
    out, i = [], 0
    while i < len(body):
        ch = body[i]
        if ch in '"[':
            close = '"' if ch == '"' else "]"
            i += 1
            while i < len(body) and body[i] != close:
                i += 2 if body[i] == "\\" else 1
            if i >= len(body):
                return None
            out.append(" ")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def check_grammar(text: str) -> list[str]:
    """Static GBNF check without llama.cpp: rules parse, root exists, no undefined names."""
    rules: dict[str, str] = {}
    errors: list[str] = []
    for no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _RULE.match(line)
        if not m:
            errors.append(f"line {no}: not a rule: {line[:60]}")
            continue
        if m.group(1) in rules:
            errors.append(f"line {no}: rule defined twice: {m.group(1)}")
        rules[m.group(1)] = m.group(2)
    if "root" not in rules:
        errors.append("no root rule")
    for name, body in rules.items():
        rest = _strip_terminals(body)
        if rest is None:
            errors.append(f"{name}: unterminated string or character class")
            continue
        for ref in _NAME.findall(rest):
            if ref not in rules:
                errors.append(f"{name}: undefined rule {ref}")
    return errors


def grammar_intents(text: str) -> tuple[str, ...]:
    m = re.search(r"^intent\s*::=\s*(.*)$", text, re.M)
    return tuple(re.findall(r'"\\"([a-z_]+)\\""', m.group(1))) if m else ()


# ── Output parsing ─────────────────────────────────────────────────────────


class IntentParser:
    """Strict validation of the model output against the output contract."""

    def __init__(self, intents: tuple[str, ...]):
        self.intents = intents

    def parse(self, raw: str) -> dict | None:
        return parse_output(raw, self.intents)


def parse_output(raw: str, intents: tuple[str, ...]) -> dict | None:
    """The model output, strictly validated. Anything unexpected -> None."""
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or tuple(obj) != FIELDS:
        return None
    if obj["intent"] not in intents:
        return None
    if obj["timeframe"] is not None and obj["timeframe"] not in TIMEFRAMES:
        return None
    if obj["entity"] is not None and not isinstance(obj["entity"], str):
        return None
    c = obj["confidence"]
    if isinstance(c, bool) or not isinstance(c, int | float) or not 0 <= c <= 1:
        return None
    return obj


def outcome(parsed: dict | None, min_confidence: float) -> str:
    if parsed is None:
        return "invalid"
    if parsed["confidence"] < min_confidence:
        return f"low confidence ({parsed['confidence']:.2f}, {parsed['intent']})"
    return parsed["intent"]


# ── Models ─────────────────────────────────────────────────────────────────


def llama_model(path: Path, threads: int, ctx: int) -> Model:
    from llama_cpp import Llama, LlamaGrammar  # imported only for real runs

    llm = Llama(model_path=str(path), n_ctx=ctx, n_threads=threads, seed=0, verbose=False)
    cache: dict[str, object] = {}

    def run(system: str, question: str, grammar: str) -> str:
        if grammar not in cache:
            cache[grammar] = LlamaGrammar.from_string(grammar, verbose=False)
        out = llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
            grammar=cache[grammar],
            temperature=0.0,
            max_tokens=80,
        )
        return out["choices"][0]["message"]["content"] or ""

    return run


def mock_model(expected: dict[str, str]) -> Model:
    """Deterministic stand-in: right for most items, wrong, unsure or broken for a few."""
    counter = {"n": 0}

    def run(_system: str, question: str, _grammar: str) -> str:
        counter["n"] += 1
        n, intent = counter["n"], expected[question.removeprefix('Question: "').removesuffix('"')]
        if n % 13 == 0:
            return "not json"
        if n % 11 == 0:
            intent = "unknown"
        conf = 0.3 if n % 17 == 0 else 0.9
        return json.dumps({"intent": intent, "timeframe": None, "entity": None, "confidence": conf})

    return run


# ── Benchmark ──────────────────────────────────────────────────────────────


class VyvyBenchRunner:
    """Runs every dataset item once through the model and scores it."""

    def __init__(self, model: Model, grammar: str, parser: IntentParser, min_confidence: float,
                 timeout_s: float, clock: Callable[[], float] = time.perf_counter):
        self.model, self.grammar, self.parser = model, grammar, parser
        self.min_confidence, self.timeout_s, self.clock = min_confidence, timeout_s, clock

    def run(self, items: list[dict]) -> dict:
        return run_benchmark(items, self.model, self.grammar, self.parser.intents,
                             self.min_confidence, self.timeout_s, self.clock)


def run_benchmark(items: list[dict], model: Model, grammar: str, intents: tuple[str, ...],
                  min_confidence: float, timeout_s: float = 4.0,
                  clock: Callable[[], float] = time.perf_counter) -> dict:
    per_intent: dict[str, dict] = {}
    for item in items:
        t0 = clock()
        raw = model(SYSTEM_PROMPT, f'Question: "{item["text"]}"', grammar)
        ms = (clock() - t0) * 1000
        got = outcome(parse_output(raw, intents), min_confidence)
        if ms > timeout_s * 1000:
            got = f"timeout ({ms:.0f} ms, {got})"
        row = per_intent.setdefault(item["expected_intent"], {"n": 0, "correct": 0, "ms": [], "misses": []})
        row["n"] += 1
        row["ms"].append(ms)
        if got == item["expected_intent"]:
            row["correct"] += 1
        else:
            row["misses"].append({"id": item["id"], "text": item["text"], "got": got})
    all_ms = [m for r in per_intent.values() for m in r["ms"]]
    return {
        "n": sum(r["n"] for r in per_intent.values()),
        "correct": sum(r["correct"] for r in per_intent.values()),
        "median_ms": round(statistics.median(all_ms)) if all_ms else 0,
        "timeout_s": timeout_s,
        "per_intent": {
            k: {"n": r["n"], "correct": r["correct"], "accuracy": r["correct"] / r["n"],
                "median_ms": round(statistics.median(r["ms"])), "mean_ms": round(statistics.mean(r["ms"])),
                "misses": r["misses"]}
            for k, r in sorted(per_intent.items())
        },
    }


def report(result: dict, header: str) -> str:
    n, c = result["n"], result["correct"]
    lines = [
        "## Vyvy AI benchmark",
        "",
        header,
        "",
        f"**Accuracy: {c}/{n} ({c / n:.1%})** · median {result['median_ms']} ms per question"
        f" · timeout {result['timeout_s']:g} s",
        "",
        "| Intent | Correct | N | Accuracy | Median ms | Mean ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, r in result["per_intent"].items():
        lines.append(f"| {k} | {r['correct']} | {r['n']} | {r['accuracy']:.1%} | {r['median_ms']} | {r['mean_ms']} |")
    misses = [(k, m) for k, r in result["per_intent"].items() for m in r["misses"]]
    if misses:
        lines += ["", "### Misses", "", "| Expected | ID | Question | Got |", "|---|---|---|---|"]
        lines += [f"| {k} | {m['id']} | {m['text']} | {m['got']} |" for k, m in misses]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Vyvy AI intent & slot extraction benchmark")
    p.add_argument("--model", type=Path, default=os.environ.get("VYVY_MODEL_PATH") or None,
                   help="GGUF model file (default: $VYVY_MODEL_PATH)")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--grammar", type=Path, default=DEFAULT_GRAMMAR)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--timeout", type=float, default=_env_float("VYVY_SLOT_TIMEOUT_S", 4.0),
                   help="seconds per question before it counts as a miss (default: $VYVY_SLOT_TIMEOUT_S or 4)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--label", default="", help="model label for the report header")
    p.add_argument("--summary", type=Path, help="append the Markdown report to this file")
    p.add_argument("--json", type=Path, help="write raw results as JSON")
    p.add_argument("--dry-run", action="store_true", help="no model: grammar check and mock inference")
    args = p.parse_args(argv)

    grammar = args.grammar.read_text(encoding="utf-8")
    errors = check_grammar(grammar)
    if errors:
        print("Grammar check failed:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    intents = grammar_intents(grammar)
    items = json.loads(args.data.read_text(encoding="utf-8"))["items"]
    unknown = sorted({i["expected_intent"] for i in items} - set(intents))
    if unknown:
        print(f"Dataset uses intents missing from the grammar: {unknown}", file=sys.stderr)
        return 1

    if args.dry_run:
        model = mock_model({i["text"]: i["expected_intent"] for i in items})
        header = "Dry run with mock inference (grammar checked statically, no model loaded)."
    else:
        if not args.model or not args.model.is_file():
            print("No model file (--model).", file=sys.stderr)
            return 2
        model = llama_model(args.model, args.threads, args.ctx)
        model(SYSTEM_PROMPT, 'Question: "Hallo"', grammar)  # warm-up, not timed
        header = (f"Model `{args.label or args.model.name}` · {args.threads} threads · "
                  f"{platform.system()} {platform.machine()} · Python {platform.python_version()}")

    runner = VyvyBenchRunner(model, grammar, IntentParser(intents), args.min_confidence, args.timeout)
    result = runner.run(items)
    text = report(result, header)
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write(text)
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
