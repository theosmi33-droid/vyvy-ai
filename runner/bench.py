#!/usr/bin/env python3
"""Vyvy AI benchmark: German intent & slot extraction with a local GGUF model.

Every question in the dataset is sent once to the model, constrained by a GBNF
grammar, at temperature 0. A prediction counts as correct only if the output
parses, the intent matches and the model's own confidence is at least
--min-confidence. The time limit is scored separately ("within timeout"), so a
slow but right answer still counts for intent accuracy. Latency is wall-clock
time per question on the machine that runs the benchmark, split into prompt
processing and generation from llama.cpp's own counters; it does not transfer
to other hardware.

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
import math
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

INTENT_HELP = {
    "appointments_today": "own appointments today",
    "appointments_tomorrow": "own appointments tomorrow",
    "tasks_open": "what is still to do",
    "tasks_overdue": "overdue, forgotten or urgent work",
    "emails_unread": "new or unread email",
    "daily_agenda": "what is on today in general",
    "create_appointment": "make a new appointment",
    "create_task": "make a new task",
    "unknown": "anything else, including general knowledge and questions about other companies",
}
CLASSIFY_EXAMPLES = (
    ("Was liegt heut im Kalender, gell?", "appointments_today"),
    ("Hab ich irgendwas Dringendes verpennt?", "tasks_overdue"),
    ("Wer hat die Glühbirne erfunden?", "unknown"),
)


def classify_prompt(intents: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    """System prompt for --mode classify and the letter for each intent (A, B, ...)."""
    letters = {intent: chr(ord("A") + i) for i, intent in enumerate(intents)}
    lines = [
        "You classify one short German question or request from an office worker "
        "(often regional dialect, colloquial wording or typos). You never answer the question. "
        "Reply with exactly one letter:",
        *(f"{letters[i]} = {i} ({INTENT_HELP.get(i, i)})" for i in intents),
        "Examples:",
        *(f'Question: "{q}" -> {letters[i]}' for q, i in CLASSIFY_EXAMPLES if i in letters),
    ]
    return "\n".join(lines), letters


def pick_intent(logits: dict[str, float]) -> tuple[str, dict[str, float]]:
    """Highest logit wins; confidence is the softmax over the intent letters only."""
    top = max(logits.values())
    exp = {i: math.exp(v - top) for i, v in logits.items()}
    total = sum(exp.values())
    probs = {i: e / total for i, e in exp.items()}
    return max(probs, key=probs.get), probs


# A model returns its raw text plus timing stats (may be empty).
Model = Callable[[str, str, str], tuple[str, dict]]
TIMING_KEYS = ("prompt_tokens", "prompt_eval_tokens", "completion_tokens",
               "prompt_ms", "gen_ms", "gen_tps")


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


def thread_info(llm) -> str:
    return f"n_threads {llm.n_threads} · n_threads_batch {llm.n_threads_batch} · n_batch {llm.n_batch}"


def _perf_stats(llama_cpp, llm) -> dict:
    perf = llama_cpp.llama_perf_context(llm.ctx)
    return {
        "prompt_eval_tokens": perf.n_p_eval,
        "prompt_ms": perf.t_p_eval_ms,
        "gen_ms": perf.t_eval_ms,
        "gen_tps": perf.n_eval / perf.t_eval_ms * 1000 if perf.t_eval_ms > 0 else None,
    }


def llama_model(path: Path, threads: int, ctx: int, lora: Path | None = None) -> Model:
    import llama_cpp  # imported only for real runs
    from llama_cpp import Llama, LlamaGrammar

    llm = Llama(model_path=str(path), n_ctx=ctx, n_threads=threads, seed=0, verbose=False,
                lora_path=str(lora) if lora else None)
    cache: dict[str, object] = {}

    def run(system: str, question: str, grammar: str) -> tuple[str, dict]:
        if grammar not in cache:
            cache[grammar] = LlamaGrammar.from_string(grammar, verbose=False)
        llama_cpp.llama_perf_context_reset(llm.ctx)
        out = llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
            grammar=cache[grammar],
            temperature=0.0,
            max_tokens=80,
        )
        # prompt_eval_tokens < prompt_tokens means the shared prefix came from the cache.
        usage = out.get("usage") or {}
        stats = {"prompt_tokens": usage.get("prompt_tokens"),
                 "completion_tokens": usage.get("completion_tokens"), **_perf_stats(llama_cpp, llm)}
        return out["choices"][0]["message"]["content"] or "", stats

    run.info = thread_info(llm)
    return run


def _chat_llm(path: Path, threads: int, ctx: int, mode: str, lora: Path | None = None):
    """Model plus a formatter for its own chat template (classify and route build prompts themselves)."""
    import llama_cpp  # imported only for real runs
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    llm = Llama(model_path=str(path), n_ctx=ctx, n_threads=threads, seed=0, verbose=False,
                lora_path=str(lora) if lora else None)
    template = llm.metadata.get("tokenizer.chat_template")
    if not template:
        raise SystemExit(f"Model has no chat template; --mode {mode} needs one.")
    eos, bos = llm.token_eos(), llm.token_bos()
    fmt = Jinja2ChatFormatter(
        template=template,
        eos_token=llm._model.token_get_text(eos) if eos != -1 else "",
        bos_token=llm._model.token_get_text(bos) if bos != -1 else "",
    )
    return llama_cpp, llm, fmt


def _eval_cached(llama_cpp, llm, tokens: list[int]) -> Callable[[int], float]:
    """Evaluate tokens, reusing the cached prefix like Llama.generate; logits of the last position."""
    prefix = 0
    for a, b in zip(llm.input_ids[: llm.n_tokens], tokens[:-1]):
        if a != b:
            break
        prefix += 1
    llm.n_tokens = prefix
    llm.eval(tokens[prefix:])
    logits = llama_cpp.llama_get_logits_ith(llm.ctx, -1)
    return lambda t: float(logits[t])


def llama_classifier(path: Path, threads: int, ctx: int, intents: tuple[str, ...],
                     lora: Path | None = None) -> Model:
    """One forward pass over prompt and question, no generation.

    The intent is the letter with the highest logit among the intent letters;
    confidence is the softmax over those letters only.
    """
    llama_cpp, llm, fmt = _chat_llm(path, threads, ctx, "classify", lora)
    system, letters = classify_prompt(intents)
    letter_ids: dict[str, int] = {}
    for intent, letter in letters.items():
        ids = llm.tokenize(letter.encode(), add_bos=False, special=False)
        if len(ids) != 1:
            raise SystemExit(f"Letter {letter} is not a single token: {ids}")
        letter_ids[intent] = ids[0]
    if len(set(letter_ids.values())) != len(letter_ids):
        raise SystemExit("Intent letters share a token.")

    def run(_system: str, question: str, _grammar: str) -> tuple[str, dict]:
        prompt = fmt(messages=[{"role": "system", "content": system},
                               {"role": "user", "content": question}]).prompt
        tokens = llm.tokenize(prompt.encode("utf-8"), special=True)
        llama_cpp.llama_perf_context_reset(llm.ctx)
        logit = _eval_cached(llama_cpp, llm, tokens)
        best, probs = pick_intent({i: logit(t) for i, t in letter_ids.items()})
        stats = {"prompt_tokens": len(tokens), "completion_tokens": 0, **_perf_stats(llama_cpp, llm),
                 "probs": {i: round(p, 4) for i, p in probs.items()}}
        return json.dumps({"intent": best, "timeframe": None, "entity": None,
                           "confidence": round(probs[best], 4)}), stats

    run.info = thread_info(llm) + f" · letters {' '.join(f'{v}={k}' for k, v in letters.items())}"
    return run


ROUTE_PREFILL = '{"intent": '


def route_candidates(tokenize: Callable[[str], list[int]], prompt: str,
                     intents: tuple[str, ...]) -> tuple[list[int], dict[str, list[int]]]:
    """Shared token prefix and the remaining tokens of each intent value.

    Every candidate is tokenized as one text (prompt + prefill + '"intent"'), so
    the token boundary at the join is the one the tokenizer itself would choose.
    """
    full = {i: tokenize(prompt + ROUTE_PREFILL + json.dumps(i)) for i in intents}
    seqs = list(full.values())
    n = 0
    while all(len(t) > n for t in seqs) and len({t[n] for t in seqs}) == 1:
        n += 1
    return seqs[0][:n], {i: t[n:] for i, t in full.items()}


def walk_route(cands: dict[str, list[int]], logit: Callable[[int], float],
               eval_more: Callable[[list[int]], Callable[[int], float]]) -> tuple[str, float, int, int]:
    """Greedy walk through the token trie of the intent values.

    Only branch points cost a model pass; forced tokens up to the next branch are
    evaluated together. This picks the intent that greedy decoding under the
    grammar would pick, and stops as soon as one intent is left.
    Returns intent, path probability, extra passes and extra tokens evaluated.
    """
    alive, depth, conf, pending, passes, tokens = list(cands), 0, 1.0, [], 0, 0
    while len(alive) > 1:
        groups: dict[int, list[str]] = {}
        for i in alive:
            if len(cands[i]) <= depth:
                raise ValueError(f"intent {i} is a token prefix of another intent")
            groups.setdefault(cands[i][depth], []).append(i)
        if len(groups) > 1:
            if pending:
                logit = eval_more(pending)
                passes, tokens, pending = passes + 1, tokens + len(pending), []
            tok, probs = pick_intent({t: logit(t) for t in groups})
            conf *= probs[tok]
            alive = groups[tok]
        else:
            tok = next(iter(groups))
        pending.append(tok)
        depth += 1
    return alive[0], conf, passes, tokens


def llama_router(path: Path, threads: int, ctx: int, intents: tuple[str, ...],
                 lora: Path | None = None) -> Model:
    """Same prompt as --mode generate, but only the intent decision is computed.

    The answer is prefilled with '{"intent": '; then walk_route reads the logits
    at each branch of the intent names. timeframe and entity stay empty.
    prompt_ms is the pass over prompt and question, gen_ms the branch passes
    (both wall-clock).
    """
    llama_cpp, llm, fmt = _chat_llm(path, threads, ctx, "route", lora)

    def tokenize(text: str) -> list[int]:
        return llm.tokenize(text.encode("utf-8"), special=True)

    def eval_more(toks: list[int]) -> Callable[[int], float]:
        llm.eval(toks)
        logits = llama_cpp.llama_get_logits_ith(llm.ctx, -1)
        return lambda t: float(logits[t])

    def run(system: str, question: str, _grammar: str) -> tuple[str, dict]:
        prompt = fmt(messages=[{"role": "system", "content": system},
                               {"role": "user", "content": question}]).prompt
        shared, cands = route_candidates(tokenize, prompt, intents)
        llama_cpp.llama_perf_context_reset(llm.ctx)
        t0 = time.perf_counter()
        logit = _eval_cached(llama_cpp, llm, shared)
        t1 = time.perf_counter()
        best, conf, passes, extra = walk_route(cands, logit, eval_more)
        t2 = time.perf_counter()
        perf = _perf_stats(llama_cpp, llm)
        stats = {"prompt_tokens": len(shared), "prompt_eval_tokens": perf["prompt_eval_tokens"],
                 "completion_tokens": extra, "prompt_ms": (t1 - t0) * 1000, "gen_ms": (t2 - t1) * 1000,
                 "gen_tps": None, "route_passes": passes}
        return json.dumps({"intent": best, "timeframe": None, "entity": None,
                           "confidence": round(conf, 4)}), stats

    run.info = thread_info(llm) + f" · prefill `{ROUTE_PREFILL}`"
    return run


def mock_model(expected: dict[str, str]) -> Model:
    """Deterministic stand-in: right for most items, wrong, unsure or broken for a few."""
    counter = {"n": 0}

    def run(_system: str, question: str, _grammar: str) -> tuple[str, dict]:
        counter["n"] += 1
        n, intent = counter["n"], expected[question.removeprefix('Question: "').removesuffix('"')]
        stats = {"prompt_tokens": 300, "prompt_eval_tokens": 20, "completion_tokens": 30,
                 "prompt_ms": 0.0, "gen_ms": 0.0, "gen_tps": None}
        if n % 13 == 0:
            return "not json", stats
        if n % 11 == 0:
            intent = "unknown"
        conf = 0.3 if n % 17 == 0 else 0.9
        return json.dumps({"intent": intent, "timeframe": None, "entity": None, "confidence": conf}), stats

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
    """Intent accuracy and the time limit are scored separately.

    intent_correct: right intent regardless of speed. within_timeout: right intent
    and at most timeout_s. A timeout never changes what the model answered.
    """
    per_intent: dict[str, dict] = {}
    records: list[dict] = []
    for item in items:
        t0 = clock()
        raw, stats = model(SYSTEM_PROMPT, f'Question: "{item["text"]}"', grammar)
        ms = (clock() - t0) * 1000
        got = outcome(parse_output(raw, intents), min_confidence)
        ok = got == item["expected_intent"]
        timed_out = ms > timeout_s * 1000
        rec = {"id": item["id"], "expected": item["expected_intent"], "got": got, "ms": round(ms),
               "timeout": timed_out, **{k: stats.get(k) for k in TIMING_KEYS}}
        if rec["prompt_ms"] is not None and rec["gen_ms"] is not None:
            rec["other_ms"] = round(ms - rec["prompt_ms"] - rec["gen_ms"])
        for extra in ("probs", "route_passes"):
            if extra in stats:
                rec[extra] = stats[extra]
        records.append(rec)
        row = per_intent.setdefault(item["expected_intent"],
                                    {"n": 0, "intent_correct": 0, "within_timeout": 0, "timeouts": 0,
                                     "ms": [], "misses": []})
        row["n"] += 1
        row["ms"].append(ms)
        row["intent_correct"] += ok
        row["within_timeout"] += ok and not timed_out
        row["timeouts"] += timed_out
        if not ok or timed_out:
            row["misses"].append({"id": item["id"], "text": item["text"], "got": got,
                                  "ms": round(ms), "timeout": timed_out})
    all_ms = [m for r in per_intent.values() for m in r["ms"]]
    return {
        "n": sum(r["n"] for r in per_intent.values()),
        "intent_correct": sum(r["intent_correct"] for r in per_intent.values()),
        "within_timeout": sum(r["within_timeout"] for r in per_intent.values()),
        "timeouts": sum(r["timeouts"] for r in per_intent.values()),
        "median_ms": round(statistics.median(all_ms)) if all_ms else 0,
        "timeout_s": timeout_s,
        "timing": timing_summary(records),
        "per_intent": {
            k: {"n": r["n"], "intent_correct": r["intent_correct"], "within_timeout": r["within_timeout"],
                "timeouts": r["timeouts"], "accuracy": r["intent_correct"] / r["n"],
                "median_ms": round(statistics.median(r["ms"])), "mean_ms": round(statistics.mean(r["ms"])),
                "misses": r["misses"]}
            for k, r in sorted(per_intent.items())
        },
        "items": records,
    }


def timing_summary(records: list[dict]) -> dict:
    out = {}
    for key in ("ms", *TIMING_KEYS, "other_ms", "route_passes"):
        vals = [r[key] for r in records if r.get(key) is not None]
        if vals:
            out[key] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
    return out


TIMING_LABELS = {
    "ms": "Total ms per question",
    "prompt_tokens": "Prompt tokens",
    "prompt_eval_tokens": "Prompt tokens evaluated (rest from cache)",
    "completion_tokens": "Completion tokens",
    "prompt_ms": "Prompt processing ms",
    "gen_ms": "Generation ms",
    "other_ms": "Other ms (sampling, grammar, overhead)",
    "gen_tps": "Generation tokens/s",
    "route_passes": "Branch passes after the prompt (route)",
}


def _fmt(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.1f}"


def report(result: dict, header: str) -> str:
    n, c, w = result["n"], result["intent_correct"], result["within_timeout"]
    lines = [
        "## Vyvy AI benchmark",
        "",
        header,
        "",
        f"**Accuracy (intent): {c}/{n} ({c / n:.1%})** · right intent, regardless of speed",
        "",
        f"**Within timeout: {w}/{n} ({w / n:.1%})** · right intent and at most {result['timeout_s']:g} s"
        f" · {result['timeouts']} timeouts · median {result['median_ms']} ms per question",
        "",
        "| Intent | Correct | Within timeout | Timeouts | N | Accuracy | Median ms | Mean ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, r in result["per_intent"].items():
        lines.append(f"| {k} | {r['intent_correct']} | {r['within_timeout']} | {r['timeouts']} | {r['n']}"
                     f" | {r['accuracy']:.1%} | {r['median_ms']} | {r['mean_ms']} |")
    if result["timing"]:
        lines += ["", "### Timing per question", "", "| Value | Median | Min | Max |", "|---|---:|---:|---:|"]
        for key, label in TIMING_LABELS.items():
            t = result["timing"].get(key)
            if t:
                lines.append(f"| {label} | {_fmt(t['median'])} | {_fmt(t['min'])} | {_fmt(t['max'])} |")
    misses = [(k, m) for k, r in result["per_intent"].items() for m in r["misses"]]
    if misses:
        lines += ["", "### Misses", "", "| Expected | ID | Question | Got | ms | Timeout |",
                  "|---|---|---|---|---:|---|"]
        lines += [f"| {k} | {m['id']} | {m['text']} | {m['got']} | {m['ms']} | {'yes' if m['timeout'] else ''} |"
                  for k, m in misses]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Vyvy AI intent & slot extraction benchmark")
    p.add_argument("--model", type=Path, default=os.environ.get("VYVY_MODEL_PATH") or None,
                   help="GGUF model file (default: $VYVY_MODEL_PATH)")
    p.add_argument("--lora", type=Path, default=os.environ.get("VYVY_LORA_PATH") or None,
                   help="LoRA adapter in GGUF format, applied on top of --model (default: $VYVY_LORA_PATH)")
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
    p.add_argument("--mode", choices=("generate", "classify", "route"), default="generate",
                   help="generate: JSON under the grammar; classify: one pass, intent letter by logits; "
                        "route: generate prompt, only the intent branches are computed")
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
        header = f"Dry run with mock inference, mode `{args.mode}` (grammar checked statically, no model loaded)."
        if args.mode == "classify":
            header += "\n\n```\n" + classify_prompt(intents)[0] + "\n```"
    else:
        if not args.model or not args.model.is_file():
            print("No model file (--model).", file=sys.stderr)
            return 2
        if args.lora and not args.lora.is_file():
            print(f"No adapter file: {args.lora}", file=sys.stderr)
            return 2
        if args.mode == "classify":
            model = llama_classifier(args.model, args.threads, args.ctx, intents, args.lora)
        elif args.mode == "route":
            model = llama_router(args.model, args.threads, args.ctx, intents, args.lora)
        else:
            model = llama_model(args.model, args.threads, args.ctx, args.lora)
        model(SYSTEM_PROMPT, 'Question: "Hallo"', grammar)  # warm-up, not timed
        adapter = f" + adapter `{args.lora.name}`" if args.lora else " · no adapter"
        header = (f"Model `{args.label or args.model.name}`{adapter} · mode `{args.mode}` · "
                  f"{platform.system()} {platform.machine()} · Python {platform.python_version()}"
                  f"\n\n{model.info}")

    runner = VyvyBenchRunner(model, grammar, IntentParser(intents), args.min_confidence, args.timeout)
    result = runner.run(items)
    result["mode"] = args.mode
    result["adapter"] = args.lora.name if args.lora else None
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
