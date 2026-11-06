# Vyvy AI: Structured German Intent & Slot Extraction Benchmarks for Local SLMs

A standalone, low-latency German intent and slot-extraction benchmark suite for
small language models (SLMs) using GBNF grammars.

How well does a small language model running on a CPU turn colloquial German
(regional dialects, slang, typos) into a structured, machine-readable request?

The benchmark sends each question once to a local GGUF model, constrained by a
GBNF grammar, and checks the resulting JSON object against the expected intent.

## Contents

| Path | Purpose |
|---|---|
| `data/holdout_benchmarks.json` | 60 synthetic German office questions in two sets, one expected intent each |
| `grammars/slots.gbnf` | Output contract: `intent`, `timeframe`, `entity`, `confidence` |
| `runner/bench.py` | Stateless benchmark runner (llama-cpp-python) |
| `.github/workflows/bench.yml` | Manual GitHub Actions run on `ubuntu-latest` |

## Output contract

```json
{"intent": "tasks_overdue", "timeframe": null, "entity": null, "confidence": 0.8}
```

Intents: `appointments_today`, `appointments_tomorrow`, `tasks_open`,
`tasks_overdue`, `emails_unread`, `daily_agenda`, `create_appointment`,
`create_task`, `unknown`. The grammar makes `unknown` always available, so the
model has a legal way out instead of inventing an intent.

## Scoring

A prediction is correct only if the output parses, the intent matches, and the
model's own `confidence` is at least `--min-confidence` (default 0.5). The
report gives two numbers:

- **Accuracy (intent):** right intent, regardless of speed.
- **Within timeout:** right intent and at most `--timeout` seconds (default 4,
  or `VYVY_SLOT_TIMEOUT_S`).

Per intent it lists both plus the number of timeouts. Every miss shows what the
model returned (`invalid`, `low confidence (…)` or the wrong intent), its time
and whether it ran over the limit.

The timing table splits each question into prompt processing, generation and
the rest (sampling, grammar, overhead), with token counts and generation speed,
taken from llama.cpp's own counters. `results.json` holds these values for every
question.

Latency is measured on the machine that runs the benchmark. Accuracy transfers
to other hardware; milliseconds do not.

## Dataset

All questions are hand-written and synthetic. They contain no real persons,
companies or customer data. The current sets cover four read intents
(`appointments_today`, `tasks_open`, `tasks_overdue`, `emails_unread`) in
Low German, Rhineland, Bavarian, Swabian, Swiss German and other regional
variants.

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Without a model: grammar check and mock inference
python runner/bench.py --dry-run

# With a model
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='models')"
python runner/bench.py --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf
# or: VYVY_MODEL_PATH=models/qwen2.5-1.5b-instruct-q4_k_m.gguf python runner/bench.py
```

## Model classes

| Class | Target hardware | Model | Choice in the workflow |
|---|---|---|---|
| Vyvy AI 4GB | 8 GB RAM | Qwen 2.5 3B Instruct, Q5_K_M | `qwen2.5-3b-instruct-q5_k_m` |
| Vyvy AI 8GB | 12–16 GB RAM | Qwen 2.5 7B Instruct, Q4_K_M | `qwen2.5-7b-instruct-q4_k_m` |

The 1.5B and 3B Q4_K_M variants stay available as reference points.

## Run on GitHub Actions

Actions → **Benchmark** → *Run workflow*, pick a model. The report appears as
the job summary; raw numbers are attached as the `results-…` artifact. Models
are downloaded from Hugging Face during the job and never stored in this
repository. Each model keeps its own license (see its Hugging Face page).

## License

All rights reserved, no license is granted. Use for AI training or data mining
is prohibited. See [LICENSE](LICENSE).
