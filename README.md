# Vyvy AI

Vyvy AI is a local language model for German office work. It understands
requests the way people actually phrase them, in regional dialect, everyday
speech or with typos, and turns each one into a structured action that an
application can carry out: show today's appointments, list overdue tasks,
create a task.

It runs entirely on the user's own computer, without a cloud service and
without a graphics card. It does not chat and does not answer questions itself;
it decides what is being asked and hands that decision over in a fixed format.

## What it does

- **German as it is spoken.** Low German, Rhineland, Bavarian, Swabian, Swiss
  German and colloquial phrasing are understood, not only standard German.
- **One fixed output format.** Every answer is a valid JSON object with
  `intent`, `timeframe`, `entity` and `confidence`. When the request is not
  covered, the answer is `unknown` instead of an invented action.
- **Fast on ordinary hardware.** A decision within 4 seconds on a CPU.
- **Voice, text and documents.** Typed requests, dictated voice messages via
  local speech recognition and, in the larger class, images and scanned
  documents.
- **Measured before release.** Every model version is checked against the
  public holdout set in this repository.

## Model classes

| Class | Memory used | Target hardware | Base model |
|---|---|---|---|
| Vyvy AI 4GB | about 4 GB | computers with 8 GB RAM | Qwen 2.5 3B Instruct |
| Vyvy AI 8GB | about 8 GB | computers with 12–16 GB RAM | Qwen 2.5 7B Instruct |

## Benchmark

This repository contains the public benchmark every Vyvy AI model has to pass:
a holdout set of German office questions, the output grammar and a runner that
measures accuracy and latency on GitHub Actions.

### Contents

| Path | Purpose |
|---|---|
| `data/holdout_benchmarks.json` | 60 synthetic German office questions in two sets, one expected intent each |
| `grammars/slots.gbnf` | Output contract: `intent`, `timeframe`, `entity`, `confidence` |
| `runner/bench.py` | Stateless benchmark runner (llama-cpp-python) |
| `.github/workflows/bench.yml` | Manual GitHub Actions run on `ubuntu-latest` |
| `data/train/` | Training data, one JSON object per line (see its README) |
| `tools/check_holdout_leak.py` | Holdout lock: rejects training sentences close to a holdout sentence |
| `.github/workflows/data-check.yml` | Runs the holdout lock on every change below `data/` |
| `training/` | Trains a LoRA adapter and exports it as GGUF (see its README) |

### Output contract

```json
{"intent": "tasks_overdue", "timeframe": null, "entity": null, "confidence": 0.8}
```

Intents: `appointments_today`, `appointments_tomorrow`, `tasks_open`,
`tasks_overdue`, `emails_unread`, `daily_agenda`, `create_appointment`,
`create_task`, `unknown`. The grammar makes `unknown` always available, so the
model has a legal way out instead of inventing an intent.

### Scoring

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

### Modes

- `--mode generate` (default): the model writes the full JSON object under the
  GBNF grammar, one token at a time.
- `--mode classify`: one forward pass over prompt and question, no generation.
  The prompt lists one letter per intent (A–I); the intent is the letter with
  the highest logit, and `confidence` is the softmax over those letters only.
  `timeframe` and `entity` stay empty in this mode. The run stops if a letter
  is not a single token of the model's tokenizer.

- `--mode route`: the same prompt as `generate`, with the answer prefilled up
  to `{"intent": `. The intent names form a token tree; the model is run only at
  its branch points (for example after `appointments_` or `tasks_`), forced
  tokens in between are evaluated together. This picks the intent greedy
  decoding under the grammar would pick, without writing the rest of the
  object. `confidence` is the probability of the chosen path; `timeframe` and
  `entity` stay empty.

All modes use the same dataset and report.

Latency is measured on the machine that runs the benchmark. Accuracy transfers
to other hardware; milliseconds do not.

### Dataset

All questions are hand-written and synthetic. They contain no real persons,
companies or customer data. The current sets cover four read intents
(`appointments_today`, `tasks_open`, `tasks_overdue`, `emails_unread`) in
Low German, Rhineland, Bavarian, Swabian, Swiss German and other regional
variants.

### Run locally

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

### Run on GitHub Actions

Actions → **Benchmark** → *Run workflow*, pick a model and a mode. To measure a
fine-tuned version, enter the release tag of this repository that holds its LoRA
adapter (one `*.gguf`); the adapter is applied on top of the base model, which
is still downloaded unchanged from Hugging Face. Locally: `--lora path/to/adapter.gguf`. The report appears as
the job summary; raw numbers are attached as the `results-…` artifact. Models
are downloaded from Hugging Face during the job and never stored in this
repository. Each model keeps its own license (see its Hugging Face page).

## License

All rights reserved, no license is granted. Use for AI training or data mining
is prohibited. See [LICENSE](LICENSE).
