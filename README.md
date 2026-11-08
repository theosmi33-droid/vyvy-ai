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
| Vyvy AI 4GB | about 4 GB | computers with 8 GB RAM | Qwen 2.5 1.5B Instruct |
| Vyvy AI 8GB | about 8 GB | computers with 12–16 GB RAM | Qwen 2.5 7B Instruct |

Both base models are released under Apache 2.0.

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
| `data/holdout_blind_v2.json` | Blind holdout v2: 270 questions, 30 per intent, from four independent writers; decides adapter acceptance |
| `data/holdout_blind_v1.json` | Blind holdout v1: 90 questions, reported for information |
| `data/train/` | Training data, one JSON object per line (see its README) |
| `tools/check_holdout_leak.py` | Holdout lock: rejects training sentences close to a holdout sentence |
| `.github/workflows/data-check.yml` | Runs the holdout lock on every change below `data/` |
| `.github/workflows/compare.yml` | Both classes and a reference in one job on the same machine |
| `tools/compare_results.py` | One table over several `results.json` |
| `.github/workflows/bench-lora.yml` | A trained adapter against its base model, one job; accepts or rejects it |
| `tools/compare_adapter.py` | Statistical acceptance rule (exact McNemar test, net gain, per-intent tolerance, no loss on `unknown`) |
| `tools/previous_adapter.py` | Finds the current default adapter (newest accepted) per base model |
| `tools/kaggle_watchdog.py` | Watches a Kaggle run, fetches the adapter or diagnoses the failure |
| `models.json` | Fixed Hugging Face commit and SHA-256 of every model file used |
| `tools/fetch_model.py` | Downloads a model exactly as pinned in `models.json` and verifies it |
| `tools/license_files.py` | Adds the base model's Apache 2.0 license and a NOTICE to every adapter release |
| `.github/workflows/mirror.yml` | Keeps a private Kaggle copy of every pinned model file |
| `tools/convert_adapter.py` | Converts a trained adapter to GGUF on a CPU runner |
| `tools/quota_guard.py` | Pauses training before the weekly Kaggle GPU quota runs out |
| `training/` | Trains a LoRA adapter and exports it as GGUF (see its README) |
| `.github/workflows/train.yml` | Trains an adapter on a hosted GPU, publishes it as a release and benchmarks it |
| `tools/build_train_data.py` | Builds the training data from hand-written core sentences |
| `.github/workflows/bench-mac.yml` | Benchmark on Apple Silicon with Metal, both classes with and without their default adapter |
| `.github/workflows/collect.yml` | Every 20 min: collects finished Kaggle runs and starts the next missing one |
| `tools/collect.py` | Collect logic: publish candidates, report failures, schedule |
| `tools/plan_train.py` | Sweep, time budget and adapter name of a training run |
| `tools/kaggle_busy.py` | Lists running kernels of the Kaggle account |
| `tools/kaggle_status.py` | Read-only table of recent kernels and their status (workflow "Kaggle status") |
| `tools/kaggle_job.py` | Writes the training job for the hosted GPU from a fixed commit |

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

### Prompts

Two system prompts, chosen by model size (`--prompt auto`, the default):
`compact` for 1.5B (short definitions, three examples) and `detailed` for 7B
(explicit boundaries, six examples). Longer definitions improved 7B and made
1.5B worse, so 1.5B is never measured or trained on the detailed prompt.
Reports and adapter manifests name the prompt they used.

Latency is measured on the machine that runs the benchmark. Accuracy transfers
to other hardware; milliseconds do not.

### Dataset

All questions are hand-written and synthetic. They contain no real persons,
companies or customer data. The current sets cover four read intents
(`appointments_today`, `tasks_open`, `tasks_overdue`, `emails_unread`) in
Low German, Rhineland, Bavarian, Swabian, Swiss German and other regional
variants.

The intent definitions in the prompt were sharpened once after looking at
errors on this holdout set (what is "on today", open vs. overdue tasks), so it
serves as the development holdout. A second set, `holdout_blind_v1.json` (90
questions, 10 per intent), was written in one pass by an independent agent that
knew only the intent definitions, not the code, prompts, training data or any
errors. A third set, `holdout_blind_v2.json` (270 questions, 30 per intent), was
written the same way by four independent agents with different focuses
(southern German and phone typing, northern and western German and dictation,
workplace variety and German as a second language, less common phrasings);
sentences close to training data or the other sets were removed. Only v2
decides whether a trained adapter is accepted.

### Accepting an adapter

On few questions per intent a zero-tolerance rule measures noise: one
borderline sentence decides. The rule therefore compares baseline and adapter
question by question. With b questions the adapter fixed and c it broke, it is
accepted only if

1. the exact one-sided McNemar test on these discordant pairs gives p < 0.05,
2. b ≥ 2c,
3. no intent loses more than max(1, ⌊5 % of its questions⌋) net, and
4. no question of `unknown` gets worse (a wrong action instead of "I don't know"
   is never acceptable).

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

Runners differ in CPU from run to run, so milliseconds from separate runs are
not comparable. Every report names its CPU. **Compare** starts by
itself when the runner, the grammar, the holdout set or the dependencies change
on `main`, and can be started by hand under Actions → **Compare**: 1.5B `route`, 7B `route` and a 7B `generate` reference
run one after the other in the same job and end with one table.

Actions → **Benchmark** → *Run workflow*, pick a model and a mode. To measure a
fine-tuned version, enter the release tag of this repository that holds its LoRA
adapter (one `*.gguf`); the adapter is applied on top of the base model, which
is still downloaded unchanged from Hugging Face. Locally: `--lora path/to/adapter.gguf`. The report appears as
the job summary; raw numbers are attached as the `results-…` artifact. Models
are downloaded from Hugging Face during the job, from the commit pinned in
`models.json`, and checked against their SHA-256; they are never stored in this
repository. Each model keeps its own license (see its Hugging Face page).

## License

All rights reserved, no license is granted. Use for AI training or data mining
is prohibited. See [LICENSE](LICENSE).
