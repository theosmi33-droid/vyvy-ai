# Training a Vyvy AI adapter

`train_lora.py` trains a LoRA adapter on `data/train/*.jsonl` and exports only
the adapter as GGUF. The base model is never merged or redistributed; the
benchmark applies the adapter on top of the unchanged base model.

Requirements: one GPU with 16 GB memory (fp16 is enough, bf16 is not needed)
and internet access for the base model.

Clone this repository on the training machine, then from its root:

```bash
pip install -q -r training/requirements.txt
python training/train_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --name vyvy-1.5b-v1
# 7B: --base Qwen/Qwen2.5-7B-Instruct --name vyvy-7b-v1 --batch 4 --grad-accum 4
```

What the script does, in this order:

1. Holdout lock: stops before training if any training sentence is close to a
   holdout sentence (`tools/check_holdout_leak.py`).
2. Validates every training line against the output contract.
3. Trains with the exact prompt of the benchmark, loss on the answer only. The
   answer is the JSON object of the contract with `confidence` fixed at 0.9.
   A checkpoint is written after every epoch to `out/checkpoints`.
4. Writes `out/adapter` (PEFT), `out/<name>.gguf` and `out/manifest.json` with
   hashes of the adapter and the training files.

Run it as a background job of the notebook so that closing the browser does not
stop it; no GitHub credentials are needed on the training machine.

## Training without a local machine

Kaggle allows 30 GPU hours per week (reset Saturday 00:00 UTC). Each training
run is a long, budgeted sweep on one GPU; everything else runs on GitHub's CPU
runners. Nothing needs manual approval or checking.

1. **Train** (`train.yml`) plans the run (`tools/plan_train.py`): size, sweep,
   time budget and the adapter name, a hash of training data, prompt, trainer
   and sweep. It checks the holdout lock, the live Kaggle quota (the whole
   budget plus a 2 h reserve must be left) and that no kernel of the account is
   running (`tools/kaggle_busy.py`, also other projects' kernels). Then it starts
   a private Kaggle kernel and ends. A push to `main` that changes the training
   data or code starts it for 1.5B.
2. **On Kaggle** the trainer runs a sweep within its budget (1.5B: 4 h over
   ranks 16/8/32 × learning rates 2e-4/1e-4/5e-4; 7B: 6 h over ranks 16/32 ×
   2e-4/1e-4). The proven default runs first. Time left after the grid goes to
   refinement around the best config: two more seeds, then double and half
   rank, half and 1.5× learning rate. Every config holds out 10 % per
   intent for validation, stops early when the validation loss no longer
   improves, and is scored by validation accuracy with the same decision the
   benchmark's route mode makes. No new config starts if it would likely not
   finish within the budget. The best config's PEFT adapter is kept; fallbacks
   handle out-of-memory and 4-bit kernel errors.
3. **Collect** (`collect.yml`) picks up finished runs. Train starts it right
   after a kernel starts, and it re-runs itself every 15 minutes while any
   kernel of the account runs (a schedule every 20 minutes is only a backup,
   because GitHub did not execute it reliably). It
   converts the adapter to GGUF on the runner, adds the base model's Apache 2.0
   license and a NOTICE, publishes a candidate (pre-release) with the sweep
   table and starts **Bench LoRA**. A failed run becomes one GitHub issue with
   the likely cause and the last 100 log lines; the same inputs are not retried
   until that issue is closed, while new data, a new prompt or a new sweep plan
   start anew. When no
   kernel is running, Collect starts the next missing run (1.5B before 7B).
4. **Bench LoRA** measures the current default and the candidate in one job,
   on blind holdout v2 (decides) and on blind holdout v1 and the development
   holdout (information). The candidate becomes a regular release, the new
   default, only under the statistical rule in the main README (exact McNemar
   p < 0.05, b ≥ 2c, per-intent tolerance, no loss on `unknown`); otherwise it
   stays a pre-release marked rejected, with the report attached.

Secrets in the environment `kaggle`: `KAGGLE_API_TOKEN` and `KAGGLE_USERNAME`.
The token can do everything in the Kaggle account; if it ever shows up in a log,
expire it on Kaggle and set a new one.

## Weekly budget

| Resource | Free limit | How it is used |
|---|---|---|
| Kaggle GPU (T4) | 30 h per week, reset Saturday 00:00 UTC | teacher synthesis up to 2.5 h per change of `training/synthesize.py` (runs first), 1.5B sweep 4 h, 7B sweep 6 h per change of data, prompt or sweep plan; every sweep uses its whole budget (grid, then seeds and neighbours of the best config); 2 h stay in reserve; other projects on the same account take turns, no run starts while another kernel runs |
| GitHub Actions, Linux | unlimited for public repositories | benchmarks, comparisons, GGUF conversion, Collect |
| GitHub Actions, macOS (Apple Silicon) | unlimited for public repositories | Bench Mac: latency with Metal on the target hardware |
| Kaggle datasets | over 200 GB private | pinned model mirror |
| Google Colab | free T4 when available, one machine at a time | the official Colab CLI starts and drives it without a browser; used for the teacher synthesis and for runs that would otherwise wait for Kaggle |

## Measuring the adapter

Publish `out/<name>.gguf` as the only `.gguf` asset of a release in this
repository (with `out/manifest.json` next to it). Then start the **Benchmark**
workflow with that release tag in the `adapter` field; leave it empty to
measure the base model alone.

The adapter keeps the license terms of its base model.
