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

Kaggle allows 30 GPU hours per week, so training runs in short bursts while
data checks and measurements run on GitHub's CPU runners. Everything runs
without manual approval or checking.

1. A push to `main` that changes `data/train/`, the training code or the
   workflow starts **Train** (1.5B, then 7B, one GPU session at a time); it can
   also be started by hand for one base model.
2. **Quota guard** (`tools/quota_guard.py`): sums the Kaggle time of all Train
   runs since Monday 00:00 UTC from the GitHub API. With less than 2 of the 30
   hours left it skips the run ("Quota paused until Monday"). GPU time used
   outside this pipeline on the same Kaggle account is not visible to it; set
   the repository variable `KAGGLE_EXTERNAL_GPU_MIN` to add it.
3. Holdout lock, then a private Kaggle kernel trains from the exact commit with
   the prompt the benchmark uses for that model size.
4. **Watchdog** (`tools/kaggle_watchdog.py`): polls every 60 s for up to 3 h.
   On success it fetches only `<name>.gguf` and `manifest.json`; on failure it
   writes the likely cause (out of memory, CUDA or 4-bit error, missing module,
   API mismatch, disk, network) and the last 100 log lines to the job summary.
5. **Fallbacks in the trainer**: out of GPU memory → batch 1 with the same
   effective batch via gradient accumulation; CUDA or 4-bit kernel error →
   plain fp16 weights. At most three attempts; the manifest lists them.
6. The adapter is checked against its manifest checksum and published as a
   candidate (pre-release); **Bench LoRA** starts by itself.
7. **Bench LoRA** measures the current default and the candidate in one job. The
   default is the newest accepted adapter for the same base model and prompt,
   or the base model if there is none. The candidate becomes a regular release
   (the new default) only with zero regressions and a higher intent accuracy;
   otherwise it stays a pre-release marked rejected, with the report and the
   missed questions attached.

Adapters are named `vyvy-<size>-d<hash of training data and prompt>`; if that
name exists, the run skips training. Secrets in the environment `kaggle`:
`KAGGLE_API_TOKEN` and `KAGGLE_USERNAME`. The token can do everything in the
Kaggle account; if it ever shows up in a log, expire it on Kaggle and set a new
one.

## Measuring the adapter

Publish `out/<name>.gguf` as the only `.gguf` asset of a release in this
repository (with `out/manifest.json` next to it). Then start the **Benchmark**
workflow with that release tag in the `adapter` field; leave it empty to
measure the base model alone.

The adapter keeps the license terms of its base model.
