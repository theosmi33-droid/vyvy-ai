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
without manual approval.

1. A push to `main` that changes `data/train/` or the workflow starts **Train**
   (1.5B, then 7B, one GPU session at a time); it can also be started by hand
   for one base model.
2. The job runs in the environment `kaggle`; its secrets are usable from `main`
   only.
3. Holdout lock, a private Kaggle kernel trains from the exact
   commit with the prompt the benchmark uses for that model size, and only
   `<name>.gguf` and `manifest.json` come back, checked against the manifest's
   checksum. The adapter is published as a candidate (pre-release).
4. **Bench LoRA** measures base model and adapter in one job. The release becomes
   a regular release only with zero regressions and a higher intent accuracy;
   otherwise it stays a pre-release marked rejected, with the reason.

Adapters are named `vyvy-<size>-d<hash of training data and prompt>`; if that
name exists, the run skips training.

Secrets in the environment `kaggle`: `KAGGLE_API_TOKEN` (Kaggle → Settings →
API Tokens) and `KAGGLE_USERNAME`. The token can do everything in the Kaggle
account; if it ever shows up in a log, expire it on Kaggle and set a new one.

## Measuring the adapter

Publish `out/<name>.gguf` as the only `.gguf` asset of a release in this
repository (with `out/manifest.json` next to it). Then start the **Benchmark**
workflow with that release tag in the `adapter` field; leave it empty to
measure the base model alone.

The adapter keeps the license terms of its base model.
