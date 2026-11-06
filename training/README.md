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

Actions → **Train**: pick the base model and an adapter name. The workflow
checks the holdout lock, sends the training job to a Kaggle GPU (private
kernel, trained from the exact commit of the run), waits for it, fetches only
`<name>.gguf` and `manifest.json`, checks the adapter against the checksum in
the manifest, publishes both as release `<name>` and starts the benchmark with
that adapter in `route` mode.

It needs two secrets in the repository environment `kaggle`, which accepts runs
from `main` only: `KAGGLE_API_TOKEN` (Kaggle → Settings → API Tokens) and
`KAGGLE_USERNAME`. The token can do everything in the Kaggle account; if it ever
shows up in a log, expire it on Kaggle and set a new one. The Kaggle account needs a verified phone number
for GPU and internet access.

## Measuring the adapter

Publish `out/<name>.gguf` as the only `.gguf` asset of a release in this
repository (with `out/manifest.json` next to it). Then start the **Benchmark**
workflow with that release tag in the `adapter` field; leave it empty to
measure the base model alone.

The adapter keeps the license terms of its base model.
