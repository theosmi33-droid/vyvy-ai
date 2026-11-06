#!/usr/bin/env python3
"""Train a LoRA adapter for Vyvy AI and export only the adapter as GGUF.

Runs on one 16 GB GPU (fp16, no bf16 needed). The base model is loaded in 4 bit
(QLoRA); only the adapter is trained and saved. Nothing is merged: the result is
a small adapter .gguf that the benchmark applies on top of the unchanged base
model (runner/bench.py --lora).

Steps:
1. Holdout lock (tools/check_holdout_leak.py); any leak aborts before training.
2. Every training line is validated against the output contract.
3. Training with the exact prompt the benchmark uses; loss only on the answer.
   Checkpoints after every epoch in <out>/checkpoints.
4. <out>/adapter (PEFT) -> <out>/<name>.gguf with convert_lora_to_gguf.py from
   the llama.cpp commit that llama-cpp-python 0.3.16 ships, plus manifest.json.

    python training/train_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --name vyvy-1.5b-v1
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))
sys.path.insert(0, str(ROOT / "tools"))

import bench  # noqa: E402  (same prompt and contract as the benchmark)
import check_holdout_leak  # noqa: E402

LLAMA_CPP_COMMIT = "4227c9be4268ac844921b90f31595f81236bd317"  # vendored by llama-cpp-python 0.3.16
TARGET_CONFIDENCE = 0.9
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(paths: list[Path], intents: tuple[str, ...]) -> list[dict]:
    rows = []
    for path in paths:
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            answer = {"intent": r.get("intent"), "timeframe": r.get("timeframe"),
                      "entity": r.get("entity"), "confidence": TARGET_CONFIDENCE}
            if not isinstance(r.get("text"), str) or bench.parse_output(json.dumps(answer), intents) is None:
                raise SystemExit(f"{path.name}:{no}: not a valid training line: {line[:120]}")
            rows.append({"text": r["text"], "answer": json.dumps(answer, ensure_ascii=False)})
    return rows


def encode(tok, row: dict) -> dict:
    messages = [{"role": "system", "content": bench.SYSTEM_PROMPT},
                {"role": "user", "content": f'Question: "{row["text"]}"'}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages + [{"role": "assistant", "content": row["answer"]}], tokenize=False)
    if not full.startswith(prompt):
        raise SystemExit("Chat template does not render the prompt as a prefix of the full turn.")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if f_ids[: len(p_ids)] != p_ids:
        raise SystemExit("Prompt tokens differ inside the full turn; cannot mask the prompt.")
    return {"input_ids": f_ids, "labels": [-100] * len(p_ids) + f_ids[len(p_ids):]}


def collate(pad_id: int):
    import torch

    def run(batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}

    return run


def llama_cpp_dir(cache: Path) -> Path:
    target = cache / f"llama.cpp-{LLAMA_CPP_COMMIT}"
    if not target.is_dir():
        url = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.tar.gz"
        with urllib.request.urlopen(url) as resp:
            tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz").extractall(cache, filter="data")
    return target


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train a Vyvy AI LoRA adapter and export it as GGUF")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face id of the base model")
    p.add_argument("--name", default="vyvy-adapter", help="file name of the exported adapter")
    p.add_argument("--train", type=Path, nargs="*", help="training files (default: data/train/*.jsonl)")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    args = p.parse_args(argv)

    paths = args.train or sorted((ROOT / "data" / "train").glob("*.jsonl"))
    if not paths:
        raise SystemExit("No training data (data/train/*.jsonl).")
    if check_holdout_leak.main(["--train", *map(str, paths)]) != 0:
        raise SystemExit("Holdout lock failed; not training.")
    intents = bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))
    rows = load_rows(paths, intents)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(args.base)
    data = [encode(tok, r) for r in rows]
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16,
                                               bnb_4bit_use_double_quant=True),
        torch_dtype=torch.float16, device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    args.out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        train_dataset=data,
        data_collator=collate(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
            logging_steps=10, save_strategy="epoch", save_total_limit=2, fp16=True,
            optim="paged_adamw_8bit", gradient_checkpointing=True, remove_unused_columns=False,
            report_to=[], seed=0,
        ),
    )
    trainer.train()
    adapter_dir = args.out / "adapter"
    model.save_pretrained(adapter_dir)

    gguf = args.out / f"{args.name}.gguf"
    convert = llama_cpp_dir(args.out / ".cache") / "convert_lora_to_gguf.py"
    subprocess.run([sys.executable, str(convert), str(adapter_dir), "--outfile", str(gguf),
                    "--outtype", "f16", "--base-model-id", args.base], check=True)

    manifest = {
        "adapter": gguf.name, "adapter_sha256": sha256(gguf), "base": args.base,
        "llama_cpp_commit": LLAMA_CPP_COMMIT, "train_files": {x.name: sha256(x) for x in paths},
        "train_sentences": len(rows), "epochs": args.epochs, "lr": args.lr, "rank": args.rank,
        "alpha": args.alpha, "batch": args.batch, "grad_accum": args.grad_accum,
        "target_confidence": TARGET_CONFIDENCE, "holdout_lock": "passed",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
