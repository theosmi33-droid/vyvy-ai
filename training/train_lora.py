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
import gc
import hashlib
import json
import math
import os
import shutil
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))
sys.path.insert(0, str(ROOT / "tools"))

import bench  # noqa: E402  (same prompt and contract as the benchmark)
import check_holdout_leak  # noqa: E402

from convert_adapter import LLAMA_CPP_COMMIT  # noqa: E402
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


def encode(tok, row: dict, system: str) -> dict:
    messages = [{"role": "system", "content": system},
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


def split_rows(rows: list[dict], share: float) -> tuple[list[dict], list[dict]]:
    """Hold out about `share` of the sentences of every intent for validation.

    Deterministic: a sentence goes to validation by the hash of its text, so the
    split never changes between runs and never depends on the order of files.
    """
    by_intent: dict[str, list[dict]] = {}
    for r in rows:
        by_intent.setdefault(json.loads(r["answer"])["intent"], []).append(r)
    train, held = [], []
    for group in by_intent.values():
        ranked = sorted(group, key=lambda r: hashlib.sha256(r["text"].encode("utf-8")).hexdigest())
        k = max(1, round(len(ranked) * share)) if share > 0 and len(ranked) > 1 else 0
        held += ranked[:k]
        train += ranked[k:]
    return train, held


def sweep_configs(spec: str, rank: int, lr: float) -> list[tuple[int, float]]:
    """(rank, lr) pairs to train; the first values of each list (or --rank/--lr) come first."""
    values = {"rank": [rank], "lr": [lr]}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        key, _, items = part.partition("=")
        if key.strip() not in values:
            raise SystemExit(f"unknown sweep key: {key}")
        cast = int if key.strip() == "rank" else float
        values[key.strip()] = [cast(x) for x in items.split(",") if x.strip()]
    pairs = [(r, x) for r in values["rank"] for x in values["lr"]]
    return sorted(pairs, key=lambda p: (p != (values["rank"][0], values["lr"][0])))


def refine_configs(rank: int, lr: float, tried: set) -> list[tuple[int, float, int]]:
    """Configs around the best one, most informative first: other seeds (is the gain
    stable?), then neighbouring ranks and learning rates. Skips anything tried."""
    cands = [(rank, lr, 1), (rank, lr, 2), (rank * 2, lr, 0), (max(4, rank // 2), lr, 0),
             (rank, lr * 0.5, 0), (rank, lr * 1.5, 0)]
    out = []
    for c in cands:
        c = (c[0], float(f"{c[1]:.3g}"), c[2])
        if c not in tried and c not in out:
            out.append(c)
    return out


def time_left_for_next(elapsed: float, durations: list[float], budget: float, margin: float = 15) -> bool:
    """Start another config only if the slowest one so far would still fit, plus a margin for export."""
    if budget <= 0:
        return True
    return elapsed + max(durations) * 1.15 + margin <= budget


def pick_best(results: list[dict]) -> int:
    """Highest validation accuracy; ties and missing accuracies decided by the lower validation loss."""
    def key(i: int):
        r = results[i]
        acc = r.get("val_accuracy")
        loss = r.get("eval_loss")
        return (-(acc if acc is not None else -1), loss if loss is not None else float("inf"), i)
    return min(range(len(results)), key=key)


def validation_accuracy(model, tok, rows: list[dict], system: str, intents: tuple[str, ...]) -> float:
    """Share of validation sentences whose expected intent scores highest after '{"intent": '.

    Every intent value is scored by the summed log-probability of its tokens, the
    same decision the benchmark's route mode makes, so selection matches what is
    later measured. Left padding with explicit positions keeps all endings aligned.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    cands = [json.dumps(i) for i in intents]
    correct = 0
    with torch.no_grad():
        for r in rows:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": f'Question: "{r["text"]}"'}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + '{"intent": '
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            seqs, spans = [], []
            for c in cands:
                f_ids = tok(prompt + c, add_special_tokens=False)["input_ids"]
                k = 0
                while k < min(len(p_ids), len(f_ids)) and p_ids[k] == f_ids[k]:
                    k += 1
                seqs.append(f_ids)
                spans.append(k)
            width = max(len(x) for x in seqs)
            keep = max(len(x) - k for x, k in zip(seqs, spans)) + 1
            ids = torch.tensor([[pad] * (width - len(x)) + x for x in seqs], device=device)
            mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in seqs], device=device)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -keep:].float()
            logp = torch.log_softmax(logits, dim=-1)
            scores = []
            for j, (x, k) in enumerate(zip(seqs, spans)):
                total = 0.0
                for q in range(width - len(x) + k, width):  # padded positions of the candidate tokens
                    total += logp[j, (q - 1) - (width - keep), ids[j, q]].item()
                scores.append(total)
            best = max(range(len(cands)), key=lambda j: scores[j])
            correct += intents[best] == json.loads(r["answer"])["intent"]
    return round(correct / len(rows), 4) if rows else None


def failure_kind(exc: BaseException) -> str:
    """Classify a training failure: out of GPU memory, a CUDA/4-bit kernel error, or other."""
    text = f"{type(exc).__name__}: {exc}"
    # cuBLAS failing to allocate its workspace is the GPU running out of memory too.
    if "OutOfMemoryError" in text or "out of memory" in text.lower() or "ALLOC_FAILED" in text:
        return "oom"
    if any(k in text for k in ("CUBLAS", "CUDA error", "cuDNN", "bitsandbytes")):
        return "kernel"
    return "other"


def fallback(kind: str, quant: str, batch: int, accum: int, fp16_fits: bool = True) -> tuple[str, int, int] | None:
    """Next configuration after a failure, or None if there is nothing sensible left to try.

    Out of memory: batch 1, gradient accumulation raised so the effective batch stays.
    Kernel error with 4-bit weights: plain fp16 weights, but only where they fit a 16 GB
    GPU (1.5B); 7B in fp16 needs about 15 GB for the weights alone, so it stops instead.
    """
    if kind == "oom" and batch > 1:
        return quant, 1, batch * accum
    if kind == "kernel" and quant == "4bit" and fp16_fits:
        return "none", batch, accum
    return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train a Vyvy AI LoRA adapter and export it as GGUF")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face id of the base model")
    p.add_argument("--name", default="vyvy-adapter", help="file name of the exported adapter")
    p.add_argument("--train", type=Path, nargs="*", help="training files (default: data/train/*.jsonl)")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--cache", type=Path, help="where llama.cpp is unpacked (default: <out>/.cache)")
    p.add_argument("--prompt", choices=("auto", *bench.PROMPTS), default="auto",
                   help="system prompt; auto = the one the benchmark uses for this base model")
    p.add_argument("--quant", choices=("auto", "4bit", "none"), default="auto",
                   help="base weights in training; auto = 4bit for 7B, fp16 otherwise")
    p.add_argument("--no-gguf", action="store_true",
                   help="only train and save the PEFT adapter; convert to GGUF elsewhere (tools/convert_adapter.py)")
    p.add_argument("--epochs", type=float, default=3, help="upper limit; training stops earlier when "
                   "the validation loss no longer improves")
    p.add_argument("--eval-share", type=float, default=0.1, help="share of training sentences held out for "
                   "validation (per intent)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--sweep", default="", help='configs to try, e.g. "rank=16,8,32;lr=2e-4,1e-4,5e-4"; '
                   "the first values are tried first; empty = only --rank and --lr")
    p.add_argument("--time-budget", type=float, default=0, help="minutes for the whole sweep; no new config "
                   "starts when it would likely not finish in time (0 = no limit)")
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
    prompt_name = bench.prompt_for(args.base) if args.prompt == "auto" else args.prompt
    system = bench.PROMPTS[prompt_name]
    print(f"Prompt: {prompt_name}")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback,
                              Trainer, TrainingArguments)

    # Fixed Hugging Face commit of the base model (models.json), never "latest".
    revision = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))["base"][args.base]
    print(f"Base model: {args.base}@{revision[:7]}")
    tok = AutoTokenizer.from_pretrained(args.base, revision=revision)
    train_rows, eval_rows = split_rows(rows, args.eval_share)
    data = [encode(tok, r, system) for r in train_rows]
    eval_data = [encode(tok, r, system) for r in eval_rows]
    print(f"Training sentences: {len(data)} · validation sentences: {len(eval_data)}")
    collator = collate(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    args.out.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.out / "adapter"

    def train_once(quant: str, batch: int, accum: int, rank: int, lr: float, seed: int, target: Path) -> dict:
        print(f"Config: rank {rank} · lr {lr:g} · seed {seed} · weights {quant} · batch {batch} · "
              f"accumulation {accum}",
              flush=True)
        if quant == "4bit":
            # QLoRA: only needed where fp16 weights would not fit next to the training state (7B).
            model = AutoModelForCausalLM.from_pretrained(
                args.base,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16,
                                                       bnb_4bit_use_double_quant=True),
                dtype=torch.float16, device_map={"": 0}, revision=revision,
            )
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16, device_map={"": 0},
                                                         revision=revision)
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
        # fp16 mixed precision needs fp32 master weights for everything that is trained.
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        model.print_trainable_parameters()
        steps = math.ceil(len(data) / (batch * accum)) * args.epochs
        trainer = Trainer(
            model=model,
            train_dataset=data,
            eval_dataset=eval_data,
            data_collator=collator,
            # Stop once the validation loss no longer improves; keep the best epoch, not the last.
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
            args=TrainingArguments(
                output_dir=str(args.out / "checkpoints"), num_train_epochs=args.epochs,
                per_device_train_batch_size=batch, gradient_accumulation_steps=accum,
                learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=max(1, round(0.03 * steps)),
                logging_steps=10, save_strategy="epoch", save_total_limit=2, fp16=True,
                eval_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="eval_loss",
                greater_is_better=False, per_device_eval_batch_size=batch,
                optim="paged_adamw_8bit" if quant == "4bit" else "adamw_torch",
                gradient_checkpointing=True, remove_unused_columns=False,
                report_to=[], seed=seed,
            ),
        )
        try:
            trainer.train()
            model.save_pretrained(target)
            history = [{k: round(v, 5) if isinstance(v, float) else v for k, v in e.items()}
                       for e in trainer.state.log_history if "eval_loss" in e]
            try:
                accuracy = validation_accuracy(model, tok, eval_rows, system, intents)
            except Exception as exc:  # noqa: BLE001  (selection then falls back to the validation loss)
                print(f"Validation accuracy not available ({type(exc).__name__}: {exc})", flush=True)
                accuracy = None
            return {"eval_loss": min((h["eval_loss"] for h in history), default=None),
                    "val_accuracy": accuracy, "history": history}
        finally:
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
            shutil.rmtree(args.out / "checkpoints", ignore_errors=True)

    def train_with_fallback(rank: int, lr: float, seed: int, target: Path) -> dict:
        quant = args.quant if args.quant != "auto" else ("4bit" if "7b" in args.base.lower() else "none")
        batch, accum = args.batch, args.grad_accum
        attempts: list[dict] = []
        while True:
            try:
                metrics = train_once(quant, batch, accum, rank, lr, seed, target)
                attempts.append({"weights": quant, "batch": batch, "grad_accum": accum, "result": "ok"})
                return {**metrics, "attempts": attempts, "weights": quant, "batch": batch, "grad_accum": accum}
            except Exception as exc:  # noqa: BLE001  (classified below, re-raised if no fallback is left)
                kind = failure_kind(exc)
                attempts.append({"weights": quant, "batch": batch, "grad_accum": accum, "result": kind,
                                 "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                nxt = fallback(kind, quant, batch, accum, fp16_fits="7b" not in args.base.lower())
                if nxt is None or len(attempts) >= 3:
                    print(json.dumps(attempts, indent=1), file=sys.stderr)
                    raise
                print(f"Training failed ({kind}: {attempts[-1]['error'][:200]}); retrying with {nxt}", flush=True)
                quant, batch, accum = nxt

    # Grid first; then, while the budget allows, refine around the best config so
    # that no paid GPU time is left unused (see refine_configs).
    queue = [(r, x, 0, "grid") for r, x in sweep_configs(args.sweep, args.rank, args.lr)]
    tried: set[tuple[int, float, int]] = set()
    started = time.monotonic()
    results: list[dict] = []
    while True:
        elapsed = (time.monotonic() - started) / 60
        if results and not time_left_for_next(elapsed, [r["minutes"] for r in results], args.time_budget):
            print(f"Time budget: stopping after {len(results)} configs ({elapsed:.0f} min)", flush=True)
            break
        if not queue:
            if not results or args.time_budget <= 0:
                break
            b0 = results[pick_best(results)]
            queue = [(*c, "refine") for c in refine_configs(b0["rank"], b0["lr"], tried)]
            if not queue:
                break
        rank, lr, seed, phase = queue.pop(0)
        if (rank, lr, seed) in tried:
            continue
        tried.add((rank, lr, seed))
        t0 = time.monotonic()
        target = args.out / "sweep" / f"{len(tried):02d}-r{rank}-lr{lr:g}-s{seed}"
        try:
            metrics = train_with_fallback(rank, lr, seed, target)
        except Exception as exc:  # noqa: BLE001  (one bad config must not lose the others)
            if not results:
                raise
            print(f"Config rank {rank} lr {lr:g} seed {seed} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        results.append({"rank": rank, "lr": lr, "seed": seed, "phase": phase, "dir": target.name,
                        "minutes": round((time.monotonic() - t0) / 60, 1), **metrics})
        # Keep only the best adapter on disk; the others are no longer needed.
        keep = results[pick_best(results)]["dir"]
        for r in results:
            if r["dir"] != keep:
                shutil.rmtree(args.out / "sweep" / r["dir"], ignore_errors=True)
        (args.out / "sweep.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"Result ({phase}): rank {rank} lr {lr:g} seed {seed} -> val accuracy {metrics['val_accuracy']} "
              f"eval loss {metrics['eval_loss']}", flush=True)
    best = results[pick_best(results)]
    print(f"Chosen: rank {best['rank']} lr {best['lr']:g} seed {best['seed']}", flush=True)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    shutil.copytree(args.out / "sweep" / best["dir"], adapter_dir)
    shutil.rmtree(args.out / "sweep", ignore_errors=True)
    attempts, history = best["attempts"], best["history"]
    quant, batch, accum = best["weights"], best["batch"], best["grad_accum"]

    manifest = {
        "name": args.name, "base": args.base, "base_revision": revision, "adapter_dir": adapter_dir.name,
        "adapter_files": {f.name: sha256(f) for f in sorted(adapter_dir.iterdir()) if f.is_file()},
        "llama_cpp_commit": LLAMA_CPP_COMMIT, "training_git_commit": os.environ.get("VYVY_COMMIT"),
        "train_files": {x.name: sha256(x) for x in paths},
        "train_sentences": len(rows), "prompt": prompt_name, "base_weights": quant,
        "attempts": attempts, "validation": history, "validation_sentences": len(eval_rows),
        "prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(), "epochs": args.epochs, "lr": best["lr"],
        "rank": best["rank"], "alpha": 2 * best["rank"], "batch": batch, "grad_accum": accum,
        "chosen_lr": best["lr"], "val_accuracy": best["val_accuracy"],
        "chosen_seed": best["seed"],
        "sweep": [{k: r[k] for k in ("rank", "lr", "seed", "phase", "minutes", "val_accuracy", "eval_loss")}
                  for r in results],
        "time_budget_min": args.time_budget,
        "target_confidence": TARGET_CONFIDENCE, "holdout_lock": "passed",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    if not args.no_gguf:
        # Locally: convert right away. On Kaggle the GitHub runner converts (no GPU time, retryable).
        import convert_adapter
        convert_adapter.convert(args.out, args.name, args.cache or args.out / ".cache")
    print((args.out / "manifest.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
