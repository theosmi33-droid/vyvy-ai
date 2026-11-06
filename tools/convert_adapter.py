#!/usr/bin/env python3
"""Convert a trained PEFT LoRA adapter to GGUF on a CPU machine.

Reads <dir>/manifest.json and <dir>/adapter/ as written by training/train_lora.py,
checks every adapter file against the manifest, runs convert_lora_to_gguf.py from
the llama.cpp commit that llama-cpp-python 0.3.16 ships, and adds the GGUF file
name and checksum to the manifest. Needs no GPU, so a failed conversion never
costs training time.

    pip install -r training/convert-requirements.txt
    python tools/convert_adapter.py out
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

LLAMA_CPP_COMMIT = "4227c9be4268ac844921b90f31595f81236bd317"  # vendored by llama-cpp-python 0.3.16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def llama_cpp_dir(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"llama.cpp-{LLAMA_CPP_COMMIT}"
    if not target.is_dir():
        url = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.tar.gz"
        with urllib.request.urlopen(url) as resp:
            tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz").extractall(cache, filter="data")
    return target


def check_files(folder: Path, manifest: dict) -> Path:
    adapter = folder / manifest.get("adapter_dir", "adapter")
    expected = manifest.get("adapter_files") or {}
    if not expected:
        raise SystemExit("manifest lists no adapter files")
    for name, digest in expected.items():
        f = adapter / name
        if not f.is_file():
            raise SystemExit(f"adapter file missing: {f}")
        if sha256(f) != digest:
            raise SystemExit(f"adapter file does not match the manifest: {f}")
    return adapter


def convert(folder: Path, name: str | None = None, cache: Path | None = None) -> Path:
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = name or manifest["name"]
    adapter = check_files(folder, manifest)
    gguf = folder / f"{name}.gguf"
    cache = cache or folder / ".cache"
    script = llama_cpp_dir(cache) / "convert_lora_to_gguf.py"
    base_args = ["--base-model-id", manifest["base"]]
    if manifest.get("base_revision"):
        # Only config and tokenizer files of the exact base commit the adapter was trained on.
        from huggingface_hub import snapshot_download

        base_dir = snapshot_download(manifest["base"], revision=manifest["base_revision"],
                                     allow_patterns=["*.json", "*.txt", "*.model"],
                                     local_dir=cache / "base-config")
        base_args = ["--base", str(base_dir)]
    subprocess.run([sys.executable, str(script), str(adapter), "--outfile", str(gguf),
                    "--outtype", "f16", *base_args], check=True)
    manifest.update({"adapter": gguf.name, "adapter_sha256": sha256(gguf), "llama_cpp_commit": LLAMA_CPP_COMMIT})
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return gguf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert a trained LoRA adapter to GGUF")
    p.add_argument("folder", type=Path, help="folder with manifest.json and adapter/")
    p.add_argument("--name", help="GGUF file name without extension (default: manifest name)")
    p.add_argument("--cache", type=Path, help="where llama.cpp is unpacked (default: <folder>/.cache)")
    args = p.parse_args(argv)
    gguf = convert(args.folder, args.name, args.cache)
    print(f"{gguf} ({gguf.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
