#!/usr/bin/env python3
"""Download a benchmark model exactly as pinned in models.json and verify it.

Every GGUF file comes from a fixed Hugging Face commit and must match its
SHA-256 from models.json; files already present are reused only if they match.
If Hugging Face cannot deliver a file, it is taken from the private Kaggle
mirror (dataset <KAGGLE_USERNAME>/vyvy-model-mirror, filled by mirror.yml) when
Kaggle credentials are set, and checked against the same SHA-256. If the
upstream repository changes or a file is replaced, the run stops instead of
measuring a different model. Prints the path to load (the first shard).

    python tools/fetch_model.py qwen2.5-7b-instruct-q4_k_m --dir models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


MIRROR = "vyvy-model-mirror"


def from_mirror(filename: str, folder: Path) -> None:
    user = os.environ.get("KAGGLE_USERNAME")
    if not user or not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("no Kaggle credentials for the mirror")
    if not shutil.which("kaggle"):
        # kaggle 2.2.5 imports six without declaring it as a dependency
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle==2.2.5", "six==1.17.0"], check=True)
    subprocess.run(["kaggle", "datasets", "download", f"{user}/{MIRROR}", "-f", filename, "-p", str(folder),
                    "--unzip", "--force"], check=True)
    zipped = folder / f"{filename}.zip"
    if zipped.exists():
        zipped.unlink()


def fetch(name: str, folder: Path, download=None, mirror=from_mirror) -> Path:
    spec = json.loads(MODELS.read_text(encoding="utf-8"))["gguf"].get(name)
    if spec is None:
        raise SystemExit(f"{name} is not pinned in models.json")
    if download is None:
        from huggingface_hub import hf_hub_download

        def download(repo: str, filename: str, revision: str) -> None:
            hf_hub_download(repo, filename, revision=revision, local_dir=folder)

    folder.mkdir(parents=True, exist_ok=True)
    for filename, digest in sorted(spec["files"].items()):
        path = folder / filename
        if not (path.is_file() and sha256(path) == digest):
            try:
                download(spec["repo"], filename, spec["revision"])
            except Exception as exc:  # noqa: BLE001  (any failure: try the mirror, then verify)
                print(f"Hugging Face failed for {filename} ({type(exc).__name__}); using the Kaggle mirror",
                      file=sys.stderr)
                mirror(filename, folder)
        actual = sha256(path)
        if actual != digest:
            raise SystemExit(f"{filename}: checksum {actual} does not match the pinned {digest}")
        print(f"ok {filename} {digest[:12]} ({spec['repo']}@{spec['revision'][:7]})", file=sys.stderr)
    return folder / sorted(spec["files"])[0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download a pinned benchmark model")
    p.add_argument("name", help="model key in models.json, e.g. qwen2.5-1.5b-instruct-q8_0")
    p.add_argument("--dir", type=Path, default=Path("models"))
    args = p.parse_args(argv)
    path = fetch(args.name, args.dir)
    print(path)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"path={path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
