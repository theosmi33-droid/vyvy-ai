#!/usr/bin/env python3
"""Write the license files that must travel with a published adapter.

An adapter is a modification of its base model, so every release carries the
base model's license text (fetched from the pinned commit in models.json) and a
NOTICE that names the base model, its license and the modification.

    python tools/license_files.py out
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOTICE = """\
{name}

This file is a LoRA adapter for {base} (revision {revision}).
{base} is licensed under the Apache License, Version 2.0; its license text is
included as LICENSE-base-model.txt. Copyright for the base model belongs to its
authors (Qwen team, Alibaba Cloud).

Modification: the adapter was trained by Vyvy AI with low-rank adaptation (LoRA)
on synthetic German office questions. It contains no weights of the base model;
it is applied on top of the unchanged base model at run time.

This adapter is distributed under the terms of the Apache License, Version 2.0,
as required for modifications of the base model. The license of the Vyvy AI
repository itself does not apply to this file.
"""


def write(folder: Path, download=None) -> list[Path]:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    base = manifest["base"]
    revision = manifest.get("base_revision") or json.loads((ROOT / "models.json").read_text())["base"][base]
    if download is None:
        from huggingface_hub import hf_hub_download

        def download(repo: str, filename: str, rev: str) -> str:
            return hf_hub_download(repo, filename, revision=rev)

    lic = folder / "LICENSE-base-model.txt"
    shutil.copyfile(download(base, "LICENSE", revision), lic)
    if "Apache License" not in lic.read_text(encoding="utf-8", errors="replace"):
        raise SystemExit(f"{base}@{revision}: LICENSE is not the Apache License; check before publishing")
    notice = folder / "NOTICE.txt"
    notice.write_text(NOTICE.format(name=manifest.get("adapter") or manifest["name"], base=base,
                                    revision=revision), encoding="utf-8")
    return [lic, notice]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write LICENSE-base-model.txt and NOTICE.txt for an adapter")
    p.add_argument("folder", type=Path, help="folder with manifest.json")
    args = p.parse_args(argv)
    for f in write(args.folder):
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
