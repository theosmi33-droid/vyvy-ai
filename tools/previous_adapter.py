#!/usr/bin/env python3
"""Find the default adapter for a base model: its newest accepted release.

An adapter counts only if its release is not a pre-release (accepted by
bench-lora), its manifest names the same base model and it was trained on the
prompt the benchmark uses today. Prints the release tag, or nothing if there is
none; bench-lora then measures against the base model alone.

    python tools/previous_adapter.py --base Qwen/Qwen2.5-1.5B-Instruct --prompt-sha256 <sha> --exclude <tag>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable


def find(releases: list[dict], manifest_of: Callable[[str], dict | None], base: str, prompt_sha: str,
         exclude: str) -> str | None:
    for rel in sorted(releases, key=lambda r: r.get("publishedAt") or "", reverse=True):
        tag = rel["tagName"]
        if tag == exclude or rel.get("isPrerelease") or rel.get("isDraft"):
            continue
        m = manifest_of(tag)
        if m and m.get("base") == base and m.get("prompt_sha256") == prompt_sha:
            return tag
    return None


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Newest accepted adapter for a base model")
    p.add_argument("--repo", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--prompt-sha256", required=True)
    p.add_argument("--exclude", default="")
    args = p.parse_args(argv)

    releases = json.loads(gh("release", "list", "--repo", args.repo, "--limit", "100",
                             "--json", "tagName,isPrerelease,isDraft,publishedAt"))

    def manifest_of(tag: str) -> dict | None:
        try:
            return json.loads(gh("release", "download", tag, "--repo", args.repo, "--pattern", "manifest.json",
                                 "--output", "-"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return None

    tag = find(releases, manifest_of, args.base, args.prompt_sha256, args.exclude)
    print(tag or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
