#!/usr/bin/env python3
"""Check an adapter before an application loads it: the four steps of the README.

1. accepted:     adapter_manifest.json says "accepted" and, online, the release title
                 is the bare tag (not "(candidate)" or "(rejected)")
2. checksum:     the GGUF file's SHA-256 equals adapter_sha256 and its name matches
3. attestation:  bench-lora on main of this repository signed the GGUF file and the
                 manifest (`gh attestation verify`, online, or offline with --bundle)
4. base model:   the base model file is the one the adapter was measured on
                 (measured_with.files_sha256)

Standard library only; step 3 calls the GitHub CLI. Prints one JSON object and
exits 0 only if every step passed. The manifest format is schemas/adapter_manifest.schema.json.

    python tools/verify_adapter.py --adapter vyvy-1.5b-dXXXXXXXX.gguf --manifest adapter_manifest.json \\
        --base qwen2.5-1.5b-instruct-q8_0.gguf [--offline --bundle-dir bundles/]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

REPO = "theosmi33-droid/vyvy-ai"
SIGNER = ".github/workflows/bench-lora.yml"
SCHEMA_FILE = Path(__file__).resolve().parent.parent / "schemas" / "adapter_manifest.schema.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def schema_problems(manifest: dict, schema: dict | None = None) -> list[str]:
    """The required fields and patterns of the schema (a subset of JSON Schema, no dependency)."""
    schema = schema or json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))

    def walk(value, node: dict, where: str) -> list[str]:
        out = []
        if "const" in node and value != node["const"]:
            out.append(f"{where}: must be {node['const']!r}")
        if "enum" in node and value not in node["enum"]:
            out.append(f"{where}: must be one of {node['enum']}")
        if "pattern" in node and isinstance(value, str) and not re.search(node["pattern"], value):
            out.append(f"{where}: does not match {node['pattern']}")
        if isinstance(value, dict):
            out += [f"{where}.{k}: missing" for k in node.get("required", []) if k not in value]
            for k, sub in node.get("properties", {}).items():
                if k in value:
                    out += walk(value[k], sub, f"{where}.{k}")
            extra = node.get("additionalProperties")
            if isinstance(extra, dict):
                for k, v in value.items():
                    if k not in node.get("properties", {}):
                        out += walk(v, extra, f"{where}.{k}")
        return out

    return walk(manifest, schema, "manifest")


def release_title(tag: str, repo: str = REPO) -> str:
    """Title of the public release (no token needed)."""
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
                                 headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["name"]


def gh_attestation(path: Path, repo: str, bundle: Path | None) -> tuple[bool, str]:
    cmd = ["gh", "attestation", "verify", str(path), "--repo", repo,
           "--signer-workflow", f"{repo}/{SIGNER}", "--source-ref", "refs/heads/main"]
    if bundle:
        cmd += ["--bundle", str(bundle)]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return False, "GitHub CLI (gh) not installed"
    lines = (done.stderr or done.stdout).strip().splitlines()
    return done.returncode == 0, lines[-1] if lines else ""


def verify(adapter: Path, manifest_path: Path, base: Path, *, repo: str = REPO, offline: bool = False,
           bundle_dir: Path | None = None,
           title: Callable[[str], str] = release_title,
           attest: Callable[[Path, str, Path | None], tuple[bool, str]] = gh_attestation) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"step": name, "ok": bool(ok), "detail": detail})

    problems = schema_problems(manifest)
    check("schema", not problems, "; ".join(problems[:5]) or "manifest matches schema 1")
    tag = adapter.name.removesuffix(".gguf")
    status = manifest.get("status")
    if offline:
        check("accepted", status == "accepted", f"manifest status {status!r} (release title not checked offline)")
    else:
        try:
            name = title(tag)
            check("accepted", status == "accepted" and name == tag, f"manifest status {status!r}, release title {name!r}")
        except Exception as exc:  # noqa: BLE001  (no network or no such release: not accepted)
            check("accepted", False, f"release not readable: {type(exc).__name__}")
    digest = sha256(adapter)
    check("checksum", manifest.get("adapter") == adapter.name and digest == manifest.get("adapter_sha256"),
          f"file {adapter.name} sha256 {digest[:12]}…, manifest {str(manifest.get('adapter_sha256'))[:12]}…")
    for path in (adapter, manifest_path):
        bundle = None
        if bundle_dir:
            found = sorted(bundle_dir.glob(f"*{sha256(path)}*"))
            bundle = found[0] if found else None
        ok, detail = (False, "offline without a bundle for this file") if offline and not bundle else attest(path, repo, bundle)
        check(f"attestation {path.name}", ok, detail or ("signed by bench-lora on main" if ok else "not verified"))
    files = (manifest.get("measured_with") or {}).get("files_sha256") or {}
    base_digest = sha256(base)
    check("base model", files.get(base.name) == base_digest,
          f"{base.name} sha256 {base_digest[:12]}…, measured with {sorted(files)}")
    return {"adapter": adapter.name, "ok": all(c["ok"] for c in checks), "checks": checks}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check an adapter before an application loads it")
    p.add_argument("--adapter", type=Path, required=True, help="the adapter GGUF file")
    p.add_argument("--manifest", type=Path, required=True, help="adapter_manifest.json of its release")
    p.add_argument("--base", type=Path, required=True, help="the base model GGUF file")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--offline", action="store_true", help="no network: manifest status and bundles only")
    p.add_argument("--bundle-dir", type=Path, help="bundles from `gh attestation download` (named by digest)")
    args = p.parse_args(argv)
    result = verify(args.adapter, args.manifest, args.base, repo=args.repo, offline=args.offline,
                    bundle_dir=args.bundle_dir)
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
