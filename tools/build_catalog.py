#!/usr/bin/env python3
"""Build catalog.json: the one checked list of Vyvy AI models an application may offer.

An application reads catalog.json without this repository. It loads, unloads and
switches models itself; here is only the list and its rules:

- Sources, never edited by hand in catalog.json: models.json (base model files and
  their SHA-256), data/catalog_releases.json (a snapshot of the releases and the
  file sizes, written by --refresh from the public GitHub and Hugging Face APIs),
  grammars/slots.gbnf and the prompts of runner/bench.py (the output contract).
- state per entry, exactly one of: selectable, accepted_unsigned, candidate, rejected,
  planned. What is selectable is decided once, by tools/verify_adapter.release_state;
  an application still runs tools/verify_adapter.py (checksum, signature, base model)
  before it loads a file.
- Origin is kept apart: the base model is a third-party work, unchanged, under its own
  licence; the adapter is Vyvy's, under the licence of its release (Apache-2.0 up to
  vyvy-1.5b-da8577af3, PolyForm Noncommercial 1.0.0 after it, see README).
- Measurements only where measured, each with the machine and the run; otherwise null
  with "not measured". The class names 4GB and 8GB are planning limits, not measurements.
- 14B and 32B are listed as planned: no files, no adapter, no measurement, no schedule.

    python tools/build_catalog.py            # write catalog.json
    python tools/build_catalog.py --check    # committed catalog.json matches its sources
    python tools/build_catalog.py --refresh  # in the cloud: rewrite data/catalog_releases.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))
sys.path.insert(0, str(ROOT / "tools"))

import bench  # noqa: E402
import verify_adapter  # noqa: E402

OUT = ROOT / "catalog.json"
SNAPSHOT = ROOT / "data" / "catalog_releases.json"
MODELS = ROOT / "models.json"
REPO = "theosmi33-droid/vyvy-ai"
CONTRACT_VERSION = 1
NOT_MEASURED = "not measured"
LICENCE_CHANGE = "2026-11-11T12:49:56Z"  # vyvy-1.5b-da8577af3, the last adapter under Apache-2.0

CLASSES = [
    {"class": "Vyvy AI 4GB", "base": "Qwen/Qwen2.5-1.5B-Instruct", "gguf": "qwen2.5-1.5b-instruct-q8_0",
     "size": "1.5b"},
    {"class": "Vyvy AI 8GB", "base": "Qwen/Qwen2.5-7B-Instruct", "gguf": "qwen2.5-7b-instruct-q4_k_m", "size": "7b"},
]
PLANNED = [
    {"class": "planned 14B", "base": "Qwen/Qwen2.5-14B-Instruct"},
    {"class": "planned 32B", "base": "Qwen/Qwen2.5-32B-Instruct"},
]
CLASS_NOTE = "planning limit for memory, not a measurement"
BASE_ORIGIN = {"origin": "third-party work, used unchanged", "author": "Qwen team, Alibaba Cloud",
               "licence": "Apache-2.0"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract() -> dict:
    grammar = bench.DEFAULT_GRAMMAR
    prompts = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in sorted(bench.PROMPTS.items())}
    return {"version": CONTRACT_VERSION, "intents": list(bench.grammar_intents(grammar.read_text(encoding="utf-8"))),
            "grammar": "grammars/slots.gbnf", "grammar_sha256": sha256_file(grammar), "prompts_sha256": prompts}


def adapter_licence(created: str) -> str:
    return "Apache-2.0" if created <= LICENCE_CHANGE else "PolyForm-Noncommercial-1.0.0"


def base_block(spec: dict, models: dict, snap: dict) -> dict:
    gguf = models["gguf"][spec["gguf"]]
    files = {name: {"sha256": digest, "size": snap["base_files"].get(name, {}).get("size")}
             for name, digest in gguf["files"].items()}
    return {"model": spec["base"], "revision": models["base"][spec["base"]], "gguf": spec["gguf"],
            "gguf_repo": gguf["repo"], "gguf_revision": gguf["revision"], "files": files, **BASE_ORIGIN}


def measured(r: dict) -> dict:
    if r.get("decision_ms_median") is None or not r.get("bench_lora_run"):
        return {"status": NOT_MEASURED, "memory_mb": None, "load_s": None, "decision_ms_median": None,
                "decision_ms_p95": None}
    return {"status": "measured", "memory_mb": None, "load_s": None,
            "decision_ms_median": r["decision_ms_median"], "decision_ms_p95": None,
            "machine": r.get("cpu"), "run": f"https://github.com/{REPO}/actions/runs/{r['bench_lora_run']}",
            "note": "decision time on the blind holdout, GitHub CPU runner; memory, load time and p95 not measured"}


#: Benchmarks on fixed platforms whose results go into the list, by workflow and artifact.
PLATFORM_RUNS = {"bench-windows.yml": "bench-windows-results", "bench-mac.yml": "bench-mac-results"}
#: Runs per platform the numbers are taken over: one run on a shared runner is no number to promise.
RUNS_PER_PLATFORM = 3
MEMORY_NOTE = "peak during the measurement, whole process (model, adapter, context, runtime and Python)"
#: How the peak is read differs per operating system, so the numbers must not be compared across platforms.
MEMORY_BY_PLATFORM = {
    "Windows": "peak working set, including the memory-mapped pages of the model file",
    "Darwin": "maximum resident set; memory on the graphics side (Metal) is not included",
    "Linux": "maximum resident set",
}
VALUES = ("intent_correct", "decision_ms_median", "decision_ms_p95", "load_s", "first_decision_s", "peak_memory_mb")


def _median(xs: list[float]) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    value = xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2
    return round(value, 2) if isinstance(value, float) else value


def platform_measurements(snap: dict, spec: dict, adapter_file: str | None) -> list[dict]:
    """The platform runs of this class with exactly this adapter (None = base model alone), one
    entry per platform: the median of each value over the newest RUNS_PER_PLATFORM runs, its
    spread (smallest and largest) and every run. Nothing is carried over to other hardware."""
    by_platform: dict[str, list[dict]] = {}
    for m in snap.get("platform_runs", []):
        if m["gguf"] == spec["gguf"] and m.get("adapter") == adapter_file:
            by_platform.setdefault(m["platform"], []).append(m)
    out = []
    for platform, runs in sorted(by_platform.items(), key=lambda x: str(x[0])):
        runs = sorted(runs, key=lambda m: m["measured_at"], reverse=True)[:RUNS_PER_PLATFORM]
        system = str(platform).split()[0]
        out.append({
            "platform": platform, "machines": sorted({m.get("machine") or "" for m in runs}),
            "threads": sorted({m.get("threads") for m in runs if m.get("threads") is not None}),
            "data": runs[0]["data"], "n": runs[0]["n"], "n_runs": len(runs),
            # A single value must not read like an average.
            "basis": (f"median over the newest {RUNS_PER_PLATFORM} runs" if len(runs) >= RUNS_PER_PLATFORM
                      else f"median over fewer than {RUNS_PER_PLATFORM} runs ({len(runs)})"),
            "runs": [m["run"] for m in runs], "measured_at": [m["measured_at"] for m in runs],
            **{k: _median([m.get(k) for m in runs]) for k in VALUES},
            "spread": {k: [min(v), max(v)] if (v := [m[k] for m in runs if m.get(k) is not None]) else None
                       for k in VALUES},
            "peak_memory_note": f"{MEMORY_NOTE}; {MEMORY_BY_PLATFORM.get(system, 'reading not described')} - "
                                "not comparable across platforms"})
    return out


def build(models: dict, snap: dict) -> dict:
    con = contract()
    entries = []
    for spec in CLASSES:
        base = base_block(spec, models, snap)
        for r in [r for r in snap["releases"] if r["tag"].startswith(f"vyvy-{spec['size']}-")]:
            state = verify_adapter.release_state(r["tag"], r["title"], r.get("signed_status"), r.get("attested"))
            entries.append({
                "id": r["tag"], "class": spec["class"], "class_note": CLASS_NOTE, "state": state,
                "selectable": state == "selectable",
                "base": base,
                "adapter": {"tag": r["tag"], "file": r["adapter_file"], "sha256": r["adapter_sha256"],
                            "size": r["adapter_size"], "manifest_sha256": r["manifest_sha256"], "lora_scale": 1.0,
                            "origin": "Vyvy AI", "licence": adapter_licence(r["created"]), "released": r["created"]},
                # The release manifest records the prompt, not the grammar: grammar_sha256 is today's
                # grammar, which an application must use with this adapter.
                "contract": {"version": con["version"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"],
                             "grammar_sha256": con["grammar_sha256"], "grammar_recorded_at_training": False},
                "attestation": r.get("attestation") or {"verified_by": "not verified"},
                "measured": measured(r),
                "measured_on_platforms": platform_measurements(snap, spec, r["adapter_file"])})
    for spec in PLANNED:
        entries.append({"id": spec["class"].replace(" ", "-").lower(), "class": spec["class"],
                        "class_note": "no files, no adapter, no measurement, no schedule", "state": "planned",
                        "selectable": False, "base": {"model": spec["base"], "files": None, **BASE_ORIGIN},
                        "adapter": None, "contract": None, "measured": {"status": NOT_MEASURED}})
    return {"schema": 2, "contract": con, "states": {
        "selectable": ("accepted, and both the GGUF and adapter_manifest.json verified as signed by bench-lora on "
                       "main (gh attestation verify, signer workflow and refs/heads/main); may be offered after "
                       "tools/verify_adapter.py passed on the downloaded files"),
        "accepted_unsigned": "accepted before the signature existed, or not yet verified; never selectable",
        "candidate": "trained, not decided", "rejected": "decided against", "planned": "listed only, never selectable"},
        "snapshot_fetched": snap["fetched"],
        # The base models alone, as they were measured on fixed platforms (no adapter).
        "base_measurements": {spec["class"]: platform_measurements(snap, spec, None) for spec in CLASSES},
        "entries": entries}


def render(catalog: dict) -> str:
    return json.dumps(catalog, indent=1, ensure_ascii=False) + "\n"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "vyvy-catalog"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


VERIFIED_BY = (f"gh attestation verify --signer-workflow {REPO}/{verify_adapter.SIGNER} "
               "--source-ref refs/heads/main (tools/verify_adapter.gh_attestation), for the GGUF and adapter_manifest.json")


def attestation(tag: str, assets: dict, attest=verify_adapter.gh_attestation, fetch=None) -> tuple[bool | None, dict]:
    """True only if bench-lora on main signed both the GGUF and adapter_manifest.json, checked with
    the same function as verify(); False if a check failed; None if it could not run. Only
    True can make an entry selectable."""
    if "adapter_manifest.json" not in assets:
        return None, {"verified_by": "not verified", "why": "no adapter_manifest.json on the release"}
    fetch = fetch or _get
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        try:
            for name in (f"{tag}.gguf", "adapter_manifest.json"):
                path = Path(tmp) / name
                path.write_bytes(fetch(f"https://github.com/{REPO}/releases/download/{tag}/{name}"))
                paths.append(path)
        except Exception as exc:  # noqa: BLE001  (no download: not verified, never selectable)
            return None, {"verified_by": "not verified", "why": f"download failed: {type(exc).__name__}"}
        results = [attest(path, REPO, None) for path in paths]
    details = {"gguf": results[0][1][-200:], "adapter_manifest": results[1][1][-200:]}
    if any("not installed" in d for d in details.values()):
        return None, {"verified_by": "not verified", "why": "GitHub CLI not available", **details}
    return all(ok for ok, _ in results), {"verified_by": VERIFIED_BY, **details}


def platform_runs(gh=None) -> list[dict]:
    """Results of the newest successful runs of each platform benchmark on main (runs in the cloud)."""
    def run_gh(*args: str) -> str:
        return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout

    gh = gh or run_gh
    out = []
    for workflow, artifact in PLATFORM_RUNS.items():
        ids = json.loads(gh("run", "list", "--repo", REPO, "--workflow", workflow, "--branch", "main", "--status",
                            "success", "--limit", str(RUNS_PER_PLATFORM), "--json", "databaseId,createdAt") or "[]")
        for run in ids:
            out += _platform_run(gh, artifact, run["databaseId"], run["createdAt"])
    return out


def _platform_run(gh, artifact: str, run_id: int, at: str) -> list[dict]:
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        gh("run", "download", str(run_id), "--repo", REPO, "--name", artifact, "--dir", tmp)
        for f in sorted(Path(tmp).glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            res = r.get("resources") or {}
            label = r.get("label", "")
            out.append({"platform": res.get("platform"), "machine": r.get("cpu"), "threads": res.get("threads"),
                        "run": f"https://github.com/{REPO}/actions/runs/{run_id}", "measured_at": at,
                        "data": "data/holdout_blind_v1.json",
                        "gguf": label.split("+")[0], "adapter": r.get("adapter"), "n": r.get("n"),
                        "intent_correct": r.get("intent_correct"), "decision_ms_median": r.get("median_ms"),
                        "decision_ms_p95": r.get("p95_ms"), "load_s": res.get("load_s"),
                        "first_decision_s": res.get("first_decision_s"), "peak_memory_mb": res.get("peak_memory_mb")})
    return out


def refresh(models: dict, now: str) -> dict:
    """Snapshot of the public releases and the base file sizes (runs in the cloud)."""
    releases = json.loads(_get(f"https://api.github.com/repos/{REPO}/releases?per_page=100"))
    base_files = {}
    for name, g in models["gguf"].items():
        tree = json.loads(_get(f"https://huggingface.co/api/models/{g['repo']}/tree/{g['revision']}"))
        for e in tree:
            if e.get("path") in g["files"] and e.get("lfs"):
                base_files[e["path"]] = {"size": e["size"], "sha256": e["lfs"]["oid"]}
    out = []
    for x in sorted(releases, key=lambda r: r["created_at"]):
        tag, assets = x["tag_name"], {a["name"]: a for a in x["assets"]}
        dl = f"https://github.com/{REPO}/releases/download/{tag}/"
        m = json.loads(_get(dl + "manifest.json"))
        signed = json.loads(_get(dl + "adapter_manifest.json")) if "adapter_manifest.json" in assets else None
        verdict = json.loads(_get(dl + "verdict.json")) if "verdict.json" in assets else None
        g = assets[f"{tag}.gguf"]
        digest = g["digest"].removeprefix("sha256:")
        if x["name"] == tag:
            attested, how = attestation(tag, assets)
        else:
            attested, how = None, {"verified_by": "not verified", "why": "not checked: the release is not accepted"}
        run = re.findall(r"bench-lora \(run (\d+)\)", x.get("body") or "")
        out.append({"tag": tag, "title": x["name"], "prerelease": x["prerelease"], "created": x["created_at"],
                    "base": m["base"], "prompt": m.get("prompt"), "prompt_sha256": m.get("prompt_sha256"),
                    "adapter_file": g["name"], "adapter_size": g["size"], "adapter_sha256": digest,
                    "manifest_sha256": assets["manifest.json"]["digest"].removeprefix("sha256:"),
                    "adapter_manifest": signed is not None, "attested": attested, "attestation": how,
                    "bench_lora_run": run[-1] if run else None,
                    "decision_ms_median": verdict.get("adapter_median_ms") if verdict else None,
                    "cpu": verdict.get("cpu") if verdict else None,
                    "signed_status": signed.get("status") if signed else None})
    return {"schema": 1, "fetched": now,
            "source": "GitHub releases of this repository and the Hugging Face API at the pinned revisions (public)",
            "base_files": base_files, "releases": out, "platform_runs": platform_runs()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build catalog.json from models.json and the release snapshot")
    p.add_argument("--check", action="store_true", help="only compare with the committed catalog.json")
    p.add_argument("--refresh", action="store_true", help="rewrite the release snapshot from the public APIs")
    args = p.parse_args(argv)
    models = json.loads(MODELS.read_text(encoding="utf-8"))
    if args.refresh:
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
        SNAPSHOT.write_text(json.dumps(refresh(models, now), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    text = render(build(models, json.loads(SNAPSHOT.read_text(encoding="utf-8"))))
    if args.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print(f"catalog.json {'matches' if same else 'differs from'} its sources")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"catalog.json written ({text.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
