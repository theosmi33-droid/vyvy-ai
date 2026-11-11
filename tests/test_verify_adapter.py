"""verify_adapter: the four steps an application runs before it loads an adapter.

    python -m pytest tests/test_verify_adapter.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import release_manifest  # noqa: E402
import verify_adapter as va  # noqa: E402

REAL = json.loads((ROOT / "tests" / "fixtures" / "adapter_manifest_d84f55739.json").read_text())


def files(tmp: Path, accepted: bool = True) -> tuple[Path, Path, Path]:
    adapter = tmp / "vyvy-1.5b-d0000000a.gguf"
    adapter.write_bytes(b"adapter bytes")
    base = tmp / "qwen2.5-1.5b-instruct-q8_0.gguf"
    base.write_bytes(b"base bytes")
    m = json.loads(json.dumps(REAL))
    m.update(adapter=adapter.name, adapter_sha256=hashlib.sha256(b"adapter bytes").hexdigest(),
             status="accepted" if accepted else "rejected")
    m["measured_with"]["files_sha256"] = {base.name: hashlib.sha256(b"base bytes").hexdigest()}
    manifest = tmp / "adapter_manifest.json"
    manifest.write_text(json.dumps(m))
    return adapter, manifest, base


def run(tmp: Path, accepted=True, title=None, attest=None, **kw) -> dict:
    a, m, b = files(tmp, accepted)
    return va.verify(a, m, b, title=title or (lambda tag: tag), attest=attest or (lambda p, r, bundle: (True, "")), **kw)


def failed(result: dict) -> list[str]:
    return [c["step"] for c in result["checks"] if not c["ok"]]


def test_the_real_manifest_matches_the_schema() -> None:
    assert va.schema_problems(REAL) == []
    assert REAL["status"] == "rejected"


def test_the_schema_covers_what_bench_lora_writes() -> None:
    schema = json.loads(va.SCHEMA_FILE.read_text())
    models = {"gguf": {"q": {"repo": "r", "revision": "a" * 40, "files": {"q.gguf": "b" * 64}}}}
    built = release_manifest.build(
        {"adapter": "vyvy-1.5b-d0000000a.gguf", "adapter_sha256": "c" * 64, "base": "B", "base_revision": "d" * 40,
         "train_files": {"t.jsonl": "e" * 64}, "prompt": "compact", "prompt_sha256": "f" * 64},
        {"accepted": True, "n": 2, "b_fixed": 1, "c_broken": 0, "p_value": 0.5, "base_correct": 1,
         "adapter_correct": 2, "reason": "r"}, "q", "1", models)
    assert set(schema["required"]) <= set(built) and va.schema_problems(built) == []


def test_all_four_steps_pass(tmp_path: Path) -> None:
    r = run(tmp_path)
    assert r["ok"] and failed(r) == []


def test_each_step_can_refuse(tmp_path: Path) -> None:
    assert failed(run(tmp_path, accepted=False)) == ["accepted"]
    assert failed(run(tmp_path, title=lambda tag: f"{tag} (candidate)")) == ["accepted"]
    sig = run(tmp_path, attest=lambda p, r, b: (p.suffix == ".json", "no attestation"))
    assert failed(sig) == ["attestation vyvy-1.5b-d0000000a.gguf"]
    a, m, b = files(tmp_path)
    a.write_bytes(b"other bytes")
    assert "checksum" in failed(va.verify(a, m, b, title=lambda t: t, attest=lambda p, r, x: (True, "")))
    b.write_bytes(b"other base")
    assert "base model" in failed(va.verify(a, m, b, title=lambda t: t, attest=lambda p, r, x: (True, "")))


def test_offline_needs_bundles(tmp_path: Path) -> None:
    r = run(tmp_path, offline=True)
    assert failed(r) == ["attestation vyvy-1.5b-d0000000a.gguf", "attestation adapter_manifest.json"]
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    a, m, _ = files(tmp_path)
    for path in (a, m):
        (bundles / f"sha256:{va.sha256(path)}.jsonl").write_text("{}")
    seen = []
    r = run(tmp_path, offline=True, bundle_dir=bundles, attest=lambda p, rep, b: (seen.append(b) or True, ""))
    assert r["ok"] and all(seen)


def test_no_network_means_not_accepted(tmp_path: Path) -> None:
    def down(tag: str) -> str:
        raise OSError("offline")
    assert failed(run(tmp_path, title=down)) == ["accepted"]
