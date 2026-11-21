"""catalog.json: one checked list, generated from its sources, honest about what is measured.

    python -m pytest tests/test_catalog.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_catalog  # noqa: E402
import verify_adapter  # noqa: E402

CATALOG = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))
SCHEMA = json.loads((ROOT / "schemas" / "catalog.schema.json").read_text(encoding="utf-8"))
MODELS = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))


def test_catalog_matches_its_sources():
    assert build_catalog.main(["--check"]) == 0


def test_schema():
    assert verify_adapter.schema_problems(CATALOG, SCHEMA) == []
    for e in CATALOG["entries"]:
        assert verify_adapter.schema_problems(e, SCHEMA["$defs"]["entry"]) == [], e["id"]
        if e["adapter"]:
            assert verify_adapter.schema_problems(e["adapter"], SCHEMA["$defs"]["adapter"]) == [], e["id"]


def test_only_the_one_rule_makes_an_entry_selectable():
    for e in CATALOG["entries"]:
        assert e["selectable"] == (e["state"] == "selectable"), e["id"]
        if e["state"] in ("planned", "accepted_unsigned", "candidate", "rejected"):
            assert not e["selectable"], e["id"]
    assert verify_adapter.release_state("t", "t", "accepted", True) == "selectable"
    assert verify_adapter.release_state("t", "t", None, None) == "accepted_unsigned"
    assert verify_adapter.release_state("t", "t", "accepted", False) == "accepted_unsigned"
    assert verify_adapter.release_state("t", "t (rejected)", "accepted", True) == "rejected"
    assert verify_adapter.release_state("t", "t (candidate)", None, None) == "candidate"


def test_planned_has_no_files_no_adapter_no_measurement():
    planned = [e for e in CATALOG["entries"] if e["state"] == "planned"]
    assert {e["base"]["model"] for e in planned} == {"Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct"}
    for e in planned:
        assert e["base"]["files"] is None and e["adapter"] is None and e["measured"] == {"status": "not measured"}
    assert not any("27b" in e["id"].lower() or "27B" in e["class"] for e in CATALOG["entries"])


def test_every_file_is_in_models_json_with_the_same_checksum():
    known = {name: digest for g in MODELS["gguf"].values() for name, digest in g["files"].items()}
    for e in CATALOG["entries"]:
        for name, f in (e["base"]["files"] or {}).items():
            assert known.get(name) == f["sha256"], (e["id"], name)
            assert isinstance(f["size"], int) and f["size"] > 0, (e["id"], name)


def test_no_measurement_without_a_run():
    for e in CATALOG["entries"]:
        m = e["measured"]
        if m["status"] == "measured":
            assert m.get("run", "").startswith("https://github.com/") and m.get("machine"), e["id"]
        else:
            assert all(m.get(k) is None for k in ("memory_mb", "load_s", "decision_ms_median", "decision_ms_p95"))


def test_origin_is_kept_apart():
    for e in CATALOG["entries"]:
        assert e["base"]["origin"] == "third-party work, used unchanged" and e["base"]["licence"] == "Apache-2.0"
        if e["adapter"]:
            assert e["adapter"]["origin"] == "Vyvy AI"
            expected = "Apache-2.0" if e["adapter"]["released"] <= build_catalog.LICENCE_CHANGE else \
                "PolyForm-Noncommercial-1.0.0"
            assert e["adapter"]["licence"] == expected, e["id"]


def test_every_adapter_names_the_contract_it_was_measured_on():
    for e in CATALOG["entries"]:
        if e["adapter"]:
            assert e["contract"]["version"] == CATALOG["contract"]["version"] == 1
            assert e["contract"]["prompt_sha256"] and e["contract"]["grammar_sha256"] == CATALOG["contract"]["grammar_sha256"]


def test_selectable_needs_both_files_verified_as_signed_by_bench_lora(monkeypatch):
    assets = {"vyvy-1.5b-dX.gguf": {}, "adapter_manifest.json": {}}
    fetch = lambda url: b"x"  # noqa: E731

    def signer(results):
        calls = iter(results)
        return lambda path, repo, bundle: next(calls)

    # Attestation exists, but not by bench-lora on main (wrong signer): not selectable.
    ok, how = build_catalog.attestation("vyvy-1.5b-dX", assets, signer([(False, "signer mismatch")] * 2), fetch)
    assert ok is False and verify_adapter.release_state("t", "t", "accepted", ok) == "accepted_unsigned"
    # Only the GGUF signed, the manifest not: not selectable.
    ok, _ = build_catalog.attestation("vyvy-1.5b-dX", assets, signer([(True, "ok"), (False, "no")]), fetch)
    assert ok is False
    # Both signed by bench-lora on main: selectable, and the record says how it was checked.
    ok, how = build_catalog.attestation("vyvy-1.5b-dX", assets, signer([(True, "ok"), (True, "ok")]), fetch)
    assert ok is True and "--signer-workflow" in how["verified_by"] and "refs/heads/main" in how["verified_by"]
    assert verify_adapter.release_state("t", "t", "accepted", ok) == "selectable"
    # The check cannot run: no CLI, no download, no manifest - never selectable.
    for case in (build_catalog.attestation("vyvy-1.5b-dX", assets, signer([(False, "GitHub CLI (gh) not installed")] * 2), fetch),
                 build_catalog.attestation("vyvy-1.5b-dX", assets, signer([]), lambda url: (_ for _ in ()).throw(OSError())),
                 build_catalog.attestation("vyvy-1.5b-dX", {"vyvy-1.5b-dX.gguf": {}}, signer([]), fetch)):
        assert case[0] is None and case[1]["verified_by"] == "not verified"
        assert verify_adapter.release_state("t", "t", "accepted", case[0]) == "accepted_unsigned"


def test_every_entry_says_how_its_signature_was_checked():
    for e in CATALOG["entries"]:
        if e["adapter"]:
            assert e["attestation"]["verified_by"], e["id"]
            if e["state"] == "selectable":
                assert e["attestation"]["verified_by"] != "not verified", e["id"]
