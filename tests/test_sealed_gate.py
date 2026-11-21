"""No GPU run before today's training data passed the Sealed check.

    python -m pytest tests/test_sealed_gate.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import sealed_gate  # noqa: E402


def done(stdout: str = "", code: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, stdout, "")


def test_the_data_state_follows_the_training_files(tmp_path):
    (tmp_path / "a.jsonl").write_text("x\n")
    first = sealed_gate.data_hash(tmp_path)
    assert first == sealed_gate.data_hash(tmp_path) and len(first) == 16
    (tmp_path / "b.jsonl").write_text("y\n")
    assert sealed_gate.data_hash(tmp_path) != first


def test_only_create_and_inject_sets_are_checked():
    assert set(sealed_gate.versions()) >= {"create-v1", "inject-v1"} and "v1" not in sealed_gate.versions()


def test_verdict_names_the_training_line_never_a_text():
    ok, _ = sealed_gate.verdict({"fail": 0.97, "sets": {"create-v1": {"failing": 0, "hits": []}}})
    assert ok
    held, why = sealed_gate.verdict({"fail": 0.97, "sets": {"create-v1": {"failing": 1, "hits": [
        {"id": "create1-0001", "score": 0.981, "kind": "train", "where": "vyvy_x.jsonl:12"}]}}})
    assert not held and "vyvy_x.jsonl:12" in why
    assert not sealed_gate.verdict({"error": {"kind": "NoResult", "line": "no result"}})[0]


def test_status_reads_the_artifact_or_waits(tmp_path, monkeypatch):
    passed = {"fail": 0.97, "sets": {"create-v1": {"failing": 0, "hits": []}}}

    def with_artifact(*args):
        if args[0] == "api":
            return done("4242")
        if args[:2] == ("run", "download"):
            target = Path(args[args.index("--dir") + 1])
            (target / "semantic_result.json").write_text(json.dumps(passed))
            return done()
        raise AssertionError(args)

    assert sealed_gate.status("o/r", "abc", run=with_artifact)[0] == "ok"

    def running(*args):
        return done("" if args[0] == "api" else "1")

    assert sealed_gate.status("o/r", "abc", run=running)[0] == "running"

    def nothing(*args):
        return done("" if args[0] == "api" else "0")

    assert sealed_gate.status("o/r", "abc", run=nothing)[0] == "missing"


def test_every_training_entry_passes_the_gate():
    for name in ("train.yml", "train-colab.yml"):
        text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        plan = text[text.index("Plan ("):text.index("Holdout lock")]
        assert "tools/sealed_gate.py" in plan and "--start" in plan, name
    assert "sealed_ready(repo)" in (ROOT / "tools" / "collect.py").read_text(encoding="utf-8")
    check = (ROOT / ".github" / "workflows" / "sealed-check.yml").read_text(encoding="utf-8")
    assert "sealed-check-${{ env.DATA }}" in check and "if: always()" in check


def test_the_check_hands_back_to_collect():
    # 21.11.2026: after the check the chain had ended; training waited for the next heartbeat.
    text = (ROOT / ".github" / "workflows" / "sealed-check.yml").read_text(encoding="utf-8")
    assert "gh workflow run collect.yml" in text and "actions: write" in text
    assert text.index("Start Collect again") > text.index("upload-artifact")
