"""bench-client: the classes of 23.11.2026 as cells, the contract check, and the numbers of one cell."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402
import client_cell  # noqa: E402

WORKFLOW = (ROOT / ".github" / "workflows" / "bench-client.yml").read_text(encoding="utf-8")


def test_linux_cells_cover_both_classes_with_hard_limits():
    linux = WORKFLOW[WORKFLOW.index("  linux:"):WORKFLOW.index("  desktop:")]
    cells = set(re.findall(r"- \{model: ([\w.-]+), cpus: (\d), memory: ([\d.]+g), size: \w+, limit_gb: [\d.]+\}", linux))
    small = {(f"qwen2.5-1.5b-instruct-{q}", c, m) for q in ("q4_k_m", "q8_0") for c in ("2", "4") for m in ("5.5g", "3.5g")}
    assert cells == small | {("qwen2.5-7b-instruct-q4_k_m", "4", "10g")}
    assert '--cpus "$CPUS" --memory "$MEMORY" --memory-swap "$MEMORY"' in linux


def test_desktop_measures_without_limit_on_macos_and_windows():
    desktop = WORKFLOW[WORKFLOW.index("  desktop:"):]
    assert "os: [macos-latest, windows-latest]" in desktop
    assert "--memory" not in desktop and "--limited" not in desktop


def test_only_by_hand():
    on = WORKFLOW[WORKFLOW.index("\non:"):WORKFLOW.index("\npermissions:")]
    assert "workflow_dispatch:" in on and "push:" not in on and "schedule:" not in on and "workflow_run" not in on


def test_adapter_of_another_contract_is_refused():
    grammar = hashlib.sha256(bench.DEFAULT_GRAMMAR.read_bytes()).hexdigest()
    assert client_cell.check_adapter({"contract_version": 2, "grammar_sha256": grammar}) is None
    assert client_cell.check_adapter({"prompt": "compact"})  # contract 1
    assert client_cell.check_adapter({"contract_version": 2, "grammar_sha256": "0" * 64})


def _result(ms: list[int], correct: int, peak_mb: float | None = None) -> dict:
    return {"n": len(ms), "intent_correct": correct, "contract": 2, "cpu": "x",
            "items": [{"ms": m, "timeout": m > 4000} for m in ms],
            "resources": {"peak_memory_mb": peak_mb} if peak_mb is not None else None}


def test_limited_cell_within_means_finished_and_not_killed():
    c = client_cell.cell(_result([100, 200, 5000, 300], 3), platform="Linux container", model="m", cpus=2,
                         limit_gb=3.5, limited=True, state={"exit": 0, "oom": False}, time_peak_gb=3.1)
    assert c["within_class"] is True and c["median_ms"] == 250 and c["over_4s"] == 0.25 and c["peak_gb"] == 3.1
    killed = client_cell.cell(None, platform="Linux container", model="m", cpus=2, limit_gb=3.5, limited=True,
                              state={"exit": 137, "oom": True}, time_peak_gb=None)
    assert killed["within_class"] is False and "killed for lack of memory" in client_cell.row(killed)


def test_unlimited_cell_compares_peak_with_the_class_limit():
    ok = client_cell.cell(_result([100] * 10, 9, peak_mb=3000), platform="Windows", model="m", cpus=4,
                          limit_gb=5.5, limited=False, state={"exit": 0, "oom": False}, time_peak_gb=None)
    over = client_cell.cell(_result([100] * 10, 9, peak_mb=6000), platform="macOS", model="m", cpus=4,
                            limit_gb=5.5, limited=False, state={"exit": 0, "oom": False}, time_peak_gb=None)
    unknown = client_cell.cell(_result([100] * 10, 9), platform="macOS", model="m", cpus=4, limit_gb=5.5,
                               limited=False, state={"exit": 0, "oom": False}, time_peak_gb=None)
    assert ok["within_class"] is True and over["within_class"] is False and unknown["within_class"] is None


def test_summary_writes_the_cell(tmp_path):
    (tmp_path / "client.json").write_text(json.dumps(_result([100, 300], 2)), encoding="utf-8")
    (tmp_path / "exit.txt").write_text("exit=0\noom=false\n", encoding="utf-8")
    (tmp_path / "time.txt").write_text("\tMaximum resident set size (kbytes): 2097152\n", encoding="utf-8")
    out = tmp_path / "cell.json"
    assert client_cell.main(["summary", "--platform", "Linux container", "--model", "m", "--cpus", "2",
                             "--limit-gb", "3.5", "--limited", "--result", str(tmp_path / "client.json"),
                             "--time", str(tmp_path / "time.txt"), "--exit", str(tmp_path / "exit.txt"),
                             "--out", str(out)]) == 0
    c = json.loads(out.read_text(encoding="utf-8"))
    assert c["peak_gb"] == 2.0 and c["within_class"] is True and c["accuracy"] == 1.0
