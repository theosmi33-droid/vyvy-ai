"""bench.py records load time, time to the first decision, peak memory and p95 - measured, never guessed.

    python -m pytest tests/test_bench_resources.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402


def test_peak_memory_is_a_number_here_or_null():
    peak = bench.peak_memory_mb()
    assert peak is None or (isinstance(peak, float) and peak > 0)


def test_dry_run_has_p95_and_no_resources(tmp_path):
    out = tmp_path / "r.json"
    assert bench.main(["--dry-run", "--data", str(ROOT / "data" / "holdout_blind_v1.json"), "--json", str(out)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(r["p95_ms"], int) and r["p95_ms"] >= r["timing"]["ms"]["median"]
    assert r["resources"] is None  # no model loaded: nothing to measure, nothing claimed


def test_the_windows_benchmark_builds_the_pinned_version():
    text = (ROOT / ".github" / "workflows" / "bench-windows.yml").read_text(encoding="utf-8")
    assert "runs-on: windows-latest" in text and "llama-cpp-python==0.3.16" in text
    assert "--accelerator" not in text and "never carried over" in text
