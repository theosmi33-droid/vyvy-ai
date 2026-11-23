"""What bench-lora tells others after a decision: Catalog, and the holdout rotation notice.

    python -m pytest tests/test_bench_lora_notices.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_remeasure import WORKFLOW, condition, steps  # noqa: E402


def test_a_run_the_pipeline_started_reaches_the_catalog():
    s = steps()
    refresh = s["Refresh the catalog"]
    assert "gh workflow run catalog.yml" in refresh and "--ref main" in refresh
    # Accepted or rejected, signed or not: the catalog shows every decision.
    assert condition(refresh) == "if: ${{ !cancelled() && github.actor == 'github-actions[bot]' }}"
    names = list(s)
    assert names.index("Refresh the catalog") > names.index("actions/attest-build-provenance@4d101475d8b20a2381f78447822ac1eab6504dd8 # v4.2.2")
    assert names.index("Refresh the catalog") < names.index("Fail when rejected")
    permissions = WORKFLOW[WORKFLOW.index("\npermissions:"):WORKFLOW.index("\nconcurrency:")]
    assert re.search(r"^  actions: write", permissions, re.M)
    catalog = (ROOT / ".github" / "workflows" / "catalog.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in catalog


def _rotation_prefix() -> str:
    decide = steps()["Decide"]
    found = re.search(r'prefix="([^"]+)"', decide)
    assert found, "the rotation notice has no prefix"
    assert 'title="${prefix}used for' in decide
    assert '--search "\\"$prefix\\" in:title"' in decide and 'grep -qF "$prefix"' in decide
    return found.group(1)


def test_an_open_notice_for_an_older_holdout_does_not_hide_the_new_one():
    prefix = _rotation_prefix()
    active = re.search(r"DECISION_SET: data/holdout_decision_(c\d+)\.json", WORKFLOW).group(1)
    assert prefix == f"Holdout rotation due (decision set {active} "
    # Issue #8, still open on 22.11.2026, hid the notice for v3; the open notices of contract 1 must not hide c2.
    for older in ("Holdout rotation due (blind v2 used for 8 decisions)",
                  "Holdout rotation due (blind v3 used for 8 decisions)"):
        assert not older.startswith(prefix)
    assert f"{prefix}used for 9 decisions)".startswith(prefix)


def test_the_decision_set_runs_in_parts_baseline_and_candidate_together():
    measure = WORKFLOW[WORKFLOW.index("  measure:"):WORKFLOW.index("  decide:")]
    parts = re.findall(r"- \{set: (\w+), part: (\d+), parts: (\d+)\}", measure)
    by_set: dict[str, list[tuple[int, int]]] = {}
    for name, part, total in parts:
        by_set.setdefault(name, []).append((int(part), int(total)))
    for name, got in by_set.items():
        total = got[0][1]
        assert sorted(got) == [(i, total) for i in range(1, total + 1)], name
    assert set(by_set) == {"decision", "shadow"} and by_set["decision"][0][1] >= 4
    step = measure[measure.index("Baseline and candidate on this part"):]
    assert 'run base "$BASELINE_GGUF"' in step and 'run lora "$GGUF"' in step
    assert '--part "$PART/$PARTS"' in step
    decide = steps()
    assert "merge_results.py" in decide["Join the parts"]
    assert list(decide).index("Join the parts") < list(decide).index("Decide")
    assert "holdout_blind_v" not in WORKFLOW


def test_an_adapter_of_another_contract_is_refused_before_measuring():
    fetch = WORKFLOW[WORKFLOW.index("Fetch the adapter and check it fits this benchmark"):]
    fetch = fetch[:fetch.index("- name: Find the current default adapter")]
    assert 'm.get("contract_version", 1) != bench.CONTRACT_VERSION' in fetch
    assert 'm.get("grammar_sha256") != grammar' in fetch


def test_no_decision_before_the_decision_set_has_its_date():
    assert 'ACTIVE_SINCE is not set' in steps()["Decide"]
