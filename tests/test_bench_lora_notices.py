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
    active = re.search(r"data/holdout_blind_(v\d+)\.json base_blind", WORKFLOW).group(1)
    assert prefix == f"Holdout rotation due (blind {active} "
    # Issue #8, still open on 22.11.2026, hid the notice for v3.
    assert not "Holdout rotation due (blind v2 used for 8 decisions)".startswith(prefix)
    assert f"{prefix}used for 9 decisions)".startswith(prefix)
