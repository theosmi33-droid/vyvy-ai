"""Every adapter release says its own license and keeps the base model's Apache notices.

    python -m pytest tests/test_license_files.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import license_files  # noqa: E402


def test_notice_names_both_licenses_and_the_required_notices() -> None:
    text = license_files.NOTICE.format(name="vyvy-1.5b-d0000000a", base="Qwen/Qwen2.5-1.5B-Instruct",
                                       revision="a" * 40)
    assert "Apache License, Version 2.0" in text and "LICENSE-base-model.txt" in text
    assert "PolyForm Noncommercial License 1.0.0" in text
    repo_notices = [l for l in (ROOT / "LICENSE").read_text().splitlines() if l.startswith("Required Notice:")]
    assert repo_notices and all(l in text for l in repo_notices)
    assert "as required for modifications" not in text
