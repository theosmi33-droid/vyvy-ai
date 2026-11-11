"""Every workflow that can start a GPU checks the loop lock first.

    python -m pytest tests/test_gpu_lock.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
# A Kaggle kernel push, or a Colab machine started by the CLI ("run" or "new").
GPU_START = re.compile(r"kaggle kernels push|colab --auth adc (run|new)\b")
LOCK = re.compile(r"loop_guard\.py --gate|gate_ok")


def test_every_gpu_start_passes_the_lock() -> None:
    starters = {p.name: p.read_text(encoding="utf-8") for p in WORKFLOWS.glob("*.yml")
                if GPU_START.search(p.read_text(encoding="utf-8"))}
    assert {"train.yml", "train-colab.yml", "synth.yml", "colab-probe.yml"} <= set(starters)
    unlocked = sorted(name for name, text in starters.items() if not LOCK.search(text))
    assert unlocked == [], f"GPU start without the loop lock in {unlocked}"


def test_the_lock_comes_before_the_gpu_step() -> None:
    for name in ("synth.yml", "colab-probe.yml", "train.yml", "train-colab.yml"):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert LOCK.search(text).start() < GPU_START.search(text).start(), name


def test_sign_ins_run_where_their_secrets_are() -> None:
    text = (WORKFLOWS / "loop-watch.yml").read_text(encoding="utf-8")
    kaggle = text[text.index("sign-in-kaggle:"):text.index("sign-in-colab:")]
    colab = text[text.index("sign-in-colab:"):text.index("  health:")]
    report = text[text.index("  report:"):]
    assert "environment: kaggle" in kaggle and "KAGGLE_API_TOKEN" in kaggle
    assert "environment: colab" in colab and "COLAB_ADC" in colab and "KAGGLE" not in colab
    assert "environment: kaggle" in report
