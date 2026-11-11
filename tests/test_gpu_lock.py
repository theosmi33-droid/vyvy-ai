"""Every workflow that can start a GPU checks the loop lock first.

    python -m pytest tests/test_gpu_lock.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
# A Kaggle kernel pushed onto an accelerator, or a Colab machine started by the CLI ("run" or "new").
# A push without --accelerator runs on CPU (sealed-eval.yml) and uses no GPU quota.
GPU_START = re.compile(r"kaggle kernels push[^\n]*--accelerator|colab --auth adc (run|new)\b")
CPU_PUSH = re.compile(r"kaggle kernels push(?![^\n]*--accelerator)")
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


def test_cpu_kernels_stay_cpu() -> None:
    import json
    import sys
    import tempfile
    sys.path.insert(0, str(ROOT / "tools"))
    import kaggle_job
    text = (WORKFLOWS / "sealed-eval.yml").read_text(encoding="utf-8")
    assert CPU_PUSH.search(text) and not GPU_START.search(text)
    with tempfile.TemporaryDirectory() as tmp:
        kaggle_job.write_sealed_job(Path(tmp), "user", "owner/repo", "a" * 40, "vyvy-1.5b-d0000000a",
                                    "vyvy-sealed-v1", "v1")
        meta = json.loads((Path(tmp) / "kernel-metadata.json").read_text())
        script = (Path(tmp) / "sealed.py").read_text()
    assert meta["enable_gpu"] is False and "machine_shape" not in meta
    assert meta["dataset_sources"] == ["user/vyvy-sealed-v1"] and meta["is_private"] is True
    assert script.startswith("# Generated") and 'dict(id=r["id"], expected=r["expected"], got=r["got"])' in script


def test_the_sealed_set_is_looked_at_once() -> None:
    text = (WORKFLOWS / "sealed-eval.yml").read_text(encoding="utf-8")
    once = text.index("Once per sealed set")
    assert once < text.index("kaggle kernels push") and "already looked at" in text
    assert "workflow_dispatch" in text and "schedule" not in text and "push:" not in text
