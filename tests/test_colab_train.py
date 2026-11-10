"""1.5B trains on one Colab T4, 7B stays on Kaggle.

    python -m pytest tests/test_colab_train.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import collect  # noqa: E402
import kaggle_job  # noqa: E402
import plan_train  # noqa: E402

SHA = "a" * 40
SMALL = "Qwen/Qwen2.5-1.5B-Instruct"
LARGE = "Qwen/Qwen2.5-7B-Instruct"


def _script(tmp_path, **kw) -> str:
    args = dict(repo="owner/vyvy-ai", commit=SHA, base=SMALL, name="vyvy-1.5b-dabc", epochs=4,
                sweep="rank=16;lr=2e-4", budget=60)
    args.update(kw)
    return kaggle_job.write_colab_job(tmp_path / "colab_train.py", **args).read_text(encoding="utf-8")


def test_colab_script_is_the_kaggle_entry_for_one_gpu(tmp_path):
    text = _script(tmp_path)
    assert 'OUT = "/content/out"' in text and "/kaggle/working" not in text
    assert "GPUS = 1" in text
    assert text.rstrip().endswith('sh("tar", "-czf", "/content/out.tgz", "-C", OUT, ".")')
    compile(text, "colab_train.py", "exec")


@pytest.mark.parametrize("budget", [0, kaggle_job.COLAB_BUDGET_MAX + 1])
def test_colab_budget_stays_inside_a_free_session(tmp_path, budget):
    with pytest.raises(SystemExit):
        _script(tmp_path, budget=budget)


def test_colab_script_checks_like_kaggle(tmp_path):
    with pytest.raises(SystemExit):
        _script(tmp_path, commit="main")
    with pytest.raises(SystemExit):
        _script(tmp_path, sweep="rank=16; rm -rf /")


def test_plans_say_where_they_run():
    assert plan_train.PLANS[SMALL]["where"] == "colab"
    assert plan_train.PLANS[SMALL]["budget"] <= kaggle_job.COLAB_BUDGET_MAX
    assert plan_train.PLANS[SMALL]["gpus"] == 1 and not plan_train.PLANS[SMALL]["gpus_after_7b"]
    assert plan_train.PLANS[LARGE]["where"] == "kaggle"
    assert set(plan_train.WORKFLOW) == {p["where"] for p in plan_train.PLANS.values()}


def test_failure_titles_name_the_platform():
    assert collect.failure_title("colab/1.5b", "vyvy-1.5b-dabc", []).startswith("Colab training failed:")
    assert collect.failure_title("u/vyvy-train-x", "n", []).startswith("Kaggle training failed:")


@pytest.fixture
def scheduler(monkeypatch):
    started: list[str] = []
    monkeypatch.setattr(collect, "any_kernel_running", lambda: False)
    monkeypatch.setattr(collect, "synth_waiting", lambda now=None: False)
    monkeypatch.setattr(collect, "synth_done", lambda now=None: True)
    monkeypatch.setattr(collect, "open_failure", lambda repo, kid, name: False)
    monkeypatch.setattr(collect, "summary", lambda text: None)
    monkeypatch.setattr(plan_train, "plan", lambda base, epochs=4: {**plan_train.PLANS[base], "name": "n"})
    # The loop lock is open here; its own tests are in test_improve_loop.py.
    monkeypatch.setattr(collect.loop_guard, "load_state", lambda: {"mode": "one_round"})
    monkeypatch.setattr(collect.loop_guard, "release_history", lambda repo: [])
    monkeypatch.setattr(collect.loop_guard, "small_verdict_on_same_data", lambda repo: None)
    monkeypatch.setattr(collect.loop_guard, "gate", lambda *a, **k: (True, "open"))

    def run(*args, check=True):
        if args[:3] == ("gh", "workflow", "run"):
            started.append(f"{args[3]} {args[-1]}")
        return ""

    monkeypatch.setattr(collect, "run", run)
    return started


def test_scheduler_starts_kaggle_and_colab_side_by_side(scheduler, monkeypatch):
    monkeypatch.setattr(plan_train, "trained_release", lambda repo, base: None)
    collect.schedule("owner/vyvy-ai", "u")
    assert scheduler == [f"train-colab.yml base={SMALL}", f"train.yml base={LARGE}"]


def test_a_running_kaggle_kernel_does_not_hold_colab(scheduler, monkeypatch):
    monkeypatch.setattr(collect, "any_kernel_running", lambda: True)
    monkeypatch.setattr(plan_train, "trained_release", lambda repo, base: None)
    collect.schedule("owner/vyvy-ai", "u")
    assert scheduler == [f"train-colab.yml base={SMALL}"]


def test_scheduler_starts_colab_when_only_1_5b_is_missing(scheduler, monkeypatch):
    monkeypatch.setattr(plan_train, "trained_release", lambda repo, base: "done" if base == LARGE else None)
    collect.schedule("owner/vyvy-ai", "u")
    assert scheduler == [f"train-colab.yml base={SMALL}"]


def test_an_open_kaggle_failure_does_not_hold_colab(scheduler, monkeypatch):
    monkeypatch.setattr(plan_train, "trained_release", lambda repo, base: None)
    monkeypatch.setattr(collect, "open_failure", lambda repo, kid, name: not kid.startswith("colab/"))
    collect.schedule("owner/vyvy-ai", "u")
    assert scheduler == [f"train-colab.yml base={SMALL}"]


def test_a_running_kernel_delays_the_synthesis_but_not_colab_later(scheduler, monkeypatch):
    monkeypatch.setattr(collect, "any_kernel_running", lambda: True)
    monkeypatch.setattr(collect, "synth_done", lambda now=None: False)
    monkeypatch.setattr(plan_train, "trained_release", lambda repo, base: None)
    collect.schedule("owner/vyvy-ai", "u")
    # New teacher sentences are coming; training now would be trained again.
    assert scheduler == []


def test_a_closed_loop_lock_starts_nothing(scheduler, monkeypatch):
    monkeypatch.setattr(plan_train, "trained_release", lambda repo, base: None)
    monkeypatch.setattr(collect, "synth_done", lambda now=None: False)
    monkeypatch.setattr(collect.loop_guard, "gate", lambda *a, **k: (False, "paused"))
    collect.schedule("owner/vyvy-ai", "u")
    assert scheduler == []


def test_the_lock_can_hold_7b_alone(scheduler, monkeypatch):
    monkeypatch.setattr(plan_train, "trained_release", lambda repo, base: None)
    monkeypatch.setattr(collect.loop_guard, "gate", lambda state, history, base, *a, **k: (base == SMALL, "7B waits"))
    collect.schedule("owner/vyvy-ai", "u")
    assert scheduler == [f"train-colab.yml base={SMALL}"]
