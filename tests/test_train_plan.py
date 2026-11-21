"""7B fits the Kaggle session: one config, two epochs, and training stops at the budget.

On 16.11. the first of six 7B configs needed about 15 h at 4 epochs; the kernel was
ended without an adapter (issue 11).

    python -m pytest tests/test_train_plan.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "training"))

import kaggle_job  # noqa: E402
import plan_train  # noqa: E402
import train_lora  # noqa: E402

SHA = "a" * 40
SMALL = "Qwen/Qwen2.5-1.5B-Instruct"
LARGE = "Qwen/Qwen2.5-7B-Instruct"


def test_7b_trains_one_config_for_at_most_two_epochs():
    p = plan_train.plan(LARGE)
    assert p["epochs"] == 2
    assert len(train_lora.sweep_configs(p["sweep"], 16, 2e-4)) == 1
    assert plan_train.plan(LARGE, epochs=1)["epochs"] == 1


def test_1_5b_plan_is_unchanged():
    p = plan_train.plan(SMALL)
    assert p["epochs"] == 4 and p["sweep"] == "rank=16;lr=2e-4" and p["budget"] == 60


def test_the_name_follows_the_capped_epochs():
    assert plan_train.plan(LARGE, epochs=4)["name"] == plan_train.plan(LARGE, epochs=2)["name"]
    assert plan_train.plan(LARGE, epochs=1)["name"] != plan_train.plan(LARGE, epochs=2)["name"]


def test_the_kaggle_kernel_stops_training_at_its_budget(tmp_path):
    kaggle_job.write_job(tmp_path, "u", "owner/vyvy-ai", SHA, LARGE, "vyvy-7b-dabc", 2,
                         sweep="rank=16;lr=2e-4", budget=600, batch=2, accum=8)
    text = (tmp_path / "train.py").read_text(encoding="utf-8")
    compile(text, "train.py", "exec")
    assert 'STOP = "600"' in text and '"--stop-after", STOP' in text


def test_without_a_budget_the_kernel_sets_no_stop(tmp_path):
    kaggle_job.write_job(tmp_path, "u", "owner/vyvy-ai", SHA, LARGE, "vyvy-7b-dabc", 2)
    assert 'STOP = "0"' in (tmp_path / "train.py").read_text(encoding="utf-8")


def test_colab_keeps_its_own_timeout(tmp_path):
    text = kaggle_job.write_colab_job(tmp_path / "colab_train.py", "owner/vyvy-ai", SHA, SMALL, "vyvy-1.5b-dabc",
                                      4, sweep="rank=16;lr=2e-4", budget=60).read_text(encoding="utf-8")
    assert 'STOP = "0"' in text


def test_the_stop_time_ends_training_with_a_last_evaluation_and_checkpoint():
    from types import SimpleNamespace

    control = SimpleNamespace(should_evaluate=False, should_save=False, should_training_stop=False)
    assert not train_lora.stop_if_due(control, None, 1e9)
    assert not train_lora.stop_if_due(control, 100.0, 99.0)
    assert not (control.should_evaluate or control.should_save or control.should_training_stop)
    assert train_lora.stop_if_due(control, 100.0, 100.0)
    assert control.should_evaluate and control.should_save and control.should_training_stop


def test_a_push_trains_only_after_this_weeks_teacher_sentences():
    # 20.11.2026: a code push during the teacher run trained 1.5B on the old data.
    for name in ("train.yml", "train-colab.yml"):
        text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        plan = text[text.index("Plan ("):text.index("Holdout lock")]
        assert '"$GITHUB_EVENT_NAME" = "push"' in plan and "collect.synth_done()" in plan, name


def test_one_config_needs_one_gpu():
    assert plan_train.grid_size("rank=16;lr=2e-4") == 1
    assert plan_train.grid_size("rank=16,32;lr=2e-4,1e-4,3e-4") == 6
    assert plan_train.grid_size("") == 1


def test_the_scheduler_looks_at_every_base(monkeypatch):
    # 20.11.2026: a decided 1.5B round returned before 7B was looked at.
    import collect

    started = []
    monkeypatch.setattr(collect, "any_kernel_running", lambda: False)
    monkeypatch.setattr(collect, "synth_waiting", lambda now=None: False)
    monkeypatch.setattr(collect, "synth_done", lambda now=None: True)
    monkeypatch.setattr(collect, "open_failure", lambda repo, kid, name: False)
    monkeypatch.setattr(collect, "summary", lambda text: None)
    monkeypatch.setattr(collect, "sealed_ready", lambda repo: (True, "passed"))
    monkeypatch.setattr(plan_train, "plan", lambda base, epochs=4: {**plan_train.PLANS[base], "name": "n"})
    monkeypatch.setattr(plan_train, "trained_release", lambda repo, base: None)
    monkeypatch.setattr(collect.loop_guard, "load_state", lambda: {"mode": "one_round"})
    monkeypatch.setattr(collect.loop_guard, "release_history", lambda repo: [])
    monkeypatch.setattr(collect.loop_guard, "small_verdict_on_same_data", lambda repo: None)
    monkeypatch.setattr(collect.loop_guard, "gate",
                        lambda state, history, base, *a, **k: (base == LARGE, "round done"))

    def run(*args, check=True):
        if args[:3] == ("gh", "workflow", "run"):
            started.append(f"{args[3]} {args[-1]}")
        return ""

    monkeypatch.setattr(collect, "run", run)
    collect.schedule("owner/vyvy-ai", "u")
    assert started == [f"train.yml base={LARGE}"]
