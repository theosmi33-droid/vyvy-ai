#!/usr/bin/env python3
"""Collect finished Kaggle training runs; runs on a schedule (collect.yml).

Training runs can take up to 11 hours, longer than a GitHub job may wait, so
Train only starts the Kaggle kernel and this collector picks up the result.
For every base model it checks the latest kernel run once:

- running or queued: nothing to do
- complete: fetch adapter/, manifest.json and sweep.json; unless the release
  already exists, convert to GGUF on this runner, add the license files, publish
  a candidate (pre-release), start bench-lora, and after 1.5B start the 7B run
- error, or complete without an adapter: open one GitHub issue per distinct
  failure with the likely cause and the last 100 log lines; the run is retried
  once automatically, a second failure waits for a fix; issues of a kernel are
  closed when a later run of it trains an adapter

Then it schedules: if no kernel of the account is running (any project), it
starts the first missing training run (1.5B before 7B) whose adapter does not
exist yet and whose kernel has no open failure issue. Train re-checks quota and
the free GPU before it pushes anything.

    python tools/collect.py --repo OWNER/REPO --owner KAGGLE_USER
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import kaggle_job  # noqa: E402
import kaggle_watchdog  # noqa: E402
import loop_guard  # noqa: E402
import plan_train  # noqa: E402

BASES = {
    "Qwen/Qwen2.5-1.5B-Instruct": {"size": "1.5b", "next": "Qwen/Qwen2.5-7B-Instruct"},
    "Qwen/Qwen2.5-7B-Instruct": {"size": "7b", "next": None},
}
PATTERN = r"(^|/)(adapter/[^/]+|manifest\.json|sweep\.json)$"
RETRY_NOTE = "Retried automatically once by Collect."


def run(*args: str, check: bool = True) -> str:
    done = subprocess.run(list(args), capture_output=True, text=True)
    if check and done.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])} failed: {done.stderr.strip()[-500:]}")
    return (done.stdout + done.stderr).strip()


def summary(text: str) -> None:
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text + "\n\n")


def release_exists(repo: str, tag: str) -> bool:
    return subprocess.run(["gh", "release", "view", tag, "--repo", repo], capture_output=True).returncode == 0


def run_name(log: str) -> str | None:
    """Adapter name of a kernel run, read from its log (the train_lora.py command line)."""
    m = re.search(r"--name['\", ]+(vyvy-[a-z0-9.-]+)", log)
    return m.group(1) if m else None


def failure_title(kid: str, name: str | None, tail: list[str]) -> str:
    key = name or hashlib.sha256("\n".join(tail[-30:]).encode("utf-8")).hexdigest()[:8]
    platform = "Colab" if kid.startswith("colab/") else "Kaggle"
    return f"{platform} training failed: {kid} {key}"


def report_failure(repo: str, kid: str, causes: list[str], tail: list[str], name: str | None = None) -> None:
    title = failure_title(kid, name, tail)
    existing = run("gh", "issue", "list", "--repo", repo, "--state", "all", "--search", f'"{title}" in:title',
                   "--json", "title", "--jq", ".[].title", check=False)
    if title in existing.splitlines():
        summary(f"{kid}: failure already reported ({title}).")
        return
    body = "\n".join([f"Kernel `{kid}` ended with an error.", "", "Likely cause: " + ", ".join(causes), "",
                      "Last log lines:", "", "```", *tail, "```", "",
                      "Opened by the Collect workflow. A new push to the training data or code starts a new run."])
    run("gh", "issue", "create", "--repo", repo, "--title", title, "--body", body)
    summary(f"{kid}: failure reported as issue ({', '.join(causes)}).")


def publish(repo: str, out: Path, base: str, sha: str) -> str | None:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    name = manifest["name"]
    if release_exists(repo, name):
        return None
    if manifest.get("holdout_lock") != "passed":
        raise RuntimeError(f"{name}: manifest does not confirm the holdout lock")
    run(sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "training" / "convert-requirements.txt"))
    run(sys.executable, str(ROOT / "tools" / "convert_adapter.py"), str(out), "--name", name,
        "--cache", "/tmp/vyvy-cache")
    run(sys.executable, str(ROOT / "tools" / "license_files.py"), str(out))
    sweep = manifest.get("sweep") or []
    lines = [f"rank {s['rank']}, lr {s['lr']:g}: validation accuracy {s['val_accuracy']}, "
             f"loss {s['eval_loss']}, {s['minutes']} min" for s in sweep]
    notes = "\n".join([
        f"Candidate. LoRA adapter for {base} (base model: Apache License 2.0), trained on data/train/ "
        f"with the {manifest.get('prompt')} prompt. Apply it on top of the unchanged base model; see "
        "training/README.md. The adapter is licensed under the PolyForm Noncommercial License 1.0.0 "
        "(commercial use needs a separate written agreement, see NOTICE.txt); the base model's license "
        "and notices apply to its use. Accepted only if "
        "bench-lora finds no regression on the blind holdout and a higher accuracy than the current default.",
        "", f"Chosen: rank {manifest.get('rank')}, lr {manifest.get('chosen_lr')}, validation accuracy "
        f"{manifest.get('val_accuracy')}.", "", "Sweep:", *[f"- {line}" for line in lines]])
    files = [out / f"{name}.gguf", out / "manifest.json", out / "LICENSE-base-model.txt", out / "NOTICE.txt"]
    if (out / "sweep.json").exists():
        files.append(out / "sweep.json")
    run("gh", "release", "create", name, *map(str, files), "--prerelease", "--repo", repo,
        "--target", sha, "--title", f"{name} (candidate)", "--notes", notes)
    run("gh", "workflow", "run", "bench-lora.yml", "--repo", repo, "--ref", "main", "-f", f"adapter={name}")
    return name


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Collect finished Kaggle training runs")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    p.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME"))
    p.add_argument("--sha", default=os.environ.get("GITHUB_SHA", "main"))
    args = p.parse_args(argv)

    failed = not collect_synth(args.repo, args.owner)
    for base, info in BASES.items():
        kid = f"{args.owner}/{kaggle_job.kernel_slug(base)}"
        status = run("kaggle", "kernels", "status", kid, check=False)
        if "404" in status or "not found" in status.lower():
            summary(f"{kid}: no kernel yet.")
            continue
        state = kaggle_watchdog.state_of(status)
        if state == "running":
            summary(f"{kid}: still running.")
            continue
        out = Path("collect") / info["size"]
        shutil.rmtree(out, ignore_errors=True)
        if state == "complete":
            run("kaggle", "kernels", "output", kid, "-p", str(out), "--file-pattern", PATTERN, check=False)
            if not (out / "manifest.json").exists():
                report_failure(args.repo, kid, ["run completed without an adapter"], [status])
                continue
            try:
                name = publish(args.repo, out, base, args.sha)
            except Exception as exc:  # noqa: BLE001  (reported, other bases still collected)
                failed = True
                report_failure(args.repo, kid, ["collecting failed on GitHub"], [str(exc)])
                continue
            if name is None:
                summary(f"{kid}: latest run already collected.")
                continue
            summary(f"{kid}: published candidate `{name}`, bench-lora started.")
            close_failures(args.repo, kid, name)
        else:
            logdir = out / "log"
            run("kaggle", "kernels", "output", kid, "-p", str(logdir), "--file-pattern", r"\.log$", check=False)
            log = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in sorted(logdir.glob("*.log")))
            causes, tail = kaggle_watchdog.diagnose(log) if log else (["no log available"], [status])
            report_failure(args.repo, kid, causes, tail, run_name(log))
    schedule(args.repo, args.owner)
    return 1 if failed else 0


# Rounds taken in before rounds followed the Kaggle week: same week, old label.
LEGACY_ROUNDS = {"2026-11-07": "2026-W45"}


def synth_round(now: datetime | None = None) -> str:
    """One teacher synthesis per Kaggle quota week (Saturday 00:00 UTC to Saturday).

    Kaggle resets the weekly GPU quota on Saturday 00:00 UTC. Rounds named after
    the ISO week (Monday) put two rounds into one quota week, and the second could
    not finish. The label is the date of the week's Saturday, e.g. 2026-11-14.
    """
    now = now or datetime.now(timezone.utc)
    saturday = now.date() - timedelta(days=(now.weekday() - 5) % 7)
    return saturday.isoformat()


def synth_name() -> str:
    return f"vyvy-synth-{synth_round()}"


def pending_since(marker: Path) -> datetime:
    """Start time written in a .pending marker ("<where> <ISO time>"); long ago if absent or unreadable."""
    try:
        return datetime.fromisoformat(marker.read_text(encoding="utf-8").split()[-1].replace("Z", "+00:00"))
    except (OSError, ValueError, IndexError):
        return datetime.min.replace(tzinfo=timezone.utc)


def synth_waiting(now: datetime | None = None) -> bool:
    """This week's synthesis runs elsewhere (a .pending marker younger than 4 h) and is not in yet."""
    pending = ROOT / "data" / "train" / f"vyvy_synth_{synth_round(now)}.pending"
    return not synth_done(now) and pending_since(pending) > datetime.now(timezone.utc) - timedelta(hours=4)


def synth_done(now: datetime | None = None) -> bool:
    rounds = {synth_round(now), LEGACY_ROUNDS.get(synth_round(now), "")} - {""}
    return any(any((ROOT / "data" / "train").glob(f"vyvy_synth_*_{r}.jsonl")) for r in rounds)


def collect_synth(repo: str, owner: str) -> bool:
    """Take a finished teacher run into data/train/ and push it to main. False if that failed."""
    kid = f"{owner}/{kaggle_job.SYNTH_SLUG}"
    status = run("kaggle", "kernels", "status", kid, check=False)
    if "404" in status or "not found" in status.lower():
        return True
    state = kaggle_watchdog.state_of(status)
    if state == "running" and not re.search(r"running|queued", status, re.I):
        return True  # no such kernel yet (the CLI answers with an access error, not 404)
    if state == "running":
        summary(f"{kid}: still running.")
        return True
    out = Path("collect") / "synth"
    shutil.rmtree(out, ignore_errors=True)
    if state != "complete":
        logdir = out / "log"
        run("kaggle", "kernels", "output", kid, "-p", str(logdir), "--file-pattern", r"\.log$", check=False)
        log = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in sorted(logdir.glob("*.log")))
        causes, tail = kaggle_watchdog.diagnose(log) if log else (["no log available"], [status])
        m = re.search(r"run (vyvy-synth-[0-9A-Za-z-]+)", log)
        report_failure(repo, kid, causes, tail, m.group(1) if m else None)
        return True
    run("kaggle", "kernels", "output", kid, "-p", str(out), "--file-pattern",
        r"(^|/)synth(_manifest\.json|\.jsonl|_dropped\.jsonl)$", check=False)
    if not (out / "synth_manifest.json").exists():
        return True
    # The semantic guard of the import needs the sentence model; installed only when a round comes in.
    run(sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "tools" / "semantic-requirements.txt"),
        check=False)
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "import_synth.py"), str(out)],
                          capture_output=True, text=True)
    if done.returncode == 3:
        return True
    if done.returncode != 0:
        report_failure(repo, kid, ["importing the teacher sentences failed"], done.stderr.splitlines()[-40:],
                       synth_name())
        return False
    path = Path(done.stdout.strip().splitlines()[-1])
    try:
        run("git", "-C", str(ROOT), "add", str(path), str(path.with_suffix(".json")))
        run("git", "-C", str(ROOT), "-c", "user.name=github-actions[bot]",
            "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
            "commit", "-m", f"Teacher sentences {path.name} (training/synthesize.py)")
        run("git", "-C", str(ROOT), "pull", "--rebase", "origin", "main")
        run("git", "-C", str(ROOT), "push", "origin", "HEAD:main")
    except Exception as exc:  # noqa: BLE001  (reported; the next Collect run retries)
        report_failure(repo, kid, ["pushing the teacher sentences failed"], [str(exc)], synth_name())
        return False
    summary(f"{kid}: took {path.name} into data/train/.\n\n```\n{done.stderr[-1500:]}\n```")
    return True


def any_kernel_running() -> bool:
    return bool(run(sys.executable, str(ROOT / "tools" / "kaggle_busy.py"), check=False).strip())


def open_failure(repo: str, kid: str, name: str) -> bool:
    """True if this exact planned run already failed, is not closed and was already retried once.

    The first time a run is found failed, it is retried automatically (a comment on the
    issue records that); only a second failure waits for a fix. Most first failures on a
    free GPU are passing ones: no GPU assigned, a stalled download, a dropped session.
    """
    title = failure_title(kid, name, [])
    found = run("gh", "issue", "list", "--repo", repo, "--state", "open", "--search", f'"{title}" in:title',
                "--json", "number,title", "--jq", f'.[] | select(.title == "{title}") | .number', check=False)
    number = found.strip().splitlines()[0] if found.strip() else ""
    if not number:
        return False
    # Only Collect's own notes count: anyone can comment on a public issue.
    comments = run("gh", "issue", "view", number, "--repo", repo, "--json", "comments", "--jq",
                   '.comments[] | select(.author.login == "github-actions" or .author.login == "app/github-actions") | .body', check=False)
    if RETRY_NOTE in comments:
        return True
    run("gh", "issue", "comment", number, "--repo", repo, "--body",
        f"{RETRY_NOTE} If it fails again, the run waits for a fix.", check=False)
    return False


def close_failures(repo: str, kid: str, name: str) -> None:
    """Close the open failure issues of a kernel once a run of it trained an adapter."""
    numbers = run("gh", "issue", "list", "--repo", repo, "--state", "open", "--search",
                  f'"Kaggle training failed: {kid}" in:title', "--json", "number", "--jq", ".[].number", check=False)
    for number in numbers.split():
        run("gh", "issue", "close", number, "--repo", repo, "--comment",
            f"Closed by Collect: `{name}` was trained on this kernel afterwards.", check=False)


def same_plan_on_kaggle(owner: str, base: str) -> bool:
    """True while this base's Kaggle kernel runs (its Colab plan was moved to Kaggle by hand)."""
    status = run("kaggle", "kernels", "status", f"{owner}/{kaggle_job.kernel_slug(base)}", check=False)
    return kaggle_watchdog.state_of(status) == "running" and bool(re.search(r"running|queued", status, re.I))


def schedule(repo: str, owner: str) -> None:
    # One Kaggle run at a time (any kernel of the account holds the GPU); a Colab
    # run does not hold it, so Colab plans are started independently of Kaggle.
    kaggle_free = not any_kernel_running()
    # Every GPU run needs the user's release of the round (data/loop_state.json); the
    # synthesis belongs to the round of the first planned base.
    state, history = loop_guard.load_state(), loop_guard.release_history(repo)
    now = datetime.now(timezone.utc).isoformat()
    go, why = loop_guard.gate(state, history, plan_train.ORDER[0], now, loop_guard.fingerprint())
    if not go:
        summary(f"Scheduler: {why}.")
        return
    # Teacher sentences first: training on data that is about to grow wastes a run.
    pending = ROOT / "data" / "train" / f"vyvy_synth_{synth_round()}.pending"
    if synth_waiting():
        # The same synthesis runs elsewhere (Colab); its result is pushed to data/train/,
        # which starts Train. Waiting keeps one GPU run from being done twice.
        summary(f"Scheduler: synthesis {synth_round()} runs elsewhere ({pending.read_text().strip()}); waiting.")
        return
    if not synth_done():
        if open_failure(repo, f"{owner}/{kaggle_job.SYNTH_SLUG}", synth_name()):
            summary("Scheduler: synthesis waits for a fix (open failure issue); training goes ahead.")
        elif not kaggle_free:
            summary("Scheduler: a kernel of the account is running; the synthesis starts after it.")
            return
        else:
            run("gh", "workflow", "run", "synth.yml", "--repo", repo, "--ref", "main")
            summary(f"Scheduler: started the teacher run {synth_name()}.")
            return
    if not kaggle_free:
        summary("Scheduler: a kernel of the account is running; no new Kaggle run.")
    started = False
    for base in plan_train.ORDER:
        p = plan_train.plan(base)
        where = p.get("where", "kaggle")
        if (where == "kaggle" and not kaggle_free) or plan_train.trained_release(repo, base):
            continue
        go, why = loop_guard.gate(state, history, base, now, loop_guard.fingerprint(),
                                  loop_guard.small_verdict_on_same_data(repo) if base == loop_guard.SEVEN_B else None)
        if not go:
            summary(f"Scheduler: {base}: {why}.")
            continue
        kid = f"{owner}/{kaggle_job.kernel_slug(base)}" if where == "kaggle" else f"colab/{p['size']}"
        if where == "colab" and not kaggle_free and same_plan_on_kaggle(owner, base):
            # A Colab plan trained on Kaggle by hand (train.yml gpu=kaggle) must not run twice.
            summary(f"Scheduler: {base} is training on Kaggle right now; no Colab run.")
            continue
        if open_failure(repo, kid, p["name"]):
            summary(f"Scheduler: {base} waits for a fix (open failure issue for {kid}).")
            if where == "kaggle":
                kaggle_free = False
            continue
        run("gh", "workflow", "run", plan_train.WORKFLOW[where], "--repo", repo, "--ref", "main",
            "-f", f"base={base}")
        summary(f"Scheduler: started training {p['name']} ({base}) on {where}.")
        started = True
        if where == "kaggle":
            kaggle_free = False
    if not started and kaggle_free:
        summary("Scheduler: every planned adapter exists; nothing to train.")


if __name__ == "__main__":
    raise SystemExit(main())
