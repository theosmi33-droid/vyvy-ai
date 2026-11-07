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
  failure with the likely cause and the last 100 log lines

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
import plan_train  # noqa: E402

BASES = {
    "Qwen/Qwen2.5-1.5B-Instruct": {"size": "1.5b", "next": "Qwen/Qwen2.5-7B-Instruct"},
    "Qwen/Qwen2.5-7B-Instruct": {"size": "7b", "next": None},
}
PATTERN = r"(^|/)(adapter/[^/]+|manifest\.json|sweep\.json)$"


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
    return f"Kaggle training failed: {kid} {key}"


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
        f"Candidate. LoRA adapter for {base} (Apache License 2.0), trained on data/train/ with the "
        f"{manifest.get('prompt')} prompt. Apply it on top of the unchanged base model; see "
        "training/README.md. The base model's license and notices apply to its use. Accepted only if "
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
        else:
            logdir = out / "log"
            run("kaggle", "kernels", "output", kid, "-p", str(logdir), "--file-pattern", r"\.log$", check=False)
            log = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in sorted(logdir.glob("*.log")))
            causes, tail = kaggle_watchdog.diagnose(log) if log else (["no log available"], [status])
            report_failure(args.repo, kid, causes, tail, run_name(log))
    schedule(args.repo, args.owner)
    return 1 if failed else 0


def synth_id() -> str:
    return hashlib.sha256((ROOT / "training" / "synthesize.py").read_bytes()).hexdigest()[:9]


def pending_since(marker: Path) -> datetime:
    """Start time written in a .pending marker ("<where> <ISO time>"); long ago if absent or unreadable."""
    try:
        return datetime.fromisoformat(marker.read_text(encoding="utf-8").split()[-1].replace("Z", "+00:00"))
    except (OSError, ValueError, IndexError):
        return datetime.min.replace(tzinfo=timezone.utc)


def synth_file() -> Path:
    return ROOT / "data" / "train" / f"vyvy_synth_{synth_id()}.jsonl"


def collect_synth(repo: str, owner: str) -> bool:
    """Take a finished teacher run into data/train/ and push it to main. False if that failed."""
    kid = f"{owner}/{kaggle_job.SYNTH_SLUG}"
    status = run("kaggle", "kernels", "status", kid, check=False)
    if "404" in status or "not found" in status.lower():
        return True
    state = kaggle_watchdog.state_of(status)
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
        m = re.search(r"run (vyvy-synth-[0-9a-f]{9})", log)
        report_failure(repo, kid, causes, tail, m.group(1) if m else None)
        return True
    run("kaggle", "kernels", "output", kid, "-p", str(out), "--file-pattern",
        r"(^|/)synth(_manifest\.json|\.jsonl|_dropped\.jsonl)$", check=False)
    if not (out / "synth_manifest.json").exists():
        return True
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "import_synth.py"), str(out)],
                          capture_output=True, text=True)
    if done.returncode == 3:
        return True
    if done.returncode != 0:
        report_failure(repo, kid, ["importing the teacher sentences failed"], done.stderr.splitlines()[-40:],
                       f"vyvy-synth-{synth_id()}")
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
        report_failure(repo, kid, ["pushing the teacher sentences failed"], [str(exc)], f"vyvy-synth-{synth_id()}")
        return False
    summary(f"{kid}: took {path.name} into data/train/.\n\n```\n{done.stderr[-1500:]}\n```")
    return True


def any_kernel_running() -> bool:
    return bool(run(sys.executable, str(ROOT / "tools" / "kaggle_busy.py"), check=False).strip())


def open_failure(repo: str, kid: str, name: str) -> bool:
    """True if this exact planned run (same inputs, same name) already failed and is not closed."""
    title = failure_title(kid, name, [])
    titles = run("gh", "issue", "list", "--repo", repo, "--state", "open", "--search",
                 f'"{title}" in:title', "--json", "title", "--jq", ".[].title", check=False)
    return title in titles.splitlines()


def schedule(repo: str, owner: str) -> None:
    if any_kernel_running():
        summary("Scheduler: a kernel of the account is running; nothing new started.")
        return
    # Teacher sentences first: training on data that is about to grow wastes a run.
    pending = synth_file().with_suffix(".pending")
    if not synth_file().exists() and pending_since(pending) > datetime.now(timezone.utc) - timedelta(hours=4):
        # The same synthesis runs elsewhere (Colab); its result is pushed to data/train/,
        # which starts Train. Waiting keeps one GPU run from being done twice.
        summary(f"Scheduler: synthesis {synth_id()} runs elsewhere ({pending.read_text().strip()}); waiting.")
        return
    if not synth_file().exists():
        if open_failure(repo, f"{owner}/{kaggle_job.SYNTH_SLUG}", f"vyvy-synth-{synth_id()}"):
            summary("Scheduler: synthesis waits for a fix (open failure issue); training goes ahead.")
        else:
            run("gh", "workflow", "run", "synth.yml", "--repo", repo, "--ref", "main")
            summary(f"Scheduler: started the teacher run vyvy-synth-{synth_id()}.")
            return
    for base in plan_train.ORDER:
        p = plan_train.plan(base)
        kid = f"{owner}/{kaggle_job.kernel_slug(base)}"
        if plan_train.trained_release(repo, base):
            continue
        if open_failure(repo, kid, p["name"]):
            summary(f"Scheduler: {base} waits for a fix (open failure issue for {kid}).")
            return
        run("gh", "workflow", "run", "train.yml", "--repo", repo, "--ref", "main", "-f", f"base={base}")
        summary(f"Scheduler: started training {p['name']} ({base}).")
        return
    summary("Scheduler: every planned adapter exists; nothing to train.")


if __name__ == "__main__":
    raise SystemExit(main())
