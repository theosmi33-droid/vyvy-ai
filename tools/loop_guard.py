#!/usr/bin/env python3
"""Hold points of the improvement loop: whether another round may start.

The loop (synthesis -> training -> bench-lora -> improve_report -> next
synthesis) runs on its own only while every hold point allows it. In order:

1. Rules: the acceptance rule (tools/compare_adapter.py) and the holdout files
   must match the fingerprint the user approved in data/loop_state.json.
   A changed rule or a new holdout needs the user's yes, never the loop's.
2. Goal: an adapter reached --goal accuracy on a sealed blind set of at least
   --min-sealed questions, looked at exactly once, with no wrong action on the
   unknown safety set. Both must hold. With --errors-unknown-only every
   remaining error must also be an "unknown" answer, never a wrong action.
3. Rotation: the last decision accepted an adapter. The active holdout has been
   looked at once too often; a fresh sealed set must come first (user's job).
4. Plateau: the last --plateau decisions were all rejected.
5. Quota: --max-rounds rounds were already decided in this ISO week.

Exit code 0 when the next round may start, 3 when the loop stops; the reason is
printed either way. The guard only reads; it changes nothing in the repository
and nothing in the application that uses the adapters.

    python tools/loop_guard.py --history decisions.json [--sealed sealed.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Outside training/ and data/train/: a push there starts Train; this file must not.
STATE = ROOT / "data" / "loop_state.json"
RULE_FILES = ["tools/compare_adapter.py", "tools/shadow_check.py"]
HOLDOUT_GLOB = "data/holdout_*.json"


def fingerprint(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    for rel in RULE_FILES + sorted(str(p.relative_to(root)) for p in root.glob(HOLDOUT_GLOB)):
        h.update(rel.encode())
        h.update((root / rel).read_bytes())
    return h.hexdigest()


def iso_week(stamp: str) -> str:
    y, w, _ = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00")).isocalendar()
    return f"{y}-W{w:02d}"


def check(history: list[dict], *, now: str, approved: str | None, current: str,
          sealed: dict | None = None, goal: float = 0.98, min_sealed: int = 270, errors_unknown_only: bool = False,
          plateau: int = 3, max_rounds: int = 5) -> tuple[bool, str]:
    """history: decisions oldest first, each {"adapter", "created" (ISO time), "accepted"}.

    sealed: {"adapter", "n", "correct", "wrong_actions", "looks", "unknown_n", "false_actions"}:
    the sealed blind set (wrong_actions = errors that are not "unknown") and the
    unknown safety set (false_actions), or None while none was measured.
    """
    if approved != current:
        return False, "rules or holdouts changed since the user approved them; the user decides"
    if sealed and sealed["looks"] == 1:
        acc = sealed["correct"] / sealed["n"]
        clean = not errors_unknown_only or sealed["wrong_actions"] == 0
        if sealed["n"] >= min_sealed and acc >= goal and sealed["false_actions"] == 0 and clean:
            return False, (f"goal reached: {sealed['adapter']} {acc:.1%} on the sealed set, "
                           f"0 wrong actions on {sealed['unknown_n']} unknown sentences")
    if sealed and sealed["looks"] > 1:
        return False, "the sealed set was looked at more than once; it no longer proves anything"
    if history and history[-1]["accepted"]:
        return False, f"{history[-1]['adapter']} was accepted: rotate the holdout before the next round"
    tail = history[-plateau:]
    if len(tail) == plateau and not any(d["accepted"] for d in tail):
        return False, f"plateau: the last {plateau} decisions were rejected ({', '.join(d['adapter'] for d in tail)})"
    week = iso_week(now)
    done = sum(1 for d in history if iso_week(d["created"]) == week)
    if done >= max_rounds:
        return False, f"{done} rounds in {week} already (limit {max_rounds}); next week"
    return True, f"next round allowed ({done} of {max_rounds} in {week})"


SEVEN_B = "Qwen/Qwen2.5-7B-Instruct"
SIZE = {"Qwen/Qwen2.5-1.5B-Instruct": "1.5b", SEVEN_B: "7b"}


def round_allowed(state: dict | None, history: list[dict], base: str, now: str) -> tuple[bool, str]:
    """May the scheduler start a GPU run (synthesis or training) for base now?

    The loop stops after every round: a decision (release) for base after the
    user's release of the round ends it. state is written only by the user:
    {"schema": 1, "fingerprint", "approved_on" (ISO time), "mode": "paused" | "one_round" | "weekly",
     "bases": [...]}. "one_round" allows one decision per base after approved_on;
    "weekly" allows one per base and ISO week, so the next one waits for the following week.
    The hold points of check() apply on top.
    """
    if not state or state.get("mode", "paused") == "paused":
        return False, "paused: the user releases the next round (data/loop_state.json)"
    if base not in state.get("bases", []):
        return False, f"{base} is not released in data/loop_state.json"
    mine = [d for d in history if d["adapter"].startswith(f"vyvy-{SIZE[base]}-")]
    if state["mode"] == "one_round":
        after = [d for d in mine if d["created"] > state["approved_on"]]
        if after:
            return False, f"round done ({after[-1]['adapter']}); report first, then the user releases the next"
        return True, "released for one round"
    if state["mode"] == "weekly":
        week = iso_week(now)
        done = [d for d in mine if iso_week(d["created"]) == week]
        if done:
            return False, f"{done[-1]['adapter']} decided in {week}; the next round comes next week"
        return True, f"weekly round for {week}"
    return False, f"unknown mode {state['mode']!r}"


def seven_b_ready(small: dict | None, max_p: float = 0.2) -> tuple[bool, str]:
    """7B only after a promising 1.5B run on the same training data.

    small: the verdict of that 1.5B adapter (p_value, per_intent), or None.
    """
    if small is None:
        return False, "no 1.5B decision on this training data yet"
    lost = small.get("per_intent", {}).get("unknown", {}).get("broken", 0)
    if small["p_value"] >= max_p or lost:
        return False, f"1.5B on this data: p={small['p_value']:.4f}, unknown broken {lost}; 7B not worth the quota"
    return True, f"1.5B on this data: p={small['p_value']:.4f}, no unknown loss"


def load_state(path: Path = STATE) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def release_history(repo: str) -> list[dict]:
    """Decisions from the releases: candidates still waiting for bench-lora are left out."""
    import subprocess

    out = subprocess.run(["gh", "release", "list", "--repo", repo, "--limit", "100", "--json", "tagName,createdAt,name"],
                         capture_output=True, text=True, check=True)
    return sorted(({"adapter": r["tagName"], "created": r["createdAt"], "accepted": "(" not in r["name"]}
                   for r in json.loads(out.stdout or "[]") if "(candidate)" not in r["name"]),
                  key=lambda d: d["created"])


def small_verdict_on_same_data(repo: str) -> dict | None:
    """verdict.json of the 1.5B release trained on today's training data, if there is one."""
    import subprocess

    import plan_train

    tag = plan_train.trained_release(repo, "Qwen/Qwen2.5-1.5B-Instruct")
    if not tag:
        return None
    got = subprocess.run(["gh", "release", "download", tag, "--repo", repo, "--pattern", "verdict.json",
                          "--output", "-"], capture_output=True, text=True)
    try:
        return json.loads(got.stdout)
    except json.JSONDecodeError:
        return None


def gate(state: dict | None, history: list[dict], base: str, now: str, current: str,
         small: dict | None = None) -> tuple[bool, str]:
    """The lock every GPU start passes (plan_train.py for Train and Train on Colab, collect.py
    for the synthesis): the user's release of the round, the approved rules and, in weekly
    mode, the hold points of check() over the decisions since that release. 7B also needs
    seven_b_ready(small)."""
    go, why = round_allowed(state, history, base, now)
    if not go:
        return go, why
    if state.get("fingerprint") != current:
        return False, "rules or holdouts changed since the user approved them; the user decides"
    if state["mode"] == "weekly":
        since = [d for d in history if d["created"] > state["approved_on"]]
        go, why = check(since, now=now, approved=current, current=current)
        if not go:
            return go, why
    if base == SEVEN_B:
        return seven_b_ready(small)
    return True, why


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Decide whether the improvement loop may start another round")
    p.add_argument("--gate", metavar="BASE", help="only the lock for a GPU start of BASE; needs --repo")
    p.add_argument("--repo", help="OWNER/REPO for --gate (history and verdicts from the releases)")
    p.add_argument("--history", type=Path, help="JSON list of decisions, oldest first")
    p.add_argument("--sealed", type=Path, help="result on the sealed set and the unknown safety set")
    p.add_argument("--state", type=Path, default=STATE, help="approved fingerprint (written by the user)")
    p.add_argument("--now", default=dt.datetime.now(dt.timezone.utc).isoformat())
    p.add_argument("--goal", type=float, default=0.98, help="accuracy on the sealed set (265 of 270 = 0.9815)")
    p.add_argument("--min-sealed", type=int, default=270, help="questions the sealed set needs at least")
    p.add_argument("--errors-unknown-only", action="store_true", help="no remaining error may be a wrong action")
    p.add_argument("--plateau", type=int, default=3)
    p.add_argument("--max-rounds", type=int, default=5)
    args = p.parse_args(argv)
    if args.gate:
        go, why = gate(load_state(args.state), release_history(args.repo), args.gate, args.now, fingerprint(),
                       small_verdict_on_same_data(args.repo) if args.gate == SEVEN_B else None)
        print(json.dumps({"ok": go, "reason": why}))
        return 0 if go else 3
    history = json.loads(args.history.read_text(encoding="utf-8"))
    now = dt.datetime.fromisoformat(args.now.replace("Z", "+00:00"))
    history = sorted((d for d in history if dt.datetime.fromisoformat(d["created"].replace("Z", "+00:00")) <= now),
                     key=lambda d: d["created"])
    approved = json.loads(args.state.read_text(encoding="utf-8")).get("fingerprint") if args.state.exists() else None
    sealed = json.loads(args.sealed.read_text(encoding="utf-8")) if args.sealed else None
    go, reason = check(history, now=args.now, approved=approved, current=fingerprint(), sealed=sealed,
                       goal=args.goal, min_sealed=args.min_sealed, errors_unknown_only=args.errors_unknown_only,
                       plateau=args.plateau, max_rounds=args.max_rounds)
    print(("Continue: " if go else "Stop: ") + reason)
    return 0 if go else 3


if __name__ == "__main__":
    raise SystemExit(main())
