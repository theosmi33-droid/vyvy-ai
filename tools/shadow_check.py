#!/usr/bin/env python3
"""Shadow holdout check: an adapter accepted on the active holdout must not be worse here.

Every acceptance decision is measured on the same active holdout (blind v2), so
over many rounds the pipeline can pick adapters that fit that set by chance.
The shadow holdout (blind v3) is never used to decide acceptance; it only
vetoes: if the adapter is significantly worse than the baseline on it (exact
one-sided McNemar on broken vs fixed, p < 0.05), or it breaks a safety intent
("unknown") net, the adapter is rejected. The result is added to verdict.json.

The script also counts decisions measured on the active holdout. At
--rotate-after or more it says so: the active holdout is then worn and should be
replaced (shadow becomes active, a newly written set becomes the shadow, the
old active set may go into training).

    python tools/shadow_check.py base_shadow.json lora_shadow.json --verdict verdict.json \
        --decisions 3 --rotate-after 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_adapter as ca  # noqa: E402


def check(base: dict, lora: dict, alpha: float = 0.05) -> dict:
    v = ca.decide(base, lora, alpha)
    b, c = v["b_fixed"], v["c_broken"]
    p_worse = ca.mcnemar_p(c, b)
    safety_net = sum(1 for r in v["regressions"] if r["expected"] in ca.SAFETY_INTENTS) - \
        sum(1 for r in v["fixed"] if r["expected"] in ca.SAFETY_INTENTS)
    veto = p_worse < alpha or safety_net > 0
    reason = (f"shadow holdout: b={b}, c={c}, p(worse)={p_worse:.4f}"
              + (", unknown worse" if safety_net > 0 else "")
              + ("; significantly worse, rejected" if veto else "; not worse"))
    return {"n": v["n"], "b_fixed": b, "c_broken": c, "p_worse": round(p_worse, 6), "veto": veto,
            "reason": reason, "base_correct": v["base_correct"], "adapter_correct": v["adapter_correct"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Veto an adapter that is worse on the shadow holdout")
    p.add_argument("base", type=Path)
    p.add_argument("adapter", type=Path)
    p.add_argument("--verdict", type=Path, required=True, help="verdict.json of the active holdout; updated")
    p.add_argument("--decisions", type=int, default=0, help="decisions measured on the active holdout so far")
    p.add_argument("--rotate-after", type=int, default=8)
    args = p.parse_args(argv)
    s = check(json.loads(args.base.read_text(encoding="utf-8")), json.loads(args.adapter.read_text(encoding="utf-8")))
    v = json.loads(args.verdict.read_text(encoding="utf-8"))
    v["shadow"] = s
    if v["accepted"] and s["veto"]:
        v["accepted"] = False
        v["reason"] = f"{v['reason']}; but {s['reason']}"
    else:
        v["reason"] = f"{v['reason']}; {s['reason']}"
    v["rotation_due"] = args.decisions + 1 >= args.rotate_after
    args.verdict.write_text(json.dumps(v, indent=1) + "\n", encoding="utf-8")
    print(f"### Shadow holdout\n\n{s['reason']}. Baseline {s['base_correct']}/{s['n']}, "
          f"adapter {s['adapter_correct']}/{s['n']}.\n")
    if v["rotation_due"]:
        print(f"Decision {args.decisions + 1} on the active holdout: rotation due "
              f"(limit {args.rotate_after}).\n")
    return 0 if v["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
