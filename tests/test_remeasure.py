"""bench-lora signs; the remeasure path must never change a decided release, and the usual
path must behave exactly as before it existed.

    python -m pytest tests/test_remeasure.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import release_manifest  # noqa: E402
import verify_adapter  # noqa: E402

WORKFLOW = (ROOT / ".github" / "workflows" / "bench-lora.yml").read_text(encoding="utf-8")

# The decide job's steps as they were before the remeasure path (21.11.2026), without
# comment and if: lines; only an "if" that keeps them off the remeasure path was added.
UNCHANGED = {
    "Fetch the adapter again (signed with the manifest)": "6eaae8c0f269c6488e2e7319951948f6ee90176c275c9fa01109597ca4045d7e",
    # 22.11.2026: only the duplicate check of the rotation notice changed (one per holdout version,
    # tests/test_bench_lora_notices.py); the decision itself is the same.
    "Decide": "c6feffa74434dea50989764716b9b736d2b21913abdb1540e3abb3f9a80992bf",
    "Mark the release": "81799e0e15a3b0d9e15db9974332c4416662301960d20ffacaf3437983b17130",
    "actions/attest-build-provenance@4d101475d8b20a2381f78447822ac1eab6504dd8 # v4.2.2":
        "f08f843825b54565145af5c1d0f5acc1f3efcd4b074a38e61a66d61258c8f973",
    "Open the rejection notice": "3896efb2593a9990b87e05883f694f694d6b1961b40ed18b34750de84ae3d747",
    "Fail when rejected": "3d12ce5fe472bd8521438c7befce39cda62cde12db777e5fd66d66a294b89462",
}


def steps() -> dict[str, str]:
    job = WORKFLOW[WORKFLOW.index("  decide:"):]
    parts = re.split(r"\n(?=      - (?:name|uses): )", job)
    return {re.match(r"\s*- (?:name: |uses: )(.*)", p.strip("\n")).group(1).strip(): p for p in parts[1:]}


def body(step: str) -> str:
    return "\n".join(line.rstrip() for line in step.splitlines()
                     if line.strip() and not line.strip().startswith(("if:", "#")))


def condition(step: str) -> str:
    found = [line.strip() for line in step.splitlines() if line.strip().startswith("if:")]
    return found[0] if found else ""


def test_the_usual_path_is_unchanged():
    s = steps()
    for name, digest in UNCHANGED.items():
        assert hashlib.sha256(body(s[name]).encode()).hexdigest() == digest, name
    assert condition(s["Mark the release"]) == "if: ${{ !inputs.remeasure }}"
    assert condition(s["Open the rejection notice"]) == "if: steps.decide.outputs.accepted != 'true' && !inputs.remeasure"
    # Without the input every step runs as before: the signature too, whatever the decision.
    attest = s["actions/attest-build-provenance@4d101475d8b20a2381f78447822ac1eab6504dd8 # v4.2.2"]
    assert condition(attest).startswith("if: ${{ !inputs.remeasure ||")
    assert "remeasure:" in WORKFLOW and "default: false" in WORKFLOW[WORKFLOW.index("remeasure:"):]


def test_the_remeasure_path_never_changes_the_release():
    s = steps()
    record = s["Record the remeasurement"]
    assert condition(record) == "if: ${{ inputs.remeasure }}"
    assert "gh release edit" not in record and "--clobber" not in record and "gh release delete" not in record
    assert '"${prefix}_$f"' in record and 'prefix="remeasure-$day"' in record
    assert 'grep -qx adapter_manifest.json <<<"$assets"' in record and "exit 1" in record
    # Accepted releases only: "(rejected)" and "(candidate)" both stop before anything is measured into the release.
    assert '[ "$title" = "$ADAPTER" ] || {' in record
    guard = record.index('[ "$title" = "$ADAPTER" ]')
    assert guard < record.index("gh release upload") and guard < record.index("release_manifest.py")
    uploads = re.findall(r"gh release upload[^\n]*", record)
    assert uploads == ['gh release upload "$ADAPTER" "${files[@]}" --repo "$GITHUB_REPOSITORY"']
    notice = s["Open the remeasurement notice"]
    assert condition(notice) == "if: steps.decide.outputs.accepted != 'true' && inputs.remeasure"
    assert "gh release edit" not in notice and "Remeasure rejected:" in notice


def test_the_signed_manifest_says_it_is_a_remeasurement():
    manifest = {"adapter": "a.gguf", "adapter_sha256": "0" * 64, "base": "Qwen/Qwen2.5-1.5B-Instruct",
                "base_revision": "r", "train_files": {}, "prompt": "compact", "prompt_sha256": "1" * 64}
    verdict = {"accepted": True, "n": 235, "b_fixed": 3, "c_broken": 0, "p_value": 0.01, "base_correct": 200,
               "adapter_correct": 203, "reason": "ok"}
    models = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    plain = release_manifest.build(manifest, verdict, "qwen2.5-1.5b-instruct-q8_0", "1", models)
    again = release_manifest.build(manifest, verdict, "qwen2.5-1.5b-instruct-q8_0", "1", models,
                                   {"remeasured_on": "2026-11-21", "rule": "v3 decides, v4 can veto",
                                    "first_decision": "released 2026-11-07"})
    assert "remeasure" not in plain and again["remeasure"]["remeasured_on"] == "2026-11-21"
    # Optional field: readers of schema 1 accept both.
    assert verify_adapter.schema_problems(plain) == verify_adapter.schema_problems(again)


def test_the_title_guard_lets_only_the_bare_tag_through():
    import subprocess

    guard = '[ "$title" = "$ADAPTER" ] || { echo "refused" >&2; exit 1; }'
    for title, ok in (("vyvy-1.5b-dX", True), ("vyvy-1.5b-dX (rejected)", False), ("vyvy-1.5b-dX (candidate)", False)):
        done = subprocess.run(["bash", "-c", f'ADAPTER=vyvy-1.5b-dX; title="{title}"; {guard}'], capture_output=True)
        assert (done.returncode == 0) is ok, title
