"""tools/holdout_build.py on invented sentences: blind agreement, filters, deterministic split, no text out."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runner"))

import bench  # noqa: E402
import holdout_build as hb  # noqa: E402

INTENTS = bench.grammar_intents(bench.DEFAULT_GRAMMAR.read_text(encoding="utf-8"))
def sentence(intent: str, i: int) -> str:
    """Invented stand-ins, far apart from each other: four hex words from a hash of intent and index."""
    h = hashlib.sha256(f"{intent}/{i}".encode()).hexdigest()
    return " ".join(h[k:k + 6] for k in range(0, 24, 6))


def write(folder: Path, rows: list[dict], blind_a: dict | None = None, blind_b: dict | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                             encoding="utf-8")
    for name, override in (("blind_a", blind_a or {}), ("blind_b", blind_b or {})):
        (folder / f"{name}.jsonl").write_text(
            "".join(json.dumps({"text": r["text"], "intent": override.get(r["text"], r["expected_intent"])},
                               ensure_ascii=False) + "\n" for r in rows if override.get(r["text"]) != "MISSING"),
            encoding="utf-8")
    return folder


def rows_for(n_per_intent: int, n_unknown: int) -> list[dict]:
    return [{"text": sentence(i, k), "expected_intent": i} for i in INTENTS
            for k in range(n_unknown if i == "unknown" else n_per_intent)]


SMALL = {"per_intent": 2, "unknown": 4, "shadow_per_intent": 1, "shadow_unknown": 2}


def test_complete_split_is_deterministic_and_disjoint(tmp_path):
    folder = write(tmp_path / "in", rows_for(4, 7))
    a = hb.build(folder, None, semantic=False, known=([], []), **SMALL)
    b = hb.build(folder, None, semantic=False, known=([], []), **SMALL)
    assert a == b and a["report"]["complete"]
    dec = [i["text"] for i in a["decision"]["items"]]
    sha = [i["text"] for i in a["shadow"]["items"]]
    assert len(dec) == 24 * 2 + 4 and len(sha) == 24 * 1 + 2 and not set(dec) & set(sha)
    assert a["report"]["per_intent"]["unknown"] == {**a["report"]["per_intent"]["unknown"], "decision": 4,
                                                    "shadow": 2, "unused": 1}
    assert a["decision"]["semantic_filter"].startswith("NOT RUN")


def test_only_sentences_both_blind_assignments_agree_on(tmp_path):
    rows = rows_for(4, 7)
    t0, t1 = rows[0]["text"], rows[1]["text"]
    folder = write(tmp_path / "in", rows, blind_a={t0: "tasks_open"}, blind_b={t1: "MISSING"})
    r = hb.build(folder, None, semantic=False, known=([], []), **SMALL)
    kept = {i["text"] for i in r["decision"]["items"] + r["shadow"]["items"]}
    assert t0 not in kept and t1 not in kept
    assert r["report"]["dropped"]["blind_disagrees"] == 1 and r["report"]["dropped"]["blind_missing"] == 1


def test_lexical_filter_against_training_holdouts_sealed_and_each_other(tmp_path):
    rows = rows_for(4, 7)
    train_copy, hold_copy, sealed_copy, twin = rows[0]["text"], rows[4]["text"], rows[8]["text"], rows[12]["text"]
    rows.append({"text": twin + "!", "expected_intent": rows[12]["expected_intent"]})  # same after normalising
    rows.append({"text": twin + " bitte", "expected_intent": rows[12]["expected_intent"]})  # near the twin
    folder = write(tmp_path / "in", rows)
    sealed = {"schema": 1, "items": [hb.fingerprint_item(sealed_copy)]}
    r = hb.build(folder, sealed, semantic=False, known=([train_copy.upper()], [hold_copy + "?"]), **SMALL)
    d = r["report"]["dropped"]
    assert d["near_training"] == 1 and d["near_holdout"] == 1 and d["near_sealed"] == 1
    assert d["duplicate"] == 1 and d["near_candidate"] == 1


def test_short_intents_are_reported_and_the_exit_code_says_so(tmp_path):
    rows = [r for r in rows_for(4, 7) if r["expected_intent"] != "complete_goal"]
    folder = write(tmp_path / "in", rows)
    out = tmp_path / "out"
    code = hb.main(["--candidates", str(folder), "--out", str(out), "--semantic", "skip",
                    "--per-intent", "2", "--unknown", "4", "--shadow-per-intent", "1", "--shadow-unknown", "2"])
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert code == 3 and not report["complete"] and report["per_intent"]["complete_goal"]["short"] == 3


def test_nothing_written_into_the_repository_and_no_text_in_report_or_fingerprint(tmp_path, capsys):
    folder = write(tmp_path / "in", rows_for(4, 7))
    assert hb.main(["--candidates", str(folder), "--out", str(ROOT / "data" / "x"), "--semantic", "skip"]) == 2
    assert hb.main(["--candidates", str(ROOT / "data"), "--out", str(tmp_path / "o"), "--semantic", "skip"]) == 2
    out = tmp_path / "out"
    hb.main(["--candidates", str(folder), "--out", str(out), "--semantic", "skip", "--per-intent", "2",
             "--unknown", "4", "--shadow-per-intent", "1", "--shadow-unknown", "2"])
    printed = capsys.readouterr()
    for name in ("report.json", "fingerprint.json"):
        body = (out / name).read_text(encoding="utf-8")
        for r in rows_for(4, 7):
            assert r["text"] not in body and r["text"] not in printed.out + printed.err


def test_semantic_filter_uses_the_model_it_is_given(tmp_path):
    class Fake:
        """Says a sentence means the same as a training sentence when it starts with a digit."""

        def __init__(self):
            self.kinds = []

        def add(self, kind, texts):
            self.kinds += [kind for _ in texts]

        def near(self, text):
            return "semantic_training" if "semantic_training" in self.kinds and text[0].isdigit() else None

    folder = write(tmp_path / "in", rows_for(6, 9))
    r = hb.build(folder, None, semantic=True, semantic_model=Fake(), known=(["irgendein Satz"], []), **SMALL)
    assert r["report"]["dropped"]["semantic_training"] > 0
    assert all(not i["text"][0].isdigit() for i in r["decision"]["items"] + r["shadow"]["items"])
    assert r["decision"]["semantic_filter"] == "run"


def test_fingerprint_matches_itself():
    item = hb.fingerprint_item("Brauche ich morgen einen Schirm?")
    lex = hb.Lexical(0.8)
    lex.add_fingerprint("near_sealed", item)
    assert lex.near("brauche ich morgen einen schirm") == "near_sealed"
    assert lex.near("Wann kommt die Lieferung der Fliesen?") is None
