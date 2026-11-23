"""tools/holdout_build.py on invented sentences: sure blind agreement, filters, quotas per round, no text out."""

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
NOTHING = {"training": [], "holdout": [], "probe": [], "template": []}
# Small quotas: decision 2 per intent from round 2 and 3 from round 1, unknown 3 and 4; shadow 1 and 2, unknown 1 and 2.
SMALL = {"decision": ("2:2,1:3", "2:3,1:4"), "shadow": ("2:1,1:2", "2:1,1:2")}


def sentence(round_name: str, intent: str, i: int) -> str:
    """Invented stand-ins, far apart from each other: four hex words from a hash."""
    h = hashlib.sha256(f"{round_name}/{intent}/{i}".encode()).hexdigest()
    return " ".join(h[k:k + 6] for k in range(0, 24, 6))


def rows(round_name: str, per_intent: int, unknown: int, writers: str) -> list[dict]:
    return [{"id": f"r{round_name}-{intent}-{i}", "writer": writers[i % len(writers)],
             "text": sentence(round_name, intent, i), "expected_intent": intent}
            for intent in INTENTS for i in range(unknown if intent == "unknown" else per_intent)]


def write_round(folder: Path, cand: str, la: str, lb: str, data: list[dict], change: dict | None = None) -> None:
    """change: candidate id -> ("a"|"b", intent or None for missing, sure)."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / cand).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
    for side, name in (("a", la), ("b", lb)):
        lines = []
        for r in data:
            intent, sure = r["expected_intent"], True
            c = (change or {}).get(r["id"])
            if c and c[0] == side:
                intent, sure = c[1], c[2]
                if intent is None:
                    continue
            lines.append(json.dumps({"id": r["id"], "intent": intent, "sicher": sure}, ensure_ascii=False))
        (folder / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def folder_with(tmp_path: Path, r1: list[dict], r2: list[dict], change: dict | None = None) -> Path:
    f = tmp_path / "in"
    write_round(f, "kandidaten.jsonl", "labels_1.jsonl", "labels_2.jsonl", r1, change)
    write_round(f, "kandidaten_hart.jsonl", "labels_hart_1.jsonl", "labels_hart_2.jsonl", r2, change)
    return f


def test_quotas_per_round_deterministic_and_disjoint(tmp_path):
    f = folder_with(tmp_path, rows("1", 6, 8, "ABCD"), rows("2", 4, 5, "FG"))
    a = hb.build(f, None, quotas=SMALL, semantic=False, known=NOTHING)
    assert a == hb.build(f, None, quotas=SMALL, semantic=False, known=NOTHING)
    assert a["report"]["complete"]
    dec, sha = a["decision"]["items"], a["shadow"]["items"]
    assert len(dec) == 24 * 5 + 7 and len(sha) == 24 * 3 + 3
    assert not {i["text"] for i in dec} & {i["text"] for i in sha}
    assert a["report"]["taken_per_round"]["decision"] == {"round 2": 24 * 2 + 3, "round 1": 24 * 3 + 4}
    assert {i["set"] for i in dec} == {"decision"} and {i["origin"] for i in sha} == {"round 1", "round 2"}
    assert set(dec[0]) == {"id", "set", "text", "expected_intent", "origin"}
    assert "same language model" in a["decision"]["description"]


def test_writers_take_turns_within_a_round(tmp_path):
    f = folder_with(tmp_path, rows("1", 8, 8, "ABCD"), rows("2", 4, 5, "FG"))
    r = hb.build(f, None, quotas={"decision": ("2:2,1:4", "2:2,1:4"), "shadow": ("2:1,1:2", "2:1,1:2")},
                 semantic=False, known=NOTHING)
    writer = {s["text"]: s["writer"] for s in rows("1", 8, 8, "ABCD")}
    for intent in INTENTS:
        got = [writer[i["text"]] for i in r["decision"]["items"]
               if i["expected_intent"] == intent and i["origin"] == "round 1"]
        assert sorted(got) == ["A", "B", "C", "D"], intent


def test_a_short_round_2_is_filled_from_round_1_and_reported(tmp_path):
    r2 = [r for r in rows("2", 4, 5, "FG")
          if not (r["expected_intent"] == "create_goal" and r["id"].endswith(("-0", "-1")))]
    f = folder_with(tmp_path, rows("1", 8, 8, "ABCD"), r2)
    r = hb.build(f, None, quotas=SMALL, semantic=False, known=NOTHING)
    goal = r["report"]["per_intent"]["create_goal"]
    assert r["report"]["complete"] and goal["taken"]["decision"] == {"round 2": 2, "round 1": 3}
    assert goal["taken"]["shadow"] == {"round 2": 0, "round 1": 3}
    assert goal["filled"] == {"shadow: round 1 for earlier rounds": 1}


def test_only_sure_agreement_with_the_writer(tmp_path):
    r1 = rows("1", 8, 8, "ABCD")
    change = {r1[0]["id"]: ("a", "tasks_open", True), r1[1]["id"]: ("b", None, True),
              r1[2]["id"]: ("b", r1[2]["expected_intent"], False)}
    f = folder_with(tmp_path, r1, rows("2", 4, 5, "FG"), change)
    r = hb.build(f, None, quotas=SMALL, semantic=False, known=NOTHING)
    d = r["report"]["dropped"]
    assert d["blind_disagrees"] == 1 and d["blind_missing"] == 1 and d["blind_unsure"] == 1
    kept = {i["text"] for i in r["decision"]["items"] + r["shadow"]["items"]}
    assert not kept & {r1[0]["text"], r1[1]["text"], r1[2]["text"]}


def test_lexical_filter_against_known_sentences_and_each_other(tmp_path):
    r1 = rows("1", 8, 8, "ABCD")
    twin = r1[12]
    r2 = rows("2", 4, 5, "FG") + [{**twin, "id": "x1", "text": twin["text"] + "!"},
                                  {**twin, "id": "x2", "text": twin["text"] + " bitte"}]
    f = folder_with(tmp_path, r1, r2)
    known = {"training": [r1[0]["text"].upper()], "holdout": [r1[4]["text"] + "?"], "probe": [r1[5]["text"]],
             "template": [r1[6]["text"]]}
    r = hb.build(f, None, quotas=SMALL, semantic=False, known=known)
    d = r["report"]["dropped"]
    assert (d["near_training"], d["near_holdout"], d["near_probe"], d["near_template"]) == (1, 1, 1, 1)
    assert d["duplicate"] == 1 and d["near_candidate"] == 1


def test_no_text_in_report_or_fingerprint_and_nothing_into_the_repository(tmp_path, capsys):
    r1, r2 = rows("1", 6, 8, "ABCD"), rows("2", 4, 5, "FG")
    f = folder_with(tmp_path, r1, r2)
    assert hb.main(["--candidates", str(f), "--out", str(ROOT / "data" / "x"), "--semantic", "skip"]) == 2
    assert hb.main(["--candidates", str(ROOT / "data"), "--out", str(tmp_path / "o"), "--semantic", "skip"]) == 2
    out = tmp_path / "out"
    code = hb.main(["--candidates", str(f), "--out", str(out), "--semantic", "skip", "--decision", "2:2,1:3",
                    "--decision-unknown", "2:3,1:4", "--shadow", "2:1,1:2", "--shadow-unknown", "2:1,1:2"])
    printed = capsys.readouterr()
    assert code in (0, 3)
    assert not (out / "fingerprint.json").exists()
    for name in ("report.json",):
        body = (out / name).read_text(encoding="utf-8")
        for r in r1 + r2:
            assert r["text"] not in body and r["text"] not in printed.out + printed.err


def test_semantic_filter_uses_the_model_it_is_given(tmp_path):
    class Fake:
        """Says a sentence means the same as a training sentence when it starts with a digit."""

        def __init__(self):
            self.kinds = []

        def add(self, kind, texts):
            self.kinds += [kind for _ in texts]

        def add_vectors(self, kind, vectors):
            assert not vectors

        def near(self, text):
            return "semantic_training" if "semantic_training" in self.kinds and text[0].isdigit() else None

    f = folder_with(tmp_path, rows("1", 6, 8, "ABCD"), rows("2", 4, 5, "FG"))
    r = hb.build(f, None, quotas=SMALL, semantic=True, semantic_model=Fake(),
                 known={**NOTHING, "training": ["irgendein Satz"]})
    assert r["report"]["dropped"]["semantic_training"] > 0
    assert all(not i["text"][0].isdigit() for i in r["decision"]["items"] + r["shadow"]["items"])
    assert r["decision"]["semantic_filter"] == "run"


def test_reserve_follows_both_sets_per_intent_and_round(tmp_path):
    f = folder_with(tmp_path, rows("1", 8, 9, "ABCD"), rows("2", 5, 6, "FG"))
    r = hb.build(f, None, quotas=SMALL, reserve=(2, 1), semantic=False, known=NOTHING)
    spare = r["reserve"]["items"]
    used = {i["text"] for i in r["decision"]["items"] + r["shadow"]["items"]}
    assert spare and not used & {i["text"] for i in spare} and {i["set"] for i in spare} == {"reserve"}
    goal = [i for i in spare if i["expected_intent"] == "create_goal"]
    assert sorted(i["origin"] for i in goal) == ["round 1", "round 1", "round 2", "round 2"]
    assert len([i for i in spare if i["expected_intent"] == "unknown"]) == 2
    assert r["decision"]["semantic_filter"].startswith("NOT RUN (local")
    assert r["report"]["checked_against"]["sealed"].startswith("NOT CHECKED")


def test_excluded_ids_are_struck_first_and_named_in_the_sets(tmp_path):
    r1 = rows("1", 8, 9, "ABCD")
    f = folder_with(tmp_path, r1, rows("2", 5, 6, "FG"))
    exclude = {"schema": 1, "geprueft_gegen": "sealed", "ausschluss_ids": [r1[0]["id"], r1[1]["id"], "unbekannt"]}
    r = hb.build(f, exclude, quotas=SMALL, semantic=False, known=NOTHING)
    kept = {i["text"] for i in r["decision"]["items"] + r["shadow"]["items"] + r["reserve"]["items"]}
    assert r["report"]["dropped"]["excluded_sealed"] == 2 and not kept & {r1[0]["text"], r1[1]["text"]}
    assert "by the auditor, 2 excluded" in r["decision"]["description"]
    assert r["report"]["checked_against"]["sealed"].endswith("2 excluded")
    assert "ausschluss_ids" not in r["report"]["sealed_check"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": 1}), encoding="utf-8")
    assert hb.main(["--candidates", str(f), "--out", str(tmp_path / "o"), "--semantic", "skip",
                    "--exclude-ids", str(bad)]) == 2
