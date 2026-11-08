import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from kaggle_status import status_word, table  # noqa: E402

LISTING = (
    "ref,title,author,lastRunTime,totalVotes\n"
    "me/old,Old,me,2026-11-01 10:00:00,0\n"
    "me/new,New,me,2026-11-07 22:00:00,0\n"
)


def test_status_word_reads_kaggle_output():
    assert status_word('me/new has status "KernelWorkerStatus.RUNNING"') == "running"
    assert status_word('me/old has status "complete"') == "complete"


def test_table_lists_newest_first_and_respects_limit():
    out = table(LISTING, lambda ref: f'{ref} has status "KernelWorkerStatus.QUEUED"', limit=1)
    assert "me/new" in out and "me/old" not in out
    assert "queued" in out


def test_table_without_kernels():
    assert "(no kernels)" in table("ref,title\n", lambda ref: "", limit=5)
