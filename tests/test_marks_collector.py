"""Guards for the marking-sheet collector.

Two failures are worth preventing, and both were live in the first version of the tool:
the judge's own verdicts sit at the end of the same file as another numbered table, so
counting them produces a confident and wrong set of marks; and a partially filled sheet
would silently reduce n, which is the one quantity the calibration exists to report.
"""

from __future__ import annotations

import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from t7h_collect_marks import read_rows  # noqa: E402

SHEET = """# Calibration sheet

| # | question | reference | answer | your mark |
| --- | --- | --- | --- | --- |
| 1 | q1 | a1 | b1 | Y |
| 2 | q2 | a2 | b2 | N |
| 3 | q3 | a3 | b3 |  |

## The judge's verdicts (read after marking)

| # | verdict | reason |
| --- | --- | --- |
| 1 | correct | it says the same thing |
| 2 | wrong | it says something else |
| 3 | correct | it matches |
"""


def test_only_the_marking_table_is_read():
    rows = read_rows(SHEET)
    assert [number for number, _mark in rows] == [1, 2, 3]
    assert [mark for _number, mark in rows] == ["Y", "N", ""]


def test_blank_rows_are_reported_not_guessed():
    rows = read_rows(SHEET)
    blank = [number for number, mark in rows if not mark]
    assert blank == [3]
    # A row that is blank must not be silently treated as either answer.
    assert all(mark in ("Y", "N", "") for _number, mark in rows)


def test_real_sheet_has_the_expected_number_of_rows():
    sheet = ROOT / "runs" / "judge_calibration_round2.md"
    if not sheet.is_file():
        pytest.skip("marking sheet not present")
    rows = read_rows(sheet.read_text(encoding="utf-8", errors="replace"))
    assert len(rows) == 24, "the round-2 sheet is 24 rows; more means the judge's table leaked in"
    assert len(set(number for number, _mark in rows)) == len(rows)
