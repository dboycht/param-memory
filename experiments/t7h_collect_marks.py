"""Turn the human marking sheet into the marks file the calibration merge reads.

The merge wants ``{"marks": "1Y 2N 3Y ..."}`` and the human wants a table they can fill
in. Transcribing between the two by hand is the step where a Y becomes an N, so it is done
here instead.

It refuses to write anything unless every row is marked: a partial sheet would silently
reduce n, and the whole point of the calibration is the size of n. It also reports the
agreement with the judge as a cross-check, but only after the marks are complete, so the
number cannot be used to decide what to write.

Usage::

    python experiments/t7h_collect_marks.py --sheet runs/judge_calibration_round2.md

    # check without writing
    python experiments/t7h_collect_marks.py --sheet runs/judge_calibration_round2.md --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROW = re.compile(r"^\|\s*(\d+)\s*\|(.*)\|\s*$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", required=True, help="the marked markdown table")
    ap.add_argument("--out", default="bench/judge_calibration_marks_round2.json")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def read_rows(text: str) -> list[tuple[int, str]]:
    """(row number, mark) for every data row; mark is '' when the cell is empty.

    Stops at the first heading after the sheet's title, because the judge's own verdicts
    are printed at the end of the same file as another numbered table -- and those cells
    are non-empty, so counting them would have produced a confident, wrong set of marks.
    """
    rows = []
    seen_title = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if seen_title:
                break
            seen_title = True
            continue
        match = ROW.match(stripped)
        if not match:
            continue
        cells = match.group(2).split("|")
        if len(cells) < 2:
            continue
        mark = cells[-1].strip().upper()
        rows.append((int(match.group(1)), mark))
    return rows


def main() -> int:
    args = parse_args()
    sheet = Path(args.sheet)
    if not sheet.is_file():
        print(f"  no sheet at {sheet}")
        return 1
    rows = read_rows(sheet.read_text(encoding="utf-8", errors="replace"))
    if not rows:
        print("  no table rows found in the sheet")
        return 1

    valid = {"Y": "Y", "N": "N", "对": "Y", "错": "N", "TRUE": "Y", "FALSE": "N",
             "YES": "Y", "NO": "N"}
    parsed = {number: valid.get(mark) for number, mark in rows}
    missing = [number for number, value in parsed.items() if value is None]
    print(f"  rows found     : {len(rows)}")
    print(f"  marked         : {len(rows) - len(missing)}")
    if missing:
        print(f"  still blank    : {missing}")
        print("\n  refusing to write: a partial sheet would silently reduce n, and the "
              "size of n is the point of the calibration")
        return 1

    marks = " ".join(f"{number}{parsed[number]}" for number, _mark in rows)
    yes = sum(1 for value in parsed.values() if value == "Y")
    print(f"  marks          : {marks}")
    print(f"  human yes      : {yes}/{len(rows)}")

    if args.dry_run:
        print("\n  dry run: nothing written")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "what": "Human marks for judge calibration, round 2",
        "how_collected": "collected from the marked sheet by "
                         "experiments/t7h_collect_marks.py, which refuses a partial sheet",
        "standard": "does the answer convey the same information as the reference",
        "marks": marks,
        "sheet": sheet.name,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  wrote {out.resolve()}")
    print("  next: python experiments/t7h_calibration.py "
          "--marks bench/judge_calibration_marks.json " + str(out) + " "
          "--payloads runs/judge_calibration.json runs/judge_calibration_round2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
