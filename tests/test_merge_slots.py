"""Every collected report must land in the merge slot it is supposed to.

Four times in one session a filename fragment matched a stage it had nothing to do with --
"multisession" caught the extraction run, "17b" caught the 1.7B multi-session run -- and each
time the payload was silently merged into the wrong slot, so a stage that had run looked
unrun and its macros rendered as n/a. Nothing failed; the numbers simply were not there.

This asserts the mapping explicitly, from the files that exist, so the next fragment that
matches too much fails a test instead of quietly emptying a table.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "runs" / "summary.json"

# filename -> the key it must appear under in the t7 section of the bundle
EXPECTED = {
    "t7i_multisession.json": "multi_session",
    "t7i_multisession_17b.json": "multi_session_large",
    "t7i_composition_demand.json": "composition_demand",
    "t7i_hosted_reader.json": "hosted_reader",
    "t7d_multisession.json": "multi_session_oracle",
    "t7j_collision_real.json": "collision_real",
    "longmemeval_evidence_shape.json": "evidence_shape",
}


def test_collected_reports_reach_their_slots():
    if not SUMMARY.is_file():
        pytest.skip("no collected bundle in this checkout")
    t7 = (json.loads(SUMMARY.read_text(encoding="utf-8")) or {}).get("t7") or {}
    if not t7:
        pytest.skip("the bundle carries no t7 section")
    missing_files = [name for name in EXPECTED if not (ROOT / "runs" / name).is_file()]
    wrong = []
    for name, key in EXPECTED.items():
        if name in missing_files:
            continue
        if key not in t7:
            wrong.append(f"{name} ran but is not merged under {key!r}")
    assert not wrong, (
        "a report on disk did not reach its merge slot:\n  " + "\n  ".join(wrong) +
        "\nThe usual cause is a branch matching a filename fragment that belongs to another "
        "stage; check the order of the t7 branch in experiments/run_all.py.")


def test_one_report_does_not_land_in_two_slots():
    """A payload merged twice means two branches matched the same filename.

    Compared by a hash of the whole payload, not by its note: the same experiment at two
    scales shares its opening sentence and was flagged as a duplicate when the comparison was
    looser, while its per-question records differ.
    """
    if not SUMMARY.is_file():
        pytest.skip("no collected bundle in this checkout")
    t7 = (json.loads(SUMMARY.read_text(encoding="utf-8")) or {}).get("t7") or {}
    seen: dict[str, list[str]] = {}
    for key, payload in t7.items():
        if isinstance(payload, dict) and "note" in payload:
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            seen.setdefault(digest, []).append(key)
    duplicated = {digest[:12]: keys for digest, keys in seen.items() if len(keys) > 1}
    assert not duplicated, f"the same payload reached several slots: {duplicated}"
