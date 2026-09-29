"""Regenerate ``paper/generated_tables.tex`` from ``runs/summary.json``.

The paper never contains hand-copied numbers: every quoted value is a macro
emitted here from the run's JSON, so a claim cannot drift away from the
measurement it came from.

Usage::

    python paper/build_tables.py                    # from runs/summary.json
    python paper/build_tables.py --bundle path.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default=str(ROOT / "runs" / "summary.json"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "generated_tables.tex"))
    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    if not bundle_path.is_file():
        print(f"missing bundle {bundle_path}; run `python experiments/run_all.py "
              "--collect-only` first", flush=True)
        return 1
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    from parammem.report import build_results_latex, stage_status

    text = build_results_latex(bundle)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    macros = text.count("\\newcommand")
    tables = text.count("\\begin{table}")
    print(f"wrote {out} : {macros} macros, {tables} tables, {len(text)} bytes",
          flush=True)
    bad = [stage for stage, status in stage_status(bundle) if status != "ok"]
    if bad:
        print(f"WARNING: stages not ok in this bundle: {bad} -- the paper will "
              "show 'n/a' for their numbers, which is intended", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
