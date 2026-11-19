"""Entrypoint for the GTA VI trophy rarity forecaster.

Reads every <id>.in in the input directory and writes <id>.out to the
output directory. Run by the grader as `python3 -P -s /bot/main.py`
(PYTHONHASHSEED=0) in python:3.13-slim, standard library only, no network.
Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for
/input and /output:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

The forecasting itself lives in model.py, imported below; -P keeps this
file's directory off sys.path, so the path line comes first.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model

OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def validate_outputs():
    """Fail loudly if any written .out is malformed.

    The grader scores an invalid case as if every quantile equalled the
    last-read rate, so a malformed row is a silent loss. Check each output
    is one row per trophy, five finite numbers in [0, 100], non-decreasing,
    and under the 65,536-byte limit; exit non-zero on the first failure so
    the problem is visible instead of scored.
    """
    for path in sorted(OUTPUT_DIR.glob("*.out")):
        if path.stat().st_size > 65536:
            raise SystemExit(f"{path.name}: over the 65,536-byte limit")
        data = json.loads(path.read_text())
        rows = data["trophies"]
        if len(rows) != 55:
            raise SystemExit(f"{path.name}: {len(rows)} rows, expected 55")
        for row in rows:
            q = row["q"]
            if len(q) != 5 or any(not (0 <= v <= 100) for v in q) or q != sorted(q):
                raise SystemExit(f"{path.name}: malformed quantiles for trophy {row['id']}")


if __name__ == "__main__":
    model.main()
    validate_outputs()
