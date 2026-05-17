"""Score human-vs-judge agreement on a calibration sheet.

Reads the output of ``scripts/run_judge_calibration.py`` after a human has
filled in the ``manual_label`` field per verdict, and reports the agreement
fraction (with a 80% threshold marker per the slice 2 plan).

Usage:
    python scripts/score_judge_calibration.py --sheet results/judge-calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score judge vs manual labels")
    parser.add_argument("--sheet", type=Path, required=True)
    args = parser.parse_args()

    with open(args.sheet) as f:
        sheet = json.load(f)

    total = 0
    agreed = 0
    unlabeled = 0
    for item in sheet:
        for v in item["verdicts"]:
            if v.get("manual_label") is None:
                unlabeled += 1
                continue
            total += 1
            if bool(v["manual_label"]) == bool(v["supported_by_judge"]):
                agreed += 1

    if unlabeled:
        print(f"WARN: {unlabeled} verdicts have no manual_label set; skipping")
    if total == 0:
        print("No labelled verdicts to score.")
        sys.exit(1)

    agreement = agreed / total
    threshold = 0.80
    print(f"n_verdicts_compared: {total}")
    print(f"agreement: {agreement:.3f}")
    if agreement >= threshold:
        print(f"PASS: agreement >= {threshold}; faithfulness metric is shippable as a headline number.")
    else:
        print(f"FAIL: agreement < {threshold}; faithfulness ships with a caveat in baseline JSON.")
        sys.exit(2)


if __name__ == "__main__":
    main()
