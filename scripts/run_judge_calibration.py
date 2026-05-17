"""Run the faithfulness judge against the labelled eval set and emit a
side-by-side calibration sheet for human spot-checks.

Usage:
    ANTHROPIC_API_KEY=sk-... \
    python scripts/run_judge_calibration.py \
        --collection quant-finance \
        --labels src/eval/queries_bound.json \
        --output results/judge-calibration.json

Per Phase 3 slice 2 plan Task 1 step 5: spot-check 10 judge verdicts against
manual labels. If agreement < 80%, faithfulness metric ships with that caveat
in the eval report rather than as a headline number.

The script:
  1. Loads the labelled queries.
  2. For each non-refusal query, runs retrieve() + synthesize_answer() to
     produce a cited answer.
  3. Runs score_faithfulness() to get the judge verdicts.
  4. Writes a JSON sheet pairing every verdict with the cited chunk content
     so a human can read both and either confirm or override.

Cost: ~$0.05 per full run (Haiku judge + Sonnet answer, ~8 queries).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.answer import synthesize_answer
from src.config import load_config
from src.eval.faithfulness import score_faithfulness
from src.pipeline.retriever import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the judge for human calibration")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--labels", type=Path, default=Path("src/eval/queries_bound.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to feed the answer layer")
    args = parser.parse_args()

    cfg = load_config()
    db = cfg.collection_db_path(args.collection)
    if not db.exists():
        print(f"collection {args.collection!r} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.labels) as f:
        queries = json.load(f)

    sheet: list[dict] = []
    for q in queries:
        # Skip refusal-expected queries — they have no citations to judge.
        if q.get("category") == "refusal_expected" or not q.get("relevant_docids"):
            continue

        results = retrieve(q["query"], db, cfg)[: args.k]
        answered = synthesize_answer(q["query"], results)

        score = score_faithfulness(q["query"], answered.answer, answered.citations)

        sheet.append({
            "id": q["id"],
            "query": q["query"],
            "answer": answered.answer,
            "judge_overall": score.overall,
            "judge_model": score.judge_model,
            "verdicts": [
                {
                    "docid": v.docid,
                    "supported_by_judge": v.supported,
                    "judge_reason": v.reason,
                    "manual_label": None,  # Fill in by hand: true/false
                    "chunk_snippet": next(
                        (c.snippet for c in answered.citations if c.docid == v.docid),
                        "",
                    ),
                }
                for v in score.verdicts
            ],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(sheet, f, indent=2)

    n_verdicts = sum(len(item["verdicts"]) for item in sheet)
    print(f"Wrote {len(sheet)} queries with {n_verdicts} verdicts to {args.output}")
    print("Next step: open the file, fill in `manual_label` per verdict, then run:")
    print(f"  python scripts/score_judge_calibration.py --sheet {args.output}")


if __name__ == "__main__":
    main()
