# Phase 3 — Slice 3 Backlog

Pulled forward from PR #1 code-review and slice 2 discoveries. Each item is
a candidate for the slice 3 work plan, not a guaranteed deliverable.

## From PR #1 code-review (P2 + P3 items)

1. **Pin the Claude answer model to a dated id.**
   `src/answer.py:31` uses `claude-3-5-sonnet-latest`. Eval baselines drift
   silently when Anthropic ships a new minor; pin to a dated alias for any
   eval-bearing release. (P2)

2. **Map Anthropic SDK errors to HTTP status codes explicitly.**
   `src/api/server.py:97` only catches `EnvironmentError`. `anthropic.APIError`,
   `RateLimitError`, `APITimeoutError`, and `BadRequestError` should map to
   502/503/504/400 with structured log entries instead of unhandled 500s. (P2)

3. **Expand the observability emit point.**
   `src/observability.py` currently logs `request_id`, `collection`, `query`
   (truncated), `k`, `status`, `latency_ms`. Add: `caller_id` (Cloudflare
   Access JWT subject), `answer_len`, `refused: bool`, `citations_count`,
   and a retrieve-vs-synth latency split. Investigator a month out should be
   able to tell whether the system refused or answered without re-running. (P2)

4. **Drop the vacuous happy-path mock in `tests/test_api_server.py`.**
   `test_query_happy_path` mocks both `retrieve()` (returns `[]`) and
   `synthesize_answer()`. With the real short-circuit path on empty retrieval,
   the test passes whether the wiring is correct or not. Either pass non-empty
   `retrieve()` results or drop the `synthesize_answer` mock. (P2)

5. **Add a model-emits-refusal test.**
   `tests/test_answer.py` covers the empty-retrieval short-circuit refusal
   path but not the path where the model itself returns the canonical refusal
   string. The "refusal is the feature" claim in the sales doc isn't covered
   by tests. (P2)

6. **Make `_DOC_TAG_RE` tolerant of model variants.**
   `[doc-1, doc-2]`, `[Doc-1]`, `[doc 1]`, `[doc-1][doc-2]` runs. Add fixture-based
   property tests before the citation UI builds further on the regex. (P2)

7. **Split dev deps out of `requirements.txt`.**
   `pytest`, `pytest-asyncio`, `httpx` are dev-only and ship to the production
   container today. Move to `requirements-dev.txt`. (P2)

8. **Pick an actual lower bound for `pypdfium2`.**
   `pypdfium2>=0.0.0` is a no-op pin. (P2)

9. **Fix the demo shell snippet in `docs/sales-demo.md`.**
   The line `ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY uvicorn ... &` then
   `sleep 2 && curl ...` is racey and assumes the var is exported. Replace
   with a robust block or remove. (P3)

10. **Reconcile ADR-006 filename pattern with ADRs 001/002/003.**
    The other ADRs use `NNN-…` (no `ADR-` prefix); ADR-006 and now ADR-007
    use the prefix. Pick one. (P3)

## From slice 2 discoveries

11. **Replace stale-on-reingest docid labels with stable identifiers.**
    Slice 2 found that 12/15 labelled docids in `src/eval/queries_bound.json`
    didn't exist in the current index — chunk uuids regenerate on every
    re-ingest. Switch the labelling scheme to `{source_pdf, page_range}` so
    labels survive a re-ingest. Convert at eval time via a lookup. (P1 for
    slice 3)

12. **Expand the labelled eval set toward 30-40 queries.**
    Slice 2 shipped with 9 queries (8 scored, 1 refusal-expected). The parent
    plan called for 50; corpus coverage caps the realistic count around 30-40
    (RSI, Bollinger, MACD have thin coverage). Expand with queries the corpus
    actually supports across the 5 categories, then bump the recall@5 target
    to 0.85 on the new set. (P1 for slice 3)

13. **Wire the eval harness into CI.**
    `docs/sales-demo.md` claims "we benchmark on every release"; today this
    is a manual `python -m src.eval.run_eval` invocation. Wire it to GitHub
    Actions on every PR to `main` with a 5%-drop regression gate. (P1)

14. **Per-PDF ingest parallelism.**
    Slice 2 measured ~80 sec per PDF on M2 (single-threaded). A 50-PDF
    corpus takes ~67 min today. With per-PDF parallelism we should land
    under ~20 min on the same hardware. (P2)

15. **OCR for scanned PDFs.**
    `src/pipeline/router.is_scanned()` already detects scanned PDFs and
    skips them with a warning. Many prospect corpora contain scanned legal
    documents; add a `--ocr` flag to `scripts/swap_corpus.py` that pipes
    detected scans through Tesseract before ingest. (P2)

16. **Live judge calibration result.**
    `scripts/run_judge_calibration.py` is in place but needs to be run once
    with a real `ANTHROPIC_API_KEY` to produce baseline agreement numbers
    against ~10 hand-spot-checked verdicts. If agreement < 80%, document the
    faithfulness caveat in the released baseline JSON. (P1)

17. **Light theme for the citation UI.**
    `src/api/static/styles.css` is dark-only. Light-theme parity adds visual
    flexibility for prospects on light-themed decks. (P3)

18. **PDF.js viewer instead of the browser iframe.**
    The current `<iframe>` renders the PDF using the browser's built-in
    viewer, which works on Chrome and Safari but is inconsistent on Firefox
    and Edge mobile. PDF.js gives consistent rendering, programmatic page
    navigation, and search highlighting. (P3)

19. **ntfy cron health-pinger across all demo systems.**
    Per the parent plan's Phases 2–6 framing item 6: a single cron-driven
    script in `infra/homelab-infra/` that pings all five demos' /health every
    30 minutes and alerts via ntfy on failure. Slice 2 only built the demo
    `/health` endpoint; the pinger lives at the homelab layer. (P2)

20. **Post-session scrub automation via cron.**
    `scripts/scrub_corpus.py` exists as a manual command. Add a cron entry
    on the host that scrubs prospect collections older than 7 days
    automatically, audit-logged. (P2)

21. **Ollama-on-Hetzner CX22 sizing check.**
    CX22 has 4 GB RAM. `nomic-embed-text` (~270 MB) + FastAPI + Ollama
    daemon plus buffers fit, but margin is thin. If embedding-during-query
    becomes a path (today's design re-embeds only at ingest), revisit the
    sizing. (P3)

22. **South African legislation corpus.**
    Slice 2 shipped against the quant-finance corpus. The sales narrative
    targets compliance/legal/ops buyers; demoing against a legislation
    corpus aligns the narrative. Source ~30 PDFs from gov.za, ingest, relabel
    a category-balanced eval set. (P1 — gate for first paid legal-sector
    engagement.)
