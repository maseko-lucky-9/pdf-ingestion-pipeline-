# Phase 3 — Slice 3 Backlog

Pulled forward from PR #1 code-review and slice 2 discoveries. Each item is
a candidate for the slice 3 work plan, not a guaranteed deliverable.

**Status legend:** ✅ landed in PR #3 (slice 3 tech-debt batch); 🟡 deferred (operator-gated or research-gated); 🔵 deferred (UI polish bundle).

## From PR #1 code-review (P2 + P3 items)

1. ✅ **Pin the Claude answer model to a dated id.**
   `src/answer.py` was using `claude-3-5-sonnet-latest`. Pinned to
   `claude-3-5-sonnet-20241022`; judge model pinned to
   `claude-3-5-haiku-20241022`. (P2 — landed)

2. ✅ **Map Anthropic SDK errors to HTTP status codes explicitly.**
   `src/api/server.py` now catches `RateLimitError` (→ 429),
   `APITimeoutError` (→ 504), `APIConnectionError` (→ 502),
   `APIStatusError` (5xx → 502, 401/403 → 503 since upstream auth is
   our misconfig). 5 new tests. (P2 — landed)

3. ✅ **Expand the observability emit point.**
   `log_rag_request` now records `retrieve_ms`, `synth_ms`, `answer_len`,
   `citations_count`, `refused`. `caller_id` is the one field that still
   waits on Cloudflare Access being live so the JWT subject is available
   in the request context. (P2 — partial, caller_id deferred)

4. ✅ **Drop the vacuous happy-path mock.**
   `test_query_happy_path_returns_full_payload` now passes a real
   SearchResult through `retrieve()`, captures the kwargs reaching
   `synthesize_answer`, and asserts the wiring. (P2 — landed)

5. ✅ **Add a model-emits-refusal test.**
   `test_synthesize_answer_propagates_model_refusal` covers the case
   where the LLM (not the empty-retrieval short-circuit) emits the
   canonical refusal string. (P2 — landed)

6. ✅ **Make the citation regex tolerant of model variants.**
   Replaced the single `_DOC_TAG_RE` with a two-stage parser:
   `_DOC_TAG_BLOCK_RE` finds `[...doc...]` blocks (any case, any spacing),
   then `_DOC_INT_RE` extracts integers. 8 new tests covering
   `[Doc-1]`, `[DOC-1]`, `[doc 1]`, `[ doc-1 ]`, `[doc-1, doc-3]`,
   `[doc-1; doc-2]`, `[doc-1][doc-2]`, and non-citation brackets. (P2 — landed)

7. ✅ **Split dev deps out of `requirements.txt`.**
   `pytest`, `pytest-asyncio` moved to a new `requirements-dev.txt` that
   sources `requirements.txt`. Production container no longer ships test
   tooling. (P2 — landed)

8. ✅ **Pick an actual lower bound for `pypdfium2`.**
   Bumped to `>=4.30.0`. (P2 — landed)

9. ✅ **Fix the demo shell snippet in `docs/sales-demo.md`.**
   Rewritten to `set -a; source .env; set +a` for the env load, poll
   `/health` in an `until` loop, capture the uvicorn PID, kill on exit.
   No more `sleep 2 && curl` race. (P3 — landed)

10. ✅ **Reconcile ADR filename pattern.**
    Renamed `001/002/003` → `ADR-001/002/003`. All ADRs now use the
    `ADR-NNN-slug` convention. (P3 — landed)

## From slice 2 discoveries

11. ✅ **Replace stale-on-reingest docid labels with stable identifiers.**
    Labels migrated from chunk uuids to `{source_pdf, pages}` tuples.
    Resolution at eval time uses page-range overlap. Recall@5 metric
    redefined as label-coverage (distinct labelled sources hit in top-k)
    rather than uuid-set intersection. 12 new tests covering the
    overlap matcher and the hit counter. New baseline: recall@5 = 0.894
    (was 0.875 against uuid labels). (P1 — landed)

12. 🟡 **Expand the labelled eval set toward 30-40 queries.**
    Slice 2 shipped with 9 queries (8 scored, 1 refusal-expected). The parent
    plan called for 50; corpus coverage caps the realistic count around 30-40
    (RSI, Bollinger, MACD have thin coverage). Expand with queries the corpus
    actually supports across the 5 categories, then bump the recall@5 target
    to 0.85 on the new set. (P1; deferred — slow LLM-as-labeller work, best
    done as its own focused session)

13. ✅ **Wire the eval harness into CI.**
    `.github/workflows/test.yml` runs pytest on every push + PR.
    `.github/workflows/eval.yml` is manual-dispatch (`workflow_dispatch`)
    with a 5% recall@5 + 5% MRR@10 regression gate vs
    `results/slice2-baseline.json`. The eval workflow needs the gitignored
    `quant-finance` collection on the runner — slice 4 will wire artifact
    upload + scheduled runs. (P1 — landed; scheduled cadence deferred to
    slice 4)

14. 🟡 **Per-PDF ingest parallelism.**
    Slice 2 measured ~80 sec per PDF on M2 (single-threaded). A 50-PDF
    corpus takes ~67 min today. With per-PDF parallelism we should land
    under ~20 min on the same hardware. (P2)

15. 🟡 **OCR for scanned PDFs.**
    `src/pipeline/router.is_scanned()` already detects scanned PDFs and
    skips them with a warning. Many prospect corpora contain scanned legal
    documents; add a `--ocr` flag to `scripts/swap_corpus.py` that pipes
    detected scans through Tesseract before ingest. (P2)

16. 🟡 **Live judge calibration result.**
    `scripts/run_judge_calibration.py` is in place but needs to be run once
    with a real `ANTHROPIC_API_KEY` to produce baseline agreement numbers
    against ~10 hand-spot-checked verdicts. If agreement < 80%, document the
    faithfulness caveat in the released baseline JSON. (P1)

17. 🔵 **Light theme for the citation UI.**
    `src/api/static/styles.css` is dark-only. Light-theme parity adds visual
    flexibility for prospects on light-themed decks. (P3)

18. 🔵 **PDF.js viewer instead of the browser iframe.**
    The current `<iframe>` renders the PDF using the browser's built-in
    viewer, which works on Chrome and Safari but is inconsistent on Firefox
    and Edge mobile. PDF.js gives consistent rendering, programmatic page
    navigation, and search highlighting. (P3)

19. 🟡 **ntfy cron health-pinger across all demo systems.**
    Per the parent plan's Phases 2–6 framing item 6: a single cron-driven
    script in `infra/homelab-infra/` that pings all five demos' /health every
    30 minutes and alerts via ntfy on failure. Slice 2 only built the demo
    `/health` endpoint; the pinger lives at the homelab layer. (P2)

20. 🟡 **Post-session scrub automation via cron.**
    `scripts/scrub_corpus.py` exists as a manual command. Add a cron entry
    on the host that scrubs prospect collections older than 7 days
    automatically, audit-logged. (P2)

21. 🟡 **Ollama-on-Hetzner CX22 sizing check.**
    CX22 has 4 GB RAM. `nomic-embed-text` (~270 MB) + FastAPI + Ollama
    daemon plus buffers fit, but margin is thin. If embedding-during-query
    becomes a path (today's design re-embeds only at ingest), revisit the
    sizing. (P3)

22. 🟡 **South African legislation corpus.**
    Slice 2 shipped against the quant-finance corpus. The sales narrative
    targets compliance/legal/ops buyers; demoing against a legislation
    corpus aligns the narrative. Source ~30 PDFs from gov.za, ingest, relabel
    a category-balanced eval set. (P1 — gate for first paid legal-sector
    engagement.)
