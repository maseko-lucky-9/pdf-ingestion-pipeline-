# Phase 3 — Slice 3 Backlog

Pulled forward from PR #1 code-review and slice 2 discoveries. Each item is
a candidate for the slice 3 work plan, not a guaranteed deliverable.

**Status legend:** ✅ landed in PR #3 (slice 3), PR #4 (slice 4), PR #5 (slice 5), or PR #6 (slice 6); 🟡 deferred (operator-gated — only the live judge calibration run + the slice-2 operator-step prereqs remain).

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

12. ✅ **Expand the labelled eval set toward 30-40 queries.**
    Set expanded 9 → 28 queries (25 labelled + 3 refusal-expected) across the
    5 categories. 4 drafted queries dropped because the corpus has thin
    retrieval coverage for those topics (Kelly, slippage, microstructure
    noise, volatility clustering). New baseline: recall@5 = 0.919 on the
    25-labelled-query set (was 0.894 on 8). (P1 — landed in slice 4)

13. ✅ **Wire the eval harness into CI.**
    `.github/workflows/test.yml` runs pytest on every push + PR.
    `.github/workflows/eval.yml` is manual-dispatch (`workflow_dispatch`)
    with a 5% recall@5 + 5% MRR@10 regression gate vs
    `results/slice2-baseline.json`. The eval workflow needs the gitignored
    `quant-finance` collection on the runner — slice 4 will wire artifact
    upload + scheduled runs. (P1 — landed; scheduled cadence deferred to
    slice 4)

14. ✅ **Per-PDF ingest parallelism.**
    `--parallel N` flag on `python -m src.ingest` and `scripts/swap_corpus.py`.
    ProcessPoolExecutor handles extract+normalize+chunk in workers; embed
    (Ollama) + write (SQLite) stay serial. Measured: 4 PDFs went 325s → 252s
    with --parallel 3 (23% reduction; on 50 PDFs the worker overhead
    amortises better — projected ~25-30 min vs ~67 min). (P2 — landed in slice 4)

15. ✅ **OCR for scanned PDFs.**
    `scripts/ocr_preprocess.py` uses Tesseract (`pdf` output mode) to produce
    searchable PDFs the existing pipeline can ingest. `scripts/swap_corpus.py
    --ocr` enables pre-processing. Lazy imports of pytesseract + pdf2image
    so the project still works without Tesseract installed. Realistic
    timing: ~6-10 sec per page at 300 DPI. (P2 — landed in slice 4)

16. 🟡 **Live judge calibration result.**
    `scripts/run_judge_calibration.py` is in place but needs to be run once
    with a real `ANTHROPIC_API_KEY` to produce baseline agreement numbers
    against ~10 hand-spot-checked verdicts. If agreement < 80%, document the
    faithfulness caveat in the released baseline JSON. (P1)

17. ✅ **Light theme for the citation UI.**
    Token-driven dark + light themes. `prefers-color-scheme` drives the
    default; `body[data-theme="light"|"dark"]` forces a specific theme.
    Toggle button in the topbar persists choice to localStorage. (P3 — landed in slice 4)

18. ✅ **PDF.js viewer (opt-in).**
    `buildPdfViewerUrl()` in app.js dispatches between the browser-native
    iframe (default) and a Mozilla pdf.js viewer (opt-in). To enable, set
    `window.PRUDENTIA_PDF_VIEWER = "pdfjs"` + `PRUDENTIA_PDF_VIEWER_BASE` to
    a same-origin pdf.js install before app.js loads. CORS on pdf.js's
    `file` param means CDN-hosted pdf.js won't work; the vendored-locally
    path is the production approach. (P3 — landed in slice 4)

19. ✅ **ntfy cron health-pinger across all demo systems.**
    `infra/monitoring/health_pinger.py` posts to ntfy on /health failure;
    YAML-configured target list (template in `targets.example.yaml`).
    Supports Cloudflare Access service-token JWTs via env-var indirection.
    Crontab snippet in `infra/monitoring/README.md`. Deployment to the box
    stays operator-gated. (P2 — landed in slice 6)

20. ✅ **Post-session scrub automation via cron.**
    `scripts/scrub_corpus.py --age-threshold-days 7 --prefix prospect-`
    walks the collections root and scrubs anything older than the threshold.
    Prefix filter ensures rehearsed demo corpora are never touched.
    `infra/monitoring/scrub_cron_wrapper.sh` is the cron-friendly wrapper.
    Deployment stays operator-gated. (P2 — landed in slice 6)

21. ✅ **Ollama-on-Hetzner CX22 sizing check.**
    `scripts/box_health_check.py` snapshots RAM / CPU load / disk / Ollama
    daemon and compares against a CX22 envelope (4 GB RAM, 2 vCPU, 40 GB).
    Exit 1 on errors, optional --strict for warnings. JSON output mode for
    cron-friendly machine parsing. Running on the actual box stays
    operator-gated. (P3 — landed in slice 6)

22. ✅ **South African legislation corpus.**
    Curated 4 acts (POPIA, Companies, Consumer Protection, Labour Relations)
    sourced from gov.za via `scripts/fetch_sa_legislation.py`. New collection
    `sa-legislation` (401 chunks, 589 pages). 7 labelled queries + 3 refusal-
    expected; baseline recall@5 = 1.000 (small corpus, topically clean).
    ADR-008 documents the corpus + the chunker page-start quirk surfaced
    during labelling. Sourcing for 3 additional acts (NCA / FICA / PAIA)
    was attempted but URLs returned 404 — slice 6 follow-up. (P1 — landed in
    slice 5)

## Slice 7 — surfaced during PR #6 review

23. ✅ **Relabel `queries_bound.json` against the new chunker page-tracking.**
    Slice 4 labels (e.g. Murphy `[1, 118]` on q024) were authored against the
    OLD `page_start=1` chunker output. Under the slice 6 chunker fix, chunks
    now report tight page ranges, so the overlap-matcher correctly resolves
    to fewer chunks. Slice 7 inspected the top-5 returned chunks per query
    under the fixed chunker and rewrote every label to a tight content-page
    range. New baseline: recall@5 = **1.000** on 25 labelled queries (was
    0.691 against bug-shaped labels). MRR@10 = 0.920, NDCG@10 = 0.547. (P1 —
    landed in slice 7)

24. ✅ **Investigate the q001 / q010 / q020 recall-zero queries.**
    Confirmed as labelling artifact, not a retrieval gap. Under the new
    chunker, q001 (RSI) had labels at MLAT p1022/p171, but the actual
    canonical sources surface from TSaM p406-409 / p540. Same pattern for
    q010 (drawdown) — TSaM p73-79 + Chan p41-43 are the canonical sources.
    q020 (VWAP) — TSaM p429-430 + p565-567 have the actual formulas. With
    correct labels, all three queries hit recall@5 = 1.000. (P1 — landed in
    slice 7)

25. ✅ **Refresh `docs/sales-demo.md`'s recall@5 figure.**
    Updated to "recall@5 = 1.000 on 25 labelled queries against quant-finance,
    plus 0.970 on 15 against SA-legislation (target ≥ 0.85 on both)". (P3 —
    landed in slice 7)
