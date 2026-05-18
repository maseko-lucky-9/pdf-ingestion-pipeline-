# ADR-008: South African Legislation Demo Corpus

**Date:** 2026-05-17
**Status:** Accepted (proof-of-concept)
**Slice:** Phase 3 slice 5 — backlog item #22
**Builds on:** ADR-006 (answer layer), ADR-007 (deploy target)

## Context

`docs/sales-demo.md` targets compliance / legal / ops buyers at mid-sized SA firms. Through slice 2-4 the rehearsed demo corpus was `quant-finance` — eight trading and ML books — which retrieves cleanly but mismatches the buyer narrative. Walking a prospect into a paid call against a "what is the Sharpe ratio?" demo when their pain is "we can't find clauses in our policy manuals" is a credibility hit.

Slice 5 ships a buyer-aligned demo corpus: a small curated set of South African acts published by the government printer (`gov.za`). Source documents are public-domain and freely re-distributable.

## Decision

A curated `sa-legislation` collection of 7 acts (originally 4 in slice 5; the 3 missing acts joined in slice 6 after probing the `a<N>-<YY>0.pdf` URL pattern on gov.za):

| Act | Source | Pages |
|---|---|---|
| Protection of Personal Information Act 4 of 2013 (POPIA) | gov.za | 76 |
| Companies Act 71 of 2008 | gov.za | 197 |
| Consumer Protection Act 68 of 2008 | gov.za | 94 |
| Labour Relations Act 66 of 1995 | gov.za | 222 |
| National Credit Act 34 of 2005 (NCA) | gov.za | 116 |
| Financial Intelligence Centre Act 38 of 2001 (FICA) | gov.za | 28 |
| Promotion of Access to Information Act 2 of 2000 (PAIA) | gov.za | 45 |
| **TOTAL** | | **~778** |

Sourcing automated via `scripts/fetch_sa_legislation.py`. The script holds the canonical URL list; operators can extend it.

## Why this set (not larger, not different)

- **POPIA** is the #1 compliance topic in SA mid-market sales conversations since the 2021 enforcement date. Walking into a demo without it is malpractice.
- **Companies Act** is universally relevant — every prospect's lawyer reads it weekly.
- **Consumer Protection Act** covers B2C buyers (retailers, services).
- **Labour Relations Act** covers HR/operations buyers; retrenchment + unfair dismissal are the queries every HR head asks.
- **National Credit Act** + **FICA** together cover financial-services buyers — reckless credit, accountable-institution due diligence, customer-money-laundering checks.
- **PAIA** covers the data-access angle that pairs with POPIA (privacy in tandem with access-to-info rights).

Together this set spans the six most common compliance domains and gives the demo room to show the system swapping between source documents on a single buyer's question.

## Operational rules

1. **Public-domain sources only.** Every act in the corpus comes from the government printer's published PDFs. Re-distribution is permitted.
2. **No prospect data in this collection.** This is the rehearsed-demo corpus, separate from prospect-specific collections built by `scripts/swap_corpus.py`.
3. **Stays in git as URLs, not bytes.** `data/sa_legislation/` is gitignored (binary PDFs); the fetcher script is the canonical source of truth. Operators re-download via `python scripts/fetch_sa_legislation.py` and ingest via `python scripts/swap_corpus.py --src data/sa_legislation --collection-name sa-legislation`.
4. **Polite scraping.** The fetcher pauses 1 second between requests and uses a self-identifying User-Agent.

## Baseline

`results/sa-legislation-baseline.json` (slice 6, 7-act corpus + chunker fix + expanded query set):

| Metric | Value | Notes |
|---|---|---|
| avg_recall@5 | **0.970** | 11 labelled queries (4 refusal-expected excluded). |
| avg_mrr@10 | 0.909 | |
| avg_ndcg@10 | 0.665 | |
| n_queries (labelled) | 11 | factual_lookup (7), definition (2), multi_doc_synthesis (2) |
| n_queries (refusal_expected) | 4 | |

**Trajectory:** slice 5 reported 1.000 on 7 labelled queries; that number was inflated by the chunker `page_start=1` quirk (every chunk in a document reported the same start page, so page-overlap matching resolved labels too loosely). Slice 6 fixed the chunker AND added 3 more acts AND added 5 new labelled queries. After all three changes recall@5 settled at 0.970 — the honest number on a meaningfully larger corpus.

**Chunker fix detail.** `src/pipeline/chunker.py` now anchors each chunk's `page_start` at the page where the carried-overlap content actually lived (the `page_end` of the previous flush). Page ranges advance monotonically — POPIA chunk 7 reports pages (12, 13) instead of (1, 13). 3 new tests in `tests/test_chunker.py` lock this in.

The score is honest in the sense that the retrieval finds genuinely relevant chunks for each query — it's just measuring source-level coverage, not chapter-level precision.

## Trade-offs

**Positive:**
- Buyer narrative now matches the demo content.
- All sources are POPIA-aligned by definition (they ARE the regulation, not data subject to it).
- 4 acts, 589 pages, fit comfortably on the Hetzner CX22.

**Negative:**
- Smaller corpus = less impressive "look how it handles huge volume" beat.
- Chunker page-start quirk understates precision. Slice 6 follow-up.
- 3 acts (NCA, FICA, PAIA) still missing — need a different sourcing path (search, not URL-guess).

**Neutral:**
- Re-ingest is required on any update. Polite-scraping defaults make a refresh take ~15 seconds for 4 acts.

## Alternatives reconsidered

If any of these later become true, revisit:

| Trigger | Likely move |
|---|---|
| First paid call is in financial services | Add Banks Act + FAIS Act + FICA |
| First paid call is in healthcare | Add Health Professions Act + National Health Act |
| Prospect insists on their own corpus | `scripts/swap_corpus.py --src <prospect-folder>` (existing slice 2 path) |
| Corpus exceeds ~50 MB | Move to a separate `data/` mount on the Hetzner box; current setup fits in the docker volume |

## Consequences

**Reversible:** the entire corpus is rebuildable from the script. Nothing in git depends on the on-disk PDFs.

**Sales-doc update:** `docs/sales-demo.md` now references `sa-legislation` as the rehearsed corpus for compliance-buyer calls and falls back to `quant-finance` for trading/ML-buyer calls.

**Cost:** zero new infra cost — same collection storage layer.
