# ADR-008: South African Legislation Demo Corpus

**Date:** 2026-05-17
**Status:** Accepted (proof-of-concept)
**Slice:** Phase 3 slice 5 — backlog item #22
**Builds on:** ADR-006 (answer layer), ADR-007 (deploy target)

## Context

`docs/sales-demo.md` targets compliance / legal / ops buyers at mid-sized SA firms. Through slice 2-4 the rehearsed demo corpus was `quant-finance` — eight trading and ML books — which retrieves cleanly but mismatches the buyer narrative. Walking a prospect into a paid call against a "what is the Sharpe ratio?" demo when their pain is "we can't find clauses in our policy manuals" is a credibility hit.

Slice 5 ships a buyer-aligned demo corpus: a small curated set of South African acts published by the government printer (`gov.za`). Source documents are public-domain and freely re-distributable.

## Decision

A curated `sa-legislation` collection of 4 acts:

| Act | Source | Pages | Chunks |
|---|---|---|---|
| Protection of Personal Information Act 4 of 2013 (POPIA) | gov.za | 76 | 48 |
| Companies Act 71 of 2008 | gov.za | 197 | 166 |
| Consumer Protection Act 68 of 2008 | gov.za | 94 | 70 |
| Labour Relations Act 66 of 1995 | gov.za | 222 | 117 |
| **TOTAL** | | **589** | **401** |

Sourcing automated via `scripts/fetch_sa_legislation.py`. The script holds the canonical URL list; operators can extend it. Three additional acts were drafted (National Credit Act, FICA, PAIA) but the URLs in the initial guess returned 404 — sourcing those is a slice 6 follow-up that needs a search-driven URL discovery step.

## Why this set (not larger, not different)

- **POPIA** is the #1 compliance topic in SA mid-market sales conversations since the 2021 enforcement date. Walking into a demo without it is malpractice.
- **Companies Act** is universally relevant — every prospect's lawyer reads it weekly.
- **Consumer Protection Act** covers B2C buyers (retailers, services).
- **Labour Relations Act** covers HR/operations buyers; retrenchment + unfair dismissal are the queries every HR head asks.

Together this set spans the four most common compliance domains and gives the demo room to show the system swapping between source documents on a single buyer's question.

## Operational rules

1. **Public-domain sources only.** Every act in the corpus comes from the government printer's published PDFs. Re-distribution is permitted.
2. **No prospect data in this collection.** This is the rehearsed-demo corpus, separate from prospect-specific collections built by `scripts/swap_corpus.py`.
3. **Stays in git as URLs, not bytes.** `data/sa_legislation/` is gitignored (binary PDFs); the fetcher script is the canonical source of truth. Operators re-download via `python scripts/fetch_sa_legislation.py` and ingest via `python scripts/swap_corpus.py --src data/sa_legislation --collection-name sa-legislation`.
4. **Polite scraping.** The fetcher pauses 1 second between requests and uses a self-identifying User-Agent.

## Baseline

`results/sa-legislation-baseline.json` (slice 5):

| Metric | Value | Notes |
|---|---|---|
| avg_recall@5 | **1.000** | 7 labelled queries (3 refusal-expected excluded from averaging) |
| avg_mrr@10 | 1.000 | Top-ranked chunk hits a labelled source for every query |
| avg_ndcg@10 | 0.951 | |
| n_queries (labelled) | 7 | factual_lookup (4), definition (1), multi_doc_synthesis (2) |
| n_queries (refusal_expected) | 3 | |

The 1.000 recall@5 reflects two things, honestly:

1. **The corpus is small and topically clean** — each query targets one act, so retrieval has less ambiguity than the 3,138-chunk quant-finance index.
2. **Page-overlap matching is forgiving here** because the chunker reports `page_start = 1` for every chunk in these acts, with `page_end` varying. A label at pages `[1, 147]` overlaps any chunk in the same act. This is a known chunker quirk surfaced by slice 5; ADR-002's atomic chunking decision should be revisited if precision-at-page becomes a buyer ask.

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
