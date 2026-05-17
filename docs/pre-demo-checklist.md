# Pre-Demo Checklist — Production RAG

**Purpose:** the prep sequence Thulani runs before every booked sales demo.
**Audience:** internal (you, future-you, any collaborator running the system).
**Companion to:** `docs/sales-demo.md` (the live-script during the call).

> **Hard rule:** if you can't run through this checklist in under 60 min, the
> demo isn't ready — postpone the call rather than wing it. The recorded
> fallback exists, but it's worse than a delayed booking.

---

## T-60 min — Prospect-corpus swap (longest step)

Realistic timing on M2 MacBook (measured 2026-05-17): **~80 sec per PDF** for
mid-size native-text financial books. A 50-PDF / 200 MB prospect corpus takes
**~60-70 min** end-to-end. For smaller prospect packs (10-15 PDFs of policy
docs, contracts, manuals) budget **15-25 min**.

If the prospect has not sent docs yet, default to the rehearsed `quant-finance`
collection and use the pre-canned questions in this checklist.

```bash
# Activate the venv first
source .venv-mac/bin/activate || source .venv/bin/activate

# One-command swap. Nukes the named collection and re-ingests from scratch.
python scripts/swap_corpus.py \
    --src ~/Desktop/prospect-<name>-pdfs \
    --collection-name prospect-<name>

# Optional: add --smoke-query "..." to test the index in the same command.
```

The pipeline auto-detects scanned PDFs and skips them with a warning
(`src/pipeline/router.is_scanned`). If the skip count is significant, ask the
prospect for native-text versions before the call rather than discovering it
live.

After ingest, check `collections/prospect-<name>/ingest_errors.log` for the
skip list.

---

## T-15 min — Boot the deploy + verify auth

Hetzner CX22 is provisioned to stay running 24/7 (ADR-007); cold-boot is not in
the pre-demo path. Verify the running stack:

```bash
# From your laptop — Cloudflare Access JWT lives in your browser cookies.
# Open https://rag.<subdomain>/health in the browser first to refresh the
# Access cookie; then `curl` works.
curl -s https://rag.<subdomain>/health | jq

# Expected: {"status":"ok","collections":["quant-finance","prospect-<name>"]}
```

If Cloudflare Access redirects to login, refresh the JWT in the browser. If
`/health` 5xxs, fall back to the local laptop tunnel:

```bash
# Local fallback — same Cloudflare Access app, tunnel from your machine
cloudflared tunnel run prudentia-rag-local &
uvicorn src.api.server:app --port 8000 &
```

---

## T-5 min — Rehearse the opening question

Pick ONE question you know the corpus answers cleanly. Run it twice — first
through the API to warm the embedding cache + the Anthropic key path, second
through the UI to check the citation render.

```bash
# CLI smoke
curl -s -X POST https://rag.<subdomain>/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"<your opening question>","collection":"prospect-<name>","k":5}' | jq

# UI smoke
open https://rag.<subdomain>/
```

The cited answer should render in <3 seconds and the `[doc-N]` markers should
all be clickable. If a click returns a 404 PDF — abort and use the rehearsed
`quant-finance` corpus instead. Don't debug live.

---

## During the call

`docs/sales-demo.md` is the live script. Reference it, don't read it. The four
beats:

1. Opening (60s) — pain framing
2. Live cited answer (90s) — the headline moment
3. Refusal moment (60s) — "the system refuses rather than guess"
4. Close (60s) — pricing teaser + the ask

If the live system glitches, pivot to the recorded fallback per
`docs/sales-demo.md:104` — don't try to fix it live.

---

## T+1 hour — Send the one-pager + named-pain email

(Sales pipeline workflow — outside the scope of this checklist.)

---

## T+24 hours — Counter-signed NDA if confidential corpora were discussed

(Legal workflow.)

---

## T+7 days — Scrub prospect data

7-day default retention per the slice 2 plan. Run for every prospect-named
collection that's not converting into a paid engagement.

```bash
python scripts/scrub_corpus.py \
    --collection prospect-<name> \
    --confirm

# Audit log entry:
tail -1 logs/scrub-audit.log
```

The audit log retains the deletion record indefinitely (POPIA-relevant).
Rotate the log file yearly if it grows large.

---

## When the checklist fails

| Symptom | Action |
|---|---|
| Ingest > 70 min on a normal-size corpus | Check `ollama` daemon load; verify `nomic-embed-text` is the active model (`ollama list`); check disk I/O. |
| Many PDFs skip as scanned | Ask prospect for native-text versions; do NOT enable OCR mid-call — slice 3 work. |
| Cloudflare Access login loop | Clear browser cookies for the subdomain; re-auth from a clean tab. |
| `/query` returns 503 | `ANTHROPIC_API_KEY` is unset or invalid on the Hetzner box; pull the `.env` from the password vault and `scp` it up. |
| `[doc-N]` click 404s | Likely a stale-collection-on-server issue; re-run `swap_corpus.py` to confirm the collection is fresh. |
| Live demo about to fail entirely | Open `assets/demos/production-rag.mp4` (Phase 3 slice 2 Task 5 deliverable). |
