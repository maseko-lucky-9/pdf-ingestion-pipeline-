# Production RAG — Sales Demo Walkthrough

**Purpose:** Script for live demos of the RAG system in booked Prudentia AI strategy sessions.
**Audience:** Compliance / legal / ops leads at mid-sized firms — people who need to find specific clauses in large document libraries.
**Total demo time:** 8 minutes. Anything longer loses them.

---

## Pre-demo (offline, before the call)

Run [`docs/pre-demo-checklist.md`](pre-demo-checklist.md). Three minutes; don't skip it.

Quick verify:
```bash
.venv/bin/python -c "from src.answer import synthesize_answer; print('imports OK')"
.venv/bin/python -m pytest tests/test_answer.py tests/test_api_server.py -q
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY uvicorn src.api.server:app --port 8000 &
sleep 2 && curl -s http://localhost:8000/health | jq
```

---

## Opening (60 seconds)

> "Every firm has the same problem. Your compliance manual is 4,000 pages. Your contract library is 80,000 pages. Your internal policies are spread across three tools nobody likes. Someone asks a question, and you spend two days finding the answer.
>
> What you're about to see is a system that turns that pile of PDFs into something you can ask in plain English — and every answer cites the exact page. We're going to point it at a real corpus and ask it real questions."

Pull up the live system. **Use the web UI or `curl` against `/query` — do not show raw retrieval results.** The cited answer is the moment that sells.

---

## Live run (4 minutes)

1. **Pick the corpus.** Use the prospect's domain if you've ingested it pre-demo, or default to the rehearsed corpus.
2. **Ask the first question.** Should be a clear factual lookup with a known good answer.
3. **Narrate the wait (2–3 seconds):**
   > "It's running hybrid retrieval — keyword and semantic search fused together — then asking Claude to answer using only the chunks that came back. Watch what comes out."
4. **The answer appears with `[doc-1]` `[doc-3]` style citations.** Click one open.
   > "This citation maps to page 47 of [filename]. That's the exact source. Not paraphrased, not generated — retrieved and shown."
5. **Ask a question the corpus doesn't have an answer for.**
   > "Now watch what happens when the context isn't there."

   The system responds: `"I cannot answer this question from the provided context."`
   > "This is the part competitors get wrong. Most demos hallucinate when they don't know. This refuses. That's the difference between a tool you can trust for compliance work and a tool you can't."

---

## The refusal moment (90 seconds — high-impact)

Compliance buyers care more about what the system *won't* say than what it will. Use the refusal to anchor the trust message.

> "If you've evaluated other AI tools, you've seen what happens when a model 'just makes something up'. That's not a bug — that's the default behaviour of every general-purpose model. The reason this one doesn't is that we constrain it to answer only from retrieved context, and we instruct it to refuse rather than guess. The refusal is the feature."

Show the system prompt briefly (terminal, not slides) if a technical buyer asks.

---

## Behind the scenes (60 seconds)

> "Three things you should know about the architecture:
>
> 1. **Your data never leaves your network for retrieval.** Docling extracts PDF text on your machine. Embeddings run via Ollama on your machine. Search runs in SQLite locally. The only external call is the final answer synthesis — and that's swappable to a local model if your compliance posture requires it.
>
> 2. **Citations are not generated — they're proven.** The numbered `[doc-N]` tags map deterministically back to retrieved chunks. We don't trust the model to format citations correctly; we extract them from the answer with a regex.
>
> 3. **Every request is logged with a request_id.** If a user disputes an answer next month, you replay the exact query, the exact retrieved chunks, the exact model response."

---

## Close (60 seconds)

> "Three things I want to point out before questions:
>
> 1. **This is your stack.** Python, SQLite, FastAPI. Self-hostable behind your firewall. POPIA-compatible without compromise.
> 2. **The eval harness is real.** We benchmark recall@5 and answer faithfulness on every change. You'll see the dashboard in our second session.
> 3. **One workflow today is the floor.** We start with one corpus, one use case. Your team uses it. We measure. We add the next workflow when the first one is paying its rent.
>
> Where in your business is the answer to a question hidden behind 50,000 pages?"

Pause. Let them answer.

---

## Common buyer questions

**"Can it run on our infrastructure?"**
Yes. Python + SQLite + Ollama. Tested on a 32 GB Linux VM. We can deploy behind Cloudflare Access or Tailscale today.

**"What's the false-answer rate?"**
The eval harness reports answer-faithfulness on every change. We don't release without baseline measurement. Current numbers ship with the engagement report.

**"How long does ingestion take?"**
~3–8 minutes per 500-page PDF on an M2 MacBook. Re-indexing is incremental.

**"How do you handle POPIA / data residency?"**
Retrieval is fully local. Answer synthesis defaults to Anthropic API but swaps to a local Llama model with one config change. See `docs/decisions/ADR-006-…` for the trade-off.

**"What does pricing look like?"**
Three tiers — Audit / Pilot / Retainer. Indicative ranges in the next session after we scope the corpus.

---

## Fallback when the live demo fails

Network drops, API quota hit, daemon died:

1. **Acknowledge immediately.** "Live system is having a moment — happens 1 in 30 demos. Showing the recorded run."
2. **Open `assets/demos/production-rag.mp4`** (5-min pre-recorded run).
3. **Do not debug live.** Looks worse than the recording.

Log the failure post-call. Two same-kind failures in a month → switch to recording-first until fixed.

---

## After the call

- Within 1 hour: send one-pager PDF + email referencing the prospect's named pain.
- Within 24 hours: counter-signed NDA if confidential corpora discussed.
- Within 7 days: demo-data scrub script runs; verify audit-log entry.
