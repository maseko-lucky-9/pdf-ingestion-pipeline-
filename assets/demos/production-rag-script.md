# Production RAG — Fallback Recording Script

**Output:** `assets/demos/production-rag.mp4` (5 min, 1080p, narrated)
**Purpose:** plays during a booked sales call when the live demo glitches.
**Referenced from:** `docs/sales-demo.md` ("Fallback when the live demo fails").

The recording is intentionally deterministic — same corpus, same opening query,
same refusal-test query, same closing visual — so any re-shoot lands within
seconds of the original timing.

## Pre-record setup (15 min)

1. **Corpus.** Use the `quant-finance` collection. Do NOT swap in a prospect
   corpus — this recording is the generic fallback.
2. **Browser.** Fresh Chrome profile. 1920×1080 window. Dark theme on, no
   extensions visible, no other tabs.
3. **Audio.** External mic (not laptop). Levels checked; -12dB peak target.
4. **Recorder.** QuickTime → File → New Screen Recording → Window → Chrome.
   1080p, 30fps, microphone source set.
5. **Trial run.** Walk the script once with the recorder off. The full
   walkthrough must land between 4:30 and 5:00 — if the trial overruns, trim
   the close, not the headline moment.

## Shot list

| Time | Visual | Voice-over |
|---|---|---|
| 0:00 – 0:10 | Static intro card: `assets/demos/intro-card.png` (Prudentia · Production RAG · 5-minute walk) | "This is a five-minute walkthrough of the production RAG system Prudentia ships. The full live demo lives behind an auth gate during a booked session; this recording exists so a flaky network never costs us a closing call." |
| 0:10 – 0:50 | UI loaded at `/`. Show the empty form. Cursor in the textarea. | "What you're about to see is a system that turns a pile of PDFs into something you can ask in plain English — and every answer cites the exact page. We're pointed at a real financial-research corpus: eight books across technical analysis, quant trading, and ML." |
| 0:50 – 1:10 | Type the opening query slowly so the audience can read it: **"What is mean reversion and how is it used in pairs trading?"** Click Ask. | "I'll ask a question I know the corpus answers cleanly — mean reversion in pairs trading. Watch the response come back." |
| 1:10 – 1:40 | Answer renders with `[doc-1]`, `[doc-2]` markers highlighted. Mouse hovers over `[doc-1]`. | "It's running hybrid retrieval — keyword and semantic search fused together — then asking Claude to answer using only the chunks that came back. The bracketed tags are not decoration; they're proof-of-source." |
| 1:40 – 2:10 | Click `[doc-1]`. PDF panel opens on the right at the cited page. Quickly scroll to highlight the relevant sentence. | "This citation maps to page 150 of Ernest Chan's Quantitative Trading. That's the exact source. Not paraphrased, not generated — retrieved and shown. Click another, see the next page." |
| 2:10 – 2:25 | Click `[doc-2]`. Different PDF page loads. | "Different book, different page. Each citation is provable." |
| 2:25 – 2:30 | Clear the form. | (silence) |
| 2:30 – 2:55 | Type: **"What is the information ratio and how does it differ from the Sharpe ratio?"** Click Ask. | "Now watch what happens when the context isn't there. The corpus covers the Sharpe ratio extensively but never defines the information ratio." |
| 2:55 – 3:10 | Answer renders as: `"I cannot answer this question from the provided context."` Refusal styling kicks in (italic, accent border). | "The system refuses. This is the part competitors get wrong. Most demos hallucinate when they don't know. This refuses — that's the difference between a tool you can trust for compliance work and a tool you can't." |
| 3:10 – 3:30 | Pull up a small terminal beside the browser. Show `tail -3 /var/log/prudentia-rag/requests.log` with structured JSON entries for the two queries (mean-reversion 200, info-ratio 200). | "Every request is logged with a request id, a hit count, and token counts. If a user disputes an answer next month, you replay the exact query, the exact retrieved chunks, the exact model response." |
| 3:30 – 4:00 | Cut back to the UI. Show the request-id pill at the bottom of the answer. Highlight the latency pill (~2 sec). | "Round-trip is around two seconds end-to-end on the demo box. That holds against a corpus up to about a hundred thousand pages on this hardware." |
| 4:00 – 4:30 | Switch to a static closing card: `assets/demos/close-card.png` (three bullets: Your data, your stack · Evals on every release · One workflow at a time · Book a strategy session). | "Three things to take away. One: this is your stack — Python, SQLite, FastAPI, self-hostable behind your firewall, POPIA-compatible. Two: we benchmark recall and answer faithfulness on every release; the dashboard ships with the engagement. Three: one workflow today is the floor — we start with one corpus, one use case, and grow from there." |
| 4:30 – 4:50 | Same close card, cursor highlights "Book a strategy session". | "Where in your business is the answer to a question hidden behind fifty thousand pages? That's the question worth a thirty-minute session." |
| 4:50 – 5:00 | Card fades to logo. | (silence) |

## Re-shoot rules

If something glitches in a take, restart the take from the most recent shot
boundary in the table above — do not splice. The visual transitions matter
less than the audio continuity.

The opening question and the refusal question are FIXED. Do not improvise a
different example — the answers above were rehearsed against the current
state of the `quant-finance` index and known to be clean. A re-pick risks an
unrehearsed refusal or weak citation.

## Post-process

1. Trim to 5:00. If under 4:30, the close was too short — re-shoot the close.
2. Add a one-second fade-in at 0:00 and fade-out at 5:00.
3. Normalize audio to -16 LUFS.
4. Export 1080p H.264, ~10 Mbps. Target file under 25 MB so it can be checked
   into git directly (otherwise honour `.gitignore` rule and reference it
   from a private bucket).
5. Save as `assets/demos/production-rag.mp4`.

## Once recorded

Update `docs/sales-demo.md` — remove the "Status (slice 1)" callout on line
~109 referencing this file. The callout is the consistency-sweep fix that
slice 2 explicitly tracks.
