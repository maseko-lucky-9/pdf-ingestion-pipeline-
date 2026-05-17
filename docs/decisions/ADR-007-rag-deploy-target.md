# ADR-007: RAG Deploy Target — Cloudflare Access in Front of Hetzner CX22

**Date:** 2026-05-17
**Status:** Accepted
**Slice:** Phase 3 slice 2 Task 2
**Plan link:** `~/.claude/plans/resume-work-on-the-curried-feigenbaum.md`

## Context

Slice 1 of the production-RAG system runs locally on Thulani's M2 MacBook. For sales-grade demos the system must run somewhere that lets the prospect click a URL during a Zoom call and see the cited-answer UI — without exposing the system publicly, without requiring the prospect to install anything, and without giving the prospect a URL that survives the engagement.

The plan capped this Task at 4 hrs and listed three options:

1. **Cloudflare Access in front of Hetzner CX22** — auth-gated hosted box.
2. **Tailscale on Thulani's homelab** — fully private, but the prospect has to install Tailscale or accept a Funnel link (semi-public).
3. **Local-only, screenshare during the call** — laptop is the demo host; no internet path.

## Decision

**Cloudflare Access in front of a Hetzner CX22.** Subdomain `rag.prudentiadigital.co.za`. Email-OTP one-time-password gate on the Access application. No public port on the Hetzner box itself — `cloudflared` tunnels traffic out, never accepts inbound.

The CX22 stays running 24/7 between calls (parent-plan budget already covers ~R140/mo). Cold-booting during the pre-demo window would add 2-5 minutes of risk we don't have during a paid call.

## Rationale

### Why Cloudflare Access + Hetzner won

- **Zero install for the prospect.** Prospect clicks a link, gets an email-OTP, enters the code, sees the system. No VPN client, no laptop screenshare arrangement.
- **Shareable URL that we can revoke.** Adding/removing prospect emails on the Access app is one API call. Tailscale gives us no equivalent if a prospect's account drifts.
- **No inbound ports on the host.** `cloudflared` makes outbound connections only; `nmap` against the public IP returns nothing. No NAT rules to maintain. Lower attack surface than Tailscale Funnel (which exposes a public hostname even when auth-gated).
- **Cloudflare Access free tier covers ≤ 50 users.** Free for our pipeline volume; no per-user billing surprises.
- **Existing Cloudflare account.** `prudentiadigital.co.za` is already on Cloudflare DNS — subdomain delegation is one record add.
- **POPIA fit.** The Hetzner CX22 is in Falkenstein (EU). Data-residency in the EU rather than US is closer to the EU model many South African enterprises align to when they harden against US discovery rules. Not a substitute for prospect-by-prospect data-residency conversations, but a defensible default.

### Why Tailscale homelab lost (this round)

- Requires the prospect to install Tailscale or trust a Funnel URL. Funnel adds a public hostname that survives the engagement.
- Homelab uptime is not 24/7 guaranteed (residential power + ISP).
- Bandwidth ceiling on residential uplink could throttle the PDF preview during a screenshare.

Tailscale **may** become the right call later for internal-only systems (developer access, monitoring), but not for the prospect-facing demo surface.

### Why local-only lost

- "Sorry my Wi-Fi just died" is the wrong sales moment.
- Cannot share the demo URL ahead of the call so the prospect arrives oriented.
- No way to leave a demo running for review between calls.

## Operational rules

1. **No public hostname.** The Cloudflare Tunnel hostname is `rag.prudentiadigital.co.za` — gated by Access, never linked from any public page (parent plan AD-5).
2. **Secrets on disk.** `ANTHROPIC_API_KEY` lives in `/opt/prudentia-rag/.env` (chmod 600, root-owned). The `.env` template is committed at `infra/deploy/.env.example`; the real file is **never** committed. Rotated at engagement end.
3. **DNS guardrail.** All DNS edits go through the Cloudflare API with a token scoped to a single sub-domain — no console clicks where a fat-finger could touch the apex `A` record.
4. **24/7 box.** The CX22 stays up between calls. The pre-demo checklist's "T-15 min" health check assumes a warm host.
5. **Collection sync.** `infra/deploy/sync-collection.sh` is the one tool that moves a built collection from Thulani's M2 to the box. Never `scp -r` ad-hoc — the script writes an audit line.
6. **Data scrub.** `scripts/scrub_corpus.py` runs on the box (not the laptop) at T+7 days via cron or manual trigger. Audit log retained.

## Provisioning sequence (one-time)

Documented in `infra/deploy/cloudflare-access-tunnel.md`. Requires:

- Hetzner Cloud API token (`HCLOUD_TOKEN` env var)
- Cloudflare API token scoped to the `rag.prudentiadigital.co.za` zone subset
- A Cloudflare Access application configured with email-OTP for Thulani's email + an empty allow-list to be populated per prospect

Until those tokens are issued, the box does not exist; the slice-2 scope-cut fallback is `cloudflared` from Thulani's laptop (same Access app, different tunnel origin).

## Alternatives reconsidered

If any of these later become true, revisit:

| Trigger | Likely move |
|---|---|
| Prospect concentration shifts to internal IT teams who already run Tailscale | Add a Tailscale path alongside Cloudflare Access |
| Hetzner pricing changes materially | Re-cost vs Linode / Vultr / OVH |
| Compliance review demands data residency outside the EU | Move to a SA-region host (currently no Cloudflare Tunnel-friendly low-cost option) |
| Inbound prospect volume exceeds free-tier 50 users | Move to Cloudflare Zero Trust paid tier (~$3/user/month) or self-host the gateway |

## Consequences

**Positive:**
- Demo URL is shareable + revocable.
- No public port on the host.
- Same Cloudflare account already manages `prudentiadigital.co.za`.

**Negative:**
- Cloudflare lock-in for the gateway (acceptable; DNS is already there).
- ~R140/mo running cost between calls regardless of demo volume.
- Requires Cloudflare and Hetzner to both work; two points of failure.

**Neutral:**
- Slice 2 fallback (laptop tunnel) is workable but loses the "URL is on a hosted box" credibility cue. Promote to dedicated host in slice 3 once tokens are in place.
