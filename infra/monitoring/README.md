# `infra/monitoring/`

Cron-driven health monitoring for the Prudentia RAG deployment + adjacent
demo systems. **The scripts here are operator-deployable** — copy them onto
the host, schedule via cron, and configure via local YAML. None of the
configuration is committed.

Slice 6, backlog item #19.

## Files

- `health_pinger.py` — one-shot cron entry. Pings configured `/health`
  endpoints and POSTs to a ntfy server when any target fails. Exit code 0
  when all healthy, 1 when any failed.
- `targets.example.yaml` — template config. Copy to `targets.yaml` on the
  host, fill in the real ntfy URL + per-target Cloudflare Access env-var
  names. chmod 600. Never commit.
- `scrub_cron_wrapper.sh` — cron-safe wrapper that runs
  `scripts/scrub_corpus.py --age-threshold-days 7` against the deployed
  collections root, sourcing the host's venv first.

## Crontab examples

```
# Health pinger every 30 minutes; only logs / alerts when something fails.
*/30 * * * *  cd /opt/prudentia-rag && .venv/bin/python infra/monitoring/health_pinger.py --config infra/monitoring/targets.yaml --quiet >> /var/log/prudentia/health.log 2>&1

# Daily scrub at 03:30 UTC; deletes any prospect collection older than 7 days.
30 3 * * *  /opt/prudentia-rag/infra/monitoring/scrub_cron_wrapper.sh >> /var/log/prudentia/scrub.log 2>&1
```

## Out of scope (still operator-gated)

- The ntfy server itself (URL, auth, retention policy)
- The cron entries on the box
- The Cloudflare Access service-token JWT that the pinger uses to reach the
  auth-gated `/health` endpoint (lives in `/opt/prudentia-rag/.env`)
