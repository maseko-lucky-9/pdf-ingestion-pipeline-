#!/usr/bin/env bash
# Cron-friendly wrapper around scripts/scrub_corpus.py.
#
# Scrubs every collection whose name starts with "prospect-" and whose
# directory mtime is older than 7 days. The rehearsed demo corpora
# (quant-finance, sa-legislation) are never touched because they don't
# match the prefix.
#
# Schedule via crontab on the deployed host:
#   30 3 * * *  /opt/prudentia-rag/infra/monitoring/scrub_cron_wrapper.sh >> /var/log/prudentia/scrub.log 2>&1
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/opt/prudentia-rag}"
VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
COLLECTIONS_ROOT="${COLLECTIONS_ROOT:-${PROJECT_ROOT}/collections}"
AUDIT_LOG="${AUDIT_LOG:-${PROJECT_ROOT}/logs/scrub-audit.log}"
AGE_DAYS="${AGE_DAYS:-7}"
PREFIX="${PREFIX:-prospect-}"

cd "${PROJECT_ROOT}"

"${VENV_PY}" scripts/scrub_corpus.py \
    --age-threshold-days "${AGE_DAYS}" \
    --prefix "${PREFIX}" \
    --collections-root "${COLLECTIONS_ROOT}" \
    --audit-log "${AUDIT_LOG}" \
    --operator "cron" \
    --confirm
