#!/usr/bin/env bash
# Push a named collection from the local M2 to the Hetzner box's mounted volume.
#
# Usage:
#   ./infra/deploy/sync-collection.sh <collection-name> [host]
#
# Writes a single audit line per push to logs/sync-collection-audit.log.
#
# Pre-conditions:
#   - Local collection exists at ./collections/<name>/index.db
#   - Operator can ssh as `thulani` to the host (default: prudentia-rag-1)
#   - `rsync` installed locally and on the host
set -euo pipefail

COLLECTION="${1:-}"
HOST="${2:-prudentia-rag-1}"
REMOTE_USER="thulani"
REMOTE_BASE="/data/collections"  # docker volume mount path on the host
AUDIT_LOG="logs/sync-collection-audit.log"

if [[ -z "$COLLECTION" ]]; then
  echo "usage: $0 <collection-name> [host]" >&2
  exit 2
fi

LOCAL_DIR="./collections/$COLLECTION"
if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "ERR: local collection not found: $LOCAL_DIR" >&2
  exit 2
fi

if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_USER}@${HOST}" true 2>/dev/null; then
  echo "ERR: cannot SSH to ${REMOTE_USER}@${HOST}" >&2
  exit 2
fi

t0=$(date +%s)

# `--delete` so the remote collection mirrors the local state exactly. Use a
# named temp dir on the remote side and atomically swap it in to avoid a
# half-written collection being queryable mid-sync.
TMP_REMOTE="${REMOTE_BASE}/.staging-${COLLECTION}"

ssh "${REMOTE_USER}@${HOST}" "mkdir -p ${TMP_REMOTE}"

rsync -avz --delete --partial \
    "${LOCAL_DIR}/" \
    "${REMOTE_USER}@${HOST}:${TMP_REMOTE}/"

ssh "${REMOTE_USER}@${HOST}" "rm -rf ${REMOTE_BASE}/${COLLECTION} && mv ${TMP_REMOTE} ${REMOTE_BASE}/${COLLECTION}"

t1=$(date +%s)
elapsed=$((t1 - t0))

mkdir -p "$(dirname "$AUDIT_LOG")"
size_local=$(du -sk "$LOCAL_DIR" | awk '{print $1}')
ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "${ts},${USER},${HOST},${COLLECTION},${size_local}KB,${elapsed}s" >> "$AUDIT_LOG"

echo "[OK] Synced ${COLLECTION} (${size_local}KB) to ${HOST} in ${elapsed}s"
echo "     Audit: $(tail -1 ${AUDIT_LOG})"
