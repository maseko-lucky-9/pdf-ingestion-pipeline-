# Cloudflare Access + Hetzner CX22 — Provisioning Runbook

**ADR:** [`docs/decisions/ADR-007-rag-deploy-target.md`](../../docs/decisions/ADR-007-rag-deploy-target.md)
**Status:** Runbook for one-time provisioning. Re-runnable for disaster recovery.
**Owner:** Thulani

> No public ports on the host. `cloudflared` makes outbound connections only;
> Cloudflare Access gates inbound at the edge.

## Pre-checks (15 min)

Before sinking the rest of the time budget, confirm:

1. `prudentiadigital.co.za` is on Cloudflare DNS (check Cloudflare dashboard → Websites).
2. Cloudflare Access **free tier** is active on the Zero Trust dashboard (covers ≤ 50 users).
3. Hetzner Cloud account exists with billing enabled.
4. `cloudflared` and `hcloud` CLIs are installed locally:
   ```bash
   brew install cloudflared hcloud
   ```
5. Cloudflare API token issued with **Zone:DNS:Edit** scoped to `prudentiadigital.co.za` and **Account:Cloudflare Tunnel:Edit**. Saved in 1Password as `CF_API_TOKEN`.
6. Hetzner API token issued from the Cloud Console (project token, not user token). Saved as `HCLOUD_TOKEN`.

If any of (1)–(5) fail, declare Task 2 blocked and use the laptop-tunnel fallback at the bottom of this file.

## Provision the Hetzner CX22 (10 min)

```bash
# Authenticate
export HCLOUD_TOKEN=...
hcloud context create prudentia

# SSH key already in the cloud console: prudentia-mb-air-2026
# Pick the EU image; ubuntu-22.04 has cloud-init out of the box.
hcloud server create \
    --type cx22 \
    --image ubuntu-22.04 \
    --location fsn1 \
    --name prudentia-rag-1 \
    --ssh-key prudentia-mb-air-2026 \
    --label app=prudentia-rag \
    --label env=prod

# Capture the public IP for ssh — but note we will NOT expose ports on it.
PUB_IP=$(hcloud server ip prudentia-rag-1)
echo "Provisioned: $PUB_IP"
```

## Harden the host (15 min)

SSH as root once, then never as root again.

```bash
ssh root@$PUB_IP <<'BOOTSTRAP'
set -euxo pipefail

# 1. Create the operator user
adduser --disabled-password --gecos "" thulani
usermod -aG sudo thulani
mkdir -p /home/thulani/.ssh
cp /root/.ssh/authorized_keys /home/thulani/.ssh/
chown -R thulani:thulani /home/thulani/.ssh
chmod 700 /home/thulani/.ssh
chmod 600 /home/thulani/.ssh/authorized_keys

# 2. Disable password auth, root login over SSH
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl reload sshd

# 3. ufw — drop everything except SSH on 22 (still need SSH for ops; tunnel
# the actual application via cloudflared so no 80/443 exposure on the host).
apt update && apt -y install ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw --force enable

# 4. Unattended upgrades
apt -y install unattended-upgrades
dpkg-reconfigure -plow unattended-upgrades

# 5. Docker
apt -y install ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release; echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
apt update && apt -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
usermod -aG docker thulani

# 6. Filesystem layout
mkdir -p /opt/prudentia-rag /data/collections /var/log/prudentia-rag
chown -R thulani:thulani /opt/prudentia-rag /data /var/log/prudentia-rag

BOOTSTRAP
```

Verify SSH as the operator user works:
```bash
ssh thulani@$PUB_IP "docker --version && ufw status"
```

## Create the Cloudflare Tunnel (10 min)

```bash
export CF_API_TOKEN=...

# 1. Auth cloudflared locally (browser flow once; cert lands in ~/.cloudflared/)
cloudflared tunnel login

# 2. Create the tunnel — this returns a UUID and writes credentials JSON locally
cloudflared tunnel create prudentia-rag

# 3. Print the token; we'll pass this to the cloudflared container on the host
TUNNEL_UUID=$(cloudflared tunnel list | awk '/prudentia-rag/{print $1}')
TUNNEL_TOKEN=$(cloudflared tunnel token "$TUNNEL_UUID")

# 4. Route the subdomain to the tunnel; DNS write via the scoped API token,
# NOT the dashboard, per the ADR's DNS guardrail.
cloudflared tunnel route dns prudentia-rag rag.prudentiadigital.co.za

# 5. Build the tunnel config — point traffic at the `api` service in the
# compose network. cloudflared resolves it via the docker DNS.
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml <<YAML
tunnel: $TUNNEL_UUID
credentials-file: /etc/cloudflared/${TUNNEL_UUID}.json
ingress:
  - hostname: rag.prudentiadigital.co.za
    service: http://api:8000
  - service: http_status:404
YAML
```

## Deploy the stack (10 min)

Copy the deploy directory + the prepared `.env` (real one, NEVER git) to the host, then bring it up.

```bash
# From the repo root locally:
ssh thulani@$PUB_IP "mkdir -p /opt/prudentia-rag/infra/deploy"
scp -r infra/deploy/Dockerfile infra/deploy/docker-compose.yml \
    thulani@$PUB_IP:/opt/prudentia-rag/infra/deploy/

# Generate the real .env from the example (one-time):
cp infra/deploy/.env.example /tmp/prudentia-rag.env
# Edit /tmp/prudentia-rag.env to fill in:
#   ANTHROPIC_API_KEY  (from 1Password)
#   TUNNEL_TOKEN       (from `cloudflared tunnel token ...` above)
#   CORS_ORIGINS       (already correct in the example)
scp /tmp/prudentia-rag.env thulani@$PUB_IP:/opt/prudentia-rag/infra/deploy/.env
ssh thulani@$PUB_IP "chmod 600 /opt/prudentia-rag/infra/deploy/.env"
rm /tmp/prudentia-rag.env

# Stage the source (needed because the Dockerfile builds from the repo root)
rsync -av --delete --exclude='.venv*/' --exclude='__pycache__' --exclude='.git/' --exclude='collections/' --exclude='data/' --exclude='results/' \
    ./ thulani@$PUB_IP:/opt/prudentia-rag/

# Bring it up
ssh thulani@$PUB_IP <<'UP'
cd /opt/prudentia-rag
docker compose -f infra/deploy/docker-compose.yml --env-file infra/deploy/.env up -d --build
docker compose -f infra/deploy/docker-compose.yml ps
UP
```

## Verify (5 min)

```bash
# Authenticated path (refresh Access cookie in browser first):
curl -sL https://rag.prudentiadigital.co.za/health | jq .

# Unauthenticated path — expect a 302 to the Cloudflare login page:
curl -sI https://rag.prudentiadigital.co.za/health | head -5

# No public ports — should show NO open TCP ports from outside:
nmap -Pn -T4 -p- "$PUB_IP" | tail -10
```

If `nmap` shows anything other than `22/tcp open ssh` (and that should be restricted to your home IP via Hetzner's firewall — slice 3), abort and lock down before sharing the URL.

## Push a collection (used during pre-demo prep)

Use `infra/deploy/sync-collection.sh` (next file in this directory) — never `scp -r` ad-hoc; the script writes an audit line per push.

## Scope-cut fallback: laptop tunnel

If Hetzner/Cloudflare credentials are not yet available, the slice-2 fallback runs the same Cloudflare Access app pointed at a tunnel originating from Thulani's laptop:

```bash
# Locally
uvicorn src.api.server:app --host 127.0.0.1 --port 8000 &
cloudflared tunnel run prudentia-rag-laptop
```

Same `rag.prudentiadigital.co.za` URL, same Access OTP. The cost is that the laptop has to stay on Wi-Fi during the demo. This is the slice-2 fallback explicitly named in the ADR.

## Disaster recovery

Re-run this entire runbook from a clean Hetzner box. The state that survives:

- The Cloudflare Tunnel UUID + credentials (recreate via `cloudflared tunnel create` if lost; route DNS again)
- The Access app config (kept in the Cloudflare Zero Trust dashboard; can re-create from the dashboard or terraform)
- The committed `collections/` data (not committed — re-build via `python -m src.ingest`)

Nothing on the box is irreplaceable. The audit log (`logs/scrub-audit.log`) is the one thing worth backing up off-host before reprovisioning.
