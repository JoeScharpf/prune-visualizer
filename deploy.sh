#!/usr/bin/env bash
# One-command deploy to the GPU box.
#
# Builds the frontend, rsyncs the bundle up, pulls the repo on the box,
# and restarts the backend service. Reads the target host from
# deploy.env (gitignored): OVERSHOOT_SSH=user@host
#
# Note: this restarts only the backend. The model servers are separate
# detached `vllm serve` processes. The UI's Stop puts them to *sleep*
# (fast wake, but still running the old code) — after deploying
# vLLM-fork changes, do a hard stop so the next Start cold-boots the
# new code:  curl -X POST 'http://localhost:8300/api/gpu/stop?hard=true'
set -euo pipefail
cd "$(dirname "$0")"

source deploy.env
: "${OVERSHOOT_SSH:?set OVERSHOOT_SSH=user@host in deploy.env}"

REMOTE_DIR='~/hiprune/prune-visualizer'

echo "==> building frontend"
npm --prefix web run build

echo "==> syncing dist to $OVERSHOOT_SSH"
rsync -az --delete web/dist/ "$OVERSHOOT_SSH:$REMOTE_DIR/web/dist/"

echo "==> pulling repo + restarting backend on the box"
ssh "$OVERSHOOT_SSH" "cd $REMOTE_DIR && git pull --ff-only && systemctl --user restart prune-visualizer.service"

echo "==> deployed"
