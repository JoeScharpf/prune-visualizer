#!/usr/bin/env bash
# One-command deploy to the GPU box.
#
# Builds the frontend, rsyncs the bundle up, pulls the repo on the box,
# and restarts the backend service. Reads the target host from
# deploy.env (gitignored): OVERSHOOT_SSH=user@host
#
# Note: this restarts only the backend. The model server is a separate
# detached `vllm serve` process; vLLM-fork changes take effect on the
# next Stop/Start GPU from the UI.
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
