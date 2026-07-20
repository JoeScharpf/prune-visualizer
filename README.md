# Prune Visualizer

Interactive visualizer for **HiPrune** and **HyDART** visual token pruning
on Qwen2.5-VL-3B, LLaVA-1.5-7B, and Gemma 4 E4B. Upload an image, pick a
model and pruning method, tune the parameters, and see the original image
next to the pruned token grid — with per-patch hover tooltips showing each
token's category, rank, and attention scores.

<img width="1406" height="496" alt="Screenshot 2026-07-20 at 10 11 38 AM" src="https://github.com/user-attachments/assets/4455ab05-5c39-4343-a180-dcb37eba5e4b" />

## What lives where

The running system is made of three pieces; this repo is only the first:

1. **This repo** — the visualizer app: a FastAPI backend (`server/app.py`)
   that starts/stops model servers and proxies inference, and a
   Vite + React + Tailwind frontend (`web/`).
2. **The vLLM fork** — [`JoeScharpf/vllm`](https://github.com/JoeScharpf/vllm),
   branch `hydart-qwen2_5-vl`. All pruning logic lives there: the
   HiPrune/HyDART selection algorithms (`vllm/multimodal/hiprune.py`),
   the per-model vision-tower instrumentation (Qwen2.5-VL, LLaVA via
   CLIP, Gemma 4), the `--enable-hiprune` serve flag, and the
   `token_pruning` / `token_pruning_metadata` fields on the
   OpenAI-compatible chat-completions API. The backend launches plain
   `vllm serve` and assumes this fork is what's installed.
3. **The GPU host** — deployment state that is intentionally not in git:
   the Python venvs, `deploy.env` (SSH host/secrets, gitignored), the
   built frontend (`web/dist`, rebuilt from source), the systemd units,
   and the ngrok config. See "Deployment" below.

```
web/       Vite + React + Tailwind frontend
server/    FastAPI backend
  app.py          GPU control (start/stop), inference proxy,
                  static serving of the built frontend
  llava_server.py Retired transformers-based LLaVA wrapper, kept as the
                  reference implementation (no longer deployed)
```

## Models

All three models are served by the vLLM fork, one at a time (the pruning
method and HyDART lambdas are fixed at serve time, so switching model or
method restarts the server — a few minutes per start).

| Model | HF checkpoint | Port | Object layer |
| --- | --- | --- | --- |
| Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` | 8124 | 16 |
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | 8125 | 9 |
| Gemma 4 E4B | `google/gemma-4-e4b-it` | 8126 | 8 |

Weights are downloaded from HuggingFace once per model into the GPU
host's cache; later starts load from disk.

## Running locally

Backend (from `server/`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 8300
```

Frontend (from `web/`):

```bash
npm install
npm run dev        # dev server on :5180, proxies /api to :8300
npm run build      # production build served by the backend
```

Configuration is via environment variables (see `server/app.py`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `HIPRUNE_HOST_MODE` | `ssh` | `ssh`: backend on a laptop, GPU reached over SSH with port forwarding. `local`: backend runs on the GPU machine and launches model servers directly |
| `HIPRUNE_SSH_HOST` | `joe@safeai-gpu3.andrew.cmu.edu` | `user@host` of the GPU machine (ssh mode; key auth required, `BatchMode=yes`) |
| `HIPRUNE_REMOTE_DIR` | `~/hiprune` | Directory on the GPU machine with `venv/` (the vLLM fork install) |
| `HIPRUNE_GPU_INDEX` | `0` | Which physical GPU to use |

## Deployment (GPU box + shareable link)

The public deployment runs entirely on the GPU host in
`HIPRUNE_HOST_MODE=local`, kept alive by two user-level systemd units,
with ngrok providing the public HTTPS link.

Layout on the box (under `~/hiprune`):

- `vllm/` — clone of the fork on `hydart-qwen2_5-vl`, installed into `venv/`
- `prune-visualizer/` — clone of this repo (with `web/dist` rsynced in)
- `app_venv/` — venv for the backend (`server/requirements.txt`)

`~/.config/systemd/user/prune-visualizer.service`:

```ini
[Unit]
Description=HiPrune/HyDART visualizer backend
After=network-online.target

[Service]
WorkingDirectory=/home/<user>/hiprune/prune-visualizer/server
Environment=HIPRUNE_HOST_MODE=local
Environment=HIPRUNE_REMOTE_DIR=/home/<user>/hiprune
Environment=HIPRUNE_GPU_INDEX=0
ExecStart=/home/<user>/hiprune/app_venv/bin/uvicorn app:app --host 127.0.0.1 --port 8300
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
```

`~/.config/systemd/user/ngrok-tunnel.service` (requires an ngrok account
with a reserved domain; `ngrok config add-authtoken ...` once):

```ini
[Unit]
Description=ngrok tunnel for prune visualizer
After=network-online.target

[Service]
ExecStart=/home/<user>/.local/bin/ngrok http --url=<your-domain>.ngrok-free.dev 8300 --log stdout
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
```

Enable both with `systemctl --user enable --now prune-visualizer ngrok-tunnel`
(and `loginctl enable-linger <user>` so they survive logout).

Deploying an update:

```bash
# on the laptop: build the frontend and sync it up
cd web && npm run build
rsync -az --delete web/dist/ <user>@<gpu-host>:~/hiprune/prune-visualizer/web/dist/

# on the box: pull both repos and restart
cd ~/hiprune/prune-visualizer && git pull
cd ~/hiprune/vllm && git pull origin hydart-qwen2_5-vl
systemctl --user restart prune-visualizer.service
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| Retention | Fraction of visual tokens kept after pruning (per request) |
| Alpha | HiPrune anchor budget fraction — paper default, fixed at serve time |
| Object layer | Vision-encoder layer for object-level attention — paper default per model (16 for Qwen, 9 for LLaVA, 8 for Gemma), fixed at serve time |
| Lambda seed / pick | HyDART greedy-MMR trade-off — applied when the GPU is started |

Hovering a patch in the pruned view shows its token index, grid position,
category (anchor / buffer / register / diverse / pruned), rank within its
category, and raw attention (or cosine-similarity) scores.
