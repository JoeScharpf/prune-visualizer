# Prune Visualizer

Interactive visualizer for **HiPrune** and **HyDART** visual token pruning on
Qwen2.5-VL-3B and LLaVA-1.5-7B. Upload an image, pick a model and pruning
method, tune the parameters, and see the original image next to the pruned
token grid — with per-patch hover tooltips showing each token's category,
rank, and attention scores.

## Architecture

```
web/       Vite + React + Tailwind frontend
server/    FastAPI backend (runs locally)
  app.py          GPU control (start/stop over SSH), inference proxy,
                  static serving of the built frontend
  llava_server.py LLaVA-1.5 wrapper — deployed to and run on the GPU host
```

- **Qwen2.5-VL** is served by a vLLM fork with HiPrune/HyDART built in
  (`token_pruning` + `token_pruning_metadata` on the OpenAI-compatible
  chat-completions API).
- **LLaVA-1.5** is served by `llava_server.py`, a thin FastAPI wrapper that
  mirrors the same response shape.
- The local backend starts/stops the remote model servers over SSH and
  reaches them through SSH port forwarding.

## Running

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

Set the GPU host in `server/app.py` (`SSH_HOST`, `REMOTE_DIR`); SSH key auth
is required (`BatchMode=yes`).

## Parameters

| Parameter | Meaning |
| --- | --- |
| Retention | Fraction of visual tokens kept after pruning |
| Alpha | HiPrune anchor budget fraction (anchors + neighbor buffers) |
| Object layer | Vision-encoder layer used for object-level attention |
| Lambda seed / pick | HyDART greedy-MMR trade-off between attention and diversity |

Hovering a patch in the pruned view shows its token index, grid position,
category (anchor / buffer / register / diverse / pruned), rank within its
category, and raw attention (or cosine-similarity) scores.
