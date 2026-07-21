"""HiPrune/HyDART visualizer backend.

Three jobs:

1. GPU control: launch/stop the model server (the vLLM fork serves both
   Qwen2.5-VL and LLaVA-1.5). In ``ssh`` mode the server runs on a
   remote GPU host reached over SSH, with an SSH tunnel forwarding its
   port; in ``local`` mode this backend runs on the GPU machine itself
   and launches it directly.
2. Inference proxy: forward an image + prompt + pruning params to the
   model server (OpenAI-compatible chat completions for both models).
3. Static serving of the built frontend (visualizer/web/dist).

Deployment config (environment variables):
- HIPRUNE_HOST_MODE: "ssh" (default; backend on a laptop, GPU remote)
  or "local" (backend on the GPU machine).
- HIPRUNE_SSH_HOST: user@host of the GPU machine (ssh mode only).
- HIPRUNE_REMOTE_DIR: directory on the GPU machine holding venv/ and the
  vLLM fork (default ~/hiprune).
- HIPRUNE_GPU_INDEX: which physical GPU to use (default 0).

Server-side facts this encodes (from the deployed vLLM fork):
- vLLM reads HIPRUNE_METHOD / HYDART_LAMBDA_SEED / HYDART_LAMBDA_PICK from
  the environment at startup, so method/lambda changes restart the server.
- The retention ratio is per-request (`token_pruning` chat-completions
  field). Alpha and object layer are constants in the fork (paper
  defaults per model), so they are display-only in the UI.

Usage:
    pip install -r requirements.txt
    uvicorn app:app --port 8300
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HOST_MODE = os.environ.get("HIPRUNE_HOST_MODE", "ssh")  # "ssh" | "local"
SSH_HOST = os.environ.get("HIPRUNE_SSH_HOST", "joe@safeai-gpu3.andrew.cmu.edu")
SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
]
REMOTE_DIR = os.environ.get("HIPRUNE_REMOTE_DIR", "~/hiprune")
QWEN_PORT = 8124
LLAVA_PORT = 8125
GEMMA_PORT = 8126
GPU_INDEX = int(os.environ.get("HIPRUNE_GPU_INDEX", "0"))

HOST_LABEL = "GPU host" if HOST_MODE == "local" else SSH_HOST.split("@")[-1]

MODELS = {
    "qwen2_5_vl": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "port": QWEN_PORT,
        "log": "serve_visualizer_qwen.log",
        "max_model_len": 32768,
    },
    "llava_1_5": {
        "hf_id": "llava-hf/llava-1.5-7b-hf",
        "port": LLAVA_PORT,
        "log": "serve_visualizer_llava.log",
        "max_model_len": 4096,
    },
    "gemma4": {
        "hf_id": "google/gemma-4-e4b-it",
        "port": GEMMA_PORT,
        "log": "serve_visualizer_gemma.log",
        "max_model_len": 8192,
    },
}

ModelKey = Literal["qwen2_5_vl", "llava_1_5", "gemma4"]
MethodKey = Literal["hiprune", "hydart", "hiprune_pp"]


class StartRequest(BaseModel):
    model: ModelKey
    method: MethodKey
    lambda_seed: float = 0.1
    lambda_pick: float = 0.5
    beta: float = Field(default=0.1, ge=0.0, le=1.0)


class InferRequest(BaseModel):
    image: str  # data URL
    model: ModelKey
    method: MethodKey
    prompt: str
    retention: float = Field(default=0.223, ge=0.01, le=1.0)
    alpha: float = 0.1
    object_layer: int = 0
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    lambda_seed: float = 0.1
    lambda_pick: float = 0.5
    beta: float = Field(default=0.1, ge=0.0, le=1.0)
    with_baseline: bool = False


class GpuState:
    """In-memory view of what we launched; verified against reality by
    /gpu/status probes."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.phase: str = "unknown"
        self.model: str | None = None
        self.method: str | None = None
        self.detail: str = ""
        self.tunnel: subprocess.Popen | None = None


STATE = GpuState()
app = FastAPI(title="HiPrune Visualizer")


def ssh_run(command: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a shell command on the GPU host (over SSH, or directly in
    local mode)."""
    if HOST_MODE == "local":
        argv = ["bash", "-c", command]
    else:
        argv = ["ssh", *SSH_OPTS, SSH_HOST, command]
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def ensure_tunnel() -> None:
    """Keep one ssh -N -L process forwarding both model ports. No-op in
    local mode: the model servers are already on localhost."""
    if HOST_MODE == "local":
        return
    with STATE.lock:
        if STATE.tunnel is not None and STATE.tunnel.poll() is None:
            return
        STATE.tunnel = subprocess.Popen(
            [
                "ssh", *SSH_OPTS, "-N",
                "-o", "ExitOnForwardFailure=yes",
                "-L", f"{QWEN_PORT}:localhost:{QWEN_PORT}",
                "-L", f"{LLAVA_PORT}:localhost:{LLAVA_PORT}",
                "-L", f"{GEMMA_PORT}:localhost:{GEMMA_PORT}",
                SSH_HOST,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def drop_tunnel() -> None:
    if HOST_MODE == "local":
        return
    with STATE.lock:
        if STATE.tunnel is not None:
            STATE.tunnel.terminate()
            STATE.tunnel = None


def health_url(model: str) -> str:
    return f"http://localhost:{MODELS[model]['port']}/health"


async def probe_health(model: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(health_url(model))
            return r.status_code == 200
    except httpx.HTTPError:
        return False


def remote_server_running(model: str) -> bool:
    port = MODELS[model]["port"]
    res = ssh_run(f"pgrep -u \"$(id -u)\" -f 'port {port}' >/dev/null && echo yes || echo no")
    return res.returncode == 0 and "yes" in res.stdout


def launch_command(req: StartRequest) -> str:
    """Build the remote launch line.

    The whole launch script is backgrounded with every fd detached from
    the SSH socket, so the ssh call returns immediately instead of
    waiting on the server process (the remote shell otherwise lingers as
    the job's parent and holds the session open).
    """
    env = (
        f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={GPU_INDEX} "
        f"HIPRUNE_METHOD={req.method} "
        f"HYDART_LAMBDA_SEED={req.lambda_seed} "
        f"HYDART_LAMBDA_PICK={req.lambda_pick} "
        f"HIPRUNE_PP_BETA={req.beta} "
        "VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0"
    )
    cfg = MODELS[req.model]
    script = (
        f"cd {REMOTE_DIR} && source venv/bin/activate && "
        f"{env} exec vllm serve {cfg['hf_id']} "
        f"--max-model-len {cfg['max_model_len']} "
        f"--enable-hiprune --enable-per-request-metrics "
        f"--gpu-memory-utilization 0.85 --port {cfg['port']}"
    )
    return (
        f"nohup bash -c '{script}' > {REMOTE_DIR}/{cfg['log']} 2>&1 < /dev/null & "
        "echo LAUNCHED"
    )


def stop_all_remote() -> None:
    # Only touches processes owned by our account. The bracketed first
    # letter keeps pkill -f from matching the shell that is executing
    # this very command string. llava_server.py is the retired
    # transformers wrapper; still killed for cleanup on old deployments.
    ssh_run(
        "pkill -u \"$(id -u)\" -f '[v]llm serve' ; "
        "pkill -u \"$(id -u)\" -f '[l]lava_server.py' ; "
        "sleep 3 ; "
        "pkill -9 -u \"$(id -u)\" -f '[v]llm serve|[E]ngineCore|[l]lava_server.py' ; true",
        timeout=30,
    )


@app.get("/api/gpu/status")
async def gpu_status() -> dict[str, Any]:
    with STATE.lock:
        phase, model, method = STATE.phase, STATE.model, STATE.method
        detail = STATE.detail
    if phase in ("starting", "ready") and model is not None:
        ensure_tunnel()
        if await probe_health(model):
            with STATE.lock:
                STATE.phase = "ready"
                STATE.detail = ""
            phase, detail = "ready", ""
        elif phase == "ready":
            # Was ready, now unhealthy: report starting so the UI shows
            # progress instead of silently failing runs.
            phase = "starting"
            detail = "server not responding; waiting for it to come back"
    return {"phase": phase, "model": model, "method": method, "detail": detail}


@app.post("/api/gpu/start")
async def gpu_start(req: StartRequest) -> dict[str, Any]:
    with STATE.lock:
        if STATE.phase == "starting":
            raise HTTPException(409, "A server is already starting")
        STATE.phase = "starting"
        STATE.model = req.model
        STATE.method = req.method
        STATE.detail = f"launching on {HOST_LABEL} (first start downloads weights)"
    try:
        stop_all_remote()
        res = ssh_run(launch_command(req), timeout=60)
        if res.returncode != 0 or "LAUNCHED" not in res.stdout:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "launch failed")
        ensure_tunnel()
    except Exception as exc:  # surface the SSH error to the UI
        with STATE.lock:
            STATE.phase = "error"
            STATE.detail = str(exc)
        raise HTTPException(500, f"GPU start failed: {exc}") from exc
    return await gpu_status()


@app.post("/api/gpu/stop")
async def gpu_stop() -> dict[str, Any]:
    with STATE.lock:
        STATE.phase = "stopping"
    try:
        stop_all_remote()
    finally:
        drop_tunnel()
        with STATE.lock:
            STATE.phase = "stopped"
            STATE.model = None
            STATE.method = None
            STATE.detail = ""
    return await gpu_status()


def chat_completion_body(req: InferRequest, pruned: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODELS[req.model]["hf_id"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": req.image}},
                    {"type": "text", "text": req.prompt},
                ],
            }
        ],
        "max_tokens": req.max_new_tokens,
        "temperature": 0,
    }
    if pruned and req.retention < 1.0:
        body["token_pruning"] = req.retention
    return body


async def infer_vllm(req: InferRequest) -> dict[str, Any]:
    port = MODELS[req.model]["port"]
    url = f"http://localhost:{port}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=300.0) as client:
        t0 = time.perf_counter()
        r = await client.post(url, json=chat_completion_body(req, pruned=True))
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:
            raise HTTPException(502, f"vLLM error {r.status_code}: {r.text[:500]}")
        resp = r.json()

        baseline_answer = None
        baseline_prompt_tokens = None
        baseline_ttft_ms = None
        if req.with_baseline:
            rb = await client.post(url, json=chat_completion_body(req, pruned=False))
            if rb.status_code == 200:
                b = rb.json()
                baseline_answer = b["choices"][0]["message"]["content"]
                baseline_prompt_tokens = (b.get("usage") or {}).get("prompt_tokens")
                baseline_ttft_ms = (b.get("metrics") or {}).get(
                    "time_to_first_token_ms"
                )

    metadata = (resp.get("token_pruning_metadata") or [None])[0]
    return {
        "answer": resp["choices"][0]["message"]["content"],
        "usage": resp.get("usage"),
        "metadata": metadata,
        "baseline_answer": baseline_answer,
        "baseline_prompt_tokens": baseline_prompt_tokens,
        "elapsed_s": elapsed,
        # Server-side TTFT (vision encoder + prefill + first token),
        # excludes upload/network time.
        "ttft_ms": (resp.get("metrics") or {}).get("time_to_first_token_ms"),
        "baseline_ttft_ms": baseline_ttft_ms,
    }


@app.post("/api/infer")
async def infer(req: InferRequest) -> dict[str, Any]:
    ensure_tunnel()
    if not await probe_health(req.model):
        raise HTTPException(503, "Model server is not ready — start the GPU first")
    return await infer_vllm(req)


# Serve the built frontend if present (`npm run build` in visualizer/web).
_dist = Path(__file__).parent.parent / "web" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8300)
