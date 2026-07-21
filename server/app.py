"""HiPrune/HyDART visualizer backend.

Three jobs:

1. GPU control: launch/stop the model servers. "Start GPU" brings up
   one vLLM server per supported model (Qwen2.5-VL, LLaVA-1.5, Gemma-4)
   side by side, each with a capped ``--gpu-memory-utilization`` slice,
   so switching models in the UI never requires a restart. "Stop GPU"
   puts the servers to *sleep* (vLLM sleep level 1: weights offload to
   host RAM, KV cache freed, GPU drops to ~0), so the next Start wakes
   them in seconds instead of a minutes-long cold boot; ``?hard=true``
   on /gpu/stop still kills everything (required after vLLM-fork
   deploys — sleeping processes run old code). In ``ssh`` mode the
   servers run on a remote GPU host reached over SSH, with an SSH
   tunnel forwarding their ports; in ``local`` mode this backend runs
   on the GPU machine itself and launches them directly.
2. Inference proxy: forward an image + prompt + pruning params to the
   requested model's server (OpenAI-compatible chat completions).
3. Static serving of the built frontend (visualizer/web/dist).

Deployment config (environment variables):
- HIPRUNE_HOST_MODE: "ssh" (default; backend on a laptop, GPU remote)
  or "local" (backend on the GPU machine).
- HIPRUNE_SSH_HOST: user@host of the GPU machine (ssh mode only).
- HIPRUNE_REMOTE_DIR: directory on the GPU machine holding venv/ and the
  vLLM fork (default ~/hiprune).
- HIPRUNE_GPU_INDEX: which physical GPU to use (default 0).

Server-side facts this encodes (from the deployed vLLM fork):
- The method and its knobs are per-request (`token_pruning_method` /
  `token_pruning_params` chat-completions fields), like the retention
  ratio (`token_pruning`). All models are resident at once, so neither
  a model nor a method change ever restarts anything.
- Each vLLM server requests ``total_memory * gpu_memory_utilization``
  and validates it against *free* memory at startup, so the per-model
  caps below are disjoint slices of the card and must sum below ~0.9.
  Servers are launched sequentially so each startup check sees the
  previous servers' slices already claimed.
- Alpha and object layer are constants in the fork (paper defaults per
  model), so they are display-only in the UI.

Usage:
    pip install -r requirements.txt
    uvicorn app:app --port 8300
"""

from __future__ import annotations

import asyncio
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
        # One image + a short prompt; profiling and CUDA graph work scale
        # with this, so keep it small for fast cold starts.
        "max_model_len": 8192,
        "gpu_mem_util": 0.15,
    },
    "llava_1_5": {
        "hf_id": "llava-hf/llava-1.5-7b-hf",
        "port": LLAVA_PORT,
        "log": "serve_visualizer_llava.log",
        "max_model_len": 4096,
        "gpu_mem_util": 0.20,
    },
    "gemma4": {
        "hf_id": "google/gemma-4-e4b-it",
        "port": GEMMA_PORT,
        "log": "serve_visualizer_gemma.log",
        "max_model_len": 8192,
        "gpu_mem_util": 0.20,
    },
}

ModelKey = Literal["qwen2_5_vl", "llava_1_5", "gemma4"]
MethodKey = Literal["hiprune", "hydart", "hiprune_pp", "dart"]

# Per-model cap on waiting for /health after a launch. Weights are
# cached on the box after the first start, so a healthy load is a few
# minutes; the cap only bounds how long a broken server blocks the
# launch sequence.
MODEL_START_TIMEOUT_S = int(os.environ.get("HIPRUNE_START_TIMEOUT_S", "900"))


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
    pivot_image: int = Field(default=4, ge=1, le=64)
    pivot_text: int = Field(default=4, ge=0, le=64)
    with_baseline: bool = False


class GpuState:
    """In-memory view of what we launched; verified against reality by
    /gpu/status probes.

    ``models`` maps every model key to its own phase (``stopped`` |
    ``starting`` | ``ready`` | ``sleeping`` | ``error``); ``phase`` is
    the aggregate: ``starting`` while the launch sequence is in flight,
    ``ready`` once it finishes with at least one healthy model,
    ``stopped`` when nothing serves (sleeping servers count as stopped —
    with a fast start available), ``error`` only if all models failed.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.phase: str = "unknown"
        self.models: dict[str, str] = {key: "unknown" for key in MODELS}
        self.detail: str = ""
        self.tunnel: subprocess.Popen | None = None
        self.start_thread: threading.Thread | None = None


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


def base_url(model: str) -> str:
    return f"http://localhost:{MODELS[model]['port']}"


async def probe_health(model: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base_url(model)}/health")
            return r.status_code == 200
    except httpx.HTTPError:
        return False


def probe_health_sync(model: str) -> bool:
    """Blocking twin of probe_health for the launcher thread."""
    try:
        r = httpx.get(f"{base_url(model)}/health", timeout=3.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


async def probe_sleeping(model: str) -> bool:
    """True iff the server answers /is_sleeping with is_sleeping=true.
    Unreachable or pre-sleep-mode servers report False."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base_url(model)}/is_sleeping")
            return r.status_code == 200 and bool(r.json().get("is_sleeping"))
    except (httpx.HTTPError, ValueError):
        return False


def probe_sleeping_sync(model: str) -> bool:
    try:
        r = httpx.get(f"{base_url(model)}/is_sleeping", timeout=3.0)
        return r.status_code == 200 and bool(r.json().get("is_sleeping"))
    except (httpx.HTTPError, ValueError):
        return False


async def probe_serving(model: str) -> bool:
    """Usable for inference: healthy AND awake. /health returns 200 even
    while the engine sleeps (it only 503s on a dead engine), so health
    alone can't gate requests anymore."""
    return await probe_health(model) and not await probe_sleeping(model)


def probe_serving_sync(model: str) -> bool:
    return probe_health_sync(model) and not probe_sleeping_sync(model)


def kill_remote_model(model: str) -> None:
    """Kill only this model's server (matched by its --port), so a stale
    or hung process can't hold the port against a relaunch. Never the
    global kill: that would tear down the healthy sibling servers. The
    bracketed first letter keeps pkill -f from matching the shell
    executing this command string; EngineCore children exit on their own
    when the serve parent dies."""
    port = MODELS[model]["port"]
    ssh_run(
        f"pkill -u \"$(id -u)\" -f '[p]ort {port}' ; sleep 2 ; "
        f"pkill -9 -u \"$(id -u)\" -f '[p]ort {port}' ; true",
        timeout=30,
    )


def launch_command(model: str) -> str:
    """Build the remote launch line.

    The whole launch script is backgrounded with every fd detached from
    the SSH socket, so the ssh call returns immediately instead of
    waiting on the server process (the remote shell otherwise lingers as
    the job's parent and holds the session open).
    """
    # VLLM_SERVER_DEV_MODE exposes the /sleep, /wake_up and /is_sleeping
    # routes — but also admin routers (rpc/rlhf/cache). --host 127.0.0.1
    # keeps all of it off the public interface; the backend and the SSH
    # tunnel both reach the servers via localhost anyway.
    env = (
        f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={GPU_INDEX} "
        "VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 "
        "VLLM_SERVER_DEV_MODE=1"
    )
    cfg = MODELS[model]
    script = (
        f"cd {REMOTE_DIR} && source venv/bin/activate && "
        f"{env} exec vllm serve {cfg['hf_id']} "
        f"--max-model-len {cfg['max_model_len']} "
        f"--enable-hiprune --enable-per-request-metrics "
        f"--enable-sleep-mode --host 127.0.0.1 "
        # Single-user demo: few sequences, few CUDA graph sizes — cuts
        # the profiling/capture part of the cold start.
        f"--max-num-seqs 8 --cudagraph-capture-sizes 1 2 4 8 "
        f"--gpu-memory-utilization {cfg['gpu_mem_util']} --port {cfg['port']}"
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


def _log_tail(model: str) -> str:
    res = ssh_run(f"tail -n 5 {REMOTE_DIR}/{MODELS[model]['log']} 2>/dev/null", timeout=15)
    return res.stdout.strip()[-500:] if res.returncode == 0 else ""


# Waking (host RAM -> GPU weight copy) is seconds, not minutes; a short
# deadline keeps a wedged wake from stalling the launch sequence.
WAKE_TIMEOUT_S = 180


def _wake_model_sync(model: str) -> bool:
    """Wake a sleeping server; True once it serves again. False falls
    back to the cold relaunch path."""
    try:
        r = httpx.post(f"{base_url(model)}/wake_up", timeout=30.0)
        if r.status_code != 200:
            return False
    except httpx.HTTPError:
        return False
    deadline = time.time() + WAKE_TIMEOUT_S
    while time.time() < deadline:
        with STATE.lock:
            if STATE.phase in ("stopping", "stopped"):
                return False
        if probe_serving_sync(model):
            return True
        time.sleep(2)
    return False


def _start_all_models() -> None:
    """Launcher thread body: bring up every model server in sequence.

    Per model: already serving -> ready; healthy but asleep -> wake
    (seconds); otherwise cold launch (kill stale process, relaunch, poll
    health). Sequential on purpose — each vLLM cold start validates its
    requested memory slice against currently *free* GPU memory, so the
    checks must see the previous servers' slices already claimed. One
    model failing never blocks the others.
    """
    first_error = ""
    for key in MODELS:
        with STATE.lock:
            if STATE.phase in ("stopping", "stopped"):
                return  # user hit Stop mid-sequence; leave state to gpu_stop
        if probe_serving_sync(key):
            with STATE.lock:
                STATE.models[key] = "ready"
            continue
        if probe_health_sync(key) and probe_sleeping_sync(key):
            with STATE.lock:
                STATE.models[key] = "starting"
                STATE.detail = f"waking {key} on {HOST_LABEL}"
            if _wake_model_sync(key):
                with STATE.lock:
                    STATE.models[key] = "ready"
                continue
            with STATE.lock:
                if STATE.phase in ("stopping", "stopped"):
                    return
            # Wake failed: fall through to a cold relaunch.
        with STATE.lock:
            STATE.models[key] = "starting"
            STATE.detail = f"starting {key} on {HOST_LABEL}"
        try:
            kill_remote_model(key)
            res = ssh_run(launch_command(key), timeout=60)
            if res.returncode != 0 or "LAUNCHED" not in res.stdout:
                raise RuntimeError(
                    res.stderr.strip() or res.stdout.strip() or "launch failed"
                )
            deadline = time.time() + MODEL_START_TIMEOUT_S
            while time.time() < deadline:
                with STATE.lock:
                    if STATE.phase in ("stopping", "stopped"):
                        # Stop landed mid-poll: the server we're waiting
                        # on was just killed. Bail out instead of probing
                        # a dead port until the timeout (which would also
                        # block a follow-up Start behind the 409 guard).
                        return
                if probe_health_sync(key):
                    break
                time.sleep(5)
            else:
                raise RuntimeError(
                    f"not healthy after {MODEL_START_TIMEOUT_S}s: {_log_tail(key)}"
                )
            with STATE.lock:
                STATE.models[key] = "ready"
        except Exception as exc:
            if not first_error:
                first_error = f"{key}: {exc}"
            with STATE.lock:
                STATE.models[key] = "error"

    with STATE.lock:
        if STATE.phase in ("stopping", "stopped"):
            return  # a Stop won the race; don't overwrite its state
        any_ready = any(p == "ready" for p in STATE.models.values())
        STATE.phase = "ready" if any_ready else "error"
        STATE.detail = "" if any_ready else (first_error or "all model servers failed")


@app.get("/api/gpu/status")
async def gpu_status() -> dict[str, Any]:
    with STATE.lock:
        phase = STATE.phase
        launching = STATE.start_thread is not None and STATE.start_thread.is_alive()
    # "unknown" is probed too: after a backend restart (every deploy) the
    # state resets, and only a probe can rediscover live or sleeping
    # servers.
    if phase in ("starting", "ready", "unknown"):
        ensure_tunnel()
        for key in MODELS:
            healthy = await probe_health(key)
            sleeping = healthy and await probe_sleeping(key)
            with STATE.lock:
                if healthy and not sleeping:
                    STATE.models[key] = "ready"
                elif sleeping:
                    # While the launcher runs it owns transitions (it may
                    # be about to wake this very server).
                    if not launching:
                        STATE.models[key] = "sleeping"
                elif not launching and STATE.models[key] == "ready":
                    # Was ready, now unhealthy: report starting so the UI
                    # shows progress instead of silently failing runs.
                    STATE.models[key] = "starting"
        if not launching:
            with STATE.lock:
                vals = list(STATE.models.values())
                if any(v == "ready" for v in vals):
                    STATE.phase = "ready"
                    if STATE.detail.startswith(("starting ", "waking ")):
                        STATE.detail = ""
                elif any(v == "sleeping" for v in vals):
                    # Nothing serving but servers are asleep: that's a
                    # stopped GPU with a fast start available.
                    STATE.phase = "stopped"
    with STATE.lock:
        return {
            "phase": STATE.phase,
            "models": dict(STATE.models),
            "detail": STATE.detail,
        }


@app.post("/api/gpu/start")
async def gpu_start() -> dict[str, Any]:
    """Start every model server that isn't already healthy. Returns
    immediately; the UI polls /gpu/status for per-model progress."""
    with STATE.lock:
        if STATE.start_thread is not None and STATE.start_thread.is_alive():
            raise HTTPException(409, "Servers are already starting")
        STATE.phase = "starting"
        STATE.detail = f"launching on {HOST_LABEL} (first start downloads weights)"
        for key in MODELS:
            STATE.models[key] = "starting"
    # Tunnel must be up before the launcher thread probes health, or a
    # healthy server would look dead and get needlessly relaunched.
    ensure_tunnel()
    with STATE.lock:
        STATE.start_thread = threading.Thread(target=_start_all_models, daemon=True)
        STATE.start_thread.start()
    return await gpu_status()


async def _sleep_model(model: str) -> bool:
    """Put one server to sleep (level 1: weights -> host RAM, KV cache
    freed). True once /is_sleeping confirms."""
    try:
        # Offloading tens of GB of weights takes a few seconds; give the
        # call room, then confirm.
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{base_url(model)}/sleep", params={"level": "1"})
        if r.status_code != 200:
            return False
    except httpx.HTTPError:
        return False
    deadline = time.time() + 60
    while time.time() < deadline:
        if await probe_sleeping(model):
            return True
        await asyncio.sleep(1)
    return False


@app.post("/api/gpu/stop")
async def gpu_stop(hard: bool = False) -> dict[str, Any]:
    """Default (UI button): put healthy servers to *sleep* — the GPU is
    freed but the processes keep their weights in host RAM, so the next
    Start is a wake (seconds) instead of a cold boot (minutes).

    ``?hard=true``: the old kill-everything path. Needed after vLLM-fork
    deploys (sleeping processes run old code) and as a recovery hammer.
    """
    with STATE.lock:
        STATE.phase = "stopping"
    try:
        if hard:
            stop_all_remote()
            with STATE.lock:
                for key in MODELS:
                    STATE.models[key] = "stopped"
        else:
            for key in MODELS:
                if await probe_health(key):
                    slept = await probe_sleeping(key) or await _sleep_model(key)
                    if not slept:
                        # e.g. stale pre-sleep-mode process: fall back to
                        # this model's port-scoped kill.
                        kill_remote_model(key)
                    with STATE.lock:
                        STATE.models[key] = "sleeping" if slept else "stopped"
                else:
                    with STATE.lock:
                        STATE.models[key] = "stopped"
    finally:
        if hard:
            # Soft stop keeps the tunnel: sleeping servers still get
            # probed by /gpu/status.
            drop_tunnel()
        with STATE.lock:
            STATE.phase = "stopped"
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
        body["token_pruning_method"] = req.method
        body["token_pruning_params"] = {
            "lambda_seed": req.lambda_seed,
            "lambda_pick": req.lambda_pick,
            "beta": req.beta,
            "pivot_image": req.pivot_image,
            "pivot_text": req.pivot_text,
        }
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
    # Serving = healthy AND awake; a sleeping engine answers /health 200
    # but can't run requests.
    if not await probe_serving(req.model):
        raise HTTPException(503, "Model server is not ready — start the GPU first")
    return await infer_vllm(req)


# Serve the built frontend if present (`npm run build` in visualizer/web).
_dist = Path(__file__).parent.parent / "web" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8300)
