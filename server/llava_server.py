"""LLaVA-1.5 HiPrune/HyDART server for the visualizer (runs on gpu3).

Wraps the HiPrune authors' LLaVA fork (cloned to ~/hiprune/HiPrune,
pinned commit) behind one HTTP endpoint with the same response shape as
the vLLM fork, so the visualizer frontend is backend-agnostic:

    POST /generate {image, prompt, method, retention, alpha,
                    object_layer, max_new_tokens, lambda_seed,
                    lambda_pick, with_baseline}
    -> {answer, usage, token_pruning_metadata, baseline_answer,
        baseline_prompt_tokens}

Method notes:
- hiprune: the fork's generate prunes internally, driven by the
  HIPRUNE_RETENTION / HIPRUNE_OBJECT_LAYER / HIPRUNE_ALPHA env vars it
  reads at call time. The token categories for the overlay are computed
  externally with the exact same arithmetic (differential-tested in the
  Colab); scores stay in fp16 so topk tie-breaking matches the model.
- hydart: HiPrune anchors/buffers + greedy-MMR diverse fill over the
  mm_projector embeddings (the vectors the LLM consumes), mirroring the
  HyDART Qwen implementation. The keep mask is injected by overriding
  the index mask that encode_images returns.
- baseline: HIPRUNE_RETENTION=576 keeps every token (mask all-True),
  which the Colab verified is a no-op path.

Deps (llava_venv): torch, transformers==4.37.2, accelerate==0.27.2,
sentencepiece, protobuf, fastapi, uvicorn, pillow.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
from typing import Any

HIPRUNE_LLAVA_DIR = os.path.expanduser("~/hiprune/HiPrune/LLaVA")
sys.path.insert(0, HIPRUNE_LLAVA_DIR)

MODEL_ID = "liuhaotian/llava-v1.5-7b"
GRID_W = GRID_H = 24
N_TOKENS = 576
DEFAULT_OBJECT_LAYER = 9  # paper README table for LLaVA-1.5

os.environ.setdefault("HIPRUNE_RETENTION", str(N_TOKENS))
os.environ.setdefault("HIPRUNE_OBJECT_LAYER", str(DEFAULT_OBJECT_LAYER))
os.environ.setdefault("HIPRUNE_ALPHA", "0.1")

import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList

# Compat shim from the Colab: transformers 4.37.2 may reference the
# removed private pytree API on modern torch.
import torch.utils._pytree as _pytree

if not hasattr(_pytree, "_register_pytree_node") and hasattr(
    _pytree, "register_pytree_node"
):
    _pytree._register_pytree_node = _pytree.register_pytree_node

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

app = FastAPI(title="LLaVA HiPrune/HyDART server")

print(f"Loading {MODEL_ID} (fp16)...", flush=True)
TOKENIZER, MODEL, IMAGE_PROCESSOR, _ = load_pretrained_model(
    MODEL_ID,
    None,
    get_model_name_from_path(MODEL_ID),
    device_map="auto",
    # sdpa for the LM; the fork's CLIP tower requests output_attentions
    # per-call, which transformers serves by falling back to eager there.
    attn_implementation="sdpa",
)
MODEL.eval()
print("Model loaded.", flush=True)

# One request at a time: generation owns the GPU and the env-var plumbing.
RUN_LOCK = threading.Lock()

# HyDART keep-mask injection point: when set, encode_images returns this
# mask instead of the fork's internal HiPrune selection.
_FORCED_KEEP_MASK: torch.Tensor | None = None

_ORIG_ENCODE_IMAGES = type(MODEL).encode_images


def _patched_encode_images(self, images):
    feats, masks = _ORIG_ENCODE_IMAGES(self, images)
    if _FORCED_KEEP_MASK is not None:
        masks = _FORCED_KEEP_MASK.to(masks.device).view(1, -1)
    return feats, masks


type(MODEL).encode_images = _patched_encode_images


class GenerateRequest(BaseModel):
    image: str  # data URL or bare base64
    prompt: str
    method: str = "hiprune"
    retention: float = 0.223
    alpha: float = 0.1
    object_layer: int = 0  # 0 = model default
    max_new_tokens: int = 128
    lambda_seed: float = 0.1
    lambda_pick: float = 0.5
    with_baseline: bool = False


def decode_image(data: str) -> Image.Image:
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    return ImageOps.exif_transpose(img.convert("RGB"))


def hiprune_select(shallow_scores, deep_scores, n_toks, gw, budget, alpha):
    """Exact selection arithmetic of the fork's encode_images (fp16)."""
    deep = deep_scores.clone()
    shallow_token_num = round(budget * alpha / 5)

    anchor_idx = torch.topk(shallow_scores, k=shallow_token_num).indices
    shallow_all = torch.cat(
        [anchor_idx, anchor_idx - 1, anchor_idx + 1, anchor_idx - gw, anchor_idx + gw]
    )
    shallow_all = shallow_all.clamp(0, n_toks - 1)
    shallow_all = torch.unique(shallow_all, sorted=False)
    buffer_idx = shallow_all[~torch.isin(shallow_all, anchor_idx)]

    deep_token_num = budget - shallow_all.shape[0]
    selected_mask = torch.zeros(n_toks, dtype=torch.bool, device=deep.device)
    selected_mask.scatter_(0, shallow_all, 1)
    deep -= selected_mask.int()
    register_idx = torch.topk(deep, k=deep_token_num).indices

    kept_mask = selected_mask.clone()
    kept_mask[register_idx] = True
    return anchor_idx, buffer_idx, register_idx, kept_mask


def hydart_select(shallow_scores, embeddings, n_toks, gw, budget, alpha,
                  lambda_seed, lambda_pick):
    """HiPrune anchors/buffers + greedy-MMR diverse fill (HyDART.py math)."""
    shallow_token_num = round(budget * alpha / 5)
    anchor_idx = torch.topk(shallow_scores, k=shallow_token_num).indices
    shallow_all = torch.cat(
        [anchor_idx, anchor_idx - 1, anchor_idx + 1, anchor_idx - gw, anchor_idx + gw]
    )
    shallow_all = shallow_all.clamp(0, n_toks - 1)
    shallow_all = torch.unique(shallow_all, sorted=False)
    buffer_idx = shallow_all[~torch.isin(shallow_all, anchor_idx)]

    if shallow_all.shape[0] > budget:
        raise HTTPException(
            400,
            f"anchors+buffers ({shallow_all.shape[0]}) exceed the keep budget "
            f"({budget}); lower alpha or raise retention",
        )

    emb = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    attn = shallow_scores.float()
    attn_hat = (attn - attn.min()) / (attn.max() - attn.min()).clamp_min(1e-12)

    if shallow_all.numel():
        r_seed = (emb @ emb[shallow_all].T).max(dim=-1).values.clamp_(0.0, 1.0)
    else:
        r_seed = torch.zeros(n_toks, device=emb.device)
    r_pick = torch.zeros_like(r_seed)
    base_score = attn_hat - lambda_seed * r_seed

    blocked = torch.zeros(n_toks, dtype=torch.bool, device=emb.device)
    blocked[shallow_all] = True

    picks: list[torch.Tensor] = []
    pick_redundancy: list[torch.Tensor] = []
    for _ in range(budget - shallow_all.shape[0]):
        score = base_score - lambda_pick * r_pick
        score[blocked] = float("-inf")
        idx = torch.argmax(score)
        picks.append(idx)
        pick_redundancy.append(torch.maximum(r_seed[idx], r_pick[idx]))
        blocked[idx] = True
        r_pick = torch.maximum(r_pick, (emb @ emb[idx]).clamp_(0.0, 1.0))
    diverse_idx = (
        torch.stack(picks) if picks
        else torch.empty(0, dtype=anchor_idx.dtype, device=emb.device)
    )

    kept_mask = torch.zeros(n_toks, dtype=torch.bool, device=shallow_scores.device)
    kept_mask[shallow_all] = True
    kept_mask[diverse_idx] = True

    sim_stats = torch.maximum(r_seed, r_pick)
    sim_stats[shallow_all] = 1.0
    if picks:
        sim_stats[diverse_idx] = torch.stack(pick_redundancy)
    return anchor_idx, buffer_idx, diverse_idx, kept_mask, sim_stats


def mean_attention(scores: torch.Tensor, idx: torch.Tensor) -> float | None:
    return float(scores.float()[idx].mean()) if idx.numel() else None


class _FirstTokenTimer(StoppingCriteria):
    """Timestamps each generated token; never stops generation. The first
    stamp marks the end of prefill (vision encoder + full forward + first
    token), i.e. TTFT."""

    def __init__(self) -> None:
        self.stamps: list[float] = []

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        torch.cuda.synchronize()
        self.stamps.append(time.perf_counter())
        return False


def generate_once(
    input_ids, images_tensor, image_sizes, max_new_tokens
) -> tuple[str, float | None]:
    """Run one generation; returns (text, ttft_ms)."""
    timer = _FirstTokenTimer()
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.inference_mode():
        new_ids, _, _ = MODEL.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([timer]),
        )
    text = TOKENIZER.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    ttft_ms = (timer.stamps[0] - t_start) * 1000 if timer.stamps else None
    return text, ttft_ms


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    global _FORCED_KEEP_MASK
    if req.method not in ("hiprune", "hydart"):
        raise HTTPException(400, f"unknown method {req.method!r}")
    object_layer = req.object_layer if req.object_layer > 0 else DEFAULT_OBJECT_LAYER
    budget = max(1, min(N_TOKENS, round(N_TOKENS * req.retention)))

    pil_image = decode_image(req.image)

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + req.prompt)
    conv.append_message(conv.roles[1], None)
    input_ids = (
        tokenizer_image_token(
            conv.get_prompt(), TOKENIZER, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .cuda()
    )
    images_tensor = process_images([pil_image], IMAGE_PROCESSOR, MODEL.config).to(
        "cuda", dtype=torch.float16
    )
    image_sizes = [pil_image.size]
    n_text_tokens = int(input_ids.shape[1]) - 1  # minus the image placeholder

    with RUN_LOCK:
        os.environ["HIPRUNE_OBJECT_LAYER"] = str(object_layer)
        os.environ["HIPRUNE_ALPHA"] = str(req.alpha)

        # Selection pass: deterministic vision-tower attentions + features.
        with torch.inference_mode():
            image_features, vt_attentions = MODEL.get_model().get_vision_tower()(
                images_tensor
            )
        sel_layer = MODEL.config.mm_vision_select_layer  # -2 for LLaVA-1.5
        shallow = vt_attentions[object_layer - 1].mean(dim=1).mean(dim=1)[0, 1:]
        deep = vt_attentions[sel_layer].mean(dim=1).mean(dim=1)[0, 1:]

        similarity = None
        if req.method == "hiprune":
            anchor_idx, buffer_idx, third_idx, kept_mask = hiprune_select(
                shallow, deep, N_TOKENS, GRID_W, budget, req.alpha
            )
            third_key = "registers"
        else:
            with torch.inference_mode():
                embeddings = MODEL.get_model().mm_projector(image_features)[0]
            anchor_idx, buffer_idx, third_idx, kept_mask, sim_stats = hydart_select(
                shallow, embeddings, N_TOKENS, GRID_W, budget, req.alpha,
                req.lambda_seed, req.lambda_pick,
            )
            third_key = "diverse"
            pruned_sim = sim_stats[~kept_mask]
            diverse_sim = (
                sim_stats[third_idx] if third_idx.numel() else torch.zeros(0)
            )
            similarity = {
                "diverse_at_selection": float(diverse_sim.mean())
                if diverse_sim.numel()
                else None,
                "pruned_vs_kept": float(pruned_sim.mean())
                if pruned_sim.numel()
                else None,
            }

        # Generation. hiprune: the fork prunes internally from the env
        # budget. hydart: force our keep mask through encode_images (the
        # fork still needs a valid env budget for its own, discarded topk).
        try:
            if req.method == "hiprune":
                os.environ["HIPRUNE_RETENTION"] = str(budget)
                _FORCED_KEEP_MASK = None
            else:
                os.environ["HIPRUNE_RETENTION"] = str(budget)
                _FORCED_KEEP_MASK = kept_mask
            answer, ttft_ms = generate_once(
                input_ids, images_tensor, image_sizes, req.max_new_tokens
            )

            baseline_answer = None
            baseline_prompt_tokens = None
            baseline_ttft_ms = None
            if req.with_baseline:
                os.environ["HIPRUNE_RETENTION"] = str(N_TOKENS)
                _FORCED_KEEP_MASK = None
                baseline_answer, baseline_ttft_ms = generate_once(
                    input_ids, images_tensor, image_sizes, req.max_new_tokens
                )
                baseline_prompt_tokens = n_text_tokens + N_TOKENS
        finally:
            _FORCED_KEEP_MASK = None
            os.environ["HIPRUNE_RETENTION"] = str(N_TOKENS)
        torch.cuda.empty_cache()

    kept_idx = kept_mask.nonzero(as_tuple=True)[0]
    pruned_idx = (~kept_mask).nonzero(as_tuple=True)[0]
    categories = {
        "anchor": anchor_idx,
        "buffer": buffer_idx,
        third_key.rstrip("s") if third_key == "registers" else third_key: third_idx,
        "kept": kept_idx,
        "pruned": pruned_idx,
    }
    metadata: dict[str, Any] = {
        "method": req.method,
        "grid": [GRID_W, GRID_H],
        "num_tokens": N_TOKENS,
        "retention": req.retention,
        "object_layer": object_layer,
        "alpha": req.alpha,
        "pruned": pruned_idx.tolist(),
        "anchors": anchor_idx.tolist(),
        "buffers": buffer_idx.tolist(),
        third_key: third_idx.tolist(),
        "mean_attention": {
            "object_layer": {
                name: mean_attention(shallow, idx) for name, idx in categories.items()
            },
            "deep_layer": {
                name: mean_attention(deep, idx) for name, idx in categories.items()
            },
        },
        # Per-token arrays for hover tooltips (index = token index).
        "scores": {
            "object_layer": shallow.float().tolist(),
            "deep_layer": deep.float().tolist(),
        },
    }
    if req.method == "hydart":
        metadata["lambda_seed"] = req.lambda_seed
        metadata["lambda_pick"] = req.lambda_pick
        metadata["similarity"] = similarity
        metadata["scores"]["similarity"] = sim_stats.float().tolist()

    n_kept = int(kept_idx.numel())
    return {
        "answer": answer,
        "usage": {
            "prompt_tokens": n_text_tokens + n_kept,
            "completion_tokens": None,
        },
        "token_pruning_metadata": [metadata],
        "baseline_answer": baseline_answer,
        "baseline_prompt_tokens": baseline_prompt_tokens,
        "ttft_ms": ttft_ms,
        "baseline_ttft_ms": baseline_ttft_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8125)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
