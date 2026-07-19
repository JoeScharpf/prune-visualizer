import type {
  GpuStatus,
  InferResult,
  MethodKey,
  ModelKey,
  Params,
} from "./types";

async function jsonOrThrow(res: Response) {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* keep the status text */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function getGpuStatus(): Promise<GpuStatus> {
  return jsonOrThrow(await fetch("/api/gpu/status"));
}

export async function startGpu(
  model: ModelKey,
  method: MethodKey,
  lambdaSeed: number,
  lambdaPick: number
): Promise<GpuStatus> {
  return jsonOrThrow(
    await fetch("/api/gpu/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        method,
        lambda_seed: lambdaSeed,
        lambda_pick: lambdaPick,
      }),
    })
  );
}

export async function stopGpu(): Promise<GpuStatus> {
  return jsonOrThrow(await fetch("/api/gpu/stop", { method: "POST" }));
}

export async function infer(
  imageDataUrl: string,
  model: ModelKey,
  method: MethodKey,
  params: Params,
  withBaseline: boolean
): Promise<InferResult> {
  return jsonOrThrow(
    await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: imageDataUrl,
        model,
        method,
        prompt: params.prompt,
        retention: params.retention,
        alpha: params.alpha,
        object_layer: params.objectLayer,
        max_new_tokens: params.maxNewTokens,
        lambda_seed: params.lambdaSeed,
        lambda_pick: params.lambdaPick,
        with_baseline: withBaseline,
      }),
    })
  );
}
