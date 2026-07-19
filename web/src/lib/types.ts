export type ModelKey = "qwen2_5_vl" | "llava_1_5";
export type MethodKey = "hiprune" | "hydart";

export interface ModelInfo {
  key: ModelKey;
  label: string;
  hfId: string;
  defaultObjectLayer: number;
}

export const MODELS: ModelInfo[] = [
  {
    key: "qwen2_5_vl",
    label: "Qwen2.5-VL-3B",
    hfId: "Qwen/Qwen2.5-VL-3B-Instruct",
    defaultObjectLayer: 16,
  },
  {
    key: "llava_1_5",
    label: "LLaVA-1.5-7B",
    hfId: "liuhaotian/llava-v1.5-7b",
    defaultObjectLayer: 9,
  },
];

export interface Params {
  retention: number;
  alpha: number;
  objectLayer: number; // 0 = model default
  maxNewTokens: number;
  lambdaSeed: number;
  lambdaPick: number;
  prompt: string;
}

export const DEFAULT_PARAMS: Params = {
  retention: 0.223,
  alpha: 0.1,
  objectLayer: 0,
  maxNewTokens: 128,
  lambdaSeed: 0.1,
  lambdaPick: 0.5,
  prompt: "Describe this image in detail.",
};

/** Per-image pruning metadata returned by the servers (same shape for
 * the vLLM fork and the LLaVA wrapper). */
export interface PruningMetadata {
  method?: MethodKey;
  grid: [number, number];
  num_tokens: number;
  retention: number;
  object_layer: number;
  alpha: number;
  pruned: number[];
  anchors: number[];
  buffers: number[];
  registers?: number[];
  diverse?: number[];
  lambda_seed?: number;
  lambda_pick?: number;
  mean_attention?: Record<string, Record<string, number | null>>;
  similarity?: Record<string, number>;
  /** Per-token arrays (index = soft-token index) for hover tooltips. */
  scores?: {
    object_layer?: number[];
    deep_layer?: number[];
    similarity?: number[];
  };
}

export interface InferResult {
  answer: string;
  usage: { prompt_tokens?: number; completion_tokens?: number } | null;
  metadata: PruningMetadata | null;
  baseline_answer?: string | null;
  baseline_prompt_tokens?: number | null;
  elapsed_s?: number;
  /** Server-side time to first token (vision encoder + prefill),
   * excluding upload/network time. */
  ttft_ms?: number | null;
  baseline_ttft_ms?: number | null;
}

export type GpuPhase =
  | "stopped"
  | "starting"
  | "ready"
  | "stopping"
  | "error"
  | "unknown";

export interface GpuStatus {
  phase: GpuPhase;
  model: ModelKey | null;
  method: MethodKey | null;
  detail?: string;
}
