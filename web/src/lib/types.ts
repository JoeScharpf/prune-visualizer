export type ModelKey = "qwen2_5_vl" | "llava_1_5" | "gemma4";
export type MethodKey = "hiprune" | "hydart" | "hiprune_pp";

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
    hfId: "llava-hf/llava-1.5-7b-hf",
    defaultObjectLayer: 9,
  },
  {
    key: "gemma4",
    label: "Gemma 4 E4B",
    hfId: "google/gemma-4-e4b-it",
    defaultObjectLayer: 8,
  },
];

export interface Params {
  retention: number;
  maxNewTokens: number;
  lambdaSeed: number;
  lambdaPick: number;
  beta: number;
  prompt: string;
}

export const DEFAULT_PARAMS: Params = {
  retention: 0.223,
  maxNewTokens: 128,
  lambdaSeed: 0.1,
  lambdaPick: 0.5,
  beta: 0.1,
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
  /** HiPrune++ text-guided picks. */
  prompt_tokens?: number[];
  beta?: number;
  lambda_seed?: number;
  lambda_pick?: number;
  mean_attention?: Record<string, Record<string, number | null>>;
  similarity?: Record<string, number>;
  text_similarity_summary?: Record<string, number | null>;
  /** Per-token arrays (index = soft-token index) for hover tooltips. */
  scores?: {
    object_layer?: number[];
    deep_layer?: number[];
    similarity?: number[];
    text_similarity?: number[];
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
