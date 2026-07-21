import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Footer, Nav } from "./Chrome";
import ControlPanel from "./ControlPanel";
import { OverlayCanvas, OverlayLegend } from "./ImageCompare";
import { getGpuStatus, infer, startGpu, stopGpu } from "./lib/api";
import type {
  GpuStatus,
  InferResult,
  MethodKey,
  ModelKey,
  Params,
} from "./lib/types";
import { DEFAULT_PARAMS, MODELS } from "./lib/types";

function DropZone({
  onImage,
  hasImage,
}: {
  onImage: (dataUrl: string) => void;
  hasImage: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const readFile = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = () => onImage(String(reader.result));
      reader.readAsDataURL(file);
    },
    [onImage]
  );

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file && file.type.startsWith("image/")) readFile(file);
      }}
      className={
        "flex flex-col items-center justify-center gap-2 border border-dashed cursor-pointer transition-colors " +
        (dragging ? "border-accent bg-orange-50" : "border-stone-300 bg-white")
      }
      style={{
        borderRadius: "var(--r-1)",
        minHeight: hasImage ? 64 : 320,
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) readFile(file);
          e.target.value = "";
        }}
      />
      <span className="text-sm text-fg-muted">
        {hasImage
          ? "Drop a new image or click to replace"
          : "Drag and drop an image here, or click to upload"}
      </span>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="demo-label text-fg-muted">{label}</span>
      <span className="font-mono text-sm text-fg">{value}</span>
    </div>
  );
}

/** Model answers arrive as markdown (Gemma especially); render it
 * instead of showing literal asterisks. react-markdown builds real DOM
 * nodes, so untrusted model output is safe without sanitization. */
function AnswerText({ text }: { text: string }) {
  return (
    <div className="answer-md text-sm text-fg">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}

function Results({
  imageUrl,
  model,
  result,
}: {
  imageUrl: string;
  model: ModelKey;
  result: InferResult;
}) {
  const md = result.metadata;
  const kept = md ? md.num_tokens - md.pruned.length : null;
  // Actual pruned share from metadata (the kept count is rounded, so
  // this can differ slightly from the requested retention slider).
  const prunedPct = md ? (100 * md.pruned.length) / md.num_tokens : null;
  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <figure className="flex flex-col gap-2">
          <figcaption className="demo-kicker text-fg-muted">Original</figcaption>
          <img
            src={imageUrl}
            alt="original upload"
            className="w-full h-auto border border-border"
            style={{ borderRadius: "var(--r-1)" }}
          />
        </figure>
        <figure className="flex flex-col gap-2">
          <figcaption className="demo-kicker text-fg-muted">
            {md?.method === "hydart"
              ? "HyDART"
              : md?.method === "hiprune_pp"
                ? "HiPrune++"
                : md?.method === "dart"
                  ? "DART"
                  : "HiPrune"}
            {md &&
              ` — ${md.pruned.length}/${md.num_tokens} pruned, grid ${md.grid[0]}x${md.grid[1]}`}
          </figcaption>
          {md ? (
            <>
              <OverlayCanvas imageUrl={imageUrl} metadata={md} model={model} />
              <OverlayLegend metadata={md} />
            </>
          ) : (
            <p className="text-sm text-fg-muted">
              No pruning metadata returned (retention 1.0 keeps everything).
            </p>
          )}
        </figure>
      </div>

      <div
        className="grid grid-cols-2 md:grid-cols-4 gap-4 border border-border bg-white p-4"
        style={{ borderRadius: "var(--r-1)" }}
      >
        <Stat
          label="Tokens kept"
          value={
            md && kept !== null
              ? `${kept} / ${md.num_tokens} (${((100 * kept) / md.num_tokens).toFixed(1)}%)`
              : "all"
          }
        />
        <Stat
          label="Prompt tokens"
          value={
            result.usage?.prompt_tokens != null
              ? String(result.usage.prompt_tokens) +
                (result.baseline_prompt_tokens != null
                  ? ` (baseline ${result.baseline_prompt_tokens})`
                  : "")
              : "—"
          }
        />
        <Stat
          label="Completion tokens"
          value={
            result.usage?.completion_tokens != null
              ? String(result.usage.completion_tokens)
              : "—"
          }
        />
        <Stat
          label="TTFT (server)"
          value={
            result.ttft_ms != null
              ? `${result.ttft_ms.toFixed(0)} ms` +
                (result.baseline_ttft_ms != null
                  ? ` (baseline ${result.baseline_ttft_ms.toFixed(0)} ms)`
                  : "")
              : result.elapsed_s != null
                ? `${result.elapsed_s.toFixed(2)}s total`
                : "—"
          }
        />
      </div>

      <div className="flex flex-col gap-4">
        {result.baseline_answer != null && (
          <div
            className="border border-border bg-white p-4 flex flex-col gap-2"
            style={{ borderRadius: "var(--r-1)" }}
          >
            <span className="demo-kicker text-fg-muted">
              Baseline answer
              {" (unpruned"}
              {result.baseline_ttft_ms != null &&
                ` · TTFT ${result.baseline_ttft_ms.toFixed(0)} ms`}
              {")"}
            </span>
            <AnswerText text={result.baseline_answer} />
          </div>
        )}
        <div
          className="border border-border bg-white p-4 flex flex-col gap-2"
          style={{ borderRadius: "var(--r-1)" }}
        >
          <span className="demo-kicker text-fg-muted">
            {result.baseline_answer != null ? "Pruned answer" : "Answer"}
            {(prunedPct != null || result.ttft_ms != null) && (
              <>
                {" ("}
                {prunedPct != null && `${prunedPct.toFixed(1)}% pruned`}
                {prunedPct != null && result.ttft_ms != null && " · "}
                {result.ttft_ms != null && `TTFT ${result.ttft_ms.toFixed(0)} ms`}
                {")"}
              </>
            )}
          </span>
          <AnswerText text={result.answer} />
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [model, setModel] = useState<ModelKey>("qwen2_5_vl");
  const [method, setMethod] = useState<MethodKey>("hiprune");
  const [params, setParams] = useState<Params>(DEFAULT_PARAMS);
  const [withBaseline, setWithBaseline] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [result, setResult] = useState<InferResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gpu, setGpu] = useState<GpuStatus>({
    phase: "unknown",
    model: null,
    method: null,
  });

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const s = await getGpuStatus();
        if (alive) setGpu(s);
      } catch {
        if (alive) setGpu({ phase: "unknown", model: null, method: null });
      }
    };
    poll();
    const t = setInterval(poll, 3000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  const handleStartStop = async () => {
    setError(null);
    try {
      if (gpu.phase === "ready" || gpu.phase === "starting") {
        setGpu({ ...gpu, phase: "stopping" });
        setGpu(await stopGpu());
      } else {
        setGpu({ phase: "starting", model, method });
        setGpu(
          await startGpu(
            model,
            method,
            params.lambdaSeed,
            params.lambdaPick,
            params.beta,
            params.pivotImage,
            params.pivotText
          )
        );
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleRun = async () => {
    if (!imageUrl) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await infer(imageUrl, model, method, params, withBaseline));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const serverMatches =
    gpu.phase === "ready" && gpu.model === model && gpu.method === method;

  return (
    <div className="min-h-screen flex flex-col">
      <Nav />
      <main className="flex-1 flex flex-col" style={{ paddingTop: 60 }}>
        <div className="container-1312 w-full py-8">
          <div className="flex flex-col xl:flex-row gap-8 items-start">
            <section className="flex-1 w-full flex flex-col gap-5 min-w-0">
              <h1 className="h-display" style={{ fontSize: 28 }}>
                Visual token pruning, side by side
              </h1>
              <DropZone onImage={(u) => setImageUrl(u)} hasImage={imageUrl != null} />
              {imageUrl && !result && (
                <img
                  src={imageUrl}
                  alt="uploaded"
                  className="w-full max-w-xl h-auto border border-border"
                  style={{ borderRadius: "var(--r-1)" }}
                />
              )}
              {error && (
                <p
                  className="text-sm border border-red-200 bg-red-50 text-red-700 p-3"
                  style={{ borderRadius: "var(--r-1)" }}
                >
                  {error}
                </p>
              )}
              {imageUrl && result && (
                <Results
                  imageUrl={imageUrl}
                  model={(MODELS.find((m) => m.key === model) ?? MODELS[0]).key}
                  result={result}
                />
              )}
            </section>

            <div className="w-full xl:w-[380px] shrink-0 xl:sticky xl:top-[84px]">
              <ControlPanel
                model={model}
                method={method}
                params={params}
                gpu={gpu}
                busy={busy}
                canRun={imageUrl != null && serverMatches}
                withBaseline={withBaseline}
                onModel={setModel}
                onMethod={setMethod}
                onParams={setParams}
                onStartStop={handleStartStop}
                onRun={handleRun}
                onWithBaseline={setWithBaseline}
              />
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
