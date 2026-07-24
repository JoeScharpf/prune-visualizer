import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
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
import { DEFAULT_PARAMS } from "./lib/types";

function useImageFileInput(onImage: (dataUrl: string) => void) {
  const inputRef = useRef<HTMLInputElement>(null);
  const readFile = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = () => onImage(String(reader.result));
      reader.readAsDataURL(file);
    },
    [onImage]
  );
  const openPicker = () => inputRef.current?.click();
  const onInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) readFile(file);
    e.target.value = "";
  };
  const takeDrop = (e: DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) readFile(file);
  };
  return { inputRef, openPicker, onInputChange, takeDrop };
}

/** Large dashed upload surface — empty state only. */
function EmptyDropZone({ onImage }: { onImage: (dataUrl: string) => void }) {
  const [dragging, setDragging] = useState(false);
  const { inputRef, openPicker, onInputChange, takeDrop } =
    useImageFileInput(onImage);

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={openPicker}
      onKeyDown={(e) => e.key === "Enter" && openPicker()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        setDragging(false);
        takeDrop(e);
      }}
      className={
        "flex h-full min-h-[280px] w-full flex-col items-center justify-center gap-2 border border-dashed cursor-pointer transition-colors " +
        (dragging ? "border-accent bg-orange-50" : "border-stone-300 bg-white")
      }
      style={{ borderRadius: "var(--r-1)" }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={onInputChange}
      />
      <span className="text-sm text-fg-muted">
        Drag and drop an image here, or click to upload
      </span>
    </div>
  );
}

/** Click / drop on an existing image to replace it. */
function ImageReplaceTarget({
  src,
  alt,
  onImage,
  className = "max-w-full max-h-[420px] w-auto h-auto",
}: {
  src: string;
  alt: string;
  onImage: (dataUrl: string) => void;
  className?: string;
}) {
  const [dragging, setDragging] = useState(false);
  const { inputRef, openPicker, onInputChange, takeDrop } =
    useImageFileInput(onImage);

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Replace image — click or drop a new file"
      onClick={openPicker}
      onKeyDown={(e) => e.key === "Enter" && openPicker()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        setDragging(false);
        takeDrop(e);
      }}
      className={
        "group relative inline-block max-w-full cursor-pointer overflow-hidden transition-shadow " +
        (dragging ? "ring-2 ring-orange-500 ring-offset-2" : "")
      }
      style={{ borderRadius: "var(--r-1)" }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={onInputChange}
      />
      <img
        src={src}
        alt={alt}
        className={className}
        style={{ borderRadius: "var(--r-1)", display: "block" }}
        draggable={false}
      />
      <div
        className={
          "pointer-events-none absolute inset-0 flex items-center justify-center p-3 transition-opacity " +
          (dragging
            ? "opacity-100 bg-black/35"
            : "opacity-0 group-hover:opacity-100 bg-black/25")
        }
        style={{ borderRadius: "var(--r-1)" }}
      >
        <span
          className="demo-label text-white bg-black/55 px-2 py-1"
          style={{ borderRadius: 2 }}
        >
          {dragging ? "Drop to replace" : "Click or drop to replace"}
        </span>
      </div>
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

function methodLabel(method: string | undefined): string {
  if (method === "hydart") return "HyDART";
  if (method === "hiprune_pp") return "HiPrune++";
  if (method === "dart") return "DART";
  if (method === "nprune") return "Lattice";
  if (method === "checkered") return "Checkered";
  if (method === "anchorprune") return "AnchorPrune";
  return "HiPrune";
}

/** Preview → Original | pruned with slide + fade when a result arrives. */
function CompareStage({
  imageUrl,
  model,
  result,
  onReplaceImage,
}: {
  imageUrl: string;
  model: ModelKey;
  result: InferResult | null;
  onReplaceImage: (dataUrl: string) => void;
}) {
  const md = result?.metadata ?? null;
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatLayerIdx, setHeatLayerIdx] = useState(0);
  // Layout (2-col) follows `result`; motion enter is a separate flip.
  const [entered, setEntered] = useState(false);

  const defaultHeatLayer = useCallback((metadata: InferResult["metadata"]) => {
    if (!metadata?.scores) return 0;
    const idx = metadata.scores.vision_layer_object_idx;
    const layers = metadata.scores.vision_layers;
    if (
      typeof idx === "number" &&
      layers &&
      idx >= 0 &&
      idx < layers.length
    ) {
      return idx;
    }
    return 0;
  }, []);

  useEffect(() => {
    if (!result) {
      setEntered(false);
      setShowHeatmap(false);
      setHeatLayerIdx(0);
      return;
    }
    setHeatLayerIdx(defaultHeatLayer(result.metadata));
    setEntered(false);
    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced) {
      setEntered(true);
      return;
    }
    const id = requestAnimationFrame(() => {
      requestAnimationFrame(() => setEntered(true));
    });
    return () => cancelAnimationFrame(id);
  }, [result, defaultHeatLayer]);

  const layerCount = md?.scores?.vision_layers?.length ?? 0;
  const heatmapCaption =
    showHeatmap && md?.scores?.object_layer
      ? layerCount > 0
        ? ` — layer ${heatLayerIdx + 1}/${layerCount}, grid ${md.grid[0]}x${md.grid[1]}`
        : ` — object-layer attention, grid ${md.grid[0]}x${md.grid[1]}`
      : null;

  return (
    <div
      className={
        "compare-stage px-2 " +
        (result ? "compare-stage--has-result " : "") +
        (entered && result ? "compare-stage--entered" : "")
      }
    >
      <figure className="compare-stage__original flex flex-col items-center gap-2 w-fit max-w-full min-w-0 mx-auto">
        <figcaption className="demo-kicker text-fg-muted text-center w-full">
          {result ? "Original" : "\u00a0"}
        </figcaption>
        <ImageReplaceTarget
          src={imageUrl}
          alt={result ? "original upload" : "uploaded"}
          onImage={onReplaceImage}
        />
      </figure>

      {result && (
        <figure
          className={
            "compare-stage__pruned flex flex-col items-center gap-2 min-w-0 " +
            (entered ? "compare-stage__pruned--in" : "")
          }
        >
          <figcaption className="demo-kicker text-fg-muted text-center">
            {showHeatmap && md?.scores?.object_layer
              ? "Attention heatmap"
              : methodLabel(md?.method)}
            {md &&
              !showHeatmap &&
              ` — ${md.pruned.length}/${md.num_tokens} pruned, grid ${md.grid[0]}x${md.grid[1]}`}
            {heatmapCaption}
          </figcaption>
          {md ? (
            <>
              <div className="w-full max-h-[420px] flex justify-center overflow-hidden [&_canvas]:max-h-[420px] [&_canvas]:max-w-full [&_canvas]:h-auto [&_canvas]:w-auto">
                <OverlayCanvas
                  imageUrl={imageUrl}
                  metadata={md}
                  model={model}
                  showHeatmap={showHeatmap}
                  heatLayerIdx={heatLayerIdx}
                />
              </div>
              <OverlayLegend
                metadata={md}
                showHeatmap={showHeatmap}
                onToggleHeatmap={setShowHeatmap}
                heatLayerIdx={heatLayerIdx}
                onHeatLayer={setHeatLayerIdx}
              />
            </>
          ) : (
            <p className="text-sm text-fg-muted">
              No pruning metadata returned (retention 1.0 keeps everything).
            </p>
          )}
        </figure>
      )}
    </div>
  );
}

function ResultsDetails({ result }: { result: InferResult }) {
  const md = result.metadata;
  const kept = md ? md.num_tokens - md.pruned.length : null;
  const prunedPct = md ? (100 * md.pruned.length) / md.num_tokens : null;

  return (
    <div className="flex flex-col gap-5">
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
  const [model] = useState<ModelKey>("gemma4");
  const [method, setMethod] = useState<MethodKey>("hiprune");
  const [params, setParams] = useState<Params>(DEFAULT_PARAMS);
  const [withBaseline, setWithBaseline] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [result, setResult] = useState<InferResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gpu, setGpu] = useState<GpuStatus>({ phase: "unknown" });

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const s = await getGpuStatus();
        if (alive) setGpu(s);
      } catch {
        if (alive) setGpu({ phase: "unknown" });
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
        setGpu({ phase: "starting" });
        setGpu(await startGpu());
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

  // All model servers run concurrently and the method (and its knobs)
  // applies per request, so a model is usable the moment its own server
  // is healthy — even while others are still starting.
  const serverMatches = gpu.models?.[model] === "ready";

  const handleImage = useCallback((dataUrl: string) => {
    setImageUrl(dataUrl);
    setResult(null);
    setError(null);
  }, []);

  // Keep the compare view frozen until the next Run — switching method
  // would otherwise redraw the old overlay with a mismatched category
  // set and look like an abrupt change.
  const handleMethod = useCallback((m: MethodKey) => {
    setMethod(m);
    setResult(null);
    setError(null);
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <Nav />
      <main className="flex-1 flex flex-col" style={{ paddingTop: 60 }}>
        <div className="container-1312 w-full py-8">
          <h1 className="h-display text-center mb-8" style={{ fontSize: 28 }}>
            Visual token pruning, side by side
          </h1>
          <div className="flex flex-col xl:flex-row gap-8 items-stretch">
            <section
              className={
                "flex-1 w-full flex flex-col gap-5 min-w-0 " +
                (!imageUrl ? "xl:min-h-0" : "")
              }
            >
              {!imageUrl && <EmptyDropZone onImage={handleImage} />}
              {imageUrl && (
                <CompareStage
                  imageUrl={imageUrl}
                  model={model}
                  result={result}
                  onReplaceImage={handleImage}
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
              {result && <ResultsDetails result={result} />}
            </section>

            <div className="w-full xl:w-[380px] shrink-0 xl:sticky xl:top-[84px] xl:self-start">
              <ControlPanel
                model={model}
                method={method}
                params={params}
                gpu={gpu}
                busy={busy}
                canRun={imageUrl != null && serverMatches}
                withBaseline={withBaseline}
                onMethod={handleMethod}
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
