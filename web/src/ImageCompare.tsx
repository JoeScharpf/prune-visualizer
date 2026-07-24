import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { ModelKey, PruningMetadata } from "./lib/types";

const ANCHOR = "rgb(255,60,60)";
const BUFFER = "rgb(255,170,40)";
const REGISTER = "rgb(50,255,80)";
const DIVERSE = "rgb(70,140,255)";
const PROMPT = "rgb(205,90,255)";
const PIVOT = "rgb(255,60,60)";
const UNIFORM = "rgb(45,212,191)";
const PRUNED_FILL = "rgba(0,0,0,0.82)";

/** Map normalized score in [0,1] to a translucent heat color (cool→hot). */
function heatFill(t: number): string {
  const x = Math.min(1, Math.max(0, t));
  // Dark blue → cyan → yellow → red
  let r: number;
  let g: number;
  let b: number;
  if (x < 0.33) {
    const u = x / 0.33;
    r = 30;
    g = Math.round(40 + 140 * u);
    b = Math.round(120 + 80 * u);
  } else if (x < 0.66) {
    const u = (x - 0.33) / 0.33;
    r = Math.round(30 + 225 * u);
    g = Math.round(180 + 40 * u);
    b = Math.round(200 - 180 * u);
  } else {
    const u = (x - 0.66) / 0.34;
    r = 255;
    g = Math.round(220 - 180 * u);
    b = Math.round(20 * (1 - u));
  }
  const a = 0.15 + 0.7 * x;
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
}

function normalizeScores(scores: number[]): number[] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of scores) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo;
  if (!(span > 0) || !Number.isFinite(span)) {
    return scores.map(() => 0);
  }
  return scores.map((v) => (v - lo) / span);
}

interface Category {
  name: string;
  indices: number[];
  color: string;
  /** Ordered categories report a rank in the tooltip (top-k / pick order). */
  ranked: boolean;
}

/** All kept-token categories in draw order, per method. */
function keptCategories(md: PruningMetadata): Category[] {
  if (md.method === "nprune" || md.method === "checkered" || md.uniform) {
    // A single unranked category: the deterministic patterns (lattice,
    // checkerboard) have no scores or order.
    return [
      {
        name: "uniform",
        indices: md.uniform ?? [],
        color: UNIFORM,
        ranked: false,
      },
    ];
  }
  if (md.method === "dart" || md.pivots) {
    return [
      { name: "pivots", indices: md.pivots ?? [], color: PIVOT, ranked: true },
      { name: "diverse", indices: md.diverse ?? [], color: DIVERSE, ranked: true },
    ];
  }
  // AnchorPrune: `expansion` is its distinguishing field (plain HiPrune
  // also reports `anchors`).
  if (md.method === "anchorprune" || md.expansion) {
    return [
      {
        name: "anchors",
        indices: md.anchors ?? [],
        color: ANCHOR,
        ranked: true,
      },
      {
        name: "expansion",
        indices: md.expansion ?? [],
        color: DIVERSE,
        ranked: true,
      },
    ];
  }
  const cats: Category[] = [
    { name: "anchors", indices: md.anchors ?? [], color: ANCHOR, ranked: true },
    { name: "buffers", indices: md.buffers ?? [], color: BUFFER, ranked: false },
  ];
  if (md.method === "hydart" || md.diverse) {
    cats.push({
      name: "diverse",
      indices: md.diverse ?? [],
      color: DIVERSE,
      ranked: true,
    });
    return cats;
  }
  cats.push({
    name: "registers",
    indices: md.registers ?? [],
    color: REGISTER,
    ranked: true,
  });
  if (md.method === "hiprune_pp" || md.prompt_tokens) {
    cats.push({
      name: "prompt",
      indices: md.prompt_tokens ?? [],
      color: PROMPT,
      ranked: true,
    });
  }
  return cats;
}

/** Singular tooltip label for a legend/category name. */
function singular(name: string): string {
  return name === "pivots"
    ? "pivot"
    : name === "anchors"
      ? "anchor"
      : name === "buffers"
        ? "buffer"
        : name === "registers"
          ? "register"
          : name;
}

/** Draw the uploaded image the way the model's preprocessor sees it:
 * LLaVA (llava-hf) resizes the shortest edge to 336 then center-crops a
 * square; Qwen resizes to the grid's aspect (its 28px-multiple resize is
 * close enough to a plain stretch for visualization). */
function drawBase(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  w: number,
  h: number,
  model: ModelKey
) {
  if (model === "llava_1_5") {
    const side = Math.min(img.width, img.height);
    const sx = (img.width - side) / 2;
    const sy = (img.height - side) / 2;
    ctx.drawImage(img, sx, sy, side, side, 0, 0, w, h);
  } else {
    ctx.drawImage(img, 0, 0, w, h);
  }
}

interface CellInfo {
  category: string;
  color: string;
  /** 1-based rank within the category, for ordered categories
   * (anchors/registers: attention top-k order; diverse: pick order). */
  rank: number | null;
  rankOf: number | null;
}

/** index -> category/rank lookup for the tooltip. */
function buildCellIndex(md: PruningMetadata): Map<number, CellInfo> {
  const map = new Map<number, CellInfo>();
  const ordered: Array<[string, number[], string, boolean]> = [
    ...keptCategories(md).map(
      (c): [string, number[], string, boolean] => [
        singular(c.name),
        c.indices,
        c.color,
        c.ranked,
      ]
    ),
    ["pruned", md.pruned, "rgb(120,113,108)", false],
  ];
  for (const [category, indices, color, ranked] of ordered) {
    indices.forEach((idx, i) => {
      map.set(idx, {
        category,
        color,
        rank: ranked ? i + 1 : null,
        rankOf: ranked ? indices.length : null,
      });
    });
  }
  return map;
}

function fmtScore(v: number | undefined, uniform: number): string {
  if (v == null) return "—";
  return `${v.toExponential(2)} (${(v / uniform).toFixed(2)}x uniform)`;
}

export function OverlayCanvas({
  imageUrl,
  metadata,
  model,
  showHeatmap = false,
  heatLayerIdx = 0,
}: {
  imageUrl: string;
  metadata: PruningMetadata;
  model: ModelKey;
  showHeatmap?: boolean;
  /** 0-based vision-encoder layer when scrubbing vision_layers. */
  heatLayerIdx?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{
    idx: number;
    xPct: number;
    yPct: number;
  } | null>(null);
  const cellIndex = useMemo(() => buildCellIndex(metadata), [metadata]);
  const visionLayers = metadata.scores?.vision_layers;
  const layerScores =
    visionLayers &&
    visionLayers.length > 0 &&
    heatLayerIdx >= 0 &&
    heatLayerIdx < visionLayers.length
      ? visionLayers[heatLayerIdx]
      : undefined;
  const heatScores = layerScores ?? metadata.scores?.object_layer;
  const canHeatmap = Boolean(
    heatScores && heatScores.length === metadata.num_tokens
  );
  const heatmapOn = showHeatmap && canHeatmap;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const [gridW, gridH] = metadata.grid;
    // Render at a cell size that lands near 700px on the long edge.
    const cell = Math.max(6, Math.round(700 / Math.max(gridW, gridH)));
    canvas.width = gridW * cell;
    canvas.height = gridH * cell;

    const img = new Image();
    img.onload = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      drawBase(ctx, img, canvas.width, canvas.height, model);

      const box = (idx: number): [number, number] => [
        (idx % gridW) * cell,
        Math.floor(idx / gridW) * cell,
      ];

      if (heatmapOn && heatScores) {
        const norm = normalizeScores(heatScores);
        for (let idx = 0; idx < norm.length; idx++) {
          const [x, y] = box(idx);
          ctx.fillStyle = heatFill(norm[idx]);
          ctx.fillRect(x, y, cell, cell);
        }
      } else {
        ctx.fillStyle = PRUNED_FILL;
        for (const idx of metadata.pruned) {
          const [x, y] = box(idx);
          ctx.fillRect(x, y, cell, cell);
        }

        const cats: Array<[number[], string]> = keptCategories(metadata).map(
          (c): [number[], string] => [c.indices, c.color]
        );
        ctx.lineWidth = 2;
        for (const [indices, color] of cats) {
          ctx.strokeStyle = color;
          for (const idx of indices) {
            const [x, y] = box(idx);
            ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
          }
        }
      }
    };
    img.src = imageUrl;
  }, [imageUrl, metadata, model, heatmapOn, heatScores]);

  const [gridW, gridH] = metadata.grid;
  const uniform = 1 / metadata.num_tokens;

  const onMove = (e: ReactMouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    const col = Math.min(gridW - 1, Math.max(0, Math.floor(x * gridW)));
    const row = Math.min(gridH - 1, Math.max(0, Math.floor(y * gridH)));
    setHover({ idx: row * gridW + col, xPct: x * 100, yPct: y * 100 });
  };

  const info = hover ? cellIndex.get(hover.idx) : undefined;
  const layerAttn = hover && heatScores ? heatScores[hover.idx] : undefined;
  const objScore =
    hover && !layerScores
      ? metadata.scores?.object_layer?.[hover.idx]
      : undefined;
  const deepScore = hover ? metadata.scores?.deep_layer?.[hover.idx] : undefined;
  const simScore = hover ? metadata.scores?.similarity?.[hover.idx] : undefined;
  const textSimScore = hover
    ? metadata.scores?.text_similarity?.[hover.idx]
    : undefined;
  const keyNorm = hover ? metadata.scores?.key_norm?.[hover.idx] : undefined;
  const pivotSim = hover
    ? metadata.scores?.pivot_similarity?.[hover.idx]
    : undefined;
  const hoverRow = hover ? Math.floor(hover.idx / gridW) : 0;
  const hoverCol = hover ? hover.idx % gridW : 0;

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        className="w-full h-auto block border border-border"
        style={{ borderRadius: "var(--r-1)", cursor: "crosshair" }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      />
      {hover && (
        <div
          aria-hidden
          className="absolute pointer-events-none border-2 border-white mix-blend-difference"
          style={{
            left: `${(hoverCol / gridW) * 100}%`,
            top: `${(hoverRow / gridH) * 100}%`,
            width: `${100 / gridW}%`,
            height: `${100 / gridH}%`,
          }}
        />
      )}
      {hover && (info || heatmapOn) && (
        <div
          className="absolute z-10 pointer-events-none bg-stone-950 text-white p-2.5 flex flex-col gap-1"
          style={{
            borderRadius: "var(--r-1)",
            left: hover.xPct < 55 ? `calc(${hover.xPct}% + 14px)` : undefined,
            right: hover.xPct >= 55 ? `calc(${100 - hover.xPct}% + 14px)` : undefined,
            top: `min(${hover.yPct}%, calc(100% - 96px))`,
            maxWidth: 260,
          }}
        >
          <span className="demo-label" style={{ color: "#a8a29e" }}>
            token {hover.idx} — row {hoverRow}, col {hoverCol}
          </span>
          {!heatmapOn && info && (
            <span className="text-xs font-mono flex items-center gap-1.5">
              <span
                aria-hidden
                className="inline-block shrink-0"
                style={{
                  width: 8,
                  height: 8,
                  background: info.color,
                  borderRadius: 1,
                }}
              />
              {info.category}
              {info.rank != null && ` — rank ${info.rank}/${info.rankOf}`}
            </span>
          )}
          {layerAttn != null && layerScores && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              layer {heatLayerIdx + 1} attn: {fmtScore(layerAttn, uniform)}
            </span>
          )}
          {objScore != null && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              obj attn: {fmtScore(objScore, uniform)}
            </span>
          )}
          {keyNorm != null && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              key L1 norm: {keyNorm.toFixed(1)}
            </span>
          )}
          {pivotSim != null && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              pivot cos sim: {pivotSim.toFixed(3)}
            </span>
          )}
          {deepScore != null && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              deep attn: {fmtScore(deepScore, uniform)}
            </span>
          )}
          {simScore != null && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              max cos sim: {simScore.toFixed(3)}
            </span>
          )}
          {textSimScore != null && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              text cos sim: {textSimScore.toFixed(3)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export function OverlayLegend({
  metadata,
  showHeatmap,
  onToggleHeatmap,
  heatLayerIdx = 0,
  onHeatLayer,
}: {
  metadata: PruningMetadata;
  showHeatmap: boolean;
  onToggleHeatmap: (next: boolean) => void;
  heatLayerIdx?: number;
  onHeatLayer?: (idx: number) => void;
}) {
  const canHeatmap = Boolean(
    metadata.scores?.object_layer &&
      metadata.scores.object_layer.length === metadata.num_tokens
  );
  const visionLayers = metadata.scores?.vision_layers;
  const layerCount = visionLayers?.length ?? 0;
  const canScrub = Boolean(
    layerCount > 1 &&
      visionLayers?.every((row) => row.length === metadata.num_tokens)
  );

  if (showHeatmap && canHeatmap) {
    return (
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <label className="inline-flex items-center gap-2 demo-label text-fg cursor-pointer select-none">
            <input
              type="checkbox"
              className="accent-fg"
              checked={showHeatmap}
              onChange={(e) => onToggleHeatmap(e.target.checked)}
            />
            Heatmap
          </label>
          <span className="inline-flex items-center gap-1.5 demo-label text-fg-muted">
            <span
              aria-hidden
              className="inline-block"
              style={{
                width: 48,
                height: 8,
                borderRadius: 1,
                background:
                  "linear-gradient(90deg, rgba(30,40,120,0.3), rgba(255,180,40,0.75), rgba(255,40,20,0.85))",
              }}
            />
            low → high attention
          </span>
        </div>
        {canScrub && onHeatLayer && (
          <label className="flex flex-col gap-1.5 max-w-md">
            <span className="demo-label text-fg-muted">
              Early — layer {heatLayerIdx + 1}/{layerCount} — Late
            </span>
            <input
              type="range"
              className="hp-range"
              min={0}
              max={layerCount - 1}
              step={1}
              value={Math.min(heatLayerIdx, layerCount - 1)}
              onChange={(e) => onHeatLayer(Number(e.target.value))}
            />
          </label>
        )}
      </div>
    );
  }

  const items: Array<[string, string, number]> = [
    ...keptCategories(metadata).map(
      (c): [string, string, number] => [c.name, c.color, c.indices.length]
    ),
    ["pruned", "rgb(12,10,9)", metadata.pruned.length],
  ];
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {canHeatmap && (
        <label className="inline-flex items-center gap-2 demo-label text-fg cursor-pointer select-none">
          <input
            type="checkbox"
            className="accent-fg"
            checked={false}
            onChange={(e) => onToggleHeatmap(e.target.checked)}
          />
          Heatmap
        </label>
      )}
      {items.map(([name, color, count]) => (
        <span
          key={name}
          className="inline-flex items-center gap-1.5 demo-label text-fg-muted"
        >
          <span
            aria-hidden
            className="inline-block"
            style={{ width: 8, height: 8, background: color, borderRadius: 1 }}
          />
          {name} {count}
        </span>
      ))}
    </div>
  );
}
