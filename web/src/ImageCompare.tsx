import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { ModelKey, PruningMetadata } from "./lib/types";

const ANCHOR = "rgb(255,60,60)";
const BUFFER = "rgb(255,170,40)";
const REGISTER = "rgb(50,255,80)";
const DIVERSE = "rgb(70,140,255)";
const PRUNED_FILL = "rgba(0,0,0,0.82)";

/** CLIP image mean, the pad color LLaVA's square-pad preprocessing uses. */
const CLIP_PAD = "rgb(122,116,104)";

function thirdCategory(md: PruningMetadata): {
  name: string;
  indices: number[];
  color: string;
} {
  if (md.method === "hydart" || md.diverse) {
    return { name: "diverse", indices: md.diverse ?? [], color: DIVERSE };
  }
  return { name: "registers", indices: md.registers ?? [], color: REGISTER };
}

/** Draw the uploaded image the way the model's preprocessor sees it:
 * LLaVA square-pads with the CLIP mean color; Qwen resizes to the grid's
 * aspect (its 28px-multiple resize is close enough to a plain stretch for
 * visualization). */
function drawBase(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  w: number,
  h: number,
  model: ModelKey
) {
  if (model === "llava_1_5") {
    ctx.fillStyle = CLIP_PAD;
    ctx.fillRect(0, 0, w, h);
    const side = Math.max(img.width, img.height);
    const dw = (img.width / side) * w;
    const dh = (img.height / side) * h;
    ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
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
  const third = thirdCategory(md);
  const ordered: Array<[string, number[], string, boolean]> = [
    ["anchor", md.anchors, ANCHOR, true],
    ["buffer", md.buffers, BUFFER, false],
    [
      third.name === "registers" ? "register" : third.name,
      third.indices,
      third.color,
      true,
    ],
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
}: {
  imageUrl: string;
  metadata: PruningMetadata;
  model: ModelKey;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{
    idx: number;
    xPct: number;
    yPct: number;
  } | null>(null);
  const cellIndex = useMemo(() => buildCellIndex(metadata), [metadata]);

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

      ctx.fillStyle = PRUNED_FILL;
      for (const idx of metadata.pruned) {
        const [x, y] = box(idx);
        ctx.fillRect(x, y, cell, cell);
      }

      const third = thirdCategory(metadata);
      const cats: Array<[number[], string]> = [
        [metadata.anchors, ANCHOR],
        [metadata.buffers, BUFFER],
        [third.indices, third.color],
      ];
      ctx.lineWidth = 2;
      for (const [indices, color] of cats) {
        ctx.strokeStyle = color;
        for (const idx of indices) {
          const [x, y] = box(idx);
          ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
        }
      }
    };
    img.src = imageUrl;
  }, [imageUrl, metadata, model]);

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
  const objScore = hover ? metadata.scores?.object_layer?.[hover.idx] : undefined;
  const deepScore = hover ? metadata.scores?.deep_layer?.[hover.idx] : undefined;
  const simScore = hover ? metadata.scores?.similarity?.[hover.idx] : undefined;
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
      {hover && info && (
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
          <span className="text-xs font-mono flex items-center gap-1.5">
            <span
              aria-hidden
              className="inline-block shrink-0"
              style={{ width: 8, height: 8, background: info.color, borderRadius: 1 }}
            />
            {info.category}
            {info.rank != null && ` — rank ${info.rank}/${info.rankOf}`}
          </span>
          <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
            obj attn: {fmtScore(objScore, uniform)}
          </span>
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
        </div>
      )}
    </div>
  );
}

export function OverlayLegend({ metadata }: { metadata: PruningMetadata }) {
  const third = thirdCategory(metadata);
  const items: Array<[string, string, number]> = [
    ["anchors", ANCHOR, metadata.anchors.length],
    ["buffers", BUFFER, metadata.buffers.length],
    [third.name, third.color, third.indices.length],
    ["pruned", "rgb(12,10,9)", metadata.pruned.length],
  ];
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {items.map(([name, color, count]) => (
        <span key={name} className="inline-flex items-center gap-1.5 demo-label text-fg-muted">
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
