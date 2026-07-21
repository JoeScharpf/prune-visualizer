import { useState } from "react";
import type {
  GpuStatus,
  MethodKey,
  ModelKey,
  Params,
} from "./lib/types";
import { MODELS } from "./lib/types";

function Chip({
  active,
  disabled,
  onClick,
  children,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={
        "btn-slide h-9 inline-flex items-center px-3.5 text-sm font-medium border " +
        (active
          ? "border-fg text-fg-invert"
          : "border-border text-fg disabled:opacity-40 disabled:cursor-not-allowed")
      }
      style={{ background: active ? "#0C0A09" : "#FFFFFF" }}
    >
      {!active && (
        <span aria-hidden className="btn-slide-fill" style={{ background: "#F3EFED" }} />
      )}
      <span className="btn-slide-content">{children}</span>
    </button>
  );
}

function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: React.ReactNode;
  hint?: string;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between">
        <span className="demo-kicker text-fg-muted">{label}</span>
        {hint && <span className="demo-label text-fg-muted">{hint}</span>}
      </div>
      {children}
    </div>
  );
}

const numInput =
  "h-9 w-full border border-border bg-white px-2.5 text-sm font-mono text-fg " +
  "focus:border-stone-400 focus:outline-none";

function Disclosure({
  label,
  open,
  onToggle,
  children,
}: {
  label: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-4 border-t border-border pt-4">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex items-center justify-between gap-2 text-left"
      >
        <span className="demo-kicker text-fg-muted">{label}</span>
        <span
          aria-hidden
          className="text-fg-muted transition-transform"
          style={{
            fontSize: 11,
            transform: open ? "rotate(90deg)" : "rotate(0deg)",
          }}
        >
          ▶
        </span>
      </button>
      {open && children}
    </div>
  );
}

function GpuChip({ status }: { status: GpuStatus }) {
  const phase = status.phase;
  const color =
    phase === "ready"
      ? "#16a34a"
      : phase === "starting" || phase === "stopping"
        ? "#ea580c"
        : phase === "error"
          ? "#dc2626"
          : "#a8a29e";
  const label =
    phase === "ready"
      ? `live — ${status.model ?? ""} / ${status.method ?? ""}`
      : phase === "starting"
        ? "starting…"
        : phase === "stopping"
          ? "stopping…"
          : phase === "error"
            ? "error"
            : "gpu stopped";
  return (
    <span
      className="inline-flex items-center gap-2 border border-border bg-white px-3 h-9 min-w-0"
      style={{ borderRadius: "var(--r-1)" }}
    >
      <span
        aria-hidden
        className={
          (phase === "ready" || phase === "starting" ? "live-dot " : "") + "shrink-0"
        }
        style={{ width: 8, height: 8, borderRadius: 999, background: color }}
      />
      <span className="demo-label text-fg truncate">{label}</span>
    </span>
  );
}

export default function ControlPanel({
  model,
  method,
  params,
  gpu,
  busy,
  canRun,
  withBaseline,
  onModel,
  onMethod,
  onParams,
  onStartStop,
  onRun,
  onWithBaseline,
}: {
  model: ModelKey;
  method: MethodKey;
  params: Params;
  gpu: GpuStatus;
  busy: boolean;
  canRun: boolean;
  withBaseline: boolean;
  onModel: (m: ModelKey) => void;
  onMethod: (m: MethodKey) => void;
  onParams: (p: Params) => void;
  onStartStop: () => void;
  onRun: () => void;
  onWithBaseline: (v: boolean) => void;
}) {
  const set = (patch: Partial<Params>) => onParams({ ...params, ...patch });
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const gpuBusy = gpu.phase === "starting" || gpu.phase === "stopping";
  const running = gpu.phase === "ready" || gpu.phase === "starting";
  const restartNeeded =
    gpu.phase === "ready" && (gpu.model !== model || gpu.method !== method);
  const modelInfo = MODELS.find((m) => m.key === model)!;

  return (
    <aside
      className="w-full flex flex-col gap-6 border border-border bg-white p-5"
      style={{ borderRadius: "var(--r-1)" }}
    >
      <div className="flex items-center justify-between gap-3">
        <GpuChip status={gpu} />
        <button
          type="button"
          onClick={onStartStop}
          disabled={gpuBusy}
          className="btn-slide h-9 inline-flex items-center shrink-0 whitespace-nowrap px-4 text-sm font-medium text-fg border border-border disabled:opacity-40"
          style={{ background: "#FFFFFF" }}
        >
          <span aria-hidden className="btn-slide-fill" style={{ background: "#F3EFED" }} />
          <span className="btn-slide-content whitespace-nowrap">
            {running ? "Stop GPU" : "Start GPU"}
          </span>
        </button>
      </div>
      {gpu.detail && gpu.phase !== "ready" && (
        <p className="text-xs font-mono text-fg-muted -mt-4">{gpu.detail}</p>
      )}
      {restartNeeded && (
        <p className="text-xs text-fg-muted -mt-3">
          The server is running {gpu.model} / {gpu.method}. Start GPU again to
          switch to {model} / {method}.
        </p>
      )}

      <Field label="Model">
        <div className="flex flex-wrap gap-2">
          {MODELS.map((m) => (
            <Chip key={m.key} active={model === m.key} onClick={() => onModel(m.key)}>
              {m.label}
            </Chip>
          ))}
        </div>
      </Field>

      <Field label="Method">
        <div className="flex flex-wrap gap-2">
          <Chip active={method === "hiprune"} onClick={() => onMethod("hiprune")}>
            HiPrune
          </Chip>
          <Chip active={method === "hydart"} onClick={() => onMethod("hydart")}>
            HyDART
          </Chip>
          <Chip
            active={method === "hiprune_pp"}
            onClick={() => onMethod("hiprune_pp")}
          >
            HiPrune++
          </Chip>
        </div>
      </Field>

      <Field
        label="Retention ratio"
        hint={`keep ${(params.retention * 100).toFixed(1)}%`}
      >
        <input
          type="range"
          className="hp-range"
          min={0.05}
          max={1}
          step={0.005}
          value={params.retention}
          onChange={(e) => set({ retention: Number(e.target.value) })}
        />
      </Field>

      <Disclosure
        label="Advanced settings"
        open={advancedOpen}
        onToggle={() => setAdvancedOpen((v) => !v)}
      >
        <div className="flex flex-col gap-6">
          {/* Alpha and the object layer are fixed at serve time in the vLLM
              fork, so they are shown but not editable. */}
          <div className="grid grid-cols-2 gap-3">
            <Field label="Alpha">
              <input
                type="number"
                className={numInput + " opacity-60 cursor-not-allowed"}
                value={0.1}
                disabled
                readOnly
              />
            </Field>
            <Field label="Object layer">
              <input
                type="number"
                className={numInput + " opacity-60 cursor-not-allowed"}
                value={modelInfo.defaultObjectLayer}
                disabled
                readOnly
              />
            </Field>
          </div>

          {method === "hiprune_pp" && (
            <Field
              label="Beta"
              hint="text-guided token share"
            >
              <input
                type="number"
                className={numInput}
                step={0.05}
                min={0}
                max={1}
                value={params.beta}
                onChange={(e) => set({ beta: Number(e.target.value) })}
              />
            </Field>
          )}

          {method === "hydart" && (
            <div className="grid grid-cols-2 gap-3">
              <Field label="Lambda seed">
                <input
                  type="number"
                  className={numInput}
                  step={0.05}
                  min={0}
                  value={params.lambdaSeed}
                  onChange={(e) => set({ lambdaSeed: Number(e.target.value) })}
                />
              </Field>
              <Field label="Lambda pick">
                <input
                  type="number"
                  className={numInput}
                  step={0.05}
                  min={0}
                  value={params.lambdaPick}
                  onChange={(e) => set({ lambdaPick: Number(e.target.value) })}
                />
              </Field>
            </div>
          )}

          <Field label="Max new tokens">
            <input
              type="number"
              className={numInput}
              step={16}
              min={1}
              max={1024}
              value={params.maxNewTokens}
              onChange={(e) => set({ maxNewTokens: Number(e.target.value) })}
            />
          </Field>
        </div>
      </Disclosure>

      <Field
        label="Prompt"
        hint={
          method === "hiprune_pp"
            ? "HiPrune++ is prompt-aware"
            : method === "hydart"
              ? "HyDART is NOT prompt-aware"
              : "HiPrune is NOT prompt-aware"
        }
      >
        <textarea
          className="w-full border border-border bg-white p-2.5 text-sm text-fg focus:border-stone-400 focus:outline-none resize-y"
          rows={3}
          value={params.prompt}
          onChange={(e) => set({ prompt: e.target.value })}
        />
      </Field>

      <label className="flex items-center gap-2 text-sm text-fg-muted cursor-pointer select-none">
        <input
          type="checkbox"
          checked={withBaseline}
          onChange={(e) => onWithBaseline(e.target.checked)}
          className="accent-stone-950"
        />
        Also run unpruned baseline for comparison
      </label>

      <button
        type="button"
        onClick={onRun}
        disabled={!canRun || busy}
        className="btn-slide h-12 inline-flex items-center justify-center text-fg-invert font-medium disabled:opacity-40 disabled:cursor-not-allowed"
        style={{ background: "#0C0A09", fontSize: 16 }}
      >
        <span aria-hidden className="btn-slide-fill" style={{ background: "#EA580C" }} />
        <span className="btn-slide-content">
          {busy && (
            <span
              aria-hidden
              className="spin inline-block"
              style={{
                width: 14,
                height: 14,
                border: "2px solid rgba(255,255,255,0.35)",
                borderTopColor: "#fff",
                borderRadius: 999,
              }}
            />
          )}
          {busy ? "Running…" : "Run"}
        </span>
      </button>
    </aside>
  );
}
