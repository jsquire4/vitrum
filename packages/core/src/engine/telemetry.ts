// Engine telemetry (T3.E).
//
// Split from the original `engine.ts` (sweep A-7). Per-frame `FrameStats` and
// long-running `ProgressStats` events surfaced via `Engine.onFrame` /
// `Engine.onProgress`. The structured GPU-memory shape lives in `./debug.ts`
// (single source of truth); we re-use it here via `import type` to avoid any
// runtime coupling.

import type { GpuMemoryBreakdown } from './debug.js';

/** Per-frame statistics surfaced via {@link Engine.onFrame}.  Optional
 *  fields reflect backend capability — `gpuTimeMs` requires
 *  `timestamp-query` on the WebGPU side; `passTimings` requires per-pass
 *  instrumentation; `spp` is meaningful only for accumulating engines. */
export interface FrameStats {
  /** Wall-clock duration of `renderFrame()` in milliseconds. */
  readonly frameTimeMs: number;
  /** GPU-side execution time if timestamp queries are available. */
  readonly gpuTimeMs?: number;
  /** Optional per-pass breakdown (label → milliseconds). */
  readonly passTimings?: Readonly<Record<string, number>>;
  /** Samples accumulated this frame (PT-style engines).  Walkaround engines
   *  emit `1`. */
  readonly spp?: number;
  /**
   * Approximate engine-owned GPU memory in bytes (sum of texture + buffer
   * bytes). Scalar form, intended as the lowest common denominator: every
   * backend can produce SOME estimate even when it can't break the bytes
   * down by category (e.g. a backend may own resources through an opaque host
   * API, so a structured split would be invented or stale).
   *
   * When a backend can produce a meaningful structured split, it emits
   * {@link FrameStats.gpuMemoryBytes} alongside this scalar. Hosts that
   * want the breakdown prefer the structured form; hosts that just need
   * a single number read this field.
   */
  readonly estimatedGpuMemoryBytes?: number;
  /**
   * Structured GPU-memory breakdown when the backend can build one.
   * Same shape as {@link EngineDebugSurface.estimatedGpuMemoryBytes}; the
   * stats hook is a streaming convenience so dev overlays can show a live
   * gauge without polling debug methods. Backends emit either this OR
   * `estimatedGpuMemoryBytes` (or neither). The structured form is
   * preferred — when both are present, prefer this and treat the scalar
   * as `gpuMemoryBytes.total`.
   */
  readonly gpuMemoryBytes?: GpuMemoryBreakdown;
  /**
   * Current denoiser lifecycle state. Present when a denoiser is
   * configured; absent (`undefined`) when no denoiser option was passed to
   * the engine constructor.
   *
   * `status` values:
   *  - `'ready'`      — denoiser is active and producing filtered output.
   *  - `'warming-up'` — async backend (OIDN model, neural weights) is
   *                     still loading; raw HDR is returned this frame.
   *  - `'in-flight'`  — async denoiser inference is running; previous
   *                     denoised frame (if any) is displayed this frame.
   *  - `'fallback'`   — denoiser was configured but is currently
   *                     pass-through (e.g. first frame before OIDN
   *                     inference completes, or no InferenceGraph).
   *  - `'failed'`     — a non-recoverable or retryable error occurred;
   *                     see `reason`. `retryable` distinguishes transient
   *                     from permanent failures.
   *  - `'disabled'`   — denoiser option is present but resolves to no-op
   *                     (e.g. no OIDN model URL supplied).
   *
   * Union is intentionally aligned with the `DenoiserState` type in
   * `walkaround-hybrid/src/pipeline/denoisers/index.ts` (extended with
   * `'disabled'` for backends that omit per-frame denoising entirely).
   */
  readonly denoiserState?: {
    readonly status: 'ready' | 'warming-up' | 'in-flight' | 'fallback' | 'failed' | 'disabled';
    readonly reason: string | null;
    readonly retryable?: boolean;
  };
}

/** Progress event surfaced via {@link Engine.onProgress}.  The discriminator
 *  is `kind`; consumers switch on it to interpret `current` / `target`. */
export interface ProgressStats {
  readonly kind: 'pt-spp' | 'denoiser-converge' | 'ddgi-warmup';
  /** Current value in the kind-appropriate unit (samples for `pt-spp`,
   *  frames for `denoiser-converge`, probe-update passes for
   *  `ddgi-warmup`). */
  readonly current: number;
  /** Target value at which `fraction` reaches 1. May be `Infinity` for
   *  open-ended walkaround warm-ups. */
  readonly target: number;
  /** Convenience: `clamp(current / target, 0, 1)`. */
  readonly fraction: number;
}
