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
  /** BVH max depth — diagnostic for traversal cost. */
  readonly bvhDepth?: number;
  /**
   * Approximate engine-owned GPU memory (sum of texture + buffer bytes).
   * Backwards-compatible scalar form — emitted by backends that have not
   * been wired up to the structured breakdown. Hosts that want a structured
   * report should prefer {@link FrameStats.gpuMemoryBytes}.
   *
   * @deprecated Prefer {@link FrameStats.gpuMemoryBytes}; this scalar
   * remains for backends that have not yet integrated the structured
   * surface.
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
