// Inverse-rendering (differentiable ray tracing) contract — public façade.
//
// "Reference image → matching 3D scene." An inverse session drives a tiny
// optimizable parameter vector (material albedo / roughness, emitter
// intensity, …) toward a target image by repeatedly rendering, measuring an
// image-space loss, and descending the parameter gradient.
//
// Design principle (the lifecycle contract, applied to optimization):
//   **The engine owns the optimization LOOP; the host owns the CADENCE.**
// Just as `renderFrame` is the host's per-frame tick, `step()` is the host's
// per-optimization-iteration tick. The host calls `step()` whenever it wants
// to advance the optimizer (typically inside its own RAF / worker loop so the
// UI can paint a preview between steps); the session never starts its own
// loop. This is what lets a host show live convergence, pause, or cancel.
//
// Two gradient strategies are exposed, mirroring the research literature:
//   • 'finite-difference' (Phase 0) — perturb each parameter by ±ε, re-render,
//     forward-difference the loss. Backend-agnostic, no adjoint shader, O(P)
//     renders per step. The honest baseline that validates the session UX.
//   • 'path-replay' (Phase 1) — re-trace the forward path with the SAME frozen
//     RNG seed and differentiate only the continuous shading (the sampled
//     direction is held constant, sidestepping the visibility / sampling
//     discontinuities). One adjoint render per step.
//     Ref: Vicini, Speierer, Jakob, "Path Replay Backpropagation," ACM TOG
//          40(4) (SIGGRAPH 2021); Nimier-David, Vicini, Zeltner, Jakob,
//          "Radiative Backpropagation," ACM TOG 39(4) (SIGGRAPH 2020).
//
// The contract is intentionally small: it names the optimizable parameters,
// the target image, the loss, and an optimizer config, and it returns a
// per-step result (loss + current values + optional preview handle). It does
// NOT expose the per-parameter gradient buffer or the adjoint internals —
// those are backend-private. Backends that cannot differentiate a requested
// parameter kind throw at `createInverseSession`, never silently no-op.

import type { BackendTexture } from './frame.js';

// ────────────────────────────────────────────────────────────────────────────
// Optimizable parameters
// ────────────────────────────────────────────────────────────────────────────

/** Which scene quantity an {@link InverseParam} addresses. The `path` string
 *  selects the target the way the illustrative API in `plan/differentiable-rt.md`
 *  does: `materials.<primitiveId>.<field>` or `emitters.<emitterId>.<field>`.
 *
 *  Phase-0/Phase-1 scope (this wave): scalar and rgb material/emitter params.
 *  `texture` is reserved for Phase 2 (texture optimization) and is part of the
 *  FIXED contract surface now so adding it later is not a breaking change —
 *  backends that don't yet differentiate textures throw on it. */
export type InverseParamKind = 'scalar' | 'rgb' | 'texture';

/** One optimizable parameter. `path` is a dotted address into the live scene
 *  (`'materials.panel-1.roughness'`, `'materials.panel-1.albedo'`,
 *  `'emitters.sun.intensity'`). The backend resolves the path against the
 *  scene it was given; an unresolvable path throws at session creation.
 *
 *  `min`/`max` clamp the parameter after each optimizer step (physically:
 *  roughness ∈ [0,1], albedo channels ∈ [0,1], intensity ≥ 0). When omitted the
 *  backend applies a physically-sensible default clamp for the resolved field. */
export interface InverseParam {
  /** Dotted address into the scene. See examples above. */
  readonly path: string;
  /** Continuity class of the parameter (drives gradient + packing). */
  readonly kind: InverseParamKind;
  /** Optional initial value override. When omitted the session reads the
   *  current scene value as the starting point. Length must match the kind
   *  (`scalar` → 1, `rgb` → 3). */
  readonly initial?: readonly number[];
  /** Optional lower clamp (a single scalar applied to EVERY component) enforced
   *  after each optimizer step. e.g. `min: 0` keeps an RGB albedo non-negative. */
  readonly min?: number;
  /** Optional upper clamp (a single scalar applied to EVERY component) enforced
   *  after each optimizer step. e.g. `max: 1` keeps an RGB albedo ≤ 1. */
  readonly max?: number;
}

// ────────────────────────────────────────────────────────────────────────────
// Loss + optimizer config
// ────────────────────────────────────────────────────────────────────────────

/** Image-space loss between the rendered image and the target. `'l2'` is the
 *  mean-squared per-pixel RGB error (the Phase-0/1 default). `'l1'` is the mean
 *  absolute error. `'ssim'` / `'lpips'` are reserved perceptual losses (named
 *  now so they don't churn the contract later); backends that don't implement a
 *  requested loss throw at session creation. */
export type InverseLoss = 'l2' | 'l1' | 'ssim' | 'lpips';

/** Gradient strategy. See the module header for the literature mapping. */
export type InverseGradientMethod = 'finite-difference' | 'path-replay';

/** Structured reason an inverse-rendering backend downgraded or scoped a
 * requested optimization path. These diagnostics are intentionally contract
 * level, not backend-log strings, so hosts can surface predictable UI and
 * compatibility reports for arbitrary assets. */
export interface InverseSessionDiagnostic {
  readonly severity: 'info' | 'warning';
  readonly code:
    | 'path-replay-hook-missing'
    | 'path-replay-unsupported-param-domain'
    | 'path-replay-unsupported-field'
    | 'path-replay-unsupported-primitive'
    | 'path-replay-unsupported-material'
    | 'path-replay-unsupported-lighting'
    | 'path-replay-unsupported-emitter'
    | 'path-replay-unsupported-receiver';
  /** Parameter path, scene path, or backend-local path the diagnostic refers to. */
  readonly path?: string;
  readonly message: string;
  readonly details?: Record<string, string | number | boolean | readonly string[]>;
}

/** Optimizer hyper-parameters. Defaults match a small-vector Adam
 *  (Kingma & Ba 2015). `learningRate` is the only knob most hosts set. */
export interface InverseOptimizerConfig {
  /** Step size. Default: 1e-2. */
  readonly learningRate?: number;
  /** Adam β1 (first-moment decay). Default: 0.9. */
  readonly beta1?: number;
  /** Adam β2 (second-moment decay). Default: 0.999. */
  readonly beta2?: number;
  /** Adam ε (numerical floor). Default: 1e-8. */
  readonly epsilon?: number;
  /** Finite-difference probe size (only used when method ===
   *  'finite-difference'). Default: 1e-3. */
  readonly fdEpsilon?: number;
}

/** Creation-time configuration for an inverse-rendering session. */
export interface InverseSessionOptions {
  /** The target image to match, as interleaved RGB (or RGBA — alpha ignored)
   *  float pixels in scanline order, plus its dimensions. The session renders
   *  at this resolution. (A host with an `ImageBitmap` / `<canvas>` decodes it
   *  to this layout once; the core contract stays DOM-free.) */
  readonly target: InverseTargetImage;
  /** Parameters to optimize. Must be non-empty; every `path` must resolve
   *  against the live scene or `createInverseSession` throws. */
  readonly parameters: readonly InverseParam[];
  /** Image-space loss. Default: 'l2'. */
  readonly loss?: InverseLoss;
  /** Gradient strategy. Default: 'finite-difference' (Phase 0). */
  readonly method?: InverseGradientMethod;
  /** Samples-per-pixel to accumulate before measuring the loss each step.
   *  Higher = less Monte-Carlo noise in the gradient, slower steps.
   *  Default: backend-specific (typically a small fixed budget). */
  readonly samplesPerStep?: number;
  /** Optimizer hyper-parameters. */
  readonly optimizer?: InverseOptimizerConfig;
  /** Optional structured diagnostics emitted during session creation, most
   *  commonly when a requested `'path-replay'` session downgrades to
   *  `'finite-difference'` for a specific asset/material/light feature. */
  readonly onDiagnostic?: (diagnostic: InverseSessionDiagnostic) => void;
}

/** A decoded target image: interleaved float RGB(A) in scanline order. */
export interface InverseTargetImage {
  /** Interleaved pixel data, scanline order. `channels` floats per pixel. */
  readonly data: Float32Array;
  readonly width: number;
  readonly height: number;
  /** 3 (RGB) or 4 (RGBA, alpha ignored by the loss). Default: 3. */
  readonly channels?: 3 | 4;
}

// ────────────────────────────────────────────────────────────────────────────
// Per-step result
// ────────────────────────────────────────────────────────────────────────────

/** The outcome of one optimizer step. */
export interface InverseStepResult {
  /** Zero-based optimizer step index (increments each `step()`). */
  readonly step: number;
  /** Image-space loss measured this step (lower = closer to the target). */
  readonly loss: number;
  /** Current value of every optimized parameter, in the same order as
   *  `InverseSessionOptions.parameters`. Each entry is the flat component list
   *  for that parameter (`scalar` → length 1, `rgb` → length 3). A copy — the
   *  host may keep or mutate it freely without affecting the session. */
  readonly values: readonly (readonly number[])[];
  /** The gradient the optimizer consumed this step, same shape as `values`.
   *  Exposed for diagnostics / convergence plots; it is a copy. */
  readonly gradient: readonly (readonly number[])[];
  /** Optional handle to the most-recent rendered preview image (backend
   *  texture). Hosts may blit it to show live convergence. Omitted by backends
   *  that don't surface an intermediate texture. */
  readonly preview?: BackendTexture;
}

// ────────────────────────────────────────────────────────────────────────────
// The session
// ────────────────────────────────────────────────────────────────────────────

/** A live inverse-rendering optimization. Created by
 *  {@link import('./engine/index.js').Engine.createInverseSession}; the engine
 *  it came from must outlive it. The host drives the loop one `step()` at a
 *  time (the session never self-schedules) and `dispose()`s it when done.
 *
 *  A session mutates the engine's scene as it optimizes (it pushes parameter
 *  updates through the engine's incremental-update path each step). On
 *  `dispose()` the final optimized values remain applied to the scene — the
 *  host keeps the fit. */
export interface InverseSession {
  /** The number of optimizable parameters (= `options.parameters.length`). */
  readonly parameterCount: number;

  /** The gradient method actually in use (resolved from options + backend
   *  capability — a backend may fall back from 'path-replay' to
   *  'finite-difference' for a parameter kind it can't yet differentiate, and
   *  reports the effective method here). */
  readonly method: InverseGradientMethod;

  /** Creation-time diagnostics for method downgrades or scoped inverse-rendering
   *  support. Empty/omitted means the requested method was accepted without a
   *  compatibility caveat. */
  readonly diagnostics?: readonly InverseSessionDiagnostic[];

  /** Advance the optimizer by one step: render at the target resolution,
   *  measure the loss, compute the gradient (finite-difference re-renders or a
   *  path-replay adjoint pass), and apply one optimizer update to the parameter
   *  vector. Async because it reads pixels back from the GPU (`mapAsync`).
   *  Resolves with the {@link InverseStepResult} for this step. */
  step(): Promise<InverseStepResult>;

  /** The current parameter values without taking a step (a copy, same shape as
   *  {@link InverseStepResult.values}). */
  currentValues(): readonly (readonly number[])[];

  /** Release session-owned GPU resources (adjoint buffers, readback buffers,
   *  optimizer state). Idempotent. Does NOT dispose the engine — the engine
   *  outlives the session — and does NOT revert the scene to its pre-session
   *  state; the optimized values stay applied. */
  dispose(): void;
}
