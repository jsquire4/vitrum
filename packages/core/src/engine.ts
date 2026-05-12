// Engine lifecycle contract.
//
// Design principle (the white-whale insight):
// **The engine accepts a device handle but does NOT own the device's lifetime.**
// The host owns: when the device is created, when it's lost, when it's reset.
// The engine owns: GPU resources allocated against that device (BVH buffers,
// reservoirs, accumulation textures, MLP weights, etc.).
//
// This is the contract that resolves the "cells go grey + reaccumulate from
// frame 0" bug class. The engine survives any host topology change (Canvas
// remount, route change, tab visibility) as long as the device is still alive.
// If the device IS lost, the host calls `engine.dispose()` and creates a
// fresh engine with the new device.
//
// Frame cadence is the host's responsibility. The engine never starts its
// own RAF loop. The host calls `engine.renderFrame(input)` whenever it wants
// a frame — typically inside `requestAnimationFrame`, but a host that wants
// to drive frames manually (offline render, headless test, video encoder)
// can do so without the engine fighting it.

import type { Scene, ScenePrimitive, SceneEmitter } from './scene.js';
import type { FrameInput, FrameOutput } from './frame.js';

// ────────────────────────────────────────────────────────────────────────────
// Engine state
// ────────────────────────────────────────────────────────────────────────────

export type EngineState =
  | 'uninitialized'
  | 'initializing'
  | 'ready'
  | 'paused'
  /** Unrecoverable init/runtime failure — GPU resources torn down; recreate the engine. */
  | 'error'
  | 'disposed';

// ────────────────────────────────────────────────────────────────────────────
// Engine capabilities (engine → host, queried after init)
// ────────────────────────────────────────────────────────────────────────────

export interface EngineCapabilities {
  /** Engine supports `updatePrimitive` / `updateEmitter` patches, falling
   *  back to full `setScene` for unsupported diffs. When false, hosts must
   *  always call `setScene` for any change. */
  readonly supportsIncrementalScene: boolean;

  /** Engine consumes `FrameInput.shutterTime`. */
  readonly supportsMotionBlur: boolean;

  /** Engine reports `FrameOutput.variance` and `FrameOutput.motionVectors`,
   *  enabling external denoisers + adaptive sampling. */
  readonly supportsAuxBuffers: boolean;

  /** Engine continues accumulating samples after temporal stability is
   *  reached (PT-style hero render). When false, engine resamples every
   *  frame (walkaround-style real-time). */
  readonly accumulates: boolean;

  /** Structural cap: the maximum samples-per-pixel this engine instance was
   *  allocated for. PT engines stop accumulating at this ceiling; walkaround
   *  engines report Infinity (they resample every frame rather than
   *  accumulating). Per-frame `FrameInput.quality.samplesTarget` is clamped
   *  to this value. */
  readonly maxSamplesPerPixel: number;

  /** Structural cap: the maximum bounces per path this engine instance was
   *  allocated for. Determined at engine creation by `EngineOptions.maxBounces`
   *  (or the backend's default if omitted). Per-frame
   *  `FrameInput.quality.bounces` is clamped to this value. */
  readonly maxBounces: number;

  /** Set of analytic-primitive `kind` values this engine supports. */
  readonly supportedAnalyticShapes: ReadonlySet<string>;

  /** Set of emitter `kind` values this engine supports. */
  readonly supportedEmitterKinds: ReadonlySet<string>;

  // ── Specular caustics (RFE-05) ──────────────────────────────────────────
  /**
   * Whether this engine instance was created with a caustic strategy.
   * 'none' means standard NEE only; consumers should not expect fast
   * caustic convergence. Other values indicate the active strategy.
   *
   * Reference: Hanika, Droske, Fascione, "Manifold Next Event Estimation,"
   * CGF 34(4), 2015.
   */
  readonly causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
}

// ────────────────────────────────────────────────────────────────────────────
// Engine — the public façade
// ────────────────────────────────────────────────────────────────────────────

/** There is intentionally NO `updateOptions()` method on this interface.
 *  Per-frame quality dials (samplesTarget, bounces, resolutionFactor,
 *  filteredGlossyFactor) live on `FrameInput.quality`. The host changes them
 *  by passing a different quality payload each frame — not by mutating the
 *  engine. Creation-time configuration (`EngineOptions`) is immutable for the
 *  engine's lifetime. If a structural change is needed (different denoiser
 *  pipeline, different structural caps), the host disposes the engine and
 *  creates a fresh one. */
export interface Engine {
  readonly state: EngineState;
  readonly capabilities: EngineCapabilities;

  // ── Scene management ────────────────────────────────────────────────────

  /** Replace the entire scene. Triggers a full BVH/light-tree rebuild. Cheap
   *  if the scene hasn't changed (engines may compare structural hashes). */
  setScene(scene: Scene): void;

  /** Patch a single primitive in-place. Engine MAY internally fall back to a
   *  full `setScene` rebuild if the diff is too disruptive (e.g., changing
   *  geometry vertex counts). Available only when
   *  `capabilities.supportsIncrementalScene = true`. */
  updatePrimitive?(id: string, patch: Partial<ScenePrimitive>): void;

  /** Patch a single emitter. Same incremental-fallback semantics as above. */
  updateEmitter?(id: string, patch: Partial<SceneEmitter>): void;

  // ── Frame-level rendering ───────────────────────────────────────────────

  /** Render one sample/frame and return references to the engine's output
   *  buffers. The host owns frame cadence; this method is the host's "tick."
   *
   *  PT-style engines: each call accumulates one sample into the running
   *  buffer. `FrameOutput.isConverged` flips true when `samplesAccumulated`
   *  reaches `min(input.quality.samplesTarget ?? Infinity, capabilities.maxSamplesPerPixel)`.
   *
   *  Walkaround-style engines: each call computes one fresh frame; output
   *  buffer is overwritten. */
  renderFrame(input: FrameInput): FrameOutput;

  /** Reset the accumulator. Hosts call this when the camera moves, the scene
   *  changes, or the user wants to start over. Engines may also reset
   *  internally on `setScene`. */
  reset(): void;

  // ── Pause / resume / dispose ────────────────────────────────────────────

  /** Skip per-frame compute work but keep all GPU resources allocated.
   *  Hosts call this when the engine's output isn't visible (Canvas hidden,
   *  tab backgrounded, route navigated away). */
  pause(): void;

  /** Resume per-frame compute. Engine state goes from 'paused' → 'ready'.
   *  Accumulator state is preserved across pause/resume. */
  resume(): void;

  /** Free all engine-owned GPU resources. The device handle remains valid;
   *  the host can dispose the device separately if it owns the device's
   *  lifetime. After dispose, the engine state is `'disposed'` and no method
   *  except `state` and `capabilities` is valid. For engines that surfaced
   *  `'error'`, callers should dispose before recreating. */
  dispose(): void;
}

// ────────────────────────────────────────────────────────────────────────────
// Backend factory contract
// ────────────────────────────────────────────────────────────────────────────

/** All engine-creation factories follow this shape. The `device` is opaque at
 *  the core level; each backend narrows `device` to its own concrete type.
 *  Examples: `@vitrum/pt-webgl` narrows to `THREE.WebGLRenderer` (the backend
 *  wraps three-gpu-pathtracer and bakes IBL); `@vitrum/pt-webgpu` narrows to
 *  `GPUDevice` (the backend uses compute shaders with no Three.js coupling).
 *  Each backend's package documents its concrete device type via the options
 *  interface that extends `EngineOptions`. */
export type EngineFactory<TOptions extends EngineOptions = EngineOptions> = (
  opts: TOptions,
) => Promise<Engine>;

/** Immutable creation-time configuration passed to an engine factory. Once
 *  the engine exists, this configuration does not change.
 *
 *  Per-frame quality dials — samplesTarget, bounces, resolutionFactor,
 *  filteredGlossyFactor — are NOT engine identity and do NOT belong here.
 *  They live on `FrameInput.quality` and are supplied by the host each frame.
 *
 *  What belongs here: the device handle (engine is bound to one device for
 *  its lifetime), the denoiser pipeline structure (changing it requires
 *  shader recompilation, i.e. a new engine), structural buffer-allocation
 *  caps (`maxBounces`, `maxSamplesPerPixel` — allocators may use these to
 *  size accumulator precision or sample-counter types), and extensions
 *  (backend-specific creation-time config). */
export interface EngineOptions {
  /** The graphics device handle. Backend-specific type is enforced via
   *  package-level overloads. */
  readonly device: unknown;

  // ── Structural caps (buffer allocation upper bounds) ─────────────────────
  /** Structural cap on per-path bounce count. Backends may use this to size
   *  path-state buffers or accumulator array dimensions. Per-frame
   *  `FrameInput.quality.bounces` is clamped to this value.
   *  Default: backend-specific (e.g., pt-webgl defaults to 12). */
  readonly maxBounces?: number;

  /** Structural cap on samples-per-pixel. Backends may use this to choose
   *  accumulator precision (e.g., FP16 vs FP32) or size sample-counter
   *  types. Per-frame `FrameInput.quality.samplesTarget` is clamped to this
   *  value. Default: backend-specific (e.g., pt-webgl defaults to 4096). */
  readonly maxSamplesPerPixel?: number;

  // ── Denoiser composition ────────────────────────────────────────────────
  /** Denoiser pipeline wired at engine creation. Changing the denoiser
   *  requires recompiling shaders and resizing auxiliary buffers — so it is
   *  a creation-time structural decision, not a per-frame dial. */
  /** `'svgf'` is a deprecated alias for `'atrous-variance'`; backends that ship
   *  à-trous + variance-scalar lookup should accept both.
   *
   *  `'svgf-real'` — T2.H1 — full Schied 2017 SVGF with bilinear motion-vector
   *  reprojection, depth+normal+objId disocclusion test (Eq. 2), per-pixel
   *  history-length texture (Eq. 3), EMA α=max(α_min, 1/(h+1)) (Eq. 4),
   *  variance-from-moments (Eq. 5), and 7×7 spatial fallback for disoccluded
   *  pixels (§4.3). Implemented in `@vitrum/shared-denoisers` and wired in
   *  `@vitrum/walkaround-hybrid`.
   *
   *  GPU memory budget for `'svgf-real'` at 1080p: ~52 MB of new persistent
   *  textures (historyLength r16uint + momentsHistory rg32float + prevRadiance
   *  rgba16float + motionVec rg32float). */
  /**
   * `'neural'` — T2.H2 — GPU U-Net denoiser. Requires backend-specific weight
   * provisioning (e.g. `HybridEngineOptions.neuralWeights` in
   * `@vitrum/walkaround-hybrid`). Opt-in; default remains `'atrous-variance'`.
   */
  readonly denoiser?: 'none' | 'atrous' | 'atrous-variance' | 'svgf' | 'svgf-real' | 'bmfr' | 'oidn-final' | 'neural';

  // ── Specular caustics strategy (RFE-05) ────────────────────────────────
  /**
   * Strategy for handling specular-chain caustic paths (LS+E, LSS+E, …).
   *
   * 'none':          No special caustic handling. Standard NEE only. Caustics
   *                  accumulate slowly via BSDF-sampled paths (may require many
   *                  thousands of samples to converge).
   *
   * 'manifold-nee':  Manifold Next-Event Estimation (Hanika et al. 2015).
   *                  At each diffuse vertex, launch a manifold walk to find
   *                  valid specular connections to sampled light positions.
   *                  Unbiased. Adds per-shading-event cost proportional to
   *                  the number of specular interfaces (typically 2–5 Newton
   *                  steps per walk attempt). May fail for highly curved or
   *                  rough specular surfaces.
   *
   * 'photon-map':    Biased photon mapping for caustics. Trace forward photons
   *                  from lights; store caustic photons in a spatial data
   *                  structure; use density estimation at diffuse shading points
   *                  to reconstruct caustic radiance. Biased but robust.
   *
   * Default: 'none'.
   *
   * Reference: Hanika, Droske, Fascione, "Manifold Next Event Estimation,"
   * Computer Graphics Forum 34(4), 2015. DOI: 10.1111/cgf.12681.
   */
  readonly causticStrategy?: 'none' | 'manifold-nee' | 'photon-map';

  /**
   * Caustic-strategy-specific tuning knobs. Backends ignore entries that don't
   * apply to the selected `causticStrategy`.
   *
   * Known keys:
   *  - `mneeMaxIterations` (number, default 8) — MNEE Newton iterations per
   *    manifold walk attempt. Active when `causticStrategy === 'manifold-nee'`.
   *  - `mneeMaxChainLength` (number, default 3) — Maximum specular vertices
   *    in an MNEE chain. Active when `causticStrategy === 'manifold-nee'`.
   *
   * The signature is open-ended so new strategies (photon-map params, etc.)
   * can add keys without churning the core contract.
   */
  readonly causticOptions?: Readonly<{
    mneeMaxIterations?: number;
    mneeMaxChainLength?: number;
    [key: string]: unknown;
  }>;

  // ── Backend-specific extensions ─────────────────────────────────────────
  /** Engines look up extension keys here for backend-specific creation-time
   *  config that doesn't fit the generic options. Backends document their own
   *  keys. */
  readonly extensions?: Readonly<Record<string, unknown>>;
}
