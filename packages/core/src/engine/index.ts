// Engine lifecycle contract — public façade.
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
//
// Sweep A-7 split this module into six sibling files; this index re-exports
// every sibling so `@vitrum/core`'s public surface is unchanged.

import type { Scene, ScenePrimitive, SceneEmitter } from '../scene/index.js';
import type { BackendTexture, FrameInput, FrameOutput } from '../frame.js';
import type { InverseSession, InverseSessionOptions } from '../inverse.js';
import type { EngineState } from './state.js';
import type { EngineCapabilities } from './capabilities.js';
import type { EngineDebugSurface } from './debug.js';
import type { FrameStats, ProgressStats, EngineError } from './telemetry.js';

export * from './state.js';
export * from './capabilities.js';
export * from './debug.js';
export * from './telemetry.js';
export * from './factory.js';
export * from './promiseLedger.js';

// ────────────────────────────────────────────────────────────────────────────
// captureFrame — pixel readback contract
// ────────────────────────────────────────────────────────────────────────────

/**
 * Options for {@link Engine.captureFrame}.
 *
 * `colorSpace` selects the source texture:
 *   - `'linear'` (default) — reads the HDR accumulator / pre-tonemap source.
 *     PT backends read the running-mean accumulation buffer; the walkaround
 *     backend reads `resolvedTexture` (the post-denoise, pre-tonemap output).
 *     The result is linear-light RGBA in scene radiance units (not display-
 *     referred), suitable for tone-mapping, EXR export, or luminance checks.
 *   - `'output'` — reads the tonemapped, OETF-encoded present output where one
 *     exists (pt-webgl2 `#presentFbo`, pt-webgpu `presentTexture`).  The
 *     walkaround backend rejects with a clear message because it writes directly
 *     to the host's swap-chain texture (not an engine-owned buffer it can read
 *     back).
 */
export interface CaptureFrameOptions {
  readonly colorSpace?: 'linear' | 'output';
}

/**
 * The result of {@link Engine.captureFrame}: a host-side CPU copy of the
 * engine's rendered output.
 *
 * **Layout convention:**
 *   - `rgba` is row-major, top-left origin (row 0 = top of the image).
 *   - Each pixel is four contiguous `Float32` values: R, G, B, A.
 *   - Stride is `width × 4` float32s per row (no padding).
 *   - For `colorSpace: 'linear'` (default) the values are linear-light HDR;
 *     they may exceed [0, 1]. For `colorSpace: 'output'` the values are
 *     display-referred (tonemapped + OETF where applicable).
 *
 * **Pipeline stall:** `captureFrame` submits a GPU → CPU readback and waits
 * for the GPU to finish. It stalls the render loop for the duration of the
 * copy. Use it for debugging, export, or test assertions — NOT per-frame.
 */
export interface CapturedFrame {
  /** Physical pixel width of the captured image. */
  readonly width: number;
  /** Physical pixel height of the captured image. */
  readonly height: number;
  /**
   * Linear-HDR (or display-referred when `colorSpace:'output'`) RGBA pixels,
   * row-major, top-left origin. Length is `width × height × 4`.
   */
  readonly rgba: Float32Array;
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

  /** Read back the engine's currently-retained scene — the inverse of
   *  {@link setScene}. Lets hosts (e.g. a hero viewer, an editor undo stack)
   *  stop shadowing scene state in parallel with the engine.
   *
   *  **Returns the canonical `@vitrum/core` {@link Scene}** — never a backend
   *  host object. Backends with host-specific internal representations still
   *  return the vitrum Scene they were handed, not a synthesized backend graph.
   *
   *  **Identity / mutation semantics (NOT a defensive copy):** the returned
   *  value is the engine's RETAINED reference to the scene it is currently
   *  rendering. {@link Scene} (and all its members) is declared deeply
   *  `readonly` in this contract, so it is immutable *by contract* — callers
   *  MUST treat the result as frozen and MUST NOT mutate it (e.g. by casting
   *  away `readonly`). A deep clone is deliberately NOT made: scenes carry
   *  large typed-array geometry buffers, and copying them on every read would
   *  be a silent O(scene) cost on a method hosts may poll. To change the scene,
   *  author a new {@link Scene} and call {@link setScene} (or use the
   *  incremental `updatePrimitive` / `addPrimitive` / … paths) — the same
   *  copy-on-write discipline the patch helpers already use.
   *
   *  **What is returned is the SUPPORTED scene, not the raw input.** Backends
   *  capability-filter the incoming scene (warn-and-skip unsupported primitive
   *  kinds / analytic shapes / emitter kinds) before retaining it, so the
   *  returned scene may have fewer primitives than the one passed to
   *  {@link setScene}. This is the scene the engine is actually rendering — the
   *  honest answer to "what is on screen."
   *
   *  Returns `null` when no scene has been set yet (or when the backend dropped
   *  its scene reference on {@link dispose}). Optional: a backend that cannot
   *  retain the canonical core Scene without inventing state omits this method
   *  entirely; hosts MUST `typeof`-check before calling. */
  getScene?(): Scene | null;

  /** Patch a single primitive in-place. Engine MAY internally fall back to a
   *  full `setScene` rebuild if the diff is too disruptive (e.g., changing
   *  geometry vertex counts). Available only when
   *  `capabilities.supportsIncrementalScene = true`. */
  updatePrimitive?(id: string, patch: Partial<ScenePrimitive>): void;

  /** Add one whole primitive to the live scene without the host re-authoring
   *  the full {@link Scene} and calling {@link setScene}. Unlike
   *  `updatePrimitive` (which mutates an EXISTING primitive), this introduces a
   *  NEW primitive — appended to the scene's primitive list — building its
   *  acceleration structure and uploading its geometry / material / emitter
   *  entries. The added primitive is renderable on the next `renderFrame`.
   *
   *  Semantics (a backend that implements this MUST honor them):
   *   • The primitive's `id` MUST be unique among the live scene's primitives.
   *     Adding a primitive whose `id` already exists throws — it is a contract
   *     violation, not a silent update (use `updatePrimitive` to mutate an
   *     existing primitive). The scene is left unchanged on throw.
   *   • Because the scene's geometry changed, accumulation history is invalid:
   *     PT-style engines reset their sample accumulator (frame restarts from
   *     sample 0); real-time engines reinitialise temporal history as on
   *     `setScene`.
   *   • Unsupported primitive kinds / analytic shapes are warn-skipped with the
   *     same capability filter `setScene` applies — they do not throw.
   *
   *  Available only when `capabilities.supportsAddRemovePrimitive = true`;
   *  hosts MUST typeof-check before calling. Backends that report `false` leave
   *  whole-primitive add/remove to a full `setScene`. */
  addPrimitive?(primitive: ScenePrimitive): void;

  /** Remove one whole primitive from the live scene by `id`, the inverse of
   *  {@link addPrimitive}. The evicted primitive's geometry, material, and
   *  emitter entries are dropped, downstream primitives are re-packed densely,
   *  and the primitive is no longer hit by any ray on the next `renderFrame`.
   *
   *  Semantics (a backend that implements this MUST honor them):
   *   • Removing an `id` that is not present in the live scene throws — it is a
   *     contract violation, not a no-op (a host that wants idempotent removal
   *     should query the scene first). The scene is left unchanged on throw.
   *   • Accumulation history is invalidated exactly as for {@link addPrimitive}.
   *   • Removing the last primitive is legal and yields a renderable empty /
   *     sky-only scene (same as `setScene` with no primitives).
   *
   *  Available only when `capabilities.supportsAddRemovePrimitive = true`;
   *  hosts MUST typeof-check before calling. */
  removePrimitive?(id: ScenePrimitive['id']): void;

  /** Patch a single emitter. Same incremental-fallback semantics as above. */
  updateEmitter?(id: string, patch: Partial<SceneEmitter>): void;

  /** Apply an environment-only update (HDRI texture / intensity / rotation
   *  swap, or transition to `kind: 'none'`) without rebuilding the BVH or
   *  re-uploading geometry/materials. Backends that can update the IBL
   *  uniforms in place — pt-webgl2 costs one accumulator reset and no BVH work —
   *  implement this for
   *  fast timeOfDay scrubs on the host side. Backends without a cheap env
   *  path (current HybridEngine is reactive to its own scene-source rather
   *  than host-driven env scrubs) may omit this method; hosts MUST
   *  typeof-check before calling. */
  updateEnvironment?(env: import('../scene/index.js').SceneEnvironment | null): void;
  /** Resize persistent backend render targets. Backends that honour
   *  `FrameInput.viewport` per frame may omit this. */
  setSize?(width: number, height: number): void;
  /**
   * Backend-specific runtime lighting update path for engines that do not map
   * lighting state 1:1 onto `SceneEnvironment`.
   *
   * **Intentional design:** the parameter is kept as `Record<string, unknown>`
   * rather than a typed struct so each backend can accept its own lighting
   * vocabulary without the core contract having to enumerate it. This is the
   * deliberate backend-specific seam — parallel to the `extensions` bag on
   * `EngineOptions`. Backends that consume this method validate and warn on
   * unrecognised keys at runtime rather than at the type layer.
   *
   * **HybridEngine (`@vitrum/walkaround-hybrid`) known keys** (all optional;
   * omitted fields are left unchanged — passing `{}` is a safe no-op):
   *  - `primaryLightDir` (`[number, number, number]`) — Primary directional
   *    light direction in world space (normalised). Triggers a DDGI probe-cache
   *    invalidation and temporal-accumulator reset.
   *  - `primaryLightIntensity` (`number`) — Linear intensity scalar for the
   *    primary directional light. Also drives the DDGI sun-intensity multiplier.
   *  - `skyTint` (`[number, number, number]`) — Diffuse sky-dome RGB tint.
   *  - `skyIrradiance` (`number`) — Sky-dome irradiance scalar paired with
   *    `skyTint`.
   *
   * Backends that honour `updateEnvironment` for env scrubs (PT-style) may
   * omit this method entirely. Hosts MUST `typeof`-check before calling.
   */
  updateLighting?(opts: Readonly<Record<string, unknown>>): void;

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

  /** Reset runtime history/accumulation state. PT engines typically clear
   *  sample accumulation buffers; real-time engines may rebuild temporal
   *  resources or reinitialize history pipelines as needed. */
  reset(): void;

  /** Seed the accumulator with an initial image as a DECAYING PRIOR — the
   *  load-bearing primitive for a progressive walkaround→PT handoff (a
   *  real-time engine's last frame is injected so a freshly-still camera shows
   *  a plausible image immediately instead of a 1-sample blizzard).
   *
   *  Correctness contract a backend that implements this MUST honor: the seed is
   *  a prior of virtual weight `opts.weight` (NOT real samples), so after `M`
   *  real samples accumulate the displayed mean is
   *  `μ + W/(W+M)·(seed − μ)`. The seed's influence W/(W+M) decays to 0, so the
   *  CONVERGED mean is exactly the no-seed result `μ` for ANY seed value — the
   *  seed only smooths the early, still-noisy frames. The virtual weight MUST
   *  NOT be counted as accumulated samples (it must not advance the SPP counter
   *  / `FrameOutput.samplesAccumulated` or convergence/telemetry would
   *  over-report). The accumulator is established CLEAN before the seed lands,
   *  so the seed is the sole prior regardless of prior accumulation.
   *
   *  `opts.width`/`opts.height` are the accumulator (destination) dims; `seed`
   *  may be a different size (the backend resamples it). The seed is treated as
   *  LINEAR HDR radiance (the host must supply linear light, not sRGB).
   *
   *  Available only when `capabilities.supportsAccumulatorSeed === true`; hosts
   *  MUST typeof-check before calling. Backends without an accumulator (e.g.
   *  real-time resample-every-frame engines) omit this method entirely. */
  seedAccumulator?(
    seed: BackendTexture,
    opts: { weight: number; width: number; height: number },
  ): void;

  /** Progressive walkaround→PT seed SOURCE (the counterpart to `seedAccumulator`).
   *  Returns this engine's latest output as a `BackendTexture` + its render dims,
   *  for seeding a *different* engine's accumulator (both engines MUST share one
   *  GPUDevice). Real-time engines (e.g. `@vitrum/walkaround-hybrid`) implement this
   *  to expose their post-denoise HDR radiance (linear, pre-tonemap — matching a PT
   *  accumulator). Null before the first rendered frame; the texture is recycled
   *  per-frame, so a consumer must use it SYNCHRONOUSLY. Available only when
   *  `capabilities.supportsProgressiveSeedSource === true`. */
  getProgressiveSeedTexture?(): {
    texture: BackendTexture;
    width: number;
    height: number;
  } | null;

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

  // ── Telemetry (T3.E) ─────────────────────────────────────────────────────

  /** Subscribe to per-frame stats. Backend invokes the callback synchronously
   *  at the end of each `renderFrame()` call. Returns an unsubscribe function;
   *  call it to stop receiving stats. Subscribers MUST NOT throw — engines
   *  catch and swallow throws to keep the render loop alive. Optional: a
   *  backend that does not implement this still satisfies the contract; the
   *  host should typeof-check before calling. */
  onFrame?(cb: (stats: FrameStats) => void): () => void;

  /** Subscribe to long-running progress events (PT samples-per-pixel
   *  accumulation, denoiser convergence, DDGI warm-up). Backends fire at
   *  their natural cadence (typically once per frame for SPP; less often
   *  for warm-up). Same throw-safety + optionality semantics as
   *  {@link onFrame}. */
  onProgress?(cb: (progress: ProgressStats) => void): () => void;

  /**
   * Subscribe to engine-level GPU/runtime errors.  Returns an unsubscribe
   * function; call it (or dispose the engine) to stop receiving errors.
   *
   * **Wired events (per backend):**
   *  - All WebGPU backends — `GPUDevice.addEventListener('uncapturederror')`
   *    with per-distinct-message throttling (one report per unique message per
   *    32 frames).  Also `GPUDevice.lost.then(...)` → `kind:'device-lost'`,
   *    `fatal:true`, which also transitions the engine to `'error'` state.
   *  - `@vitrum/pt-webgl2` — `webglcontextlost` canvas event →
   *    `kind:'context-lost'`, `fatal:true`.
   *
   * **Contract:** callbacks MUST NOT throw — the engine catches and ignores
   * any thrown exceptions to keep the render loop alive.
   *
   * **Optional** (follows the same opt-in convention as `onFrame` /
   * `onProgress`): all three shipping backends implement this method; a
   * minimal backend that omits it still satisfies the core `Engine` contract.
   * Hosts MUST typeof-check before calling:
   * ```ts
   * const unsub = engine.onError?.(e => console.error(e));
   * ```
   */
  onError?(cb: (error: EngineError) => void): () => void;

  /** Optional debug-introspection surface for dev overlays. When present,
   *  exposes engine-internal state (DDGI atlases, BVH nodes, GI signal
   *  textures, denoiser toggle) that visualisation tools can blit / draw.
   *  Backends that don't implement this still satisfy the core contract;
   *  consumers MUST typeof-check before calling any method. See
   *  {@link EngineDebugSurface}. */
  debug?: EngineDebugSurface;

  // ── Inverse rendering (differentiable RT) ────────────────────────────────

  /** Open an inverse-rendering (differentiable ray tracing) session that
   *  optimizes a tiny scene-parameter vector toward a target image. Returns an
   *  {@link InverseSession} the host drives one `step()` at a time (the engine
   *  owns the optimization loop; the host owns the cadence — same lifecycle
   *  contract as `renderFrame`).
   *
   *  Throws if a requested parameter `path` can't be resolved against the live
   *  scene, or if a requested loss / parameter kind / gradient method isn't
   *  supported by this backend (never a silent no-op). Backends that have no
   *  inverse-rendering path omit this method entirely; hosts MUST typeof-check
   *  before calling. See {@link InverseSessionOptions}. */
  createInverseSession?(opts: InverseSessionOptions): InverseSession;

  // ── Pixel readback ───────────────────────────────────────────────────────

  /**
   * Capture the engine's rendered output as a host-side CPU {@link CapturedFrame}
   * (linear-HDR RGBA Float32, row-major, top-left origin).
   *
   * **Source textures per backend:**
   *   - `@vitrum/pt-webgpu` `colorSpace:'linear'` — reads `accumTexture`
   *     (rgba16float, the Welford running-mean accumulator written by
   *     `accumulateFrame`).  This is the canonical HDR source: every sample
   *     landed here, and it is what OIDN / the present pass reads.
   *     `colorSpace:'output'` reads `presentTexture` (rgba16float, the
   *     tonemapped output written by the present pass after each frame).
   *   - `@vitrum/pt-webgl2` `colorSpace:'linear'` — reads the RGBA32F
   *     accumulation FBO via `gl.readPixels`.  `colorSpace:'output'` reads
   *     the RGBA32F present FBO (the tonemapped output).  Rows are flipped
   *     from GL's bottom-left origin to top-left before returning.
   *   - `@vitrum/walkaround-hybrid` `colorSpace:'linear'` — reads
   *     `resolvedTexture` (rgba16float, the post-denoiser, pre-tonemap output;
   *     the same texture exposed by `getProgressiveSeedTexture()`).
   *     `colorSpace:'output'` rejects: the walkaround backend writes directly
   *     to the host's swap-chain texture (an engine-external buffer), so there
   *     is no engine-owned display-referred surface to read back.
   *
   * **Pipeline stall:** submits a GPU → CPU readback and waits for the GPU.
   * Use for debugging, export, or test assertions — NOT per-frame.
   *
   * Returns `null` before the first frame has been rendered (no source
   * texture allocated yet). Optional: a minimal backend that cannot implement
   * readback omits this method; hosts MUST `typeof`-check before calling.
   *
   * @param opts - `colorSpace: 'linear'` (default) or `'output'`.
   */
  captureFrame?(opts?: CaptureFrameOptions): Promise<CapturedFrame | null>;

  // ── Experimental / backend-specific result buffers ───────────────────────

  /**
   * H14-C — Returns the EXPERIMENTAL ReSTIR-PT resolve-pass output buffer (the
   * per-pixel reconnection-indirect estimate, one vec4f / px = 16 B). Present
   * ONLY when the backend was constructed with `restirPtReuse: true` (gated on
   * the `'pt-webgpu-restir-pt-reuse'` experimental feature) AND the reuse passes
   * have been dispatched at least once (the buffer may be null before the first
   * successful frame). The returned value is backend-opaque (`unknown`) so hosts
   * must narrow it before use (e.g. cast to `GPUBuffer` when the backend is
   * known to be pt-webgpu — see `capabilities.experimentalFeatures`).
   *
   * This result buffer is a SEPARATE debug output from the beauty image: the
   * reuse path is validated in isolation before it composites into the beauty
   * buffer (road-to-100 A1). Hosts MUST typeof-check before calling; backends
   * that do not implement ReSTIR-PT reuse omit this method entirely.
   */
  getRestirPtResultBuffer?(): unknown | null;
}
