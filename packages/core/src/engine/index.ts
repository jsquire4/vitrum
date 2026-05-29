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
import type { FrameInput, FrameOutput } from '../frame.js';
import type { InverseSession, InverseSessionOptions } from '../inverse.js';
import type { EngineState } from './state.js';
import type { EngineCapabilities } from './capabilities.js';
import type { EngineDebugSurface } from './debug.js';
import type { FrameStats, ProgressStats } from './telemetry.js';

export * from './state.js';
export * from './capabilities.js';
export * from './debug.js';
export * from './telemetry.js';
export * from './factory.js';
export * from './promiseLedger.js';

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
   *  uniforms in place — pt-webgl wraps WebGLPathTracer.updateEnvironment(),
   *  which costs one accumulator reset and no BVH work — implement this for
   *  fast timeOfDay scrubs on the host side. Backends without a cheap env
   *  path (current HybridEngine is reactive to its own scene-source rather
   *  than host-driven env scrubs) may omit this method; hosts MUST
   *  typeof-check before calling. */
  updateEnvironment?(env: import('../scene/index.js').SceneEnvironment | null): void;
  /** Resize persistent backend render targets. Backends that honour
   *  `FrameInput.viewport` per frame may omit this. */
  setSize?(width: number, height: number): void;
  /** Backend-specific runtime lighting update path for engines that do not map
   *  lighting state 1:1 onto `SceneEnvironment`. */
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
}
