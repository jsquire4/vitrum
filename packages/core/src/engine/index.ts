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

import type { Scene, ScenePrimitive, SceneEmitter } from '../scene.js';
import type { FrameInput, FrameOutput } from '../frame.js';
import type { EngineState } from './state.js';
import type { EngineCapabilities } from './capabilities.js';
import type { EngineDebugSurface } from './debug.js';
import type { FrameStats, ProgressStats } from './telemetry.js';

export * from './state.js';
export * from './capabilities.js';
export * from './debug.js';
export * from './telemetry.js';
export * from './factory.js';

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

  /** Apply an environment-only update (HDRI texture / intensity / rotation
   *  swap, or transition to `kind: 'none'`) without rebuilding the BVH or
   *  re-uploading geometry/materials. Backends that can update the IBL
   *  uniforms in place — pt-webgl wraps WebGLPathTracer.updateEnvironment(),
   *  which costs one accumulator reset and no BVH work — implement this for
   *  fast timeOfDay scrubs on the host side. Backends without a cheap env
   *  path (current HybridEngine is reactive to its own scene-source rather
   *  than host-driven env scrubs) may omit this method; hosts MUST
   *  typeof-check before calling. */
  updateEnvironment?(env: import('../scene.js').SceneEnvironment | null): void;

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
}
