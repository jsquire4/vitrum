// @vitrum/dev — shared types for debug overlay components.
//
// FrameStats and ProgressStats are the shapes that T3.E will add to the Engine
// interface (engine.onFrame / engine.onProgress). They're declared here so
// @vitrum/dev can compile independently of T3.E landing. When T3.E ships,
// @vitrum/core will export these same shapes; imports can migrate at that point.

import type { Engine } from '@vitrum/core';

// ────────────────────────────────────────────────────────────────────────────
// T3.E telemetry shapes (anticipated — engine.onFrame / engine.onProgress)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Per-frame stats emitted by engine.onFrame (T3.E).
 * Each field is optional so this type is forward-compatible as engines add more.
 */
export interface FrameStats {
  /** Wall-clock time for this frame in milliseconds. */
  frameTimeMs: number;
  /** GPU timestamp query result, if the backend supports it. */
  gpuTimeMs?: number;
  /** Per-render-pass timing breakdown. Keys are backend-specific (e.g. 'restir-di', 'svgf'). */
  passTimings?: Record<string, number>;
  /** Samples per pixel accumulated (PT-style engines). */
  spp?: number;
  /** BVH depth at root (diagnostic). */
  bvhDepth?: number;
  /** Estimated GPU memory in bytes (from resourceManager totals). */
  estimatedGpuMemoryBytes?: number;
}

/**
 * Progress notification emitted by engine.onProgress (T3.E).
 */
export interface ProgressStats {
  kind: 'pt-spp' | 'denoiser-converge' | 'ddgi-warmup';
  current: number;
  target: number;
  fraction: number;
}

// ────────────────────────────────────────────────────────────────────────────
// Debug engine extension interface (approach (a) per plan)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Optional debug introspection surface an engine may implement.
 * @vitrum/dev components check for `engine.debug` at runtime and degrade
 * gracefully when absent — none of these fields are required by the Engine
 * contract, and HybridEngine does NOT implement them yet.
 *
 * Plan: implement in HybridEngine as part of a T3.G followup sprint.
 * See packages/dev/README.md for the full API plan.
 */
export interface EngineDebugSurface {
  /**
   * Returns the current DDGI irradiance atlas texture, or null if DDGI is
   * not active or the atlas is not yet allocated.
   * Requires engine.debug to be present (T3.G followup).
   */
  atlasTexture?(): GPUTexture | null;

  /**
   * Returns the current DDGI visibility atlas texture, or null.
   * Requires engine.debug to be present (T3.G followup).
   */
  visibilityAtlasTexture?(): GPUTexture | null;

  /**
   * Returns the BVH node bounding-box list as a flat Float32Array of
   * [minX, minY, minZ, maxX, maxY, maxZ, depth, ...] per node.
   * Null when BVH is not built or introspection is not supported.
   * Requires engine.debug to be present (T3.G followup).
   */
  bvhNodes?(): Float32Array | null;

  /**
   * Given a screen-space (x, y) in pixel coords, returns the primitive ID
   * of the hit surface, or null if nothing was hit.
   * Used by MaterialInspector for click-to-inspect.
   * Requires engine.debug to be present (T3.G followup).
   */
  pickPrimitive?(x: number, y: number): string | null;

  /**
   * Returns the current denoiser pass enabled state.
   * Reflects the last setDenoiserEnabled() call (or engine default).
   */
  isDenoiserEnabled?(): boolean;

  /**
   * Enable or disable the denoiser pass for the next frame.
   * When absent, DenoiserABToggle falls back to a console.warn.
   */
  setDenoiserEnabled?(enabled: boolean): void;

  /**
   * Returns the GI signal split textures (direct, indirect, AO).
   * All are null when the engine doesn't expose split-signal outputs.
   * Requires engine.debug to be present (T3.G followup).
   */
  giSignalTextures?(): {
    direct: GPUTexture | null;
    indirect: GPUTexture | null;
    ao: GPUTexture | null;
    total: GPUTexture | null;
  } | null;
}

/**
 * Engine with the optional T3.E telemetry hooks and T3.G debug surface.
 * Cast a real Engine to this type to access debug APIs — all fields are
 * optional, so it's safe to cast even when the engine doesn't implement them.
 */
export interface DebuggableEngine extends Engine {
  /**
   * Subscribe to per-frame stats (T3.E). Returns an unsubscribe function.
   * When absent (T3.E not yet landed), overlays fall back to rAF timing.
   */
  onFrame?(cb: (stats: FrameStats) => void): () => void;

  /**
   * Subscribe to long-running progress notifications (T3.E).
   * Returns an unsubscribe function.
   */
  onProgress?(cb: (progress: ProgressStats) => void): () => void;

  /**
   * Optional debug introspection surface (T3.G followup).
   * Components check for this before calling any debug API.
   */
  debug?: EngineDebugSurface;
}
