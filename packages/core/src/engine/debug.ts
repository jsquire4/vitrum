// Debug-introspection surface (T3.G + T3.G followup).
//
// Split from the original `engine.ts` (sweep A-7). Optional surface that a
// backend may expose via `engine.debug`. Dev overlays consume this; every
// method is optional so production hosts pay nothing.

/** Optional debug-introspection surface an engine may expose via `engine.debug`.
 *  Every method is optional — dev tools call only those they need and
 *  fall back when absent. Returned WebGPU texture handles are owned by the
 *  engine (the caller MUST NOT destroy them); they're invalidated on the
 *  next setScene() / dispose(). */
export interface EngineDebugSurface {
  /** A3 (2026-05-19) — the engine's GPUDevice handle (WebGPU backends)
   *  or null (WebGL backends or pre-init). Dev overlays (DDGIAtlasViewer,
   *  BVHVisualizer, GISignalSplit) need this to issue
   *  `copyTextureToBuffer` + `mapAsync(READ)` readbacks for their
   *  canvas-blit rendering. Backends MUST NOT destroy or hand off
   *  ownership — the caller is read-only. */
  device?(): GPUDevice | null;

  /** DDGI irradiance atlas (the GPUTexture the probe-update pass writes
   *  to). Returns null when DDGI is disabled or not yet initialised. */
  atlasTexture?(): GPUTexture | null;

  /** DDGI visibility-atlas companion to {@link atlasTexture}. */
  visibilityAtlasTexture?(): GPUTexture | null;

  /** Flat per-node AABB list: 8 floats per node — `[minX, minY, minZ,
   *  maxX, maxY, maxZ, depth, pad]`. Returns null when the BVH is not
   *  built or introspection is not wired. */
  bvhNodes?(): Float32Array | null;

  /** Screen-space pixel hit-test. Returns the primitive ID at (x, y) or
   *  null when nothing is hit. Used by MaterialInspector for click-pick. */
  pickPrimitive?(x: number, y: number): string | null;

  /** Current denoiser-pass enabled state; mirrors the last
   *  {@link setDenoiserEnabled} call (or the engine default). */
  isDenoiserEnabled?(): boolean;

  /** Toggle the denoiser pass for the next frame. */
  setDenoiserEnabled?(enabled: boolean): void;

  /** Per-channel GI signal textures for split-screen visualisation. Any
   *  field may be null when the backend doesn't separate that signal. */
  giSignalTextures?(): {
    direct: GPUTexture | null;
    indirect: GPUTexture | null;
    ao: GPUTexture | null;
    total: GPUTexture | null;
  } | null;

  /**
   * Approximate engine-owned GPU memory broken down by algorithm category,
   * texture format, and buffer-usage class. Numbers are estimates:
   *
   *   - WebGPU does not expose actual driver-allocated size (alignment,
   *     padding, mip-chain rounding all live inside the implementation).
   *     We infer texture bytes as `width × height × bytesPerTexel(format)`
   *     and buffer bytes as the requested `size`.
   *   - Sub-byte / block-compressed formats (BC*, ETC*, ASTC*) are reported
   *     as their uncompressed equivalent for safety; if a backend allocates
   *     a BC texture it should bias the report up, never down.
   *   - The split by `byCategory` follows the per-algorithm sub-structs in
   *     each backend's FrameResources (e.g. `common`, `restirDI`,
   *     `restirGI`, `svgf`, `gtao`, `ddgi`, `ppg`, `neural`). Backends
   *     without an algorithm report `0` for that key (or omit it).
   *
   * Returns `null` when the engine hasn't allocated any frame resources
   * yet (pre-init, between dispose+recreate). Callers MUST tolerate a
   * `null` return; budget gauges typically render a "—" tile in that case.
   *
   * Hook into {@link FrameStats.gpuMemoryBytes} for streaming consumption
   * during the render loop — pulling once at engine creation is fine for
   * a one-shot budget audit.
   */
  estimatedGpuMemoryBytes?(): GpuMemoryBreakdown | null;
}

/**
 * Per-frame GPU-memory budget breakdown surfaced via
 * {@link EngineDebugSurface.estimatedGpuMemoryBytes}. All values are bytes.
 *
 * Invariant: `total === sum(byCategory)`. The two secondary tables
 * (`byTextureFormat`, `byBufferUsage`) are independent decompositions of
 * the same total — they sum to `total` when the engine has at least one
 * resource of each type, otherwise they sum to `total − unaccounted` (e.g.
 * samplers are not counted in either secondary table because WebGPU does
 * not expose their footprint). Consumers MUST NOT cross-multiply between
 * the three views.
 */
export interface GpuMemoryBreakdown {
  /** Sum of all engine-owned texture + buffer bytes. */
  readonly total: number;
  /** Per-algorithm bytes (keyed by FrameResources sub-struct name). */
  readonly byCategory: Readonly<Record<string, number>>;
  /** Texture bytes split by `GPUTextureFormat` literal. */
  readonly byTextureFormat: Readonly<Record<string, number>>;
  /**
   * Buffer bytes split by usage class (`'storage' | 'uniform' | 'vertex'
   * | 'index' | 'other'`). A buffer with multiple usage bits is attributed
   * to its dominant declared purpose: STORAGE > UNIFORM > VERTEX > INDEX >
   * other (the order chosen so the visible budget reflects path-tracer
   * allocator pressure rather than incidental copy bits).
   */
  readonly byBufferUsage: Readonly<Record<string, number>>;
}
