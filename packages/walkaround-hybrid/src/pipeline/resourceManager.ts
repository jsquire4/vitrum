/**
 * Resource manager — GPU buffer + texture creation helpers for the
 * WalkaroundGPUPipeline.
 *
 * `uploadBuffer` is the lowest-level primitive: creates a mappedAtCreation
 * buffer, copies host data in, unmaps. The pipeline class uses it for all
 * static BVH buffers and re-uses it for emitter re-uploads.
 *
 * `createFrameResources` creates all per-frame GPU objects (reservoir
 * buffers, HDR color textures, ping-pong textures, accum textures, UBO,
 * samplers, DDGI placeholders). Returns a typed bundle so the caller can
 * store each handle as a private field.
 *
 * `createPPGBuffers` allocates the three PPG storage buffers (Sprint 11).
 * Gated by `ppgEnabled` option — no allocation when PPG is disabled so
 * the default engine behaviour is unchanged.
 */

import {
  PPG_MAX_SPATIAL_CELLS,
  PPG_CELL_BYTE_STRIDE,
  PPG_LEAF_BYTE_STRIDE,
  PPG_DIRECTIONS,
} from '../ppg/types.js';
import type { PPGBufferOptions } from '../ppg/types.js';

export interface FrameResources {
  reservoirCurrentBuffer: GPUBuffer;
  reservoirPreviousBuffer: GPUBuffer;
  reservoirSpatialBuffer: GPUBuffer;
  hdrColorTexture: GPUTexture;
  gNormalDepthTexture: GPUTexture;
  denoisedPingTexture: GPUTexture;
  denoisedPongTexture: GPUTexture;
  accumTextureA: GPUTexture;
  accumTextureB: GPUTexture;
  placeholderTexture: GPUTexture;
  uboBuffer: GPUBuffer;
  nearestSampler: GPUSampler;
  compositeLinearSampler: GPUSampler;
  ddgiPlaceholderRgba16f: GPUTexture;
  ddgiPlaceholderRg16f: GPUTexture;
  ddgiUboBuffer: GPUBuffer;
  /**
   * Sprint 9 — Per-pixel Welford variance buffer (RG32Float storage texture).
   * Layout per texel: r = mean (running luminance average), g = M2 (sum of
   * squared deltas). Variance = M2 / (n - 1) where n is the sample count
   * passed per-frame as a uniform.
   *
   * Allocated here (low-risk, Sprint 10a SVGF will also use it). NOT written
   * by any dispatch in Sprint 9; Sprint 10a SVGF and the deferred
   * sprint-9-walkaround-integration will wire the write path.
   *
   * See: packages/walkaround-hybrid/src/shaders/common.wgsl.ts — WelfordVariance
   * struct, welfordUpdate, welfordVariance for the matching WGSL definition.
   *
   * @since Sprint 9, 2026-05-09
   */
  varianceBuffer: GPUTexture;

  /**
   * Sprint 11 — PPG (path guiding) buffers. Only present when `ppgEnabled`
   * is true in the options passed to `createFrameResources`. When PPG is
   * disabled (the default), this field is absent — all existing consumers
   * are unaffected.
   *
   * See `createPPGBuffers` for the allocation details and byte-layout rationale.
   *
   * exactOptionalPropertyTypes: the field is entirely absent (not undefined)
   * when PPG is disabled — consumers must use `'ppgBuffers' in res` to check.
   *
   * @since Sprint 11, 2026-05-09
   */
  readonly ppgBuffers?: PPGBuffers;
}

/**
 * GPU buffer bundle allocated by `createPPGBuffers`. All three buffers are
 * needed together to run the PPG sample + update passes.
 *
 * @since Sprint 11, 2026-05-09
 */
export interface PPGBuffers {
  /**
   * Spatial cell buffer — one PPGSpatialCell per active cell.
   *
   * Size: `maxCells × PPG_CELL_BYTE_STRIDE` (32 bytes/cell).
   * Usage: STORAGE | COPY_DST | COPY_SRC
   *   - STORAGE: both ppgSample (read) and ppgUpdate (read) shaders bind it.
   *   - COPY_DST: host uploads initial cell positions on init or scene change.
   *   - COPY_SRC: allows CPU readback for test validation.
   */
  cellBuffer: GPUBuffer;

  /**
   * Directional leaf buffer — one PPGDirectionalLeaf per spatial cell.
   *
   * Size: `maxCells × PPG_LEAF_BYTE_STRIDE` (256 bytes/leaf).
   * Half of each leaf (128 bytes) holds 16 atomic-u32 pairs
   * (radianceSum_fixed + sampleCount per bin); the other 128 bytes are
   * reserved for future split-tracking fields.
   * Usage: STORAGE | COPY_DST | COPY_SRC
   *   - STORAGE: ppgSample (read) and ppgUpdate (read_write/atomic) bind it.
   *   - COPY_DST: cleared to zero on init; updated per-frame by ppgUpdate.
   *   - COPY_SRC: CPU readback for test validation.
   */
  leafBuffer: GPUBuffer;

  /**
   * Per-frame path-completion sample buffer.
   *
   * Receives PPGPathSample records written by the shade pass (one record per
   * completed indirect bounce). Consumed by the ppgUpdate compute pass on
   * the next even frame. Size = `maxCells × 48 bytes` (one sample-record
   * slot per cell — in practice the shade pass fills far fewer than maxCells
   * samples per frame, but headroom is allocated so the buffer never overflows
   * without bounds-checking).
   * Usage: STORAGE | COPY_DST | COPY_SRC
   */
  sampleBuffer: GPUBuffer;

  /** Maximum number of spatial cells this buffer set was allocated for. */
  readonly maxCells: number;
}

/**
 * Allocate the three PPG storage buffers for the walkaround engine.
 *
 * Called from `createFrameResources` when `ppgEnabled` is true, or
 * from `HybridEngine.setPPGEnabled(true)` if PPG is toggled at runtime.
 * (Sprint 11: runtime toggle is a no-op for dispatch; buffers are allocated
 * and available but the update/sample passes are not yet wired.)
 *
 * Buffer sizing:
 *   - cellBuffer:   maxCells × 32 bytes  = 320 KB at default 10K cap.
 *   - leafBuffer:   maxCells × 256 bytes = 2.56 MB at default 10K cap.
 *   - sampleBuffer: maxCells × 48 bytes  = 480 KB at default 10K cap.
 * Total: ~3.36 MB — negligible alongside the main BVH + reservoir buffers.
 *
 * WebGPU minimum buffer size is 4 bytes; all allocations are well above that.
 *
 * @param device   - Live GPUDevice.
 * @param options  - Optional overrides (see PPGBufferOptions).
 * @returns        Three allocated PPGBuffers, zero-initialised by the GPU driver.
 *
 * @since Sprint 11, 2026-05-09
 */
export function createPPGBuffers(device: GPUDevice, options?: PPGBufferOptions): PPGBuffers {
  const maxCells = options?.maxCells ?? PPG_MAX_SPATIAL_CELLS;

  const storageUsage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

  // Each leaf occupies PPG_LEAF_BYTE_STRIDE bytes (256), of which 128 are used
  // for the 16 atomic-u32-pair bins; 128 bytes are reserved.
  // The leaf count equals maxCells (1:1 cell-to-leaf mapping in Sprint 11).
  const leafCount = maxCells;

  // sampleBuffer: one PPGPathSample (48 bytes) per cell as an upper bound.
  const PPG_PATH_SAMPLE_BYTE_STRIDE = 48;

  const cellBuffer = device.createBuffer({
    label: 'ppg-cell-buffer',
    size: Math.max(maxCells * PPG_CELL_BYTE_STRIDE, 16),
    usage: storageUsage,
  });

  const leafBuffer = device.createBuffer({
    label: 'ppg-leaf-buffer',
    size: Math.max(leafCount * PPG_LEAF_BYTE_STRIDE, 16),
    usage: storageUsage,
  });

  const sampleBuffer = device.createBuffer({
    label: 'ppg-sample-buffer',
    size: Math.max(maxCells * PPG_PATH_SAMPLE_BYTE_STRIDE, 16),
    usage: storageUsage,
  });

  // Suppress unused-import lint: PPG_DIRECTIONS is part of the public type
  // contract surfaced through tests; referenced here to keep the import live.
  void PPG_DIRECTIONS;

  return { cellBuffer, leafBuffer, sampleBuffer, maxCells };
}

/**
 * Destroy all buffers in a PPGBuffers bundle. Safe to call from dispose()
 * or when PPG is toggled off. Nullability is not enforced here — callers
 * guard with `if (ppgBuffers)`.
 *
 * @since Sprint 11, 2026-05-09
 */
export function destroyPPGBuffers(buffers: PPGBuffers): void {
  buffers.cellBuffer.destroy();
  buffers.leafBuffer.destroy();
  buffers.sampleBuffer.destroy();
}

/**
 * Upload a CPU-side ArrayBuffer into a GPU storage buffer.
 * Enforces a minimum size of 16 bytes (WebGPU spec requirement).
 */
export function uploadBuffer(device: GPUDevice, data: ArrayBuffer, usage: number): GPUBuffer {
  const size = Math.max(data.byteLength, 16);
  const buf = device.createBuffer({
    size,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data));
  buf.unmap();
  return buf;
}

/**
 * Build the DDGI "placeholder" UBO data — the zero-grid uniform that causes
 * shade.wgsl's `isDDGIWired()` check to return false (dimsX ≤ 1).
 *
 * DDGIGridUniform layout (64 bytes = 16 × f32):
 *   f32[0..2]  origin xyz   — (0,0,0)
 *   f32[3]     spacing      — 24 (matches probeGrid default, irrelevant when wired=false)
 *   u32[4..6]  dims xyz     — (1,1,1) — dimsX=1 gates isDDGIWired() to false
 *   u32[7]     padding      — 0
 *   f32[8..11] irrW,irrH,visW,visH — 1×1 (match 1×1 placeholder textures)
 *   f32[12..15] reserved    — 0
 *
 * This is the canonical, single definition of the placeholder UBO layout.
 * Both `createFrameResources` and `WalkaroundGPUPipeline.setDDGIInputs(null)`
 * call this function rather than duplicating the pack inline.
 *
 * The returned `Float32Array` is freshly allocated each call. For the hot path
 * in `setDDGIInputs(null)`, callers should cache the result — see
 * `WalkaroundGPUPipeline._ddgiPlaceholderUBO`.
 */
export function buildDDGIPlaceholderUBO(): Float32Array {
  const placeholder = new Float32Array(16);
  placeholder[3] = 24;                              // spacing (default probe spacing)
  new Uint32Array(placeholder.buffer)[4] = 1;       // dimsX — isDDGIWired() checks dimsX > 1u
  new Uint32Array(placeholder.buffer)[5] = 1;       // dimsY
  new Uint32Array(placeholder.buffer)[6] = 1;       // dimsZ
  placeholder[8]  = 1;                              // irrW (matches 1×1 placeholder texture)
  placeholder[9]  = 1;                              // irrH
  placeholder[10] = 1;                              // visW
  placeholder[11] = 1;                              // visH
  return placeholder;
}

/**
 * Create a per-pixel Welford variance buffer (RG32Float storage texture).
 *
 * Sprint 9 — Decision 13 (locked 2026-05-09): the WelfordVariance struct
 * in common.wgsl.ts pins the layout to RG32Float (r=mean, g=M2). This helper
 * creates the matching GPU texture so the layout is enforced in one place.
 *
 * Usage flags:
 *   - STORAGE_BINDING: the temporal accumulator (Sprint 10a) will write to it
 *     via a compute pass; the sample-budget shader reads from it.
 *   - TEXTURE_BINDING: Sprint 10a's SVGF spatial filter reads variance as a
 *     sampled texture.
 *   - COPY_SRC: allows CPU readback for test validation (caustic harness).
 *
 * @param device - Live GPUDevice.
 * @param w      - Render-surface pixel width.
 * @param h      - Render-surface pixel height.
 * @returns      A GPUTexture ready to be bound as a storage/sampled texture.
 */
export function createVarianceBuffer(device: GPUDevice, w: number, h: number): GPUTexture {
  return device.createTexture({
    label: 'welford-variance',
    size: [w, h],
    format: 'rg32float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });
}

/**
 * Options for `createFrameResources`.
 *
 * @since Sprint 11, 2026-05-09 (ppgEnabled added)
 */
export interface FrameResourceOptions {
  /**
   * When true, allocates the three PPG storage buffers (cellBuffer, leafBuffer,
   * sampleBuffer). Defaults to false — existing callers are unaffected.
   *
   * PPG buffers add ~3.36 MB of GPU memory at the default 10K cell cap.
   * The Sprint 11 default is disabled; only opt-in consumers (tests or
   * hosts that explicitly set `ppgEnabled: true` in HybridEngineOptions)
   * will allocate them.
   *
   * @since Sprint 11
   */
  ppgEnabled?: boolean;
}

/**
 * Create all per-frame GPU resources for the pipeline. Called once from
 * `initialize()` after BVH upload and before shader compilation.
 *
 * Pass `{ ppgEnabled: true }` to also allocate PPG buffers (Sprint 11).
 * The default is `ppgEnabled: false` — no behavioural change for existing callers.
 */
export function createFrameResources(
  device: GPUDevice,
  W: number,
  H: number,
  options?: FrameResourceOptions,
): FrameResources {
  // Reservoir DI: 16 bytes/pixel (4 × u32)
  const RESERVOIR_STRIDE = 16;
  const totalReservoirBytes = Math.max(W * H * RESERVOIR_STRIDE, 256);

  const reservoirCurrentBuffer = device.createBuffer({
    size: totalReservoirBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const reservoirPreviousBuffer = device.createBuffer({
    size: totalReservoirBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const reservoirSpatialBuffer = device.createBuffer({
    size: totalReservoirBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // HDR color output (rgba16float — written by shade, read by atrous).
  // COPY_SRC enables GPU pixel readback for the caustic validation harness.
  const hdrColorTexture = device.createTexture({
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  // G-buffer (normal + depth) — written by shade, read by atrous denoiser.
  const gNormalDepthTexture = device.createTexture({
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  // Ping-pong denoised textures.
  // COPY_SRC enables GPU pixel readback for the caustic validation harness.
  const denoisedPingTexture = device.createTexture({
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const denoisedPongTexture = device.createTexture({
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  // Temporal accumulator ping-pong (rgba16float). Read prev / write
  // current within a single dispatch — must be separate textures.
  const accumTextureA = device.createTexture({
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const accumTextureB = device.createTexture({
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  // 1×1 placeholder texture for G-buffer bind group slots.
  const placeholderTexture = device.createTexture({
    size: [1, 1],
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  // Fill placeholder with a valid forward-facing normal so the atrous denoiser
  // does not produce NaN. The atrous shader decodes normal as: n = raw * 2 - 1.
  // For a forward-facing normal (0,0,1): raw = (0.5, 0.5, 1.0, 0.0).
  // Using (0,0,0) for the zero-depth (sky) placeholder causes dot(n,n) = 3 →
  // pow(3, sigmaN=128) → Inf, and Inf/Inf = NaN propagation through the denoiser.
  const placeholderData = new Float32Array([0.5, 0.5, 1.0, 0.0]); // encodes normal=(0,0,1), depth=0
  device.queue.writeTexture({ texture: placeholderTexture }, placeholderData, { bytesPerRow: 16 }, [1, 1]);

  // UBO: camera matrices + per-frame params (256 bytes).
  const uboBuffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const nearestSampler = device.createSampler({
    magFilter: 'nearest',
    minFilter: 'nearest',
  });
  const compositeLinearSampler = device.createSampler({
    magFilter: 'nearest',
    minFilter: 'nearest',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  // DDGI placeholder textures + UBO. The shade pipeline's 4th bind group
  // always binds these, so the pipeline validates even when no real DDGI
  // atlas has been supplied via setDDGIInputs().
  const ddgiPlaceholderRgba16f = device.createTexture({
    label: 'ddgi-placeholder-irr',
    size: [1, 1],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  // Visibility placeholder must match the live atlas format
  // (probeUpdatePass creates rgba16float). Format consistency is the
  // safer invariant — bug fix isolated during 2026-05-07 sweep (B5).
  const ddgiPlaceholderRg16f = device.createTexture({
    label: 'ddgi-placeholder-vis',
    size: [1, 1],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const ddgiUboBuffer = device.createBuffer({
    label: 'ddgi-ubo',
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // Default DDGI uniform — origin (0,0,0), spacing 24, dims (1,1,1),
  // atlas dims 1×1. shade.wgsl gates DDGI consumption on isDDGIWired()
  // which checks dimsX > 1u; the placeholder writes dimsX=1 so the gate
  // returns false and Lo_ddgi=0 until setDDGIInputs() supplies real
  // grid params from HybridLayeredStage.
  device.queue.writeBuffer(ddgiUboBuffer, 0, buildDDGIPlaceholderUBO().buffer);

  // Sprint 9 — Per-pixel Welford variance buffer (RG32Float, r=mean g=M2).
  // Allocated here; no compute pass writes to it in Sprint 9. Sprint 10a SVGF
  // and the deferred sprint-9-walkaround-integration will wire the write path.
  const varianceBuffer = createVarianceBuffer(device, W, H);

  // Sprint 11 — PPG buffers (opt-in via ppgEnabled, default: disabled).
  // No behavioural change for existing callers when ppgEnabled is false/unset.
  // Use a conditional spread so the ppgBuffers key is absent (not undefined)
  // which satisfies exactOptionalPropertyTypes.
  const ppgExt: Pick<FrameResources, 'ppgBuffers'> =
    options?.ppgEnabled === true
      ? { ppgBuffers: createPPGBuffers(device) }
      : {};

  return {
    reservoirCurrentBuffer,
    reservoirPreviousBuffer,
    reservoirSpatialBuffer,
    hdrColorTexture,
    gNormalDepthTexture,
    denoisedPingTexture,
    denoisedPongTexture,
    accumTextureA,
    accumTextureB,
    placeholderTexture,
    uboBuffer,
    nearestSampler,
    compositeLinearSampler,
    ddgiPlaceholderRgba16f,
    ddgiPlaceholderRg16f,
    ddgiUboBuffer,
    varianceBuffer,
    ...ppgExt,
  };
}

/**
 * Destroy all resources returned by `createFrameResources`. Safe to call
 * in dispose(); callers must also destroy the static BVH buffers separately.
 */
export function destroyFrameResources(r: FrameResources): void {
  r.reservoirCurrentBuffer.destroy();
  r.reservoirPreviousBuffer.destroy();
  r.reservoirSpatialBuffer.destroy();
  r.hdrColorTexture.destroy();
  r.gNormalDepthTexture.destroy();
  r.denoisedPingTexture.destroy();
  r.denoisedPongTexture.destroy();
  r.accumTextureA.destroy();
  r.accumTextureB.destroy();
  r.placeholderTexture.destroy();
  r.uboBuffer.destroy();
  r.ddgiPlaceholderRgba16f.destroy();
  r.ddgiPlaceholderRg16f.destroy();
  r.ddgiUboBuffer.destroy();
  r.varianceBuffer.destroy();  // Sprint 9 — Welford variance buffer
  if (r.ppgBuffers) {          // Sprint 11 — PPG buffers (opt-in, may be absent)
    destroyPPGBuffers(r.ppgBuffers);
  }
}
