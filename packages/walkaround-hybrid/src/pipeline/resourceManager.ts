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
 * `createPPGBuffers` allocates the PPG storage buffers (Sprint 11).
 * Gated by `ppgEnabled` option — no allocation when PPG is disabled so
 * the default engine behaviour is unchanged.
 */

import {
  PPG_MAX_SPATIAL_CELLS,
  PPG_CELL_BYTE_STRIDE,
  PPG_LEAF_BYTE_STRIDE,
  PPG_KD_MAX_NODES,
  PPG_KD_NODE_BYTE_STRIDE,
} from '../ppg/types.js';
import type { PPGBufferOptions } from '../ppg/types.js';
import { encodePpgKdDisabledRoot, buildPpgKdTreeGpuBytes } from '../ppg/buildPpgKdTree.js';

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
  compositeSampler: GPUSampler;
  ddgiPlaceholderRgba16f: GPUTexture;
  ddgiPlaceholderRg16f: GPUTexture;
  ddgiUboBuffer: GPUBuffer;
  /**
   * Sprint 9 — Per-pixel Welford variance buffer (RG32Float storage texture).
   * Ping-pong pair with {@link varianceBufferAux} when the SVGF path runs
   * `welfordTemporalMain` each frame.
   */
  varianceBuffer: GPUTexture;
  /** Second Welford ping-pong half (SVGF path only). */
  varianceBufferAux: GPUTexture;
  /** SVGF variance-estimation output (.r = scalar variance, .g = frame tag). */
  svgfVarianceEstimateTexture: GPUTexture;
  /** Screen-space motion (RG32F); zeros until a motion-vector pass exists. */
  motionVectorTexture: GPUTexture;
  /**
   * Sprint 9 — Per-pixel sample tier (r32uint, 1 / 2 / 4). Written by the
   * sample-budget pass each frame; available for downstream consumption by
   * shade or RIS. Always allocated — adaptive sampling is part of the
   * standard pipeline as of Sprint 9 wire-in.
   */
  tierTexture: GPUTexture;
  /**
   * Sprint 9 — Final resolved radiance (rgba16float). Written by the
   * resolve pass (between temporalAccum and composite); read by composite.
   * Always allocated — sparse-shade resolve is part of the standard pipeline
   * as of Sprint 9 wire-in.
   */
  resolvedTexture: GPUTexture;

  /**
   * Sprint 15 — Half-resolution GTAO occlusion factor (r16float). Written by
   * `gtaoMain`; consumed by `gtaoUpsampleMain` to reconstruct full-res AO.
   */
  aoHalfTexture: GPUTexture;
  /**
   * Sprint 15 — Full-resolution GTAO occlusion factor (r16float). Written by
   * `gtaoUpsampleMain`; sampled by `shade.wgsl` to modulate the diffuse
   * indirect / direct terms. 1-frame lagged from current shade (AO computes
   * from current frame's gNormalDepth but shade reads the *previous* frame's
   * AO texture for binding-order simplicity).
   */
  aoFullTexture: GPUTexture;
  /** Sprint 15 — GTAO uniforms (16 bytes: tanFovHalf, radiusPx, intensity, depthThresh). */
  gtaoUboBuffer: GPUBuffer;

  /**
   * Sprint 16 — half-res ReSTIR-GI reservoir buffer.
   * Layout: RESERVOIR_GI_STRIDE = 20 u32 (80 bytes) per pixel.
   * Size: (W/2) × (H/2) × 80 bytes. At 2688×1344 → ~58 MB.
   * Written by `risGiMain`; read by temporal/spatial passes and shade.
   */
  reservoirGiCurrentBuffer: GPUBuffer;
  /**
   * Sprint 17 — previous-frame GI reservoir (temporal reuse input).
   * Updated at end-of-frame via copyBufferToBuffer(current → previous).
   */
  reservoirGiPreviousBuffer: GPUBuffer;
  /**
   * Sprint 17 — spatial-reuse scratch GI reservoir. Ping-ponged with
   * `reservoirGiCurrentBuffer` across the two spatial passes.
   */
  reservoirGiSpatialBuffer: GPUBuffer;

  /**
   * Sprint 18 — separate indirect-channel HDR target (rgba16float, full-res).
   * Written by shade as `Lo_indirect * ao`; read by the indirect-combine
   * pass which bilateral-blurs it with broader sigmas and sums into
   * `combinedDenoisedTexture`.
   */
  hdrIndirectTexture: GPUTexture;
  /**
   * Sprint 18 — final per-channel combined output. Written by the
   * indirect-combine pass; consumed by temporalAccum in place of the
   * raw direct-denoiser output.
   */
  combinedDenoisedTexture: GPUTexture;
  /**
   * Sprint 18 follow-up — total (direct + indirect) HDR signal, written
   * by shade alongside hdrColor and hdrIndirect.  Used as the welford
   * input so the per-pixel variance estimate (and the sample-budget
   * tier derived from it) reflects the full radiance, not just the
   * direct channel.  SVGF variance / atrous still read hdrColorTexture
   * (direct-only) so the denoiser sees the channel it's tuned for.
   */
  hdrTotalTexture: GPUTexture;
  /**
   * Sprint 18 — indirect-channel à-trous ping-pong pair.  Four iterations
   * with widening step (1, 2, 4, 8) on hdrIndirectTexture produce a smooth
   * indirect signal that the indirect-combine pass sums with the direct
   * SVGF output.  Broader sigmas than the direct chain — indirect is
   * already temporally smoothed by ReSTIR-GI and tolerates wider blurs.
   */
  indirectDenoisedPingTexture: GPUTexture;
  indirectDenoisedPongTexture: GPUTexture;
  /**
   * Sprint 18 follow-up — indirect-channel temporal-accumulator ping-pong.
   * Inserted between shade and atrous-indirect.  The pre-atrous accumulator
   * applies a TCBB (temporal color bounding box) clip on the history using
   * the current frame's 3×3 neighborhood [min, max] to reject fireflies +
   * anti-ghost, then α-blends with the raw current frame.  atrous-indirect
   * reads the accumulator output (much more temporally coherent than the
   * raw indirect signal), so its spatial smoothing actually converges
   * instead of just reshuffling new noise every frame.
   */
  indirectAccumPingTexture: GPUTexture;
  indirectAccumPongTexture: GPUTexture;

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
 * GPU buffer bundle allocated by `createPPGBuffers`. All buffers are
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

  /**
   * Atomic slot counter for training samples (cleared each frame before shade).
   * sizeof = 16 bytes for WebGPU min buffer alignment.
   */
  sampleHeadBuffer: GPUBuffer;

  /**
   * kd-tree over spatial cell centroids (16-byte nodes, see `buildPpgKdTree.ts`).
   * Initialised with a sentinel that forces brute-force lookup in WGSL until
   * the host uploads `buildPpgKdTreeGpuBytes(...)`.
   */
  kdBuffer: GPUBuffer;

  /** Maximum number of spatial cells this buffer set was allocated for. */
  readonly maxCells: number;
}

/**
 * Allocate PPG storage buffers for the walkaround engine.
 *
 * Called from `createFrameResources` when `ppgEnabled` is true, or
 * from `HybridEngine.setPPGEnabled(true)` if PPG is toggled at runtime.
 *
 * Buffer sizing:
 *   - cellBuffer:   maxCells × 32 bytes  = 320 KB at default 10K cap.
 *   - leafBuffer:   maxCells × 256 bytes = 2.56 MB at default 10K cap.
 *   - sampleBuffer: maxCells × 48 bytes  = 480 KB at default 10K cap.
 *   - sampleHeadBuffer: 16 bytes (atomic counter + alignment).
 *   - kdBuffer:     ≤ (2×maxCells+8) × 16 bytes (~321 KB at 10K cap).
 * Total: ~3.7 MB — negligible alongside the main BVH + reservoir buffers.
 *
 * WebGPU minimum buffer size is 4 bytes; all allocations are well above that.
 *
 * @param device   - Live GPUDevice.
 * @param options  - Optional overrides (see PPGBufferOptions).
 * @returns        Allocated PPGBuffers (cells, leaves, samples, sample head, kd-tree).
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

  const kdByteSize = Math.max(PPG_KD_MAX_NODES * PPG_KD_NODE_BYTE_STRIDE, 16);
  const kdBuffer = device.createBuffer({
    label: 'ppg-kd-buffer',
    size: kdByteSize,
    usage: storageUsage,
  });
  const sampleHeadBuffer = device.createBuffer({
    label: 'ppg-sample-head',
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(sampleHeadBuffer, 0, new Uint32Array(4));
  const kdSentinel = encodePpgKdDisabledRoot();
  device.queue.writeBuffer(
    kdBuffer,
    0,
    kdSentinel.buffer as ArrayBuffer,
    kdSentinel.byteOffset,
    kdSentinel.byteLength,
  );

  return { cellBuffer, leafBuffer, sampleBuffer, sampleHeadBuffer, kdBuffer, maxCells };
}

export function writePpgKdTree(
  queue: GPUQueue,
  kdBuffer: GPUBuffer,
  cells: ReadonlyArray<{ readonly position: readonly [number, number, number] }>,
  activeCellCount: number,
): void {
  const bytes = buildPpgKdTreeGpuBytes(cells, activeCellCount);
  queue.writeBuffer(
    kdBuffer,
    0,
    bytes.buffer as ArrayBuffer,
    bytes.byteOffset,
    bytes.byteLength,
  );
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
  buffers.sampleHeadBuffer.destroy();
  buffers.kdBuffer.destroy();
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
 * Pack live DDGI grid params into the canonical 64-byte UBO layout expected
 * by shade.wgsl. Single source of truth; HybridEngine.renderFrame() used to
 * inline this packing, which drifted vs. buildDDGIPlaceholderUBO above.
 */
export function packDDGIGridParams(p: {
  origin: { x: number; y: number; z: number };
  spacing: number;
  dims: { x: number; y: number; z: number };
  irradianceAtlasW: number;
  irradianceAtlasH: number;
  visibilityAtlasW: number;
  visibilityAtlasH: number;
}): ArrayBuffer {
  const buf = new ArrayBuffer(64);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  f32[0] = p.origin.x;
  f32[1] = p.origin.y;
  f32[2] = p.origin.z;
  f32[3] = p.spacing;
  u32[4] = p.dims.x;
  u32[5] = p.dims.y;
  u32[6] = p.dims.z;
  u32[7] = 0;
  f32[8]  = p.irradianceAtlasW;
  f32[9]  = p.irradianceAtlasH;
  f32[10] = p.visibilityAtlasW;
  f32[11] = p.visibilityAtlasH;
  return buf;
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
   * When true, allocates PPG storage buffers (cellBuffer, leafBuffer,
   * sampleBuffer, sampleHeadBuffer, kdBuffer). Defaults to false — existing callers are unaffected.
   *
   * PPG buffers add about 3.7 MB of GPU memory at the default 10K cell cap.
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
  // Sprint 18 — separate indirect-channel HDR target.
  const hdrIndirectTexture = device.createTexture({
    label: 'hdrIndirect',
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  // Sprint 18 — combined output of indirect-combine pass; fed to temporalAccum.
  const combinedDenoisedTexture = device.createTexture({
    label: 'combinedDenoised',
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  // Sprint 18 follow-up — total radiance fed to welford so the tier
  // classification sees the full signal (direct + indirect).
  const hdrTotalTexture = device.createTexture({
    label: 'hdrTotal',
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  // Sprint 18 — indirect-channel à-trous ping-pong pair.
  const indirectDenoisedPingTexture = device.createTexture({
    label: 'indirectDenoisedPing',
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const indirectDenoisedPongTexture = device.createTexture({
    label: 'indirectDenoisedPong',
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  // Sprint 18 follow-up — indirect temporal-accumulator ping-pong.
  const indirectAccumPingTexture = device.createTexture({
    label: 'indirectAccumPing',
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const indirectAccumPongTexture = device.createTexture({
    label: 'indirectAccumPong',
    size: [W, H],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
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
  const compositeSampler = device.createSampler({
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

  // Sprint 9 / 10a — Welford ping-pong + SVGF variance map + motion placeholder.
  const varianceBuffer = createVarianceBuffer(device, W, H);
  const varianceBufferAux = createVarianceBuffer(device, W, H);
  const svgfVarianceEstimateTexture = device.createTexture({
    label: 'svgf-variance-estimate',
    size: [W, H],
    format: 'rg32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const motionVectorTexture = device.createTexture({
    label: 'motion-vectors-zero',
    size: [W, H],
    format: 'rg32float',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  });
  const rowBytes = 8 * W;
  const bytesPerRow = Math.max(256, Math.ceil(rowBytes / 256) * 256);
  const motionZero = new Uint8Array(bytesPerRow * H);
  device.queue.writeTexture(
    { texture: motionVectorTexture },
    motionZero,
    { offset: 0, bytesPerRow },
    { width: W, height: H, depthOrArrayLayers: 1 },
  );

  // Sprint 9 — Adaptive sampling textures (tier + resolved). Both are
  // unconditional now: sample-budget + resolve are standard pipeline passes.
  const tierTexture = device.createTexture({
    label: 'sample-tier',
    size: [W, H],
    format: 'r32uint',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const resolvedTexture = device.createTexture({
    label: 'resolved-radiance',
    size: [W, H],
    format: 'rgba16float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });

  // Sprint 11 — PPG buffers (opt-in via ppgEnabled, default: disabled).
  // No behavioural change for existing callers when ppgEnabled is false/unset.
  // Use a conditional spread so the ppgBuffers key is absent (not undefined)
  // which satisfies exactOptionalPropertyTypes.
  const ppgExt: Pick<FrameResources, 'ppgBuffers'> =
    options?.ppgEnabled === true
      ? { ppgBuffers: createPPGBuffers(device) }
      : {};

  // Sprint 15 — GTAO textures (half-res input, full-res upsampled output).
  // `aoFullTexture` is initialised to 1.0 by uploading a buffer of f16 ones
  // so the first frame (before gtao has executed) doesn't darken shade to
  // black via an uninitialised AO read.
  const halfW = Math.max(1, Math.floor(W / 2));
  const halfH = Math.max(1, Math.floor(H / 2));
  const aoHalfTexture = device.createTexture({
    label: 'gtao-half',
    size: [halfW, halfH],
    format: 'r16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const aoFullTexture = device.createTexture({
    label: 'gtao-full',
    size: [W, H],
    format: 'r16float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST,
  });
  // Seed aoFullTexture with 1.0 (f16 1.0 = 0x3C00) so first-frame shade
  // reads return "unoccluded" before gtao writes a real value.
  {
    const onesRowBytes = Math.max(256, Math.ceil(W * 2 / 256) * 256);
    const onesBuf = new Uint8Array(onesRowBytes * H);
    // f16 1.0 = 0x3C00. r16float stores 2 bytes/texel.
    for (let y = 0; y < H; y++) {
      const rowOff = y * onesRowBytes;
      for (let x = 0; x < W; x++) {
        const o = rowOff + x * 2;
        onesBuf[o] = 0x00;
        onesBuf[o + 1] = 0x3C;
      }
    }
    device.queue.writeTexture(
      { texture: aoFullTexture },
      onesBuf,
      { offset: 0, bytesPerRow: onesRowBytes },
      { width: W, height: H, depthOrArrayLayers: 1 },
    );
  }
  const gtaoUboBuffer = device.createBuffer({
    label: 'gtao-ubo',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Sprint 16 — GI reservoir buffer (half-res). RESERVOIR_GI_STRIDE = 20 u32
  // (80 bytes) per pixel. Aligned to 256 to match WebGPU storage binding rules.
  const reservoirGiSize = halfW * halfH * 80;
  const reservoirGiCurrentBuffer = device.createBuffer({
    label: 'reservoir-gi-current',
    size: Math.max(256, reservoirGiSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  // Sprint 17 — temporal + spatial reservoir buffers. Both same size as
  // current; previous is read-only during temporal reuse, copied at end of
  // frame; spatial is scratch for the two ping-ponged spatial passes.
  const reservoirGiPreviousBuffer = device.createBuffer({
    label: 'reservoir-gi-previous',
    size: Math.max(256, reservoirGiSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const reservoirGiSpatialBuffer = device.createBuffer({
    label: 'reservoir-gi-spatial',
    size: Math.max(256, reservoirGiSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

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
    compositeSampler,
    ddgiPlaceholderRgba16f,
    ddgiPlaceholderRg16f,
    ddgiUboBuffer,
    varianceBuffer,
    varianceBufferAux,
    svgfVarianceEstimateTexture,
    motionVectorTexture,
    tierTexture,
    resolvedTexture,
    aoHalfTexture,
    aoFullTexture,
    gtaoUboBuffer,
    reservoirGiCurrentBuffer,
    reservoirGiPreviousBuffer,
    reservoirGiSpatialBuffer,
    hdrIndirectTexture,
    combinedDenoisedTexture,
    hdrTotalTexture,
    indirectDenoisedPingTexture,
    indirectDenoisedPongTexture,
    indirectAccumPingTexture,
    indirectAccumPongTexture,
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
  r.varianceBuffer.destroy();
  r.varianceBufferAux.destroy();
  r.svgfVarianceEstimateTexture.destroy();
  r.motionVectorTexture.destroy();
  r.tierTexture.destroy();
  r.resolvedTexture.destroy();
  r.aoHalfTexture.destroy();
  r.aoFullTexture.destroy();
  r.gtaoUboBuffer.destroy();
  r.reservoirGiCurrentBuffer.destroy();
  r.reservoirGiPreviousBuffer.destroy();
  r.reservoirGiSpatialBuffer.destroy();
  r.hdrIndirectTexture.destroy();
  r.combinedDenoisedTexture.destroy();
  r.hdrTotalTexture.destroy();
  r.indirectDenoisedPingTexture.destroy();
  r.indirectDenoisedPongTexture.destroy();
  r.indirectAccumPingTexture.destroy();
  r.indirectAccumPongTexture.destroy();
  if (r.ppgBuffers) {          // Sprint 11 — PPG buffers (opt-in, may be absent)
    destroyPPGBuffers(r.ppgBuffers);
  }
}
