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
 */

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
   * Ping-pong pair with {@link varianceBufferAux} when the atrous-variance path runs
   * `welfordTemporalMain` each frame.
   */
  varianceBuffer: GPUTexture;
  /** Second Welford ping-pong half (atrous-variance path only). */
  varianceBufferAux: GPUTexture;
  /** Atrous-variance estimation output (.r = scalar variance, .g = frame tag). */
  atrousVarianceEstimateTexture: GPUTexture;
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
   * Sprint 15 — Half-resolution GTAO occlusion factor (rgba16float). Written by
   * `gtaoMain`; consumed by `gtaoUpsampleMain` to reconstruct full-res AO.
   * E1: bumped from r16float to rgba16float to carry per-channel multi-bounce
   * AO (Jiménez 2016 §5.2 / Eq. 16). The upsample reduces to scalar luminance.
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
   * Item 24 — visible-point diffuse albedo (rgba16float, full-res). Written by
   * shade alongside hdrIndirectOut. indirectCombine reads this to re-modulate
   * the denoised lighting signal: `output = filtered_lighting × albedo`.
   * Albedo demodulation (Schied 2017 §4.1) keeps high-frequency albedo
   * variation out of the à-trous chain, preventing material-boundary bleed.
   */
  albedoTexture: GPUTexture;

  // ── T2.H1 — Real SVGF ('svgf-real' mode) persistent textures ─────────────
  /**
   * T2.H1 — 1×1 r32uint placeholder for object-ID inputs. Object IDs are
   * not available in the current walkaround pipeline; this placeholder makes
   * both currObjId and prevObjId read as 0, which means the object-id mismatch
   * test (oPrev != objIdCurr → 0 != 0 = false) never rejects reprojection.
   */
  svgfObjIdPlaceholderTexture: GPUTexture;

  /**
   * T2.H1 — Per-pixel history length A (r16uint, full-res).
   * Ping-pong pair with svgfHistoryLengthTextureB.
   * Increments on accepted reprojection; resets to 1 on disocclusion (Eq. 3).
   * Memory per texture at 1080p: 1920×1080×2 ≈ 4 MB.
   */
  svgfHistoryLengthTextureA: GPUTexture;
  /** T2.H1 — Per-pixel history length B (ping-pong pair). */
  svgfHistoryLengthTextureB: GPUTexture;
  /**
   * T2.H1 — Per-pixel first + second luminance moments A (rg32float, full-res).
   * Ping-pong pair with svgfMomentsTextureB.
   * M1 = E[L] (Eq. 4), M2 = E[L²] (Eq. 4). Used by variance pass Eq. 5.
   * Memory per texture at 1080p: 1920×1080×8 ≈ 16 MB.
   */
  svgfMomentsTextureA: GPUTexture;
  /** T2.H1 — Per-pixel moments B (ping-pong pair). */
  svgfMomentsTextureB: GPUTexture;
  /**
   * T2.H1 — Previous-frame EMA radiance A (rgba16float, full-res).
   * Ping-pong pair with svgfPrevRadianceTextureB.
   * Read as prevColor; written as colorOut. Swapped each frame.
   * Memory per texture at 1080p: 1920×1080×8 ≈ 16 MB.
   */
  svgfPrevRadianceTextureA: GPUTexture;
  /** T2.H1 — Previous-frame EMA radiance B (ping-pong pair). */
  svgfPrevRadianceTextureB: GPUTexture;
  /**
   * T2.H1 — SVGF variance output texture (rg32float, full-res). Written by
   * the 7×7 fallback pass; read by the à-trous chain.
   * Memory at 1080p: 1920×1080×8 ≈ 16 MB.
   */
  svgfVarianceTexture: GPUTexture;
  /**
   * T2.H1 — Intermediate variance from moments (rg32float, full-res). Written
   * by svgfVarianceFromMomentsMain; read by svgf7x7FallbackMain.
   */
  svgfVarianceMomentsIntermedTexture: GPUTexture;
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
 */
export interface FrameResourceOptions {
  // Reserved for future options.
}

/**
 * Create all per-frame GPU resources for the pipeline. Called once from
 * `initialize()` after BVH upload and before shader compilation.
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

  // UBO: camera matrices + per-frame params + library-generality tunables
  // (304 bytes — see WALKAROUND_UBO_SIZE_BYTES in uboUpdater.ts and the
  // WalkaroundUBO struct in common.wgsl).
  const uboBuffer = device.createBuffer({
    size: 304,
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

  // Sprint 9 / 10a — Welford ping-pong + atrous-variance estimate map + motion placeholder.
  const varianceBuffer = createVarianceBuffer(device, W, H);
  const varianceBufferAux = createVarianceBuffer(device, W, H);
  const atrousVarianceEstimateTexture = device.createTexture({
    label: 'atrous-variance-estimate',
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

  // Sprint 15 — GTAO textures (half-res input, full-res upsampled output).
  // `aoFullTexture` is initialised to 1.0 by uploading a buffer of f16 ones
  // so the first frame (before gtao has executed) doesn't darken shade to
  // black via an uninitialised AO read.
  const halfW = Math.max(1, Math.floor(W / 2));
  const halfH = Math.max(1, Math.floor(H / 2));
  // E1: rgba16float (was r16float) to carry per-channel multi-bounce AO.
  const aoHalfTexture = device.createTexture({
    label: 'gtao-half',
    size: [halfW, halfH],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  // Was r16float, but WebGPU base spec doesn't allow r16float as a storage
  // texture (requires the optional `texture-formats-tier1` feature, which
  // three.js's WebGPURenderer ignores — its hardcoded GPUFeatureName enum
  // omits it). rgba16float IS base-spec storage-capable; we keep the AO
  // value in .r (other channels unused) so existing texture-binding reads
  // in shade.wgsl (textureLoad(...).r) still work without changes.
  const aoFullTexture = device.createTexture({
    label: 'gtao-full',
    size: [W, H],
    format: 'rgba16float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST,
  });
  // Seed aoFullTexture with 1.0 (f16 1.0 = 0x3C00) so first-frame shade
  // reads return "unoccluded" before gtao writes a real value.
  // rgba16float = 8 bytes/texel; pack 1.0 in .r, zeros in .g/.b/.a.
  {
    const bytesPerTexel = 8;
    const rowBytes = Math.max(256, Math.ceil(W * bytesPerTexel / 256) * 256);
    const buf = new Uint8Array(rowBytes * H);
    for (let y = 0; y < H; y++) {
      const rowOff = y * rowBytes;
      for (let x = 0; x < W; x++) {
        const o = rowOff + x * bytesPerTexel;
        // .r = 1.0 (0x3C00); .g/.b/.a = 0
        buf[o]     = 0x00;
        buf[o + 1] = 0x3C;
        // (remaining 6 bytes stay 0 from Uint8Array init)
      }
    }
    device.queue.writeTexture(
      { texture: aoFullTexture },
      buf,
      { offset: 0, bytesPerRow: rowBytes },
      { width: W, height: H, depthOrArrayLayers: 1 },
    );
  }
  // 32 bytes: GTAOUniforms struct {tanFovHalf, radiusPx, intensity,
  // depthThresh, bilateralDepthSigma, _pad0, _pad1, _pad2}. The bilateral
  // sigma + pads were added per audit B3 so gtaoUpsample can read the
  // host-configurable depth-edge falloff from the same UBO.
  const gtaoUboBuffer = device.createBuffer({
    label: 'gtao-ubo',
    size: 32,
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

  // Item 24 — albedo demodulation (Schied 2017 §4.1). Full-res rgba16float
  // texture that carries the visible-point diffuse albedo written by shade.
  // indirectCombine reads it to re-multiply the denoised lighting signal.
  const albedoTexture = device.createTexture({
    label: 'albedo-demodulation',
    size: [W, H],
    format: 'rgba16float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });

  // ── T2.H1 — 1×1 r32uint zero placeholder for object IDs (svgf-real).
  // Object IDs are not available in this pipeline; reading 0 from both
  // curr and prev means the id-mismatch test never rejects reprojection.
  const svgfObjIdPlaceholderTexture = device.createTexture({
    label: 'svgf-real-objid-placeholder',
    size: [1, 1],
    format: 'r32uint',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: svgfObjIdPlaceholderTexture },
    new Uint32Array([0]),
    { bytesPerRow: 4 },
    [1, 1],
  );

  // ── T2.H1 — Real SVGF persistent textures (always allocated; zero overhead
  // when mode is not 'svgf-real' because they're idle). At 1080p total ≈52 MB.
  // historyLength: 1920×1080×2 ≈  4 MB
  // momentsHistory: 1920×1080×8 ≈ 16 MB
  // prevRadiance: 1920×1080×8 ≈  16 MB
  // varianceTexture: 1920×1080×8 ≈ 16 MB (merged final)
  // varianceMomentsIntermed: 1920×1080×8 ≈ 16 MB
  const svgfHistUsage =
    GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
  // Was r16uint; base-spec WebGPU disallows it as a storage texture (needs
  // texture-formats-tier1). r32uint is base-spec storage-capable. Counter
  // values stay well under u16 max so the wider format is just 2× memory,
  // no behavioural change.
  const svgfHistoryLengthTextureA = device.createTexture({
    label: 'svgf-real-history-length-a',
    size: [W, H], format: 'r32uint', usage: svgfHistUsage,
  });
  const svgfHistoryLengthTextureB = device.createTexture({
    label: 'svgf-real-history-length-b',
    size: [W, H], format: 'r32uint', usage: svgfHistUsage,
  });
  // Initialise both to 0 so the first frame treats all pixels as disoccluded.
  // r32uint = 4 bytes/texel.
  {
    const bpr = Math.max(256, Math.ceil(W * 4 / 256) * 256);
    const zeroBuf = new Uint8Array(bpr * H);
    device.queue.writeTexture({ texture: svgfHistoryLengthTextureA }, zeroBuf, { bytesPerRow: bpr }, { width: W, height: H, depthOrArrayLayers: 1 });
    device.queue.writeTexture({ texture: svgfHistoryLengthTextureB }, zeroBuf, { bytesPerRow: bpr }, { width: W, height: H, depthOrArrayLayers: 1 });
  }
  const svgfMomUsage =
    GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
  const svgfMomentsTextureA = device.createTexture({
    label: 'svgf-real-moments-a',
    size: [W, H], format: 'rg32float', usage: svgfMomUsage,
  });
  const svgfMomentsTextureB = device.createTexture({
    label: 'svgf-real-moments-b',
    size: [W, H], format: 'rg32float', usage: svgfMomUsage,
  });
  const svgfRadUsage =
    GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
  const svgfPrevRadianceTextureA = device.createTexture({
    label: 'svgf-real-prev-radiance-a',
    size: [W, H], format: 'rgba16float', usage: svgfRadUsage,
  });
  const svgfPrevRadianceTextureB = device.createTexture({
    label: 'svgf-real-prev-radiance-b',
    size: [W, H], format: 'rgba16float', usage: svgfRadUsage,
  });
  const svgfVarianceTexture = device.createTexture({
    label: 'svgf-real-variance',
    size: [W, H],
    format: 'rg32float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING,
  });
  const svgfVarianceMomentsIntermedTexture = device.createTexture({
    label: 'svgf-real-variance-moments-intermed',
    size: [W, H],
    format: 'rg32float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING,
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
    atrousVarianceEstimateTexture,
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
    albedoTexture,
    svgfObjIdPlaceholderTexture,
    svgfHistoryLengthTextureA,
    svgfHistoryLengthTextureB,
    svgfMomentsTextureA,
    svgfMomentsTextureB,
    svgfPrevRadianceTextureA,
    svgfPrevRadianceTextureB,
    svgfVarianceTexture,
    svgfVarianceMomentsIntermedTexture,
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
  r.atrousVarianceEstimateTexture.destroy();
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
  r.albedoTexture.destroy();
  r.svgfObjIdPlaceholderTexture.destroy();
  r.svgfHistoryLengthTextureA.destroy();
  r.svgfHistoryLengthTextureB.destroy();
  r.svgfMomentsTextureA.destroy();
  r.svgfMomentsTextureB.destroy();
  r.svgfPrevRadianceTextureA.destroy();
  r.svgfPrevRadianceTextureB.destroy();
  r.svgfVarianceTexture.destroy();
  r.svgfVarianceMomentsIntermedTexture.destroy();
}
