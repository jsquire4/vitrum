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
 * W1-R2 (2026-05-17) — the formerly 41-field flat `FrameResources` god-struct
 * is now split into 8 per-algorithm sub-structs (`common`, `restirDI`,
 * `restirGI`, `ddgi`, `gtao`, `svgf`, `ppg`, `neural`). Sub-struct boundaries
 * map to the per-algorithm passes in `pipeline/passes/*`. The `ppg` and
 * `neural` sub-structs were empty placeholders at the W1-R2 landing time;
 * both have since been populated — `ppg` by the W9 finish (PPG GPU dTree
 * traversal + per-pixel position binding), `neural` — InferenceGraph owns its
 * tensor buffers; NeuralFrameResources is an empty placeholder. See
 * plan/premium-grade-refactor-20260517.md §W1-R2
 * and complexity-sweep-20260517 findings A3 + B6.
 */

import { createCommonFrameResources } from './frameResources/createCommonFrameResources.js';

// `buildDDGIPlaceholderUBO` is re-exported here because two pipeline-side
// consumers (frameResources/createDdgiFrameResources.ts, DDGIBindingState.ts)
// import it via this module. `packDDGIGridParams` / `DDGIGridParamsInput` were
// also re-exported but had zero importers-via-this-path — all consumers pull
// them from the canonical `../ddgi/ddgiGridUbo.js` directly — so they were
// dropped (T10-DDGI dead-re-export cleanup).
export { buildDDGIPlaceholderUBO } from '../ddgi/ddgiGridUbo.js';
import { createDdgiFrameResources } from './frameResources/createDdgiFrameResources.js';
import { createRestirDIFrameResources } from './frameResources/createRestirDIFrameResources.js';
import { createRestirGIFrameResources } from './frameResources/createRestirGIFrameResources.js';
import { createGtaoFrameResources } from './frameResources/createGtaoFrameResources.js';
import { createNeuralFrameResources } from './frameResources/createNeuralFrameResources.js';
import { createSvgfFrameResources } from './frameResources/createSvgfFrameResources.js';

// ─── Per-algorithm sub-struct interfaces ─────────────────────────────────────

/**
 * Cross-cutting GPU resources shared by every pass — primary HDR targets,
 * temporal accumulator ping-pong, UBO, samplers, motion vectors, the
 * sample-tier + resolved-radiance textures used by sample-budget /
 * resolve, Sprint-18 indirect + albedo + variance scaffolding, and the
 * 1×1 placeholder used by the G-buffer bind group slots.
 *
 * Anything that belongs to a single algorithm (DDGI atlas placeholders,
 * GTAO half-/full-res output, SVGF history textures, ReSTIR reservoir
 * buffers) lives in its dedicated sub-struct below.
 */
export interface CommonFrameResources {
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
  /** Screen-space motion (RG32F), written each frame by MotionVectorsPass. */
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
   * Sprint 18 follow-up — total (direct + indirect) HDR signal, written
   * by shade alongside hdrColor and hdrIndirect.  Used as the welford
   * input so the per-pixel variance estimate (and the sample-budget
   * tier derived from it) reflects the full radiance, not just the
   * direct channel.  SVGF variance / atrous still read hdrColorTexture
   * (direct-only) so the denoiser sees the channel it's tuned for.
   */
  hdrTotalTexture: GPUTexture;
  /**
   * Item 24 — visible-point diffuse albedo (rgba16float, full-res). Written by
   * shade alongside hdrIndirectOut. indirectCombine reads this to re-modulate
   * the denoised lighting signal: `output = filtered_lighting × albedo`.
   * Albedo demodulation (Schied 2017 §4.1) keeps high-frequency albedo
   * variation out of the à-trous chain, preventing material-boundary bleed.
   */
  albedoTexture: GPUTexture;
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
   * Sprint 9 — Per-pixel Welford variance buffer (RG32Float storage texture).
   * Ping-pong pair with {@link varianceBufferAux} when the atrous-variance path runs
   * `welfordTemporalMain` each frame.
   */
  varianceBuffer: GPUTexture;
  /** Second Welford ping-pong half (atrous-variance path only). */
  varianceBufferAux: GPUTexture;
  /** Atrous-variance estimation output (.r = scalar variance, .g = frame tag). */
  atrousVarianceEstimateTexture: GPUTexture;
}

/** ReSTIR direct-illumination reservoir buffers (current / previous / spatial). */
export interface RestirDIFrameResources {
  reservoirCurrentBuffer: GPUBuffer;
  reservoirPreviousBuffer: GPUBuffer;
  reservoirSpatialBuffer: GPUBuffer;
}

/** ReSTIR global-illumination reservoir buffers (current / previous / spatial). */
export interface RestirGIFrameResources {
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
}

/** DDGI 1×1 atlas placeholders + the DDGI uniform buffer (gate UBO). */
export interface DDGIFrameResources {
  ddgiPlaceholderRgba16f: GPUTexture;
  ddgiPlaceholderRg16f: GPUTexture;
  ddgiUboBuffer: GPUBuffer;
}

/** GTAO half- and full-resolution AO textures + GTAO uniform buffer. */
export interface GTAOFrameResources {
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
}

/** SVGF ('svgf-real' mode) persistent textures — history, moments, prev-rad, variance. */
export interface SVGFFrameResources {
  /**
   * T2.H1 — 1×1 r32uint placeholder for object-ID inputs. Object IDs are
   * not available in the current walkaround pipeline; this placeholder makes
   * both currObjId and prevObjId read as 0, which means the object-id mismatch
   * test (oPrev != objIdCurr → 0 != 0 = false) never rejects reprojection.
   */
  svgfObjIdPlaceholderTexture: GPUTexture;
  /** Conservative prev-object-id placeholder (value 1). When bound as prevObjId
   *  against currObjId=0, reprojection rejects history instead of accepting
   *  stale cross-object reuse while true object IDs are unavailable. */
  svgfPrevObjIdPlaceholderTexture: GPUTexture;
  /**
   * Previous-frame normal+depth history for reprojection validity checks.
   * Copied from `common.gNormalDepthTexture` after each denoiser dispatch.
   */
  svgfPrevNormalDepthTexture: GPUTexture;
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
 * Path-guiding (PPG) GPU resources — Müller 2017 Practical Path Guiding
 * (W9, opt-in via `HybridEngineOptions.ppgEnabled`).
 *
 * When PPG is not enabled, every field is `undefined` and the PPG passes are
 * never registered (see `PPGCoordinator.enabled`). When enabled,
 * these buffers are uploaded each rebuild cycle and bound into the PPG guide
 * + update passes:
 *
 *   - sTreeBuf       — serialised spatial kd-tree (Float32Array)
 *   - dTreeBuf       — concatenated per-cell directional quadtrees
 *   - dTreeOffsetsBuf — sTree-cell → dTreeBuf base-offset table
 *   - fluxAtomicsBuf — atomic u32 flux accumulator (one slot per dTree node)
 *   - samplesPosBuf  — per-pixel sample positions (training input; W9 P1 stub-filled)
 *   - samplesDirBuf  — per-pixel sample directions (training input)
 *   - samplesLiBuf   — per-pixel incoming radiance L_i (deviation-3 binding)
 *   - sampleOutBuf   — per-pixel guide sample output (xyz=dir, w=pdf)
 *   - guideUboBuffer — guide kernel UBO (pixelCount, alpha, scene bounds)
 *   - updateUboBuffer — update kernel UBO (sampleCount, fluxBudget)
 *
 * See `ppg/serialise.ts` for the buffer layout.
 */
export interface PPGFrameResources {
  /** Set only when `ppgEnabled` was true at engine init. */
  sTreeBuf?: GPUBuffer;
  dTreeBuf?: GPUBuffer;
  dTreeOffsetsBuf?: GPUBuffer;
  /** Atomic u32 accumulator — one slot per dTree node (matches dTreeBuf layout). */
  fluxAtomicsBuf?: GPUBuffer;
  /** Per-pixel sample inputs to the update kernel (training). */
  samplesPosBuf?: GPUBuffer;
  samplesDirBuf?: GPUBuffer;
  samplesLiBuf?: GPUBuffer;
  /** Per-pixel guide sample output (xyz=dir world, w=pdf). */
  sampleOutBuf?: GPUBuffer;
  /** Guide + update kernel UBOs. */
  guideUboBuffer?: GPUBuffer;
  updateUboBuffer?: GPUBuffer;
}

/**
 * Neural denoiser GPU resources — empty placeholder.
 *
 * W10 (neural denoiser finish) will populate this sub-struct with the
 * weight buffers, intermediate tensors, and any required UBOs. Declared
 * now so consumers can pattern-match on `res.neural` without conditional
 * access.
 */
export interface NeuralFrameResources {
  /** Reserved for W10. */
  readonly _empty?: never;
}

/**
 * All per-frame GPU resources, grouped by owning algorithm.
 *
 * Premium-library rationale (complexity sweep 2026-05-17 findings A3 + B6):
 * the legacy flat 41-field interface forced every consumer to scan a wall of
 * sibling fields and made it impossible to know at a glance which algorithm
 * owned a given resource. With per-algorithm sub-structs, a Pass that only
 * touches GTAO can take `GTAOFrameResources` directly; a SVGF Pass takes
 * `SVGFFrameResources`; nothing needs to be aware of fields it doesn't use.
 */
export interface FrameResources {
  common: CommonFrameResources;
  restirDI: RestirDIFrameResources;
  restirGI: RestirGIFrameResources;
  ddgi: DDGIFrameResources;
  gtao: GTAOFrameResources;
  svgf: SVGFFrameResources;
  ppg: PPGFrameResources;
  neural: NeuralFrameResources;
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

/** 16-byte zeroed storage buffer for unused scene-BGL slots (merged BVH mode). */
export function createDummyStorageBuffer(device: GPUDevice, _label: string): GPUBuffer {
  return uploadBuffer(device, new ArrayBuffer(16), GPUBufferUsage.STORAGE);
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
 *
 * Allocation order is load-bearing: every test + reference render that
 * captures GPU-call traces (createTexture / createBuffer / writeTexture /
 * writeBuffer) relies on the exact ordering below. Do not reorder without
 * regenerating references. W1-R2 preserves the legacy order verbatim — only
 * the assembled return shape changes.
 */
export function createFrameResources(
  device: GPUDevice,
  W: number,
  H: number,
  _options?: FrameResourceOptions,
): FrameResources {
  const common = createCommonFrameResources(device, W, H);
  const restirDI = createRestirDIFrameResources(device, W, H);
  const restirGI = createRestirGIFrameResources(device, W, H);
  const gtao = createGtaoFrameResources(device, W, H);
  const ddgi = createDdgiFrameResources(device);
  const svgf = createSvgfFrameResources(device, W, H);

  // ── Assemble per-algorithm sub-structs ────────────────────────────────────
  // Allocation above is unchanged from the legacy flat layout; the bucketing
  // below is a pure organisational layer. W1-R2 maps each of the 41 legacy
  // sibling fields to exactly one sub-struct — see plan/premium-grade-refactor
  // -20260517.md §W1-R2 for the canonical mapping table.

  // PPG resources are allocated lazily by `allocatePPGResources` when
  // `HybridEngineOptions.ppgEnabled === true`. Default leaves every slot
  // undefined so the pipeline can treat PPG as truly opt-in.
  const ppg: PPGFrameResources = {};
  const neural = createNeuralFrameResources();

  return { common, restirDI, restirGI, ddgi, gtao, svgf, ppg, neural };
}

/**
 * Destroy all resources returned by `createFrameResources`. Safe to call
 * in dispose(); callers must also destroy the static BVH buffers separately.
 *
 * Destruction order mirrors the legacy flat-struct destroy order verbatim so
 * any GPU-call-trace test or telemetry recording continues to observe the
 * same sequence of `.destroy()` calls.
 */
export function destroyFrameResources(r: FrameResources): void {
  // restirDI
  r.restirDI.reservoirCurrentBuffer.destroy();
  r.restirDI.reservoirPreviousBuffer.destroy();
  r.restirDI.reservoirSpatialBuffer.destroy();

  // common (first wave — matches legacy order)
  r.common.hdrColorTexture.destroy();
  r.common.gNormalDepthTexture.destroy();
  r.common.denoisedPingTexture.destroy();
  r.common.denoisedPongTexture.destroy();
  r.common.accumTextureA.destroy();
  r.common.accumTextureB.destroy();
  r.common.placeholderTexture.destroy();
  r.common.uboBuffer.destroy();

  // ddgi
  r.ddgi.ddgiPlaceholderRgba16f.destroy();
  r.ddgi.ddgiPlaceholderRg16f.destroy();
  r.ddgi.ddgiUboBuffer.destroy();

  // common (second wave — variance + motion + tier + resolved)
  r.common.varianceBuffer.destroy();
  r.common.varianceBufferAux.destroy();
  r.common.atrousVarianceEstimateTexture.destroy();
  r.common.motionVectorTexture.destroy();
  r.common.tierTexture.destroy();
  r.common.resolvedTexture.destroy();

  // gtao
  r.gtao.aoHalfTexture.destroy();
  r.gtao.aoFullTexture.destroy();
  r.gtao.gtaoUboBuffer.destroy();

  // restirGI
  r.restirGI.reservoirGiCurrentBuffer.destroy();
  r.restirGI.reservoirGiPreviousBuffer.destroy();
  r.restirGI.reservoirGiSpatialBuffer.destroy();

  // common (Sprint-18 indirect / combined / hdrTotal / indirect ping-pong / albedo)
  r.common.hdrIndirectTexture.destroy();
  r.common.combinedDenoisedTexture.destroy();
  r.common.hdrTotalTexture.destroy();
  r.common.indirectDenoisedPingTexture.destroy();
  r.common.indirectDenoisedPongTexture.destroy();
  r.common.indirectAccumPingTexture.destroy();
  r.common.indirectAccumPongTexture.destroy();
  r.common.albedoTexture.destroy();

  // svgf
  r.svgf.svgfObjIdPlaceholderTexture.destroy();
  r.svgf.svgfPrevObjIdPlaceholderTexture.destroy();
  r.svgf.svgfPrevNormalDepthTexture.destroy();
  r.svgf.svgfHistoryLengthTextureA.destroy();
  r.svgf.svgfHistoryLengthTextureB.destroy();
  r.svgf.svgfMomentsTextureA.destroy();
  r.svgf.svgfMomentsTextureB.destroy();
  r.svgf.svgfPrevRadianceTextureA.destroy();
  r.svgf.svgfPrevRadianceTextureB.destroy();
  r.svgf.svgfVarianceTexture.destroy();
  r.svgf.svgfVarianceMomentsIntermedTexture.destroy();

  // ppg — destroy all allocated buffers (each is optional; null-safe).
  r.ppg.sTreeBuf?.destroy();
  r.ppg.dTreeBuf?.destroy();
  r.ppg.dTreeOffsetsBuf?.destroy();
  r.ppg.fluxAtomicsBuf?.destroy();
  r.ppg.samplesPosBuf?.destroy();
  r.ppg.samplesDirBuf?.destroy();
  r.ppg.samplesLiBuf?.destroy();
  r.ppg.sampleOutBuf?.destroy();
  r.ppg.guideUboBuffer?.destroy();
  r.ppg.updateUboBuffer?.destroy();
  // neural — empty placeholder; nothing to destroy until W10.
}

// ────────────────────────────────────────────────────────────────────────────
// PPG resource allocator (W9 — opt-in via HybridEngineOptions.ppgEnabled)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Allocate the PPG (Müller 2017 path-guiding) GPU buffers and attach them to
 * an existing FrameResources struct. Called once at engine init, AFTER
 * `createFrameResources`, only when `ppgEnabled` is true.
 *
 * Buffer-size policy:
 *   - sTreeBuf:        sized for `maxSpatialCells` × STREE_NODE_F32 + header.
 *   - dTreeBuf:        sized for `maxSpatialCells × maxDTreeNodesPerCell`
 *                      × DTREE_NODE_F32 + per-cell header.
 *   - dTreeOffsetsBuf: u32 × maxSpatialCells.
 *   - fluxAtomicsBuf:  u32 × maxSpatialCells × maxDTreeNodesPerCell.
 *   - samples*Buf:     vec4f × (width × height).
 *   - sampleOutBuf:    vec4f × (width × height).
 *   - guideUboBuffer:  48 bytes (12 × f32 — see PPGGuideUBO in WGSL).
 *   - updateUboBuffer: 16 bytes (4 × u32 — see PPGUpdateUBO in WGSL).
 *
 * The host can resize via {@link allocatePPGResources} on a new
 * FrameResources struct; the old buffers are NOT destroyed here (the caller
 * owns their lifecycle).
 */
export function allocatePPGResources(
  device: GPUDevice,
  res: FrameResources,
  width: number,
  height: number,
  opts?: {
    /**
     * Hard cap on sTree leaf count. Default 1 024 — large enough for
     * meaningful spatial refinement while keeping VRAM bounded at ~6 MB.
     * Hosts that expect dense scenes can raise this up to
     * `PPG_MAX_SPATIAL_CELLS` (16 384).
     */
    maxSpatialCells?: number;
    /**
     * Upper bound on dTree nodes per cell.
     * Default 341 = 1 + 4 + 16 + 64 + 256 (full 4^4 quadtree = depth 4).
     * Matches `PPG_DTREE_MAX_DEPTH = 4`.
     */
    maxDTreeNodesPerCell?: number;
  },
): void {
  const maxSpatialCells = opts?.maxSpatialCells ?? 1_024;
  const maxDTreeNodesPerCell = opts?.maxDTreeNodesPerCell ?? 341;
  const pixelCount = Math.max(1, width * height);

  // Layout constants — must match serialise.ts.
  const STREE_HEADER_F32 = 4;
  const STREE_NODE_F32 = 16;
  const DTREE_HEADER_F32 = 4;
  const DTREE_NODE_F32 = 8;

  const sTreeBufF32 = STREE_HEADER_F32 + maxSpatialCells * STREE_NODE_F32;
  const dTreeBufF32 =
    maxSpatialCells * (DTREE_HEADER_F32 + maxDTreeNodesPerCell * DTREE_NODE_F32);
  const fluxAtomicsCount = maxSpatialCells * maxDTreeNodesPerCell;

  res.ppg.sTreeBuf = device.createBuffer({
    label: 'ppg-sTreeBuf',
    size: Math.max(16, sTreeBufF32 * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  res.ppg.dTreeBuf = device.createBuffer({
    label: 'ppg-dTreeBuf',
    size: Math.max(16, dTreeBufF32 * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  res.ppg.dTreeOffsetsBuf = device.createBuffer({
    label: 'ppg-dTreeOffsets',
    size: Math.max(16, maxSpatialCells * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  res.ppg.fluxAtomicsBuf = device.createBuffer({
    label: 'ppg-fluxAtomics',
    size: Math.max(16, fluxAtomicsCount * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  // Sample input buffers — vec4f per pixel.
  const samplesSize = Math.max(16, pixelCount * 16);
  res.ppg.samplesPosBuf = device.createBuffer({
    label: 'ppg-samplesPos',
    size: samplesSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  res.ppg.samplesDirBuf = device.createBuffer({
    label: 'ppg-samplesDir',
    size: samplesSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  res.ppg.samplesLiBuf = device.createBuffer({
    label: 'ppg-samplesLi',
    size: samplesSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  res.ppg.sampleOutBuf = device.createBuffer({
    label: 'ppg-sampleOut',
    size: samplesSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  res.ppg.guideUboBuffer = device.createBuffer({
    label: 'ppg-guide-ubo',
    size: 48, // 12 × f32 — see PPGGuideUBO in ppgGuide.wgsl.ts.
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  res.ppg.updateUboBuffer = device.createBuffer({
    label: 'ppg-update-ubo',
    size: 16, // 4 × u32 — see PPGUpdateUBO in ppgUpdate.wgsl.ts.
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}
