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
 * is split into resource-owning per-algorithm sub-structs (`common`, `restirDI`,
 * `restirGI`, `ddgi`, `gtao`, `svgf`, `ppg`). Sub-struct boundaries map to the
 * passes in `pipeline/passes/*`. PPG owns its query/training buffers here;
 * neural resources are owned by InferenceGraph, so no empty placeholder bucket
 * is retained. See
 * plan/premium-grade-refactor-20260517.md §W1-R2
 * and complexity-sweep-20260517 findings A3 + B6.
 */

import { createCommonFrameResources } from './frameResources/createCommonFrameResources.js';
import {
  assertFrameResourcePlanSupported,
  planFrameResources,
} from './frameResourcePlan.js';
export {
  DEFAULT_FRAME_RESOURCE_BUDGET_BYTES,
  assertFrameResourceReservoirScale,
  assertFrameResourcePlanSupported,
  planFrameResources,
  resolveFrameResourcePlan,
  type FrameResourceAllocation,
  type FrameResourceFootprint,
  type FrameResourcePlanningOptions,
  type FrameResourceResolutionOptions,
  type FrameResourceResolutionPolicy,
  type ResolvedFrameResourcePlan,
} from './frameResourcePlan.js';

// `buildDDGIPlaceholderUBO` is re-exported here because two pipeline-side
// consumers (frameResources/createDdgiFrameResources.ts, OptionalSubsystemBindingState.ts)
// import it via this module. `packDDGIGridParams` / `DDGIGridParamsInput` were
// also re-exported but had zero importers-via-this-path — all consumers pull
// them from the canonical `../ddgi/ddgiGridUbo.js` directly — so they were
// dropped (T10-DDGI dead-re-export cleanup).
export { buildDDGIPlaceholderUBO } from '../ddgi/ddgiGridUbo.js';
import { createDdgiFrameResources } from './frameResources/createDdgiFrameResources.js';
import { createRestirDIFrameResources } from './frameResources/createRestirDIFrameResources.js';
import { createRestirGIFrameResources } from './frameResources/createRestirGIFrameResources.js';
import { createGtaoFrameResources } from './frameResources/createGtaoFrameResources.js';
import { createSvgfFrameResources } from './frameResources/createSvgfFrameResources.js';
import {
  PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL,
  PPG_DEFAULT_SPATIAL_CELLS,
  PPG_MAX_DTREE_NODES_PER_CELL,
  PPG_MAX_SPATIAL_CELLS,
} from '../ppg/ppgConstants.js';
import {
  createPpgQueryArenaLayout,
  type PpgQueryArenaLayout,
} from '../ppg/ppgQueryArena.js';

// ─── Per-algorithm sub-struct interfaces ─────────────────────────────────────

/**
 * Cross-cutting GPU resources shared by every pass — primary HDR targets,
 * temporal accumulator ping-pong, UBO, samplers, motion vectors, the
 * sample-tier + resolved-radiance textures used by sample-budget /
 * resolve, and Sprint-18 indirect + albedo + variance scaffolding.
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
  uboBuffer: GPUBuffer;
  nearestSampler: GPUSampler;
  /** Screen-space motion (RG32F), written each frame by MotionVectorsPass. */
  motionVectorTexture: GPUTexture;
  /**
   * Immutable current-frame radiance snapshot used by the checkerboard
   * pre-denoiser reconstruction pass. The pass writes gap pixels back into
   * `hdrColorTexture`, so WebGPU cannot legally sample that same subresource;
   * this copy supplies the four fresh shaded neighbours used for temporal
   * color-box rejection. It is full resolution only when checkerboard is on.
   */
  checkerboardRadianceSnapshotTexture: GPUTexture;
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
   * Camera-visible alpha blend composition target. The transparent-OIT pass
   * reads `combinedDenoisedTexture` as the opaque/background radiance, walks
   * fractional `alphaMode:'blend'` layers front-to-back, and writes the
   * composited radiance here. `TemporalAccumPass` consumes this texture after
   * the pass publishes it through `frameState.combinedDenoised`.
   */
  transparentCompositeTexture: GPUTexture;
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
  /** Second Welford ping-pong half (all production denoiser modes). */
  varianceBufferAux: GPUTexture;
  /** Atrous-variance estimation output (`r32float`; `.r` is scalar variance). */
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
   * Half-res ReSTIR-GI reservoir buffer. The sole live layout is 28 u32
   * (112 bytes) per pixel: the original sample prefix at [0..19], followed by
   * generalized reconnection-shift metadata at [20..27]. The historical
   * 20-u32 layout is accepted only by snapshot migration code and is never
   * allocated for execution. See shaders/reservoirGi.wgsl.ts.
   * Size: (W/2) × (H/2) × stride bytes.
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
  ddgiPlaceholderVisRgba16f: GPUTexture;
  ddgiUboBuffer: GPUBuffer;
}

/** GTAO low- and full-resolution AO textures + GTAO uniform buffer. */
export interface GTAOFrameResources {
  /**
   * Sprint 15 — Low-resolution GTAO occlusion factor (rgba16float). Written by
   * `gtaoMain`; consumed by `gtaoUpsampleMain` to reconstruct full-res AO.
   * Allocated at `W/downscale × H/downscale` where `downscale` is 2 for
   * `gtaoMode:'on'` (half-res, the Sprint-15 default) or 4 for
   * `gtaoMode:'quarter'` (quarter-res — 1/16 the AO compute footprint). The
   * field name retains the historical `aoHalf` spelling; "half" is no longer
   * literal once `quarter` mode is selected.
   * E1: bumped from r16float to rgba16float to carry per-channel multi-bounce
   * AO (Jiménez 2016 §5.2 / Eq. 16). The upsample reduces to scalar luminance.
   */
  aoHalfTexture: GPUTexture;
  /**
   * Sprint 15 — Full-resolution GTAO occlusion factor (rgba16float). Written by
   * `gtaoUpsampleMain`; sampled by `shade.wgsl` to modulate the diffuse
   * indirect / direct terms. 1-frame lagged from current shade (AO computes
   * from current frame's gNormalDepth but shade reads the *previous* frame's
   * AO texture for binding-order simplicity).
   */
  aoFullTexture: GPUTexture;
  /** Sprint 15 — GTAO uniforms (16 bytes: tanFovHalf, radiusPx, intensity, depthThresh). */
  gtaoUboBuffer: GPUBuffer;
}

/** SVGF ('svgf-real' mode) persistent textures — object IDs, history, moments, prev-rad, variance. */
export interface SVGFFrameResources {
  /**
   * Current-frame stable object/primitive/triangle ID (r32uint).
   * Full-res only when `svgf-real` is active; otherwise a 1×1 frame-layout
   * placeholder guarded by shade's object-id store. Read by SVGF reprojection as
   * currObjId. 0 is reserved for sky/miss pixels.
   */
  svgfCurrentObjectIdTexture: GPUTexture;
  /**
   * Previous-frame stable object/primitive/triangle ID (r32uint).
   * Full-res only when `svgf-real` is active. Read by SVGF reprojection as
   * prevObjId, then refreshed from svgfCurrentObjectIdTexture at the end of SVGF
   * dispatch.
   */
  svgfPreviousObjectIdTexture: GPUTexture;
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
   * T2.H1 — Per-pixel first + second luminance moments A (rgba32float, full-res).
   * Ping-pong pair with svgfMomentsTextureB.
   * M1 = E[L] (Eq. 4), M2 = E[L²] (Eq. 4). Used by variance pass Eq. 5.
   * Memory per texture at 1080p: 1920×1080×8 ≈ 16 MB.
   */
  svgfMomentsTextureA: GPUTexture;
  /** T2.H1 — Per-pixel moments B (ping-pong pair). */
  svgfMomentsTextureB: GPUTexture;
  /**
   * T2.H1 — Previous-frame first-wavelet radiance A (rgba16float, full-res).
   * Ping-pong pair with svgfPrevRadianceTextureB.
   * Reprojection writes its temporal color, then à-trous iteration zero
   * replaces it with the filtered history prescribed by Schied 2017 §4.3.
   * Read as prevColor on the next frame; swapped each accepted submission.
   * Memory per texture at 1080p: 1920×1080×8 ≈ 16 MB.
   */
  svgfPrevRadianceTextureA: GPUTexture;
  /** T2.H1 — Previous-frame EMA radiance B (ping-pong pair). */
  svgfPrevRadianceTextureB: GPUTexture;
  /**
   * T2.H1 — SVGF variance output texture (r32float, full-res). Written by
   * the 7×7 fallback pass; read by the à-trous chain.
   * Memory at 1080p: 1920×1080×4 ≈ 8 MB.
   */
  svgfVarianceTexture: GPUTexture;
  /**
   * T2.H1 — Intermediate variance from moments (r32float, full-res). Written
   * by svgfVarianceFromMomentsMain; read by svgf7x7FallbackMain.
   */
  svgfVarianceMomentsIntermedTexture: GPUTexture;
}

/**
 * Path-guiding (PPG) GPU resources — Müller 2017 Practical Path Guiding
 * (W9, opt-in via `HybridEngineOptions.ppgEnabled`).
 *
 * All fields are required: instances are only created by
 * {@link allocatePPGResources}, which allocates and returns all four buffers
 * together. Exported so `PPGCoordinator` and bind-group builders can reference
 * it directly (D3.12 factory pattern).
 *
 * The update pass trains directly from `restirGI.reservoirGiCurrentBuffer`.
 * See `ppg/serialise.ts` for the tree buffer layout.
 */
export interface PPGFrameResources {
  /** Versioned sTree+dTree+offset packed query arena. */
  queryArenaBuf: GPUBuffer;
  readonly queryArenaLayout: PpgQueryArenaLayout;
  queryArenaEpoch: number;
  /** Atomic u32 accumulator — one slot per dTree node (matches dTreeBuf layout). */
  fluxAtomicsBuf: GPUBuffer;
  /**
   * A2 — per-spatial-cell training-sample counter (one atomic u32 per sTree
   * leaf cell, indexed by `dTreeIndex`). The update kernel increments this once
   * per accepted training record; the coordinator reads it back each window and
   * feeds `splitOverflowLeaves(sTree, counts)` so high-traffic cells subdivide.
   */
  cellSampleCountsBuf: GPUBuffer;
  /** Update kernel UBO. */
  updateUboBuffer: GPUBuffer;
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
  /** Populated by {@link allocatePPGResources} when PPG is enabled; empty record otherwise. */
  ppg: PPGFrameResources | Record<never, never>;
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
  try {
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data));
    buf.unmap();
    return buf;
  } catch (error) {
    try { buf.destroy(); } catch { /* preserve the upload failure */ }
    throw error;
  }
}

/**
 * Like {@link uploadBuffer} but allocates `data.byteLength + extraBytes`,
 * copies `data` into the front, and leaves the trailing `extraBytes` zeroed.
 *
 * Used for the COMBINED light-tree + ReGIR-grid storage buffer: the light-tree
 * nodes occupy the front (uploaded once), the trailing region is the ReGIR grid
 * the grid-build pass writes each frame. Co-locating them in ONE buffer keeps
 * the RIS pipeline at the eight-storage-buffer floor (a second @group(3) buffer
 * would push it to nine). `extraBytes == 0` is exactly
 * `uploadBuffer`.
 */
export function uploadBufferPadded(
  device: GPUDevice,
  data: ArrayBuffer,
  extraBytes: number,
  usage: number,
): GPUBuffer {
  if (!Number.isSafeInteger(extraBytes) || extraBytes < 0) {
    throw new RangeError(
      `[ReGIR] storage-buffer padding must be a non-negative safe integer; ` +
        `received ${String(extraBytes)}.`,
    );
  }
  const combinedBytes = data.byteLength + extraBytes;
  if (!Number.isSafeInteger(combinedBytes)) {
    throw new RangeError('[ReGIR] combined light-tree/grid buffer size is not a safe integer.');
  }
  if (extraBytes > 0) {
    if ((data.byteLength & 3) !== 0 || (extraBytes & 3) !== 0) {
      throw new RangeError(
        '[ReGIR] combined light-tree/grid storage must be four-byte aligned.',
      );
    }
    const combinedFloatElements = BigInt(combinedBytes / 4);
    if (combinedFloatElements > 0x1_0000_0000n) {
      throw new RangeError(
        '[ReGIR] combined light-tree/grid storage exceeds the WGSL u32 element-index domain.',
      );
    }
  }
  const size = Math.max(combinedBytes, 16);
  const maxBufferSize = reportedDeviceLimit(device, 'maxBufferSize');
  if (maxBufferSize !== undefined && size > maxBufferSize) {
    throw new RangeError(
      `[ReGIR] combined light-tree/grid buffer requires ${size} bytes, exceeding ` +
        `device maxBufferSize=${maxBufferSize}.`,
    );
  }
  const maxStorageBindingSize = reportedDeviceLimit(
    device,
    'maxStorageBufferBindingSize',
  );
  if (maxStorageBindingSize !== undefined && size > maxStorageBindingSize) {
    throw new RangeError(
      `[ReGIR] combined light-tree/grid buffer requires ${size} bytes, exceeding ` +
        `device maxStorageBufferBindingSize=${maxStorageBindingSize}.`,
    );
  }
  const buf = device.createBuffer({
    size,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  try {
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data));
    buf.unmap();
    return buf;
  } catch (error) {
    try { buf.destroy(); } catch { /* preserve the upload failure */ }
    throw error;
  }
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
      GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
}

/**
 * Options for `createFrameResources`.
 */
export interface FrameResourceOptions {
  /** GTAO compute downscale factor. `2` ⇒ half-res AO target (`gtaoMode:'on'`,
   *  default); `4` ⇒ quarter-res AO target (`gtaoMode:'quarter'`). Sizes the
   *  `gtao.aoHalfTexture` at `W/factor × H/factor`. Defaults to `2`. */
  readonly gtaoDownscale?: number;
  /** Allocate SVGF-real's ~80-90 MB @1080p of full-res history/moments/
   *  radiance textures. `true` only when the active denoiser is `svgf-real`
   *  (the sole reader — `SVGFRealDenoiser.dispatch`). When `false` (the default
   *  `atrous-variance` and every other denoiser) those heavy textures collapse
   *  to 1×1 placeholders. The full-res r32uint object-ID pair remains allocated
   *  because shade writes it through the shared frame bind group. Defaults to
   *  `true` so callers that omit it keep the legacy full-allocation behavior. */
  readonly svgfEnabled?: boolean;
  /** Allocate full-res Welford ping-pong textures.
   *  Production uses this for every denoiser: `atrous-variance` writes it
   *  itself and the shared variance tracker covers all other modes. Defaults
   *  to true; direct resource harnesses may opt into 1x1 placeholders. */
  readonly welfordPingPong?: boolean;
  /** Allocate the à-trous-only full-resolution scalar variance estimate.
   *  Defaults to true for direct callers preserving the legacy allocation. */
  readonly atrousVarianceEstimate?: boolean;
  /** Allocate the full-resolution checkerboard reconstruction snapshot. */
  readonly checkerboard?: boolean;
  /** Allocate GTAO targets at their configured resolution. When false, both
   *  targets are seeded 1x1 neutral placeholders and the gated passes never
   *  dispatch against them. */
  readonly gtaoEnabled?: boolean;
  /** Optional host-owned steady-state frame-resource budget. */
  readonly maxPersistentBytes?: number;
  /** Integer ReSTIR reservoir-grid scale relative to shading resolution. */
  readonly reservoirScale?: number;
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
  options?: FrameResourceOptions,
): FrameResources {
  const plan = planFrameResources(W, H, options);
  assertFrameResourcePlanSupported(device, plan, options);
  const created: DestroyableResource[] = [];
  const trackedDevice = new Proxy(device, {
    get(target, property) {
      if (property === 'createBuffer') {
        return (descriptor: GPUBufferDescriptor): GPUBuffer => {
          const resource = target.createBuffer(descriptor);
          created.push(resource);
          return resource;
        };
      }
      if (property === 'createTexture') {
        return (descriptor: GPUTextureDescriptor): GPUTexture => {
          const resource = target.createTexture(descriptor);
          created.push(resource);
          return resource;
        };
      }
      const value = Reflect.get(target, property, target) as unknown;
      if (typeof value !== 'function') return value;
      return (...args: unknown[]): unknown =>
        Reflect.apply(value, target, args) as unknown;
    },
  });

  try {
    const common = createCommonFrameResources(trackedDevice, W, H, {
      welfordPingPong: options?.welfordPingPong ?? true,
      atrousVarianceEstimate:
        options?.atrousVarianceEstimate ?? options?.welfordPingPong ?? true,
      checkerboardSnapshot: options?.checkerboard === true,
    });
    const reservoirScale = options?.reservoirScale ?? 1;
    const restirDI = createRestirDIFrameResources(
      trackedDevice,
      W,
      H,
      reservoirScale,
    );
    const restirGI = createRestirGIFrameResources(
      trackedDevice,
      W,
      H,
      reservoirScale,
    );
    const gtao = createGtaoFrameResources(
      trackedDevice,
      W,
      H,
      options?.gtaoDownscale ?? 2,
      options?.gtaoEnabled ?? true,
    );
    const ddgi = createDdgiFrameResources(trackedDevice);
    const svgf = createSvgfFrameResources(trackedDevice, W, H, options?.svgfEnabled ?? true);

    // PPG resources are allocated lazily by `allocatePPGResources` when
    // `HybridEngineOptions.ppgEnabled === true`.
    const ppg: FrameResources['ppg'] = {};
    return { common, restirDI, restirGI, ddgi, gtao, svgf, ppg };
  } catch (error) {
    for (let i = created.length - 1; i >= 0; i--) {
      try {
        created[i]!.destroy();
      } catch {
        // Cleanup must preserve the allocation failure that caused rollback.
      }
    }
    throw error;
  }
}

/** Minimal interface for GPU resources that own a `destroy()` handle. */
type DestroyableResource = { destroy(): void };

/**
 * Assemble the ordered destroy queue for a `FrameResources` bundle (D3.11).
 *
 * The sequence matches the legacy 35-call inline order verbatim so any
 * GPU-call-trace test or telemetry recording continues to observe the same
 * sequence of `.destroy()` calls. `destroyFrameResources` iterates this
 * array — the ordering is load-bearing, do not reorder.
 */
function buildDestroyQueue(r: FrameResources): DestroyableResource[] {
  const ppg = r.ppg as Partial<PPGFrameResources>;
  const queue: DestroyableResource[] = [
    // restirDI
    r.restirDI.reservoirCurrentBuffer,
    r.restirDI.reservoirPreviousBuffer,
    r.restirDI.reservoirSpatialBuffer,
    // common (first wave — matches legacy order)
    r.common.hdrColorTexture,
    r.common.gNormalDepthTexture,
    r.common.denoisedPingTexture,
    r.common.denoisedPongTexture,
    r.common.accumTextureA,
    r.common.accumTextureB,
    r.common.uboBuffer,
    // ddgi
    r.ddgi.ddgiPlaceholderRgba16f,
    r.ddgi.ddgiPlaceholderVisRgba16f,
    r.ddgi.ddgiUboBuffer,
    // common (second wave — variance + motion + tier + resolved)
    r.common.varianceBuffer,
    r.common.varianceBufferAux,
    r.common.atrousVarianceEstimateTexture,
    r.common.motionVectorTexture,
    r.common.checkerboardRadianceSnapshotTexture,
    r.common.tierTexture,
    r.common.resolvedTexture,
    // gtao
    r.gtao.aoHalfTexture,
    r.gtao.aoFullTexture,
    r.gtao.gtaoUboBuffer,
    // restirGI
    r.restirGI.reservoirGiCurrentBuffer,
    r.restirGI.reservoirGiPreviousBuffer,
    r.restirGI.reservoirGiSpatialBuffer,
    // common (Sprint-18 indirect / combined / hdrTotal / indirect ping-pong / albedo)
    r.common.hdrIndirectTexture,
    r.common.combinedDenoisedTexture,
    r.common.transparentCompositeTexture,
    r.common.hdrTotalTexture,
    r.common.indirectDenoisedPingTexture,
    r.common.indirectDenoisedPongTexture,
    r.common.indirectAccumPingTexture,
    r.common.indirectAccumPongTexture,
    r.common.albedoTexture,
    // svgf
    r.svgf.svgfCurrentObjectIdTexture,
    r.svgf.svgfPreviousObjectIdTexture,
    r.svgf.svgfPrevNormalDepthTexture,
    r.svgf.svgfHistoryLengthTextureA,
    r.svgf.svgfHistoryLengthTextureB,
    r.svgf.svgfMomentsTextureA,
    r.svgf.svgfMomentsTextureB,
    r.svgf.svgfPrevRadianceTextureA,
    r.svgf.svgfPrevRadianceTextureB,
    r.svgf.svgfVarianceTexture,
    r.svgf.svgfVarianceMomentsIntermedTexture,
  ];
  // ppg — only present when allocatePPGResources was called (opt-in).
  if (ppg.queryArenaBuf) queue.push(ppg.queryArenaBuf);
  if (ppg.fluxAtomicsBuf) queue.push(ppg.fluxAtomicsBuf);
  if (ppg.cellSampleCountsBuf) queue.push(ppg.cellSampleCountsBuf);
  if (ppg.updateUboBuffer) queue.push(ppg.updateUboBuffer);
  return queue;
}

/**
 * Destroy all resources returned by `createFrameResources`. Safe to call
 * in dispose(); callers must also destroy the static BVH buffers separately.
 *
 * Destruction order mirrors the legacy flat-struct destroy order verbatim so
 * any GPU-call-trace test or telemetry recording continues to observe the
 * same sequence of `.destroy()` calls (see `buildDestroyQueue`).
 */
export function destroyFrameResources(r: FrameResources): void {
  const destroyed = new Set<DestroyableResource>();
  for (const res of buildDestroyQueue(r)) {
    if (destroyed.has(res)) continue;
    destroyed.add(res);
    try {
      res.destroy();
    } catch {
      // GPU wrappers can throw during device-loss teardown. Each object is an
      // independent owner, so one hostile destroy must not leak the remainder.
    }
  }
}

// ────────────────────────────────────────────────────────────────────────────
// PPG resource allocator (W9 — opt-in via HybridEngineOptions.ppgEnabled)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Allocate the PPG (Müller 2017 path-guiding) GPU buffers and return them as
 * a fully-populated `PPGFrameResources`. Called once at engine init, AFTER
 * `createFrameResources`, only when `ppgEnabled` is true (D3.12 factory
 * pattern — the caller assigns the result to `frameResources.ppg`).
 *
 * Buffer-size policy:
 *   - sTreeBuf:        sized for `maxSpatialCells` × STREE_NODE_F32 + header.
 *   - dTreeBuf:        sized for `maxSpatialCells × maxDTreeNodesPerCell`
 *                      × DTREE_NODE_F32 + per-cell header.
 *   - dTreeOffsetsBuf: u32 × maxSpatialCells.
 *   - fluxAtomicsBuf:  u32 × maxSpatialCells × maxDTreeNodesPerCell.
 *   - updateUboBuffer: 16 bytes (4 × u32 — see PPGUpdateUBO in WGSL).
 *
 * The old buffers are NOT destroyed here (the caller owns their lifecycle).
 */
export interface PPGResourceFootprint {
  readonly maxSpatialCells: number;
  readonly maxDTreeNodesPerCell: number;
  readonly maxSTreeNodes: number;
  readonly sTreeBytes: number;
  readonly dTreeBytes: number;
  readonly dTreeOffsetsBytes: number;
  readonly queryArenaBytes: number;
  readonly fluxAtomicsBytes: number;
  readonly cellSampleCountsBytes: number;
  readonly updateUboBytes: number;
  readonly totalBytes: number;
}

function checkedIntegerProduct(label: string, ...factors: number[]): number {
  let value = 1;
  for (const factor of factors) {
    if (!Number.isSafeInteger(factor) || factor < 0) {
      throw new RangeError(`[PPG] ${label} has an invalid sizing factor: ${factor}.`);
    }
    value *= factor;
    if (!Number.isSafeInteger(value)) {
      throw new RangeError(`[PPG] ${label} exceeds JavaScript safe-integer sizing.`);
    }
  }
  return value;
}

/** Compute the exact persistent GPU allocation made by PPG. */
export function computePPGResourceFootprint(
  maxSpatialCells: number = PPG_DEFAULT_SPATIAL_CELLS,
  maxDTreeNodesPerCell: number = PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL,
): PPGResourceFootprint {
  if (!Number.isSafeInteger(maxSpatialCells) || maxSpatialCells < 1 || maxSpatialCells > PPG_MAX_SPATIAL_CELLS) {
    throw new RangeError(`[PPG] maxSpatialCells must be an integer in [1, ${PPG_MAX_SPATIAL_CELLS}].`);
  }
  if (!Number.isSafeInteger(maxDTreeNodesPerCell) || maxDTreeNodesPerCell < 1 || maxDTreeNodesPerCell > PPG_MAX_DTREE_NODES_PER_CELL) {
    throw new RangeError(`[PPG] maxDTreeNodesPerCell must be an integer in [1, ${PPG_MAX_DTREE_NODES_PER_CELL}].`);
  }
  // A full binary sTree with L leaf cells has 2L-1 nodes. The previous
  // allocation used L nodes and overflowed once refinement crossed L/2 leaves.
  const maxSTreeNodes = checkedIntegerProduct('sTree node capacity', 2, maxSpatialCells) - 1;
  const sTreeBytes = Math.max(16, checkedIntegerProduct('sTree buffer', 4 + 16 * maxSTreeNodes, 4));
  const dTreeBytes = Math.max(16, checkedIntegerProduct('dTree buffer', maxSpatialCells, 4 + 8 * maxDTreeNodesPerCell, 4));
  const dTreeOffsetsBytes = Math.max(16, checkedIntegerProduct('dTree offsets', maxSpatialCells, 4));
  const queryArenaLayout = createPpgQueryArenaLayout({
    sTreeCapacityBytes: sTreeBytes,
    dTreeCapacityBytes: dTreeBytes,
    dTreeOffsetsCapacityBytes: dTreeOffsetsBytes,
    maxSpatialCells,
    maxDTreeNodesPerCell,
  });
  const queryArenaBytes = queryArenaLayout.byteLength;
  const fluxAtomicsBytes = Math.max(16, checkedIntegerProduct('flux atomics', maxSpatialCells, maxDTreeNodesPerCell, 4));
  const cellSampleCountsBytes = Math.max(16, checkedIntegerProduct('cell sample counts', maxSpatialCells, 4));
  const updateUboBytes = 16;
  const totalBytes = checkedIntegerProduct(
    'total allocation',
    queryArenaBytes + fluxAtomicsBytes + cellSampleCountsBytes + updateUboBytes,
  );
  return {
    maxSpatialCells,
    maxDTreeNodesPerCell,
    maxSTreeNodes,
    sTreeBytes,
    dTreeBytes,
    dTreeOffsetsBytes,
    queryArenaBytes,
    fluxAtomicsBytes,
    cellSampleCountsBytes,
    updateUboBytes,
    totalBytes,
  };
}

function reportedDeviceLimit(device: GPUDevice, name: string): number | undefined {
  const raw = (device.limits as unknown as Record<string, unknown> | undefined)?.[name];
  return typeof raw === 'number' && Number.isFinite(raw) && raw > 0 ? raw : undefined;
}

function assertPPGDeviceLimits(
  device: GPUDevice,
  footprint: PPGResourceFootprint,
  width: number,
  height: number,
): void {
  if (!Number.isSafeInteger(width) || width <= 0
      || !Number.isSafeInteger(height) || height <= 0) {
    throw new RangeError(
      `[PPG] render dimensions must be positive safe integers; got ${width}x${height}.`,
    );
  }
  const storageLimit = reportedDeviceLimit(device, 'maxStorageBufferBindingSize');
  const bufferLimit = reportedDeviceLimit(device, 'maxBufferSize');
  const storageBuffers: ReadonlyArray<readonly [string, number]> = [
    ['queryArenaBuf', footprint.queryArenaBytes],
    ['fluxAtomicsBuf', footprint.fluxAtomicsBytes],
    ['cellSampleCountsBuf', footprint.cellSampleCountsBytes],
  ];
  for (const [label, bytes] of storageBuffers) {
    if (storageLimit !== undefined && bytes > storageLimit) {
      throw new RangeError(`[PPG] ${label} requires ${bytes} bytes, exceeding device maxStorageBufferBindingSize=${storageLimit}.`);
    }
    if (bufferLimit !== undefined && bytes > bufferLimit) {
      throw new RangeError(`[PPG] ${label} requires ${bytes} bytes, exceeding device maxBufferSize=${bufferLimit}.`);
    }
  }
  const workgroupSize = 64;
  const invocations = reportedDeviceLimit(device, 'maxComputeInvocationsPerWorkgroup');
  const workgroupX = reportedDeviceLimit(device, 'maxComputeWorkgroupSizeX');
  if (invocations !== undefined && workgroupSize > invocations) {
    throw new RangeError(`[PPG] update workgroup size ${workgroupSize} exceeds maxComputeInvocationsPerWorkgroup=${invocations}.`);
  }
  if (workgroupX !== undefined && workgroupSize > workgroupX) {
    throw new RangeError(`[PPG] update workgroup X ${workgroupSize} exceeds maxComputeWorkgroupSizeX=${workgroupX}.`);
  }
  const halfW = Math.max(1, Math.floor(width / 2));
  const halfH = Math.max(1, Math.floor(height / 2));
  const sampleCount = checkedIntegerProduct('training sample count', halfW, halfH);
  if (sampleCount > 0xffff_ffff) {
    throw new RangeError(
      `[PPG] training sample count ${sampleCount} cannot be represented by the update UBO u32.`,
    );
  }
  const dispatchX = Math.max(1, Math.ceil(sampleCount / workgroupSize));
  const dispatchLimit = reportedDeviceLimit(device, 'maxComputeWorkgroupsPerDimension');
  if (dispatchLimit !== undefined && dispatchX > dispatchLimit) {
    throw new RangeError(`[PPG] update dispatch requires ${dispatchX} workgroups in X, exceeding maxComputeWorkgroupsPerDimension=${dispatchLimit}.`);
  }
}

export function allocatePPGResources(
  device: GPUDevice,
  width: number,
  height: number,
  opts?: {
    /**
     * Hard cap on sTree leaf count. Default 1 024 — large enough for
     * meaningful spatial refinement while keeping persistent PPG buffers bounded at ~12.2 MiB.
     * Hosts that expect dense scenes can raise this up to
     * 16 384 (the maximum supported).
     */
    maxSpatialCells?: number;
    /**
     * Upper bound on dTree nodes per cell.
     * Default 341 = 1 + 4 + 16 + 64 + 256 (full 4^4 quadtree = depth 4).
     * Matches `PPG_DTREE_MAX_DEPTH = 4`.
     */
    maxDTreeNodesPerCell?: number;
  },
): PPGFrameResources {
  const maxSpatialCells = opts?.maxSpatialCells ?? PPG_DEFAULT_SPATIAL_CELLS;
  const maxDTreeNodesPerCell = opts?.maxDTreeNodesPerCell ?? PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL;
  const footprint = computePPGResourceFootprint(maxSpatialCells, maxDTreeNodesPerCell);
  assertPPGDeviceLimits(device, footprint, width, height);

  const created: GPUBuffer[] = [];
  const create = (descriptor: GPUBufferDescriptor): GPUBuffer => {
    const buffer = device.createBuffer(descriptor);
    created.push(buffer);
    return buffer;
  };

  try {
    const queryArenaLayout = createPpgQueryArenaLayout({
      sTreeCapacityBytes: footprint.sTreeBytes,
      dTreeCapacityBytes: footprint.dTreeBytes,
      dTreeOffsetsCapacityBytes: footprint.dTreeOffsetsBytes,
      maxSpatialCells,
      maxDTreeNodesPerCell,
    });
    return {
      queryArenaBuf: create({
        label: 'ppg-query-arena',
        size: footprint.queryArenaBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      queryArenaLayout,
      queryArenaEpoch: 0,
      fluxAtomicsBuf: create({
        label: 'ppg-fluxAtomics',
        size: footprint.fluxAtomicsBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      }),
      // A2 — per-spatial-cell sample counter: one atomic u32 per spatial cell.
      cellSampleCountsBuf: create({
        label: 'ppg-cellSampleCounts',
        size: footprint.cellSampleCountsBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      }),
      updateUboBuffer: create({
        label: 'ppg-update-ubo',
        size: footprint.updateUboBytes, // 4 × u32 — see PPGUpdateUBO in ppgUpdate.wgsl.ts.
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    };
  } catch (error) {
    for (const buffer of created) {
      try { buffer.destroy(); } catch { /* preserve the allocation failure */ }
    }
    throw error;
  }
}
