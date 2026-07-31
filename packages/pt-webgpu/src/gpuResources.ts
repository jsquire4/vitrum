/**
 * GpuResources — the cohesive GPU-resource-lifecycle cluster extracted from the
 * `PTEngineWebGPU` god-class (T14-followup, mirrors the W1-R2 FrameResources split
 * on `@vitrum/walkaround-hybrid`).
 *
 * Owns, as a single sub-struct on the engine (`#gpu`):
 *   - the accumulation + aux textures (accum / normalDepth / albedo / variance /
 *     motionVectors) and their cached views,
 *   - the accum / varianceMoments storage buffers + the params uniform buffer,
 *   - the path-trace compute pipeline sharing ONE explicit GPUPipelineLayout,
 *     and the explicit per-group bind-group
 *     layouts that pipeline layout is built from,
 *   - the cached per-frame bind groups (group 0/1/2), and
 *   - the current accum dims (width / height / byte size).
 *
 * Behavior is preserved verbatim from the prior inline implementation. The only
 * cross-cutting state that stays on the engine is `#samplesAccumulated`: methods
 * that reset it (`ensureAccumResources` on recreate) report that back to the
 * caller (return `recreated: boolean`) rather than reaching into engine state.
 * Bind-group *construction* takes the scene buffers as explicit parameters
 * (those live on the engine), but the resulting groups are
 * cached here because their lifetime is tied to the accum views + pipeline.
 */

import type { EngineWarning } from '@vitrum/core';
import type { PtWebgpuTraceTier } from './traceTier.js';
import type { UploadedSceneBuffers } from './scene/uploadSceneBuffers.js';
import { FRAME_PARAMS_BYTE_SIZE } from './scene/frameParamsLayout.js';
import type { LiteLightTexData, LiteEnvTexData, LiteEnvCdfData } from './scene/litePackedTextures.js';
import {
  composePtWebgpuTraceWgsl,
  composePtWebgpuCompositeTraceWgsl,
  composeSppmPhotonPassWgsl,
} from './wgsl/pathTraceBruteforce.wgsl.js';
import {
  SPPM_PHOTON_CELLS_BYTES,
  SPPM_CELL_COUNTERS_BYTES,
  SPPM_STATS_BYTES,
  SPPM_PHOTON_CELLS_MAX_BYTES,
  SPPM_PIXEL_STATS_BYTES_PER_PIXEL,
} from './wgsl/pathTrace/sppmBindings.wgsl.js';
import { composePtWebgpuTraceLiteWgsl } from './wgsl/pathTraceBruteforceLite.wgsl.js';
import type { PtWebgpuSamplingMode } from './wgsl/common.wgsl.js';
import { PT_WEBGPU_SEED_BLIT_WGSL } from './wgsl/seedBlit.wgsl.js';
import { PT_WEBGPU_PRESENT_WGSL } from './wgsl/present.wgsl.js';
import {
  composeRestirPtProducerWgsl,
  composeRestirPtTemporalWgsl,
  composeRestirPtSpatialWgsl,
  composeRestirPtResolveWgsl,
  RPT_GROUP0_BINDING_BASE,
} from './wgsl/pathTrace/restirPtCompose.wgsl.js';
import {
  RESTIR_PT_PARAMS_BYTES,
} from './wgsl/pathTrace/reservoirPtHero.wgsl.js';

export type PtWebgpuBvhTraversalMode = 'binary' | 'cwbvh-closest';

// ── Module-level binding-layout helpers (D8.3) ─────────────────────────────

/** Two-phase replacement token used by setScene's outer resource transaction. */
export interface LiteTextureReplacement {
  /** Publish candidate views while retaining the previous textures for rollback. */
  commit(): void;
  /** Restore the previous views (if committed) and destroy every candidate. */
  rollback(): void;
  /** Make a committed replacement permanent and destroy the previous textures. */
  finalize(): void;
}

type DestroyableGpuResource = GPUTexture | GPUBuffer;

function destroyGpuResourcesBestEffort(
  resources: readonly (DestroyableGpuResource | null)[],
  preservedResources: readonly (DestroyableGpuResource | null)[] = [],
): void {
  const destroyed = new Set<object>(
    preservedResources.filter((resource): resource is DestroyableGpuResource => resource != null),
  );
  for (const resource of resources) {
    if (resource == null || destroyed.has(resource)) continue;
    destroyed.add(resource);
    try { resource.destroy(); } catch { /* preserve the transaction outcome */ }
  }
}
// These are hoisted out of the per-method local-scope to eliminate the
// duplication that previously existed between #buildSharedPipelineLayout and
// #buildReservoirGroup0Layout.  All values are pure constants; the only
// runtime dependency is GPUShaderStage.COMPUTE which is defined before any
// GpuResources method can execute.

/** All layout entries in this file are COMPUTE-only. */
function _vis(): number { return GPUShaderStage.COMPUTE; }
/** WGSL `var<storage, read>` → 'read-only-storage'. */
const _ro: GPUBufferBindingLayout = { type: 'read-only-storage' };
/** WGSL `var<storage, read_write>` → 'storage'. */
const _rw: GPUBufferBindingLayout = { type: 'storage' };
/** WGSL `var<uniform>` → 'uniform'. */
const _uniform: GPUBufferBindingLayout = { type: 'uniform' };
/** WGSL `texture_storage_2d<rgba16float, write>`. */
const _storageTex: GPUStorageTextureBindingLayout = {
  access: 'write-only',
  format: 'rgba16float',
  viewDimension: '2d',
};
/** Buffer binding entry. */
function _buf(binding: number, layout: GPUBufferBindingLayout): GPUBindGroupLayoutEntry {
  return { binding, visibility: _vis(), buffer: layout };
}
/** Storage texture (rgba16float write-only) binding entry. */
function _tex(binding: number): GPUBindGroupLayoutEntry {
  return { binding, visibility: _vis(), storageTexture: _storageTex };
}
/** WGSL scalar variance `texture_storage_2d<r32float, write>`. */
function _varianceTex(binding: number): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: _vis(),
    storageTexture: {
      access: 'write-only',
      format: 'r32float',
      viewDimension: '2d',
    },
  };
}
/**
 * Sampled texture (unfilterable-float 2d) binding entry — used by B12 lite-tier
 * textures.  Trust-audit F2 (2026-06-10): rgba32float is UNFILTERABLE; declaring
 * 'float' (filterable) made every lite-tier pipeline fail validation.
 */
function _sampledTex(binding: number): GPUBindGroupLayoutEntry {
  return { binding, visibility: _vis(), texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } };
}

interface SharedPipelineLayoutCandidate {
  readonly pipelineLayout: GPUPipelineLayout;
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly bindGroupLayout1: GPUBindGroupLayout | null;
  readonly bindGroupLayout2: GPUBindGroupLayout | null;
  readonly bindGroupLayout3: GPUBindGroupLayout | null;
}

// ── D8.1 Sub-object: ReservoirResources ─────────────────────────────────────
/**
 * Owns the ReSTIR-PT reuse resources: two full-frame reservoir buffers, result
 * buffer, params UBO, the four reuse compute pipelines + composite megakernel,
 * and the cached per-pass reuse bind groups. Gated by `restirPtReuse &&
 * traceTier === 'full'`. GpuResources delegates to this sub-object and exposes
 * all its fields via read-write accessors so the existing external surface
 * (index.ts / tests) remains byte-identical.
 */
class ReservoirResources {
  /**
   * Two full-resolution ReservoirPTHero buffers (64 B/px = 16 u32). `Cur` is
   * producer/temporal scratch. The spatial pass writes its hazard-free output
   * directly into `Prev`; resolve reads `Prev`, which is already next frame's
   * history. No third full-frame reservoir or end-of-frame rotation is required.
   */
  rptReservoirCur: GPUBuffer | null = null;
  rptReservoirPrev: GPUBuffer | null = null;
  /** `rpt_result`: one vec4f / px (16 B) — the resolve pass's reconnection
   *  indirect (.rgb) + contributing flag (.a). STORAGE | COPY_SRC. */
  rptResultBuffer: GPUBuffer | null = null;
  /** RestirPtParams UBO (16 B: mClamp plus three alignment pads). */
  rptParamsBuffer: GPUBuffer | null = null;
  rptReservoirByteSize = 0;
  rptResultByteSize = 0;
  /** The four reuse compute pipelines (lazy, gated). */
  rptProducerPipeline: GPUComputePipeline | null = null;
  rptTemporalPipeline: GPUComputePipeline | null = null;
  rptSpatialPipeline: GPUComputePipeline | null = null;
  rptResolvePipeline: GPUComputePipeline | null = null;
  /** A1 — the COMPOSITE megakernel (E0-direct-only + resolve indirect). */
  rptCompositePipeline: GPUComputePipeline | null = null;
  /** Explicit group-0 layout for the reuse passes (megakernel g0 + reuse bindings 20..25). */
  rptGroup0Layout: GPUBindGroupLayout | null = null;
  /** Cached per-pass reuse bind groups (rebuilt on scene-buffer / reservoir recreation). */
  rptProducerGroup0: GPUBindGroup | null = null;
  rptTemporalGroup0: GPUBindGroup | null = null;
  rptSpatialGroup0: GPUBindGroup | null = null;
  rptResolveGroup0: GPUBindGroup | null = null;

  dispose(): void {
    this.rptReservoirCur?.destroy();
    this.rptReservoirCur = null;
    this.rptReservoirPrev?.destroy();
    this.rptReservoirPrev = null;
    this.rptResultBuffer?.destroy();
    this.rptResultBuffer = null;
    this.rptParamsBuffer?.destroy();
    this.rptParamsBuffer = null;
    this.rptReservoirByteSize = 0;
    this.rptResultByteSize = 0;
    this.rptProducerPipeline = null; // GPUComputePipeline has no destroy()
    this.rptTemporalPipeline = null;
    this.rptSpatialPipeline = null;
    this.rptResolvePipeline = null;
    this.rptCompositePipeline = null;
    this.rptGroup0Layout = null;
    this.rptProducerGroup0 = null;
    this.rptTemporalGroup0 = null;
    this.rptSpatialGroup0 = null;
    this.rptResolveGroup0 = null;
  }
}

// ── D8.1 Sub-object: SppmResources ───────────────────────────────────────────
/**
 * Owns the SPPM photon-map resources: photon cells buffer, cell counters,
 * stats UBO, per-pixel statistics buffer, and the photon-emission pipeline.
 * Gated by `causticStrategy === 'photon-map' && traceTier === 'full'`.
 * GpuResources delegates to this sub-object and exposes all its fields via
 * read-write accessors so the existing external surface remains unchanged.
 */
class SppmResources {
  /**
   * A4 — SPPM photon hash-grid cells buffer.
   * One 48-byte PhotonRecord per emitted lane (65,536 records = 3 MiB).
   * Bound at group(3) binding(6).  A 64-byte placeholder is used when SPPM is
   * off so the group-3 layout slot is satisfied without allocating the real buffer.
   */
  sppmPhotonCellsBuffer: GPUBuffer | null = null;
  /**
   * A4 — SPPM cell-insertion counters: atomic<u32>[SPPM_MAX_CELLS] ≈ 256 KiB.
   * Cleared at the start of each frame before the photon pass writes into it.
   * Bound at group(3) binding(7).
   */
  sppmCellCountersBuffer: GPUBuffer | null = null;
  /**
   * A4 — SppmStats UBO (32 bytes): currentRadius, r0, frameAccumulated,
   * photonCount, sceneExtent, sceneCenter.xyz. Written per-frame by the host.
   * Bound at group(3) binding(8).
   */
  sppmStatsBuffer: GPUBuffer | null = null;
  /**
   * A4-progressive — per-pixel SPPM statistics buffer.
   * SppmPixelStats[W×H×2] = separate surface/volume
   * {tau.rgb, linearRadius, N, _pad×3} records × 32 bytes = 64 bytes/px.
   * Allocated in `ensureSppmPixelStatsBuffer`; reset (GPU-cleared) whenever
   * the PT accumulator resets. Bound at group(3) binding(9).
   */
  sppmPixelStatsBuffer: GPUBuffer | null = null;
  sppmPixelStatsWidth  = 0;
  sppmPixelStatsHeight = 0;
  /** Compute pipeline for the photon-emission pre-pass (sppmEmitPhotons). */
  sppmPhotonPipeline: GPUComputePipeline | null = null;
  /** true if the SPPM buffers have been allocated at full size (not placeholder). */
  sppmBuffersReady = false;

  dispose(): void {
    this.sppmPhotonCellsBuffer?.destroy();
    this.sppmPhotonCellsBuffer = null;
    this.sppmCellCountersBuffer?.destroy();
    this.sppmCellCountersBuffer = null;
    this.sppmStatsBuffer?.destroy();
    this.sppmStatsBuffer = null;
    this.sppmPixelStatsBuffer?.destroy();
    this.sppmPixelStatsBuffer = null;
    this.sppmPixelStatsWidth  = 0;
    this.sppmPixelStatsHeight = 0;
    this.sppmPhotonPipeline = null; // GPUComputePipeline has no destroy()
    // The SPPM bind-group layout is part of group-3 (bindGroupLayout3 / the shared
    // pipeline layout); it is released via the normal bindGroupLayout3 null-out in
    // GpuResources.dispose() — no separate ref to drop here.
    this.sppmBuffersReady = false;
  }
}

// ── D8.1 Sub-object: PresentResources ────────────────────────────────────────
/**
 * Owns the present-pass pipeline, params UBO, seed-blit pipeline + sampler +
 * params + variance-placeholder, and the present texture/view. GpuResources
 * delegates to this sub-object; the present texture is ALSO exposed as a
 * top-level field on GpuResources for the accum-size-tied lifecycle
 * (destroyAccumTexture clears it; ensureAccumResources recreates it).
 */
class PresentResources {
  /**
   * Present-pass output texture (rgba16float, TEXTURE_BINDING | STORAGE_BINDING).
   * Sized to match accumTexture; recreated when the accum dims change.
   * The present compute pass reads accumTexture and writes the tonemapped +
   * OETF-encoded result here. Null before the first ensureAccumResources call.
   */
  presentTexture: GPUTexture | null = null;
  presentView: GPUTextureView | null = null;

  /**
   * Accepted linear-HDR OIDN output. It is deliberately separate from the
   * stable present texture so presentation changes can reuse it without a new
   * inference or CPU upload.
   */
  denoisedLinearTexture: GPUTexture | null = null;


  /**
   * PresentParams UBO (16 bytes): tonemapMode (u32), exposure (f32),
   * outputColorSpace (u32), _pad (u32).
   */
  presentParamsBuffer: GPUBuffer | null = null;
  /** The present compute pipeline (lazily built on first ensurePresentPipeline). */
  presentPipeline: GPUComputePipeline | null = null;

  /** Seed-blit compute pipeline (P8 progressive handoff). */
  seedBlitPipeline: GPUComputePipeline | null = null;
  seedBlitParamsBuffer: GPUBuffer | null = null;
  seedBlitSampler: GPUSampler | null = null;
  /**
   * Placeholder storage buffer at the seed-blit's varianceMoments slot on the
   * lite tier (which has no real varianceMomentsBuffer). Discarded at write time.
   */
  seedBlitVarPlaceholder: GPUBuffer | null = null;
  seedBlitVarPlaceholderByteSize = 0;

  dispose(): void {
    // Seed-blit resources (freed before present, matching original dispose order).
    this.seedBlitParamsBuffer?.destroy();
    this.seedBlitParamsBuffer = null;
    this.seedBlitVarPlaceholder?.destroy();
    this.seedBlitVarPlaceholder = null;
    this.seedBlitVarPlaceholderByteSize = 0;
    this.seedBlitSampler = null;     // GPUSampler has no destroy(); drop the ref
    this.seedBlitPipeline = null;    // GPUComputePipeline has no destroy(); drop the ref
    // Present-pass resources.
    this.presentParamsBuffer?.destroy();
    this.presentParamsBuffer = null;
    this.presentPipeline = null;     // GPUComputePipeline has no destroy(); drop the ref
    // presentTexture / presentView are destroyed by GpuResources.destroyAccumTexture()
    // (their lifecycle is tied to the accum dims); PresentResources.dispose() only
    // drops the refs if the texture has already been nulled by that path.
    this.presentTexture = null;
    this.presentView = null;
    this.denoisedLinearTexture = null;
  }
}

export class GpuResources {
  readonly #device: GPUDevice;
  readonly #traceTier: PtWebgpuTraceTier;
  readonly #bdpt: boolean;
  readonly #sampling: PtWebgpuSamplingMode;
  readonly #cwbvhClosest: boolean;
  readonly #onWarning: ((warning: EngineWarning) => void) | undefined;
  /**
   * Compile-time opt-in for the ReSTIR-PT reservoir/reuse pre-passes (the hero-
   * stack temporal reconnection-reuse path). OFF by default and full-tier only.
   * When OFF, NONE of the reuse resources/pipelines below are ever created, and
   * `renderFrame`'s default megakernel path is byte-identical — the wgslContract
   * SHA pin + every existing test stay green. Mirrors the `#bdpt` flag.
   */
  readonly #restirPtReuse: boolean;

  // ── D8.1 Sub-objects (cohesive resource clusters) ─────────────────────────
  /** D8.1 — ReSTIR-PT reservoir + pipeline + bind-group cluster. */
  readonly #rsvr: ReservoirResources = new ReservoirResources();
  /** D8.1 — SPPM photon-map resource cluster. */
  readonly #sppm: SppmResources = new SppmResources();
  /** D8.1 — Present-pass + seed-blit resource cluster. */
  readonly #present: PresentResources = new PresentResources();

  accumTexture: GPUTexture | null = null;
  accumView: GPUTextureView | null = null;
  normalDepthTexture: GPUTexture | null = null;
  normalDepthView: GPUTextureView | null = null;
  albedoTexture: GPUTexture | null = null;
  albedoView: GPUTextureView | null = null;
  varianceTexture: GPUTexture | null = null;
  varianceView: GPUTextureView | null = null;
  motionVectorsTexture: GPUTexture | null = null;
  motionVectorsView: GPUTextureView | null = null;
  accumBuffer: GPUBuffer | null = null;
  varianceMomentsBuffer: GPUBuffer | null = null;
  /**
   * BDPT t=1 per-frame RGB splat sum (four atomic u32 words per pixel).
   * Allocated only for a full-tier bdpt:true engine, cleared before each trace
   * dispatch, and resolved into accumBuffer by bdptCameraSplatResolvePipeline.
   */
  bdptCameraSplatBuffer: GPUBuffer | null = null;
  accumBufferByteSize = 0;
  accumWidth = 0;
  accumHeight = 0;

  /** Reused bind groups until scene buffers or accum views are recreated. */
  pathTraceBindGroup: GPUBindGroup | null = null;
  pathTraceBindGroup1: GPUBindGroup | null = null;
  pathTraceBindGroup2: GPUBindGroup | null = null;
  /** WS2 — light-tree node buffer bind group (full tier only). */
  pathTraceBindGroup3: GPUBindGroup | null = null;

  paramsBuffer: GPUBuffer | null = null;
  computePipeline: GPUComputePipeline | null = null;
  /** Second entry point in the BDPT megakernel module; null when BDPT is off. */
  bdptCameraSplatResolvePipeline: GPUComputePipeline | null = null;
  /**
   * Explicit group-0 bind-group layout. Used to build `pathTraceBindGroup`.
   *
   * The path-trace pipeline uses one explicit `GPUPipelineLayout`. BDPT light
   * prefixes are generated inside `main`; there is no auxiliary BDPT pipeline.
   */
  bindGroupLayout: GPUBindGroupLayout | null = null;
  /** Explicit group-1 layout (full tier only): analytics + env + area lights. */
  bindGroupLayout1: GPUBindGroupLayout | null = null;
  /** Explicit group-2 layout (full tier only): TLAS table. */
  bindGroupLayout2: GPUBindGroupLayout | null = null;
  /** Explicit group-3 layout (full tier only): WS2 light-tree node buffer. */
  bindGroupLayout3: GPUBindGroupLayout | null = null;

  // ── D8.1 sub-object public accessors (D9-2: sub-objects are the real owners) ──
  /** ReSTIR-PT reuse reservoir/pipeline/bind-group cluster. Access fields via
   *  `gpu.reservoir.<field>` (was the former delegating get/set accessor pairs). */
  get reservoir(): ReservoirResources { return this.#rsvr; }
  /** SPPM photon-map resource cluster. Access fields via `gpu.sppm.<field>`. */
  get sppm(): SppmResources { return this.#sppm; }
  /** Present-pass + seed-blit resource cluster. Access fields via `gpu.present.<field>`. */
  get present(): PresentResources { return this.#present; }

  // ── B12 — Lite-tier packed textures (group-0 bindings 12–14) ─────────────────
  /**
   * B12 — env radiance+pdf texture (binding 12), RGBA32F, envWidth×envHeight.
   * 1×1 black placeholder when no HDRI/sky is loaded.  Lite tier only; null on full.
   */
  liteEnvTexture: GPUTexture | null = null;
  liteEnvTextureView: GPUTextureView | null = null;
  /**
   * B12 — env CDF texture (binding 13), RGBA32F (.r = CDF entry), envWidth×envHeight.
   * 1×1 zero placeholder when no HDRI/sky is loaded.  Lite tier only; null on full.
   */
  liteEnvCdfTexture: GPUTexture | null = null;
  liteEnvCdfTextureView: GPUTextureView | null = null;
  /**
   * B12 — light data texture (binding 14), RGBA32F, liteLightTexWidth×1.
   * Packs point/spot/rect-area light records contiguously.  1×1 black placeholder
   * when the scene has no such lights.  Lite tier only; null on full.
   */
  liteLightTexture: GPUTexture | null = null;
  liteLightTextureView: GPUTextureView | null = null;

  /** Bytes per compact ReservoirPTHero (16 u32). MUST equal the shader stride·4
   *  in reservoirPtHero.wgsl.ts (pinned by reservoirPtHeroLayout.test.ts). */
  static readonly RESERVOIR_PT_HERO_BYTES = 64;
  /** RestirPtParams UBO byte size (4 × 4-byte fields). */
  static readonly RESTIR_PT_PARAMS_BYTES = RESTIR_PT_PARAMS_BYTES;
  /**
   * Portable per-binding ceiling. A 64 B/px 1920×1080 reservoir is 126.6 MiB,
   * fitting the WebGPU default 128 MiB maxStorageBufferBindingSize.
   */
  static readonly RESTIR_PT_RESERVOIR_MAX_BYTES = 128 * 1024 * 1024;
  /** Aggregate budget for Cur + Prev + the 16 B/px resolved-result buffer. */
  static readonly RESTIR_PT_TOTAL_MAX_BYTES = 320 * 1024 * 1024;

  /** Byte size of the PresentParams UBO: 4 × 4 bytes. */
  static readonly PRESENT_PARAMS_BYTES = 16;

  /**
   * H14-F — once-gate set for per-frame buffer-ceiling warnings. A warning fires
   * at most once per key (e.g. 'bdptEyeStack', 'restirPtReservoir') for the
   * lifetime of this GpuResources instance. Keys are added on the first warning;
   * subsequent frames that hit the same ceiling are silently suppressed.
   */
  readonly #ceilingWarnedKeys = new Set<string>();

  constructor(
    device: GPUDevice,
    traceTier: PtWebgpuTraceTier,
    bdpt: boolean,
    restirPtReuse = false,
    onWarning?: (warning: EngineWarning) => void,
    sampling: PtWebgpuSamplingMode = 'pcg',
    bvhTraversal: PtWebgpuBvhTraversalMode = 'binary',
  ) {
    this.#device = device;
    this.#traceTier = traceTier;
    this.#bdpt = bdpt;
    this.#sampling = sampling;
    this.#onWarning = onWarning;
    this.#cwbvhClosest = bvhTraversal === 'cwbvh-closest' && traceTier === 'full';
    // Reuse is full-tier only: the per-pass layouts bind the full-tier scene
    // groups (analytics/TLAS/lights). On the lite tier the flag is inert.
    this.#restirPtReuse = restirPtReuse && traceTier === 'full';
  }

  #emitResourceCeilingWarning(key: string, warning: EngineWarning): void {
    if (this.#ceilingWarnedKeys.has(key)) return;
    this.#ceilingWarnedKeys.add(key);
    if (this.#onWarning != null) {
      this.#onWarning(warning);
      return;
    }
    console.warn(warning.message);
  }

  destroyAccumTexture(): void {
    this.accumTexture?.destroy();
    this.accumTexture = null;
    this.accumView = null;
    this.normalDepthTexture?.destroy();
    this.normalDepthTexture = null;
    this.normalDepthView = null;
    this.albedoTexture?.destroy();
    this.albedoTexture = null;
    this.albedoView = null;
    this.varianceTexture?.destroy();
    this.varianceTexture = null;
    this.varianceView = null;
    this.motionVectorsTexture?.destroy();
    this.motionVectorsTexture = null;
    this.motionVectorsView = null;
    this.accumBuffer?.destroy();
    this.accumBuffer = null;
    this.varianceMomentsBuffer?.destroy();
    this.varianceMomentsBuffer = null;
    this.bdptCameraSplatBuffer?.destroy();
    this.bdptCameraSplatBuffer = null;
    this.accumBufferByteSize = 0;
    this.accumWidth = 0;
    this.accumHeight = 0;
    // The present texture is sized to accumTexture; destroy it together.
    this.#present.presentTexture?.destroy();
    this.#present.presentTexture = null;
    this.#present.presentView = null;
    this.#present.denoisedLinearTexture?.destroy();
    this.#present.denoisedLinearTexture = null;

  }

  clearAccumBuffer(): void {
    if (this.accumBuffer == null) return;
    const encoder = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.clearAccum' });
    encoder.clearBuffer(this.accumBuffer);
    if (this.varianceMomentsBuffer != null) {
      encoder.clearBuffer(this.varianceMomentsBuffer);
    }
    if (this.bdptCameraSplatBuffer != null) {
      encoder.clearBuffer(this.bdptCameraSplatBuffer);
    }
    this.#device.queue.submit([encoder.finish()]);
  }

  /**
   * Clear every scene-dependent temporal buffer in one command submission.
   *
   * Besides avoiding three queue submissions on an ordinary reset, this gives
   * setScene a single synchronous failure boundary: no later clear can throw
   * after an earlier clear has already been accepted by the queue.
   */
  clearTemporalBuffers(): void {
    const buffers = new Set<GPUBuffer>();
    if (this.accumBuffer != null) buffers.add(this.accumBuffer);
    if (this.varianceMomentsBuffer != null) {
      buffers.add(this.varianceMomentsBuffer);
    }
    if (this.bdptCameraSplatBuffer != null) {
      buffers.add(this.bdptCameraSplatBuffer);
    }
    if (this.#rsvr.rptReservoirCur != null) {
      buffers.add(this.#rsvr.rptReservoirCur);
    }
    if (this.#rsvr.rptReservoirPrev != null) {
      buffers.add(this.#rsvr.rptReservoirPrev);
    }
    if (
      this.#sppm.sppmPixelStatsBuffer != null &&
      this.#sppm.sppmPixelStatsWidth !== 0
    ) {
      buffers.add(this.#sppm.sppmPixelStatsBuffer);
    }
    if (buffers.size === 0) return;

    const encoder = this.#device.createCommandEncoder({
      label: 'vitrum.pt-webgpu.clearTemporalBuffers',
    });
    for (const buffer of buffers) encoder.clearBuffer(buffer);
    this.#device.queue.submit([encoder.finish()]);
  }

  /**
   * (Re)allocate the accum + aux textures and the accum / varianceMoments
   * buffers to the requested dims. Returns `true` if a recreate happened (which
   * means the caller must reset its sample counter — the prior inline version
   * set `#samplesAccumulated = 0` here; that one piece of engine state is now
   * reported back rather than reached into). Returns `false` on the cache-hit
   * fast path where nothing was touched.
   */
  ensureAccumResources(width: number, height: number): boolean {
    if (!Number.isSafeInteger(width) || width <= 0 ||
        !Number.isSafeInteger(height) || height <= 0) {
      throw new RangeError(
        `pt-webgpu render-target dimensions must be positive safe integers ` +
        `(got ${String(width)}x${String(height)})`,
      );
    }
    const pixelCount = width * height;
    const targetByteSize = pixelCount * 16;
    if (!Number.isSafeInteger(pixelCount) || !Number.isSafeInteger(targetByteSize)) {
      throw new RangeError(
        `pt-webgpu render-target allocation ${width}x${height} exceeds the safe integer range`,
      );
    }
    // `GPUDevice.limits` is mandatory in browsers. Keep this runtime guard for
    // lightweight injected/test devices so validation still fails only on an
    // advertised finite ceiling rather than dereferencing an absent mock field.
    const limits = this.#device.limits as GPUSupportedLimits | undefined;
    const maxTextureDimension2D = Number(limits?.maxTextureDimension2D);
    if (Number.isFinite(maxTextureDimension2D) && maxTextureDimension2D > 0 &&
        (width > maxTextureDimension2D || height > maxTextureDimension2D)) {
      throw new RangeError(
        `pt-webgpu render-target allocation ${width}x${height} exceeds ` +
        `maxTextureDimension2D=${maxTextureDimension2D}`,
      );
    }
    const maxBufferSize = Number(limits?.maxBufferSize);
    const maxStorageBindingSize = Number(limits?.maxStorageBufferBindingSize);
    const finiteBufferLimits = [maxBufferSize, maxStorageBindingSize]
      .filter((limit) => Number.isFinite(limit) && limit > 0);
    const effectiveBufferLimit = finiteBufferLimits.length > 0
      ? Math.min(...finiteBufferLimits)
      : Number.POSITIVE_INFINITY;
    if (targetByteSize > effectiveBufferLimit) {
      throw new RangeError(
        `pt-webgpu render-target allocation ${width}x${height} requires a ` +
        `${targetByteSize}-byte storage buffer; device limit is ${effectiveBufferLimit}`,
      );
    }
    const textureReady =
      this.accumTexture != null && this.accumWidth === width && this.accumHeight === height;
    const bufferReady =
      this.accumBuffer != null &&
      this.accumBufferByteSize === targetByteSize &&
      (!this.#bdpt || this.bdptCameraSplatBuffer != null);
    if (textureReady && bufferReady) {
      return false;
    }
    const previous = [
      this.accumTexture, this.normalDepthTexture, this.albedoTexture,
      this.varianceTexture, this.motionVectorsTexture, this.accumBuffer,
      this.varianceMomentsBuffer, this.bdptCameraSplatBuffer,
      this.#present.presentTexture,
      this.#present.denoisedLinearTexture,
    ];
    const created: Array<GPUTexture | GPUBuffer> = [];
    const forbiddenResources = new Set<object>(
      previous.filter((resource) => resource != null),
    );
    const registerCandidate = <T extends DestroyableGpuResource>(resource: T, label: string): T => {
      if (forbiddenResources.has(resource)) {
        throw new Error(`[pt-webgpu] ${label} candidate aliased an existing GPU resource`);
      }
      created.push(resource);
      forbiddenResources.add(resource);
      return resource;
    };
    let accumTexture: GPUTexture;
    let accumView: GPUTextureView;
    let normalDepthTexture: GPUTexture;
    let normalDepthView: GPUTextureView;
    let albedoTexture: GPUTexture;
    let albedoView: GPUTextureView;
    let varianceTexture: GPUTexture;
    let varianceView: GPUTextureView;
    let motionVectorsTexture: GPUTexture | null = null;
    let motionVectorsView: GPUTextureView | null = null;
    let accumBuffer: GPUBuffer;
    let varianceMomentsBuffer: GPUBuffer | null = null;
    let bdptCameraSplatBuffer: GPUBuffer | null = null;
    let presentTexture: GPUTexture;
    let presentView: GPUTextureView;
    try {
      const createTexture = (descriptor: GPUTextureDescriptor): GPUTexture => {
        return registerCandidate(this.#device.createTexture(descriptor), String(descriptor.label));
      };
      accumTexture = createTexture({
        label: 'vitrum.pt-webgpu.accum', size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
      accumView = accumTexture.createView();
      normalDepthTexture = createTexture({
        label: 'vitrum.pt-webgpu.normalDepth', size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
      normalDepthView = normalDepthTexture.createView();
      albedoTexture = createTexture({
        label: 'vitrum.pt-webgpu.albedo', size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
      albedoView = albedoTexture.createView();
      varianceTexture = createTexture({
        label: 'vitrum.pt-webgpu.variance', size: { width, height, depthOrArrayLayers: 1 },
        format: 'r32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      varianceView = varianceTexture.createView();
      if (this.#traceTier === 'full') {
        motionVectorsTexture = createTexture({
          label: 'vitrum.pt-webgpu.motionVectors', size: { width, height, depthOrArrayLayers: 1 },
          format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        motionVectorsView = motionVectorsTexture.createView();
      }
      accumBuffer = registerCandidate(this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.accum.buffer', size: Math.max(16, targetByteSize),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }), 'vitrum.pt-webgpu.accum.buffer');
      if (this.#traceTier === 'full') {
        varianceMomentsBuffer = registerCandidate(this.#device.createBuffer({
          label: 'vitrum.pt-webgpu.varianceMoments.buffer', size: Math.max(16, targetByteSize),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }), 'vitrum.pt-webgpu.varianceMoments.buffer');
      }
      if (this.#bdpt) {
        bdptCameraSplatBuffer = registerCandidate(this.#device.createBuffer({
          label: 'vitrum.pt-webgpu.bdpt.cameraSplats.buffer',
          size: Math.max(16, targetByteSize),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }), 'vitrum.pt-webgpu.bdpt.cameraSplats.buffer');
      }
      presentTexture = createTexture({
        label: 'vitrum.pt-webgpu.present', size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
      presentView = presentTexture.createView();
      const encoder = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.clearAccum.candidate' });
      encoder.clearBuffer(accumBuffer);
      if (varianceMomentsBuffer != null) encoder.clearBuffer(varianceMomentsBuffer);
      if (bdptCameraSplatBuffer != null) {
        encoder.clearBuffer(bdptCameraSplatBuffer);
      }
      this.#device.queue.submit([encoder.finish()]);
    } catch (error) {
      destroyGpuResourcesBestEffort(created, previous);
      throw error;
    }

    this.accumTexture = accumTexture;
    this.accumView = accumView;
    this.normalDepthTexture = normalDepthTexture;
    this.normalDepthView = normalDepthView;
    this.albedoTexture = albedoTexture;
    this.albedoView = albedoView;
    this.varianceTexture = varianceTexture;
    this.varianceView = varianceView;
    this.motionVectorsTexture = motionVectorsTexture;
    this.motionVectorsView = motionVectorsView;
    this.accumBuffer = accumBuffer;
    this.varianceMomentsBuffer = varianceMomentsBuffer;
    this.bdptCameraSplatBuffer = bdptCameraSplatBuffer;
    this.accumBufferByteSize = targetByteSize;
    this.accumWidth = width;
    this.accumHeight = height;
    this.#present.presentTexture = presentTexture;
    this.#present.presentView = presentView;
    this.#present.denoisedLinearTexture = null;
    // Drop ALL cached bind groups: the path-trace groups AND (when reuse is on)
    // the reuse group-0, which references the just-recreated accum/aux views.
    this.invalidateBindGroups();
    destroyGpuResourcesBestEffort(previous, created);
    return true;
  }

  /**
   * Build the explicit `GPUBindGroupLayout`s for the current tier and wrap them
   * in the path-trace pipeline's single explicit `GPUPipelineLayout`. The
   * binding indices / resource types / visibility MUST match what the prior
   * `layout:'auto'` derived from the WGSL `@group/@binding` decls (see
   * `wgsl/pathTrace/material.wgsl.ts`) so the existing bind-group construction
   * in `buildBindGroups` stays valid unchanged.
   *
   * All bindings are COMPUTE-visible (the entire kernel is one compute stage).
   * BDPT keeps its path vertices invocation-private but adds binding 14 for
   * cross-pixel t=1 camera splats and a second resolver entry point.
   */
  /**
   * D8.2 — The full-tier group-0 bind-group layout entries (bindings 0..13,
   * plus BDPT-only atomic camera splats at binding 14).
   * This is the CANONICAL definition shared by BOTH `#buildSharedPipelineLayout`
   * (which builds the megakernel's group-0 layout) and `#buildReservoirGroup0Layout`
   * (which extends the same 0..13 base with the reuse bindings 20..25). Keeping
   * one definition eliminates the historical F1 full-tier crash class: any future
   * binding-slot change only needs to be made HERE and is automatically reflected
   * in both pipeline layouts, guaranteeing byte-identical group-0 layouts across
   * the megakernel and the ReSTIR-PT reuse passes.
   *
   * Full tier only — callers must not invoke this on the lite tier.
   */
  #makeGroup0LayoutEntries(): GPUBindGroupLayoutEntry[] {
    return [
      _tex(0),        // outputTexture (storage texture, write)
      _buf(1, _uniform), // params (uniform)
      _buf(2, _rw),   // accumBuffer (read_write)
      _buf(3, _ro),   // positions
      _buf(4, _ro),   // indices
      _buf(5, _ro),   // triMaterialIds
      _buf(6, _ro),   // materials
      _buf(7, _ro),   // bvhNodes
      _buf(8, _ro),   // normals
      _tex(9),        // normalDepthTexture
      _tex(10),       // albedoTexture
      _varianceTex(11), // varianceTexture (scalar r32float)
      _tex(12),       // motionVectorsTexture (full tier)
      _buf(13, _rw),  // varianceMomentsBuffer (read_write, full tier)
      ...(this.#bdpt
        ? [_buf(14, _rw)] // BDPT t=1 atomic RGB camera-splat buffer
        : []),
    ];
  }

  /**
   * D8.2 — The full-tier group-0 bind-group entries (bindings 0..13) that bind
   * the current accum/aux views + scene geometry buffers.  Both `buildBindGroups`
   * and `buildReservoirBindGroups` need an identical 0..13 prefix; extracting it
   * here eliminates the prior duplication and guarantees the two sites always bind
   * the same resources to the same slots — any change is propagated automatically.
   *
   * Callers supply the `sb` scene buffers; this method reads the current accum/
   * aux views + varianceMomentsBuffer off `this`.  Full tier only.
   */
  /**
   * D9-6 — The group-0 bindings 0..11 shared byte-identically by BOTH tiers
   * (accum/params/geometry/aux-views). The lite and full group-0 entry lists
   * DIVERGE only at 12+ (lite: B12 sampled env/light textures 12-14; full:
   * motion-vectors storage texture 12 + variance-moments buffer 13). Extracting
   * the common prefix here means the two tiers can never drift on bindings 0..11.
   */
  #makeGroup0SharedPrefixEntries(sb: UploadedSceneBuffers): GPUBindGroupEntry[] {
    return [
      { binding: 0, resource: this.accumView! },
      { binding: 1, resource: { buffer: this.paramsBuffer! } },
      { binding: 2, resource: { buffer: this.accumBuffer! } },
      { binding: 3, resource: { buffer: sb.positionsBuffer } },
      { binding: 4, resource: { buffer: sb.indicesBuffer } },
      { binding: 5, resource: { buffer: sb.triMaterialIdsBuffer } },
      { binding: 6, resource: { buffer: sb.materialsBuffer } },
      { binding: 7, resource: { buffer: sb.bvhNodesBuffer } },
      { binding: 8, resource: { buffer: sb.normalsBuffer } },
      { binding: 9, resource: this.normalDepthView! },
      { binding: 10, resource: this.albedoView! },
      { binding: 11, resource: this.varianceView! },
    ];
  }

  #makeGroup0BindGroupEntries(sb: UploadedSceneBuffers): GPUBindGroupEntry[] {
    return [
      ...this.#makeGroup0SharedPrefixEntries(sb),
      { binding: 12, resource: this.motionVectorsView! },
      { binding: 13, resource: { buffer: this.varianceMomentsBuffer! } },
      ...(this.#bdpt
        ? [{
            binding: 14,
            resource: { buffer: this.bdptCameraSplatBuffer! },
          }]
        : []),
    ];
  }

  /**
   * D9-6 — The full-tier group-2 bind-group entries (TLAS table bindings 0..4).
   * BDPT eye/light stacks are invocation-private WGSL values, so this group has
   * no BDPT-owned resource. Full tier only.
   */
  #makeGroup2BindGroupEntries(sb: UploadedSceneBuffers): GPUBindGroupEntry[] {
    return [
      { binding: 0, resource: { buffer: sb.tlasNodesBuffer } },
      { binding: 1, resource: { buffer: sb.tlasInstanceIndicesBuffer } },
      { binding: 2, resource: { buffer: sb.tlasBlasRootsBuffer } },
      { binding: 3, resource: { buffer: sb.tlasInstanceWorldToLocalBuffer } },
      { binding: 4, resource: { buffer: sb.tlasInstanceLocalToWorldBuffer } },
    ];
  }

  #buildSharedPipelineLayout(): SharedPipelineLayoutCandidate {
    // Group 0 — bindings 0..11 (both tiers) + 12..14 (lite texture slots) / 12..13 (full).
    // Mirrors material.wgsl.ts.
    let group0Entries: GPUBindGroupLayoutEntry[];
    if (this.#traceTier === 'full') {
      // D8.2: delegate to the shared extractor — keeps megakernel + reuse passes
      // byte-identical at group 0.
      group0Entries = this.#makeGroup0LayoutEntries();
    } else {
      // Lite tier: bindings 0-11 shared + B12 sampled texture bindings 12-14.
      // Uses maxSampledTexturesPerShaderStage (≥ 16 baseline), NOT storage buffers.
      group0Entries = [
        _tex(0),              // outputTexture (storage texture, write)
        _buf(1, _uniform),    // params (uniform)
        _buf(2, _rw),         // accumBuffer (read_write)
        _buf(3, _ro),         // positions
        _buf(4, _ro),         // indices
        _buf(5, _ro),         // triMaterialIds
        _buf(6, _ro),         // materials
        _buf(7, _ro),         // bvhNodes
        _buf(8, _ro),         // normals
        _tex(9),              // normalDepthTexture
        _tex(10),             // albedoTexture
        _varianceTex(11),     // varianceTexture (scalar r32float)
        _sampledTex(12),      // liteEnvTex    — RGBA32F env radiance+pdf
        _sampledTex(13),      // liteEnvCdfTex — RGBA32F (.r = CDF entry)
        _sampledTex(14),      // liteLightTex  — RGBA32F packed light data
      ];
    }
    const bindGroupLayout = this.#device.createBindGroupLayout({
      label: `vitrum.pt-webgpu.layout.group0.${this.#traceTier}`,
      entries: group0Entries,
    });
    const bindGroupLayouts: GPUBindGroupLayout[] = [bindGroupLayout];
    let bindGroupLayout1: GPUBindGroupLayout | null = null;
    let bindGroupLayout2: GPUBindGroupLayout | null = null;
    let bindGroupLayout3: GPUBindGroupLayout | null = null;

    if (this.#traceTier === 'full') {
      // Group 1 — 11 read-only storage buffers (analytics + env + area lights).
      bindGroupLayout1 = this.#device.createBindGroupLayout({
        label: 'vitrum.pt-webgpu.layout.group1.full',
        entries: Array.from({ length: 11 }, (_unused, binding) => _buf(binding, _ro)),
      });
      // Group 2 — TLAS table. BDPT stacks are invocation-private.
      bindGroupLayout2 = this.#device.createBindGroupLayout({
        label: 'vitrum.pt-webgpu.layout.group2.full',
        entries: [
          _buf(0, _ro), // tlasNodes
          _buf(1, _ro), // tlasInstanceIndices
          _buf(2, _ro), // tlasBlasRoots
          _buf(3, _ro), // tlasInstanceWorldToLocal
          _buf(4, _ro), // tlasInstanceLocalToWorld
        ],
      });
      // Group 3 — WS2 light-tree node buffer + P2 material textures (per-vertex
      // UVs/tangents, per-material descriptors, the baseColor texture_2d_array, a sampler).
      // A DEDICATED group so the lite tier (which never reaches this branch) carries
      // no group-3 layout, and so adding it leaves groups 0/1/2 byte-identical.
      // A4 — Group 3 extended with SPPM bindings 6/7/8 (photonCells + cellCounters
      // + sppmStats). Using group 3 instead of a new group 4 avoids requiring
      // maxBindGroups ≥ 5 (lavapipe only supports 4 bind groups). Placeholder
      // buffers are bound when SPPM is off; the gather code is guarded by
      // causticMode() == 2u so they are never accessed.
      const VIS = _vis();
      bindGroupLayout3 = this.#device.createBindGroupLayout({
        label: 'vitrum.pt-webgpu.layout.group3.full',
        entries: [
          _buf(0, _ro), // lightTree (read-only storage)
          _buf(1, _ro), // meshUvs (P2)
          _buf(2, _ro), // materialTexDescriptors (P2)
          { binding: 3, visibility: VIS, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // materialTextures sRGB (P2)
          { binding: 4, visibility: VIS, sampler: { type: 'filtering' } }, // materialTexSampler (P2)
          { binding: 5, visibility: VIS, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // materialTexturesLinear normal/scalar data maps (P2)
          // T1-6 — dedicated rgba16float emissive array (sampled as float 2d-array).
          { binding: 17, visibility: VIS, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // materialTexturesEmissive (HDR emissive)
          _buf(6, _rw), // A4: sppmPhotonCells (read_write storage)
          _buf(7, _rw), // A4: sppmCellCounters (read_write storage, atomic)
          _buf(8, _uniform), // A4: sppmStats (uniform)
          _buf(9, _rw), // A4-progressive: sppmPixelStats (read_write storage)
          _buf(10, _ro), // tangent.xyzw (authored/generated TBN handedness)
          _buf(11, _ro), // vertex color.rgba (glTF COLOR_0)
          ...(this.#cwbvhClosest
            ? [
                _buf(12, _ro), // CWBVH parent node bounds
                _buf(13, _ro), // CWBVH child bounds packed u16 pairs
                _buf(14, _ro), // CWBVH child metadata
                _buf(15, _ro), // CWBVH live child counts
                _buf(16, _ro), // TLAS BLAS roots remapped to CWBVH wide roots
              ]
            : []),
        ],
      });
      bindGroupLayouts.push(bindGroupLayout1, bindGroupLayout2, bindGroupLayout3);
    }

    const pipelineLayout = this.#device.createPipelineLayout({
      label: `vitrum.pt-webgpu.pipelineLayout.${this.#traceTier}`,
      bindGroupLayouts,
    });
    return {
      pipelineLayout,
      bindGroupLayout,
      bindGroupLayout1,
      bindGroupLayout2,
      bindGroupLayout3,
    };
  }

  ensurePipeline(): void {
    const layoutsReady =
      this.bindGroupLayout != null &&
      (this.#traceTier === 'lite' || (
        this.bindGroupLayout1 != null &&
        this.bindGroupLayout2 != null &&
        this.bindGroupLayout3 != null
      ));
    if (
      this.computePipeline != null &&
      this.paramsBuffer != null &&
      layoutsReady &&
      (!this.#bdpt || this.bdptCameraSplatResolvePipeline != null)
    ) {
      return;
    }

    const previousParamsBuffer = this.paramsBuffer;
    let paramsBuffer: GPUBuffer | null = null;
    try {
      paramsBuffer = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.params',
        size: FRAME_PARAMS_BYTE_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      // WS4 — every full-tier variant composes the volumetric random walk.
      // BDPT carries matching light/eye medium vertices, segment densities, and
      // transmittance, so enabling it does not remove volume transport.
      const samplingOpts = { sampling: this.#sampling, cwbvhClosest: this.#cwbvhClosest };
      const traceWgsl =
        this.#traceTier === 'lite'
          ? composePtWebgpuTraceLiteWgsl(samplingOpts)
          : composePtWebgpuTraceWgsl(this.#bdpt, samplingOpts);
      const module = this.#device.createShaderModule({
        label: `vitrum.pt-webgpu.pathTrace.${this.#traceTier}`,
        code: traceWgsl,
      });
      // Build the entire explicit-layout + pipeline cohort off to the side.
      // Publishing any member before the optional BDPT pipeline succeeds would
      // poison the readiness guard and make a transient creation failure
      // permanently non-retryable.
      const layouts = this.#buildSharedPipelineLayout();
      const computePipeline = this.#device.createComputePipeline({
        label: 'vitrum.pt-webgpu.pathTrace.pipeline',
        layout: layouts.pipelineLayout,
        compute: {
          module,
          entryPoint: 'main',
        },
      });
      const bdptCameraSplatResolvePipeline = this.#bdpt
        ? this.#device.createComputePipeline({
            label: 'vitrum.pt-webgpu.bdpt.cameraSplatResolve.pipeline',
            layout: layouts.pipelineLayout,
            compute: {
              module,
              entryPoint: 'bdptResolveCameraSplats',
            },
          })
        : null;
      this.paramsBuffer = paramsBuffer;
      this.computePipeline = computePipeline;
      this.bdptCameraSplatResolvePipeline =
        bdptCameraSplatResolvePipeline;
      this.bindGroupLayout = layouts.bindGroupLayout;
      this.bindGroupLayout1 = layouts.bindGroupLayout1;
      this.bindGroupLayout2 = layouts.bindGroupLayout2;
      this.bindGroupLayout3 = layouts.bindGroupLayout3;
      this.invalidateBindGroups();
      destroyGpuResourcesBestEffort([previousParamsBuffer], [paramsBuffer]);
    } catch (error) {
      destroyGpuResourcesBestEffort([paramsBuffer], [previousParamsBuffer]);
      throw error;
    }
  }

  // ── ReSTIR-PT reuse resource lifecycle (gated; full tier only) ───────────────

  /**
   * Ensure the two compact full-frame reservoirs (`Cur` scratch + `Prev`
   * spatial/history) and the 16 B/px resolved-result buffer. The method either
   * returns with an exact-size cohort or throws before publishing any candidate;
   * an explicitly requested ReSTIR mode is never silently disabled.
   */
  ensureReservoirBuffers(width: number, height: number): boolean {
    if (!this.#restirPtReuse) return false;
    if (
      !Number.isSafeInteger(width) || width < 1 || width > 0xffffffff ||
      !Number.isSafeInteger(height) || height < 1 || height > 0xffffffff
    ) {
      throw new RangeError(
        `[vitrum/pt-webgpu] ReSTIR-PT dimensions must be positive u32 integers (got ${width}×${height}).`,
      );
    }
    const px = width * height;
    const reservoirBytes = px * GpuResources.RESERVOIR_PT_HERO_BYTES;
    const resultBytes = px * 16;
    const totalBytes = reservoirBytes * 2 + resultBytes;
    if (
      !Number.isSafeInteger(reservoirBytes) ||
      !Number.isSafeInteger(resultBytes) ||
      !Number.isSafeInteger(totalBytes)
    ) {
      throw new RangeError(
        `[vitrum/pt-webgpu] ReSTIR-PT dimensions ${width}×${height} overflow a safe GPU allocation size.`,
      );
    }

    const limits = this.#device.limits as Partial<GPUSupportedLimits> | undefined;
    const finiteLimit = (value: number | undefined): number => (
      value != null && Number.isFinite(value) && value > 0
        ? value
        : Number.POSITIVE_INFINITY
    );
    const perBufferLimit = Math.min(
      GpuResources.RESTIR_PT_RESERVOIR_MAX_BYTES,
      finiteLimit(limits?.maxStorageBufferBindingSize),
      finiteLimit(limits?.maxBufferSize),
    );
    const exceedsPerBuffer = reservoirBytes > perBufferLimit
      || resultBytes > perBufferLimit;
    const exceedsTotal = totalBytes > GpuResources.RESTIR_PT_TOTAL_MAX_BYTES;
    if (exceedsPerBuffer || exceedsTotal) {
      const message =
        `[vitrum/pt-webgpu] ReSTIR-PT at ${width}×${height} requires ` +
        `${(reservoirBytes / (1024 * 1024)).toFixed(2)} MiB per reservoir and ` +
        `${(totalBytes / (1024 * 1024)).toFixed(2)} MiB total; limits are ` +
        `${(perBufferLimit / (1024 * 1024)).toFixed(2)} MiB per binding and ` +
        `${(GpuResources.RESTIR_PT_TOTAL_MAX_BYTES / (1024 * 1024)).toFixed(0)} MiB total.`;
      this.#emitResourceCeilingWarning('restirPtReservoir', {
        code: 'pt-webgpu.restir-pt-reservoir-ceiling',
        backend: 'pt-webgpu',
        phase: 'renderFrame',
        method: 'renderFrame',
        message,
        details: {
          width,
          height,
          bytesPerPixel: GpuResources.RESERVOIR_PT_HERO_BYTES,
          reservoirCount: 2,
          reservoirBytes,
          resultBytes,
          totalBytes,
          perBufferLimit,
          totalLimit: GpuResources.RESTIR_PT_TOTAL_MAX_BYTES,
          fallback: 'throw',
        },
      });
      throw new RangeError(message);
    }

    const ready =
      this.#rsvr.rptReservoirCur != null &&
      this.#rsvr.rptReservoirPrev != null &&
      this.#rsvr.rptResultBuffer != null &&
      this.#rsvr.rptParamsBuffer != null &&
      this.#rsvr.rptReservoirByteSize === reservoirBytes &&
      this.#rsvr.rptResultByteSize === resultBytes;
    if (ready) return true;

    const previous = [
      this.#rsvr.rptReservoirCur,
      this.#rsvr.rptReservoirPrev,
      this.#rsvr.rptResultBuffer,
      this.#rsvr.rptParamsBuffer,
    ];
    const usage =
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const candidates: GPUBuffer[] = [];
    try {
      const cur = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.restirPt.reservoir.cur',
        size: reservoirBytes,
        usage,
      });
      candidates.push(cur);
      const prev = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.restirPt.reservoir.prev',
        size: reservoirBytes,
        usage,
      });
      candidates.push(prev);
      const result = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.restirPt.result',
        size: resultBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      candidates.push(result);
      const params = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.restirPt.params',
        size: GpuResources.RESTIR_PT_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      candidates.push(params);

      const enc = this.#device.createCommandEncoder({
        label: 'vitrum.pt-webgpu.restirPt.clear',
      });
      enc.clearBuffer(cur);
      enc.clearBuffer(prev);
      this.#device.queue.submit([enc.finish()]);

      this.#rsvr.rptReservoirCur = cur;
      this.#rsvr.rptReservoirPrev = prev;
      this.#rsvr.rptResultBuffer = result;
      this.#rsvr.rptParamsBuffer = params;
      this.#rsvr.rptReservoirByteSize = reservoirBytes;
      this.#rsvr.rptResultByteSize = resultBytes;
      this.#rsvr.rptProducerGroup0 = null;
      this.#rsvr.rptTemporalGroup0 = null;
      this.#rsvr.rptSpatialGroup0 = null;
      this.#rsvr.rptResolveGroup0 = null;
      destroyGpuResourcesBestEffort(previous, candidates);
      return true;
    } catch (error) {
      destroyGpuResourcesBestEffort(candidates, previous);
      throw error;
    }
  }

  /**
   * Write the RestirPtParams UBO (mClamp).
   * No-op only when reuse is OFF. Once requested, an absent params buffer or an
   * unrepresentable value is an invariant failure rather than a silent skip.
   * Called per-frame by the engine before dispatch.
   */
  writeReservoirParams(mClamp: number): void {
    if (!this.#restirPtReuse) return;
    if (this.#rsvr.rptParamsBuffer == null) {
      throw new Error(
        '[vitrum/pt-webgpu] ReSTIR-PT params buffer is absent after reuse was enabled.',
      );
    }
    if (!Number.isInteger(mClamp) || mClamp < 1 || mClamp > 4095) {
      throw new RangeError(
        `[vitrum/pt-webgpu] ReSTIR-PT mClamp must be an integer in 1..4095 (got ${mClamp}).`,
      );
    }
    const ubo = new ArrayBuffer(GpuResources.RESTIR_PT_PARAMS_BYTES);
    const u = new Uint32Array(ubo);
    u[0] = mClamp;
    this.#device.queue.writeBuffer(this.#rsvr.rptParamsBuffer, 0, ubo);
  }

  /**
   * Build the extended group-0 bind-group layout for the reuse passes: the
   * megakernel's full-tier group-0 bindings (0..13, IDENTICAL to #buildShared-
   * PipelineLayout's group0) PLUS the relocated reuse bindings (20..25). This is
   * the one group the reuse passes carry their own resources in; groups 1/2/3 are
   * the megakernel's existing explicit layouts (reused verbatim), so a reuse
   * pipeline's layout is [g0', g1, g2, g3] — exactly 4 groups, portable on a
   * guaranteed maxBindGroups = 4 adapter. Full tier only (the reuse passes
   * statically use the full-tier scene groups).
   */
  #buildReservoirGroup0Layout(): GPUBindGroupLayout {
    const B = RPT_GROUP0_BINDING_BASE; // 20
    // D8.2: Megakernel group-0 (0..13) sourced from #makeGroup0LayoutEntries() —
    // byte-for-byte the same as the trace layout's group0 (full tier) so the same
    // group-0 scene/G-buffer resources bind.  Previously duplicated inline; using
    // the shared extractor guarantees the two layouts can never diverge.
    const entries: GPUBindGroupLayoutEntry[] = [
      ...this.#makeGroup0LayoutEntries(),
      // Relocated reuse bindings (20..25). The composed WGSL keeps rpt_resPrev /
      // rpt_resResolved as `var<storage, read>` (the naga-gap fix is to MONOMORPHISE
      // the reservoir helpers so the read-only global is indexed DIRECTLY rather
      // than passed as a storage pointer — see restirPtCompose.wgsl.ts
      // monomorphiseReservoirHelpers; the access modes are unchanged). So the layout
      // matches the shader access modes exactly: rpt_resPrev (b22) is read-only.
      _buf(B + 0, _rw),      // rpt_reservoirOut  (producer write)
      _buf(B + 1, _rw),      // rpt_resCurrent    (temporal in/out; spatial reads same slot as `read`)
      _buf(B + 2, _ro),      // rpt_resPrev       (temporal history, read-only)
      _buf(B + 3, _rw),      // rpt_result        (resolve write)
      _buf(B + 4, _uniform), // rptParams
      _buf(B + 5, _rw),      // rpt_resSpatial    (spatial write; resolve reads same slot as `read`)
    ];
    return this.#device.createBindGroupLayout({
      label: 'vitrum.pt-webgpu.restirPt.layout.group0',
      entries,
    });
  }

  /**
   * Lazily build the four reuse compute pipelines + their shared 4-group layout
   * [g0', g1, g2, g3]. Requires `ensurePipeline()` to have run first (so the
   * megakernel's group-1/2/3 explicit layouts exist — the reuse passes reuse them
   * for the scene/TLAS/light bindings their trace + NEE statically use). No-op
   * when reuse is OFF, on the lite tier, or already built. Each entry point gets
   * its own module (the combined unit has duplicate @group/@binding slots — see
   * restirPtCompose.wgsl.ts), composed standalone with the reuse bindings
   * relocated into group 0.
   */
  ensureReservoirPipelines(): void {
    if (!this.#restirPtReuse) return;
    if (
      this.#rsvr.rptGroup0Layout != null &&
      this.#rsvr.rptProducerPipeline != null &&
      this.#rsvr.rptTemporalPipeline != null &&
      this.#rsvr.rptSpatialPipeline != null &&
      this.#rsvr.rptResolvePipeline != null &&
      this.#rsvr.rptCompositePipeline != null
    ) return;
    if (
      this.bindGroupLayout1 == null ||
      this.bindGroupLayout2 == null ||
      this.bindGroupLayout3 == null
    ) {
      // ensurePipeline() must build the full-tier group-1/2/3 layouts first.
      throw new Error(
        'ensureReservoirPipelines: full-tier group-1/2/3 layouts missing — call ensurePipeline() first.',
      );
    }
    const group0Layout = this.#buildReservoirGroup0Layout();
    const pipelineLayout = this.#device.createPipelineLayout({
      label: 'vitrum.pt-webgpu.restirPt.pipelineLayout',
      bindGroupLayouts: [
        group0Layout,
        this.bindGroupLayout1,
        this.bindGroupLayout2,
        this.bindGroupLayout3,
      ],
    });
    const mk = (label: string, code: string, entryPoint: string): GPUComputePipeline => {
      const module = this.#device.createShaderModule({ label, code });
      return this.#device.createComputePipeline({
        label,
        layout: pipelineLayout,
        compute: { module, entryPoint },
      });
    };
    // Candidate-first publication is essential here: a transient failure in a
    // later pass must not leave producer non-null and poison the retry guard.
    const producerPipeline = mk(
      'vitrum.pt-webgpu.restirPt.producer',
      composeRestirPtProducerWgsl({ sampling: this.#sampling }),
      'restirPtProduce',
    );
    const temporalPipeline = mk(
      'vitrum.pt-webgpu.restirPt.temporal',
      composeRestirPtTemporalWgsl({ sampling: this.#sampling }),
      'restirPtTemporal',
    );
    const spatialPipeline = mk(
      'vitrum.pt-webgpu.restirPt.spatial',
      composeRestirPtSpatialWgsl({ sampling: this.#sampling }),
      'restirPtSpatial',
    );
    const resolvePipeline = mk(
      'vitrum.pt-webgpu.restirPt.resolve',
      composeRestirPtResolveWgsl({ sampling: this.#sampling }),
      'restirPtResolve',
    );
    // A1 — the COMPOSITE megakernel uses the SAME [g0', g1, g2, g3] layout (it reads
    // rpt_result at the relocated group-0 binding 23 + the scene groups). Composed
    // for this engine's BDPT mode (matches the default megakernel's SSS/BDPT gate).
    const compositePipeline = mk(
      'vitrum.pt-webgpu.restirPt.compositeMegakernel',
      composePtWebgpuCompositeTraceWgsl(this.#bdpt, {
        sampling: this.#sampling,
        cwbvhClosest: this.#cwbvhClosest,
      }),
      'main',
    );

    this.#rsvr.rptGroup0Layout = group0Layout;
    this.#rsvr.rptProducerPipeline = producerPipeline;
    this.#rsvr.rptTemporalPipeline = temporalPipeline;
    this.#rsvr.rptSpatialPipeline = spatialPipeline;
    this.#rsvr.rptResolvePipeline = resolvePipeline;
    this.#rsvr.rptCompositePipeline = compositePipeline;
    this.#rsvr.rptProducerGroup0 = null;
    this.#rsvr.rptTemporalGroup0 = null;
    this.#rsvr.rptSpatialGroup0 = null;
    this.#rsvr.rptResolveGroup0 = null;
  }

  /**
   * Build (and cache) the per-pass reuse group-0 bind groups. Each provides the
   * megakernel group-0 scene/G-buffer resources (IDENTICAL to the trace group 0)
   * PLUS the reuse bindings (20..25). Four distinct groups keep each pass's
   * bindings non-aliasing: producer writes Cur at b20; temporal reads/writes Cur
   * at b21 and reads Prev at b22; spatial reads Cur at b21 and writes Prev at
   * b25; resolve reads Prev at b25 and writes result at b23. WebGPU tracks every
   * resource in the bound explicit layout for the dispatch, including entries
   * unused by that pass's shader. Therefore unused b22 slots bind a scene buffer
   * already used read-only, never either reservoir or the writable result.
   * Returns nothing; the engine reads the cached groups off this struct.
   * Idempotent.
   *
   * The pass sequence provides the inter-pass visibility boundary. Cur is the
   * producer/temporal scratch reservoir; Prev is both temporal history and the
   * spatial output/resolve input, so it becomes the next frame's history without
   * an end-of-frame swap.
   */
  buildReservoirBindGroups(sb: UploadedSceneBuffers): void {
    if (!this.#restirPtReuse) return;
    if (this.#rsvr.rptGroup0Layout == null) {
      throw new Error(
        '[vitrum/pt-webgpu] ReSTIR-PT group-0 layout is absent after pipeline initialization.',
      );
    }
    if (
      this.#rsvr.rptProducerGroup0 != null &&
      this.#rsvr.rptTemporalGroup0 != null &&
      this.#rsvr.rptSpatialGroup0 != null &&
      this.#rsvr.rptResolveGroup0 != null
    ) {
      return;
    }
    if (
      this.#rsvr.rptReservoirCur == null ||
      this.#rsvr.rptReservoirPrev == null ||
      this.#rsvr.rptResultBuffer == null ||
      this.#rsvr.rptParamsBuffer == null ||
      this.accumView == null ||
      this.normalDepthView == null ||
      this.albedoView == null ||
      this.varianceView == null ||
      this.motionVectorsView == null ||
      this.varianceMomentsBuffer == null ||
      (this.#bdpt && this.bdptCameraSplatBuffer == null) ||
      this.paramsBuffer == null
    ) {
      throw new Error(
        '[vitrum/pt-webgpu] ReSTIR-PT bind-group resources are incomplete after requested-mode setup.',
      );
    }
    const B = RPT_GROUP0_BINDING_BASE;
    // D8.2: Shared megakernel group-0 scene/G-buffer entries (0..13, plus
    // BDPT-only binding 14) sourced from
    // #makeGroup0BindGroupEntries() — guaranteed to match the layout produced by
    // #makeGroup0LayoutEntries() that #buildReservoirGroup0Layout() also uses.
    const sceneG0: GPUBindGroupEntry[] = [
      ...this.#makeGroup0BindGroupEntries(sb),
      { binding: B + 4, resource: { buffer: this.#rsvr.rptParamsBuffer } },
    ];
    const makeGroup = (
      label: string,
      out: GPUBuffer,
      current: GPUBuffer,
      historyRead: GPUBuffer,
      spatialOutput: GPUBuffer,
    ): GPUBindGroup => this.#device.createBindGroup({
      label,
      layout: this.#rsvr.rptGroup0Layout!,
      entries: [
        ...sceneG0,
        { binding: B + 0, resource: { buffer: out } },
        { binding: B + 1, resource: { buffer: current } },
        { binding: B + 2, resource: { buffer: historyRead } },
        { binding: B + 3, resource: { buffer: this.#rsvr.rptResultBuffer! } },
        { binding: B + 5, resource: { buffer: spatialOutput } },
      ],
    });
    const cur = this.#rsvr.rptReservoirCur;
    const prev = this.#rsvr.rptReservoirPrev;
    const readOnlyPlaceholder = sb.positionsBuffer;
    // Separate groups keep every explicit-layout read/write binding non-aliasing,
    // not merely the bindings statically referenced by the shader. Spatial reads
    // Cur at b21 and writes Prev at b25; resolve then reads Prev at b25, which is
    // already next frame's b22 temporal history. For passes that do not consume
    // b22, the positions buffer is a compatible read-only placeholder (and is
    // already bound read-only in the inherited scene portion of this same group).
    // Publish the cohort only after every per-pass group was created. A
    // transient failure can then be retried without a partial-cache false hit.
    const producerGroup =
      makeGroup(
        'vitrum.pt-webgpu.restirPt.bindgroup0.producer',
        cur,
        cur,
        readOnlyPlaceholder,
        cur,
      );
    const temporalGroup =
      makeGroup('vitrum.pt-webgpu.restirPt.bindgroup0.temporal', cur, cur, prev, cur);
    const spatialGroup =
      makeGroup(
        'vitrum.pt-webgpu.restirPt.bindgroup0.spatial',
        cur,
        cur,
        readOnlyPlaceholder,
        prev,
      );
    const resolveGroup =
      makeGroup(
        'vitrum.pt-webgpu.restirPt.bindgroup0.resolve',
        cur,
        cur,
        readOnlyPlaceholder,
        prev,
      );
    this.#rsvr.rptProducerGroup0 = producerGroup;
    this.#rsvr.rptTemporalGroup0 = temporalGroup;
    this.#rsvr.rptSpatialGroup0 = spatialGroup;
    this.#rsvr.rptResolveGroup0 = resolveGroup;
  }

  /** Tear down all ReSTIR-PT reuse resources. Called from dispose(). */
  #disposeReservoirResources(): void {
    // D8.1: delegate to the ReservoirResources sub-object.
    this.#rsvr.dispose();
  }

  /**
   * Build (and cache) the path-trace bind group(s) from the current accum views,
   * params buffer, pipeline layout, and the supplied scene buffers. Returns
   * group 0 (the always-present group). Groups 1/2 are
   * only created on the `full` tier and are read back off this struct by the
   * caller. Idempotent: if group 0 is already cached, returns it unchanged.
   *
   * Callers must have already run `ensureAccumResources` + `ensurePipeline` and
   * validated that the views / pipeline / layout / params / scene buffers are
   * non-null (renderFrame's preconditions handle this).
   */
  buildBindGroups(sb: UploadedSceneBuffers): GPUBindGroup {
    const complete =
      this.pathTraceBindGroup != null &&
      (this.#traceTier === 'lite' || (
        this.pathTraceBindGroup1 != null &&
        this.pathTraceBindGroup2 != null &&
        this.pathTraceBindGroup3 != null
      ));
    if (complete) return this.pathTraceBindGroup!;
    // Bindings 0–11 are shared between the lite and full tiers; the tiers then
    // DIVERGE at 12+ (lite: B12 sampled light/env textures at 12–14; full:
    // motion-vectors storage texture at 12 + variance-moments buffer at 13).
    // Trust-audit F1 (2026-06-10): the full entries previously spread the WHOLE
    // lite array (duplicate bindings 12/13 + a stray 14) — WebGPU rejected the
    // bind group, so EVERY full-tier render crashed before frame 1. The mock
    // device suite validated nothing; the lavapipe behavioral harness caught it.
    // Lite-tier group-0: bindings 0-11 (shared geometry/accum) + 12-14 (B12 textures).
    const liteEntries: GPUBindGroupEntry[] = [
      ...this.#makeGroup0SharedPrefixEntries(sb),
      // B12 — lite-tier texture bindings 12–14 (sampled, not storage).
      { binding: 12, resource: this.liteEnvTextureView! },
      { binding: 13, resource: this.liteEnvCdfTextureView! },
      { binding: 14, resource: this.liteLightTextureView! },
    ];
    // D8.2: Full-tier group-0: delegate to the shared extractor — the same
    // 0..13 definition used by the ReSTIR-PT reuse group-0.
    const fullGroup0Entries: GPUBindGroupEntry[] = this.#makeGroup0BindGroupEntries(sb);
    const fullGroup1Entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: sb.analyticHeadersBuffer } },
      { binding: 1, resource: { buffer: sb.analyticParamsBuffer } },
      { binding: 2, resource: { buffer: sb.analyticLocalToWorldBuffer } },
      { binding: 3, resource: { buffer: sb.analyticWorldToLocalBuffer } },
      { binding: 4, resource: { buffer: sb.environmentMapTexelsBuffer } },
      { binding: 5, resource: { buffer: sb.environmentMapCdfBuffer } },
      { binding: 6, resource: { buffer: sb.pointLightsBuffer } },
      { binding: 7, resource: { buffer: sb.spotLightsBuffer } },
      { binding: 8, resource: { buffer: sb.rectAreaLightsBuffer } },
      { binding: 9, resource: { buffer: sb.meshAreaLightsBuffer } },
      // N-directional: directionalLights storage buffer (group 1 binding 10).
      { binding: 10, resource: { buffer: sb.directionalLightsBuffer } },
    ];
    const fullGroup2Entries: GPUBindGroupEntry[] = this.#traceTier === 'full'
      ? this.#makeGroup2BindGroupEntries(sb)
      : [];
    const bindGroup = this.#device.createBindGroup({
      label: `vitrum.pt-webgpu.pathTrace.bindgroup0.${this.#traceTier}`,
      layout: this.bindGroupLayout!,
      entries: this.#traceTier === 'lite' ? liteEntries : fullGroup0Entries,
    });
    let bindGroup1: GPUBindGroup | null = null;
    let bindGroup2: GPUBindGroup | null = null;
    let bindGroup3: GPUBindGroup | null = null;
    if (this.#traceTier === 'full') {
      // Built against the explicit layouts used by the path-trace pipeline.
      bindGroup1 = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.pathTrace.bindgroup1.full',
        layout: this.bindGroupLayout1!,
        entries: fullGroup1Entries,
      });
      bindGroup2 = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.pathTrace.bindgroup2.full',
        layout: this.bindGroupLayout2!,
        entries: fullGroup2Entries,
      });
      bindGroup3 = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.pathTrace.bindgroup3.full',
        layout: this.bindGroupLayout3!,
        entries: [
          { binding: 0, resource: { buffer: sb.lightTreeBuffer } },
          { binding: 1, resource: { buffer: sb.uvsBuffer } },
          { binding: 2, resource: { buffer: sb.materialTexDescriptorsBuffer } },
          { binding: 3, resource: sb.materialTextureView },
          { binding: 4, resource: sb.materialTextureSampler },
          { binding: 5, resource: sb.materialLinearTextureView },
          // T1-6 — dedicated rgba16float emissive array view (HDR emissive).
          { binding: 17, resource: sb.materialEmissiveTextureView },
          // A4 — SPPM photon hash-grid (bindings 6/7/8) + A4-progressive per-pixel
          // stats (binding 9). Placeholder buffers are bound when SPPM is off so the
          // layout slot is satisfied; the gather code in caustic.wgsl.ts is guarded
          // by causticMode() == 2u so the placeholders are never accessed.
          { binding: 6, resource: { buffer: this.#sppm.sppmPhotonCellsBuffer! } },
          { binding: 7, resource: { buffer: this.#sppm.sppmCellCountersBuffer! } },
          { binding: 8, resource: { buffer: this.#sppm.sppmStatsBuffer! } },
          { binding: 9, resource: { buffer: this.#sppm.sppmPixelStatsBuffer! } },
          { binding: 10, resource: { buffer: sb.tangentsBuffer } },
          { binding: 11, resource: { buffer: sb.colorsBuffer } },
          ...(this.#cwbvhClosest
            ? [
                { binding: 12, resource: { buffer: sb.cwbvhNodeBoundsBuffer } },
                { binding: 13, resource: { buffer: sb.cwbvhChildBoundsPackedBuffer } },
                { binding: 14, resource: { buffer: sb.cwbvhChildMetaBuffer } },
                { binding: 15, resource: { buffer: sb.cwbvhChildCountBuffer } },
                { binding: 16, resource: { buffer: sb.cwbvhTlasBlasRootsBuffer } },
              ]
            : []),
        ],
      });
    }

    // Commit all four groups together. If a later createBindGroup throws, the
    // previous complete cohort remains published and the next call retries.
    this.pathTraceBindGroup = bindGroup;
    this.pathTraceBindGroup1 = bindGroup1;
    this.pathTraceBindGroup2 = bindGroup2;
    this.pathTraceBindGroup3 = bindGroup3;
    return bindGroup;
  }

  /**
   * Progressive walkaround→PT handoff (P8) — seed the accumulation buffers from
   * `seedTex` as a DECAYING PRIOR of virtual weight `weight`. Writes
   * `accumBuffer[i] = vec4f(seedRGB·W, W)` and
   * `varianceMomentsBuffer[i] = vec3(lum·W, lum²·W, W)` (full tier; lite tier
   * discards the variance write to a placeholder). The converged mean is
   * UNCHANGED because the seed's influence is W/(W+M) → 0 as M real samples land
   * (see seedBlit.wgsl.ts header for the derivation).
   *
   * MUST be called AFTER `ensureAccumResources` (so the accum buffers exist) and
   * AFTER `clearAccumBuffer`/`reset` (so the seed isn't subsequently zeroed). A
   * no-op if the accum buffer is absent. `width`/`height` are the accum
   * (destination) dims; `seedTex` may be any size (bilinearly resampled).
   *
   * Does NOT touch the engine's `#samplesAccumulated`: `weight` is a
   * virtual-sample prior, distinct from the real-SPP counter — the converged-mean
   * math depends on that separation (the caller in `index.ts` enforces it).
   */
  seedAccumBuffer(seedTex: GPUTexture, weight: number, width: number, height: number): void {
    if (this.accumBuffer == null) return;
    const W = Math.max(0, weight);

    const placeholderByteSize = Math.max(this.accumBufferByteSize, 16);
    const needsPlaceholder = this.varianceMomentsBuffer == null;
    const previousPipeline = this.#present.seedBlitPipeline;
    const previousSampler = this.#present.seedBlitSampler;
    const previousParamsBuffer = this.#present.seedBlitParamsBuffer;
    const previousPlaceholder = this.#present.seedBlitVarPlaceholder;
    const cohortReady =
      previousPipeline != null &&
      previousSampler != null &&
      previousParamsBuffer != null &&
      (!needsPlaceholder ||
        (previousPlaceholder != null &&
          this.#present.seedBlitVarPlaceholderByteSize === placeholderByteSize));

    let pipeline = previousPipeline;
    let sampler = previousSampler;
    let paramsBuffer = previousParamsBuffer;
    let placeholder = previousPlaceholder;
    const candidateBuffers: GPUBuffer[] = [];

    try {
      // If any member is missing (or a lite-tier placeholder is stale after a
      // resize), build a complete replacement cohort locally. Publishing fields
      // piecemeal makes a later bind-group/submit failure poison every retry.
      if (!cohortReady) {
        const module = this.#device.createShaderModule({
          label: 'vitrum.pt-webgpu.seedBlit',
          code: PT_WEBGPU_SEED_BLIT_WGSL,
        });
        pipeline = this.#device.createComputePipeline({
          label: 'vitrum.pt-webgpu.seedBlit.pipeline',
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        });
        // Filtering sampler so a differently-sized seed is bilinearly resampled
        // onto the accum grid; clamp so edge UVs don't wrap.
        sampler = this.#device.createSampler({
          label: 'vitrum.pt-webgpu.seedBlit.sampler',
          magFilter: 'linear',
          minFilter: 'linear',
          addressModeU: 'clamp-to-edge',
          addressModeV: 'clamp-to-edge',
        });
        paramsBuffer = this.#device.createBuffer({
          label: 'vitrum.pt-webgpu.seedBlit.params',
          size: 32, // vec4u seedDim (16) + vec4f seedWeight (16)
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        candidateBuffers.push(paramsBuffer);
        placeholder = needsPlaceholder
          ? this.#device.createBuffer({
              label: 'vitrum.pt-webgpu.seedBlit.varPlaceholder',
              size: placeholderByteSize,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            })
          : null;
        if (placeholder != null) candidateBuffers.push(placeholder);
      }

      // Narrowing follows from cohortReady or the complete candidate build.
      if (pipeline == null || sampler == null || paramsBuffer == null) {
        throw new Error('pt-webgpu: incomplete seed-blit resource cohort');
      }
      const varBuffer = this.varianceMomentsBuffer ?? placeholder;
      if (varBuffer == null) {
        throw new Error('pt-webgpu: seed-blit variance target is unavailable');
      }

      // SeedParams UBO: seedDim (accum dims) as uvec4, seedWeight as vec4f.
      const ubo = new ArrayBuffer(32);
      new Uint32Array(ubo, 0, 4).set([width >>> 0, height >>> 0, 0, 0]);
      new Float32Array(ubo, 16, 4).set([W, 0, 0, 0]);
      this.#device.queue.writeBuffer(paramsBuffer, 0, ubo);

      const bindGroup = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.seedBlit.bindgroup',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: seedTex.createView() },
          { binding: 2, resource: sampler },
          { binding: 3, resource: { buffer: this.accumBuffer } },
          { binding: 4, resource: { buffer: varBuffer } },
        ],
      });

      const encoder = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.seedBlit.encoder' });
      const pass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.seedBlit.pass' });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      pass.end();
      this.#device.queue.submit([encoder.finish()]);

      if (!cohortReady) {
        this.#present.seedBlitPipeline = pipeline;
        this.#present.seedBlitSampler = sampler;
        this.#present.seedBlitParamsBuffer = paramsBuffer;
        this.#present.seedBlitVarPlaceholder = placeholder;
        this.#present.seedBlitVarPlaceholderByteSize = needsPlaceholder
          ? placeholderByteSize
          : 0;
        destroyGpuResourcesBestEffort(
          [previousParamsBuffer, previousPlaceholder],
          candidateBuffers,
        );
      }
    } catch (error) {
      destroyGpuResourcesBestEffort(
        candidateBuffers,
        [previousParamsBuffer, previousPlaceholder],
      );
      throw error;
    }
  }

  /**
   * D8.5 — Returns `true` when all GPU resources required for `renderFrame` are
   * in a valid (non-null) state.  Codifies the 12 null-guard conditions that
   * `index.ts renderFrame` tests inline immediately after `ensurePipeline()`.
   *
   * The 13th condition in `renderFrame` (`this.#sceneBuffers == null`) lives on the
   * engine, not on GpuResources, so it is NOT included here.  Callers MUST check
   * both this method AND that the engine's scene buffers are present before
   * dispatching.
   *
   * MAINTENANCE: This method MUST be kept in sync with the `buildBindGroups` resource
   * set.  Any new resource that becomes required for a valid dispatch must be added
   * here simultaneously.
   */
  isReadyToRender(): boolean {
    if (
      this.accumView == null ||
      this.normalDepthView == null ||
      this.albedoView == null ||
      this.varianceView == null ||
      this.accumBuffer == null ||
      this.paramsBuffer == null ||
      this.computePipeline == null ||
      this.bindGroupLayout == null
    ) {
      return false;
    }
    if (this.#traceTier === 'full') {
      if (this.motionVectorsView == null || this.varianceMomentsBuffer == null) {
        return false;
      }
      if (
        this.#bdpt &&
        (
          this.bdptCameraSplatBuffer == null ||
          this.bdptCameraSplatResolvePipeline == null
        )
      ) {
        return false;
      }
    } else {
      // B12 — lite-tier texture views must be present before rendering.
      if (
        this.liteEnvTextureView == null ||
        this.liteEnvCdfTextureView == null ||
        this.liteLightTextureView == null
      ) {
        return false;
      }
    }
    return true;
  }

  /** Invalidate the cached bind groups (scene-buffer / accum-view recreation). */
  invalidateBindGroups(): void {
    this.pathTraceBindGroup = null;
    this.pathTraceBindGroup1 = null;
    this.pathTraceBindGroup2 = null;
    this.pathTraceBindGroup3 = null;
    // The reuse bind groups reference the same scene buffers + accum views, so a
    // scene-buffer / accum-view recreation invalidates them too.
    this.#rsvr.rptProducerGroup0 = null;
    this.#rsvr.rptTemporalGroup0 = null;
    this.#rsvr.rptSpatialGroup0 = null;
    this.#rsvr.rptResolveGroup0 = null;
  }

  // ── SPPM photon-map lifecycle (A4) ───────────────────────────────────────────

  /**
   * Predicate: would the full SPPM photon-cells allocation exceed the effective
   * ceiling (min of the static SPPM ceiling and the live device limits)?  This
   * mirrors the ceiling check inside {@link ensureSppmBuffers} but is side-effect
   * free, so the engine can decide the caustic-mode fallback BEFORE it packs the
   * per-frame params buffer (V2-1: the packed `causticMode` must reflect the
   * manifold-nee fallback, not the placeholder photon path).  Returns `false` on
   * the lite tier (SPPM never runs there) and when SPPM is already allocated.
   */
  sppmWouldExceedCeiling(): boolean {
    if (this.#traceTier !== 'full') return false;
    if (
      this.#sppm.sppmBuffersReady &&
      this.#sppm.sppmPhotonCellsBuffer != null &&
      this.#sppm.sppmCellCountersBuffer != null &&
      this.#sppm.sppmStatsBuffer != null &&
      this.#sppm.sppmPhotonPipeline != null
    ) {
      return false;
    }
    const deviceMaxBuffer = this.#device.limits?.maxBufferSize ?? 256 * 1024 * 1024;
    const deviceMaxBinding =
      this.#device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
    const sppmCeiling = Math.min(SPPM_PHOTON_CELLS_MAX_BYTES, deviceMaxBuffer, deviceMaxBinding);
    return SPPM_PHOTON_CELLS_BYTES > sppmCeiling;
  }

  /**
   * (Re)allocate or ensure the SPPM buffers and pipeline exist.  Returns `true`
   * when the full photon-cells + counters + stats buffers are allocated and the
   * photon-emission pipeline is built; `false` on the lite tier or when the
   * allocation would exceed the ceiling.
   *
   * SPPM bindings live in group-3 (bindings 6/7/8), appended to the existing
   * light-tree / material-texture entries (0–5).  No new bind group is needed;
   * only the three buffer handles in group-3 are managed here.
   *
   * A 64-byte placeholder buffer is always created when SPPM is off so the
   * group-3 layout slots 6/7/8 are satisfied without allocating the real data.
   * The gather code in caustic.wgsl.ts is guarded by `causticMode() == 2u` so
   * the placeholder buffers are never accessed on the GPU.
   */
  ensureSppmBuffers(sppmActive: boolean): boolean {
    if (this.#traceTier !== 'full') return false;
    // The group-3 layout (which includes SPPM bindings 6/7/8) is built in
    // #buildSharedPipelineLayout via ensurePipeline. We don't need the layout
    // here — just ensure the buffer handles exist so buildBindGroups can bind them.
    // When SPPM is not active, ensure at least a placeholder buffer for the
    // layout. Trust-audit F1b (2026-06-10): the placeholders were 16/16/32 bytes
    // but the WGSL runtime-sized arrays impose a min-binding-size of ONE ELEMENT
    // (photonCells element = 48 B) — WebGPU rejected the group-3 bind group, so
    // EVERY full-tier render with causticStrategy != 'photon-map' failed. This is
    // the 4th occurrence of the min-binding-size placeholder class (ea88803 /
    // 0bedd92 / the 32B BVHNode dummies); placeholders sized 64 B to clear any
    // current element stride with headroom.
    if (!sppmActive) {
      const placeholdersReady =
        this.#sppm.sppmPhotonCellsBuffer != null &&
        this.#sppm.sppmCellCountersBuffer != null &&
        this.#sppm.sppmStatsBuffer != null &&
        this.#sppm.sppmPixelStatsBuffer != null;
      if (!placeholdersReady) {
        const previous = [
          this.#sppm.sppmPhotonCellsBuffer,
          this.#sppm.sppmCellCountersBuffer,
          this.#sppm.sppmStatsBuffer,
          this.#sppm.sppmPixelStatsBuffer,
        ];
        const candidates: GPUBuffer[] = [];
        try {
          const photonCells = this.#device.createBuffer({
            label: 'vitrum.pt-webgpu.sppm.photonCells.placeholder',
            size: 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          candidates.push(photonCells);
          const cellCounters = this.#device.createBuffer({
            label: 'vitrum.pt-webgpu.sppm.cellCounters.placeholder',
            size: 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          candidates.push(cellCounters);
          const stats = this.#device.createBuffer({
            label: 'vitrum.pt-webgpu.sppm.stats.placeholder',
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          candidates.push(stats);
          const pixelStats = this.#device.createBuffer({
            label: 'vitrum.pt-webgpu.sppm.pixelStats.placeholder',
            size: 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          candidates.push(pixelStats);

          this.#sppm.sppmPhotonCellsBuffer = photonCells;
          this.#sppm.sppmCellCountersBuffer = cellCounters;
          this.#sppm.sppmStatsBuffer = stats;
          this.#sppm.sppmPixelStatsBuffer = pixelStats;
          this.#sppm.sppmPixelStatsWidth = 0;
          this.#sppm.sppmPixelStatsHeight = 0;
          this.#sppm.sppmBuffersReady = false;
          this.invalidateGroup3BindGroup();
          destroyGpuResourcesBestEffort(previous, candidates);
        } catch (error) {
          destroyGpuResourcesBestEffort(candidates, previous);
          throw error;
        }
      }
      return false;
    }
    // Full allocation path.
    const fullCohortReady =
      this.#sppm.sppmBuffersReady &&
      this.#sppm.sppmPhotonCellsBuffer != null &&
      this.#sppm.sppmCellCountersBuffer != null &&
      this.#sppm.sppmStatsBuffer != null &&
      this.#sppm.sppmPhotonPipeline != null;
    if (fullCohortReady) return true;

    // Validate the exact 3 MiB record buffer against both live WebGPU limits
    // before creating it. The caller treats false as an explicit readiness
    // failure and throws before packing caustic mode 2.
    const deviceMaxBuffer = this.#device.limits?.maxBufferSize ?? 256 * 1024 * 1024;
    const deviceMaxBinding = this.#device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
    const sppmCeiling = Math.min(SPPM_PHOTON_CELLS_MAX_BYTES, deviceMaxBuffer, deviceMaxBinding);
    if (SPPM_PHOTON_CELLS_BYTES > sppmCeiling) {
      const mib = (SPPM_PHOTON_CELLS_BYTES / (1024 * 1024)).toFixed(1);
      this.#emitResourceCeilingWarning('sppmPhotonCells', {
        code: 'pt-webgpu.sppm-photon-cells-ceiling',
        backend: 'pt-webgpu',
        phase: 'renderFrame',
        method: 'renderFrame',
        message:
          `[vitrum/pt-webgpu] SPPM photon-cells buffer would be ${mib} MiB, ` +
          `exceeding the ${(sppmCeiling / (1024 * 1024)).toFixed(0)} MiB ceiling ` +
          '(min of the static SPPM ceiling and live device limits). ' +
          'The requested photon-map frame cannot be rendered.',
        details: {
          targetBytes: SPPM_PHOTON_CELLS_BYTES,
          ceilingBytes: sppmCeiling,
          deviceMaxBuffer,
          deviceMaxBinding,
          fallback: 'throw',
        },
      });
      return false;
    }
    // Build the photon-emission pipeline lazily.
    // The photon pass uses groups 0–3.  Group 3 now carries the SPPM bindings
    // (6/7/8) in addition to the light-tree / material-texture entries (0–5).
    // Its pipeline layout is [g0, g1, g2, g3] — the SAME 4-group layout as the
    // megakernel.  This is safe on ALL adapters (maxBindGroups = 4 is guaranteed
    // by the WebGPU spec).  Must be called AFTER ensurePipeline().
    if (
      this.bindGroupLayout == null ||
      this.bindGroupLayout1 == null ||
      this.bindGroupLayout2 == null ||
      this.bindGroupLayout3 == null
    ) {
      return false;
    }

    const previous = [
      this.#sppm.sppmPhotonCellsBuffer,
      this.#sppm.sppmCellCountersBuffer,
      this.#sppm.sppmStatsBuffer,
    ];
    const candidates: GPUBuffer[] = [];
    try {
      const photonCells = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.sppm.photonCells',
        size: SPPM_PHOTON_CELLS_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      candidates.push(photonCells);
      const cellCounters = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.sppm.cellCounters',
        size: SPPM_CELL_COUNTERS_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      candidates.push(cellCounters);
      const stats = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.sppm.stats',
        size: SPPM_STATS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      candidates.push(stats);

      const photonPipeline = this.#sppm.sppmPhotonPipeline ?? (() => {
        const module = this.#device.createShaderModule({
          label: 'vitrum.pt-webgpu.sppm.photonPass',
          code: composeSppmPhotonPassWgsl({ sampling: this.#sampling }),
        });
        const photonLayout = this.#device.createPipelineLayout({
          label: 'vitrum.pt-webgpu.sppm.photonPass.layout',
          bindGroupLayouts: [
            this.bindGroupLayout,
            this.bindGroupLayout1,
            this.bindGroupLayout2,
            this.bindGroupLayout3,
          ],
        });
        return this.#device.createComputePipeline({
          label: 'vitrum.pt-webgpu.sppm.photonPass.pipeline',
          layout: photonLayout,
          compute: { module, entryPoint: 'sppmEmitPhotons' },
        });
      })();

      this.#sppm.sppmPhotonCellsBuffer = photonCells;
      this.#sppm.sppmCellCountersBuffer = cellCounters;
      this.#sppm.sppmStatsBuffer = stats;
      this.#sppm.sppmPhotonPipeline = photonPipeline;
      this.#sppm.sppmBuffersReady = true;
      this.invalidateGroup3BindGroup();
      destroyGpuResourcesBestEffort(previous, candidates);
      return true;
    } catch (error) {
      destroyGpuResourcesBestEffort(candidates, previous);
      throw error;
    }
  }

  /**
   * Allocate, upload, and create every lite view without publishing any of it.
   * The returned token lets the engine include these textures in its wider
   * scene/BDPT/reset transaction.
   */
  stageLiteTextureReplacement(
    lightData: LiteLightTexData,
    envData: LiteEnvTexData,
    cdfData: LiteEnvCdfData,
    additionalForbiddenResources: readonly object[] = [],
  ): LiteTextureReplacement {
    if (this.#traceTier !== 'lite') {
      return { commit() {}, rollback() {}, finalize() {} };
    }
    const previousTextures = [
      this.liteEnvTexture,
      this.liteEnvCdfTexture,
      this.liteLightTexture,
    ];
    const created: GPUTexture[] = [];
    const forbiddenResources = new Set<object>([
      ...previousTextures.filter((texture) => texture != null),
      ...additionalForbiddenResources,
    ]);
    const registerCandidate = (texture: GPUTexture, label: string): GPUTexture => {
      if (forbiddenResources.has(texture)) {
        throw new Error(
          `[pt-webgpu] ${label} candidate aliased an existing GPU texture`,
        );
      }
      created.push(texture);
      forbiddenResources.add(texture);
      return texture;
    };
    let envTexture: GPUTexture;
    let envView: GPUTextureView;
    let cdfTexture: GPUTexture;
    let cdfView: GPUTextureView;
    let lightTexture: GPUTexture;
    let lightView: GPUTextureView;
    try {
      envTexture = registerCandidate(this.#device.createTexture({
        label: 'vitrum.pt-webgpu.lite.envTex',
        size: { width: envData.width, height: envData.height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      }), 'vitrum.pt-webgpu.lite.envTex');
      this.#device.queue.writeTexture(
        { texture: envTexture },
        envData.texels as unknown as Float32Array<ArrayBuffer>,
        { bytesPerRow: envData.width * 16, rowsPerImage: envData.height },
        { width: envData.width, height: envData.height },
      );
      envView = envTexture.createView();

      cdfTexture = registerCandidate(this.#device.createTexture({
        label: 'vitrum.pt-webgpu.lite.envCdfTex',
        size: { width: cdfData.width, height: cdfData.height, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      }), 'vitrum.pt-webgpu.lite.envCdfTex');
      this.#device.queue.writeTexture(
        { texture: cdfTexture },
        cdfData.data as unknown as Float32Array<ArrayBuffer>,
        { bytesPerRow: cdfData.width * 16, rowsPerImage: cdfData.height },
        { width: cdfData.width, height: cdfData.height },
      );
      cdfView = cdfTexture.createView();

      lightTexture = registerCandidate(this.#device.createTexture({
        label: 'vitrum.pt-webgpu.lite.lightTex',
        size: { width: lightData.width, height: 1, depthOrArrayLayers: 1 },
        format: 'rgba32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      }), 'vitrum.pt-webgpu.lite.lightTex');
      this.#device.queue.writeTexture(
        { texture: lightTexture },
        lightData.data as unknown as Float32Array<ArrayBuffer>,
        { bytesPerRow: lightData.width * 16, rowsPerImage: 1 },
        { width: lightData.width, height: 1 },
      );
      lightView = lightTexture.createView();
    } catch (error) {
      destroyGpuResourcesBestEffort(created, previousTextures);
      throw error;
    }

    const previous = {
      envTexture: this.liteEnvTexture,
      envView: this.liteEnvTextureView,
      cdfTexture: this.liteEnvCdfTexture,
      cdfView: this.liteEnvCdfTextureView,
      lightTexture: this.liteLightTexture,
      lightView: this.liteLightTextureView,
    };
    let state: 'staged' | 'committed' | 'rolled-back' | 'finalized' = 'staged';
    const commit = (): void => {
      if (state !== 'staged') return;
      this.liteEnvTexture = envTexture;
      this.liteEnvTextureView = envView;
      this.liteEnvCdfTexture = cdfTexture;
      this.liteEnvCdfTextureView = cdfView;
      this.liteLightTexture = lightTexture;
      this.liteLightTextureView = lightView;
      this.pathTraceBindGroup = null;
      state = 'committed';
    };
    return {
      commit,
      rollback: () => {
        if (state === 'rolled-back' || state === 'finalized') return;
        if (state === 'committed') {
          this.liteEnvTexture = previous.envTexture;
          this.liteEnvTextureView = previous.envView;
          this.liteEnvCdfTexture = previous.cdfTexture;
          this.liteEnvCdfTextureView = previous.cdfView;
          this.liteLightTexture = previous.lightTexture;
          this.liteLightTextureView = previous.lightView;
          this.pathTraceBindGroup = null;
        }
        destroyGpuResourcesBestEffort(created, previousTextures);
        state = 'rolled-back';
      },
      finalize: () => {
        if (state === 'finalized' || state === 'rolled-back') return;
        commit();
        destroyGpuResourcesBestEffort(previousTextures, created);
        state = 'finalized';
      },
    };
  }

  /**
   * Invalidate the group-3 bind group (which now includes the SPPM bindings
   * at 6/7/8) when the SPPM buffers are reallocated.  Called by
   * `ensureSppmBuffers` after a buffer realloc; the next `buildBindGroups` call
   * will re-create group-3 with the new buffer handles.
   *
   * Item-1 fix (2026-06-10): also clears `pathTraceBindGroup` (group-0 cache) so
   * `buildBindGroups` is forced to rebuild ALL groups — not just group-3.
   * Previously, group-0 stayed cached after placeholder→real buffer swap, causing
   * `buildBindGroups` to return the cached group-0 early (its `pathTraceBindGroup != null` fast-out)
   * while leaving `pathTraceBindGroup3` null on every subsequent frame.  Both
   * the photon-emission pass and the megakernel guard group-3 with a null-check
   * and silently skip the bind — the photon pass wrote nothing, the gather read
   * stale/zero data, and the full-tier pipeline missed the light-tree binding.
   * Root cause: the 4th occurrence of the placeholder-→-real buffer invalidation
   * bug class (ea88803 / 0bedd92 / item-F1b); fix is identical in principle.
   */
  invalidateGroup3BindGroup(): void {
    this.pathTraceBindGroup = null;    // force buildBindGroups to rebuild all groups
    this.pathTraceBindGroup3 = null;
  }

  #ensureSppmPixelStatsPlaceholder(): void {
    // A ceiling rejection must not downgrade a previously usable real buffer.
    // The placeholder exists only to satisfy the full-tier static layout before
    // the first accepted pixel-stats allocation.
    if (this.#sppm.sppmPixelStatsBuffer != null) return;
    const candidate = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.sppm.pixelStats.placeholder',
      size: 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.#sppm.sppmPixelStatsBuffer = candidate;
    this.#sppm.sppmPixelStatsWidth = 0;
    this.#sppm.sppmPixelStatsHeight = 0;
    this.invalidateGroup3BindGroup();
  }

  /** Write the SppmStats UBO per-frame. No-op if the buffer is not allocated. */
  writeSppmStats(
    currentRadius: number,
    r0: number,
    frameAccumulated: number,
    photonCount: number,
    sceneExtent: number,
    sceneCenter: readonly [number, number, number],
  ): void {
    if (this.#sppm.sppmStatsBuffer == null) return;
    const ubo = new ArrayBuffer(SPPM_STATS_BYTES);
    const f = new Float32Array(ubo);
    const u = new Uint32Array(ubo);
    f[0] = currentRadius;
    f[1] = r0;
    u[2] = frameAccumulated >>> 0;
    u[3] = photonCount >>> 0;
    f[4] = sceneExtent;
    f[5] = sceneCenter[0];
    f[6] = sceneCenter[1];
    f[7] = sceneCenter[2];
    this.#device.queue.writeBuffer(this.#sppm.sppmStatsBuffer, 0, ubo);
  }

  /**
   * A4-progressive — (Re)allocate the per-pixel SPPM statistics buffer to
   * `width × height`.  Returns `true` when the buffer exists at the requested
   * dims (freshly created or already cached); `false` on the lite tier or when
   * the requested size exceeds the device's maxBufferSize / maxStorageBufferBindingSize.
   *
   * On (re)allocation the buffer is GPU-cleared so every pixel's (τ, R, N)
   * starts at zero — the gather treats R=0 as "first frame" and seeds from r₀.
   * Called each frame from `renderFrame` just before `buildBindGroups`; also
   * called from `reset()` + `setScene()` so the stats are wiped when the view
   * moves or a new scene is loaded.
   */
  ensureSppmPixelStatsBuffer(width: number, height: number): boolean {
    if (this.#traceTier !== 'full') return false;
    const targetBytes = Math.max(
      64, // minimum for valid binding
      width * height * SPPM_PIXEL_STATS_BYTES_PER_PIXEL,
    );
    // Guard against device limits before allocating the photon cells.
    const deviceMaxBuffer  = this.#device.limits?.maxBufferSize ?? 256 * 1024 * 1024;
    const deviceMaxBinding = this.#device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
    if (targetBytes > Math.min(deviceMaxBuffer, deviceMaxBinding)) {
      const mib = (targetBytes / (1024 * 1024)).toFixed(1);
      this.#emitResourceCeilingWarning('sppmPixelStats', {
        code: 'pt-webgpu.sppm-pixel-stats-ceiling',
        backend: 'pt-webgpu',
        phase: 'renderFrame',
        method: 'renderFrame',
        message:
          `[vitrum/pt-webgpu] SPPM per-pixel stats buffer would be ${mib} MiB ` +
          `(${width}×${height} × ${SPPM_PIXEL_STATS_BYTES_PER_PIXEL} B), exceeding the device limit. ` +
          'The requested photon-map frame cannot be rendered. ' +
          '(This warning fires once per engine instance before the caller throws.)',
        details: {
          width,
          height,
          targetBytes,
          ceilingBytes: Math.min(deviceMaxBuffer, deviceMaxBinding),
          deviceMaxBuffer,
          deviceMaxBinding,
          fallback: 'throw',
        },
      });
      this.#ensureSppmPixelStatsPlaceholder();
      return false;
    }
    // Cache hit — buffer already at the right size.
    if (
      this.#sppm.sppmPixelStatsBuffer != null &&
      this.#sppm.sppmPixelStatsWidth  === width &&
      this.#sppm.sppmPixelStatsHeight === height
    ) {
      return true;
    }
    const previous = this.#sppm.sppmPixelStatsBuffer;
    let candidate: GPUBuffer | null = null;
    try {
      candidate = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.sppm.pixelStats',
        size: targetBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      // Clear before publication so a clear/submit failure cannot replace a
      // usable history buffer with an uninitialized candidate.
      const enc = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.sppm.pixelStats.clear' });
      enc.clearBuffer(candidate);
      this.#device.queue.submit([enc.finish()]);

      this.#sppm.sppmPixelStatsBuffer = candidate;
      this.#sppm.sppmPixelStatsWidth = width;
      this.#sppm.sppmPixelStatsHeight = height;
      this.invalidateGroup3BindGroup();
      destroyGpuResourcesBestEffort([previous], [candidate]);
      return true;
    } catch (error) {
      destroyGpuResourcesBestEffort([candidate], [previous]);
      throw error;
    }
  }

  /** Dispose SPPM-specific GPU resources. Called from dispose(). */
  #disposeSppmResources(): void {
    // D8.1: delegate to the SppmResources sub-object.
    this.#sppm.dispose();
  }

  // ── Present-pass public API ──────────────────────────────────────────────────

  /**
   * Lazily build the present compute pipeline and its PresentParams UBO.
   * No-op if already built. Must be called before `dispatchPresentPass`.
   */
  ensurePresentPipeline(): void {
    if (
      this.#present.presentPipeline != null &&
      this.#present.presentParamsBuffer != null
    ) {
      return;
    }
    const previousParams = this.#present.presentParamsBuffer;
    let params: GPUBuffer | null = null;
    try {
      const module = this.#device.createShaderModule({
        label: 'vitrum.pt-webgpu.present',
        code: PT_WEBGPU_PRESENT_WGSL,
      });
      // The shader uses textureLoad rather than filtering. Declare that
      // contract explicitly so both the rgba16float accumulator and the
      // rgba32float OIDN upload are legal without the optional
      // `float32-filterable` device feature. `layout: 'auto'` would infer a
      // filterable-float binding from texture_2d<f32> and reject rgba32float.
      const bindGroupLayout = this.#device.createBindGroupLayout({
        label: 'vitrum.pt-webgpu.present.bindGroupLayout',
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform' },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            texture: {
              sampleType: 'unfilterable-float',
              viewDimension: '2d',
            },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: {
              access: 'write-only',
              format: 'rgba16float',
              viewDimension: '2d',
            },
          },
        ],
      });
      const pipelineLayout = this.#device.createPipelineLayout({
        label: 'vitrum.pt-webgpu.present.pipelineLayout',
        bindGroupLayouts: [bindGroupLayout],
      });
      const pipeline = this.#device.createComputePipeline({
        label: 'vitrum.pt-webgpu.present.pipeline',
        layout: pipelineLayout,
        compute: { module, entryPoint: 'presentMain' },
      });
      params = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.present.params',
        size: GpuResources.PRESENT_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      this.#present.presentPipeline = pipeline;
      this.#present.presentParamsBuffer = params;
      destroyGpuResourcesBestEffort([previousParams], [params]);
    } catch (error) {
      destroyGpuResourcesBestEffort([params], [previousParams]);
      throw error;
    }
  }

  /**
   * Write the PresentParams UBO for the current frame.  Converts the
   * FrameQualitySettings string dials to their WGSL-side numeric indices:
   *   tonemapMode:      aces=0(default), agx=1, reinhard=2, linear=3, none=4
   *   exposure:         linear multiplier, default 1.0
   *   outputColorSpace: srgb=0(default, apply OETF), linear=1(skip OETF)
   *
   * Called per-frame by the engine just before `dispatchPresentPass`.
   * No-op if `ensurePresentPipeline` has not been called yet.
   */
  writePresentParams(
    tonemapMode: number,
    exposure: number,
    outputColorSpace: number,
  ): void {
    if (this.#present.presentParamsBuffer == null) return;
    const ubo = new ArrayBuffer(GpuResources.PRESENT_PARAMS_BYTES);
    const u = new Uint32Array(ubo);
    const f = new Float32Array(ubo);
    u[0] = tonemapMode >>> 0;
    f[1] = exposure;
    u[2] = outputColorSpace >>> 0;
    u[3] = 0; // _pad
    this.#device.queue.writeBuffer(this.#present.presentParamsBuffer, 0, ubo);
  }

  /**
   * Encode the present compute pass onto `encoder`.  Reads `accumTexture`
   * (the running-mean linear HDR) and writes the tonemapped+OETF result to
   * `presentTexture`.  Must be called AFTER the path-trace pass has written
   * `accumTexture` this frame.
   *
   * Preconditions: `ensurePresentPipeline` + `writePresentParams` have run;
   * `accumTexture` / `presentTexture` / `presentView` / `accumView` are non-null.
   */
  dispatchPresentPass(
    encoder: GPUCommandEncoder,
    width: number,
    height: number,
    sourceTexture: GPUTexture | null = this.accumTexture,
  ): void {
    if (
      this.#present.presentPipeline == null ||
      this.#present.presentParamsBuffer == null ||
      sourceTexture == null ||
      this.#present.presentTexture == null ||
      this.#present.presentView == null
    ) {
      return;
    }
    // Per-frame bind group: PresentParams UBO + accumTex (sampled) + presentTex (storage write).
    const bg = this.#device.createBindGroup({
      label: 'vitrum.pt-webgpu.present.bindgroup',
      layout: this.#present.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#present.presentParamsBuffer } },
        { binding: 1, resource: sourceTexture.createView() },
        { binding: 2, resource: this.#present.presentView },
      ],
    });
    const pass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.present.pass' });
    pass.setPipeline(this.#present.presentPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), 1);
    pass.end();
  }


  /** Whether an accepted OIDN linear-HDR source is retained for presentation. */
  hasDenoisedResult(): boolean {
    return this.#present.denoisedLinearTexture != null;
  }

  /** Stop presenting the accepted OIDN source and return to the accumulator. */
  clearDenoisedResult(): void {
    this.#present.denoisedLinearTexture?.destroy();
    this.#present.denoisedLinearTexture = null;
  }

  /**
   * Upload an accepted OIDN RGB frame and publish it only after the present
   * submission succeeds. All GPU mutation occurs synchronously inside the
   * host's explicit render cadence; the inference completion hook only queues
   * CPU data on the engine.
   */
  presentDenoisedResult(
    frame: { readonly rgb: Float32Array; readonly width: number; readonly height: number },
    tonemapMode: number,
    exposure: number,
    outputColorSpace: number,
  ): void {
    if (frame.width !== this.accumWidth || frame.height !== this.accumHeight) {
      throw new RangeError(
        `pt-webgpu: OIDN result ${frame.width}×${frame.height} does not match accumulation ` +
          `${this.accumWidth}×${this.accumHeight}`,
      );
    }
    const pixelCount = frame.width * frame.height;
    if (frame.rgb.length !== pixelCount * 3) {
      throw new RangeError(
        `pt-webgpu: OIDN result expected ${pixelCount * 3} RGB floats, got ${frame.rgb.length}`,
      );
    }

    const rgba = new Float32Array(pixelCount * 4);
    for (let i = 0; i < pixelCount; i += 1) {
      const src = i * 3;
      const dst = i * 4;
      rgba[dst] = frame.rgb[src]!;
      rgba[dst + 1] = frame.rgb[src + 1]!;
      rgba[dst + 2] = frame.rgb[src + 2]!;
      rgba[dst + 3] = 1;
    }

    const previous = this.#present.denoisedLinearTexture;
    const candidate = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.oidn.linear',
      size: { width: frame.width, height: frame.height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const forbiddenTextures = new Set<GPUTexture>();
    for (const texture of [
      previous,
      this.accumTexture,
      this.normalDepthTexture,
      this.albedoTexture,
      this.varianceTexture,
      this.motionVectorsTexture,
      this.#present.presentTexture,
      this.liteEnvTexture,
      this.liteEnvCdfTexture,
      this.liteLightTexture,
    ]) {
      if (texture != null) forbiddenTextures.add(texture);
    }
    const candidateAliasesLiveTexture = forbiddenTextures.has(candidate);
    try {
      if (candidateAliasesLiveTexture) {
        throw new Error('pt-webgpu: OIDN upload candidate aliased a live GPU texture');
      }
      this.#device.queue.writeTexture(
        { texture: candidate },
        rgba,
        { bytesPerRow: frame.width * 16, rowsPerImage: frame.height },
        { width: frame.width, height: frame.height, depthOrArrayLayers: 1 },
      );
      this.ensurePresentPipeline();
      this.writePresentParams(tonemapMode, exposure, outputColorSpace);
      const encoder = this.#device.createCommandEncoder({
        label: 'vitrum.pt-webgpu.oidn.present.encoder',
      });
      this.dispatchPresentPass(encoder, frame.width, frame.height, candidate);
      this.#device.queue.submit([encoder.finish()]);
    } catch (error) {
      // A non-conforming/mock device may return one of our live textures from
      // createTexture. Never destroy that live object while rejecting it; every
      // genuinely fresh candidate is disposed on all pre-publication failures.
      if (!candidateAliasesLiveTexture) {
        try { candidate.destroy(); } catch { /* preserve the presentation error */ }
      }
      throw error;
    }
    this.#present.denoisedLinearTexture = candidate;
    if (previous != null) {
      try { previous.destroy(); } catch { /* candidate is already live */ }
    }
  }

  /** Re-run presentation over the retained OIDN source, or the accumulator. */
  presentCurrentResult(
    tonemapMode: number,
    exposure: number,
    outputColorSpace: number,
  ): void {
    const source = this.#present.denoisedLinearTexture ?? this.accumTexture;
    if (source == null || this.accumWidth <= 0 || this.accumHeight <= 0) return;
    this.ensurePresentPipeline();
    this.writePresentParams(tonemapMode, exposure, outputColorSpace);
    const encoder = this.#device.createCommandEncoder({
      label: 'vitrum.pt-webgpu.present.current.encoder',
    });
    this.dispatchPresentPass(encoder, this.accumWidth, this.accumHeight, source);
    this.#device.queue.submit([encoder.finish()]);
  }
  /**
   * Full GPU-resource teardown for engine `dispose()`: destroy the accum textures
   * + buffers, drop the cached bind groups, destroy + null the params buffer, and
   * null the pipeline / layout handles. Mirrors the prior inline dispose order.
   */
  dispose(): void {
    this.destroyAccumTexture();
    this.pathTraceBindGroup = null;
    this.pathTraceBindGroup1 = null;
    this.pathTraceBindGroup2 = null;
    this.pathTraceBindGroup3 = null;
    // D8.1: PresentResources.dispose() tears down seed-blit resources (in order:
    // seedBlitParamsBuffer → seedBlitVarPlaceholder → seedBlitSampler → seedBlitPipeline)
    // then present-pass resources (presentParamsBuffer → presentPipeline).
    // presentTexture/presentView are already nulled by destroyAccumTexture() above.
    this.#present.dispose();
    this.paramsBuffer?.destroy();
    this.paramsBuffer = null;
    this.computePipeline = null;
    this.bdptCameraSplatResolvePipeline = null;
    this.bindGroupLayout = null;
    this.bindGroupLayout1 = null;
    this.bindGroupLayout2 = null;
    this.bindGroupLayout3 = null;
    this.#disposeReservoirResources();
    this.#disposeSppmResources();
    // B12 — lite-tier textures (no-op on full tier since they are null).
    this.liteEnvTexture?.destroy();
    this.liteEnvTexture = null;
    this.liteEnvTextureView = null;
    this.liteEnvCdfTexture?.destroy();
    this.liteEnvCdfTexture = null;
    this.liteEnvCdfTextureView = null;
    this.liteLightTexture?.destroy();
    this.liteLightTexture = null;
    this.liteLightTextureView = null;
  }
}
