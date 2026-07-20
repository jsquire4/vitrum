/**
 * GpuResources — the cohesive GPU-resource-lifecycle cluster extracted from the
 * `PTEngineWebGPU` god-class (T14-followup, mirrors the W1-R2 FrameResources split
 * on `@vitrum/walkaround-hybrid`).
 *
 * Owns, as a single sub-struct on the engine (`#gpu`):
 *   - the accumulation + aux textures (accum / normalDepth / albedo / variance /
 *     motionVectors) and their cached views,
 *   - the accum / varianceMoments storage buffers + the params uniform buffer,
 *   - the compute pipeline(s) (path-trace + optional BDPT light-subpath) sharing
 *     ONE explicit GPUPipelineLayout, and the explicit per-group bind-group
 *     layouts that pipeline layout is built from,
 *   - the cached per-frame bind groups (group 0/1/2), and
 *   - the current accum dims (width / height / byte size).
 *
 * Behavior is preserved verbatim from the prior inline implementation. The only
 * cross-cutting state that stays on the engine is `#samplesAccumulated`: methods
 * that reset it (`ensureAccumResources` on recreate) report that back to the
 * caller (return `recreated: boolean`) rather than reaching into engine state.
 * Bind-group *construction* takes the scene buffers + BDPT light-path view as
 * explicit parameters (those live on the engine), but the resulting groups are
 * cached here because their lifetime is tied to the accum views + pipeline.
 */

import type { EngineWarning } from '@vitrum/core';
import type { PtWebgpuTraceTier } from './traceTier.js';
import type { UploadedSceneBuffers } from './scene/uploadSceneBuffers.js';
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

export type PtWebgpuBvhTraversalMode = 'binary' | 'cwbvh-closest-experimental';

// ── Module-level binding-layout helpers (D8.3) ─────────────────────────────
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
/**
 * Sampled texture (unfilterable-float 2d) binding entry — used by B12 lite-tier
 * textures.  Trust-audit F2 (2026-06-10): rgba32float is UNFILTERABLE; declaring
 * 'float' (filterable) made every lite-tier pipeline fail validation.
 */
function _sampledTex(binding: number): GPUBindGroupLayoutEntry {
  return { binding, visibility: _vis(), texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } };
}

// ── D8.1 Sub-object: ReservoirResources ─────────────────────────────────────
/**
 * Owns the ReSTIR-PT reuse resources: reservoir ping-pong buffers, result
 * buffer, params UBO, the four reuse compute pipelines + composite megakernel,
 * and the cached per-pass reuse bind groups. Gated by `restirPtReuse &&
 * traceTier === 'full'`. GpuResources delegates to this sub-object and exposes
 * all its fields via read-write accessors so the existing external surface
 * (index.ts / tests) remains byte-identical.
 */
class ReservoirResources {
  /**
   * Full-res ReservoirPTHero ping-pong buffers (224 B/px = 56 u32). `Cur` is the
   * producer output that the temporal pass fuses in place; `Prev` is last frame's
   * temporal output (read-only this frame). `swapReservoirs()` exchanges them at
   * frame end so this frame's resolved reservoir becomes next frame's history.
   * STORAGE | COPY_SRC | COPY_DST (COPY_* so a future readback / clear works).
   */
  rptReservoirCur: GPUBuffer | null = null;
  rptReservoirPrev: GPUBuffer | null = null;
  /** A1 — the SPATIAL pass output (post-temporal → spatial → resolve). The spatial
   *  pass reads the temporal output (`Cur`) for neighbour sampling and writes here;
   *  resolve reads this. A dedicated buffer (not a ping-pong slot) so the spatial
   *  pass never writes the slot it samples (hazard-free neighbour reads). */
  rptReservoirSpatial: GPUBuffer | null = null;
  /** `rpt_result`: one vec4f / px (16 B) — the resolve pass's reconnection
   *  indirect (.rgb) + contributing flag (.a). STORAGE | COPY_SRC. */
  rptResultBuffer: GPUBuffer | null = null;
  /** RestirPtParams UBO (32 B: width/height/mClamp/allowGlossyReuse u32 + wCap/3×_pad f32). */
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
    this.rptReservoirSpatial?.destroy();
    this.rptReservoirSpatial = null;
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
   * PhotonRecord[SPPM_MAX_CELLS × SPPM_CELL_CAPACITY], 48 B/record ≈ 402 MiB.
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
   * photonCount, sceneExtent, _pad×3.  Written per-frame by the host.
   * Bound at group(3) binding(8).
   */
  sppmStatsBuffer: GPUBuffer | null = null;
  /**
   * A4-progressive — per-pixel SPPM statistics buffer.
   * SppmPixelStats[W×H] = {tau.rgb, radius2, N, _pad×3} × 32 bytes/px.
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

  dispose(): void {
    // Seed-blit resources (freed before present, matching original dispose order).
    this.seedBlitParamsBuffer?.destroy();
    this.seedBlitParamsBuffer = null;
    this.seedBlitVarPlaceholder?.destroy();
    this.seedBlitVarPlaceholder = null;
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
  bdptSubpathPipeline: GPUComputePipeline | null = null;
  /**
   * Explicit group-0 bind-group layout. Used to build `pathTraceBindGroup`.
   *
   * Both compute pipelines (path-trace `main` + BDPT `bdptExtendLightSubpath`)
   * share ONE explicit `GPUPipelineLayout` built from these layouts. A bind
   * group built against an explicit layout is NOT pipeline-exclusive, so the
   * SAME bind groups can be set on both pipelines — which is exactly what the
   * BDPT light-subpath pass needs (it reuses the path-trace scene/params/light
   * bindings). Auto-generated layouts (`layout:'auto'`) are pipeline-exclusive
   * per the WebGPU spec, so they would reject a cross-pipeline `setBindGroup`.
   */
  bindGroupLayout: GPUBindGroupLayout | null = null;
  /** Explicit group-1 layout (full tier only): analytics + env + area lights. */
  bindGroupLayout1: GPUBindGroupLayout | null = null;
  /** Explicit group-2 layout (full tier only): TLAS table + BDPT scratch buffers. */
  bindGroupLayout2: GPUBindGroupLayout | null = null;
  /** Explicit group-3 layout (full tier only): WS2 light-tree node buffer. */
  bindGroupLayout3: GPUBindGroupLayout | null = null;

  /**
   * BDPT eye-subpath scratch stack (D2): a per-pixel × maxEyeDepth read_write
   * storage buffer (2× vec4 / eye vertex = 32 B). Bound at group(2) binding(6) on
   * the full tier. A 32-byte placeholder is kept when BDPT is off so the explicit
   * group-2 layout (which always declares binding 6) stays satisfied.
   */
  bdptEyeStackBuffer: GPUBuffer | null = null;
  bdptEyeStackByteSize = 0;

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

  /** Bytes per ReservoirPTHero (56 u32). MUST equal RESERVOIR_PT_HERO_STRIDE·4
   *  in reservoirPtHero.wgsl.ts (pinned by reservoirPtHeroLayout.test.ts). */
  static readonly RESERVOIR_PT_HERO_BYTES = 224;
  /** RestirPtParams UBO byte size (8 × 4-byte fields). */
  static readonly RESTIR_PT_PARAMS_BYTES = 32;
  /**
   * Safety ceiling for EACH reservoir ping-pong buffer. 224 B/px is ~464 MB at
   * 1920×1080; above this ceiling we refuse to grow (skip reuse this frame) rather
   * than silently allocate. Mirrors the BDPT eye-stack ceiling discipline.
   */
  static readonly RESTIR_PT_RESERVOIR_MAX_BYTES = 416 * 1024 * 1024; // 416 MiB

  /** Byte size of the PresentParams UBO: 4 × 4 bytes. */
  static readonly PRESENT_PARAMS_BYTES = 16;

  /** Bytes per eye vertex in the scratch stack: 2× vec4f = 32. */
  static readonly BDPT_EYE_VERTEX_BYTES = 32;
  /**
   * Safety ceiling for the eye-stack allocation. The full-depth (8) per-pixel
   * stack is ~530 MB at 1920×1080; above this ceiling we refuse to grow the
   * buffer and warn rather than silently allocating a multi-hundred-MB region.
   */
  static readonly BDPT_EYE_STACK_MAX_BYTES = 384 * 1024 * 1024; // 384 MiB

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
    this.#cwbvhClosest = bvhTraversal === 'cwbvh-closest-experimental' && traceTier === 'full';
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

  /** Whether ReSTIR-PT reuse is active for this engine (compile-time + full-tier). */
  get restirPtReuseEnabled(): boolean {
    return this.#restirPtReuse;
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
    this.accumBufferByteSize = 0;
    this.accumWidth = 0;
    this.accumHeight = 0;
    // The present texture is sized to accumTexture; destroy it together.
    this.#present.presentTexture?.destroy();
    this.#present.presentTexture = null;
    this.#present.presentView = null;
  }

  clearAccumBuffer(): void {
    if (this.accumBuffer == null) return;
    const encoder = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.clearAccum' });
    encoder.clearBuffer(this.accumBuffer);
    if (this.varianceMomentsBuffer != null) {
      encoder.clearBuffer(this.varianceMomentsBuffer);
    }
    this.#device.queue.submit([encoder.finish()]);
  }

  /**
   * Item 2e — Clear all allocated ReSTIR-PT reservoir buffers (Cur/Prev/Spatial)
   * when the scene changes or the engine resets so stale temporal history from a
   * previous scene does not bleed into the new one. No-op when the buffers have
   * not yet been allocated. Called by `index.ts reset()` and full setScene.
   */
  clearReservoirBuffers(): void {
    if (this.#rsvr.rptReservoirCur == null) return;
    const encoder = this.#device.createCommandEncoder({
      label: 'vitrum.pt-webgpu.restirPt.clearReservoirs',
    });
    encoder.clearBuffer(this.#rsvr.rptReservoirCur);
    if (this.#rsvr.rptReservoirPrev != null) encoder.clearBuffer(this.#rsvr.rptReservoirPrev);
    if (this.#rsvr.rptReservoirSpatial != null) encoder.clearBuffer(this.#rsvr.rptReservoirSpatial);
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
    const targetByteSize = width * height * 16;
    const textureReady =
      this.accumTexture != null && this.accumWidth === width && this.accumHeight === height;
    const bufferReady = this.accumBuffer != null && this.accumBufferByteSize === targetByteSize;
    if (textureReady && bufferReady) {
      return false;
    }
    this.destroyAccumTexture();
    this.accumTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.accum',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.accumView = this.accumTexture.createView();
    this.normalDepthTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.normalDepth',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.normalDepthView = this.normalDepthTexture.createView();
    this.albedoTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.albedo',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.albedoView = this.albedoTexture.createView();
    this.varianceTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.variance',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.varianceView = this.varianceTexture.createView();
    if (this.#traceTier === 'full') {
      this.motionVectorsTexture = this.#device.createTexture({
        label: 'vitrum.pt-webgpu.motionVectors',
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.motionVectorsView = this.motionVectorsTexture.createView();
    }
    this.accumBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.accum.buffer',
      size: Math.max(16, targetByteSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (this.#traceTier === 'full') {
      this.varianceMomentsBuffer = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.varianceMoments.buffer',
        size: Math.max(16, targetByteSize),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    this.accumBufferByteSize = targetByteSize;
    this.accumWidth = width;
    this.accumHeight = height;
    // Present texture — same dims as accumTexture. Needs STORAGE_BINDING (write)
    // for the present compute pass and TEXTURE_BINDING so hosts can read it.
    // COPY_SRC so snapshot/debug paths can read it back.
    this.#present.presentTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.present',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.#present.presentView = this.#present.presentTexture.createView();
    // Drop ALL cached bind groups: the path-trace groups AND (when reuse is on)
    // the reuse group-0, which references the just-recreated accum/aux views.
    this.invalidateBindGroups();
    this.clearAccumBuffer();
    return true;
  }

  /**
   * Build the explicit `GPUBindGroupLayout`s for the current tier and wrap them
   * in a single `GPUPipelineLayout` shared by both compute pipelines. The
   * binding indices / resource types / visibility MUST match what the prior
   * `layout:'auto'` derived from the WGSL `@group/@binding` decls (see
   * `wgsl/pathTrace/material.wgsl.ts`) so the existing bind-group construction
   * in `buildBindGroups` stays valid unchanged.
   *
   * All bindings are COMPUTE-visible (the entire kernel is one compute stage).
   * The explicit layout is a SUPERSET that satisfies both `main` (uses every
   * binding) and `bdptExtendLightSubpath` (uses a subset) — an explicit layout
   * may declare bindings an entry point doesn't statically use, so one layout
   * serves both pipelines.
   */
  /**
   * D8.2 — The full-tier group-0 bind-group layout entries (bindings 0..13).
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
      _tex(11),       // varianceTexture
      _tex(12),       // motionVectorsTexture (full tier)
      _buf(13, _rw),  // varianceMomentsBuffer (read_write, full tier)
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
    ];
  }

  /**
   * D9-6 — The full-tier group-2 bind-group entries (TLAS table bindings 0..4 +
   * BDPT light-path/eye-stack bindings 5/6). Both `buildBindGroups` and
   * `rebuildGroup2Only` build this identical entry list; extracting it here
   * eliminates the prior duplication (any change propagates to both). The
   * `binding:5` light-path buffer is caller-supplied (the full build passes the
   * lazily-resolved `bdptLightPathBuffer()`; the H9 fast rebuild passes the
   * host-supplied next-frame buffer). Full tier only.
   */
  #makeGroup2BindGroupEntries(
    sb: UploadedSceneBuffers,
    lightPathBuffer: GPUBuffer,
  ): GPUBindGroupEntry[] {
    return [
      { binding: 0, resource: { buffer: sb.tlasNodesBuffer } },
      { binding: 1, resource: { buffer: sb.tlasInstanceIndicesBuffer } },
      { binding: 2, resource: { buffer: sb.tlasBlasRootsBuffer } },
      { binding: 3, resource: { buffer: sb.tlasInstanceWorldToLocalBuffer } },
      { binding: 4, resource: { buffer: sb.tlasInstanceLocalToWorldBuffer } },
      { binding: 5, resource: { buffer: lightPathBuffer } },
      { binding: 6, resource: { buffer: this.bdptEyeStackBuffer! } },
    ];
  }

  #buildSharedPipelineLayout(): GPUPipelineLayout {
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
        _tex(11),             // varianceTexture
        _sampledTex(12),      // liteEnvTex    — RGBA32F env radiance+pdf
        _sampledTex(13),      // liteEnvCdfTex — RGBA32F (.r = CDF entry)
        _sampledTex(14),      // liteLightTex  — RGBA32F packed light data
      ];
    }
    this.bindGroupLayout = this.#device.createBindGroupLayout({
      label: `vitrum.pt-webgpu.layout.group0.${this.#traceTier}`,
      entries: group0Entries,
    });
    const bindGroupLayouts: GPUBindGroupLayout[] = [this.bindGroupLayout];

    if (this.#traceTier === 'full') {
      // Group 1 — 11 read-only storage buffers (analytics + env + area lights).
      this.bindGroupLayout1 = this.#device.createBindGroupLayout({
        label: 'vitrum.pt-webgpu.layout.group1.full',
        entries: Array.from({ length: 11 }, (_unused, binding) => _buf(binding, _ro)),
      });
      // Group 2 — TLAS table (5 read-only) + BDPT light-path + eye-stack (read_write).
      this.bindGroupLayout2 = this.#device.createBindGroupLayout({
        label: 'vitrum.pt-webgpu.layout.group2.full',
        entries: [
          _buf(0, _ro), // tlasNodes
          _buf(1, _ro), // tlasInstanceIndices
          _buf(2, _ro), // tlasBlasRoots
          _buf(3, _ro), // tlasInstanceWorldToLocal
          _buf(4, _ro), // tlasInstanceLocalToWorld
          _buf(5, _rw), // bdptLightPath (read_write)
          _buf(6, _rw), // bdptEyeStack (read_write)
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
      this.bindGroupLayout3 = this.#device.createBindGroupLayout({
        label: 'vitrum.pt-webgpu.layout.group3.full',
        entries: [
          _buf(0, _ro), // lightTree (read-only storage)
          _buf(1, _ro), // meshUvs (P2)
          _buf(2, _ro), // materialTexDescriptors (P2)
          { binding: 3, visibility: VIS, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // materialTextures sRGB (P2)
          { binding: 4, visibility: VIS, sampler: { type: 'filtering' } }, // materialTexSampler (P2)
          { binding: 5, visibility: VIS, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // materialTexturesLinear normal/scalar data maps (P2)
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
      bindGroupLayouts.push(this.bindGroupLayout1, this.bindGroupLayout2, this.bindGroupLayout3);
    } else {
      this.bindGroupLayout1 = null;
      this.bindGroupLayout2 = null;
      this.bindGroupLayout3 = null;
    }

    return this.#device.createPipelineLayout({
      label: `vitrum.pt-webgpu.pipelineLayout.${this.#traceTier}`,
      bindGroupLayouts,
    });
  }

  ensurePipeline(): void {
    if (this.computePipeline != null && this.bindGroupLayout != null && this.paramsBuffer != null) {
      return;
    }
    this.paramsBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.params',
      size: 512,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // WS4 — the full-tier kernel is composed for this engine's integrator
    // config: the volumetric SSS random walk is compiled in only when BDPT is
    // OFF (structural gate — energy conservation; BDPT has no medium logic).
    const samplingOpts = { sampling: this.#sampling, cwbvhClosest: this.#cwbvhClosest };
    const traceWgsl =
      this.#traceTier === 'lite'
        ? composePtWebgpuTraceLiteWgsl(samplingOpts)
        : composePtWebgpuTraceWgsl(this.#bdpt, samplingOpts);
    const module = this.#device.createShaderModule({
      label: `vitrum.pt-webgpu.pathTrace.${this.#traceTier}`,
      code: traceWgsl,
    });
    // ONE explicit pipeline layout shared by BOTH pipelines. Auto layouts are
    // pipeline-exclusive (WebGPU spec), so a bind group built against the
    // path-trace pipeline's auto layout cannot be set on the BDPT pipeline (and
    // vice versa). The BDPT light-subpath pass reuses the path-trace bind groups,
    // so it requires a shared explicit layout to dispatch on real hardware.
    const pipelineLayout = this.#buildSharedPipelineLayout();
    this.computePipeline = this.#device.createComputePipeline({
      label: 'vitrum.pt-webgpu.pathTrace.pipeline',
      layout: pipelineLayout,
      compute: {
        module,
        entryPoint: 'main',
      },
    });
    if (this.#traceTier === 'full' && this.#bdpt) {
      this.bdptSubpathPipeline = this.#device.createComputePipeline({
        label: 'vitrum.pt-webgpu.bdptLightSubpath.pipeline',
        layout: pipelineLayout,
        compute: {
          module,
          entryPoint: 'bdptExtendLightSubpath',
        },
      });
    } else {
      this.bdptSubpathPipeline = null;
    }
  }

  /**
   * (Re)allocate the BDPT eye-subpath scratch stack for the given render dims and
   * per-pixel eye depth. Returns `true` if BDPT connections may proceed this
   * frame, `false` if the allocation would exceed the safety ceiling (caller must
   * skip the BDPT pass; a stale/placeholder buffer remains bound so the pipeline
   * still validates). Sizes the buffer to `width·height·maxDepth·32 B`. When
   * BDPT is off, keeps only a 32-byte placeholder.
   *
   * The full-tier explicit group-2 layout always declares the `bdptEyeStack`
   * binding (6), so a non-null buffer must always exist on the full tier.
   */
  ensureBdptEyeStack(width: number, height: number, maxDepth: number, bdptActive: boolean): boolean {
    if (this.#traceTier !== 'full') {
      return false;
    }
    const targetBytes = bdptActive
      ? Math.max(
          GpuResources.BDPT_EYE_VERTEX_BYTES,
          width * height * Math.max(1, maxDepth) * GpuResources.BDPT_EYE_VERTEX_BYTES,
        )
      : GpuResources.BDPT_EYE_VERTEX_BYTES;

    if (bdptActive && targetBytes > GpuResources.BDPT_EYE_STACK_MAX_BYTES) {
      // H14-F: once-gate — warn only on the first frame that hits the ceiling.
      const mib = (targetBytes / (1024 * 1024)).toFixed(1);
      this.#emitResourceCeilingWarning('bdptEyeStack', {
        code: 'pt-webgpu.bdpt-eye-stack-ceiling',
        backend: 'pt-webgpu',
        phase: 'renderFrame',
        method: 'renderFrame',
        message:
          `[vitrum/pt-webgpu] BDPT eye-stack scratch would be ${mib} MiB ` +
          `(${width}×${height} × depth ${maxDepth} × 32 B), exceeding the ` +
          `${(GpuResources.BDPT_EYE_STACK_MAX_BYTES / (1024 * 1024)).toFixed(0)} MiB ceiling. ` +
          'Skipping BDPT connections this frame — lower resolutionFactor, cap bounces, or tile. ' +
          '(This warning fires once per engine instance.)',
        details: {
          width,
          height,
          maxDepth,
          targetBytes,
          ceilingBytes: GpuResources.BDPT_EYE_STACK_MAX_BYTES,
          fallback: 'skip-bdpt-connections',
        },
      });
      if (this.bdptEyeStackBuffer == null) {
        this.bdptEyeStackBuffer = this.#device.createBuffer({
          label: 'vitrum.pt-webgpu.bdpt.eyeStack.placeholder',
          size: GpuResources.BDPT_EYE_VERTEX_BYTES,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.bdptEyeStackByteSize = GpuResources.BDPT_EYE_VERTEX_BYTES;
        this.invalidateBindGroups();
      }
      return false;
    }

    if (this.bdptEyeStackBuffer != null && this.bdptEyeStackByteSize === targetBytes) {
      return bdptActive;
    }
    this.bdptEyeStackBuffer?.destroy();
    this.bdptEyeStackBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.bdpt.eyeStack',
      size: targetBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.bdptEyeStackByteSize = targetBytes;
    this.invalidateBindGroups();
    return bdptActive;
  }

  // ── ReSTIR-PT reuse resource lifecycle (gated; full tier only) ───────────────

  /**
   * (Re)allocate the ReSTIR-PT reservoir ping-pong buffers (`Cur` / `Prev`,
   * 144 B/px), the `rpt_result` buffer (16 B/px), and the RestirPtParams UBO to
   * the requested dims. No-op (returns `false`) when reuse is OFF. Returns `true`
   * when buffers for the requested frame dimensions are available, whether they
   * were freshly created or already cached. Refuses to grow past the per-buffer
   * safety ceiling (returns `false`, leaving any prior buffers as-is); the caller
   * skips the reuse passes that frame instead of dispatching against stale sizes.
   *
   * Both reservoir buffers are zero-cleared on (re)allocation so the FIRST frame's
   * temporal pass reads an empty (M=0) history rather than garbage.
   */
  ensureReservoirBuffers(width: number, height: number): boolean {
    if (!this.#restirPtReuse) return false;
    const px = Math.max(1, width) * Math.max(1, height);
    const reservoirBytes = px * GpuResources.RESERVOIR_PT_HERO_BYTES;
    const resultBytes = px * 16;
    if (reservoirBytes > GpuResources.RESTIR_PT_RESERVOIR_MAX_BYTES) {
      // H14-F: once-gate — warn only on the first frame that hits the ceiling.
      const mib = (reservoirBytes / (1024 * 1024)).toFixed(1);
      this.#emitResourceCeilingWarning('restirPtReservoir', {
        code: 'pt-webgpu.restir-pt-reservoir-ceiling',
        backend: 'pt-webgpu',
        phase: 'renderFrame',
        method: 'renderFrame',
        message:
          `[vitrum/pt-webgpu] ReSTIR-PT reservoir buffer would be ${mib} MiB ` +
          `(${width}×${height} × ${GpuResources.RESERVOIR_PT_HERO_BYTES} B), exceeding the ` +
          `${(GpuResources.RESTIR_PT_RESERVOIR_MAX_BYTES / (1024 * 1024)).toFixed(0)} MiB ceiling. ` +
          'Skipping ReSTIR-PT reuse this frame — lower resolutionFactor or tile. ' +
          '(This warning fires once per engine instance.)',
        details: {
          width,
          height,
          targetBytes: reservoirBytes,
          ceilingBytes: GpuResources.RESTIR_PT_RESERVOIR_MAX_BYTES,
          fallback: 'skip-restir-pt-reuse',
        },
      });
      return false;
    }
    const ready =
      this.#rsvr.rptReservoirCur != null &&
      this.#rsvr.rptReservoirByteSize === reservoirBytes &&
      this.#rsvr.rptResultByteSize === resultBytes;
    if (ready) return true;

    this.#rsvr.rptReservoirCur?.destroy();
    this.#rsvr.rptReservoirPrev?.destroy();
    this.#rsvr.rptReservoirSpatial?.destroy();
    this.#rsvr.rptResultBuffer?.destroy();
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.#rsvr.rptReservoirCur = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.restirPt.reservoir.cur',
      size: reservoirBytes,
      usage,
    });
    this.#rsvr.rptReservoirPrev = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.restirPt.reservoir.prev',
      size: reservoirBytes,
      usage,
    });
    this.#rsvr.rptReservoirSpatial = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.restirPt.reservoir.spatial',
      size: reservoirBytes,
      usage,
    });
    this.#rsvr.rptResultBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.restirPt.result',
      size: resultBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    if (this.#rsvr.rptParamsBuffer == null) {
      this.#rsvr.rptParamsBuffer = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.restirPt.params',
        size: GpuResources.RESTIR_PT_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    this.#rsvr.rptReservoirByteSize = reservoirBytes;
    this.#rsvr.rptResultByteSize = resultBytes;
    // Zero the ping-pong so frame 0's temporal history is an empty reservoir.
    const enc = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.restirPt.clear' });
    enc.clearBuffer(this.#rsvr.rptReservoirCur);
    enc.clearBuffer(this.#rsvr.rptReservoirPrev);
    enc.clearBuffer(this.#rsvr.rptReservoirSpatial);
    this.#device.queue.submit([enc.finish()]);
    // New buffers → the cached reuse bind groups are stale.
    this.#rsvr.rptProducerGroup0 = null;
    this.#rsvr.rptTemporalGroup0 = null;
    this.#rsvr.rptSpatialGroup0 = null;
    this.#rsvr.rptResolveGroup0 = null;
    return true;
  }

  /**
   * Write the RestirPtParams UBO (width/height/mClamp/allowGlossyReuse + wCap).
   * No-op when reuse is OFF or the buffer is absent. Called per-frame by the
   * engine before dispatch.
   */
  writeReservoirParams(
    width: number,
    height: number,
    mClamp: number,
    wCap: number,
    allowGlossyReuse: boolean,
  ): void {
    if (!this.#restirPtReuse || this.#rsvr.rptParamsBuffer == null) return;
    const ubo = new ArrayBuffer(GpuResources.RESTIR_PT_PARAMS_BYTES);
    const u = new Uint32Array(ubo);
    const f = new Float32Array(ubo);
    u[0] = width >>> 0;
    u[1] = height >>> 0;
    u[2] = Math.max(1, Math.floor(mClamp)) >>> 0;
    u[3] = allowGlossyReuse ? 1 : 0;
    f[4] = wCap;
    f[5] = 0; f[6] = 0; f[7] = 0; // _padB/_padC/_padD
    this.#device.queue.writeBuffer(this.#rsvr.rptParamsBuffer, 0, ubo);
  }

  /**
   * Build the extended group-0 bind-group layout for the reuse passes: the
   * megakernel's full-tier group-0 bindings (0..13, IDENTICAL to #buildShared-
   * PipelineLayout's group0) PLUS the relocated reuse bindings (20..24). This is
   * the ONE group the reuse passes carry their own resources in; groups 1/2/3 are
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
      // Relocated reuse bindings (20..24). The composed WGSL keeps rpt_resPrev /
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
   * Lazily build the three reuse compute pipelines + their shared 4-group layout
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
    if (this.#rsvr.rptProducerPipeline != null) return;
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
    this.#rsvr.rptGroup0Layout = this.#buildReservoirGroup0Layout();
    const pipelineLayout = this.#device.createPipelineLayout({
      label: 'vitrum.pt-webgpu.restirPt.pipelineLayout',
      bindGroupLayouts: [
        this.#rsvr.rptGroup0Layout,
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
    this.#rsvr.rptProducerPipeline = mk(
      'vitrum.pt-webgpu.restirPt.producer',
      composeRestirPtProducerWgsl({ sampling: this.#sampling }),
      'restirPtProduce',
    );
    this.#rsvr.rptTemporalPipeline = mk(
      'vitrum.pt-webgpu.restirPt.temporal',
      composeRestirPtTemporalWgsl({ sampling: this.#sampling }),
      'restirPtTemporal',
    );
    this.#rsvr.rptSpatialPipeline = mk(
      'vitrum.pt-webgpu.restirPt.spatial',
      composeRestirPtSpatialWgsl({ sampling: this.#sampling }),
      'restirPtSpatial',
    );
    this.#rsvr.rptResolvePipeline = mk(
      'vitrum.pt-webgpu.restirPt.resolve',
      composeRestirPtResolveWgsl({ sampling: this.#sampling }),
      'restirPtResolve',
    );
    // A1 — the COMPOSITE megakernel uses the SAME [g0', g1, g2, g3] layout (it reads
    // rpt_result at the relocated group-0 binding 23 + the scene groups). Composed
    // for this engine's BDPT mode (matches the default megakernel's SSS/BDPT gate).
    this.#rsvr.rptCompositePipeline = mk(
      'vitrum.pt-webgpu.restirPt.compositeMegakernel',
      composePtWebgpuCompositeTraceWgsl(this.#bdpt, {
        sampling: this.#sampling,
        cwbvhClosest: this.#cwbvhClosest,
      }),
      'main',
    );
  }

  /**
   * Build (and cache) the per-pass reuse group-0 bind groups. Each provides the
   * megakernel group-0 scene/G-buffer resources (IDENTICAL to the trace group 0)
   * PLUS the reuse bindings (20..24). The producer reads `Cur` as its OUTPUT slot
   * (binding 21 is the "current" reservoir in every pass by binding number); the
   * temporal pass reads `Cur`(21)+`Prev`(22); the resolve pass reads `Cur`(21) +
   * writes `result`(23). All three also bind 20/23/24 even if a given pass does
   * not declare them (extra layout entries are legal; the layout is uniform), so
   * one group-0 bind-group construction serves all three with the SAME resources
   * except the producer's binding-20 write target. Returns nothing; the engine
   * reads the cached groups off this struct. Idempotent.
   *
   * NOTE on binding 20 vs 21: the producer writes `rpt_reservoirOut` at b20 and
   * the temporal/resolve read `rpt_resCurrent`/`rpt_resResolved` at b21 — these
   * are the SAME logical "current" reservoir. We bind `rptReservoirCur` to BOTH
   * b20 and b21 so the producer's output IS the temporal's input (one buffer),
   * and `rptReservoirPrev` to b22.
   *
   * **Aliasing-safety rationale** — `rptReservoirCur` is bound simultaneously at
   * b20 (`rpt_reservoirOut`, read_write) and b21 (`rpt_resCurrent`, read_write).
   * This is safe because the producer, temporal, spatial, and resolve passes are
   * SEQUENTIAL compute passes: each pass encodes into its own
   * `GPUComputePassEncoder` and the encoder submits them in order. WebGPU (§19.4)
   * guarantees an implicit memory barrier at each `computePassEncoder.end()`, so
   * the producer's writes to b20 are fully visible before the temporal pass reads
   * b21 — there is no within-pass read-write hazard. Within a single pass the
   * producer only WRITES b20 (b21 is read by temporal, not by producer), so the
   * dual binding is data-race-free at all times.
   */
  buildReservoirBindGroups(sb: UploadedSceneBuffers): void {
    if (!this.#restirPtReuse || this.#rsvr.rptGroup0Layout == null) return;
    if (this.#rsvr.rptProducerGroup0 != null) return;
    if (
      this.#rsvr.rptReservoirCur == null ||
      this.#rsvr.rptReservoirPrev == null ||
      this.#rsvr.rptReservoirSpatial == null ||
      this.#rsvr.rptResultBuffer == null ||
      this.#rsvr.rptParamsBuffer == null ||
      this.accumView == null ||
      this.normalDepthView == null ||
      this.albedoView == null ||
      this.varianceView == null ||
      this.motionVectorsView == null ||
      this.varianceMomentsBuffer == null ||
      this.paramsBuffer == null
    ) {
      return;
    }
    const B = RPT_GROUP0_BINDING_BASE;
    // D8.2: Shared megakernel group-0 scene/G-buffer entries (0..13) sourced from
    // #makeGroup0BindGroupEntries() — guaranteed to match the layout produced by
    // #makeGroup0LayoutEntries() that #buildReservoirGroup0Layout() also uses.
    const sceneG0: GPUBindGroupEntry[] = [
      ...this.#makeGroup0BindGroupEntries(sb),
      { binding: B + 4, resource: { buffer: this.#rsvr.rptParamsBuffer } },
    ];
    // The reuse-reservoir slots differ only in which buffer is the "current"
    // ping-pong half. All three passes share ONE bind group: b20 = b21 = Cur (the
    // producer writes b20, temporal/resolve read b21 — same buffer), b22 = Prev,
    // b23 = result.
    const group0 = this.#device.createBindGroup({
      label: 'vitrum.pt-webgpu.restirPt.bindgroup0',
      layout: this.#rsvr.rptGroup0Layout,
      entries: [
        ...sceneG0,
        { binding: B + 0, resource: { buffer: this.#rsvr.rptReservoirCur } }, // rpt_reservoirOut → Cur
        { binding: B + 1, resource: { buffer: this.#rsvr.rptReservoirCur } }, // rpt_resCurrent  → Cur
        { binding: B + 2, resource: { buffer: this.#rsvr.rptReservoirPrev } }, // rpt_resPrev    → Prev
        { binding: B + 3, resource: { buffer: this.#rsvr.rptResultBuffer } }, // rpt_result      → result
        { binding: B + 5, resource: { buffer: this.#rsvr.rptReservoirSpatial } }, // rpt_resSpatial → Spatial
      ],
    });
    // Same resources for all four passes (the layout + bindings are uniform).
    this.#rsvr.rptProducerGroup0 = group0;
    this.#rsvr.rptTemporalGroup0 = group0;
    this.#rsvr.rptSpatialGroup0 = group0;
    this.#rsvr.rptResolveGroup0 = group0;
  }

  /**
   * Ping-pong the reservoir buffers: this frame's RESOLVED reservoir is the SPATIAL
   * pass output (producer→Cur, temporal Prev→Cur, spatial Cur→Spatial, resolve reads
   * Spatial). For the temporal feedback loop to carry the spatially-improved estimate
   * forward, next frame's `Prev` history must be THIS frame's `Spatial` — so we swap
   * Prev↔Spatial (the old Prev buffer becomes the new Spatial scratch). `Cur` is
   * producer-overwritten every frame, so it does NOT rotate. Invalidates the cached
   * reuse bind groups (they reference the now-swapped buffers). No-op when reuse OFF.
   *
   * **ORDERING CONSTRAINT** — this method MUST be called AFTER `device.queue.submit()`
   * for the current frame. Calling it before submit would swap the buffer references
   * while the GPU is still reading/writing them inside the submitted command buffer,
   * causing temporal history corruption on the very next frame (Prev and Spatial
   * buffers in the new bind groups would point to the wrong ping-pong half).
   * The call site in `index.ts` (the `gpu.swapReservoirs()` call immediately after
   * `this.#device.queue.submit(…)` in `renderFrame`) enforces this ordering: the swap
   * and bind-group invalidation always happen after the GPU command stream has been
   * handed to the driver.
   */
  swapReservoirs(): void {
    if (!this.#restirPtReuse) return;
    const tmp = this.#rsvr.rptReservoirPrev;
    this.#rsvr.rptReservoirPrev = this.#rsvr.rptReservoirSpatial;
    this.#rsvr.rptReservoirSpatial = tmp;
    this.#rsvr.rptProducerGroup0 = null;
    this.#rsvr.rptTemporalGroup0 = null;
    this.#rsvr.rptSpatialGroup0 = null;
    this.#rsvr.rptResolveGroup0 = null;
  }

  /** Tear down all ReSTIR-PT reuse resources. Called from dispose(). */
  #disposeReservoirResources(): void {
    // D8.1: delegate to the ReservoirResources sub-object.
    this.#rsvr.dispose();
  }

  /**
   * Build (and cache) the path-trace bind group(s) from the current accum views,
   * params buffer, pipeline layout, the supplied scene buffers, and the BDPT
   * light-path buffer. Returns group 0 (the always-present group). Groups 1/2 are
   * only created on the `full` tier and are read back off this struct by the
   * caller. Idempotent: if group 0 is already cached, returns it unchanged.
   *
   * `bdptLightPathBuffer` is a thunk so the engine's lazy placeholder-buffer
   * creation only fires on the construction branch (matching the prior inline
   * code, which only called `#bdptLightPathBuffer()` when the group was rebuilt).
   *
   * Callers must have already run `ensureAccumResources` + `ensurePipeline` and
   * validated that the views / pipeline / layout / params / scene buffers are
   * non-null (renderFrame's preconditions handle this).
   */
  buildBindGroups(sb: UploadedSceneBuffers, bdptLightPathBuffer: () => GPUBuffer): GPUBindGroup {
    if (this.pathTraceBindGroup != null) return this.pathTraceBindGroup;
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
    const fullGroup2Entries: GPUBindGroupEntry[] = this.#makeGroup2BindGroupEntries(
      sb,
      bdptLightPathBuffer(),
    );
    const bindGroup = this.#device.createBindGroup({
      label: `vitrum.pt-webgpu.pathTrace.bindgroup0.${this.#traceTier}`,
      layout: this.bindGroupLayout!,
      entries: this.#traceTier === 'lite' ? liteEntries : fullGroup0Entries,
    });
    this.pathTraceBindGroup = bindGroup;
    if (this.#traceTier === 'full') {
      // Built against the SAME explicit layouts the shared pipeline layout uses,
      // so these groups set cleanly on BOTH the path-trace and BDPT pipelines.
      this.pathTraceBindGroup1 = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.pathTrace.bindgroup1.full',
        layout: this.bindGroupLayout1!,
        entries: fullGroup1Entries,
      });
      this.pathTraceBindGroup2 = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.pathTrace.bindgroup2.full',
        layout: this.bindGroupLayout2!,
        entries: fullGroup2Entries,
      });
      this.pathTraceBindGroup3 = this.#device.createBindGroup({
        label: 'vitrum.pt-webgpu.pathTrace.bindgroup3.full',
        layout: this.bindGroupLayout3!,
        entries: [
          { binding: 0, resource: { buffer: sb.lightTreeBuffer } },
          { binding: 1, resource: { buffer: sb.uvsBuffer } },
          { binding: 2, resource: { buffer: sb.materialTexDescriptorsBuffer } },
          { binding: 3, resource: sb.materialTextureView },
          { binding: 4, resource: sb.materialTextureSampler },
          { binding: 5, resource: sb.materialLinearTextureView },
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
    return bindGroup;
  }

  /**
   * H9 — Reconstruct ONLY bind group 2 (TLAS table + BDPT light-path/eye-stack)
   * while leaving groups 0, 1, and 3 intact.  Called by `bdptAdvanceFrame` when
   * the host supplies a new external light-path buffer for the NEXT frame; the
   * full-group rebuild in `buildBindGroups` is NOT triggered because group 0 is
   * still cached (and returning it early is the correct fast path for all other
   * frames).
   *
   * Fast-out: if `lightPathBuffer` is the same reference that was used to build
   * the currently-cached group 2, the group is left in place (pointer equality
   * suffices because GPUBuffer identity is stable for the same host allocation).
   *
   * Preconditions (enforced by the `bdptAdvanceFrame` caller):
   *  - full tier only (`this.#traceTier === 'full'`)
   *  - `bindGroupLayout2` non-null (ensurePipeline must have run)
   *  - `sb` non-null (a scene has been set)
   *  - `bdptEyeStackBuffer` non-null (ensureBdptEyeStack must have run)
   */
  rebuildGroup2Only(sb: UploadedSceneBuffers, lightPathBuffer: GPUBuffer): void {
    if (this.#traceTier !== 'full' || this.bindGroupLayout2 == null) return;
    // Pointer-equality fast-out: if the buffer didn't change, the cached group
    // is still valid — avoid a redundant createBindGroup call.
    if (this.#lastBdptLightPathBuffer === lightPathBuffer && this.pathTraceBindGroup2 != null) {
      return;
    }
    this.#lastBdptLightPathBuffer = lightPathBuffer;
    this.pathTraceBindGroup2 = this.#device.createBindGroup({
      label: 'vitrum.pt-webgpu.pathTrace.bindgroup2.full.bdptRebuild',
      layout: this.bindGroupLayout2,
      entries: this.#makeGroup2BindGroupEntries(sb, lightPathBuffer),
    });
  }
  /** The light-path buffer reference used to build the most-recent group 2.
   *  Enables the pointer-equality fast-out in `rebuildGroup2Only`. */
  #lastBdptLightPathBuffer: GPUBuffer | null = null;

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

    // Lazily build the seed-blit pipeline + sampler + params UBO (engine-owned).
    if (this.#present.seedBlitPipeline == null) {
      const module = this.#device.createShaderModule({
        label: 'vitrum.pt-webgpu.seedBlit',
        code: PT_WEBGPU_SEED_BLIT_WGSL,
      });
      this.#present.seedBlitPipeline = this.#device.createComputePipeline({
        label: 'vitrum.pt-webgpu.seedBlit.pipeline',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
    }
    if (this.#present.seedBlitSampler == null) {
      // Filtering sampler so a differently-sized seed is bilinearly resampled
      // onto the accum grid; clamp so edge UVs don't wrap.
      this.#present.seedBlitSampler = this.#device.createSampler({
        label: 'vitrum.pt-webgpu.seedBlit.sampler',
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      });
    }
    if (this.#present.seedBlitParamsBuffer == null) {
      this.#present.seedBlitParamsBuffer = this.#device.createBuffer({
        label: 'vitrum.pt-webgpu.seedBlit.params',
        size: 32, // vec4u seedDim (16) + vec4f seedWeight (16)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }

    // SeedParams UBO: seedDim (accum dims) as uvec4, seedWeight as vec4f.
    const ubo = new ArrayBuffer(32);
    new Uint32Array(ubo, 0, 4).set([width >>> 0, height >>> 0, 0, 0]);
    new Float32Array(ubo, 16, 4).set([W, 0, 0, 0]);
    this.#device.queue.writeBuffer(this.#present.seedBlitParamsBuffer, 0, ubo);

    // varianceMoments slot: the real buffer on the full tier; a discardable
    // placeholder on the lite tier (which has none) so the layout stays valid.
    let varBuffer = this.varianceMomentsBuffer;
    if (varBuffer == null) {
      if (this.#present.seedBlitVarPlaceholder == null) {
        this.#present.seedBlitVarPlaceholder = this.#device.createBuffer({
          label: 'vitrum.pt-webgpu.seedBlit.varPlaceholder',
          size: this.accumBufferByteSize > 0 ? this.accumBufferByteSize : 16,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }
      varBuffer = this.#present.seedBlitVarPlaceholder;
    }

    const bindGroup = this.#device.createBindGroup({
      label: 'vitrum.pt-webgpu.seedBlit.bindgroup',
      layout: this.#present.seedBlitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#present.seedBlitParamsBuffer } },
        { binding: 1, resource: seedTex.createView() },
        { binding: 2, resource: this.#present.seedBlitSampler },
        { binding: 3, resource: { buffer: this.accumBuffer } },
        { binding: 4, resource: { buffer: varBuffer } },
      ],
    });

    const encoder = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.seedBlit.encoder' });
    const pass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.seedBlit.pass' });
    pass.setPipeline(this.#present.seedBlitPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), 1);
    pass.end();
    this.#device.queue.submit([encoder.finish()]);
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
    // A full invalidation also clears the fast-out reference so the next
    // rebuildGroup2Only call unconditionally rebuilds against fresh scene buffers.
    this.#lastBdptLightPathBuffer = null;
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
    if (this.#sppm.sppmBuffersReady) return false;
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
   * A 16-byte placeholder buffer is always created when SPPM is off so the
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
      if (this.#sppm.sppmPhotonCellsBuffer == null) {
        this.#sppm.sppmPhotonCellsBuffer = this.#device.createBuffer({
          label: 'vitrum.pt-webgpu.sppm.photonCells.placeholder',
          size: 64,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.#sppm.sppmCellCountersBuffer = this.#device.createBuffer({
          label: 'vitrum.pt-webgpu.sppm.cellCounters.placeholder',
          size: 64,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.#sppm.sppmStatsBuffer = this.#device.createBuffer({
          label: 'vitrum.pt-webgpu.sppm.stats.placeholder',
          size: 64,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // A4-progressive: placeholder for binding(9) when SPPM is off.
        if (this.#sppm.sppmPixelStatsBuffer == null) {
          this.#sppm.sppmPixelStatsBuffer = this.#device.createBuffer({
            label: 'vitrum.pt-webgpu.sppm.pixelStats.placeholder',
            size: 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
        }
        // SPPM buffers just created — invalidate group-3 so it rebuilds
        // with the new placeholder handles.
        this.invalidateGroup3BindGroup();
      }
      return false;
    }
    // Full allocation path.
    if (this.#sppm.sppmBuffersReady) return true;

    // R7a behavioral-gate fix (2026-06-10): the static ceiling alone let a
    // 402 MiB allocation through onto devices whose maxBufferSize is the
    // WebGPU default 256 MiB — buffer creation failed validation and photon-map
    // rendered black.  Guard against the LIVE device limit too, same degrade.
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
          '(min of the static SPPM ceiling and device.limits.maxBufferSize). ' +
          "Falling back to 'manifold-nee' caustic strategy. " +
          '(This warning fires once per engine instance.)',
        details: {
          targetBytes: SPPM_PHOTON_CELLS_BYTES,
          ceilingBytes: sppmCeiling,
          deviceMaxBuffer,
          deviceMaxBinding,
          fallback: 'manifold-nee',
        },
      });
      return false;
    }
    // Destroy any placeholder buffers before allocating the real ones.
    this.#sppm.sppmPhotonCellsBuffer?.destroy();
    this.#sppm.sppmCellCountersBuffer?.destroy();
    this.#sppm.sppmStatsBuffer?.destroy();
    this.#sppm.sppmPhotonCellsBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.sppm.photonCells',
      size: SPPM_PHOTON_CELLS_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.#sppm.sppmCellCountersBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.sppm.cellCounters',
      size: SPPM_CELL_COUNTERS_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.#sppm.sppmStatsBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.sppm.stats',
      size: SPPM_STATS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // New SPPM buffers — invalidate group-3 so it rebuilds with the real handles.
    this.invalidateGroup3BindGroup();
    this.#sppm.sppmBuffersReady = true;

    // Build the photon-emission pipeline lazily.
    // The photon pass uses groups 0–3.  Group 3 now carries the SPPM bindings
    // (6/7/8) in addition to the light-tree / material-texture entries (0–5).
    // Its pipeline layout is [g0, g1, g2, g3] — the SAME 4-group layout as the
    // megakernel.  This is safe on ALL adapters (maxBindGroups = 4 is guaranteed
    // by the WebGPU spec).  Must be called AFTER ensurePipeline().
    if (this.#sppm.sppmPhotonPipeline == null) {
      if (
        this.bindGroupLayout != null &&
        this.bindGroupLayout1 != null &&
        this.bindGroupLayout2 != null &&
        this.bindGroupLayout3 != null
      ) {
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
        this.#sppm.sppmPhotonPipeline = this.#device.createComputePipeline({
          label: 'vitrum.pt-webgpu.sppm.photonPass.pipeline',
          layout: photonLayout,
          compute: { module, entryPoint: 'sppmEmitPhotons' },
        });
      }
    }
    return true;
  }

  // ── B12 — Lite-tier texture upload ──────────────────────────────────────────

  /**
   * B12 — (Re)allocate and upload the lite-tier packed textures for the current
   * scene.  Called from `index.ts` after `uploadSceneBuffers` (full or incremental
   * rebuild) whenever the trace tier is `'lite'`.
   *
   * Destroys any previously-allocated lite textures before re-creating them so
   * the bind group always holds fresh views.  Invalidates the cached group-0 bind
   * group so `buildBindGroups` will rebuild it with the new texture views.
   *
   * No-op on the full tier (lite textures are null/unused there).
   */
  uploadLiteTextures(
    lightData: LiteLightTexData,
    envData:   LiteEnvTexData,
    cdfData:   LiteEnvCdfData,
  ): void {
    if (this.#traceTier !== 'lite') return;

    // Destroy previous textures.
    this.liteEnvTexture?.destroy();
    this.liteEnvCdfTexture?.destroy();
    this.liteLightTexture?.destroy();

    // Create and upload env radiance+pdf texture (envWidth × envHeight, RGBA32F).
    this.liteEnvTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.lite.envTex',
      size: { width: envData.width, height: envData.height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.#device.queue.writeTexture(
      { texture: this.liteEnvTexture },
      // Cast: packed texels are always new Float32Array() with a plain ArrayBuffer.
      envData.texels as unknown as Float32Array<ArrayBuffer>,
      { bytesPerRow: envData.width * 16, rowsPerImage: envData.height },
      { width: envData.width, height: envData.height },
    );
    this.liteEnvTextureView = this.liteEnvTexture.createView();

    // Create and upload env CDF texture (envWidth × envHeight, RGBA32F, .r = CDF).
    this.liteEnvCdfTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.lite.envCdfTex',
      size: { width: cdfData.width, height: cdfData.height, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.#device.queue.writeTexture(
      { texture: this.liteEnvCdfTexture },
      cdfData.data as unknown as Float32Array<ArrayBuffer>,
      { bytesPerRow: cdfData.width * 16, rowsPerImage: cdfData.height },
      { width: cdfData.width, height: cdfData.height },
    );
    this.liteEnvCdfTextureView = this.liteEnvCdfTexture.createView();

    // Create and upload light data texture (liteLightTexWidth × 1, RGBA32F).
    this.liteLightTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.lite.lightTex',
      size: { width: lightData.width, height: 1, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.#device.queue.writeTexture(
      { texture: this.liteLightTexture },
      lightData.data as unknown as Float32Array<ArrayBuffer>,
      { bytesPerRow: lightData.width * 16, rowsPerImage: 1 },
      { width: lightData.width, height: 1 },
    );
    this.liteLightTextureView = this.liteLightTexture.createView();

    // Invalidate group-0 so buildBindGroups picks up the new texture views.
    this.pathTraceBindGroup = null;
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
    if (
      this.#sppm.sppmPixelStatsBuffer != null &&
      this.#sppm.sppmPixelStatsWidth === 0 &&
      this.#sppm.sppmPixelStatsHeight === 0
    ) {
      return;
    }
    this.#sppm.sppmPixelStatsBuffer?.destroy();
    this.#sppm.sppmPixelStatsBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.sppm.pixelStats.placeholder',
      size: 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
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
    f[5] = 0; f[6] = 0; f[7] = 0; // _pad
    this.#device.queue.writeBuffer(this.#sppm.sppmStatsBuffer, 0, ubo);
  }

  /**
   * A4-progressive — (Re)allocate the per-pixel SPPM statistics buffer to
   * `width × height`.  Returns `true` when the buffer exists at the requested
   * dims (freshly created or already cached); `false` on the lite tier or when
   * the requested size exceeds the device's maxBufferSize / maxStorageBufferBindingSize.
   *
   * On (re)allocation the buffer is GPU-cleared so every pixel's (τ, R², N)
   * starts at zero — the gather treats R²=0 as "first frame" and seeds from r₀².
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
    // Guard against device limits (same discipline as BDPT eye-stack / photon cells).
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
          'SPPM progressive stats disabled — caustic quality will be reduced. ' +
          '(This warning fires once per engine instance.)',
        details: {
          width,
          height,
          targetBytes,
          ceilingBytes: Math.min(deviceMaxBuffer, deviceMaxBinding),
          deviceMaxBuffer,
          deviceMaxBinding,
          fallback: 'disable-sppm-progressive-stats',
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
    // Reallocate.
    this.#sppm.sppmPixelStatsBuffer?.destroy();
    this.#sppm.sppmPixelStatsBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.sppm.pixelStats',
      size: targetBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.#sppm.sppmPixelStatsWidth  = width;
    this.#sppm.sppmPixelStatsHeight = height;
    // GPU-clear: all zeros → τ=0, R²=0 (→ seed r₀ on first frame), N=0.
    const enc = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.sppm.pixelStats.clear' });
    enc.clearBuffer(this.#sppm.sppmPixelStatsBuffer);
    this.#device.queue.submit([enc.finish()]);
    // New buffer handle — invalidate group-3 so buildBindGroups picks it up.
    this.invalidateGroup3BindGroup();
    return true;
  }

  /**
   * GPU-clear the per-pixel SPPM statistics buffer (τ/R²/N → 0).
   * Called whenever the PT accumulator resets (camera move / setScene / reset())
   * so the progressive estimate restarts from a clean state — consistent with
   * SPPM's static-eye-point assumption (accumulation only proceeds on a still view).
   * No-op if the buffer has not been allocated yet.
   */
  clearSppmPixelStats(): void {
    if (this.#sppm.sppmPixelStatsBuffer == null || this.#sppm.sppmPixelStatsWidth === 0) return;
    const enc = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.sppm.pixelStats.reset' });
    enc.clearBuffer(this.#sppm.sppmPixelStatsBuffer);
    this.#device.queue.submit([enc.finish()]);
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
   *
   * Uses `layout: 'auto'` (the present pass has no cross-pipeline layout sharing).
   */
  ensurePresentPipeline(): void {
    if (this.#present.presentPipeline != null) return;
    const module = this.#device.createShaderModule({
      label: 'vitrum.pt-webgpu.present',
      code: PT_WEBGPU_PRESENT_WGSL,
    });
    this.#present.presentPipeline = this.#device.createComputePipeline({
      label: 'vitrum.pt-webgpu.present.pipeline',
      layout: 'auto',
      compute: { module, entryPoint: 'presentMain' },
    });
    this.#present.presentParamsBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.present.params',
      size: GpuResources.PRESENT_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
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
  dispatchPresentPass(encoder: GPUCommandEncoder, width: number, height: number): void {
    if (
      this.#present.presentPipeline == null ||
      this.#present.presentParamsBuffer == null ||
      this.accumTexture == null ||
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
        { binding: 1, resource: this.accumTexture.createView() },
        { binding: 2, resource: this.#present.presentView },
      ],
    });
    const pass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.present.pass' });
    pass.setPipeline(this.#present.presentPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), 1);
    pass.end();
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
    this.bdptEyeStackBuffer?.destroy();
    this.bdptEyeStackBuffer = null;
    this.bdptEyeStackByteSize = 0;
    // D8.1: PresentResources.dispose() tears down seed-blit resources (in order:
    // seedBlitParamsBuffer → seedBlitVarPlaceholder → seedBlitSampler → seedBlitPipeline)
    // then present-pass resources (presentParamsBuffer → presentPipeline).
    // presentTexture/presentView are already nulled by destroyAccumTexture() above.
    this.#present.dispose();
    this.paramsBuffer?.destroy();
    this.paramsBuffer = null;
    this.computePipeline = null;
    this.bdptSubpathPipeline = null;
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
