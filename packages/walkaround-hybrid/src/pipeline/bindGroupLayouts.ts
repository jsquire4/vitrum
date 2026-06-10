/**
 * Bind group layout factories — one function per layout used by the
 * WalkaroundGPUPipeline. Each function is memoised on first call via the
 * `cache` argument (a plain object on the class) so the GPUDevice only
 * allocates each layout once.
 *
 * Layouts:
 *   frame   — per-frame G-buffer textures + reservoir buffers + HDR output
 *   scene   — static BVH + emitter buffers
 *   ubo     — 256-byte WalkaroundUBO uniform
 *   atrous  — denoiser I/O textures + per-pass UBO
 *   accum   — temporal accumulator I/O textures + AccumUBO
 *   composite — final blit (fragment stage, unfilterable-float + sampler)
 *   hybridLayers — DDGI atlas textures + grid uniform (shade pass slot 3)
 *
 * T9-stepB: the *uniform* families (frame/scene/ubo/gtao/gtaoUpsample/
 * temporalGi/spatialGi/indirectCombine/indirectTemporalAccum/motionVectors/
 * resolve/sampleBudget/composite) now derive their layout entries from the
 * single descriptor table in `bindGroupDescriptors.ts` via {@link bglEntriesFor}
 * — same table the builders consume, so the two can no longer drift. The three
 * NON-uniform layouts (atrous / accum / hybridLayers — lazy-UBO or
 * placeholder-fallback builders) stay hand-written below.
 */

import { bglEntriesFor } from './bindGroupDescriptors.js';

export interface BGLCache {
  frame?: GPUBindGroupLayout;
  scene?: GPUBindGroupLayout;
  ubo?: GPUBindGroupLayout;
  atrous?: GPUBindGroupLayout;
  composite?: GPUBindGroupLayout;
  accum?: GPUBindGroupLayout;
  hybridLayers?: GPUBindGroupLayout;
  /** Sprint 9 — sample-budget pass bind group layout. */
  sampleBudget?: GPUBindGroupLayout;
  /** Sprint 9 — resolve pass bind group layout. */
  resolve?: GPUBindGroupLayout;
  /** Motion-vector generation pass bind group layout. */
  motionVectors?: GPUBindGroupLayout;
  /** Sprint 15 — GTAO half-res compute pass bind group layout. */
  gtao?: GPUBindGroupLayout;
  /** Sprint 15 — GTAO bilateral upsample pass bind group layout. */
  gtaoUpsample?: GPUBindGroupLayout;
  /** Sprint 17 — GI temporal-reuse pass bind group layout. */
  temporalGi?: GPUBindGroupLayout;
  /** Sprint 17 — GI spatial-reuse pass bind group layout. */
  spatialGi?: GPUBindGroupLayout;
  /** Sprint 18 — indirect-blur + combine pass bind group layout. */
  indirectCombine?: GPUBindGroupLayout;
  /** Sprint 18 follow-up — indirect-channel pre-atrous temporal accumulator. */
  indirectTemporalAccum?: GPUBindGroupLayout;
  /** Light-tree DI light-selection BGL (RIS-only group 3). */
  lightTree?: GPUBindGroupLayout;
  /** ReGIR grid-build pass BGL (own group 0: combined buffer rw + emitters + ubo). */
  regirBuild?: GPUBindGroupLayout;
  /** NRC (Müller 2021) gi-ris @group(4) BGL — present ONLY when nrcEnabled is
   *  compile-time on (MLP weights/biases + hash tables + level descs + record
   *  gather + encoding-config UBO). */
  nrc?: GPUBindGroupLayout;
}

// frame BGL entries (incl. inert/placeholder slots 0-4 + shade-only 10/12/13/14/15)
// are declared in bindGroupDescriptors.ts with per-binding rationale notes.
export function getFrameBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.frame) return cache.frame;
  cache.frame = device.createBindGroupLayout({
    label: 'frame-bgl',
    entries: bglEntriesFor('frame'),
  });
  return cache.frame;
}

export function getSceneBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.scene) return cache.scene;
  cache.scene = device.createBindGroupLayout({
    label: 'scene-bgl',
    entries: bglEntriesFor('scene'),
  });
  return cache.scene;
}

/**
 * Light-tree DI light-SELECTION bind group layout — a RIS-ONLY 4th bind group
 * (group 3). Deliberately NOT folded into the shared `scene` group: the shade
 * pass already references 16 storage buffers (4 frame + 11 scene + 1 RC
 * cascade0), exactly at the `maxStorageBuffersPerShaderStage = 16` full-tier
 * floor, so adding a 12th scene-group storage buffer would push shade to 17 and
 * fail pipeline creation. As a separate RIS-only group, the tree adds its 1
 * storage buffer only to the RIS pipeline layout (frame 4 + scene 11 + tree 1 =
 * 16, at the floor), leaving temporal/spatial/shade untouched. RIS uses
 * `maxBindGroups = 4` (frame/scene/ubo/lightTree), within the Lovelace cap.
 *
 *   0 — light-tree node buffer (read-only-storage, flat array<f32>)
 */
export function getLightTreeBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.lightTree) return cache.lightTree;
  cache.lightTree = device.createBindGroupLayout({
    label: 'light-tree-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
  });
  return cache.lightTree;
}

/**
 * ReGIR grid-build pass BGL — a DEDICATED group(0) bound only by the grid-build
 * compute pipeline. It writes the ReGIR grid region of the COMBINED light-tree
 * buffer and reads the tree region + the emitter list.
 *
 *   0 — combined light-tree + ReGIR-grid buffer (READ_WRITE storage). The SAME
 *       GPUBuffer RIS binds read-only at its group(3) binding 0; binding it
 *       read_write here (different bind group, different access) keeps RIS at
 *       16 storage buffers while letting the build pass write the grid region.
 *   1 — emitters (read-only storage) — for the per-cell target q̂_c eval.
 *   2 — WalkaroundUBO (uniform) — grid geometry + M/K + frameSeed + gate.
 *
 * The grid-build pipeline uses a SINGLE bind group (group 0), so it consumes
 * only 2 storage buffers — far under any tier floor.
 */
/**
 * NRC (Müller et al. 2021) gi-ris @group(4) bind group layout. Present ONLY on
 * the gi-ris pipeline when `nrcEnabled` is compile-time on; the default gi-ris
 * pipeline (4 groups) never references it, so the default pipeline structure is
 * byte-for-byte pre-NRC (the GRIS-class regression discipline — f8df9a4).
 *
 *   0 — MLP weights      (read-only storage, f32) — concatenated weight matrices
 *   1 — MLP biases       (read-only storage, f32)
 *   2 — hash-grid tables (read-only storage, f32) — trainable feature tables
 *   3 — level descriptors (read-only storage, NrcLevelDesc)
 *   4 — record gather     (read_write storage, f32) — self-training records
 *   5 — encoding config   (uniform, NrcCfgUBO)
 *
 * Storage-buffer budget on the gi-ris pipeline (NRC ON): gi-ris reuses the shade
 * layout (frame/scene/ubo/hybrid) whose scene+frame groups carry 16 storage
 * buffers at the full-tier floor — BUT @group(4) adds 5 MORE storage buffers,
 * which would push gi-ris to 21 > the 16 floor. So unlike GRIS (which kept under
 * the floor), the NRC gi-ris layout must NOT reuse the 16-buffer shade layout's
 * scene group verbatim if it also binds 5 NRC storage buffers. This is handled
 * in compilePipelines by binding NRC as a 5th group on a layout that the device
 * accepts (full-tier maxStorageBuffersPerShaderStage is the gate; NRC is
 * full-tier-only and the host must confirm the budget — see V20). The 4 NRC
 * storage buffers + 1 uniform here are declared read-only except the record
 * gather, matching nrcQuery.wgsl.
 */
export function getNrcBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.nrc) return cache.nrc;
  cache.nrc = device.createBindGroupLayout({
    label: 'nrc-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      // H27 — per-slot atomic claim flags (one u32 per recordCap slot). The
      // host clears this buffer to zero each frame. The GPU shader uses
      // atomicCompareExchangeWeak to claim a slot before writing the record,
      // preventing torn records when two invocations alias to the same slot.
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  return cache.nrc;
}

export function getRegirBuildBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.regirBuild) return cache.regirBuild;
  cache.regirBuild = device.createBindGroupLayout({
    label: 'regir-build-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  return cache.regirBuild;
}

// ubo BGL entries (incl. inert slot 2 — adaptive-sampling tier, read only by
// risGi) are declared in bindGroupDescriptors.ts with per-binding notes.
export function getUboBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.ubo) return cache.ubo;
  cache.ubo = device.createBindGroupLayout({
    label: 'ubo-bgl',
    entries: bglEntriesFor('ubo'),
  });
  return cache.ubo;
}

export function getAtrousBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.atrous) return cache.atrous;
  cache.atrous = device.createBindGroupLayout({
    label: 'atrous-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  return cache.atrous;
}

export function getCompositeBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.composite) return cache.composite;
  cache.composite = device.createBindGroupLayout({
    label: 'composite-bgl',
    entries: bglEntriesFor('composite'),
  });
  return cache.composite;
}

export function getAccumBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.accum) return cache.accum;
  cache.accum = device.createBindGroupLayout({
    label: 'accum-bgl',
    entries: [
      // 0: currentAtrous (read)
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      // 1: prevAccum (read)
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      // 2: accumOut (write)
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      // 3: AccumUBO
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  return cache.accum;
}

/**
 * Phase 2B — combined hybrid-layers bind group (slot 3, shade only).
 * DDGI + RC inputs packed into one group because Lovelace's adapter
 * caps `maxBindGroups = 4` (verified empirically); a 5th group is
 * rejected. maxBindingsPerBindGroup is 1000 so 6 bindings is fine.
 * Layout:
 *   DDGI section
 *     0 — irradiance atlas (texture_2d<f32>, unfilterable)
 *     1 — visibility atlas (texture_2d<f32>, unfilterable)
 *     2 — non-filtering sampler
 *     3 — DDGI grid uniform (64 bytes)
 *   RC section (W8 Phase 3, 2026-05-18)
 *     4 — cascade-0 storage buffer (read-only) — `array<vec4f>` packed
 *         (probeX·probeY·probeZ·rays) entries. Always bound; a 1-vec4f
 *         placeholder backs the slot when RC is disabled.
 *     5 — RCParams uniform (64 bytes) — probeOrigin/roomSize/probeCount/
 *         raysPerProbe/rayGridSize + rcWeight + enabled bit. The
 *         `enabled == 0u` short-circuits sampleCascadeC0 to vec3f(0),
 *         so the same bind group works for rcEnabled=true and false
 *         without a pipeline recompile.
 *   PPG section (W9 guided-sampling landing)
 *     6 — PPG sTree storage buffer (read-only) — serialised spatial kd-tree.
 *     7 — PPG dTree storage buffer (read-only) — concatenated per-cell
 *         directional quadtrees (the learned guiding distribution).
 *     8 — PPG dTreeOffsets storage buffer (read-only) — sTree-cell →
 *         dTreeBuf base-offset table.
 *         All three are ALWAYS bound; a 16-byte zeroed placeholder backs
 *         each slot when PPG is disabled. gi-ris reads them only when
 *         `ubo.ppgEnabled == 1` (the kernel guards on the gate before any
 *         dTree descent), so the placeholders are never dereferenced when
 *         PPG is off.
 *
 * shade.wgsl reads bindings 0-5; risGi.wgsl reads 0-3 (DDGI) + 6-8 (PPG).
 * WebGPU spec allows pipelines to reference a subset of layout entries, so
 * the unified BGL works for both. Bind groups must still provide resources
 * for all 9 entries (placeholders for unused). The PPG storage buffers are
 * read-only-storage, so they bind against the STORAGE-flagged PPG buffers
 * (or placeholders) without any usage-flag change.
 *
 * Storage-buffer budget: in TLAS mode gi-ris references 9 storage buffers
 * (1 frame reservoir + 8 scene BVH/TLAS) + these 3 PPG buffers = 12, under
 * the `HYBRID_WEBGPU_REQUIRED_LIMITS.maxStorageBuffersPerShaderStage = 16`
 * full-tier floor. PPG is forbidden on the lite tier, so its lower (10)
 * floor is never asked to host the PPG buffers.
 */
export function getHybridLayersBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.hybridLayers) return cache.hybridLayers;
  cache.hybridLayers = device.createBindGroupLayout({
    label: 'hybrid-layers-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'non-filtering' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      // PPG guided-sampling tree buffers (W9). Read-only-storage; gi-ris
      // descends them when ubo.ppgEnabled == 1.
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
  });
  return cache.hybridLayers;
}

/**
 * Sprint 9 — sample-budget BGL. Matches the @group(0) bindings in
 * sampleBudget.wgsl.ts:
 *   0 — SampleBudgetUniforms ubo (thresholds + screen size)
 *   1 — variance source (rg32float, sampled, unfilterable)
 *   2 — tier output (r32uint, write-only storage)
 *   3 — SampleCountUniforms ubo (sample-count counter)
 */
export function getSampleBudgetBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.sampleBudget) return cache.sampleBudget;
  cache.sampleBudget = device.createBindGroupLayout({
    label: 'sample-budget-bgl',
    entries: bglEntriesFor('sampleBudget'),
  });
  return cache.sampleBudget;
}

/**
 * Sprint 9 — resolve BGL. Matches the @group(0) bindings in
 * resolve.wgsl.ts:
 *   0 — ResolveUniforms ubo (screen size + frame parity)
 *   1 — current radiance (rgba16float, sampled, unfilterable)
 *   2 — previous radiance (rgba16float, sampled, unfilterable)
 *   3 — motion vectors (rg32float, sampled, unfilterable)
 *   4 — resolved out (rgba16float, write-only storage)
 */
export function getResolveBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.resolve) return cache.resolve;
  cache.resolve = device.createBindGroupLayout({
    label: 'resolve-bgl',
    entries: bglEntriesFor('resolve'),
  });
  return cache.resolve;
}

/**
 * Motion-vectors pass BGL. Matches `motionVectors.wgsl.ts`:
 *   0 — gNormalDepth in (rgba16float sampled, unfilterable)
 *   1 — motion out (rg32float write-only storage)
 *   2 — WalkaroundUBO (uniform)
 */
export function getMotionVectorsBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.motionVectors) return cache.motionVectors;
  cache.motionVectors = device.createBindGroupLayout({
    label: 'motion-vectors-bgl',
    entries: bglEntriesFor('motionVectors'),
  });
  return cache.motionVectors;
}

/**
 * Sprint 15 — GTAO half-res compute pass BGL. Matches `gtao.wgsl.ts`:
 *   0 — gNormalDepth (rgba16float, sampled)
 *   1 — aoHalf out (rgba16float, write-only storage — E1 multi-bounce: bumped from r16float)
 *   2 — GTAOUniforms ubo
 *   3 — gtao_albedo (rgba16float, sampled — E1 Jiménez 2016 §5.2 multi-bounce term)
 */
export function getGTAOBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.gtao) return cache.gtao;
  cache.gtao = device.createBindGroupLayout({
    label: 'gtao-bgl',
    entries: bglEntriesFor('gtao'),
  });
  return cache.gtao;
}

/**
 * Sprint 15 — GTAO bilateral upsample BGL. Matches `gtaoUpsample.wgsl.ts`:
 *   0 — aoHalf in (rgba16float, sampled — carries per-channel multi-bounce AO)
 *   1 — gNormalDepth (rgba16float, sampled)
 *   2 — aoFull out (rgba16float, write-only storage — Tier-G fix: per-channel
 *       Jiménez 2016 §5.2 AO stored in `.rgb`; previously collapsed to a
 *       luminance scalar in `.r`)
 *   3 — GTAOUniforms (uniform, audit B3 — for bilateralDepthSigma)
 */
export function getGTAOUpsampleBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.gtaoUpsample) return cache.gtaoUpsample;
  cache.gtaoUpsample = device.createBindGroupLayout({
    label: 'gtao-upsample-bgl',
    entries: bglEntriesFor('gtaoUpsample'),
  });
  return cache.gtaoUpsample;
}

/**
 * Sprint 17 — temporal GI BGL. Matches `temporalGi.wgsl.ts`:
 *   0 — reservoirGiCurrent  (storage, read_write)
 *   1 — reservoirGiPrevious (storage, read)
 *   2 — WalkaroundUBO       (uniform)
 */
export function getTemporalGiBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.temporalGi) return cache.temporalGi;
  cache.temporalGi = device.createBindGroupLayout({
    label: 'temporal-gi-bgl',
    entries: bglEntriesFor('temporalGi'),
  });
  return cache.temporalGi;
}

/**
 * Sprint 17 — spatial GI BGL. Two ping-pong bind groups share this layout:
 *   pass 1 → in = reservoirGiCurrent, out = reservoirGiSpatial
 *   pass 2 → in = reservoirGiSpatial, out = reservoirGiCurrent
 * Matches `spatialGi.wgsl.ts`:
 *   0 — input  reservoir (storage, read)
 *   1 — output reservoir (storage, read_write)
 *   2 — WalkaroundUBO    (uniform)
 */
export function getSpatialGiBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.spatialGi) return cache.spatialGi;
  cache.spatialGi = device.createBindGroupLayout({
    label: 'spatial-gi-bgl',
    entries: bglEntriesFor('spatialGi'),
  });
  return cache.spatialGi;
}

/**
 * Sprint 18 follow-up — indirect temporal accumulator BGL.  Matches
 * `indirectTemporalAccum.wgsl.ts`:
 *   0 — currentRaw (rgba16float sampled, hdrIndirectTexture)
 *   1 — prevAccum  (rgba16float sampled, previous frame's accumulator output)
 *   2 — outAccum   (rgba16float storage write, this frame's accumulator output)
 */
export function getIndirectTemporalAccumBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.indirectTemporalAccum) return cache.indirectTemporalAccum;
  cache.indirectTemporalAccum = device.createBindGroupLayout({
    label: 'indirect-temporal-accum-bgl',
    entries: bglEntriesFor('indirectTemporalAccum'),
  });
  return cache.indirectTemporalAccum;
}

/**
 * Sprint 18 — indirect-combine BGL. Matches `indirectCombine.wgsl.ts`:
 *   0 — denoisedDirect (rgba16float, sampled, unfilterable)
 *   1 — hdrIndirect    (rgba16float, sampled, unfilterable)
 *   2 — combinedOut    (rgba16float, write-only storage)
 *   3 — albedo         (rgba16float, sampled, unfilterable) — Item 24
 *
 * W5-I2 cleanup (2026-05-18): the previous slot-2 `gNormalDepth` entry was
 * declared "for BGL compat" but never read by the shader; dropped along
 * with its host-side BGL entry + builder argument.
 */
export function getIndirectCombineBindGroupLayout(
  device: GPUDevice,
  cache: BGLCache,
): GPUBindGroupLayout {
  if (cache.indirectCombine) return cache.indirectCombine;
  cache.indirectCombine = device.createBindGroupLayout({
    label: 'indirect-combine-bgl',
    entries: bglEntriesFor('indirectCombine'),
  });
  return cache.indirectCombine;
}
