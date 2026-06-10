/**
 * Bind group builders — construct GPUBindGroup instances from the per-frame
 * resource handles held by WalkaroundGPUPipeline.
 *
 * All functions are pure in the sense that they only read their arguments and
 * call device.createBindGroup(); no mutable state is kept here. The one
 * exception is buildAtrousBindGroup / buildAccumBindGroup, which lazily
 * create a small per-builder UBO via the `uboRef` out-param object — callers
 * pass a wrapper `{ buf: GPUBuffer | undefined }` that is initialised on the
 * first call and reused thereafter.
 */

import { defineUbo } from '@vitrum/shared-samplers';
import {
  getFrameBindGroupLayout,
  getSceneBindGroupLayout,
  getUboBindGroupLayout,
  getAtrousBindGroupLayout,
  getAccumBindGroupLayout,
  getCompositeBindGroupLayout,
  getHybridLayersBindGroupLayout,
  getSampleBudgetBindGroupLayout,
  getResolveBindGroupLayout,
  getMotionVectorsBindGroupLayout,
  getGTAOBindGroupLayout,
  getGTAOUpsampleBindGroupLayout,
  getTemporalGiBindGroupLayout,
  getSpatialGiBindGroupLayout,
  getIndirectCombineBindGroupLayout,
  getIndirectTemporalAccumBindGroupLayout,
  getLightTreeBindGroupLayout,
  getRegirBuildBindGroupLayout,
  type BGLCache,
} from './bindGroupLayouts.js';
import { buildBindGroupFromTable } from './bindGroupDescriptors.js';

interface TextureViewCache {
  textureView(texture: GPUTexture): GPUTextureView;
}

function textureView(texture: GPUTexture, cache?: TextureViewCache): GPUTextureView {
  return cache?.textureView(texture) ?? texture.createView();
}

// ─── W2-C13 follow-up: UBO codegen for builder-managed UBOs ───────────────────
// AtrousUBO (atrous.wgsl.ts): {stepWidth, sigmaN, sigmaZ, sigmaC} — 4×f32 = 16 B.
const ATROUS_UBO = defineUbo([
  { name: 'stepWidth', type: 'f32' },
  { name: 'sigmaN',    type: 'f32' },
  { name: 'sigmaZ',    type: 'f32' },
  { name: 'sigmaC',    type: 'f32' },
] as const);
// AccumUBO (temporalAccum.wgsl.ts): {alpha, _pad1, _pad2, _pad3} — 1 active f32
// padded to the 16-byte WebGPU minimum-binding floor. Only `alpha` is read by
// the shader; the three trailing pads are zero-filled by defineUbo.pack.
const ACCUM_UBO = defineUbo([
  { name: 'alpha', type: 'f32' },
] as const);

// ── Frame bind group ─────────────────────────────────────────────────────────

interface FrameBindGroupResources {
  placeholderView: GPUTextureView;
  reservoirCurrentBuffer: GPUBuffer;
  reservoirPreviousBuffer: GPUBuffer;
  reservoirSpatialBuffer: GPUBuffer;
  hdrColorTexture: GPUTexture;
  nearestSampler: GPUSampler;
  gNormalDepthTexture: GPUTexture;
  /** Sprint 16 — half-res GI reservoir (write target for risGi, read for shade). */
  reservoirGiCurrentBuffer: GPUBuffer;
  /** Sprint 18 — indirect-channel HDR output texture (shade write target). */
  hdrIndirectTexture: GPUTexture;
  /** Sprint 18 follow-up — total-radiance HDR output (shade write, welford read). */
  hdrTotalTexture: GPUTexture;
  /** Item 24 — visible-point diffuse albedo (shade write, indirectCombine read). */
  albedoTexture: GPUTexture;
  /** SVGF-real current object ID (shade write, SVGF reprojection read). */
  svgfCurrentObjectIdTexture: GPUTexture;
}

// Positional resource order MUST match the 'frame' descriptor table in
// bindGroupDescriptors.ts (which carries the per-binding rationale notes —
// incl. the inert/placeholder G-buffer slots 0-4 and shade-only 10/12/13/14/15).
export function buildFrameBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  r: FrameBindGroupResources,
  viewCache?: TextureViewCache,
): GPUBindGroup {
  return buildBindGroupFromTable(device, 'frame', getFrameBindGroupLayout(device, cache), [
    r.placeholderView,                          // 0 gDepth (placeholder)
    r.placeholderView,                          // 1 gNormal (placeholder)
    r.placeholderView,                          // 2 gAlbedo (placeholder)
    r.placeholderView,                          // 3 gRough (placeholder)
    r.placeholderView,                          // 4 motionVec (placeholder)
    { buffer: r.reservoirCurrentBuffer },       // 5
    { buffer: r.reservoirPreviousBuffer },      // 6
    { buffer: r.reservoirSpatialBuffer },       // 7
    textureView(r.hdrColorTexture, viewCache),  // 8
    r.nearestSampler,                           // 9
    textureView(r.gNormalDepthTexture, viewCache), // 10 gNormalDepth (shade-write only)
    { buffer: r.reservoirGiCurrentBuffer },     // 11 GI reservoir (risGi-write/shade-read)
    textureView(r.hdrIndirectTexture, viewCache), // 12 hdrIndirect (shade-write only)
    textureView(r.hdrTotalTexture, viewCache),  // 13 hdrTotal (shade-write only)
    textureView(r.albedoTexture, viewCache),    // 14 albedo (shade-write only)
    textureView(r.svgfCurrentObjectIdTexture, viewCache), // 15 SVGF current object ID
  ]);
}

// ── Scene bind group ─────────────────────────────────────────────────────────

interface SceneBindGroupResources {
  bvhNodesBuffer: GPUBuffer;
  bvhIndexBuffer: GPUBuffer;
  bvhPositionBuffer: GPUBuffer;
  emitterBuffer: GPUBuffer;
  emitterCdfBuffer: GPUBuffer;
  /** WS1 — beer is now a uint texture (binding 5), not a storage buffer. */
  bvhBeerTextureView: GPUTextureView;
  /** WS1 — per-vertex world-space normals storage buffer (binding 11). */
  bvhNormalBuffer: GPUBuffer;
  /** Camera-visible emitters — per-tri HDR emissive Le, rgba32float texture
   *  (binding 12). Shade reads it via `textureLoad` (lo_emitterGlow). */
  bvhEmissiveTextureView: GPUTextureView;
  /** B1 — per-tri roughness+metalness, r32uint texture (binding 14). */
  bvhRoughMetalTextureView: GPUTextureView;
  tlasNodesBuffer: GPUBuffer;
  tlasInstanceIndicesBuffer: GPUBuffer;
  tlasBlasRootsBuffer: GPUBuffer;
  tlasInstanceWorldToLocalBuffer: GPUBuffer;
  tlasInstanceLocalToWorldBuffer: GPUBuffer;
  /** H41 — packed point/spot analytic lights (binding 13). 64-byte stride. */
  analyticLightsBuffer: GPUBuffer;
  /** B3 — directional IBL (bindings 15-19). Placeholders + envParams.hasEnv=0
   *  for non-HDRI scenes (scalar-tint fallback, no-HDRI byte-identity). */
  envMapTextureView: GPUTextureView;
  envMarginalTextureView: GPUTextureView;
  envConditionalTextureView: GPUTextureView;
  envSampler: GPUSampler;
  envParamsBuffer: GPUBuffer;
}

export function buildSceneBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  r: SceneBindGroupResources,
): GPUBindGroup {
  return buildBindGroupFromTable(device, 'scene', getSceneBindGroupLayout(device, cache), [
    { buffer: r.bvhNodesBuffer },                   // 0
    { buffer: r.bvhIndexBuffer },                   // 1 vec4u: [0..2]=indices, [3]=RGBA8 raw attCol
    { buffer: r.bvhPositionBuffer },                // 2
    { buffer: r.emitterBuffer },                    // 3
    { buffer: r.emitterCdfBuffer },                 // 4
    r.bvhBeerTextureView,                           // 5 WS1 r32uint texture: per-tri Beer-Lambert color
    { buffer: r.tlasNodesBuffer },                  // 6
    { buffer: r.tlasInstanceIndicesBuffer },        // 7
    { buffer: r.tlasBlasRootsBuffer },              // 8
    { buffer: r.tlasInstanceWorldToLocalBuffer },   // 9
    { buffer: r.tlasInstanceLocalToWorldBuffer },   // 10
    { buffer: r.bvhNormalBuffer },                  // 11 WS1 per-vertex world-space smooth normals
    r.bvhEmissiveTextureView,                       // 12 camera-visible emitters: per-tri HDR emissive Le
    { buffer: r.analyticLightsBuffer },             // 13 H41 analytic point/spot lights for shade NEE
    r.bvhRoughMetalTextureView,                     // 14 B1 per-tri roughness+metalness (r32uint texture)
    r.envMapTextureView,                            // 15 B3 directional IBL radiance + per-texel pdf
    r.envMarginalTextureView,                       // 16 B3 marginal inverse-CDF
    r.envConditionalTextureView,                    // 17 B3 conditional inverse-CDF
    r.envSampler,                                   // 18 B3 env sampler (textureLoad path; declared for layout)
    { buffer: r.envParamsBuffer },                  // 19 B3 EnvParams uniform
  ]);
}

// ── Light-tree bind group (RIS-only group 3) ─────────────────────────────────

/**
 * Build the RIS-only light-tree bind group (group 3, binding 0 = node buffer).
 * Bound only by the RIS pipeline so the extra storage buffer stays off the
 * shade/temporal/spatial layouts (see getLightTreeBindGroupLayout).
 */
export function buildLightTreeBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  lightTreeBuffer: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'light-tree-bg',
    layout: getLightTreeBindGroupLayout(device, cache),
    entries: [{ binding: 0, resource: { buffer: lightTreeBuffer } }],
  });
}

// ── ReGIR grid-build bind group (its own group 0) ────────────────────────────

/**
 * Build the ReGIR grid-build bind group: the COMBINED light-tree + grid buffer
 * bound READ_WRITE (binding 0 — same GPUBuffer RIS reads read-only at its
 * group(3)), the emitter list (binding 1), and the WalkaroundUBO (binding 2).
 * Bound only by the grid-build pipeline so the read_write access never touches
 * the RIS / shade layouts.
 */
export function buildRegirBuildBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  combinedLightTreeBuffer: GPUBuffer,
  emitterBuffer: GPUBuffer,
  uboBuffer: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'regir-build-bg',
    layout: getRegirBuildBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: { buffer: combinedLightTreeBuffer } },
      { binding: 1, resource: { buffer: emitterBuffer } },
      { binding: 2, resource: { buffer: uboBuffer } },
    ],
  });
}

// ── UBO bind group ───────────────────────────────────────────────────────────

export function buildUboBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  uboBuffer: GPUBuffer,
  aoFullView: GPUTextureView,
  tierView: GPUTextureView,
): GPUBindGroup {
  return buildBindGroupFromTable(device, 'ubo', getUboBindGroupLayout(device, cache), [
    { buffer: uboBuffer },  // 0
    aoFullView,             // 1 Sprint 15 — GTAO occlusion factor
    tierView,               // 2 Sprint 9 — adaptive-sampling tier (inert except risGi)
  ]);
}

// ── Atrous bind group ────────────────────────────────────────────────────────

/** Mutable wrapper so the UBO buffer is lazily created once and reused. */
export interface UboRef { buf: GPUBuffer | undefined }

/**
 * Atrous edge-stop sigmas. Defaults are the direct-channel tuning
 * (tight stops to preserve hard shadow boundaries); the Sprint 18
 * indirect chain overrides these with broader values since the indirect
 * signal is already temporally smoothed by ReSTIR-GI and tolerates
 * wider blurs across depth / normal / chroma transitions.
 */
export interface AtrousSigmas {
  sigmaN: number;
  sigmaZ: number;
  sigmaC: number;
}

/** Direct-channel default — tight stops, preserves shadow / caustic edges.
 *  Exported for AtrousDenoiser's default; per-frame overrides flow from
 *  `HybridEngineOptions.atrousDirectSigmas` through `PipelineFrameInputs`
 *  (B3a, 2026-05-19). */
export const ATROUS_DIRECT_SIGMAS: Readonly<AtrousSigmas> = Object.freeze({
  sigmaN: 128.0,
  sigmaZ: 5.0,
  sigmaC: 0.05,
});

/**
 * Indirect-channel sigmas. Broader on every axis because ReSTIR-GI already
 * smooths the indirect signal temporally + spatially; the remaining 2×2 quad
 * variance (from half-res GI reservoir reads) just needs a wide low-pass.
 *   σn=32  → still rejects perpendicular surfaces but happily blurs through
 *            mild curvature.
 *   σz=20  → ~4× the direct depth tolerance — fine for indirect, which has
 *            no hard-shadow edges to preserve.
 *   σc=0.5 → ~10× direct's color tolerance — allows blur across color-bleed
 *            transitions which are low-frequency anyway.
 *
 * Per-frame overrides flow from `HybridEngineOptions.atrousIndirectSigmas`
 * through `PipelineFrameInputs` (B3a, 2026-05-19).
 */
export const ATROUS_INDIRECT_SIGMAS: Readonly<AtrousSigmas> = Object.freeze({
  sigmaN: 32.0,
  sigmaZ: 20.0,
  sigmaC: 0.5,
});

export function buildAtrousBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  uboRef: UboRef,
  inputView: GPUTextureView,
  outputView: GPUTextureView,
  gNormalView: GPUTextureView,
  gDepthView: GPUTextureView,
  stepWidth: number,
  sigmas: Readonly<AtrousSigmas>,
): GPUBindGroup {
  if (!uboRef.buf) {
    uboRef.buf = device.createBuffer({
      size: ATROUS_UBO.sizeBytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }
  // W2-C13 follow-up: same byte layout as the prior Float32Array([stepWidth,
  // sigmaN, sigmaZ, sigmaC]) — four f32 fields at offsets 0/4/8/12.
  const uboData = new ArrayBuffer(ATROUS_UBO.sizeBytes);
  ATROUS_UBO.pack(new DataView(uboData), 0, {
    stepWidth,
    sigmaN: sigmas.sigmaN,
    sigmaZ: sigmas.sigmaZ,
    sigmaC: sigmas.sigmaC,
  });
  device.queue.writeBuffer(uboRef.buf, 0, uboData);

  return device.createBindGroup({
    label: `atrous-bg-step${stepWidth}`,
    layout: getAtrousBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: inputView },
      { binding: 1, resource: outputView },
      { binding: 2, resource: gNormalView },
      { binding: 3, resource: gDepthView },
      { binding: 4, resource: { buffer: uboRef.buf } },
    ],
  });
}

// ── Accum bind group ─────────────────────────────────────────────────────────

export function buildAccumBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  uboRef: UboRef,
  currentAtrousView: GPUTextureView,
  prevAccumView: GPUTextureView,
  accumOutView: GPUTextureView,
  alpha: number,
): GPUBindGroup {
  if (!uboRef.buf) {
    uboRef.buf = device.createBuffer({
      size: ACCUM_UBO.sizeBytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }
  // W2-C13 follow-up: AccumUBO is now {alpha, _pad1, _pad2, _pad3} — the
  // temporal accum shader uses an AABB clamp on the 3×3 neighborhood (not
  // k·std_dev), so the former varianceK slot is unused padding. defineUbo
  // zero-fills the trailing pad bytes to match the prior
  // Float32Array([alpha, 0, 0, 0]) write byte-for-byte.
  const accumUboData = new ArrayBuffer(ACCUM_UBO.sizeBytes);
  ACCUM_UBO.pack(new DataView(accumUboData), 0, { alpha });
  device.queue.writeBuffer(uboRef.buf, 0, accumUboData);
  return device.createBindGroup({
    label: 'accum-bg',
    layout: getAccumBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: currentAtrousView },
      { binding: 1, resource: prevAccumView },
      { binding: 2, resource: accumOutView },
      { binding: 3, resource: { buffer: uboRef.buf } },
    ],
  });
}

// ── Hybrid layers bind group (DDGI, shade pass slot 3) ───────────────────────

interface HybridLayersResources {
  ddgiIrrTex: GPUTexture | null;
  ddgiVisTex: GPUTexture | null;
  ddgiPlaceholderRgba16f: GPUTexture;
  ddgiPlaceholderVisRgba16f: GPUTexture;
  nearestSampler: GPUSampler;
  ddgiUboBuffer: GPUBuffer;
  // W8 Phase 3 (2026-05-18) — RC cascade-0 + params. Both fields are
  // always-present GPUBuffers (a 16-byte and a 64-byte placeholder
  // when RC is disabled) so the bind group can be built unconditionally;
  // the shader's `rcParams.enabled` bit gates whether the cascade-0
  // sample is actually integrated into Lo_indirect.
  rcCascade0Buffer: GPUBuffer;
  rcParamsBuffer:   GPUBuffer;
  // W9 guided sampling — PPG tree buffers (sTree / dTree / dTreeOffsets).
  // Always present GPUBuffers: the real STORAGE-flagged PPG buffers when PPG
  // is enabled, or a shared 16-byte zeroed placeholder when disabled. gi-ris
  // descends them only when ubo.ppgEnabled == 1, so the placeholders are
  // never dereferenced in the PPG-off path.
  ppgSTreeBuffer:        GPUBuffer;
  ppgDTreeBuffer:        GPUBuffer;
  ppgDTreeOffsetsBuffer: GPUBuffer;
}

export function buildHybridLayersBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  r: HybridLayersResources,
  viewCache?: TextureViewCache,
): GPUBindGroup {
  const irrTex = r.ddgiIrrTex ?? r.ddgiPlaceholderRgba16f;
  const visTex = r.ddgiVisTex ?? r.ddgiPlaceholderVisRgba16f;
  return device.createBindGroup({
    label: 'hybrid-layers-bg',
    layout: getHybridLayersBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: textureView(irrTex, viewCache) },
      { binding: 1, resource: textureView(visTex, viewCache) },
      { binding: 2, resource: r.nearestSampler },
      { binding: 3, resource: { buffer: r.ddgiUboBuffer } },
      { binding: 4, resource: { buffer: r.rcCascade0Buffer } },
      { binding: 5, resource: { buffer: r.rcParamsBuffer } },
      // W9 — PPG guided-sampling tree buffers.
      { binding: 6, resource: { buffer: r.ppgSTreeBuffer } },
      { binding: 7, resource: { buffer: r.ppgDTreeBuffer } },
      { binding: 8, resource: { buffer: r.ppgDTreeOffsetsBuffer } },
    ],
  });
}

// ── Composite bind group ─────────────────────────────────────────────────────

export function buildCompositeBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  texView: GPUTextureView,
  compositeSampler: GPUSampler,
): GPUBindGroup {
  return buildBindGroupFromTable(device, 'composite', getCompositeBindGroupLayout(device, cache), [
    texView,            // 0
    compositeSampler,   // 1
  ]);
}

// ── Sample-budget bind group (Sprint 9) ──────────────────────────────────────

export function buildSampleBudgetBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  varianceView: GPUTextureView,
  tierWriteView: GPUTextureView,
  budgetUbo: GPUBuffer,
  sampleCountUbo: GPUBuffer,
): GPUBindGroup {
  return buildBindGroupFromTable(device, 'sampleBudget', getSampleBudgetBindGroupLayout(device, cache), [
    { buffer: budgetUbo },        // 0
    varianceView,                 // 1 welford variance source (rg32float)
    tierWriteView,                // 2 tier output (r32uint)
    { buffer: sampleCountUbo },   // 3
  ]);
}

// ── Resolve bind group (Sprint 9) ────────────────────────────────────────────

export function buildResolveBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  resolveUbo: GPUBuffer,
  currentRadianceView: GPUTextureView,
  prevRadianceView: GPUTextureView,
  motionVectorsView: GPUTextureView,
  resolvedWriteView: GPUTextureView,
): GPUBindGroup {
  return buildBindGroupFromTable(device, 'resolve', getResolveBindGroupLayout(device, cache), [
    { buffer: resolveUbo },   // 0
    currentRadianceView,      // 1
    prevRadianceView,         // 2
    motionVectorsView,        // 3
    resolvedWriteView,        // 4
  ]);
}

export function buildMotionVectorsBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  gNormalDepthView: GPUTextureView,
  motionVectorsWriteView: GPUTextureView,
  uboBuffer: GPUBuffer,
): GPUBindGroup {
  return buildBindGroupFromTable(device, 'motionVectors', getMotionVectorsBindGroupLayout(device, cache), [
    gNormalDepthView,         // 0
    motionVectorsWriteView,   // 1
    { buffer: uboBuffer },    // 2
  ]);
}

// ── GTAO bind groups (Sprint 15) ─────────────────────────────────────────────

export function buildGTAOBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  gNormalDepthView: GPUTextureView,
  aoHalfWriteView: GPUTextureView,
  gtaoUbo: GPUBuffer,
  /** E1 — hdrAlbedoOut view for Jiménez 2016 §5.2 multi-bounce term. */
  albedoView: GPUTextureView,
): GPUBindGroup {
  return buildBindGroupFromTable(device, 'gtao', getGTAOBindGroupLayout(device, cache), [
    gNormalDepthView,       // 0
    aoHalfWriteView,        // 1
    { buffer: gtaoUbo },    // 2
    albedoView,             // 3 E1 — visible-point albedo from shade (Item 24)
  ]);
}

export function buildGTAOUpsampleBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  aoHalfReadView: GPUTextureView,
  gNormalDepthView: GPUTextureView,
  aoFullWriteView: GPUTextureView,
  gtaoUbo: GPUBuffer,
): GPUBindGroup {
  return buildBindGroupFromTable(device, 'gtaoUpsample', getGTAOUpsampleBindGroupLayout(device, cache), [
    aoHalfReadView,         // 0
    gNormalDepthView,       // 1
    aoFullWriteView,        // 2
    { buffer: gtaoUbo },    // 3
  ]);
}

// ── GI temporal + spatial bind groups (Sprint 17) ────────────────────────────

export function buildTemporalGiBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  reservoirGiCurrent: GPUBuffer,
  reservoirGiPrevious: GPUBuffer,
  uboBuffer: GPUBuffer,
): GPUBindGroup {
  return buildBindGroupFromTable(device, 'temporalGi', getTemporalGiBindGroupLayout(device, cache), [
    { buffer: reservoirGiCurrent },    // 0
    { buffer: reservoirGiPrevious },   // 1
    { buffer: uboBuffer },             // 2
  ]);
}

export function buildSpatialGiBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  inBuffer: GPUBuffer,
  outBuffer: GPUBuffer,
  uboBuffer: GPUBuffer,
  label: string,
): GPUBindGroup {
  // spatialGi uses two distinct ping-pong labels ('spatial-gi-bg-1/-2'); pass
  // the caller-supplied label through as the GPU debug-label override.
  return buildBindGroupFromTable(
    device, 'spatialGi', getSpatialGiBindGroupLayout(device, cache),
    [
      { buffer: inBuffer },    // 0
      { buffer: outBuffer },   // 1
      { buffer: uboBuffer },   // 2
    ],
    label,
  );
}

// ── Indirect temporal accumulator (Sprint 18 follow-up) ──────────────────────

export function buildIndirectTemporalAccumBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  currentRawView: GPUTextureView,
  prevAccumView: GPUTextureView,
  outAccumView: GPUTextureView,
): GPUBindGroup {
  return buildBindGroupFromTable(device, 'indirectTemporalAccum', getIndirectTemporalAccumBindGroupLayout(device, cache), [
    currentRawView,   // 0
    prevAccumView,    // 1
    outAccumView,     // 2
  ]);
}

// ── Indirect-combine bind group (Sprint 18) ──────────────────────────────────

export function buildIndirectCombineBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  denoisedDirectView: GPUTextureView,
  hdrIndirectView: GPUTextureView,
  combinedOutView: GPUTextureView,
  /** Item 24 — albedo texture view for re-modulation after indirect denoising. */
  albedoView: GPUTextureView,
): GPUBindGroup {
  return buildBindGroupFromTable(device, 'indirectCombine', getIndirectCombineBindGroupLayout(device, cache), [
    denoisedDirectView,   // 0
    hdrIndirectView,      // 1
    combinedOutView,      // 2
    albedoView,           // 3 Item 24 — re-modulate by albedo (Schied 2017 §4.1)
  ]);
}

// ── Shared à-trous variance atrous bind group ────────────────────────────────
// Both SVGFRealDenoiser and AtrousVarianceDenoiser build the same 6-binding
// layout for the atrous-variance kernel's `svgfAtrousMain` entry. This builder
// is the single source so the binding order is enforced in one place.
// Binding order mirrors AtrousVarianceAtrousBindGroupLayout in
// @vitrum/shared-denoisers (bindings 0..5):
//   0 = color input (rgba16float, texture_2d)
//   1 = color output (rgba16float, storage write)
//   2 = gNormalDepth (rgba32float or rgba16float, texture_2d) — normal
//   3 = gNormalDepth (same view) — depth
//   4 = variance estimate (rg32float, texture_2d)
//   5 = AtrousVarianceAtrousUBO (uniform)

/**
 * Build the atrous-variance atrous-pass bind group.  Both the
 * `SVGFRealDenoiser` and the `AtrousVarianceDenoiser` bind this identical
 * 6-entry layout; sharing it here prevents the two sites from drifting.
 *
 * @param atrousPipeline  The compiled atrous pipeline (layout: 'auto').
 * @param inputTex        View of the color input texture.
 * @param outputTex       View of the color output texture (storage write).
 * @param gNormalDepth    View of the normal+depth G-buffer (bound at slots 2 AND 3).
 * @param varianceEstimate View of the variance estimate texture.
 * @param ubo             The per-iteration AtrousVarianceAtrousUBO buffer.
 * @param label           Optional GPUBindGroup debug label.
 */
export function buildAtrousVarianceAtrousBindGroup(
  device: GPUDevice,
  atrousPipeline: GPUComputePipeline,
  inputTex: GPUTextureView,
  outputTex: GPUTextureView,
  gNormalDepth: GPUTextureView,
  varianceEstimate: GPUTextureView,
  ubo: GPUBuffer,
  label = 'atrous-variance-atrous-bg',
): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout: atrousPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: inputTex },
      { binding: 1, resource: outputTex },
      { binding: 2, resource: gNormalDepth },
      { binding: 3, resource: gNormalDepth },
      { binding: 4, resource: varianceEstimate },
      { binding: 5, resource: { buffer: ubo } },
    ],
  });
}

// ── PPG bind groups (W9 — Müller 2017 path guiding) ──────────────────────────
//
// The PPG update kernel uses `layout: 'auto'` (the WGSL declares its own
// bindings rather than referencing a cached BGL family), so this helper takes
// the pipeline's `getBindGroupLayout` accessor instead of a `BGLCache`.
// Guided sampling itself is inlined in gi-ris via ppgPdf.wgsl.

/** Pipeline auto-layout accessor — `GPUComputePipeline.getBindGroupLayout`. */
export type AutoLayoutFor = (index: number) => GPUBindGroupLayout;

export interface PpgUpdateBindGroupResources {
  reservoirGiCurrentBuffer: GPUBuffer;
  fluxAtomicsBuf: GPUBuffer;
  sTreeBuf: GPUBuffer;
  dTreeBuf: GPUBuffer;
  dTreeOffsetsBuf: GPUBuffer;
  /** A2 — per-spatial-cell sample counter (binding 5). */
  cellSampleCountsBuf: GPUBuffer;
  updateUboBuffer: GPUBuffer;
}

/**
 * Build the two auto-layout bind groups for the PPG update kernel
 * (ppgUpdate.wgsl.ts):
 *   group(0): reservoirGiCurrent / fluxAtomics / sTree / dTree / dTreeOffsets /
 *            cellSampleCounts
 *   group(1): updateUbo
 */
export function buildPpgUpdateBindGroups(
  device: GPUDevice,
  getBindGroupLayout: AutoLayoutFor,
  r: PpgUpdateBindGroupResources,
): readonly [GPUBindGroup, GPUBindGroup] {
  const bg0 = device.createBindGroup({
    label: 'ppg-update-bg0',
    layout: getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: r.reservoirGiCurrentBuffer } },
      { binding: 1, resource: { buffer: r.fluxAtomicsBuf } },
      { binding: 2, resource: { buffer: r.sTreeBuf } },
      { binding: 3, resource: { buffer: r.dTreeBuf } },
      { binding: 4, resource: { buffer: r.dTreeOffsetsBuf } },
      { binding: 5, resource: { buffer: r.cellSampleCountsBuf } },
    ],
  });
  const bg1 = device.createBindGroup({
    label: 'ppg-update-bg1',
    layout: getBindGroupLayout(1),
    entries: [{ binding: 0, resource: { buffer: r.updateUboBuffer } }],
  });
  return [bg0, bg1];
}
