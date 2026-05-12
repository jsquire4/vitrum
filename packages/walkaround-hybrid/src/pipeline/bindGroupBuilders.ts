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
  getGTAOBindGroupLayout,
  getGTAOUpsampleBindGroupLayout,
  getTemporalGiBindGroupLayout,
  getSpatialGiBindGroupLayout,
  getIndirectCombineBindGroupLayout,
  getIndirectTemporalAccumBindGroupLayout,
  type BGLCache,
} from './bindGroupLayouts.js';

// ── Frame bind group ─────────────────────────────────────────────────────────

export interface FrameBindGroupResources {
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
}

export function buildFrameBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  r: FrameBindGroupResources,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'frame-bg',
    layout: getFrameBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: r.placeholderView },   // gDepth (placeholder — not used in primary-ray-cast mode)
      { binding: 1, resource: r.placeholderView },   // gNormal
      { binding: 2, resource: r.placeholderView },   // gAlbedo
      { binding: 3, resource: r.placeholderView },   // gRough
      { binding: 4, resource: r.placeholderView },   // motionVec
      { binding: 5, resource: { buffer: r.reservoirCurrentBuffer } },
      { binding: 6, resource: { buffer: r.reservoirPreviousBuffer } },
      { binding: 7, resource: { buffer: r.reservoirSpatialBuffer } },
      { binding: 8, resource: r.hdrColorTexture.createView() },
      { binding: 9, resource: r.nearestSampler },
      // gNormalDepth — only the shade pass writes to it; other passes
      // declare it (in the BGL) but never reference the symbol, so it's
      // inert for them. Bound to the same texture in every dispatch.
      { binding: 10, resource: r.gNormalDepthTexture.createView() },
      // Sprint 16 — half-res ReSTIR-GI reservoir. Only risGi writes to it
      // (and shade reads it); other DI passes declare it via the BGL but
      // never reference the symbol.
      { binding: 11, resource: { buffer: r.reservoirGiCurrentBuffer } },
      // Sprint 18 — indirect-channel HDR output. Only shade writes to it;
      // bound to all frame-BGL pipelines for layout compatibility.
      { binding: 12, resource: r.hdrIndirectTexture.createView() },
      // Sprint 18 follow-up — total-radiance output (welford input).
      { binding: 13, resource: r.hdrTotalTexture.createView() },
      // Item 24 — albedo demodulation: shade writes visible-point albedo here;
      // indirectCombine reads it to re-modulate the denoised indirect signal.
      { binding: 14, resource: r.albedoTexture.createView() },
    ],
  });
}

// ── Scene bind group ─────────────────────────────────────────────────────────

export interface SceneBindGroupResources {
  bvhNodesBuffer: GPUBuffer;
  bvhIndexBuffer: GPUBuffer;
  bvhPositionBuffer: GPUBuffer;
  emitterBuffer: GPUBuffer;
  emitterCdfBuffer: GPUBuffer;
  bvhBeerBuffer: GPUBuffer;
}

export function buildSceneBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  r: SceneBindGroupResources,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'scene-bg',
    layout: getSceneBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: { buffer: r.bvhNodesBuffer } },
      { binding: 1, resource: { buffer: r.bvhIndexBuffer } },   // vec4u: [0..2]=indices, [3]=RGBA8 raw attCol
      { binding: 2, resource: { buffer: r.bvhPositionBuffer } },
      { binding: 3, resource: { buffer: r.emitterBuffer } },
      { binding: 4, resource: { buffer: r.emitterCdfBuffer } },
      { binding: 5, resource: { buffer: r.bvhBeerBuffer } },    // u32: per-tri Beer-Lambert visible color
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
  return device.createBindGroup({
    label: 'ubo-bg',
    layout: getUboBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: { buffer: uboBuffer } },
      { binding: 1, resource: aoFullView },  // Sprint 15 — GTAO occlusion factor
      { binding: 2, resource: tierView },    // Sprint 9 — adaptive-sampling tier
    ],
  });
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

/** Direct-channel default — tight stops, preserves shadow / caustic edges. */
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
  sigmas: Readonly<AtrousSigmas> = ATROUS_DIRECT_SIGMAS,
): GPUBindGroup {
  if (!uboRef.buf) {
    uboRef.buf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }
  const uboData = new Float32Array([stepWidth, sigmas.sigmaN, sigmas.sigmaZ, sigmas.sigmaC]);
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
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }
  // AccumUBO is now {alpha, _pad1, _pad2, _pad3} — the temporal accum shader
  // uses an AABB clamp on the 3×3 neighborhood (not k·std_dev), so the former
  // varianceK slot is unused padding.
  device.queue.writeBuffer(
    uboRef.buf, 0,
    new Float32Array([alpha, 0, 0, 0]),
  );
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

export interface HybridLayersResources {
  ddgiIrrTex: GPUTexture | null;
  ddgiVisTex: GPUTexture | null;
  ddgiPlaceholderRgba16f: GPUTexture;
  ddgiPlaceholderRg16f: GPUTexture;
  nearestSampler: GPUSampler;
  ddgiUboBuffer: GPUBuffer;
}

export function buildHybridLayersBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  r: HybridLayersResources,
): GPUBindGroup {
  const irrTex = r.ddgiIrrTex ?? r.ddgiPlaceholderRgba16f;
  const visTex = r.ddgiVisTex ?? r.ddgiPlaceholderRg16f;
  return device.createBindGroup({
    label: 'hybrid-layers-bg',
    layout: getHybridLayersBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: irrTex.createView() },
      { binding: 1, resource: visTex.createView() },
      { binding: 2, resource: r.nearestSampler },
      { binding: 3, resource: { buffer: r.ddgiUboBuffer } },
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
  return device.createBindGroup({
    label: 'composite-bg',
    layout: getCompositeBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: texView },
      { binding: 1, resource: compositeSampler },
    ],
  });
}

// ── À-trous + variance bind group builders (Sprint 10a) ──────────────────────
//
// These pipelines use 'auto' bindgroup layouts, so the layout source is the
// pipeline itself (via `getBindGroupLayout(0)`) rather than the BGL cache.
// These builders centralize the layout-binding wiring that previously lived
// inline inside renderFrame().

export function buildWelfordBindGroup(
  device: GPUDevice,
  welfordPipeline: GPUComputePipeline,
  hdrColor: GPUTextureView,
  welfordRead: GPUTextureView,
  welfordWrite: GPUTextureView,
  ubo: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'welford-bg',
    layout: welfordPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: hdrColor },
      { binding: 1, resource: welfordRead },
      { binding: 2, resource: welfordWrite },
      { binding: 3, resource: { buffer: ubo } },
    ],
  });
}

export function buildAtrousVarianceVarianceBindGroup(
  device: GPUDevice,
  variancePipeline: GPUComputePipeline,
  hdrColor: GPUTextureView,
  welfordWrite: GPUTextureView,
  varianceEstimate: GPUTextureView,
  ubo: GPUBuffer,
): GPUBindGroup {
  // The variance kernel reads only inputColor (0), varianceIn (5),
  // and writes varianceOut (6) + reads varUBO (7). Bindings 1..4 are
  // DECLARED in the WGSL but UNREFERENCED by the kernel body — Dawn's
  // `layout: 'auto'` drops unreferenced bindings, so trying to bind
  // them yields "binding index N not present in the bind group layout"
  // and the whole command buffer is rejected. We just don't pass them.
  return device.createBindGroup({
    label: 'atrous-variance-variance-bg',
    layout: variancePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: hdrColor },
      { binding: 5, resource: welfordWrite },
      { binding: 6, resource: varianceEstimate },
      { binding: 7, resource: { buffer: ubo } },
    ],
  });
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
  return device.createBindGroup({
    label: 'sample-budget-bg',
    layout: getSampleBudgetBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: { buffer: budgetUbo } },
      { binding: 1, resource: varianceView },        // welford variance source (rg32float)
      { binding: 2, resource: tierWriteView },       // tier output (r32uint)
      { binding: 3, resource: { buffer: sampleCountUbo } },
    ],
  });
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
  return device.createBindGroup({
    label: 'resolve-bg',
    layout: getResolveBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: { buffer: resolveUbo } },
      { binding: 1, resource: currentRadianceView },
      { binding: 2, resource: prevRadianceView },
      { binding: 3, resource: motionVectorsView },
      { binding: 4, resource: resolvedWriteView },
    ],
  });
}

// ── GTAO bind groups (Sprint 15) ─────────────────────────────────────────────

export function buildGTAOBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  gNormalDepthView: GPUTextureView,
  aoHalfWriteView: GPUTextureView,
  gtaoUbo: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'gtao-bg',
    layout: getGTAOBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: gNormalDepthView },
      { binding: 1, resource: aoHalfWriteView },
      { binding: 2, resource: { buffer: gtaoUbo } },
    ],
  });
}

export function buildGTAOUpsampleBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  aoHalfReadView: GPUTextureView,
  gNormalDepthView: GPUTextureView,
  aoFullWriteView: GPUTextureView,
  gtaoUbo: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'gtao-upsample-bg',
    layout: getGTAOUpsampleBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: aoHalfReadView },
      { binding: 1, resource: gNormalDepthView },
      { binding: 2, resource: aoFullWriteView },
      { binding: 3, resource: { buffer: gtaoUbo } },
    ],
  });
}

// ── GI temporal + spatial bind groups (Sprint 17) ────────────────────────────

export function buildTemporalGiBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  reservoirGiCurrent: GPUBuffer,
  reservoirGiPrevious: GPUBuffer,
  uboBuffer: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'temporal-gi-bg',
    layout: getTemporalGiBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: { buffer: reservoirGiCurrent } },
      { binding: 1, resource: { buffer: reservoirGiPrevious } },
      { binding: 2, resource: { buffer: uboBuffer } },
    ],
  });
}

export function buildSpatialGiBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  inBuffer: GPUBuffer,
  outBuffer: GPUBuffer,
  uboBuffer: GPUBuffer,
  label: string,
): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout: getSpatialGiBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: { buffer: inBuffer } },
      { binding: 1, resource: { buffer: outBuffer } },
      { binding: 2, resource: { buffer: uboBuffer } },
    ],
  });
}

// ── Indirect temporal accumulator (Sprint 18 follow-up) ──────────────────────

export function buildIndirectTemporalAccumBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  currentRawView: GPUTextureView,
  prevAccumView: GPUTextureView,
  outAccumView: GPUTextureView,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'indirect-temporal-accum-bg',
    layout: getIndirectTemporalAccumBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: currentRawView },
      { binding: 1, resource: prevAccumView },
      { binding: 2, resource: outAccumView },
    ],
  });
}

// ── Indirect-combine bind group (Sprint 18) ──────────────────────────────────

export function buildIndirectCombineBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  denoisedDirectView: GPUTextureView,
  hdrIndirectView: GPUTextureView,
  gNormalDepthView: GPUTextureView,
  combinedOutView: GPUTextureView,
  /** Item 24 — albedo texture view for re-modulation after indirect denoising. */
  albedoView: GPUTextureView,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'indirect-combine-bg',
    layout: getIndirectCombineBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: denoisedDirectView },
      { binding: 1, resource: hdrIndirectView },
      { binding: 2, resource: gNormalDepthView },
      { binding: 3, resource: combinedOutView },
      // Item 24 — re-modulate denoised indirect by albedo (Schied 2017 §4.1).
      { binding: 4, resource: albedoView },
    ],
  });
}

export function buildAtrousVarianceAtrousBindGroup(
  device: GPUDevice,
  atrousPipeline: GPUComputePipeline,
  inputTex: GPUTextureView,
  outputTex: GPUTextureView,
  gNormalDepth: GPUTextureView,
  varianceEstimate: GPUTextureView,
  ubo: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'atrous-variance-atrous-bg',
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
