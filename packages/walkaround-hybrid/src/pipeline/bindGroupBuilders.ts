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
  getHybridLayersBindGroupLayoutWithPpg,
  getSampleBudgetBindGroupLayout,
  getResolveBindGroupLayout,
  getGTAOBindGroupLayout,
  getGTAOUpsampleBindGroupLayout,
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
): GPUBindGroup {
  return device.createBindGroup({
    label: 'ubo-bg',
    layout: getUboBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: { buffer: uboBuffer } },
      { binding: 1, resource: aoFullView },  // Sprint 15 — GTAO occlusion factor
    ],
  });
}

// ── Atrous bind group ────────────────────────────────────────────────────────

/** Mutable wrapper so the UBO buffer is lazily created once and reused. */
export interface UboRef { buf: GPUBuffer | undefined }

export function buildAtrousBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  uboRef: UboRef,
  inputView: GPUTextureView,
  outputView: GPUTextureView,
  gNormalView: GPUTextureView,
  gDepthView: GPUTextureView,
  stepWidth: number,
): GPUBindGroup {
  if (!uboRef.buf) {
    uboRef.buf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }
  // sigmaN=128, sigmaZ=5.0, sigmaC=0.05 are tuned for an HDR-linear ReSTIR
  // à-trous denoiser at this scene's scale (camera-relative ray distances
  // ~30..200 units). The shade pass writes a real per-pixel normal+depth
  // G-buffer into `gNormalDepthTexture`, so the normal-based edge-stop
  // (`pow(dot(n,n)^k, sigmaN)`) and depth-based stop (`exp(-|Δz|/sigmaZ)`)
  // are meaningful.
  //   σn=128 → tight normal-stop; only near-coplanar surfaces blur together.
  //   σz=5   → tolerates ~5 unit depth differences (one floor-tile receding
  //            from camera at stepWidth=16); rejects floor↔wall transitions.
  //   σc=0.05 → handles low-chroma-delta caustic boundaries. At 0.15, RED
  //            caustics on warm-oak floor blocked cleanly, but BLUE/GREEN
  //            caustics had small Δc — atrous bled cool-tinted caustic into
  //            the warm floor. At σc=0.05: Δ=0.36 blue-caustic-vs-warm-floor
  //            weighs ≈ 0 (blocked); Δ=0.05 within-patch noise still weighs
  //            ≈ 0.37 so within-patch denoising continues.
  const uboData = new Float32Array([stepWidth, 128.0, 5.0, 0.05]);
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
  /** When set, binds PPG training (4–5) + guiding buffers (6–9) for group 3. */
  readonly ppgTrainBuffers?: {
    sampleBuffer: GPUBuffer;
    headBuffer: GPUBuffer;
    cellBuffer: GPUBuffer;
    leafBuffer: GPUBuffer;
    kdBuffer: GPUBuffer;
    shadeMetaBuffer: GPUBuffer;
  };
}

export function buildHybridLayersBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  r: HybridLayersResources,
): GPUBindGroup {
  const irrTex = r.ddgiIrrTex ?? r.ddgiPlaceholderRgba16f;
  const visTex = r.ddgiVisTex ?? r.ddgiPlaceholderRg16f;
  const ppg = r.ppgTrainBuffers;
  const layout = ppg
    ? getHybridLayersBindGroupLayoutWithPpg(device, cache)
    : getHybridLayersBindGroupLayout(device, cache);
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: irrTex.createView() },
    { binding: 1, resource: visTex.createView() },
    { binding: 2, resource: r.nearestSampler },
    { binding: 3, resource: { buffer: r.ddgiUboBuffer } },
  ];
  if (ppg) {
    entries.push(
      { binding: 4, resource: { buffer: ppg.sampleBuffer } },
      { binding: 5, resource: { buffer: ppg.headBuffer } },
      { binding: 6, resource: { buffer: ppg.cellBuffer } },
      { binding: 7, resource: { buffer: ppg.leafBuffer } },
      { binding: 8, resource: { buffer: ppg.kdBuffer } },
      { binding: 9, resource: { buffer: ppg.shadeMetaBuffer } },
    );
  }
  return device.createBindGroup({
    label: 'hybrid-layers-bg',
    layout,
    entries,
  });
}

// ── Composite bind group ─────────────────────────────────────────────────────

export function buildCompositeBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  texView: GPUTextureView,
  compositeLinearSampler: GPUSampler,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'composite-bg',
    layout: getCompositeBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: texView },
      { binding: 1, resource: compositeLinearSampler },
    ],
  });
}

// ── SVGF bind group builders (Sprint 10a) ────────────────────────────────────
//
// SVGF pipelines use 'auto' bindgroup layouts, so the layout source is the
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

export function buildSVGFVarianceBindGroup(
  device: GPUDevice,
  variancePipeline: GPUComputePipeline,
  hdrColor: GPUTextureView,
  _prevAccum: GPUTextureView,
  _gNormalDepth: GPUTextureView,
  _motionVectors: GPUTextureView,
  welfordWrite: GPUTextureView,
  varianceEstimate: GPUTextureView,
  ubo: GPUBuffer,
): GPUBindGroup {
  // The svgf-variance kernel reads only inputColor (0), varianceIn (5),
  // and writes varianceOut (6) + reads varUBO (7). Bindings 1..4 are
  // DECLARED in the WGSL but UNREFERENCED by the kernel body — Dawn's
  // `layout: 'auto'` drops unreferenced bindings, so trying to bind
  // them yields "binding index N not present in the bind group layout"
  // and the whole command buffer is rejected. The trailing `_` params
  // are preserved so existing callers keep compiling without churn;
  // a future cleanup can shrink the signature when the SVGF spatial
  // term grows to actually consume those buffers.
  return device.createBindGroup({
    label: 'svgf-variance-bg',
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
): GPUBindGroup {
  return device.createBindGroup({
    label: 'gtao-upsample-bg',
    layout: getGTAOUpsampleBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: aoHalfReadView },
      { binding: 1, resource: gNormalDepthView },
      { binding: 2, resource: aoFullWriteView },
    ],
  });
}

export function buildSVGFAtrousBindGroup(
  device: GPUDevice,
  atrousPipeline: GPUComputePipeline,
  inputTex: GPUTextureView,
  outputTex: GPUTextureView,
  gNormalDepth: GPUTextureView,
  varianceEstimate: GPUTextureView,
  ubo: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'svgf-atrous-bg',
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
