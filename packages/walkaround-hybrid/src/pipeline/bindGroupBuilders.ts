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
): GPUBindGroup {
  return device.createBindGroup({
    label: 'ubo-bg',
    layout: getUboBindGroupLayout(device, cache),
    entries: [{ binding: 0, resource: { buffer: uboBuffer } }],
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
  // varianceK: 1.5 — std-dev multiplier for the temporal-history clamp box.
  // Karis recommends k=1 in his original temporal-AA writeup; 1.5 is a small
  // relaxation that lets stable samples blend more smoothly without
  // re-introducing sparkle/cross-talk at edges. Higher = more history weight
  // (more smoothing, more sparkle risk), lower = stricter clamp (sharper
  // edges, more noise).
  device.queue.writeBuffer(
    uboRef.buf, 0,
    new Float32Array([alpha, 1.5, 0, 0]),
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

export interface SampleBudgetBindGroupResources {
  varianceView: GPUTextureView;
  sampleTierView: GPUTextureView;
  thresholdLow: number;
  thresholdHigh: number;
  width: number;
  height: number;
  sampleCount: number;
}

export function buildSampleBudgetBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  uboRef: UboRef,
  sampleCountRef: UboRef,
  r: SampleBudgetBindGroupResources,
): GPUBindGroup {
  if (!uboRef.buf) {
    uboRef.buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  if (!sampleCountRef.buf) {
    sampleCountRef.buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  const uboF32 = new Float32Array([r.thresholdLow, r.thresholdHigh, 0, 0]);
  const uboU32 = new Uint32Array(uboF32.buffer);
  uboU32[2] = r.width >>> 0;
  uboU32[3] = r.height >>> 0;
  device.queue.writeBuffer(uboRef.buf, 0, uboF32);
  device.queue.writeBuffer(sampleCountRef.buf, 0, new Uint32Array([r.sampleCount >>> 0, 0, 0, 0]));

  return device.createBindGroup({
    label: 'sample-budget-bg',
    layout: getSampleBudgetBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: { buffer: uboRef.buf } },
      { binding: 1, resource: r.varianceView },
      { binding: 2, resource: r.sampleTierView },
      { binding: 3, resource: { buffer: sampleCountRef.buf } },
    ],
  });
}

export interface ResolveBindGroupResources {
  currentView: GPUTextureView;
  prevView: GPUTextureView;
  motionView: GPUTextureView;
  resolvedOutView: GPUTextureView;
  width: number;
  height: number;
  frameParity: number;
}

export function buildResolveBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  uboRef: UboRef,
  r: ResolveBindGroupResources,
): GPUBindGroup {
  if (!uboRef.buf) {
    uboRef.buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  const u = new Uint32Array([r.width >>> 0, r.height >>> 0, r.frameParity >>> 0, 0]);
  device.queue.writeBuffer(uboRef.buf, 0, u);
  return device.createBindGroup({
    label: 'resolve-bg',
    layout: getResolveBindGroupLayout(device, cache),
    entries: [
      { binding: 0, resource: { buffer: uboRef.buf } },
      { binding: 1, resource: r.currentView },
      { binding: 2, resource: r.prevView },
      { binding: 3, resource: r.motionView },
      { binding: 4, resource: r.resolvedOutView },
    ],
  });
}
