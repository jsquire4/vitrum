/**
 * svgfRealWebGPU.ts — One-shot host pipeline for the real Schied 2017 SVGF denoiser.
 *
 * Runs the full SVGF chain on CPU-backed linear HDR RGB via WebGPU:
 *   1. svgfReprojMain           — bilinear reprojection + disocclusion + EMA history
 *   2. svgfVarianceFromMomentsMain — Eq. 5 variance from blended moments
 *   3. svgf7x7FallbackMain      — 7×7 spatial fallback for history < 4 pixels
 *   4. svgfAtrousMain (×5)      — variance-guided à-trous chain (reuses ATROUS_VARIANCE_WGSL)
 *
 * GPU memory budget (see svgfRealConstants.ts for full breakdown):
 *   New persistent textures: historyLength (r16uint) + momentsHistory (rg32float)
 *   + prevRadiance (rgba16float) + motionVec (rg32float) ≈ 52 MB at 1080p.
 *
 * This one-shot path allocates and destroys all textures per call. In the
 * WalkaroundGPUPipeline (persistent mode), textures are allocated once in
 * createFrameResources() and referenced across frames.
 *
 * Previously named: (new file — no prior SVGF one-shot path existed for real SVGF).
 *
 * References:
 *   Schied et al. "Spatiotemporal Variance-Guided Filtering" HPG 2017.
 */

import { SVGF_REAL_REPROJECTION_WORKGROUP_SIZE } from './wgsl/svgfReprojection.wgsl.js';
import { SVGF_VARIANCE_FROM_MOMENTS_WORKGROUP_SIZE } from './wgsl/svgfVarianceFromMoments.wgsl.js';
import { SVGF_7X7_SPATIAL_FALLBACK_WORKGROUP_SIZE } from './wgsl/svgf7x7SpatialFallback.wgsl.js';
import { ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE } from './wgsl/atrousVariance.wgsl.js';
import { svgfRealPipelines } from './svgfRealPipelineCache.js';
import {
  SVGF_REPROJ_UNIFORMS_SIZE_BYTES,
  SVGF_REPROJ_DEFAULT_UNIFORMS,
  packSVGFReprojUniforms,
  type SVGFReprojUniforms,
} from './svgfRealBindings.js';
import {
  SVGF_REAL_DEFAULT_ATROUS_ITERATIONS,
  SVGF_REAL_MAX_ATROUS_ITERATIONS,
} from './svgfRealConstants.js';
import {
  ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
  ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS,
  packAtrousVarianceAtrousUniforms,
  packAtrousVarianceVarianceUniforms,
  ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES,
} from './atrousVarianceBindings.js';
import { getSharedWebGPUDevice } from './sharedWebGpuDevice.js';
import { alignedTextureCopyBytesPerRow } from './webGpuTextureCopy.js';
import {
  uploadRgbAsRgba16f,
  uploadRg32f,
  uploadR32f,
  uploadR32Uint,
  uploadR16Uint,
  fillR16Uint,
  fillRg32f,
  fillRgba32f,
  readRgba16fToRgb,
} from './webGpuTextureUpload.js';

// ============================================================
// Albedo demodulation helpers — Schied 2017 §4.1.
// ============================================================
//
// Before SVGF's variance + à-trous chain, divide the HDR radiance by the
// per-pixel albedo so the spatial filter operates on pure lighting (the
// "lighting estimate" L = c/ρ in Schied 2017 §4.1). After the chain, the
// filtered lighting is re-multiplied by albedo to restore the physically
// correct outgoing radiance.
//
// The benefit is that high-frequency albedo variation (e.g. a red/green
// checkerboard) no longer participates in the cross-bilateral weights of
// the à-trous kernel, so the filter cannot bleed colors across material
// boundaries that share the same depth + normal.
//
// `demodulateAlbedo`: out[i] = rgb[i] / max(albedo[i], 1e-3)   — returns a new array.
// `remodulateAlbedo`: rgb[i] *= albedo[i]                       — mutates in-place.
//
// Both helpers mirror the implementation in `atrousVarianceWebGPU.ts` exactly,
// so a denoiser switch (atrous-variance ↔ svgf-real) is invariant under the
// modulation step.

/** Divide rgb by albedo; returns a new Float32Array of the demodulated signal. */
export function svgfRealDemodulateAlbedo(
  rgb: Float32Array,
  albedo: Float32Array,
  pixelCount: number,
): Float32Array {
  const out = new Float32Array(rgb.length);
  for (let i = 0; i < pixelCount; i += 1) {
    const si = i * 3;
    const ar = Math.max(albedo[si]     ?? 0, 1e-3);
    const ag = Math.max(albedo[si + 1] ?? 0, 1e-3);
    const ab = Math.max(albedo[si + 2] ?? 0, 1e-3);
    out[si]     = (rgb[si]     ?? 0) / ar;
    out[si + 1] = (rgb[si + 1] ?? 0) / ag;
    out[si + 2] = (rgb[si + 2] ?? 0) / ab;
  }
  return out;
}

/** Multiply rgb by albedo in-place; returns the same Float32Array. */
export function svgfRealRemodulateAlbedo(
  rgb: Float32Array,
  albedo: Float32Array,
  pixelCount: number,
): Float32Array {
  for (let i = 0; i < pixelCount; i += 1) {
    const si = i * 3;
    const ar = albedo[si]     !== undefined ? albedo[si]     : 1;
    const ag = albedo[si + 1] !== undefined ? albedo[si + 1]! : 1;
    const ab = albedo[si + 2] !== undefined ? albedo[si + 2]! : 1;
    rgb[si]     = (rgb[si]     ?? 0) * ar;
    rgb[si + 1] = (rgb[si + 1] ?? 0) * ag;
    rgb[si + 2] = (rgb[si + 2] ?? 0) * ab;
  }
  return rgb;
}

// CPU emulation oracles (`svgfReprojCPU`, `svgfVarianceFromMomentsCPU`,
// `svgf7x7FallbackCPU`) hoisted to svgfRealCpu.ts (W4-A7 extraction;
// the duplicate copies in this file were removed in the 2026-05-18
// dead-code sweep after the package index was verified to re-export
// them only via svgfRealCpu.js).

// Pipeline cache hoisted to svgfRealPipelineCache.ts (W4-A7) — both the
// type and the function come from that module now. Earlier revisions
// inlined the cache here and the extracted module became an orphan
// canonical (created, never imported). Routing through the canonical
// activates the extracted module and removes ~40 lines of duplication.
//
// Texture upload / readback helpers were similarly inlined here until W4-A7.
// They now live in webGpuTextureUpload.ts (see imports above); routing
// through that canonical removes ~150 lines of duplication and ensures
// the row-padding alignment math has a single source of truth.

// ============================================================
// Public API
// ============================================================

export interface SVGFRealWebGPUOptions {
  /** Current-frame noisy HDR radiance (row-major RGB, length W*H*3). */
  readonly rgb: Float32Array;
  readonly width:  number;
  readonly height: number;

  /** Previous-frame EMA color. If omitted, mirrors `rgb` (first frame). */
  readonly prevRadianceRgb?: Float32Array;
  /** Screen-space motion vector (pixel delta, RG interleaved), length W*H*2. */
  readonly motionRg?: Float32Array;
  /** Linear depth per pixel, length W*H. Defaults to constant 1.0. */
  readonly linearDepth?: Float32Array;
  /** World-space normals (packed 0..1 XYZ), length W*H*3. Defaults to forward-facing. */
  readonly gbufferNormalsRgb?: Float32Array;
  /** Per-pixel object ID (u32), length W*H. Defaults to 0. */
  readonly objectIds?: Uint32Array;

  /**
   * Per-pixel surface albedo ρ (linear RGB row-major, length W*H*3). When
   * supplied, the SVGF chain runs on demodulated lighting (L = c/ρ) per
   * Schied 2017 §4.1 and re-modulates after the à-trous chain. When omitted,
   * the chain operates directly on `rgb` (legacy / variance-test path).
   */
  readonly albedoRgb?: Float32Array;

  /** Per-pixel history length (u32) from previous frame. Defaults to 0. */
  readonly historyLengthIn?: Uint32Array;
  /** Previous-frame moments M1, M2 (interleaved), length W*H*2. Defaults to 0. */
  readonly momentsIn?: Float32Array;

  /** Disocclusion + EMA tunables. Uses paper defaults when omitted. */
  readonly reprojUniforms?: Partial<SVGFReprojUniforms>;

  /** À-trous iterations (default SVGF_REAL_DEFAULT_ATROUS_ITERATIONS = 5). */
  readonly atrousIterations?: number;
  /** À-trous σ values. Uses ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS when omitted. */
  readonly sigmaColor?:  number;
  readonly sigmaNormal?: number;
  readonly sigmaDepth?:  number;

  /** Explicit GPU device (never destroyed by this call). */
  readonly device?: GPUDevice;
  /** When false, requests an ephemeral device (and destroys it after). Default: true. */
  readonly reuseSharedWebGpuDevice?: boolean;
}

/**
 * Run a full Schied 2017 SVGF denoiser pass on the given inputs.
 * Returns the filtered HDR RGB (length width*height*3).
 */
export async function runSVGFRealWebGPU(opts: SVGFRealWebGPUOptions): Promise<Float32Array> {
  const w = opts.width;
  const h = opts.height;
  const rawAtrous = opts.atrousIterations ?? SVGF_REAL_DEFAULT_ATROUS_ITERATIONS;
  const atrousIterations = Math.min(SVGF_REAL_MAX_ATROUS_ITERATIONS, Math.max(1, Math.floor(rawAtrous)));
  const reuseShared = opts.reuseSharedWebGpuDevice !== false && opts.device == null;
  const sigmaColor  = opts.sigmaColor  ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaColor;
  const sigmaNormal = opts.sigmaNormal ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaNormal;
  const sigmaDepth  = opts.sigmaDepth  ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaDepth;

  const reprojU: SVGFReprojUniforms = {
    sigmaDepth:  opts.reprojUniforms?.sigmaDepth  ?? SVGF_REPROJ_DEFAULT_UNIFORMS.sigmaDepth,
    sigmaNormal: opts.reprojUniforms?.sigmaNormal ?? SVGF_REPROJ_DEFAULT_UNIFORMS.sigmaNormal,
    alphaMin:    opts.reprojUniforms?.alphaMin    ?? SVGF_REPROJ_DEFAULT_UNIFORMS.alphaMin,
  };

  if (typeof navigator === 'undefined' || navigator.gpu == null) {
    throw new Error('runSVGFRealWebGPU: WebGPU not available');
  }

  let device: GPUDevice;
  let destroyEphemeral: (() => void) | null = null;
  if (opts.device != null) {
    device = opts.device;
  } else if (reuseShared) {
    device = await getSharedWebGPUDevice();
  } else {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter == null) throw new Error('runSVGFRealWebGPU: failed to request GPU adapter');
    device = await adapter.requestDevice();
    destroyEphemeral = () => { device.destroy(); };
  }

  const { reprojPipeline, momentsPipeline, fallbackPipeline, atrousPipeline } =
    svgfRealPipelines(device);

  // ── Texture creation ──────────────────────────────────────────────────────
  const texB = GPUTextureUsage.TEXTURE_BINDING;
  const texS = GPUTextureUsage.STORAGE_BINDING;
  const texC = GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;

  const currColorTex = device.createTexture({ label: 'svgf-curr-color', size: [w,h], format: 'rgba16float', usage: texB|texC });
  const prevColorTex = device.createTexture({ label: 'svgf-prev-color', size: [w,h], format: 'rgba16float', usage: texB|texC });
  const motionTex    = device.createTexture({ label: 'svgf-motion',     size: [w,h], format: 'rg32float',   usage: texB|texC });
  const currDepthTex = device.createTexture({ label: 'svgf-depth',      size: [w,h], format: 'r32float',    usage: texB|texC });
  const currNormTex  = device.createTexture({ label: 'svgf-norm',       size: [w,h], format: 'rgba32float', usage: texB|texC });
  const currObjTex   = device.createTexture({ label: 'svgf-obj',        size: [w,h], format: 'r32uint',     usage: texB|texC });
  const prevDepthTex = device.createTexture({ label: 'svgf-prev-depth', size: [w,h], format: 'r32float',    usage: texB|texC });
  const prevNormTex  = device.createTexture({ label: 'svgf-prev-norm',  size: [w,h], format: 'rgba32float', usage: texB|texC });
  const prevObjTex   = device.createTexture({ label: 'svgf-prev-obj',   size: [w,h], format: 'r32uint',     usage: texB|texC });
  const histInTex    = device.createTexture({ label: 'svgf-hist-in',    size: [w,h], format: 'r16uint',     usage: texB|texC });
  const momentsInTex = device.createTexture({ label: 'svgf-mom-in',     size: [w,h], format: 'rg32float',   usage: texB|texC });

  // Reprojection outputs
  const colorOutTex  = device.createTexture({ label: 'svgf-color-out', size: [w,h], format: 'rgba16float', usage: texS|texB|texC });
  const histOutTex   = device.createTexture({ label: 'svgf-hist-out',  size: [w,h], format: 'r16uint',     usage: texS|texB|texC });
  const momOutTex    = device.createTexture({ label: 'svgf-mom-out',   size: [w,h], format: 'rg32float',   usage: texS|texB|texC });

  // Variance from moments output
  const varMomOutTex  = device.createTexture({ label: 'svgf-var-mom',  size: [w,h], format: 'rg32float', usage: texS|texB|texC });
  // Merged variance (after 7×7 fallback)
  const varFinalTex   = device.createTexture({ label: 'svgf-var-final',size: [w,h], format: 'rg32float', usage: texS|texB|texC });

  // Atrous-variance pass variance output (not actually used — we feed varFinalTex directly to atrous)
  // We need a "varianceIn" for the atrous-variance's own svgfVarianceMain pass.
  // For svgf-real, we skip the atrous-variance pass's variance computation and use
  // the already-computed varFinalTex as the varianceMap directly fed to svgfAtrousMain.

  // Atrous ping-pong
  const pingPongUsage = texS | texB | texC;
  const pingTex = device.createTexture({ label: 'svgf-ping', size: [w,h], format: 'rgba16float', usage: pingPongUsage });
  const pongTex = device.createTexture({ label: 'svgf-pong', size: [w,h], format: 'rgba16float', usage: pingPongUsage });

  // ── Upload inputs ─────────────────────────────────────────────────────────
  // Schied 2017 §4.1 — albedo demodulation. When albedoRgb is supplied,
  // divide both the current and previous-frame HDR radiance by per-pixel
  // albedo BEFORE the SVGF chain so reprojection blending, moment tracking,
  // variance estimation, and the à-trous spatial filter all see the
  // demodulated lighting estimate L = c/ρ. This keeps high-frequency albedo
  // variation (e.g. material-boundary checkerboards) out of the cross-
  // bilateral weights and prevents color bleed across material edges that
  // share the same depth + normal. After the à-trous chain finishes, we
  // multiply the filtered lighting back by albedo to restore physically
  // correct outgoing radiance.
  const px = w * h;
  const rgbForChain = opts.albedoRgb != null
    ? svgfRealDemodulateAlbedo(opts.rgb, opts.albedoRgb, px)
    : opts.rgb;
  const prevForChain = opts.albedoRgb != null
    ? svgfRealDemodulateAlbedo(opts.prevRadianceRgb ?? opts.rgb, opts.albedoRgb, px)
    : (opts.prevRadianceRgb ?? opts.rgb);
  uploadRgbAsRgba16f(device, currColorTex, rgbForChain, w, h);
  uploadRgbAsRgba16f(device, prevColorTex, prevForChain, w, h);

  if (opts.motionRg != null) {
    uploadRg32f(device, motionTex, opts.motionRg, w, h);
  } else {
    fillRg32f(device, motionTex, w, h, 0, 0);
  }
  if (opts.linearDepth != null) {
    uploadR32f(device, currDepthTex, opts.linearDepth, w, h);
    uploadR32f(device, prevDepthTex, opts.linearDepth, w, h); // prev same as curr for one-shot
  } else {
    const ones = new Float32Array(w * h).fill(1);
    uploadR32f(device, currDepthTex, ones, w, h);
    uploadR32f(device, prevDepthTex, ones, w, h);
  }
  if (opts.gbufferNormalsRgb != null) {
    // Upload packed normals to both curr and prev normal textures.
    const bpr = alignedTextureCopyBytesPerRow(w, 16);
    const stride = bpr / 4;
    const normBuf = new Float32Array(stride * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const si = (y * w + x) * 3;
        const o  = y * stride + x * 4;
        normBuf[o]   = opts.gbufferNormalsRgb[si]     ?? 0.5;
        normBuf[o+1] = opts.gbufferNormalsRgb[si + 1] ?? 0.5;
        normBuf[o+2] = opts.gbufferNormalsRgb[si + 2] ?? 1.0;
        normBuf[o+3] = 0;
      }
    }
    device.queue.writeTexture({ texture: currNormTex }, normBuf.buffer, { bytesPerRow: bpr }, [w,h]);
    device.queue.writeTexture({ texture: prevNormTex }, normBuf.buffer, { bytesPerRow: bpr }, [w,h]);
  } else {
    fillRgba32f(device, currNormTex, w, h, [0.5, 0.5, 1.0, 0.0]);
    fillRgba32f(device, prevNormTex, w, h, [0.5, 0.5, 1.0, 0.0]);
  }
  if (opts.objectIds != null) {
    uploadR32Uint(device, currObjTex, opts.objectIds, w, h);
    uploadR32Uint(device, prevObjTex, opts.objectIds, w, h);
  } else {
    const zeros32 = new Uint32Array(w * h);
    uploadR32Uint(device, currObjTex, zeros32, w, h);
    uploadR32Uint(device, prevObjTex, zeros32, w, h);
  }
  if (opts.historyLengthIn != null) {
    uploadR16Uint(device, histInTex, opts.historyLengthIn, w, h);
  } else {
    fillR16Uint(device, histInTex, w, h, 0);
  }
  if (opts.momentsIn != null) {
    uploadRg32f(device, momentsInTex, opts.momentsIn, w, h);
  } else {
    fillRg32f(device, momentsInTex, w, h, 0, 0);
  }

  // ── UBOs ─────────────────────────────────────────────────────────────────
  const reprojUboScratch = new ArrayBuffer(SVGF_REPROJ_UNIFORMS_SIZE_BYTES);
  packSVGFReprojUniforms(reprojU, reprojUboScratch);
  const reprojUboGpu = device.createBuffer({
    size: SVGF_REPROJ_UNIFORMS_SIZE_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(reprojUboGpu, 0, reprojUboScratch);

  // Atrous UBOs (one per iteration)
  const atrousUbos: GPUBuffer[] = [];
  const atrousScratch = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
  for (let iter = 0; iter < atrousIterations; iter++) {
    const ubo = device.createBuffer({
      label: `svgf-real-atrous-ubo-${iter}`,
      size: ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    packAtrousVarianceAtrousUniforms({ iteration: iter, sigmaColor, sigmaNormal, sigmaDepth }, atrousScratch);
    device.queue.writeBuffer(ubo, 0, atrousScratch);
    atrousUbos.push(ubo);
  }

  // ── Bind groups ───────────────────────────────────────────────────────────
  const reprojBG = device.createBindGroup({
    layout: reprojPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0,  resource: currColorTex.createView() },
      { binding: 1,  resource: prevColorTex.createView() },
      { binding: 2,  resource: motionTex.createView() },
      { binding: 3,  resource: currDepthTex.createView() },
      { binding: 4,  resource: currNormTex.createView() },
      { binding: 5,  resource: currObjTex.createView() },
      { binding: 6,  resource: prevDepthTex.createView() },
      { binding: 7,  resource: prevNormTex.createView() },
      { binding: 8,  resource: prevObjTex.createView() },
      { binding: 9,  resource: histInTex.createView() },
      { binding: 10, resource: momentsInTex.createView() },
      { binding: 11, resource: colorOutTex.createView() },
      { binding: 12, resource: histOutTex.createView() },
      { binding: 13, resource: momOutTex.createView() },
      { binding: 14, resource: { buffer: reprojUboGpu } },
    ],
  });

  const momentsBG = device.createBindGroup({
    layout: momentsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: momOutTex.createView() },
      { binding: 1, resource: histOutTex.createView() },
      { binding: 2, resource: varMomOutTex.createView() },
    ],
  });

  const fallbackBG = device.createBindGroup({
    layout: fallbackPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: colorOutTex.createView() },
      { binding: 1, resource: histOutTex.createView() },
      { binding: 2, resource: varMomOutTex.createView() },
      { binding: 3, resource: varFinalTex.createView() },
    ],
  });

  // Initial copy of colorOut → pingTex for atrous input
  // We need a dummy variance UBO and varianceIn for the svgfVarianceMain we won't use —
  // Instead we skip the atrous-variance's own variance pass and use varFinalTex directly
  // as the varianceMap in all atrous iterations.

  const atrousBGs = atrousUbos.map((ubo, iter) => {
    const isEven = iter % 2 === 0;
    return device.createBindGroup({
      layout: atrousPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: (isEven ? pingTex : pongTex).createView() },
        { binding: 1, resource: (isEven ? pongTex : pingTex).createView() },
        { binding: 2, resource: currNormTex.createView() },
        { binding: 3, resource: currDepthTex.createView() },
        { binding: 4, resource: varFinalTex.createView() },
        { binding: 5, resource: { buffer: ubo } },
      ],
    });
  });

  // ── Command encoding ──────────────────────────────────────────────────────
  const wg = SVGF_REAL_REPROJECTION_WORKGROUP_SIZE;
  const wg2 = SVGF_VARIANCE_FROM_MOMENTS_WORKGROUP_SIZE;
  const wg3 = SVGF_7X7_SPATIAL_FALLBACK_WORKGROUP_SIZE;
  const wgA = ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE;

  const encoder = device.createCommandEncoder({ label: 'svgf-real-batched' });

  // 1. Reprojection
  {
    const pass = encoder.beginComputePass({ label: 'svgf-reproj' });
    pass.setPipeline(reprojPipeline);
    pass.setBindGroup(0, reprojBG);
    pass.dispatchWorkgroups(Math.ceil(w / wg), Math.ceil(h / wg));
    pass.end();
  }

  // 2. Variance from moments
  {
    const pass = encoder.beginComputePass({ label: 'svgf-moments' });
    pass.setPipeline(momentsPipeline);
    pass.setBindGroup(0, momentsBG);
    pass.dispatchWorkgroups(Math.ceil(w / wg2), Math.ceil(h / wg2));
    pass.end();
  }

  // 3. 7×7 spatial fallback (merges into varFinalTex)
  {
    const pass = encoder.beginComputePass({ label: 'svgf-7x7' });
    pass.setPipeline(fallbackPipeline);
    pass.setBindGroup(0, fallbackBG);
    pass.dispatchWorkgroups(Math.ceil(w / wg3), Math.ceil(h / wg3));
    pass.end();
  }

  // 4. Copy colorOut → pingTex (atrous input for iter 0)
  encoder.copyTextureToTexture({ texture: colorOutTex }, { texture: pingTex }, [w, h]);

  // 5. À-trous chain
  for (let iter = 0; iter < atrousIterations; iter++) {
    const pass = encoder.beginComputePass({ label: `svgf-atrous-${iter}` });
    pass.setPipeline(atrousPipeline);
    pass.setBindGroup(0, atrousBGs[iter]);
    pass.dispatchWorkgroups(Math.ceil(w / wgA), Math.ceil(h / wgA));
    pass.end();
  }

  device.queue.submit([encoder.finish()]);

  // Read result: after N iterations, last write is in pong (odd) or ping (even).
  const readTex = atrousIterations % 2 === 0 ? pingTex : pongTex;
  const result  = await readRgba16fToRgb(device, readTex, w, h);

  // Schied 2017 §4.1 — albedo re-modulation: multiply the filtered lighting
  // by per-pixel albedo to restore physically correct denoised outgoing
  // radiance. In-place mutation of `result`. No-op when albedoRgb is omitted.
  if (opts.albedoRgb != null) {
    svgfRealRemodulateAlbedo(result, opts.albedoRgb, px);
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────
  for (const t of [
    currColorTex, prevColorTex, motionTex, currDepthTex, currNormTex,
    currObjTex, prevDepthTex, prevNormTex, prevObjTex, histInTex,
    momentsInTex, colorOutTex, histOutTex, momOutTex,
    varMomOutTex, varFinalTex, pingTex, pongTex,
  ]) {
    t.destroy();
  }
  reprojUboGpu.destroy();
  for (const ubo of atrousUbos) ubo.destroy();
  destroyEphemeral?.();

  return result;
}
