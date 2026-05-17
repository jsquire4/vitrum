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

import { SVGF_REPROJECTION_WGSL, SVGF_REAL_REPROJECTION_WORKGROUP_SIZE } from './wgsl/svgfReprojection.wgsl.js';
import { SVGF_VARIANCE_FROM_MOMENTS_WGSL, SVGF_VARIANCE_FROM_MOMENTS_WORKGROUP_SIZE } from './wgsl/svgfVarianceFromMoments.wgsl.js';
import { SVGF_7X7_SPATIAL_FALLBACK_WGSL, SVGF_7X7_SPATIAL_FALLBACK_WORKGROUP_SIZE } from './wgsl/svgf7x7SpatialFallback.wgsl.js';
import { ATROUS_VARIANCE_WGSL, ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE } from './wgsl/atrousVariance.wgsl.js';
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
import { float32ToFloat16Bits, float16BitsToFloat32 } from './halfFloat.js';
import { getSharedTestWebGPUDevice } from './sharedWebGpuDevice.js';
import { alignedTextureCopyBytesPerRow } from './webGpuTextureCopy.js';

// ============================================================
// CPU-emulation helpers (used by tests; no GPU required)
// ============================================================

/**
 * CPU emulation of svgfReprojMain — for unit tests only.
 *
 * Performs the per-pixel bilinear reprojection + disocclusion test + EMA
 * blend identical to the WGSL kernel, operating on flat Float32Array/Uint32Array
 * buffers instead of GPU textures.
 *
 * @returns Updated { color, historyLength, moments } arrays.
 */
export interface SVGFReprojCPUInput {
  /** Current-frame RGB, row-major, length W*H*3. */
  readonly currColor:       Float32Array;
  /** Previous-frame RGB (post-EMA), row-major, length W*H*3. */
  readonly prevColor:       Float32Array;
  /** Screen-space motion vector (pixel delta), RG interleaved, length W*H*2. */
  readonly motionVec:       Float32Array;
  /** Current linear depth per pixel, length W*H. */
  readonly currDepth:       Float32Array;
  /** Current world-space normals, packed 0..1 in XYZ, row-major, length W*H*3. */
  readonly currNormal:      Float32Array;
  /** Current object ID per pixel (u32), length W*H. */
  readonly currObjId:       Uint32Array;
  /** Previous frame depth per pixel, length W*H. */
  readonly prevDepth:       Float32Array;
  /** Previous frame normals, packed 0..1, length W*H*3. */
  readonly prevNormal:      Float32Array;
  /** Previous object ID per pixel, length W*H. */
  readonly prevObjId:       Uint32Array;
  /** Previous history-length per pixel (u32), length W*H. */
  readonly historyLengthIn: Uint32Array;
  /** Previous moments M1, M2 interleaved, length W*H*2. */
  readonly momentsIn:       Float32Array;
  readonly width:  number;
  readonly height: number;
  readonly sigmaDepth?:  number;
  readonly sigmaNormal?: number;
  readonly alphaMin?:    number;
}

export interface SVGFReprojCPUOutput {
  readonly colorOut:        Float32Array;
  readonly historyLengthOut: Uint32Array;
  readonly momentsOut:      Float32Array;
}

/** Rec. 709 luminance. */
function lumCPU(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * CPU emulation of svgfReprojMain (used by unit tests).
 * Identical logic to the WGSL kernel; operates on flat arrays.
 */
export function svgfReprojCPU(input: SVGFReprojCPUInput): SVGFReprojCPUOutput {
  const W = input.width;
  const H = input.height;
  const sigmaDepth  = input.sigmaDepth  ?? SVGF_REPROJ_DEFAULT_UNIFORMS.sigmaDepth;
  const sigmaNormal = input.sigmaNormal ?? SVGF_REPROJ_DEFAULT_UNIFORMS.sigmaNormal;
  const alphaMin    = input.alphaMin    ?? SVGF_REPROJ_DEFAULT_UNIFORMS.alphaMin;

  const colorOut   = new Float32Array(W * H * 3);
  const histOut    = new Uint32Array(W * H);
  const momentsOut = new Float32Array(W * H * 2);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      const mvX = input.motionVec[pi * 2]     ?? 0;
      const mvY = input.motionVec[pi * 2 + 1] ?? 0;

      const prevFX = x + mvX;
      const prevFY = y + mvY;
      const prevX  = Math.floor(prevFX);
      const prevY  = Math.floor(prevFY);
      const fracX  = prevFX - prevX;
      const fracY  = prevFY - prevY;

      const zCurr = input.currDepth[pi] ?? 0;
      const nCurrX = (input.currNormal[pi * 3]     ?? 0) * 2 - 1;
      const nCurrY = (input.currNormal[pi * 3 + 1] ?? 0) * 2 - 1;
      const nCurrZ = (input.currNormal[pi * 3 + 2] ?? 0) * 2 - 1;
      const objIdCurr = input.currObjId[pi] ?? 0;

      // Bilinear taps
      const tapOffsets: [number, number][] = [[0,0],[1,0],[0,1],[1,1]];
      const tapWeights = [
        (1 - fracX) * (1 - fracY),
        fracX       * (1 - fracY),
        (1 - fracX) * fracY,
        fracX       * fracY,
      ];

      let accR = 0, accG = 0, accB = 0;
      let accM1 = 0, accM2 = 0, accH = 0, accW = 0;

      for (let t = 0; t < 4; t++) {
        const [ox, oy] = tapOffsets[t]!;
        const tx = prevX + ox;
        const ty = prevY + oy;
        if (tx < 0 || ty < 0 || tx >= W || ty >= H) continue;

        const ti = ty * W + tx;
        const zPrev = input.prevDepth[ti] ?? 0;
        const nPX = (input.prevNormal[ti * 3]     ?? 0) * 2 - 1;
        const nPY = (input.prevNormal[ti * 3 + 1] ?? 0) * 2 - 1;
        const nPZ = (input.prevNormal[ti * 3 + 2] ?? 0) * 2 - 1;
        const oPrev = input.prevObjId[ti] ?? 0;

        // Disocclusion test (Eq. 2)
        if (Math.abs(zCurr - zPrev) > sigmaDepth * Math.max(zCurr, zPrev) + 1e-4) continue;
        const nDot = nCurrX * nPX + nCurrY * nPY + nCurrZ * nPZ;
        if (nDot < sigmaNormal) continue;
        if (oPrev !== objIdCurr) continue;

        const w = tapWeights[t]!;
        accR += (input.prevColor[ti * 3]     ?? 0) * w;
        accG += (input.prevColor[ti * 3 + 1] ?? 0) * w;
        accB += (input.prevColor[ti * 3 + 2] ?? 0) * w;
        accM1 += (input.momentsIn[ti * 2]     ?? 0) * w;
        accM2 += (input.momentsIn[ti * 2 + 1] ?? 0) * w;
        accH  += (input.historyLengthIn[ti] ?? 0) * w;
        accW  += w;
      }

      const currR = input.currColor[pi * 3]     ?? 0;
      const currG = input.currColor[pi * 3 + 1] ?? 0;
      const currB = input.currColor[pi * 3 + 2] ?? 0;

      let newH: number;
      let alpha: number;
      let prevR = 0, prevG = 0, prevB = 0;
      let prevM1 = 0, prevM2 = 0;

      if (accW > 1e-6) {
        const invW = 1 / accW;
        prevR = accR * invW;
        prevG = accG * invW;
        prevB = accB * invW;
        prevM1 = accM1 * invW;
        prevM2 = accM2 * invW;
        newH = Math.round(accH * invW) + 1;
        alpha = Math.max(alphaMin, 1 / (newH + 1));
      } else {
        newH = 1;
        alpha = 1;
      }

      const blendR = alpha * currR + (1 - alpha) * prevR;
      const blendG = alpha * currG + (1 - alpha) * prevG;
      const blendB = alpha * currB + (1 - alpha) * prevB;
      const lCurr  = lumCPU(currR, currG, currB);
      const newM1  = alpha * lCurr          + (1 - alpha) * prevM1;
      const newM2  = alpha * lCurr * lCurr  + (1 - alpha) * prevM2;

      colorOut[pi * 3]     = blendR;
      colorOut[pi * 3 + 1] = blendG;
      colorOut[pi * 3 + 2] = blendB;
      histOut[pi]          = newH;
      momentsOut[pi * 2]   = newM1;
      momentsOut[pi * 2+1] = newM2;
    }
  }

  return { colorOut, historyLengthOut: histOut, momentsOut };
}

/**
 * CPU emulation of svgfVarianceFromMomentsMain.
 * Returns a Float32Array of scalar variance per pixel (length W*H).
 */
export function svgfVarianceFromMomentsCPU(opts: {
  readonly momentsIn:   Float32Array; // M1,M2 interleaved, length W*H*2
  readonly historyIn:   Uint32Array;  // length W*H
  readonly width:       number;
  readonly height:      number;
  readonly historyMin?: number;       // default 4
}): Float32Array {
  const { width: W, height: H, momentsIn, historyIn } = opts;
  const threshold = opts.historyMin ?? 4;
  const out = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const h = historyIn[i] ?? 0;
    if (h >= threshold) {
      const m1 = momentsIn[i * 2]     ?? 0;
      const m2 = momentsIn[i * 2 + 1] ?? 0;
      out[i] = Math.max(0, m2 - m1 * m1);
    }
  }
  return out;
}

/**
 * CPU emulation of svgf7x7FallbackMain.
 * Returns a Float32Array of scalar variance per pixel (length W*H),
 * merging spatial estimates for pixels with insufficient history.
 */
export function svgf7x7FallbackCPU(opts: {
  readonly currColor:   Float32Array; // RGB row-major, length W*H*3
  readonly historyIn:   Uint32Array;  // length W*H
  readonly varianceIn:  Float32Array; // from svgfVarianceFromMomentsCPU, length W*H
  readonly width:       number;
  readonly height:      number;
  readonly historyMin?: number;       // default 4
}): Float32Array {
  const { width: W, height: H, currColor, historyIn, varianceIn } = opts;
  const threshold = opts.historyMin ?? 4;
  const out = new Float32Array(W * H);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      const h = historyIn[pi] ?? 0;

      if (h >= threshold) {
        out[pi] = varianceIn[pi] ?? 0;
        continue;
      }

      // 7×7 spatial variance estimate
      let sumL = 0, sumL2 = 0, n = 0;
      for (let dy = -3; dy <= 3; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          const ni = ny * W + nx;
          const lum = lumCPU(
            currColor[ni * 3]     ?? 0,
            currColor[ni * 3 + 1] ?? 0,
            currColor[ni * 3 + 2] ?? 0,
          );
          sumL  += lum;
          sumL2 += lum * lum;
          n++;
        }
      }
      if (n > 1) {
        const mean = sumL / n;
        out[pi] = Math.max(0, sumL2 / n - mean * mean);
      }
    }
  }

  return out;
}

// ============================================================
// Pipeline cache (keyed by GPUDevice)
// ============================================================

interface SVGFRealPipelineBundle {
  readonly reprojPipeline:    GPUComputePipeline;
  readonly momentsPipeline:   GPUComputePipeline;
  readonly fallbackPipeline:  GPUComputePipeline;
  readonly atrousPipeline:    GPUComputePipeline;
}

const svgfRealPipelinesByDevice = new WeakMap<GPUDevice, SVGFRealPipelineBundle>();

function svgfRealPipelines(device: GPUDevice): SVGFRealPipelineBundle {
  let bundle = svgfRealPipelinesByDevice.get(device);
  if (bundle == null) {
    const reprojSM   = device.createShaderModule({ label: 'svgf-reproj',    code: SVGF_REPROJECTION_WGSL });
    const momentsSM  = device.createShaderModule({ label: 'svgf-moments',   code: SVGF_VARIANCE_FROM_MOMENTS_WGSL });
    const fallbackSM = device.createShaderModule({ label: 'svgf-7x7',       code: SVGF_7X7_SPATIAL_FALLBACK_WGSL });
    const atrousVarianceSM = device.createShaderModule({ label: 'svgf-real-atrous-variance', code: ATROUS_VARIANCE_WGSL });

    bundle = {
      reprojPipeline: device.createComputePipeline({
        label: 'svgf-real-reproj',
        layout: 'auto',
        compute: { module: reprojSM, entryPoint: 'svgfReprojMain' },
      }),
      momentsPipeline: device.createComputePipeline({
        label: 'svgf-real-moments',
        layout: 'auto',
        compute: { module: momentsSM, entryPoint: 'svgfVarianceFromMomentsMain' },
      }),
      fallbackPipeline: device.createComputePipeline({
        label: 'svgf-real-7x7',
        layout: 'auto',
        compute: { module: fallbackSM, entryPoint: 'svgf7x7FallbackMain' },
      }),
      atrousPipeline: device.createComputePipeline({
        label: 'svgf-real-atrous',
        layout: 'auto',
        compute: { module: atrousVarianceSM, entryPoint: 'svgfAtrousMain' },
      }),
    };
    svgfRealPipelinesByDevice.set(device, bundle);
  }
  return bundle;
}

// ============================================================
// Helpers — texture upload / readback
// ============================================================

function uploadRgbAsRgba16f(
  device: GPUDevice,
  texture: GPUTexture,
  rgb: Float32Array,
  w: number,
  h: number,
): void {
  const bpr = alignedTextureCopyBytesPerRow(w, 8);
  const buf = new Uint8Array(bpr * h);
  const dv = new DataView(buf.buffer);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const si = (y * w + x) * 3;
      const byte = y * bpr + x * 8;
      dv.setUint16(byte + 0, float32ToFloat16Bits(rgb[si]     ?? 0), true);
      dv.setUint16(byte + 2, float32ToFloat16Bits(rgb[si + 1] ?? 0), true);
      dv.setUint16(byte + 4, float32ToFloat16Bits(rgb[si + 2] ?? 0), true);
      dv.setUint16(byte + 6, float32ToFloat16Bits(1),                true);
    }
  }
  device.queue.writeTexture({ texture }, buf.buffer as GPUAllowSharedBufferSource, { bytesPerRow: bpr }, [w, h]);
}

function uploadRg32f(
  device: GPUDevice,
  texture: GPUTexture,
  rg: Float32Array,
  w: number,
  h: number,
): void {
  const bpr = alignedTextureCopyBytesPerRow(w, 8);
  // bpr is aligned to 256 bytes (WebGPU minimum), which is always divisible by 4,
  // so stride (f32 elements per row) is always an integer.
  const stride = bpr / 4;
  const buf = new Float32Array(stride * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const si = (y * w + x) * 2;
      const o  = y * stride + x * 2;
      buf[o]     = rg[si]     ?? 0;
      buf[o + 1] = rg[si + 1] ?? 0;
    }
  }
  device.queue.writeTexture(
    { texture },
    buf.buffer as GPUAllowSharedBufferSource,
    { bytesPerRow: bpr },
    [w, h],
  );
}

function fillR16Uint(device: GPUDevice, texture: GPUTexture, w: number, h: number, value: number): void {
  const bpr = alignedTextureCopyBytesPerRow(w, 2);
  const buf = new Uint8Array(bpr * h);
  const dv  = new DataView(buf.buffer);
  const v   = value & 0xFFFF;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      dv.setUint16(y * bpr + x * 2, v, true);
    }
  }
  device.queue.writeTexture({ texture }, buf.buffer as GPUAllowSharedBufferSource, { bytesPerRow: bpr }, [w, h]);
}

function fillRg32f(device: GPUDevice, texture: GPUTexture, w: number, h: number, r: number, g: number): void {
  // Allocate tight RG buffer; uploadRg32f handles row-alignment internally.
  const data = new Float32Array(w * h * 2);
  if (r !== 0 || g !== 0) {
    for (let i = 0; i < w * h; i++) { data[i * 2] = r; data[i * 2 + 1] = g; }
  }
  uploadRg32f(device, texture, data, w, h);
}

function fillRgba32f(device: GPUDevice, texture: GPUTexture, w: number, h: number, rgba: [number,number,number,number]): void {
  const bpr = alignedTextureCopyBytesPerRow(w, 16);
  const stride = bpr / 4;
  const buf = new Float32Array(stride * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = y * stride + x * 4;
      buf[o] = rgba[0]; buf[o+1] = rgba[1]; buf[o+2] = rgba[2]; buf[o+3] = rgba[3];
    }
  }
  device.queue.writeTexture(
    { texture },
    buf.buffer as GPUAllowSharedBufferSource,
    { bytesPerRow: bpr },
    [w, h],
  );
}

function uploadR32f(device: GPUDevice, texture: GPUTexture, data: Float32Array, w: number, h: number): void {
  const bpr = alignedTextureCopyBytesPerRow(w, 4);
  const stride = bpr / 4;
  const buf = new Float32Array(stride * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      buf[y * stride + x] = data[y * w + x] ?? 0;
    }
  }
  device.queue.writeTexture(
    { texture },
    buf.buffer as GPUAllowSharedBufferSource,
    { bytesPerRow: bpr },
    [w, h],
  );
}

function uploadR32Uint(device: GPUDevice, texture: GPUTexture, data: Uint32Array, w: number, h: number): void {
  const bpr = alignedTextureCopyBytesPerRow(w, 4);
  const stride = bpr / 4;
  const buf = new Uint32Array(stride * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      buf[y * stride + x] = data[y * w + x] ?? 0;
    }
  }
  device.queue.writeTexture(
    { texture },
    buf.buffer as GPUAllowSharedBufferSource,
    { bytesPerRow: bpr },
    [w, h],
  );
}

async function readRgba16fToRgb(device: GPUDevice, texture: GPUTexture, w: number, h: number): Promise<Float32Array> {
  const bpr = alignedTextureCopyBytesPerRow(w, 8);
  const buf = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
  device.queue.submit([encoder.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const raw = new Uint8Array(buf.getMappedRange());
  const dv  = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const out = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const byte = y * bpr + x * 8;
      const di   = (y * w + x) * 3;
      out[di]     = float16BitsToFloat32(dv.getUint16(byte + 0, true));
      out[di + 1] = float16BitsToFloat32(dv.getUint16(byte + 2, true));
      out[di + 2] = float16BitsToFloat32(dv.getUint16(byte + 4, true));
    }
  }
  buf.unmap();
  buf.destroy();
  return out;
}

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

  /** Explicit GPU device (never destroyed by this call). REQUIRED unless `reuseSharedWebGpuDevice: true` is set. */
  readonly device?: GPUDevice;
  /**
   * Opt-in to the test/demo singleton from `sharedWebGpuDevice.ts`.
   * Default `false` (W6-E1, 2026-05-17): production callers MUST pass `device`.
   * Set to `true` only from tests / Cornell-style demos that intentionally
   * share a process-wide device for adapter latency.
   *
   * If neither `device` nor `reuseSharedWebGpuDevice: true` is supplied, the
   * call throws — the singleton fallback is no longer implicit.
   */
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
  const reuseShared = opts.reuseSharedWebGpuDevice === true && opts.device == null;
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
  if (opts.device != null) {
    device = opts.device;
  } else if (reuseShared) {
    device = await getSharedTestWebGPUDevice();
  } else {
    throw new Error(
      'runSVGFRealWebGPU: pass an explicit `device: GPUDevice` (host owns lifecycle, ' +
        'CLAUDE.md design principle #2). Tests / demos may opt in to the shared singleton ' +
        'with `reuseSharedWebGpuDevice: true`.',
    );
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
  uploadRgbAsRgba16f(device, currColorTex, opts.rgb, w, h);
  uploadRgbAsRgba16f(device, prevColorTex, opts.prevRadianceRgb ?? opts.rgb, w, h);

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
    device.queue.writeTexture({ texture: currNormTex }, normBuf.buffer as GPUAllowSharedBufferSource, { bytesPerRow: bpr }, [w,h]);
    device.queue.writeTexture({ texture: prevNormTex }, normBuf.buffer as GPUAllowSharedBufferSource, { bytesPerRow: bpr }, [w,h]);
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
    fillR16Uint(device, histInTex, w, h, 0);
    // Upload actual values
    const bpr = alignedTextureCopyBytesPerRow(w, 2);
    const hBuf = new Uint8Array(bpr * h);
    const dv = new DataView(hBuf.buffer);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        dv.setUint16(y * bpr + x * 2, (opts.historyLengthIn[y * w + x] ?? 0) & 0xFFFF, true);
      }
    }
    device.queue.writeTexture({ texture: histInTex }, hBuf.buffer as GPUAllowSharedBufferSource, { bytesPerRow: bpr }, [w,h]);
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
    pass.setBindGroup(0, atrousBGs[iter]!);
    pass.dispatchWorkgroups(Math.ceil(w / wgA), Math.ceil(h / wgA));
    pass.end();
  }

  device.queue.submit([encoder.finish()]);

  // Read result: after N iterations, last write is in pong (odd) or ping (even).
  const readTex = atrousIterations % 2 === 0 ? pingTex : pongTex;
  const result  = await readRgba16fToRgb(device, readTex, w, h);

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
  // Device ownership stays with the caller (W6-E1) — never destroyed here.

  return result;
}
