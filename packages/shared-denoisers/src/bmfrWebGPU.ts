/**
 * bmfrWebGPU.ts — One-shot host pipeline for the BMFR denoiser.
 *
 * BMFR = Koskela, Immonen, Mäkitalo, Foi, Viitanen, Jääskeläinen, Kultala,
 * Takala. "Blockwise Multi-Order Feature Regression for Real-Time Path-Tracing
 * Reconstruction." ACM Transactions on Graphics 38(5), 2019.
 *
 * Runs the per-32×32-block least-squares feature regression on CPU-backed
 * linear HDR RGB via WebGPU, then (optionally) temporally accumulates against
 * a previous-frame reconstruction. One workgroup per block fits the noisy color
 * to a 10-feature matrix [1, p.xyz, n.xyz, p².xyz] via Householder QR on the
 * normal equations and reconstructs `color = T·α` (see `wgsl/bmfr.wgsl.ts`).
 *
 * Albedo demodulation (Schied 2017 §4.1 convention, matching the other vitrum
 * denoisers): when `albedoRgb` is supplied the fit runs on `L = c/ρ` and the
 * result is re-multiplied by ρ. This keeps high-frequency albedo variation out
 * of the smooth feature fit so material edges stay crisp.
 *
 * This one-shot path allocates and destroys all textures per call. In the
 * WalkaroundGPUPipeline (persistent mode), the registry entry
 * (`walkaround-hybrid/.../denoisers/bmfr.ts`) reuses the same WGSL kernel with
 * persistent ping-pong history textures.
 */

import { BMFR_WGSL, BMFR_WORKGROUP_SIZE } from './wgsl/bmfr.wgsl.js';
import { BMFR_BLOCK_SIZE } from './bmfrRegression.js';
import {
  BMFR_DEFAULT_UNIFORMS,
  BMFR_UNIFORMS_SIZE_BYTES,
  packBmfrUniforms,
  type BmfrUniforms,
} from './bmfrBindings.js';
import { acquireDenoiseDevice } from './sharedWebGpuDevice.js';
import { demodulateAlbedo, remodulateAlbedo } from './albedoModulation.js';
import { alignedTextureCopyBytesPerRow } from './webGpuTextureCopy.js';
import {
  uploadRgbAsRgba16f,
  readRgba16fToRgb,
} from './webGpuTextureUpload.js';

const BMFR_ENTRY = 'bmfrMain';

interface BmfrPipelineBundle {
  readonly pipeline: GPUComputePipeline;
}

const bmfrPipelinesByDevice = new WeakMap<GPUDevice, BmfrPipelineBundle>();

function bmfrPipeline(device: GPUDevice): GPUComputePipeline {
  let bundle = bmfrPipelinesByDevice.get(device);
  if (bundle == null) {
    const module = device.createShaderModule({ label: 'bmfr', code: BMFR_WGSL });
    bundle = {
      pipeline: device.createComputePipeline({
        label: 'bmfr',
        layout: 'auto',
        compute: { module, entryPoint: BMFR_ENTRY },
      }),
    };
    bmfrPipelinesByDevice.set(device, bundle);
  }
  return bundle.pipeline;
}

export interface BmfrWebGPUOptions {
  /** Current-frame noisy HDR radiance (row-major RGB, length W*H*3). */
  readonly rgb: Float32Array;
  readonly width: number;
  readonly height: number;

  /**
   * World-space position per pixel (row-major XYZ, length W*H*3). REQUIRED for
   * a meaningful fit — BMFR's feature matrix is dominated by the position
   * columns. When omitted, the kernel sees a flat z=0 plane and the regression
   * degrades to a normal-only fit (still valid, but loses the spatial term).
   */
  readonly worldPosRgb?: Float32Array;
  /**
   * Per-pixel surface validity / linear depth (length W*H). Pixels with
   * value <= 0 are treated as sky/miss and pass through unfiltered. When
   * omitted, all pixels with a supplied worldPos are considered valid.
   */
  readonly validityW?: Float32Array;
  /** World-space normals (packed 0..1 XYZ), length W*H*3. Defaults to +Z. */
  readonly gbufferNormalsRgb?: Float32Array;
  /** Previous-frame reconstruction (RGB, length W*H*3) for temporal blend. */
  readonly historyRgb?: Float32Array;
  /** Per-pixel diffuse albedo (RGB, length W*H*3); enables demodulation. */
  readonly albedoRgb?: Float32Array;

  /** Square block edge in pixels (default BMFR_BLOCK_SIZE = 32). */
  readonly blockSize?: number;
  /** Block grid stride in pixels (default = blockSize; < blockSize = overlap). */
  readonly blockStride?: number;
  /** World-space normalisation scale for the squared features. */
  readonly positionScale?: number;
  /** Temporal EMA weight on the current frame (only used when historyRgb set). */
  readonly temporalAlpha?: number;
  /** Tikhonov diagonal loading for the QR solve. */
  readonly regularisation?: number;

  /** Explicit GPU device (never destroyed by this call). */
  readonly device?: GPUDevice;
  /** When true, uses the process-shared WebGPU device. Default: false. */
  readonly reuseSharedWebGpuDevice?: boolean;
}

/**
 * Run a single BMFR reconstruction pass. Returns the filtered HDR RGB
 * (length width*height*3).
 */
export async function runBmfrWebGPU(opts: BmfrWebGPUOptions): Promise<Float32Array> {
  const w = opts.width;
  const h = opts.height;
  const px = w * h;
  if (w <= 0 || h <= 0 || opts.rgb.length < px * 3) {
    throw new Error('runBmfrWebGPU: invalid rgb buffer or dimensions');
  }

  // The kernel's thread tiling is 16×16 threads × 2×2 patch = a 32×32 footprint
  // per workgroup, so blockSize is clamped to [2, 32]: a larger block would
  // leave its trailing pixels uncovered by any thread.
  const blockSize = Math.min(BMFR_BLOCK_SIZE, Math.max(2, Math.floor(opts.blockSize ?? BMFR_BLOCK_SIZE)));
  const blockStride = Math.max(1, Math.floor(opts.blockStride ?? blockSize));

  const uniforms: BmfrUniforms = {
    blockSize,
    blockStride,
    positionScale: opts.positionScale ?? BMFR_DEFAULT_UNIFORMS.positionScale,
    temporalAlpha: opts.temporalAlpha ?? BMFR_DEFAULT_UNIFORMS.temporalAlpha,
    regularisation: opts.regularisation ?? BMFR_DEFAULT_UNIFORMS.regularisation,
    hasHistory: opts.historyRgb != null ? 1 : 0,
    // The one-shot host path supplies a real world-position buffer (mode 0).
    positionMode: 0,
  };

  const { device, dispose: destroyEphemeral } = await acquireDenoiseDevice({
    device: opts.device,
    reuseSharedWebGpuDevice: opts.reuseSharedWebGpuDevice,
    errorLabel: 'runBmfrWebGPU',
  });

  const pipeline = bmfrPipeline(device);

  const texB = GPUTextureUsage.TEXTURE_BINDING;
  const texC = GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
  const texS = GPUTextureUsage.STORAGE_BINDING;

  const colorTex   = device.createTexture({ label: 'bmfr-color',   size: [w, h], format: 'rgba16float', usage: texB | texC });
  const normalTex  = device.createTexture({ label: 'bmfr-normal',  size: [w, h], format: 'rgba16float', usage: texB | texC });
  const worldPosTex = device.createTexture({ label: 'bmfr-worldpos', size: [w, h], format: 'rgba32float', usage: texB | texC });
  const historyTex = device.createTexture({ label: 'bmfr-history', size: [w, h], format: 'rgba16float', usage: texB | texC });
  const outTex     = device.createTexture({ label: 'bmfr-out',     size: [w, h], format: 'rgba16float', usage: texS | texB | texC });

  // Color (demodulated by albedo when supplied).
  const rgbForFit = opts.albedoRgb != null
    ? demodulateAlbedo(opts.rgb, opts.albedoRgb, px)
    : opts.rgb;
  uploadRgbAsRgba16f(device, colorTex, rgbForFit, w, h);

  // Normals (packed 0..1); default forward-facing +Z = packed (0.5,0.5,1).
  if (opts.gbufferNormalsRgb != null) {
    uploadRgbAsRgba16f(device, normalTex, opts.gbufferNormalsRgb, w, h);
  } else {
    uploadRgbAsRgba16f(device, normalTex, packedFlatNormals(px), w, h);
  }

  // World position (XYZ in .xyz, validity/depth in .w). The kernel uses .w<=0
  // as the sky/miss sentinel.
  uploadWorldPosAsRgba32f(device, worldPosTex, opts, w, h);

  // History (only sampled when hasHistory; zero-fill otherwise so the
  // texture is initialised — the kernel skips the read unless hasHistory).
  uploadRgbAsRgba16f(device, historyTex, opts.historyRgb ?? new Float32Array(px * 3), w, h);

  // UBO.
  const uboScratch = new ArrayBuffer(BMFR_UNIFORMS_SIZE_BYTES);
  packBmfrUniforms(uniforms, uboScratch);
  const ubo = device.createBuffer({
    label: 'bmfr-ubo',
    size: BMFR_UNIFORMS_SIZE_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ubo, 0, uboScratch);

  const bg = device.createBindGroup({
    label: 'bmfr-bg',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: colorTex.createView() },
      { binding: 1, resource: normalTex.createView() },
      { binding: 2, resource: worldPosTex.createView() },
      { binding: 3, resource: historyTex.createView() },
      { binding: 4, resource: outTex.createView() },
      { binding: 5, resource: { buffer: ubo } },
    ],
  });

  // One workgroup per block origin. Grid covers all block origins that touch
  // the image (ceil so the trailing partial block is included).
  const blocksX = Math.ceil(w / blockStride);
  const blocksY = Math.ceil(h / blockStride);

  const encoder = device.createCommandEncoder({ label: 'bmfr' });
  const pass = encoder.beginComputePass({ label: 'bmfr-fit' });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(blocksX, blocksY, 1);
  pass.end();
  device.queue.submit([encoder.finish()]);

  const result = await readRgba16fToRgb(device, outTex, w, h);

  if (opts.albedoRgb != null) {
    remodulateAlbedo(result, opts.albedoRgb, px);
  }

  for (const t of [colorTex, normalTex, worldPosTex, historyTex, outTex]) t.destroy();
  ubo.destroy();
  destroyEphemeral();

  return result;
}

/** Forward-facing +Z normal, packed 0..1 → (0.5, 0.5, 1) per pixel. */
function packedFlatNormals(pixelCount: number): Float32Array {
  const n = new Float32Array(pixelCount * 3);
  for (let i = 0; i < pixelCount; i++) {
    n[i * 3] = 0.5;
    n[i * 3 + 1] = 0.5;
    n[i * 3 + 2] = 1.0;
  }
  return n;
}

/**
 * Upload world position (XYZ) + validity (.w) into an rgba32float texture.
 * Validity comes from `validityW` when supplied, else 1.0 for pixels with a
 * supplied worldPos (and 0.0 — sky sentinel — when neither worldPos nor
 * validity is given).
 */
function uploadWorldPosAsRgba32f(
  device: GPUDevice,
  texture: GPUTexture,
  opts: BmfrWebGPUOptions,
  width: number,
  height: number,
): void {
  const bpr = alignedTextureCopyBytesPerRow(width, 16);
  const stride = bpr / 4;
  const buf = new Float32Array(stride * height);
  const wp = opts.worldPosRgb;
  const valid = opts.validityW;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pi = y * width + x;
      const si = pi * 3;
      const o = y * stride + x * 4;
      buf[o] = wp?.[si] ?? 0;
      buf[o + 1] = wp?.[si + 1] ?? 0;
      buf[o + 2] = wp?.[si + 2] ?? 0;
      // .w = validity (>0 => fit this pixel). Default: 1 when worldPos given,
      // else 0 (so a worldPos-less call passes everything through unfiltered).
      buf[o + 3] = valid != null ? (valid[pi] ?? 0) : (wp != null ? 1 : 0);
    }
  }
  device.queue.writeTexture({ texture }, buf.buffer, { bytesPerRow: bpr, rowsPerImage: height }, [width, height]);
}
