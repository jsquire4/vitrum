/**
 * bmfrWebGPU.ts — One-shot host pipeline for the BMFR denoiser.
 *
 * BMFR = Koskela, Immonen, Mäkitalo, Foi, Viitanen, Jääskeläinen, Kultala,
 * Takala. "Blockwise Multi-Order Feature Regression for Real-Time Path-Tracing
 * Reconstruction." ACM Transactions on Graphics 38(5), 2019.
 *
 * Runs overlapping blockwise least-squares feature regression on CPU-backed
 * linear HDR RGB via WebGPU, then (optionally) temporally accumulates against
 * a previous-frame reconstruction. The fit pass applies Householder QR directly
 * to the regularized rectangular system for [1, p.xyz, n.xyz, p².xyz]; a second
 * pass deterministically resolves overlapping block fits into `color = T·α`
 * (see `wgsl/bmfr.wgsl.ts`).
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

import {
  BMFR_BLOCK_FIT_SIZE_BYTES,
  BMFR_RESOLVE_WORKGROUP_SIZE,
  BMFR_WGSL,
} from './wgsl/bmfr.wgsl.js';
import { BMFR_BLOCK_SIZE } from './bmfrRegression.js';
import {
  BMFR_DEFAULT_UNIFORMS,
  BMFR_UNIFORMS_SIZE_BYTES,
  packBmfrUniforms,
  type BmfrUniforms,
} from './bmfrBindings.js';
import { acquireDenoiseDevice, makePerDevicePipelineCache } from './sharedWebGpuDevice.js';
import { makeResourceTracker } from './atrousChain.js';
import {
  assertFiniteFloat16Slice,
  assertFiniteFloatSlice,
  assertFiniteNumber,
  assertOneShotArrayLength,
  assertOneShotDeviceLimits,
  assertOneShotDimensions,
} from './webGpuOneShotValidation.js';
import { demodulateAlbedo, remodulateAlbedo } from './albedoModulation.js';
import { alignedTextureCopyBytesPerRow } from './webGpuTextureCopy.js';
import {
  uploadRgbAsRgba16f,
  readRgba16fToRgb,
} from './webGpuTextureUpload.js';

const BMFR_FIT_ENTRY = 'bmfrMain';
const BMFR_RESOLVE_ENTRY = 'bmfrResolve';

const bmfrFitPipeline = makePerDevicePipelineCache<GPUComputePipeline>(
  (device) => {
    const module = device.createShaderModule({ label: 'bmfr-fit', code: BMFR_WGSL });
    return device.createComputePipeline({
      label: 'bmfr-fit',
      layout: 'auto',
      compute: { module, entryPoint: BMFR_FIT_ENTRY },
    });
  },
);

const bmfrResolvePipeline = makePerDevicePipelineCache<GPUComputePipeline>(
  (device) => {
    const module = device.createShaderModule({ label: 'bmfr-resolve', code: BMFR_WGSL });
    return device.createComputePipeline({
      label: 'bmfr-resolve',
      layout: 'auto',
      compute: { module, entryPoint: BMFR_RESOLVE_ENTRY },
    });
  },
);

export interface BmfrWebGPUOptions {
  /** Current-frame noisy HDR radiance (row-major RGB, length W*H*3). */
  readonly rgb: Float32Array;
  readonly width: number;
  readonly height: number;

  /**
   * World-space position per pixel (row-major XYZ, length W*H*3). BMFR's
   * feature matrix requires this stream; a normal-only fit path is not
   * implemented.
   */
  readonly worldPosRgb: Float32Array;
  /**
   * Per-pixel surface validity / linear depth (length W*H). Pixels with
   * value <= 0 are treated as sky/miss and pass through unfiltered. When
   * omitted, all pixels with a supplied worldPos are considered valid.
   */
  readonly validityW?: Float32Array;
  /** World-space normals (packed 0..1 XYZ), length W*H*3. Defaults to +Z. */
  readonly gbufferNormalsRgb?: Float32Array;
  /**
   * Previous-frame reconstruction (remodulated RGB, length W*H*3) for temporal
   * blend. This is the same radiance domain returned by `runBmfrWebGPU`.
   */
  readonly historyRgb?: Float32Array;
  /**
   * Previous-frame albedo corresponding to `historyRgb`. Required when both
   * `albedoRgb` and `historyRgb` are supplied so historical radiance is
   * demodulated in its own material domain rather than the current frame's.
   */
  readonly historyAlbedoRgb?: Float32Array;
  /** Per-pixel diffuse albedo (RGB, length W*H*3); enables demodulation. */
  readonly albedoRgb?: Float32Array;

  /** Square block edge in pixels (default BMFR_BLOCK_SIZE = 32). */
  readonly blockSize?: number;
  /**
   * Block-grid stride. Values from ceil(blockSize/2) through blockSize are
   * supported. Overlapping fits are averaged deterministically by a resolve
   * pass. The default is half a block.
   */
  readonly blockStride?: number;
  /** World-space normalisation scale for the squared features. */
  readonly positionScale?: number;
  /** Temporal EMA weight on the current frame (only used when historyRgb set). */
  readonly temporalAlpha?: number;
  /** Tikhonov loading represented as augmented identity rows in direct QR. */
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
  const label = 'runBmfrWebGPU';
  const px = assertOneShotDimensions(label, w, h);
  const check = (name: string, value: Float32Array, length: number): void => {
    assertOneShotArrayLength(label, name, value, length);
    assertFiniteFloatSlice(label, name, value, length);
  };
  check('rgb', opts.rgb, px * 3);
  if (opts.worldPosRgb == null) {
    throw new TypeError(`${label}: worldPosRgb is required`);
  }
  check('worldPosRgb', opts.worldPosRgb, px * 3);
  if (opts.validityW != null) check('validityW', opts.validityW, px);
  if (opts.gbufferNormalsRgb != null) {
    check('gbufferNormalsRgb', opts.gbufferNormalsRgb, px * 3);
  }
  if (opts.historyRgb != null) check('historyRgb', opts.historyRgb, px * 3);
  if (opts.historyAlbedoRgb != null) {
    check('historyAlbedoRgb', opts.historyAlbedoRgb, px * 3);
  }
  if (opts.albedoRgb != null) check('albedoRgb', opts.albedoRgb, px * 3);
  if (opts.historyAlbedoRgb != null && opts.albedoRgb == null) {
    throw new Error(`${label}: historyAlbedoRgb requires albedoRgb`);
  }
  if (opts.historyAlbedoRgb != null && opts.historyRgb == null) {
    throw new Error(`${label}: historyAlbedoRgb requires historyRgb`);
  }

  const blockSize = opts.blockSize ?? BMFR_BLOCK_SIZE;
  assertFiniteNumber(label, 'blockSize', blockSize, {
    integer: true,
    min: 2,
    max: BMFR_BLOCK_SIZE,
  });
  const blockStride = opts.blockStride ?? Math.ceil(blockSize / 2);
  assertFiniteNumber(label, 'blockStride', blockStride, {
    integer: true,
    min: Math.ceil(blockSize / 2),
    max: blockSize,
  });
  const positionScale = opts.positionScale ?? BMFR_DEFAULT_UNIFORMS.positionScale;
  const temporalAlpha = opts.temporalAlpha ?? BMFR_DEFAULT_UNIFORMS.temporalAlpha;
  const regularisation = opts.regularisation ?? BMFR_DEFAULT_UNIFORMS.regularisation;
  assertFiniteNumber(label, 'positionScale', positionScale, { min: 0 });
  assertFiniteNumber(label, 'temporalAlpha', temporalAlpha, { min: 0, max: 1 });
  assertFiniteNumber(label, 'regularisation', regularisation, { min: 0 });

  if (positionScale <= 0) {
    throw new Error(`${label}: positionScale must be > 0; received ${positionScale}`);
  }

  const blocksX = Math.ceil(w / blockStride);
  const blocksY = Math.ceil(h / blockStride);
  const blockFitCount = blocksX * blocksY;
  const blockFitBytes = blockFitCount * BMFR_BLOCK_FIT_SIZE_BYTES;
  if (!Number.isSafeInteger(blockFitBytes)) {
    throw new Error(`${label}: block-fit storage size exceeds the safe integer range`);
  }

  const uniforms: BmfrUniforms = {
    blockSize,
    blockStride,
    positionScale,
    temporalAlpha,
    regularisation,
    hasHistory: opts.historyRgb != null ? 1 : 0,
    // The one-shot host path supplies a real world-position buffer (mode 0).
    positionMode: 0,
  };
  const rgbForFit = opts.albedoRgb != null
    ? demodulateAlbedo(opts.rgb, opts.albedoRgb, px)
    : opts.rgb;
  const normalsForUpload =
    opts.gbufferNormalsRgb ?? packedFlatNormals(px);
  let historyForTemporal: Float32Array;
  if (opts.historyRgb == null) {
    historyForTemporal = new Float32Array(px * 3);
  } else if (opts.albedoRgb == null) {
    historyForTemporal = opts.historyRgb;
  } else {
    const historyAlbedo = opts.historyAlbedoRgb;
    if (historyAlbedo == null) {
      throw new Error(
        `${label}: historyAlbedoRgb is required when albedoRgb and historyRgb are supplied`,
      );
    }
    historyForTemporal = demodulateAlbedo(opts.historyRgb, historyAlbedo, px);
  }
  assertFiniteFloat16Slice(label, 'rgbForFit', rgbForFit, px * 3);
  assertFiniteFloat16Slice(
    label,
    'gbufferNormalsRgb',
    normalsForUpload,
    px * 3,
  );
  assertFiniteFloat16Slice(
    label,
    'historyForTemporal',
    historyForTemporal,
    px * 3,
  );

  const { device, dispose: destroyEphemeral } = await acquireDenoiseDevice({
    device: opts.device,
    reuseSharedWebGpuDevice: opts.reuseSharedWebGpuDevice,
    errorLabel: label,
  });

  const { trackTexture, trackBuffer, dispose: disposeResources } =
    makeResourceTracker(destroyEphemeral);

  try {
    assertOneShotDeviceLimits(device, label, w, h, 8);
    const fitPipeline = bmfrFitPipeline(device);
    const resolvePipeline = bmfrResolvePipeline(device);

    const maxBufferSize = device.limits?.maxBufferSize;
    const maxStorageBinding = device.limits?.maxStorageBufferBindingSize;
    if (typeof maxBufferSize === 'number' && blockFitBytes > maxBufferSize) {
      throw new Error(`${label}: block-fit buffer exceeds maxBufferSize (${maxBufferSize})`);
    }
    if (
      typeof maxStorageBinding === 'number' &&
      blockFitBytes > maxStorageBinding
    ) {
      throw new Error(
        `${label}: block-fit buffer exceeds maxStorageBufferBindingSize (${maxStorageBinding})`,
      );
    }

    const texB = GPUTextureUsage.TEXTURE_BINDING;
    const texC = GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
    const texS = GPUTextureUsage.STORAGE_BINDING;

    const colorTex   = trackTexture(device.createTexture({ label: 'bmfr-color',   size: [w, h], format: 'rgba16float', usage: texB | texC }));
    const normalTex  = trackTexture(device.createTexture({ label: 'bmfr-normal',  size: [w, h], format: 'rgba16float', usage: texB | texC }));
    const worldPosTex = trackTexture(device.createTexture({ label: 'bmfr-worldpos', size: [w, h], format: 'rgba32float', usage: texB | texC }));
    const historyTex = trackTexture(device.createTexture({ label: 'bmfr-history', size: [w, h], format: 'rgba16float', usage: texB | texC }));
    const outTex     = trackTexture(device.createTexture({ label: 'bmfr-out',     size: [w, h], format: 'rgba16float', usage: texS | texB | texC }));
    const blockFits = trackBuffer(device.createBuffer({
      label: 'bmfr-block-fits',
      size: blockFitBytes,
      usage: GPUBufferUsage.STORAGE,
    }));

    // Color (demodulated by albedo when supplied).
    uploadRgbAsRgba16f(device, colorTex, rgbForFit, w, h);

    // Normals (packed 0..1); default forward-facing +Z = packed (0.5,0.5,1).
    uploadRgbAsRgba16f(device, normalTex, normalsForUpload, w, h);

    // World position (XYZ in .xyz, validity/depth in .w). The kernel uses .w<=0
    // as the sky/miss sentinel.
    uploadWorldPosAsRgba32f(device, worldPosTex, opts, w, h);

    // History must enter the resolve kernel in the same domain as the current
    // fit. `historyRgb` is a prior public result and is therefore remodulated
    // radiance, while `rgbForFit` is c/ρ when albedo is supplied. Demodulate
    // history with its own frame's albedo before the EMA; the result is
    // remodulated with the current albedo exactly once after readback below.
    // Only sampled when hasHistory; zero-fill otherwise so the texture is
    // initialised — the kernel skips the read unless hasHistory.
    uploadRgbAsRgba16f(
      device,
      historyTex,
      historyForTemporal,
      w,
      h,
    );

    // UBO.
    const uboScratch = new ArrayBuffer(BMFR_UNIFORMS_SIZE_BYTES);
    packBmfrUniforms(uniforms, uboScratch);
    const ubo = trackBuffer(device.createBuffer({
      label: 'bmfr-ubo',
      size: BMFR_UNIFORMS_SIZE_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }));
    device.queue.writeBuffer(ubo, 0, uboScratch);

    const fitBg = device.createBindGroup({
      label: 'bmfr-fit-bg',
      layout: fitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: colorTex.createView() },
        { binding: 1, resource: normalTex.createView() },
        { binding: 2, resource: worldPosTex.createView() },
        { binding: 4, resource: { buffer: blockFits } },
        { binding: 5, resource: { buffer: ubo } },
      ],
    });
    const resolveBg = device.createBindGroup({
      label: 'bmfr-resolve-bg',
      layout: resolvePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: colorTex.createView() },
        { binding: 1, resource: normalTex.createView() },
        { binding: 2, resource: worldPosTex.createView() },
        { binding: 3, resource: historyTex.createView() },
        { binding: 4, resource: { buffer: blockFits } },
        { binding: 5, resource: { buffer: ubo } },
        { binding: 6, resource: outTex.createView() },
      ],
    });

    const encoder = device.createCommandEncoder({ label: 'bmfr' });
    const fitPass = encoder.beginComputePass({ label: 'bmfr-fit' });
    fitPass.setPipeline(fitPipeline);
    fitPass.setBindGroup(0, fitBg);
    fitPass.dispatchWorkgroups(blocksX, blocksY, 1);
    fitPass.end();
    const resolvePass = encoder.beginComputePass({ label: 'bmfr-resolve' });
    resolvePass.setPipeline(resolvePipeline);
    resolvePass.setBindGroup(0, resolveBg);
    resolvePass.dispatchWorkgroups(
      Math.ceil(w / BMFR_RESOLVE_WORKGROUP_SIZE),
      Math.ceil(h / BMFR_RESOLVE_WORKGROUP_SIZE),
      1,
    );
    resolvePass.end();
    device.queue.submit([encoder.finish()]);

    const result = await readRgba16fToRgb(device, outTex, w, h);

    if (opts.albedoRgb != null) {
      remodulateAlbedo(result, opts.albedoRgb, px);
    }

    return result;
  } finally {
    disposeResources();
  }
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
 * Validity comes from `validityW` when supplied, else every pixel is valid.
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
      buf[o] = wp[si]!;
      buf[o + 1] = wp[si + 1]!;
      buf[o + 2] = wp[si + 2]!;
      // .w = validity (>0 => fit this pixel). Default: all supplied positions
      // are valid; callers mark sky/misses explicitly through validityW.
      buf[o + 3] = valid != null ? (valid[pi] ?? 0) : 1;
    }
  }
  device.queue.writeTexture({ texture }, buf.buffer, { bytesPerRow: bpr, rowsPerImage: height }, [width, height]);
}
