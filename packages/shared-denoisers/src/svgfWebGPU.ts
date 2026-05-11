/**
 * One-shot SVGF (variance + à-trous) on CPU-backed linear HDR RGB via WebGPU.
 *
 * When optional g-buffer slices (`gbufferNormalsRgb`, `linearDepth`, `motionRg`,
 * `welfordMeanM2`) are omitted, fills synthetic buffers from
 * SVGF_SYNTHETIC_GBUFFER_DEFAULTS unless `syntheticGbufferFallback` overrides them.
 *
 * For temporal variance (`frameCount >= SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT`), hosts should
 * supply `welfordMeanM2` from the path accumulator (RG mean + M₂). Cornell-style demos
 * may omit it and accept the console warning plus zero-filled variance input.
 */

import { SVGF_WGSL, SVGF_COMPUTE_WORKGROUP_SIZE } from './wgsl/svgf.wgsl.js';
import {
  SVGF_DEFAULT_UNIFORMS,
  SVGF_UNIFORMS_SIZE_BYTES,
  SVGF_VARIANCE_UNIFORMS_SIZE_BYTES,
  packSVGFUniforms,
  packSVGFVarianceUniforms,
} from './svgfBindings.js';
import {
  SVGF_DEFAULT_ATROUS_ITERATIONS,
  SVGF_MAX_ATROUS_ITERATIONS,
  SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT,
} from './svgfConstants.js';
import { float32ToFloat16Bits, float16BitsToFloat32 } from './halfFloat.js';
import { getSharedWebGPUDevice } from './sharedWebGpuDevice.js';
import { alignedTextureCopyBytesPerRow } from './webGpuTextureCopy.js';

export interface SvgfSyntheticGbufferFallback {
  readonly normalRgb?: readonly [number, number, number];
  readonly linearDepth?: number;
}

/** Defaults when optional G-buffer slices are omitted (Cornell-style synthetic prepass). */
export const SVGF_SYNTHETIC_GBUFFER_DEFAULTS = {
  normalRgb: [0, 1, 0] as const,
  linearDepth: 2,
} as const;

const VARIANCE_ENTRY = 'svgfVarianceMain';
const ATROUS_ENTRY = 'svgfAtrousMain';

interface SvgfPipelineBundle {
  readonly variance: GPUComputePipeline;
  readonly atrous: GPUComputePipeline;
}

const svgfPipelinesByDevice = new WeakMap<GPUDevice, SvgfPipelineBundle>();

function svgfPipelines(device: GPUDevice): SvgfPipelineBundle {
  let bundle = svgfPipelinesByDevice.get(device);
  if (bundle == null) {
    const shaderModule = device.createShaderModule({ label: 'svgf', code: SVGF_WGSL });
    bundle = {
      variance: device.createComputePipeline({
        label: 'svgf-variance',
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: VARIANCE_ENTRY },
      }),
      atrous: device.createComputePipeline({
        label: 'svgf-atrous',
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: ATROUS_ENTRY },
      }),
    };
    svgfPipelinesByDevice.set(device, bundle);
  }
  return bundle;
}

function fillRgba32fTexture(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
  rgbaPerPixel: readonly [number, number, number, number],
): void {
  const bpp = 16;
  const bpr = alignedTextureCopyBytesPerRow(width, bpp);
  const upload = new Float32Array((bpr / 4) * height);
  for (let y = 0; y < height; y += 1) {
    const row = (y * bpr) / 4;
    for (let x = 0; x < width; x += 1) {
      const o = row + x * 4;
      upload[o] = rgbaPerPixel[0]!;
      upload[o + 1] = rgbaPerPixel[1]!;
      upload[o + 2] = rgbaPerPixel[2]!;
      upload[o + 3] = rgbaPerPixel[3]!;
    }
  }
  device.queue.writeTexture({ texture }, upload.buffer as GPUAllowSharedBufferSource, { bytesPerRow: bpr, rowsPerImage: height }, [
    width,
    height,
  ]);
}

function uploadRgbAsRgba32f(
  device: GPUDevice,
  texture: GPUTexture,
  rgb: Float32Array,
  width: number,
  height: number,
): void {
  const bpp = 16;
  const bpr = alignedTextureCopyBytesPerRow(width, bpp);
  const upload = new Float32Array((bpr / 4) * height);
  for (let y = 0; y < height; y += 1) {
    const row = (y * bpr) / 4;
    for (let x = 0; x < width; x += 1) {
      const si = (y * width + x) * 3;
      const o = row + x * 4;
      upload[o] = rgb[si] ?? 0;
      upload[o + 1] = rgb[si + 1] ?? 0;
      upload[o + 2] = rgb[si + 2] ?? 0;
      upload[o + 3] = 1;
    }
  }
  device.queue.writeTexture({ texture }, upload.buffer as GPUAllowSharedBufferSource, { bytesPerRow: bpr, rowsPerImage: height }, [
    width,
    height,
  ]);
}

function uploadRgbAsRgba16f(
  device: GPUDevice,
  texture: GPUTexture,
  rgb: Float32Array,
  width: number,
  height: number,
): void {
  const bpp = 8;
  const bpr = alignedTextureCopyBytesPerRow(width, bpp);
  const upload = new Uint8Array(bpr * height);
  const dv = new DataView(upload.buffer);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const si = (y * width + x) * 3;
      const byte = y * bpr + x * 8;
      dv.setUint16(byte + 0, float32ToFloat16Bits(rgb[si] ?? 0), true);
      dv.setUint16(byte + 2, float32ToFloat16Bits(rgb[si + 1] ?? 0), true);
      dv.setUint16(byte + 4, float32ToFloat16Bits(rgb[si + 2] ?? 0), true);
      dv.setUint16(byte + 6, float32ToFloat16Bits(1), true);
    }
  }
  device.queue.writeTexture({ texture }, upload.buffer as GPUAllowSharedBufferSource, { bytesPerRow: bpr, rowsPerImage: height }, [
    width,
    height,
  ]);
}

function fillRg32f(device: GPUDevice, texture: GPUTexture, width: number, height: number, r: number, g: number): void {
  const bpp = 8;
  const bpr = alignedTextureCopyBytesPerRow(width, bpp);
  const upload = new Float32Array((bpr / 4) * height);
  for (let y = 0; y < height; y += 1) {
    const row = (y * bpr) / 4;
    for (let x = 0; x < width; x += 1) {
      const o = row + x * 2;
      upload[o] = r;
      upload[o + 1] = g;
    }
  }
  device.queue.writeTexture({ texture }, upload.buffer as GPUAllowSharedBufferSource, { bytesPerRow: bpr, rowsPerImage: height }, [
    width,
    height,
  ]);
}

/** Linear depth → rgba32float texel `.r` (matches SVGF gbufferDepth sampling). */
function uploadLinearDepthAsRgba32f(
  device: GPUDevice,
  texture: GPUTexture,
  depth: Float32Array,
  width: number,
  height: number,
): void {
  const bpp = 16;
  const bpr = alignedTextureCopyBytesPerRow(width, bpp);
  const upload = new Float32Array((bpr / 4) * height);
  for (let y = 0; y < height; y += 1) {
    const row = (y * bpr) / 4;
    for (let x = 0; x < width; x += 1) {
      const si = y * width + x;
      const o = row + x * 4;
      upload[o] = depth[si] ?? 0;
      upload[o + 1] = 0;
      upload[o + 2] = 0;
      upload[o + 3] = 0;
    }
  }
  device.queue.writeTexture({ texture }, upload.buffer as GPUAllowSharedBufferSource, { bytesPerRow: bpr, rowsPerImage: height }, [
    width,
    height,
  ]);
}

/** Interleaved RG floats per pixel → rg32float texture (motion or Welford RG). */
function uploadInterleavedRgAsRg32f(
  device: GPUDevice,
  texture: GPUTexture,
  rg: Float32Array,
  width: number,
  height: number,
): void {
  const bpp = 8;
  const bpr = alignedTextureCopyBytesPerRow(width, bpp);
  const upload = new Float32Array((bpr / 4) * height);
  for (let y = 0; y < height; y += 1) {
    const row = (y * bpr) / 4;
    for (let x = 0; x < width; x += 1) {
      const si = (y * width + x) * 2;
      const o = row + x * 2;
      upload[o] = rg[si] ?? 0;
      upload[o + 1] = rg[si + 1] ?? 0;
    }
  }
  device.queue.writeTexture({ texture }, upload.buffer as GPUAllowSharedBufferSource, { bytesPerRow: bpr, rowsPerImage: height }, [
    width,
    height,
  ]);
}

/**
 * Validates sizes of optional SVGF G-buffer CPU slices.
 * Call from hosts before `runSvgfWebGPU` when assembling buffers manually.
 */
export function assertSvgfWebGPUBufferShapes(opts: SvgfWebGPUOptions): void {
  const w = opts.width;
  const h = opts.height;
  const px = w * h;
  if (w <= 0 || h <= 0 || opts.rgb.length < px * 3) {
    throw new Error('runSvgfWebGPU: invalid rgb buffer or dimensions');
  }
  const need = (cond: boolean, detail: string): void => {
    if (!cond) throw new Error(`runSvgfWebGPU: ${detail}`);
  };
  if (opts.prevRadianceRgb != null) {
    need(opts.prevRadianceRgb.length >= px * 3, 'prevRadianceRgb length must be >= width * height * 3');
  }
  if (opts.gbufferNormalsRgb != null) {
    need(opts.gbufferNormalsRgb.length >= px * 3, 'gbufferNormalsRgb length must be >= width * height * 3');
  }
  if (opts.linearDepth != null) {
    need(opts.linearDepth.length >= px, 'linearDepth length must be >= width * height');
  }
  if (opts.motionRg != null) {
    need(opts.motionRg.length >= px * 2, 'motionRg length must be >= width * height * 2');
  }
  if (opts.welfordMeanM2 != null) {
    need(opts.welfordMeanM2.length >= px * 2, 'welfordMeanM2 length must be >= width * height * 2');
  }
}

const TEMPORAL_VARIANCE_FRAME_THRESHOLD = 4;

function warnMissingWelfordTemporal(frameCount: number): void {
  if (frameCount < SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT) return;
  if (typeof console === 'undefined' || typeof console.warn !== 'function') return;
  console.warn(
    `[@vitrum/shared-denoisers] runSvgfWebGPU: frameCount >= ${SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT} but welfordMeanM2 was not supplied — temporal variance uses zeros. Provide RG32 mean+M₂ from the path accumulator when using temporal SVGF.`,
  );
}

function readRgba16fToRgbFloat(device: GPUDevice, texture: GPUTexture, width: number, height: number): Promise<Float32Array> {
  const bpp = 8;
  const bpr = alignedTextureCopyBytesPerRow(width, bpp);
  const buf = device.createBuffer({
    size: bpr * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow: bpr }, [width, height]);
  device.queue.submit([encoder.finish()]);
  return buf.mapAsync(GPUMapMode.READ).then(() => {
    const raw = new Uint8Array(buf.getMappedRange());
    const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const out = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const byte = y * bpr + x * 8;
        const di = (y * width + x) * 3;
        out[di] = float16BitsToFloat32(dv.getUint16(byte + 0, true));
        out[di + 1] = float16BitsToFloat32(dv.getUint16(byte + 2, true));
        out[di + 2] = float16BitsToFloat32(dv.getUint16(byte + 4, true));
      }
    }
    buf.unmap();
    buf.destroy();
    return out;
  });
}

export interface SvgfWebGPUOptions {
  readonly rgb: Float32Array;
  readonly width: number;
  readonly height: number;
  /** Frames since reset; below SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT selects spatial variance in shader. Default 0. */
  readonly frameCount?: number;
  /**
   * À-trous iterations (step 1, 2, 4, …). Default SVGF_DEFAULT_ATROUS_ITERATIONS.
   * Values are clamped to [1, SVGF_MAX_ATROUS_ITERATIONS].
   */
  readonly atrousIterations?: number;
  /** Explicit device; never destroyed by this call. */
  readonly device?: GPUDevice;
  /**
   * When true (default), uses getSharedWebGPUDevice. When false, one-shot device + destroy.
   */
  readonly reuseSharedWebGpuDevice?: boolean;

  /**
   * Previous-frame noisy HDR radiance (same layout as `rgb`). Defaults to mirroring `rgb` when omitted.
   * For motion-aware temporal filtering or TAA-style history, pass reprojected radiance here.
   */
  readonly prevRadianceRgb?: Float32Array;
  /** World-space (or view-space) unit normals: row-major RGB, length `width * height * 3`. */
  readonly gbufferNormalsRgb?: Float32Array;
  /** Linear depth per pixel; sampled as `.r` of gbuffer depth texture. Length `width * height`. */
  readonly linearDepth?: Float32Array;
  /** Screen-space motion vector per pixel (e.g. UV delta); RG interleaved, length `width * height * 2`. */
  readonly motionRg?: Float32Array;
  /**
   * Welford RG texel matching SVGF_WGSL / Sprint 9 Welford buffer: `.r = mean luminance`, `.g = M₂`.
   * Length `width * height * 2`. Supply when frameCount >= SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT for temporal variance.
   */
  readonly welfordMeanM2?: Float32Array;

  /** Overrides SVGF_SYNTHETIC_GBUFFER_DEFAULTS when real G-buffer slices are omitted. */
  readonly syntheticGbufferFallback?: SvgfSyntheticGbufferFallback;
  /** À-trous edge-stop σ; defaults from SVGF_DEFAULT_UNIFORMS. */
  readonly sigmaColor?: number;
  readonly sigmaNormal?: number;
  readonly sigmaDepth?: number;
}

/**
 * Runs SVGF variance estimation then ping-pong à-trous filtering.
 * Transient textures and buffers are freed per call; the GPU device is pooled by default.
 */
export async function runSvgfWebGPU(opts: SvgfWebGPUOptions): Promise<Float32Array> {
  const w = opts.width;
  const h = opts.height;
  const frameCount = opts.frameCount ?? 0;
  const rawAtrous = opts.atrousIterations ?? SVGF_DEFAULT_ATROUS_ITERATIONS;
  const atrousIterations = Math.min(SVGF_MAX_ATROUS_ITERATIONS, Math.max(1, Math.floor(rawAtrous)));
  const reuseShared = opts.reuseSharedWebGpuDevice !== false && opts.device == null;
  const sigmaColor = opts.sigmaColor ?? SVGF_DEFAULT_UNIFORMS.sigmaColor;
  const sigmaNormal = opts.sigmaNormal ?? SVGF_DEFAULT_UNIFORMS.sigmaNormal;
  const sigmaDepth = opts.sigmaDepth ?? SVGF_DEFAULT_UNIFORMS.sigmaDepth;
  const synNormal = opts.syntheticGbufferFallback?.normalRgb ?? SVGF_SYNTHETIC_GBUFFER_DEFAULTS.normalRgb;
  const synDepth = opts.syntheticGbufferFallback?.linearDepth ?? SVGF_SYNTHETIC_GBUFFER_DEFAULTS.linearDepth;
  assertSvgfWebGPUBufferShapes(opts);
  if (opts.welfordMeanM2 == null) {
    warnMissingWelfordTemporal(frameCount);
  }
  if (typeof navigator === 'undefined' || navigator.gpu == null) {
    throw new Error('runSvgfWebGPU: WebGPU not available');
  }

  let device: GPUDevice;
  let destroyEphemeral: (() => void) | null = null;
  if (opts.device != null) {
    device = opts.device;
  } else if (reuseShared) {
    device = await getSharedWebGPUDevice();
  } else {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter == null) {
      throw new Error('runSvgfWebGPU: failed to request GPU adapter');
    }
    device = await adapter.requestDevice();
    destroyEphemeral = () => {
      device.destroy();
    };
  }

  const { variance: variancePipeline, atrous: atrousPipeline } = svgfPipelines(device);

  const texRgba32Usage =
    GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT;

  const inputColor = device.createTexture({
    label: 'svgf-input-color',
    size: [w, h],
    format: 'rgba32float',
    usage: texRgba32Usage,
  });
  const prevRadiance = device.createTexture({
    label: 'svgf-prev',
    size: [w, h],
    format: 'rgba32float',
    usage: texRgba32Usage,
  });
  const gbufferNormal = device.createTexture({
    label: 'svgf-normal',
    size: [w, h],
    format: 'rgba32float',
    usage: texRgba32Usage,
  });
  const gbufferDepth = device.createTexture({
    label: 'svgf-depth',
    size: [w, h],
    format: 'rgba32float',
    usage: texRgba32Usage,
  });
  const motionVectors = device.createTexture({
    label: 'svgf-motion',
    size: [w, h],
    format: 'rg32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });
  const varianceIn = device.createTexture({
    label: 'svgf-welford-in',
    size: [w, h],
    format: 'rg32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });
  const varianceOut = device.createTexture({
    label: 'svgf-variance-out',
    size: [w, h],
    format: 'rg32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  const pingPongUsage =
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.COPY_DST |
    GPUTextureUsage.COPY_SRC;

  const colorPingA = device.createTexture({
    label: 'svgf-color-a',
    size: [w, h],
    format: 'rgba16float',
    usage: pingPongUsage,
  });
  const colorPingB = device.createTexture({
    label: 'svgf-color-b',
    size: [w, h],
    format: 'rgba16float',
    usage: pingPongUsage,
  });

  uploadRgbAsRgba32f(device, inputColor, opts.rgb, w, h);
  if (opts.prevRadianceRgb != null) {
    uploadRgbAsRgba32f(device, prevRadiance, opts.prevRadianceRgb, w, h);
  } else {
    uploadRgbAsRgba32f(device, prevRadiance, opts.rgb, w, h);
  }
  if (opts.gbufferNormalsRgb != null) {
    uploadRgbAsRgba32f(device, gbufferNormal, opts.gbufferNormalsRgb, w, h);
  } else {
    fillRgba32fTexture(device, gbufferNormal, w, h, [synNormal[0]!, synNormal[1]!, synNormal[2]!, 0]);
  }
  if (opts.linearDepth != null) {
    uploadLinearDepthAsRgba32f(device, gbufferDepth, opts.linearDepth, w, h);
  } else {
    fillRgba32fTexture(device, gbufferDepth, w, h, [synDepth, 0, 0, 0]);
  }
  if (opts.motionRg != null) {
    uploadInterleavedRgAsRg32f(device, motionVectors, opts.motionRg, w, h);
  } else {
    fillRg32f(device, motionVectors, w, h, 0, 0);
  }
  if (opts.welfordMeanM2 != null) {
    uploadInterleavedRgAsRg32f(device, varianceIn, opts.welfordMeanM2, w, h);
  } else {
    fillRg32f(device, varianceIn, w, h, 0, 0);
  }

  const varianceUbo = device.createBuffer({
    size: SVGF_VARIANCE_UNIFORMS_SIZE_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const varianceUboScratch = new ArrayBuffer(SVGF_VARIANCE_UNIFORMS_SIZE_BYTES);
  packSVGFVarianceUniforms({ frameCount }, varianceUboScratch);
  device.queue.writeBuffer(varianceUbo, 0, varianceUboScratch);

  const varianceBind = device.createBindGroup({
    layout: variancePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: inputColor.createView() },
      { binding: 1, resource: prevRadiance.createView() },
      { binding: 2, resource: gbufferNormal.createView() },
      { binding: 3, resource: gbufferDepth.createView() },
      { binding: 4, resource: motionVectors.createView() },
      { binding: 5, resource: varianceIn.createView() },
      { binding: 6, resource: varianceOut.createView() },
      { binding: 7, resource: { buffer: varianceUbo } },
    ],
  });

  const wg = SVGF_COMPUTE_WORKGROUP_SIZE;
  const encV = device.createCommandEncoder();
  const passV = encV.beginComputePass();
  passV.setPipeline(variancePipeline);
  passV.setBindGroup(0, varianceBind);
  passV.dispatchWorkgroups(Math.ceil(w / wg), Math.ceil(h / wg));
  passV.end();
  device.queue.submit([encV.finish()]);

  uploadRgbAsRgba16f(device, colorPingA, opts.rgb, w, h);

  const atrousUbo = device.createBuffer({
    label: 'svgf-atrous-ubo',
    size: SVGF_UNIFORMS_SIZE_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const atrousScratch = new ArrayBuffer(SVGF_UNIFORMS_SIZE_BYTES);

  let readTex = colorPingA;
  let writeTex = colorPingB;

  for (let iter = 0; iter < atrousIterations; iter += 1) {
    packSVGFUniforms(
      {
        iteration: iter,
        sigmaColor,
        sigmaNormal,
        sigmaDepth,
      },
      atrousScratch,
    );
    device.queue.writeBuffer(atrousUbo, 0, atrousScratch);

    const atrousBind = device.createBindGroup({
      layout: atrousPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: readTex.createView() },
        { binding: 1, resource: writeTex.createView() },
        { binding: 2, resource: gbufferNormal.createView() },
        { binding: 3, resource: gbufferDepth.createView() },
        { binding: 4, resource: varianceOut.createView() },
        { binding: 5, resource: { buffer: atrousUbo } },
      ],
    });

    const encA = device.createCommandEncoder();
    const passA = encA.beginComputePass();
    passA.setPipeline(atrousPipeline);
    passA.setBindGroup(0, atrousBind);
    passA.dispatchWorkgroups(Math.ceil(w / wg), Math.ceil(h / wg));
    passA.end();
    device.queue.submit([encA.finish()]);

    const nextRead = writeTex;
    writeTex = readTex;
    readTex = nextRead;
  }

  const finalTex = readTex;
  const rgbOut = await readRgba16fToRgbFloat(device, finalTex, w, h);

  inputColor.destroy();
  prevRadiance.destroy();
  gbufferNormal.destroy();
  gbufferDepth.destroy();
  motionVectors.destroy();
  varianceIn.destroy();
  varianceOut.destroy();
  colorPingA.destroy();
  colorPingB.destroy();
  varianceUbo.destroy();
  atrousUbo.destroy();
  destroyEphemeral?.();

  return rgbOut;
}
