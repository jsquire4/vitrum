/**
 * One-shot à-trous + variance denoiser on CPU-backed linear HDR RGB via WebGPU.
 *
 * à-trous + variance denoiser; not Schied SVGF — see svgfRealWebGPU.ts for real SVGF.
 *
 * When optional g-buffer slices (`gbufferNormalsRgb`, `linearDepth`, `motionRg`,
 * `welfordMeanM2`) are omitted, fills synthetic buffers from
 * ATROUS_VARIANCE_SYNTHETIC_GBUFFER_DEFAULTS unless `syntheticGbufferFallback` overrides them.
 *
 * For temporal variance (`frameCount >= ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT`), hosts should
 * supply `welfordMeanM2` from the path accumulator (RG mean + M₂). Cornell-style demos
 * may omit it and accept the console warning plus zero-filled variance input.
 */

import { ATROUS_VARIANCE_WGSL, ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE } from './wgsl/atrousVariance.wgsl.js';
import {
  ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS,
  ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
  ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES,
  packAtrousVarianceAtrousUniforms,
  packAtrousVarianceVarianceUniforms,
} from './atrousVarianceBindings.js';
import {
  ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS,
  ATROUS_VARIANCE_MAX_ATROUS_ITERATIONS,
  ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT,
} from './atrousVarianceConstants.js';
import { acquireDenoiseDevice, makePerDevicePipelineCache } from './sharedWebGpuDevice.js';
import { demodulateAlbedo, remodulateAlbedo } from './albedoModulation.js';
import {
  fillRg32f,
  fillRgba32f as fillRgba32fTexture,
  uploadInterleavedRgAsRg32f,
  uploadLinearDepthAsRgba32f,
  uploadRgbAsRgba16f,
  uploadRgbAsRgba32f,
  readRgba16fToRgb,
} from './webGpuTextureUpload.js';

export interface AtrousVarianceSyntheticGbufferFallback {
  readonly normalRgb?: readonly [number, number, number];
  readonly linearDepth?: number;
}

/** Defaults when optional G-buffer slices are omitted (Cornell-style synthetic prepass). */
export const ATROUS_VARIANCE_SYNTHETIC_GBUFFER_DEFAULTS = {
  normalRgb: [0, 1, 0] as const,
  linearDepth: 2,
} as const;

const VARIANCE_ENTRY = 'svgfVarianceMain';
const ATROUS_ENTRY = 'svgfAtrousMain';

interface AtrousVariancePipelineBundle {
  readonly variance: GPUComputePipeline;
  readonly atrous: GPUComputePipeline;
}

const atrousVariancePipelines = makePerDevicePipelineCache<AtrousVariancePipelineBundle>(
  (device) => {
    const shaderModule = device.createShaderModule({ label: 'atrous-variance', code: ATROUS_VARIANCE_WGSL });
    return {
      variance: device.createComputePipeline({
        label: 'atrous-variance-variance',
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: VARIANCE_ENTRY },
      }),
      atrous: device.createComputePipeline({
        label: 'atrous-variance-atrous',
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: ATROUS_ENTRY },
      }),
    };
  },
);

// Albedo demodulate / remodulate helpers (Schied 2017 §4.1) live in
// albedoModulation.ts — shared with the svgf-real host path.

/**
 * Validates sizes of optional à-trous + variance G-buffer CPU slices.
 * Call from hosts before `runAtrousVarianceWebGPU` when assembling buffers manually.
 */
export function assertAtrousVarianceWebGPUBufferShapes(opts: AtrousVarianceWebGPUOptions): void {
  const w = opts.width;
  const h = opts.height;
  const px = w * h;
  if (w <= 0 || h <= 0 || opts.rgb.length < px * 3) {
    throw new Error('runAtrousVarianceWebGPU: invalid rgb buffer or dimensions');
  }
  const need = (cond: boolean, detail: string): void => {
    if (!cond) throw new Error(`runAtrousVarianceWebGPU: ${detail}`);
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
  if (opts.albedoRgb != null) {
    need(opts.albedoRgb.length >= px * 3, 'albedoRgb length must be >= width * height * 3');
  }
}

function warnMissingWelfordTemporal(frameCount: number): void {
  if (frameCount < ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT) return;
  if (typeof console === 'undefined' || typeof console.warn !== 'function') return;
  console.warn(
    `[@vitrum/shared-denoisers] runAtrousVarianceWebGPU: frameCount >= ${ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT} but welfordMeanM2 was not supplied — temporal variance uses zeros. Provide RG32 mean+M₂ from the path accumulator when using temporal à-trous variance denoiser.`,
  );
}

export interface AtrousVarianceWebGPUOptions {
  readonly rgb: Float32Array;
  readonly width: number;
  readonly height: number;
  /** Frames since reset; below ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT selects spatial variance in shader. Default 0. */
  readonly frameCount?: number;
  /**
   * À-trous iterations (step 1, 2, 4, …). Default ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS.
   * Values are clamped to [1, ATROUS_VARIANCE_MAX_ATROUS_ITERATIONS].
   */
  readonly atrousIterations?: number;
  /** Explicit device; never destroyed by this call. */
  readonly device?: GPUDevice;
  /**
   * When true, uses getSharedWebGPUDevice. Default: false.
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
   * Welford RG texel matching ATROUS_VARIANCE_WGSL / Sprint 9 Welford buffer: `.r = mean luminance`, `.g = M₂`.
   * Length `width * height * 2`. Supply when frameCount >= ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT for temporal variance.
   */
  readonly welfordMeanM2?: Float32Array;

  /**
   * Per-pixel diffuse albedo (row-major RGB, length `width * height * 3`).
   *
   * When supplied, enables albedo demodulation per Schied 2017 §4.1:
   *   1. Before filtering: `lighting = rgb / max(albedo, 1e-3)` (per channel).
   *   2. À-trous chain filters the pure lighting signal.
   *   3. After filtering: `output = filtered_lighting × albedo`.
   *
   * Without this, albedo-correlated high-frequency variation (texture
   * boundaries, material edges) bleeds into the lighting during spatial
   * filtering and produces blurry material boundaries.
   */
  readonly albedoRgb?: Float32Array;

  /** Overrides ATROUS_VARIANCE_SYNTHETIC_GBUFFER_DEFAULTS when real G-buffer slices are omitted. */
  readonly syntheticGbufferFallback?: AtrousVarianceSyntheticGbufferFallback;
  /** À-trous edge-stop σ; defaults from ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS. */
  readonly sigmaColor?: number;
  readonly sigmaNormal?: number;
  readonly sigmaDepth?: number;
}

/**
 * Runs à-trous variance estimation then ping-pong à-trous filtering.
 * Transient textures and buffers are freed per call; the GPU device is pooled by default.
 */
export async function runAtrousVarianceWebGPU(opts: AtrousVarianceWebGPUOptions): Promise<Float32Array> {
  const w = opts.width;
  const h = opts.height;
  const frameCount = opts.frameCount ?? 0;
  const rawAtrous = opts.atrousIterations ?? ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS;
  const atrousIterations = Math.min(ATROUS_VARIANCE_MAX_ATROUS_ITERATIONS, Math.max(1, Math.floor(rawAtrous)));
  const sigmaColor = opts.sigmaColor ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaColor;
  const sigmaNormal = opts.sigmaNormal ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaNormal;
  const sigmaDepth = opts.sigmaDepth ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaDepth;
  const synNormal = opts.syntheticGbufferFallback?.normalRgb ?? ATROUS_VARIANCE_SYNTHETIC_GBUFFER_DEFAULTS.normalRgb;
  const synDepth = opts.syntheticGbufferFallback?.linearDepth ?? ATROUS_VARIANCE_SYNTHETIC_GBUFFER_DEFAULTS.linearDepth;
  assertAtrousVarianceWebGPUBufferShapes(opts);
  if (opts.welfordMeanM2 == null) {
    warnMissingWelfordTemporal(frameCount);
  }

  const { device, dispose: destroyEphemeral } = await acquireDenoiseDevice({
    device: opts.device,
    reuseSharedWebGpuDevice: opts.reuseSharedWebGpuDevice,
    errorLabel: 'runAtrousVarianceWebGPU',
  });

  const { variance: variancePipeline, atrous: atrousPipeline } = atrousVariancePipelines(device);

  // G-buffer inputs are written via writeTexture and read as texture_2d<f32>
  // inside the compute shaders. They are never used as render attachments,
  // so RENDER_ATTACHMENT is omitted. COPY_SRC stays so test / debug readbacks
  // remain possible without re-allocating with new flags.
  const texRgba32Usage =
    GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;

  const inputColor = device.createTexture({
    label: 'atrous-variance-input-color',
    size: [w, h],
    format: 'rgba32float',
    usage: texRgba32Usage,
  });
  const prevRadiance = device.createTexture({
    label: 'atrous-variance-prev',
    size: [w, h],
    format: 'rgba32float',
    usage: texRgba32Usage,
  });
  const gbufferNormal = device.createTexture({
    label: 'atrous-variance-normal',
    size: [w, h],
    format: 'rgba32float',
    usage: texRgba32Usage,
  });
  const gbufferDepth = device.createTexture({
    label: 'atrous-variance-depth',
    size: [w, h],
    format: 'rgba32float',
    usage: texRgba32Usage,
  });
  const motionVectors = device.createTexture({
    label: 'atrous-variance-motion',
    size: [w, h],
    format: 'rg32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });
  const varianceIn = device.createTexture({
    label: 'atrous-variance-welford-in',
    size: [w, h],
    format: 'rg32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });
  const varianceOut = device.createTexture({
    label: 'atrous-variance-variance-out',
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
    label: 'atrous-variance-color-a',
    size: [w, h],
    format: 'rgba16float',
    usage: pingPongUsage,
  });
  const colorPingB = device.createTexture({
    label: 'atrous-variance-color-b',
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
    fillRgba32fTexture(device, gbufferNormal, w, h, [synNormal[0], synNormal[1], synNormal[2], 0]);
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
    size: ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const varianceUboScratch = new ArrayBuffer(ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES);
  packAtrousVarianceVarianceUniforms({ frameCount }, varianceUboScratch);
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

  // Pre-allocate one UBO per à-trous iteration so each pass reads its own
  // uniforms. With a shared UBO + per-iter writeBuffer, the driver can re-
  // order writes vs dispatches and produce wrong stepWidth per pass; per-iter
  // UBOs are tiny (32 B × N) and remove that hazard. Write all UBOs once,
  // then batch every pass (variance + N × atrous) into a single encoder /
  // single queue.submit — replaces what used to be up to 13 separate submits.
  const atrousUbos: GPUBuffer[] = [];
  const atrousScratch = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
  for (let iter = 0; iter < atrousIterations; iter += 1) {
    const ubo = device.createBuffer({
      label: `atrous-variance-atrous-ubo-${iter}`,
      size: ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    packAtrousVarianceAtrousUniforms(
      { iteration: iter, sigmaColor, sigmaNormal, sigmaDepth },
      atrousScratch,
    );
    device.queue.writeBuffer(ubo, 0, atrousScratch);
    atrousUbos.push(ubo);
  }

  // Item 24 — albedo demodulation (Schied 2017 §4.1).
  // When albedoRgb is supplied, divide the input rgb by albedo before
  // uploading to the GPU so the à-trous chain filters pure lighting.
  // The demodulated buffer is used ONLY for colorPingA (the à-trous input);
  // inputColor (the variance pass input) receives the original rgb so the
  // variance estimate reflects the actual noisy signal energy.
  const rgbForAtrous = opts.albedoRgb != null
    ? demodulateAlbedo(opts.rgb, opts.albedoRgb, w * h)
    : opts.rgb;
  uploadRgbAsRgba16f(device, colorPingA, rgbForAtrous, w, h);

  // Build the alternating bind groups up front: A→B for even iterations,
  // B→A for odd. Each pair is paired with its iteration's UBO.
  const atrousBindGroups: GPUBindGroup[] = atrousUbos.map((ubo, iter) => {
    const isEven = iter % 2 === 0;
    return device.createBindGroup({
      layout: atrousPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: (isEven ? colorPingA : colorPingB).createView() },
        { binding: 1, resource: (isEven ? colorPingB : colorPingA).createView() },
        { binding: 2, resource: gbufferNormal.createView() },
        { binding: 3, resource: gbufferDepth.createView() },
        { binding: 4, resource: varianceOut.createView() },
        { binding: 5, resource: { buffer: ubo } },
      ],
    });
  });

  const wg = ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE;
  const encoder = device.createCommandEncoder({ label: 'atrous-variance-batched' });

  const passV = encoder.beginComputePass();
  passV.setPipeline(variancePipeline);
  passV.setBindGroup(0, varianceBind);
  passV.dispatchWorkgroups(Math.ceil(w / wg), Math.ceil(h / wg));
  passV.end();

  for (let iter = 0; iter < atrousIterations; iter += 1) {
    const passA = encoder.beginComputePass();
    passA.setPipeline(atrousPipeline);
    passA.setBindGroup(0, atrousBindGroups[iter]);
    passA.dispatchWorkgroups(Math.ceil(w / wg), Math.ceil(h / wg));
    passA.end();
  }

  device.queue.submit([encoder.finish()]);

  // After N iterations, the last write went into colorPingB when N is odd,
  // colorPingA when N is even (the loop's pre-fix swap convention).
  const readTex = atrousIterations % 2 === 0 ? colorPingA : colorPingB;

  const finalTex = readTex;
  const rgbOut = await readRgba16fToRgb(device, finalTex, w, h);

  // Item 24 — albedo re-modulation: multiply the filtered lighting by albedo
  // to restore the correct denoised outgoing radiance.
  if (opts.albedoRgb != null) {
    remodulateAlbedo(rgbOut, opts.albedoRgb, w * h);
  }

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
  for (const ubo of atrousUbos) ubo.destroy();
  destroyEphemeral();

  return rgbOut;
}
