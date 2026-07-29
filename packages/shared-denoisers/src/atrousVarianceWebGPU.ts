/**
 * One-shot à-trous + variance denoiser on CPU-backed linear HDR RGB via WebGPU.
 *
 * à-trous + variance denoiser; not Schied SVGF — see svgfRealWebGPU.ts for real SVGF.
 *
 * When optional g-buffer slices (`gbufferNormalsRgb`, `linearDepth`,
 * `welfordMeanM2`) are omitted, fills synthetic buffers from
 * ATROUS_VARIANCE_SYNTHETIC_GBUFFER_DEFAULTS unless `syntheticGbufferFallback` overrides them.
 *
 * For temporal variance (`frameCount >= ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT`), hosts should
 * supply `welfordMeanM2` from the path accumulator (RG mean + M₂). Cornell-style demos
 * may omit it and accept the console warning plus zero-filled variance input.
 */

import {
  ATROUS_VARIANCE_WGSL,
  ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE,
} from './wgsl/atrousVariance.wgsl.js';
import {
  ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS,
  ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES,
  packAtrousVarianceVarianceUniforms,
} from './atrousVarianceBindings.js';
import {
  ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS,
  ATROUS_VARIANCE_MAX_ATROUS_ITERATIONS,
  ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT,
} from './atrousVarianceConstants.js';
import { acquireDenoiseDevice, makePerDevicePipelineCache } from './sharedWebGpuDevice.js';
import { buildAtrousChain, makeResourceTracker } from './atrousChain.js';
import {
  assertFiniteFloatSlice,
  assertFiniteNumber,
  assertOneShotArrayLength,
  assertOneShotDeviceLimits,
  assertOneShotDimensions,
} from './webGpuOneShotValidation.js';
// NOTE: albedo demodulation differs INTENTIONALLY between this à-trous+variance
// denoiser and svgfRealWebGPU.ts — see the demodulation cross-reference comment
// at the `rgbForAtrous` site below (`demodulateAlbedo`/`remodulateAlbedo`).
import { demodulateAlbedo, remodulateAlbedo } from './albedoModulation.js';
import {
  fillRg32f,
  fillRgba32f as fillRgba32fTexture,
  uploadInterleavedRgAsRg32f,
  uploadLinearDepthAsRgba32f,
  uploadRgbAsRgba16f,
  uploadRgbAsRgba32f,
  uploadUnitNormalsAsRgba32f,
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
    const shaderModule = device.createShaderModule({
      label: 'atrous-variance',
      code: ATROUS_VARIANCE_WGSL,
    });
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
  const label = 'runAtrousVarianceWebGPU';
  const px = assertOneShotDimensions(label, w, h);
  const check = (name: string, value: Float32Array, length: number): void => {
    assertOneShotArrayLength(label, name, value, length);
    assertFiniteFloatSlice(label, name, value, length);
  };
  check('rgb', opts.rgb, px * 3);
  if (opts.gbufferNormalsRgb != null) {
    check('gbufferNormalsRgb', opts.gbufferNormalsRgb, px * 3);
    for (let index = 0; index < opts.gbufferNormalsRgb.length; index += 1) {
      const component = opts.gbufferNormalsRgb[index]!;
      if (component < -1 || component > 1) {
        throw new RangeError(
          `${label}: gbufferNormalsRgb[${index}] must be in signed normal range [-1, 1] ` +
          `(got ${String(component)})`,
        );
      }
    }
  }
  if (opts.linearDepth != null) {
    check('linearDepth', opts.linearDepth, px);
  }
  if (opts.welfordMeanM2 != null) {
    check('welfordMeanM2', opts.welfordMeanM2, px * 2);
  }
  if (opts.albedoRgb != null) {
    check('albedoRgb', opts.albedoRgb, px * 3);
  }

  assertFiniteNumber(label, 'frameCount', opts.frameCount ?? 0, {
    integer: true,
    min: 0,
    max: 0xFFFFFFFF,
  });
  assertFiniteNumber(
    label,
    'atrousIterations',
    opts.atrousIterations ?? ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS,
  );
  assertFiniteNumber(label, 'sigmaColor', opts.sigmaColor ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaColor, { min: 0 });
  assertFiniteNumber(label, 'sigmaNormal', opts.sigmaNormal ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaNormal, { min: 0 });
  assertFiniteNumber(label, 'sigmaDepth', opts.sigmaDepth ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaDepth, { min: 0 });

  const fallback = opts.syntheticGbufferFallback;
  if (fallback?.normalRgb != null) {
    assertFiniteFloatSlice(label, 'syntheticGbufferFallback.normalRgb', fallback.normalRgb, 3);
    for (let index = 0; index < fallback.normalRgb.length; index += 1) {
      const component = fallback.normalRgb[index]!;
      if (component < -1 || component > 1) {
        throw new RangeError(
          `${label}: syntheticGbufferFallback.normalRgb[${index}] must be in signed ` +
          `normal range [-1, 1] (got ${String(component)})`,
        );
      }
    }
  }
  if (fallback?.linearDepth != null) {
    assertFiniteNumber(label, 'syntheticGbufferFallback.linearDepth', fallback.linearDepth);
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

  /** World-space (or view-space) unit normals: row-major RGB, length `width * height * 3`. */
  readonly gbufferNormalsRgb?: Float32Array;
  /** Linear depth per pixel; sampled as `.r` of gbuffer depth texture. Length `width * height`. */
  readonly linearDepth?: Float32Array;
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
export async function runAtrousVarianceWebGPU(
  opts: AtrousVarianceWebGPUOptions,
): Promise<Float32Array> {
  const w = opts.width;
  const h = opts.height;
  const frameCount = opts.frameCount ?? 0;
  const rawAtrous = opts.atrousIterations ?? ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS;
  const atrousIterations = Math.min(
    ATROUS_VARIANCE_MAX_ATROUS_ITERATIONS,
    Math.max(1, Math.floor(rawAtrous)),
  );
  const sigmaColor = opts.sigmaColor ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaColor;
  const sigmaNormal = opts.sigmaNormal ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaNormal;
  const sigmaDepth = opts.sigmaDepth ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaDepth;
  const synNormal =
    opts.syntheticGbufferFallback?.normalRgb ??
    ATROUS_VARIANCE_SYNTHETIC_GBUFFER_DEFAULTS.normalRgb;
  const synDepth =
    opts.syntheticGbufferFallback?.linearDepth ??
    ATROUS_VARIANCE_SYNTHETIC_GBUFFER_DEFAULTS.linearDepth;
  assertAtrousVarianceWebGPUBufferShapes(opts);
  if (opts.welfordMeanM2 == null) {
    warnMissingWelfordTemporal(frameCount);
  }

  const { device, dispose: destroyEphemeral } = await acquireDenoiseDevice({
    device: opts.device,
    reuseSharedWebGpuDevice: opts.reuseSharedWebGpuDevice,
    errorLabel: 'runAtrousVarianceWebGPU',
  });

  const { trackTexture, trackBuffer, dispose: disposeResources } =
    makeResourceTracker(destroyEphemeral);

  try {
    assertOneShotDeviceLimits(device, 'runAtrousVarianceWebGPU', w, h, 8);
    const { variance: variancePipeline, atrous: atrousPipeline } = atrousVariancePipelines(device);

    // G-buffer inputs are written via writeTexture and read as texture_2d<f32>
    // inside the compute shaders. They are never used as render attachments,
    // so RENDER_ATTACHMENT is omitted. COPY_SRC stays so test / debug readbacks
    // remain possible without re-allocating with new flags.
    const texRgba32Usage =
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;

    const inputColor = trackTexture(
      device.createTexture({
        label: 'atrous-variance-input-color',
        size: [w, h],
        format: 'rgba32float',
        usage: texRgba32Usage,
      }),
    );
    const gbufferNormal = trackTexture(
      device.createTexture({
        label: 'atrous-variance-normal',
        size: [w, h],
        format: 'rgba32float',
        usage: texRgba32Usage,
      }),
    );
    const gbufferDepth = trackTexture(
      device.createTexture({
        label: 'atrous-variance-depth',
        size: [w, h],
        format: 'rgba32float',
        usage: texRgba32Usage,
      }),
    );
    const varianceIn = trackTexture(
      device.createTexture({
        label: 'atrous-variance-welford-in',
        size: [w, h],
        format: 'rg32float',
        usage:
          GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
      }),
    );
    const varianceOut = trackTexture(
      device.createTexture({
        label: 'atrous-variance-variance-out',
        size: [w, h],
        format: 'rgba32float',
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
      }),
    );

    const pingPongUsage =
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.COPY_SRC;

    const colorPingA = trackTexture(
      device.createTexture({
        label: 'atrous-variance-color-a',
        size: [w, h],
        format: 'rgba16float',
        usage: pingPongUsage,
      }),
    );
    const colorPingB = trackTexture(
      device.createTexture({
        label: 'atrous-variance-color-b',
        size: [w, h],
        format: 'rgba16float',
        usage: pingPongUsage,
      }),
    );

    uploadRgbAsRgba32f(device, inputColor, opts.rgb, w, h);
    if (opts.gbufferNormalsRgb != null) {
      uploadUnitNormalsAsRgba32f(device, gbufferNormal, opts.gbufferNormalsRgb, w, h);
    } else {
      fillRgba32fTexture(device, gbufferNormal, w, h, [
        synNormal[0] * 0.5 + 0.5,
        synNormal[1] * 0.5 + 0.5,
        synNormal[2] * 0.5 + 0.5,
        0,
      ]);
    }
    if (opts.linearDepth != null) {
      uploadLinearDepthAsRgba32f(device, gbufferDepth, opts.linearDepth, w, h);
    } else {
      fillRgba32fTexture(device, gbufferDepth, w, h, [synDepth, 0, 0, 0]);
    }
    if (opts.welfordMeanM2 != null) {
      uploadInterleavedRgAsRg32f(device, varianceIn, opts.welfordMeanM2, w, h);
    } else {
      fillRg32f(device, varianceIn, w, h, 0, 0);
    }

    const varianceUbo = trackBuffer(
      device.createBuffer({
        size: ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    );
    const varianceUboScratch = new ArrayBuffer(ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES);
    packAtrousVarianceVarianceUniforms({ frameCount }, varianceUboScratch);
    device.queue.writeBuffer(varianceUbo, 0, varianceUboScratch);

    const varianceBind = device.createBindGroup({
      layout: variancePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: inputColor.createView() },
        { binding: 1, resource: varianceIn.createView() },
        { binding: 2, resource: varianceOut.createView() },
        { binding: 3, resource: { buffer: varianceUbo } },
      ],
    });

    // Item 24 — albedo demodulation (Schied 2017 §4.1).
    // When albedoRgb is supplied, divide the input rgb by albedo before
    // uploading to the GPU so the à-trous chain filters pure lighting.
    // The demodulated buffer is used ONLY for colorPingA (the à-trous input);
    // inputColor (the variance pass input) receives the original rgb so the
    // variance estimate reflects the actual noisy signal energy.
    //
    // ── Demodulation cross-reference (D14-3) ─────────────────────────────────
    // This is DELIBERATELY narrower than svgfRealWebGPU.ts's demodulation. Here
    // ONLY the à-trous input (colorPingA) sees demodulated lighting; the
    // variance pass reads the ORIGINAL noisy `rgb` so the variance estimate
    // reflects true signal energy. In svgfRealWebGPU.ts BOTH current AND
    // previous radiance are demodulated up front so reprojection blending,
    // moment tracking, and variance all operate on lighting L = c/ρ (that path
    // has real temporal moments; this à-trous+variance denoiser does not).
    // The divergence is intentional — do NOT unify these two demodulation
    // strategies; the algorithms differ (Schied SVGF vs à-trous lookup).
    const rgbForAtrous =
      opts.albedoRgb != null ? demodulateAlbedo(opts.rgb, opts.albedoRgb, w * h) : opts.rgb;
    uploadRgbAsRgba16f(device, colorPingA, rgbForAtrous, w, h);

    const wg = ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE;
    // Batch every pass (variance + N × atrous) into a single encoder /
    // single queue.submit — replaces what used to be up to 13 separate submits.
    const encoder = device.createCommandEncoder({ label: 'atrous-variance-batched' });

    const passV = encoder.beginComputePass();
    passV.setPipeline(variancePipeline);
    passV.setBindGroup(0, varianceBind);
    passV.dispatchWorkgroups(Math.ceil(w / wg), Math.ceil(h / wg));
    passV.end();

    // Shared à-trous ping-pong chain (atrousChain.ts) — per-iter UBOs +
    // alternating bind groups + parity-based readTex. Single source of truth
    // with svgfRealWebGPU.ts.
    const readTex = buildAtrousChain({
      device,
      atrousPipeline,
      encoder,
      atrousIterations,
      width: w,
      height: h,
      sigmaColor,
      sigmaNormal,
      sigmaDepth,
      pingTex: colorPingA,
      pongTex: colorPingB,
      normalView: gbufferNormal.createView(),
      depthView: gbufferDepth.createView(),
      varianceView: varianceOut.createView(),
      uboLabelPrefix: 'atrous-variance-atrous-ubo-',
      trackBuffer,
    });

    device.queue.submit([encoder.finish()]);

    const rgbOut = await readRgba16fToRgb(device, readTex, w, h);

    // Item 24 — albedo re-modulation: multiply the filtered lighting by albedo
    // to restore the correct denoised outgoing radiance.
    if (opts.albedoRgb != null) {
      remodulateAlbedo(rgbOut, opts.albedoRgb, w * h);
    }

    return rgbOut;
  } finally {
    disposeResources();
  }
}
