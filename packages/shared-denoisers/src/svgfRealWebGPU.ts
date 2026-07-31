/**
 * svgfRealWebGPU.ts — One-shot host pipeline for the real Schied 2017 SVGF denoiser.
 *
 * Runs the full SVGF chain on CPU-backed linear HDR RGB via WebGPU:
 *   1. svgfReprojMain           — bilinear reprojection + disocclusion + EMA history
 *   2. svgfVarianceFromMomentsMain — Eq. 5 variance from blended moments
 *   3. svgf7x7FallbackMain      — 7×7 spatial fallback for history < 4 pixels
 *   4. svgfRealAtrousMain (×5)  — variance-prefiltered, variance-propagating à-trous chain
 *
 * GPU memory budget (see svgfRealConstants.ts for the compact input side):
 *   persistent inputs remain compact where portable; storage-write outputs use
 *   rgba32float where WebGPU storage-format support requires it.
 *
 * This one-shot path allocates and destroys all textures per call. In the
 * WalkaroundGPUPipeline (persistent mode), textures are allocated once in
 * createFrameResources() and referenced across frames.
 *
 * This is the real Schied 2017 SVGF; the à-trous + variance lookup denoiser
 * in atrousVariance*.ts is a different (non-Schied) filter.
 *
 * References:
 *   Schied et al. "Spatiotemporal Variance-Guided Filtering" HPG 2017.
 */

import { SVGF_REAL_REPROJECTION_WORKGROUP_SIZE } from './wgsl/svgfReprojection.wgsl.js';
import { SVGF_VARIANCE_FROM_MOMENTS_WORKGROUP_SIZE } from './wgsl/svgfVarianceFromMoments.wgsl.js';
import { SVGF_7X7_SPATIAL_FALLBACK_WORKGROUP_SIZE } from './wgsl/svgf7x7SpatialFallback.wgsl.js';
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
import { ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS } from './atrousVarianceBindings.js';
import { acquireDenoiseDevice } from './sharedWebGpuDevice.js';
import { buildAtrousChain, makeResourceTracker } from './atrousChain.js';
import {
  assertFiniteFloat16Slice,
  assertFiniteFloatSlice,
  assertFiniteNumber,
  assertOneShotArrayLength,
  assertOneShotDeviceLimits,
  assertOneShotDimensions,
} from './webGpuOneShotValidation.js';
// NOTE: albedo demodulation differs INTENTIONALLY between this real-SVGF path
// and atrousVarianceWebGPU.ts — see the demodulation cross-reference comment at
// the `rgbForChain`/`prevForChain` site below.
import { demodulateAlbedo, remodulateAlbedo } from './albedoModulation.js';
import {
  uploadRgbAsRgba16f,
  uploadRgbAsRgba32fPacked,
  uploadInterleavedRgAsRg32f,
  uploadR32f,
  uploadR32Uint,
  uploadR16Uint,
  fillR16Uint,
  fillRg32f,
  fillRgba32f,
  readRgba16fToRgb,
  readRgba32fToRg,
  readR32UintToU32,
} from './webGpuTextureUpload.js';

// ============================================================
// Albedo demodulation helpers — Schied 2017 §4.1.
// ============================================================
//
// The implementations live in albedoModulation.ts (shared with the
// atrous-variance host path). These exports preserve the historic
// svgfReal*-prefixed public names for backward-compatible test / host imports.

/** Divide rgb by albedo; returns a new Float32Array of the demodulated signal. */
export const svgfRealDemodulateAlbedo = demodulateAlbedo;

/** Multiply rgb by albedo in-place; returns the same Float32Array. */
export const svgfRealRemodulateAlbedo = remodulateAlbedo;

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
  readonly width: number;
  readonly height: number;

  /**
   * Previous-frame first-wavelet color. If omitted, mirrors `rgb` (first
   * frame). A chainable call returns the matching value as
   * `prevRadianceOut`.
   */
  readonly prevRadianceRgb?: Float32Array;
  /**
   * Previous-frame albedo corresponding to `prevRadianceRgb`. Required when
   * both `albedoRgb` and `prevRadianceRgb` are supplied so previous radiance is
   * demodulated in its own material domain rather than the current frame's.
   */
  readonly prevAlbedoRgb?: Float32Array;
  /** Screen-space motion vector (pixel delta, RG interleaved), length W*H*2. */
  readonly motionRg?: Float32Array;
  /** Linear depth per pixel, length W*H. Defaults to constant 1.0. */
  readonly linearDepth?: Float32Array;
  /** World-space normals (packed 0..1 XYZ), length W*H*3. Defaults to forward-facing. */
  readonly gbufferNormalsRgb?: Float32Array;
  /** Per-pixel object ID (u32), length W*H. Defaults to 0. */
  readonly objectIds?: Uint32Array;

  // ── Previous-frame G-buffer (enables multi-frame chaining) ────────────────
  // When a prev* input is supplied it is uploaded to the corresponding prev
  // texture used by reprojection disocclusion. When absent, the one-shot
  // fallback mirrors the current-frame value (prev == curr) as before — so a
  // caller that omits these keeps the historic single-frame behavior exactly.

  /** Previous-frame linear depth, length W*H. Falls back to `linearDepth` (curr) when absent. */
  readonly prevLinearDepth?: Float32Array;
  /** Previous-frame packed normals (0..1 XYZ), length W*H*3. Falls back to `gbufferNormalsRgb` (curr) when absent. */
  readonly prevNormalsRgb?: Float32Array;
  /** Previous-frame object IDs (u32), length W*H. Falls back to `objectIds` (curr) when absent. */
  readonly prevObjectIds?: Uint32Array;
  /** Previous-frame per-pixel history length (u32), length W*H. Alias of `historyLengthIn`; when both set, this wins for the prev texture. */
  readonly prevHistoryLength?: Uint32Array;

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
  readonly sigmaColor?: number;
  readonly sigmaNormal?: number;
  readonly sigmaDepth?: number;

  /** Explicit GPU device (never destroyed by this call). */
  readonly device?: GPUDevice;
  /** When true, uses process-shared WebGPU device; otherwise uses ephemeral device. Default: false. */
  readonly reuseSharedWebGpuDevice?: boolean;

  /**
   * When true, the first-wavelet color history plus reprojection outputs
   * (blended moments + per-pixel history length) are read back BEFORE the
   * transient textures are destroyed and returned as `prevRadianceOut`,
   * `momentsOut`, and `historyLengthOut`. Feed them back as
   * `prevRadianceRgb`, `momentsIn`, and `historyLengthIn` next frame. When
   * albedo demodulation is enabled, also feed the current `albedoRgb` back as
   * the next call's `prevAlbedoRgb`.
   * Default false (single-frame; chaining outputs remain undefined and no
   * extra readback is submitted).
   */
  readonly chainable?: boolean;
}

/**
 * Result of a one-shot SVGF pass.
 *
 * `rgb` is always the filtered HDR RGB (length width*height*3). When the caller
 * sets `chainable: true`, `prevRadianceOut` (the first-wavelet color history),
 * `momentsOut` (blended M1/M2 interleaved), and `historyLengthOut` (per-pixel
 * u32 history length) are also returned so they can be threaded into the next
 * frame; otherwise all three are undefined.
 */
export interface SVGFRealWebGPUResult {
  readonly rgb: Float32Array;
  readonly prevRadianceOut?: Float32Array;
  readonly momentsOut?: Float32Array;
  readonly historyLengthOut?: Uint32Array;
}

/** Validate all CPU-backed inputs before device acquisition or GPU allocation. */
export function assertSVGFRealWebGPUInputs(opts: SVGFRealWebGPUOptions): number {
  const label = 'runSVGFRealWebGPU';
  const px = assertOneShotDimensions(label, opts.width, opts.height);
  const checkFloat = (name: string, value: Float32Array, length: number): void => {
    assertOneShotArrayLength(label, name, value, length);
    assertFiniteFloatSlice(label, name, value, length);
  };
  const checkU32 = (
    name: string,
    value: Uint32Array,
    length: number,
    max: number = 0xFFFFFFFF,
  ): void => {
    assertOneShotArrayLength(label, name, value, length);
    if (max < 0xFFFFFFFF) {
      for (let i = 0; i < length; i += 1) {
        if (value[i]! > max) {
          throw new Error(
            `${label}: ${name}[${i}] must be <= ${max}; received ${value[i]}`,
          );
        }
      }
    }
  };

  checkFloat('rgb', opts.rgb, px * 3);
  if (opts.prevRadianceRgb != null) {
    checkFloat('prevRadianceRgb', opts.prevRadianceRgb, px * 3);
  }
  if (opts.prevAlbedoRgb != null) {
    checkFloat('prevAlbedoRgb', opts.prevAlbedoRgb, px * 3);
  }
  if (opts.motionRg != null) checkFloat('motionRg', opts.motionRg, px * 2);
  if (opts.linearDepth != null) checkFloat('linearDepth', opts.linearDepth, px);
  if (opts.gbufferNormalsRgb != null) {
    checkFloat('gbufferNormalsRgb', opts.gbufferNormalsRgb, px * 3);
  }
  if (opts.prevLinearDepth != null) {
    checkFloat('prevLinearDepth', opts.prevLinearDepth, px);
  }
  if (opts.prevNormalsRgb != null) {
    checkFloat('prevNormalsRgb', opts.prevNormalsRgb, px * 3);
  }
  if (opts.albedoRgb != null) checkFloat('albedoRgb', opts.albedoRgb, px * 3);
  if (
    opts.albedoRgb != null &&
    opts.prevRadianceRgb != null &&
    opts.prevAlbedoRgb == null
  ) {
    throw new Error(
      `${label}: prevAlbedoRgb is required when albedoRgb and prevRadianceRgb are supplied`,
    );
  }
  if (opts.prevAlbedoRgb != null && opts.albedoRgb == null) {
    throw new Error(`${label}: prevAlbedoRgb requires albedoRgb`);
  }
  if (opts.prevAlbedoRgb != null && opts.prevRadianceRgb == null) {
    throw new Error(`${label}: prevAlbedoRgb requires prevRadianceRgb`);
  }
  if (opts.momentsIn != null) checkFloat('momentsIn', opts.momentsIn, px * 2);

  if (opts.objectIds != null) checkU32('objectIds', opts.objectIds, px);
  if (opts.prevObjectIds != null) checkU32('prevObjectIds', opts.prevObjectIds, px);
  if (opts.historyLengthIn != null) {
    checkU32('historyLengthIn', opts.historyLengthIn, px, 0xFFFF);
  }
  if (opts.prevHistoryLength != null) {
    checkU32('prevHistoryLength', opts.prevHistoryLength, px, 0xFFFF);
  }

  assertFiniteNumber(
    label,
    'atrousIterations',
    opts.atrousIterations ?? SVGF_REAL_DEFAULT_ATROUS_ITERATIONS,
  );
  assertFiniteNumber(label, 'sigmaColor', opts.sigmaColor ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaColor, { min: 0 });
  assertFiniteNumber(label, 'sigmaNormal', opts.sigmaNormal ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaNormal, { min: 0 });
  assertFiniteNumber(label, 'sigmaDepth', opts.sigmaDepth ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaDepth, { min: 0 });

  const reproj = opts.reprojUniforms;
  assertFiniteNumber(label, 'reprojUniforms.sigmaDepth', reproj?.sigmaDepth ?? SVGF_REPROJ_DEFAULT_UNIFORMS.sigmaDepth, { min: 0 });
  assertFiniteNumber(label, 'reprojUniforms.sigmaNormal', reproj?.sigmaNormal ?? SVGF_REPROJ_DEFAULT_UNIFORMS.sigmaNormal, { min: 0 });
  assertFiniteNumber(label, 'reprojUniforms.alphaMin', reproj?.alphaMin ?? SVGF_REPROJ_DEFAULT_UNIFORMS.alphaMin, { min: 0, max: 1 });
  assertFiniteNumber(label, 'reprojUniforms.forceReset', reproj?.forceReset ?? SVGF_REPROJ_DEFAULT_UNIFORMS.forceReset ?? 0, { integer: true, min: 0, max: 1 });
  return px;
}

/**
 * Run a full Schied 2017 SVGF denoiser pass on the given inputs.
 * Returns `{ rgb, prevRadianceOut?, momentsOut?, historyLengthOut? }`: `rgb`
 * is the filtered HDR RGB (length width*height*3); the remaining fields are
 * populated only when `opts.chainable` is true (for multi-frame chaining).
 */
export async function runSVGFRealWebGPU(
  opts: SVGFRealWebGPUOptions,
): Promise<SVGFRealWebGPUResult> {
  const w = opts.width;
  const h = opts.height;
  const px = assertSVGFRealWebGPUInputs(opts);
  const rawAtrous = opts.atrousIterations ?? SVGF_REAL_DEFAULT_ATROUS_ITERATIONS;
  const atrousIterations = Math.min(
    SVGF_REAL_MAX_ATROUS_ITERATIONS,
    Math.max(1, Math.floor(rawAtrous)),
  );
  const sigmaColor = opts.sigmaColor ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaColor;
  const sigmaNormal = opts.sigmaNormal ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaNormal;
  const sigmaDepth = opts.sigmaDepth ?? ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaDepth;

  const reprojU: SVGFReprojUniforms = {
    sigmaDepth: opts.reprojUniforms?.sigmaDepth ?? SVGF_REPROJ_DEFAULT_UNIFORMS.sigmaDepth,
    sigmaNormal: opts.reprojUniforms?.sigmaNormal ?? SVGF_REPROJ_DEFAULT_UNIFORMS.sigmaNormal,
    alphaMin: opts.reprojUniforms?.alphaMin ?? SVGF_REPROJ_DEFAULT_UNIFORMS.alphaMin,
    forceReset: opts.reprojUniforms?.forceReset ?? SVGF_REPROJ_DEFAULT_UNIFORMS.forceReset ?? 0,
  };
  const rgbForChain =
    opts.albedoRgb != null
      ? svgfRealDemodulateAlbedo(opts.rgb, opts.albedoRgb, px)
      : opts.rgb;
  const prevForChain =
    opts.albedoRgb != null
      ? svgfRealDemodulateAlbedo(
          opts.prevRadianceRgb ?? opts.rgb,
          opts.prevAlbedoRgb ?? opts.albedoRgb,
          px,
        )
      : (opts.prevRadianceRgb ?? opts.rgb);
  assertFiniteFloat16Slice(
    'runSVGFRealWebGPU',
    'rgbForChain',
    rgbForChain,
    px * 3,
  );
  assertFiniteFloat16Slice(
    'runSVGFRealWebGPU',
    'prevRadianceForChain',
    prevForChain,
    px * 3,
  );

  const { device, dispose: destroyEphemeral } = await acquireDenoiseDevice({
    device: opts.device,
    reuseSharedWebGpuDevice: opts.reuseSharedWebGpuDevice,
    errorLabel: 'runSVGFRealWebGPU',
  });

  const { trackTexture, trackBuffer, dispose: disposeResources } =
    makeResourceTracker(destroyEphemeral);

  try {
    assertOneShotDeviceLimits(device, 'runSVGFRealWebGPU', w, h, opts.chainable === true ? 16 : 8);
    const { reprojPipeline, momentsPipeline, fallbackPipeline, atrousPipeline } =
      svgfRealPipelines(device);

    // ── Texture creation ──────────────────────────────────────────────────────
    const texB = GPUTextureUsage.TEXTURE_BINDING;
    const texS = GPUTextureUsage.STORAGE_BINDING;
    const texC = GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;

    const currColorTex = trackTexture(
      device.createTexture({
        label: 'svgf-curr-color',
        size: [w, h],
        format: 'rgba16float',
        usage: texB | texC,
      }),
    );
    const prevColorTex = trackTexture(
      device.createTexture({
        label: 'svgf-prev-color',
        size: [w, h],
        format: 'rgba16float',
        usage: texB | texC,
      }),
    );
    const motionTex = trackTexture(
      device.createTexture({
        label: 'svgf-motion',
        size: [w, h],
        format: 'rg32float',
        usage: texB | texC,
      }),
    );
    const currDepthTex = trackTexture(
      device.createTexture({
        label: 'svgf-depth',
        size: [w, h],
        format: 'r32float',
        usage: texB | texC,
      }),
    );
    const currNormTex = trackTexture(
      device.createTexture({
        label: 'svgf-norm',
        size: [w, h],
        format: 'rgba32float',
        usage: texB | texC,
      }),
    );
    const currObjTex = trackTexture(
      device.createTexture({
        label: 'svgf-obj',
        size: [w, h],
        format: 'r32uint',
        usage: texB | texC,
      }),
    );
    const prevDepthTex = trackTexture(
      device.createTexture({
        label: 'svgf-prev-depth',
        size: [w, h],
        format: 'r32float',
        usage: texB | texC,
      }),
    );
    const prevNormTex = trackTexture(
      device.createTexture({
        label: 'svgf-prev-norm',
        size: [w, h],
        format: 'rgba32float',
        usage: texB | texC,
      }),
    );
    const prevObjTex = trackTexture(
      device.createTexture({
        label: 'svgf-prev-obj',
        size: [w, h],
        format: 'r32uint',
        usage: texB | texC,
      }),
    );
    const histInTex = trackTexture(
      device.createTexture({
        label: 'svgf-hist-in',
        size: [w, h],
        format: 'r16uint',
        usage: texB | texC,
      }),
    );
    const momentsInTex = trackTexture(
      device.createTexture({
        label: 'svgf-mom-in',
        size: [w, h],
        format: 'rg32float',
        usage: texB | texC,
      }),
    );

    // Reprojection outputs
    const colorOutTex = trackTexture(
      device.createTexture({
        label: 'svgf-color-out',
        size: [w, h],
        format: 'rgba16float',
        usage: texS | texB | texC,
      }),
    );
    const histOutTex = trackTexture(
      device.createTexture({
        label: 'svgf-hist-out',
        size: [w, h],
        format: 'r32uint',
        usage: texS | texB | texC,
      }),
    );
    const momOutTex = trackTexture(
      device.createTexture({
        label: 'svgf-mom-out',
        size: [w, h],
        format: 'rgba32float',
        usage: texS | texB | texC,
      }),
    );

    // Variance from moments output
    const varMomOutTex = trackTexture(
      device.createTexture({
        label: 'svgf-var-mom',
        size: [w, h],
        format: 'r32float',
        usage: texS | texB | texC,
      }),
    );
    // Merged variance (after 7×7 fallback)
    const varFinalTex = trackTexture(
      device.createTexture({
        label: 'svgf-var-final',
        size: [w, h],
        format: 'r32float',
        usage: texS | texB | texC,
      }),
    );

    // The real-SVGF wavelet chain starts from the merged moment/short-history
    // estimate in varFinalTex and then propagates variance in color alpha.

    // Atrous ping-pong
    const pingPongUsage = texS | texB | texC;
    const pingTex = trackTexture(
      device.createTexture({
        label: 'svgf-ping',
        size: [w, h],
        format: 'rgba16float',
        usage: pingPongUsage,
      }),
    );
    const pongTex = trackTexture(
      device.createTexture({
        label: 'svgf-pong',
        size: [w, h],
        format: 'rgba16float',
        usage: pingPongUsage,
      }),
    );

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
    //
    // ── Demodulation cross-reference (D14-3) ─────────────────────────────────
    // This is DELIBERATELY BROADER than atrousVarianceWebGPU.ts's demodulation.
    // Here BOTH current AND previous radiance are demodulated up front so the
    // ENTIRE chain — reprojection blending, moment tracking, variance
    // estimation, and the à-trous spatial filter — operates on lighting L = c/ρ
    // (this path carries real temporal moments). In atrousVarianceWebGPU.ts only
    // the à-trous input is demodulated while the variance pass reads the
    // original noisy `rgb` (no temporal moments there). The divergence is
    // intentional — do NOT unify; the algorithms differ (Schied SVGF vs
    // à-trous lookup).
    uploadRgbAsRgba16f(device, currColorTex, rgbForChain, w, h);
    uploadRgbAsRgba16f(device, prevColorTex, prevForChain, w, h);

    if (opts.motionRg != null) {
      uploadInterleavedRgAsRg32f(device, motionTex, opts.motionRg, w, h);
    } else {
      fillRg32f(device, motionTex, w, h, 0, 0);
    }
    // Depth: curr from linearDepth (or 1.0); prev from prevLinearDepth when
    // supplied, else mirror curr (one-shot fallback).
    if (opts.linearDepth != null) {
      uploadR32f(device, currDepthTex, opts.linearDepth, w, h);
    } else {
      uploadR32f(device, currDepthTex, new Float32Array(w * h).fill(1), w, h);
    }
    {
      const prevDepth = opts.prevLinearDepth ?? opts.linearDepth;
      if (prevDepth != null) {
        uploadR32f(device, prevDepthTex, prevDepth, w, h);
      } else {
        uploadR32f(device, prevDepthTex, new Float32Array(w * h).fill(1), w, h);
      }
    }
    // Normals: curr from gbufferNormalsRgb (or rest pose); prev from
    // prevNormalsRgb when supplied, else mirror curr.
    const NORMAL_REST: readonly [number, number, number, number] = [0.5, 0.5, 1.0, 0.0];
    if (opts.gbufferNormalsRgb != null) {
      uploadRgbAsRgba32fPacked(device, currNormTex, opts.gbufferNormalsRgb, w, h, NORMAL_REST);
    } else {
      fillRgba32f(device, currNormTex, w, h, NORMAL_REST);
    }
    {
      const prevNormals = opts.prevNormalsRgb ?? opts.gbufferNormalsRgb;
      if (prevNormals != null) {
        uploadRgbAsRgba32fPacked(device, prevNormTex, prevNormals, w, h, NORMAL_REST);
      } else {
        fillRgba32f(device, prevNormTex, w, h, NORMAL_REST);
      }
    }
    // Object IDs: curr from objectIds (or 0); prev from prevObjectIds when
    // supplied, else mirror curr.
    if (opts.objectIds != null) {
      uploadR32Uint(device, currObjTex, opts.objectIds, w, h);
    } else {
      uploadR32Uint(device, currObjTex, new Uint32Array(w * h), w, h);
    }
    {
      const prevObj = opts.prevObjectIds ?? opts.objectIds;
      if (prevObj != null) {
        uploadR32Uint(device, prevObjTex, prevObj, w, h);
      } else {
        uploadR32Uint(device, prevObjTex, new Uint32Array(w * h), w, h);
      }
    }
    // History length in: prevHistoryLength wins over historyLengthIn (both name
    // the previous-frame history feeding reprojection); fall back to 0.
    {
      const histIn = opts.prevHistoryLength ?? opts.historyLengthIn;
      if (histIn != null) {
        uploadR16Uint(device, histInTex, histIn, w, h);
      } else {
        fillR16Uint(device, histInTex, w, h, 0);
      }
    }
    if (opts.momentsIn != null) {
      uploadInterleavedRgAsRg32f(device, momentsInTex, opts.momentsIn, w, h);
    } else {
      fillRg32f(device, momentsInTex, w, h, 0, 0);
    }

    // ── UBOs ─────────────────────────────────────────────────────────────────
    const reprojUboScratch = new ArrayBuffer(SVGF_REPROJ_UNIFORMS_SIZE_BYTES);
    packSVGFReprojUniforms(reprojU, reprojUboScratch);
    const reprojUboGpu = trackBuffer(
      device.createBuffer({
        size: SVGF_REPROJ_UNIFORMS_SIZE_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    );
    device.queue.writeBuffer(reprojUboGpu, 0, reprojUboScratch);

    // Atrous UBOs + alternating ping-pong bind groups are built by the shared
    // buildAtrousChain (atrousChain.ts) during command encoding below.

    // ── Bind groups ───────────────────────────────────────────────────────────
    const reprojBG = device.createBindGroup({
      layout: reprojPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: currColorTex.createView() },
        { binding: 1, resource: prevColorTex.createView() },
        { binding: 2, resource: motionTex.createView() },
        { binding: 3, resource: currDepthTex.createView() },
        { binding: 4, resource: currNormTex.createView() },
        { binding: 5, resource: currObjTex.createView() },
        { binding: 6, resource: prevDepthTex.createView() },
        { binding: 7, resource: prevNormTex.createView() },
        { binding: 8, resource: prevObjTex.createView() },
        { binding: 9, resource: histInTex.createView() },
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
        // Short-history variance is a spatial estimate of the current noisy
        // signal, not of the already temporally blended reprojection output.
        { binding: 0, resource: currColorTex.createView() },
        { binding: 1, resource: histOutTex.createView() },
        { binding: 2, resource: varMomOutTex.createView() },
        { binding: 3, resource: varFinalTex.createView() },
        { binding: 4, resource: currNormTex.createView() },
        { binding: 5, resource: currDepthTex.createView() },
      ],
    });

    // Initial copy of colorOut → pingTex for wavelet iteration zero. The
    // ping-pong scaffolding is delegated to buildAtrousChain below.

    // ── Command encoding ──────────────────────────────────────────────────────
    const wg = SVGF_REAL_REPROJECTION_WORKGROUP_SIZE;
    const wg2 = SVGF_VARIANCE_FROM_MOMENTS_WORKGROUP_SIZE;
    const wg3 = SVGF_7X7_SPATIAL_FALLBACK_WORKGROUP_SIZE;
    // The à-trous workgroup size (ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE) is
    // owned by buildAtrousChain (atrousChain.ts) now.

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

    // 5. À-trous chain — shared ping-pong scaffolding with a dedicated real-
    // SVGF shader. Iteration zero reads varFinalTex; each output stores the
    // squared-weight propagated variance in alpha for the next wavelet level.
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
      pingTex,
      pongTex,
      normalView: currNormTex.createView(),
      depthView: currDepthTex.createView(),
      varianceView: varFinalTex.createView(),
      uboLabelPrefix: 'svgf-real-atrous-ubo-',
      trackBuffer,
      // Schied 2017 §4.3 uses the first wavelet level as the next frame's
      // color history. colorOutTex is free after its initial copy to pingTex,
      // so preserve level zero there before later ping-pong iterations run.
      ...(opts.chainable === true
        ? {
            afterIteration: (iteration: number, _input: GPUTexture, output: GPUTexture) => {
              if (iteration !== 0) return;
              encoder.copyTextureToTexture(
                { texture: output },
                { texture: colorOutTex },
                [w, h],
              );
            },
          }
        : {}),
    });

    device.queue.submit([encoder.finish()]);

    const result = await readRgba16fToRgb(device, readTex, w, h);

    // Schied 2017 §4.1 — albedo re-modulation: multiply the filtered lighting
    // by per-pixel albedo to restore physically correct denoised outgoing
    // radiance. In-place mutation of `result`. No-op when albedoRgb is omitted.
    if (opts.albedoRgb != null) {
      svgfRealRemodulateAlbedo(result, opts.albedoRgb, px);
    }

    // Multi-frame chaining: read the first-wavelet color plus reprojection
    // outputs (blended moments + per-pixel history length) back BEFORE the
    // `finally` destroys the textures. colorOutTex was overwritten by the
    // level-zero copy above; momOutTex is rgba32float (M1/M2 in .rg);
    // histOutTex is r32uint. Skipped entirely when `chainable` is false.
    if (opts.chainable === true) {
      const prevRadianceOut = await readRgba16fToRgb(device, colorOutTex, w, h);
      if (opts.albedoRgb != null) {
        svgfRealRemodulateAlbedo(prevRadianceOut, opts.albedoRgb, px);
      }
      const momentsOut = await readRgba32fToRg(device, momOutTex, w, h);
      const historyLengthOut = await readR32UintToU32(device, histOutTex, w, h);
      return { rgb: result, prevRadianceOut, momentsOut, historyLengthOut };
    }

    return { rgb: result };
  } finally {
    disposeResources();
  }
}
