/**
 * Swap-chain presentation policy for the walkaround renderer.
 *
 * The composite shader writes floating-point color. Keep the accepted target
 * set to color-renderable WebGPU formats with at least RGB channels whose
 * fragment-output type is compatible with `vec4f`. Integer, depth/stencil,
 * compressed, and one/two-channel targets are deliberately rejected before a
 * pipeline or frame resource can be mutated.
 */

import { TONEMAP_MAX_FINITE_F32 } from '@vitrum/shared-samplers';

export const HYBRID_PRESENTATION_FORMATS = Object.freeze([
  'rgba8unorm',
  'rgba8unorm-srgb',
  'bgra8unorm',
  'bgra8unorm-srgb',
  'rgb10a2unorm',
  'rg11b10ufloat',
  'rgba16float',
  'rgba32float',
] as const satisfies readonly GPUTextureFormat[]);

export type HybridPresentationFormat =
  (typeof HYBRID_PRESENTATION_FORMATS)[number];

const HYBRID_PRESENTATION_FORMAT_SET: ReadonlySet<string> =
  new Set(HYBRID_PRESENTATION_FORMATS);

const HYBRID_PRESENTATION_MAX_RGB: Readonly<
  Record<HybridPresentationFormat, readonly [number, number, number]>
> = {
  rgba8unorm: [1, 1, 1],
  'rgba8unorm-srgb': [1, 1, 1],
  bgra8unorm: [1, 1, 1],
  'bgra8unorm-srgb': [1, 1, 1],
  rgb10a2unorm: [1, 1, 1],
  // rg11b10ufloat has 6 mantissa bits for R/G and 5 for B.
  rg11b10ufloat: [65_024, 65_024, 64_512],
  rgba16float: [65_504, 65_504, 65_504],
  rgba32float: [
    TONEMAP_MAX_FINITE_F32,
    TONEMAP_MAX_FINITE_F32,
    TONEMAP_MAX_FINITE_F32,
  ],
};

export interface HybridPresentationTarget {
  readonly format: HybridPresentationFormat;
  readonly outputColorSpace: 'srgb' | 'linear';
  /** True when the render attachment performs the linear-to-sRGB conversion. */
  readonly attachmentSrgb: boolean;
  /**
   * True when the shader/CPU sky clear must apply the sRGB OETF itself.
   * Exactly one of this and `attachmentSrgb` is true for sRGB output.
   */
  readonly applySoftwareSrgbOetf: boolean;
}

export function assertHybridSwapChainFormat(
  value: unknown,
  label = 'walkaround-hybrid swapChainFormat',
): asserts value is HybridPresentationFormat {
  if (
    typeof value !== 'string'
    || !HYBRID_PRESENTATION_FORMAT_SET.has(value)
  ) {
    throw new RangeError(
      `${label} is unsupported: ${String(value)}. Supported formats: `
      + HYBRID_PRESENTATION_FORMATS.join(', '),
    );
  }
}

export function hybridSwapChainFormatIsSrgb(
  format: HybridPresentationFormat,
): boolean {
  return format.endsWith('-srgb');
}

/** Largest finite RGB value representable by one accepted target format. */
export function hybridPresentationMaxRgb(
  formatValue: unknown,
): readonly [number, number, number] {
  assertHybridSwapChainFormat(formatValue, 'presentation target format');
  return HYBRID_PRESENTATION_MAX_RGB[formatValue];
}

/** Clamp one already-tonemapped/encoded RGB value to its concrete attachment. */
export function clampHybridPresentationRgb(
  rgb: readonly [number, number, number],
  formatValue: unknown,
): [number, number, number] {
  const maximum = hybridPresentationMaxRgb(formatValue);
  return [
    Math.min(maximum[0], Math.max(0, rgb[0])),
    Math.min(maximum[1], Math.max(0, rgb[1])),
    Math.min(maximum[2], Math.max(0, rgb[2])),
  ];
}

/** Resolve and validate the browser's preferred canvas format. */
export function getPreferredHybridSwapChainFormat(): HybridPresentationFormat {
  const preferred =
    typeof navigator !== 'undefined' && 'gpu' in navigator
      ? (
          navigator.gpu as {
            getPreferredCanvasFormat?: () => GPUTextureFormat;
          }
        ).getPreferredCanvasFormat?.() ?? 'bgra8unorm'
      : 'bgra8unorm';
  assertHybridSwapChainFormat(
    preferred,
    'navigator.gpu.getPreferredCanvasFormat() result',
  );
  return preferred;
}

/**
 * Resolve the exact attachment-transfer policy before any frame mutation.
 *
 * An sRGB attachment always encodes fragment/clear values in hardware. It can
 * therefore satisfy `outputColorSpace:'srgb'` without a shader OETF, but cannot
 * honestly expose a linear output contract. Non-sRGB targets preserve the
 * historical software-OETF path.
 */
export function resolveHybridPresentationTarget(
  formatValue: unknown,
  outputColorSpaceValue: unknown,
  label = 'HybridEngine.renderFrame',
): HybridPresentationTarget {
  const format =
    formatValue === undefined
      ? getPreferredHybridSwapChainFormat()
      : formatValue;
  assertHybridSwapChainFormat(format, `${label}: swapChainFormat`);

  const outputColorSpace =
    outputColorSpaceValue === undefined ? 'srgb' : outputColorSpaceValue;
  if (outputColorSpace !== 'srgb' && outputColorSpace !== 'linear') {
    throw new RangeError(
      `${label}: quality.outputColorSpace is unsupported: `
      + `${String(outputColorSpaceValue)}.`,
    );
  }

  const attachmentSrgb = hybridSwapChainFormatIsSrgb(format);
  if (attachmentSrgb && outputColorSpace === 'linear') {
    throw new RangeError(
      `${label}: quality.outputColorSpace 'linear' is incompatible with `
      + `sRGB swap-chain format '${format}'; the attachment always applies `
      + 'an sRGB encoding on writes.',
    );
  }

  return {
    format,
    outputColorSpace,
    attachmentSrgb,
    applySoftwareSrgbOetf:
      outputColorSpace === 'srgb' && !attachmentSrgb,
  };
}

/** WebGPU override constants paired exactly with one presentation format. */
export function hybridCompositeFragmentConstants(
  format: unknown,
): Readonly<
  Record<
    | 'VT_ATTACHMENT_SRGB'
    | 'VT_TARGET_MAX_R'
    | 'VT_TARGET_MAX_G'
    | 'VT_TARGET_MAX_B',
    number
  >
> {
  assertHybridSwapChainFormat(format, 'composite pipeline target format');
  const maximum = hybridPresentationMaxRgb(format);
  return {
    VT_ATTACHMENT_SRGB: hybridSwapChainFormatIsSrgb(format) ? 1 : 0,
    VT_TARGET_MAX_R: maximum[0],
    VT_TARGET_MAX_G: maximum[1],
    VT_TARGET_MAX_B: maximum[2],
  };
}
