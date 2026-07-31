import type { FrameQualitySettings } from '@vitrum/core';
import { requireNonNegativeFloat32 } from './scene/float32Policy.js';

export interface ResolvedWebGl2FrameQuality {
  readonly samplesTarget: number;
  readonly bounces: number;
  readonly filteredGlossyFactor: number;
  readonly resolutionFactor: number;
  readonly exposure: number;
  readonly tonemap: NonNullable<FrameQualitySettings['tonemap']>;
  readonly outputColorSpace: NonNullable<FrameQualitySettings['outputColorSpace']>;
}

const TONEMAP_MODES = new Set<NonNullable<FrameQualitySettings['tonemap']>>([
  'aces',
  'agx',
  'reinhard',
  'linear',
  'none',
]);
const OUTPUT_COLOR_SPACES = new Set<NonNullable<FrameQualitySettings['outputColorSpace']>>([
  'srgb',
  'linear',
]);
const QUALITY_KEYS = new Set<keyof FrameQualitySettings>([
  'samplesTarget',
  'bounces',
  'filteredGlossyFactor',
  'resolutionFactor',
  'exposure',
  'tonemap',
  'outputColorSpace',
]);

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function describeValidationValue(value: unknown): string {
  return value !== null && typeof value === 'object'
    ? Object.prototype.toString.call(value)
    : String(value);
}

function assertFiniteNumber(label: string, value: unknown): asserts value is number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite (got ${String(value)})`);
  }
}

/** Fail closed on malformed or typo-bearing per-frame quality payloads. */
export function validateWebGl2FrameQuality(quality: unknown): asserts quality is FrameQualitySettings | undefined {
  if (quality === undefined) return;
  if (quality === null || typeof quality !== 'object' || Array.isArray(quality)) {
    throw new TypeError('renderFrame: quality must be a non-array object when supplied');
  }
  if (Object.getOwnPropertySymbols(quality).length !== 0) {
    throw new TypeError('renderFrame: quality contains unsupported symbol keys');
  }
  const q = quality as Record<string, unknown>;
  for (const key of Object.keys(q)) {
    if (!QUALITY_KEYS.has(key as keyof FrameQualitySettings)) {
      throw new TypeError(`renderFrame: quality contains unsupported field "${key}"`);
    }
  }
  for (const field of ['samplesTarget', 'bounces'] as const) {
    const value = q[field];
    if (value === undefined) continue;
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(
          `renderFrame: quality.${field} must be a positive safe integer (got ${describeValidationValue(value)})`,
      );
    }
  }
  if (q.resolutionFactor !== undefined) {
    assertFiniteNumber('renderFrame: quality.resolutionFactor', q.resolutionFactor);
    if (q.resolutionFactor <= 0 || q.resolutionFactor > 1) {
      throw new RangeError(
        `renderFrame: quality.resolutionFactor must be in (0, 1] (got ${q.resolutionFactor})`,
      );
    }
  }
  if (q.filteredGlossyFactor !== undefined) {
    assertFiniteNumber('renderFrame: quality.filteredGlossyFactor', q.filteredGlossyFactor);
    if (q.filteredGlossyFactor < 0 || q.filteredGlossyFactor > 1) {
      throw new RangeError(
        `renderFrame: quality.filteredGlossyFactor must be in [0, 1] ` +
          `(got ${q.filteredGlossyFactor})`,
      );
    }
    requireNonNegativeFloat32(
      q.filteredGlossyFactor,
      'renderFrame: quality.filteredGlossyFactor',
    );
  }
  if (q.exposure !== undefined) {
    assertFiniteNumber('renderFrame: quality.exposure', q.exposure);
    if (q.exposure < 0) {
      throw new RangeError(
        `renderFrame: quality.exposure must be >= 0 (got ${q.exposure})`,
      );
    }
    requireNonNegativeFloat32(
      q.exposure,
      'renderFrame: quality.exposure',
    );
  }
  if (q.tonemap !== undefined &&
      (typeof q.tonemap !== 'string' || !TONEMAP_MODES.has(q.tonemap as never))) {
    throw new RangeError(
        `renderFrame: quality.tonemap is unsupported (got ${describeValidationValue(q.tonemap)})`,
    );
  }
  if (q.outputColorSpace !== undefined &&
      (typeof q.outputColorSpace !== 'string' ||
        !OUTPUT_COLOR_SPACES.has(q.outputColorSpace as never))) {
    throw new RangeError(
      `renderFrame: quality.outputColorSpace is unsupported ` +
          `(got ${describeValidationValue(q.outputColorSpace)})`,
    );
  }
}

/**
 * Resolve the runtime quality payload to finite, contract-valid values before
 * it can affect allocation, accumulation convergence, or shader uniforms.
 */
export function resolveWebGl2FrameQuality(
  quality: FrameQualitySettings | undefined,
  maxBounces: number,
  maxSamplesPerPixel: number,
  defaultSamplesTarget: number,
): ResolvedWebGl2FrameQuality {
  validateWebGl2FrameQuality(quality);
  for (const [label, value] of [
    ['maxBounces', maxBounces],
    ['maxSamplesPerPixel', maxSamplesPerPixel],
    ['defaultSamplesTarget', defaultSamplesTarget],
  ] as const) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`resolveWebGl2FrameQuality: ${label} must be a positive safe integer`);
    }
  }
  const q = quality ?? {};
  const requestedBounces = q.bounces ?? maxBounces;
  const requestedSamples = q.samplesTarget ?? defaultSamplesTarget;
  const requestedResolution = q.resolutionFactor ?? 1;
  const requestedGlossy = requireNonNegativeFloat32(
    q.filteredGlossyFactor ?? 0,
    'renderFrame: quality.filteredGlossyFactor',
  );
  const requestedExposure = requireNonNegativeFloat32(
    q.exposure ?? 1,
    'renderFrame: quality.exposure',
  );

  return {
    bounces: clamp(requestedBounces, 1, maxBounces),
    samplesTarget: clamp(requestedSamples, 1, maxSamplesPerPixel),
    filteredGlossyFactor: requestedGlossy,
    resolutionFactor: requestedResolution,
    exposure: requestedExposure,
    tonemap: q.tonemap ?? 'aces',
    outputColorSpace: q.outputColorSpace ?? 'srgb',
  };
}

export function withResolvedWebGl2FrameQuality(
  quality: ResolvedWebGl2FrameQuality,
): FrameQualitySettings {
  return {
    samplesTarget: quality.samplesTarget,
    bounces: quality.bounces,
    filteredGlossyFactor: quality.filteredGlossyFactor,
    resolutionFactor: quality.resolutionFactor,
    exposure: quality.exposure,
    tonemap: quality.tonemap,
    outputColorSpace: quality.outputColorSpace,
  };
}
