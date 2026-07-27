// materialScalarDerivations.ts — backend-agnostic scalar derivations shared by
// the pt-webgpu and pt-webgl2 material packers.
//
// Both backends pack materials into per-backend texel layouts (a vec4 stream for
// pt-webgpu, an RGBA32F record for pt-webgl2), so the LAYOUT stays per-backend.
// But several SCALAR DERIVATIONS — the math that turns a `@vitrum/core`
// `MaterialSpec` field into a packed number — were duplicated verbatim (or nearly
// so) in both packers. This module single-sources exactly those derivations:
//
//   - `sigmaAFromAttenuation` — Beer-Lambert σ_a from KHR_materials_volume
//     attenuationColor/attenuationDistance (identical in both backends).
//   - `sampleSpectralCurve` — strict linear-interpolated μ(λ) lookup into a core
//     `SpectralCurve` without silently canonicalizing malformed input.
//   - `sampleSpectralGrid` — the 32-sample 380→780 nm uniform grid sampling loop.
//   - `dispersionStrengthFromAbbe` — dispersion strength from the Abbe number,
//     evaluated at the Fraunhofer C/F lines (pt-webgl2's `dispersionStrength`).
//   - `resolveEmissiveIntensity` — the `emissiveIntensity ?? 1` default.
//
// The Fraunhofer C/F line constants are re-used from `cauchyIor.ts` so there is
// one source for those wavelengths across the whole package.

import { FRAUNHOFER_C_NM, FRAUNHOFER_F_NM } from './cauchyIor.js';

/** RGB triple (matches `@vitrum/core`'s `Vec3`, restated here to avoid a core
 *  dependency in the leaf sampler package). */
export type MaterialScalarVec3 = readonly [number, number, number];

/** Minimal shape of `@vitrum/core`'s `SpectralCurve` consumed by the samplers.
 *  `values` is `ArrayLike<number>` so a core `Float32Array` (its runtime type)
 *  and a plain `number[]` (test fixtures) both satisfy it. */
export interface SpectralCurveLike {
  readonly wavelengthStart: number;
  readonly wavelengthEnd: number;
  readonly values: ArrayLike<number>;
}

/** Canonical 32-sample uniform spectral grid (380→780 nm inclusive) shared by
 *  both packers' spectral-attenuation blocks. */
export const SPECTRAL_GRID_SAMPLE_COUNT = 32;
export const SPECTRAL_GRID_START_NM = 380.0;
export const SPECTRAL_GRID_END_NM = 780.0;

/**
 * Beer-Lambert absorption coefficient σ_a from KHR_materials_volume
 * `attenuationColor` (transmittance at `attenuationDistance`) and
 * `attenuationDistance`:
 *
 *   T(d) = attenuationColor = exp(-σ_a · attenuationDistance)
 *   ⇒ σ_a = -ln(attenuationColor) / attenuationDistance   (≥ 0)
 *
 * A zero transmittance channel maps to positive infinity, its exact
 * Beer-Lambert coefficient. Positive-infinite distance means no attenuation.
 * Every other malformed value is rejected instead of being rewritten.
 */
export function sigmaAFromAttenuation(
  attenuationColor: MaterialScalarVec3,
  attenuationDistance: number,
): [number, number, number] {
  for (let channelIndex = 0; channelIndex < 3; channelIndex += 1) {
    const channel = attenuationColor[channelIndex]!;
    if (!Number.isFinite(channel) || channel < 0 || channel > 1) {
      throw new RangeError(
        `sigmaAFromAttenuation.attenuationColor[${channelIndex}] must be finite and in [0, 1]`,
      );
    }
  }
  if (attenuationDistance === Number.POSITIVE_INFINITY) {
    return [0.0, 0.0, 0.0];
  }
  if (!Number.isFinite(attenuationDistance) || attenuationDistance <= 0.0) {
    throw new RangeError(
      'sigmaAFromAttenuation.attenuationDistance must be positive or +Infinity',
    );
  }
  const sigmaAChannel = (channel: number): number => {
    if (channel === 0) return Number.POSITIVE_INFINITY;
    if (channel === 1) return 0;
    return -Math.log(channel) / attenuationDistance;
  };
  return [
    sigmaAChannel(attenuationColor[0]),
    sigmaAChannel(attenuationColor[1]),
    sigmaAChannel(attenuationColor[2]),
  ];
}

/** Validation options for {@link sampleSpectralCurve}. */
export interface SpectralCurveSampleOptions {
  /** Minimum number of curve values required to sample. Default 1. */
  readonly minValueCount?: number;
}

interface ValidatedSpectralCurve {
  readonly wavelengthStart: number;
  readonly wavelengthEnd: number;
  readonly values: readonly number[];
}

function validateSpectralCurve(
  curve: SpectralCurveLike,
  minValueCount: number,
): ValidatedSpectralCurve {
  const values = curve.values;
  if (
    values == null ||
    !Number.isSafeInteger(values.length) ||
    values.length < minValueCount
  ) {
    throw new RangeError(
      `sampleSpectralCurve.values must contain at least ${minValueCount} sample(s)`,
    );
  }
  if (
    !Number.isFinite(curve.wavelengthStart) ||
    !Number.isFinite(curve.wavelengthEnd) ||
    !(curve.wavelengthEnd > curve.wavelengthStart)
  ) {
    throw new RangeError(
      'sampleSpectralCurve wavelength bounds must be finite with wavelengthEnd > wavelengthStart',
    );
  }
  const checkedValues = new Array<number>(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value) || Number(value) < 0) {
      throw new RangeError(
        `sampleSpectralCurve.values[${index}] must be finite and non-negative`,
      );
    }
    checkedValues[index] = Number(value);
  }
  return {
    wavelengthStart: curve.wavelengthStart,
    wavelengthEnd: curve.wavelengthEnd,
    values: checkedValues,
  };
}

function sampleValidatedSpectralCurve(
  curve: ValidatedSpectralCurve,
  lambdaNm: number,
): number {
  const t = Math.min(
    1,
    Math.max(
      0,
      (lambdaNm - curve.wavelengthStart) /
        (curve.wavelengthEnd - curve.wavelengthStart),
    ),
  );
  const f = t * (curve.values.length - 1);
  const i0 = Math.floor(f);
  const i1 = Math.min(i0 + 1, curve.values.length - 1);
  const a = curve.values[i0]!;
  const b = curve.values[i1]!;
  return a + (b - a) * (f - i0);
}

/**
 * Sample a core `SpectralCurve` at `lambdaNm` with linear interpolation between
 * the two nearest grid values. A missing curve evaluates to zero; a present but
 * malformed curve is rejected.
 */
export function sampleSpectralCurve(
  curve: SpectralCurveLike | null | undefined,
  lambdaNm: number,
  options: SpectralCurveSampleOptions = {},
): number {
  const minValueCount = options.minValueCount ?? 1;
  if (!Number.isSafeInteger(minValueCount) || minValueCount < 1) {
    throw new RangeError('sampleSpectralCurve.minValueCount must be a positive safe integer');
  }
  if (curve == null) return 0;
  if (!Number.isFinite(lambdaNm)) {
    throw new RangeError('sampleSpectralCurve.lambdaNm must be finite');
  }
  return sampleValidatedSpectralCurve(
    validateSpectralCurve(curve, minValueCount),
    lambdaNm,
  );
}

/** Result of {@link sampleSpectralGrid}: the 32 grid samples plus the summary
 *  statistics pt-webgpu folds in the same loop (avg/max/count). Consumers use
 *  whichever fields they need — pt-webgl2 uses only `samples`, pt-webgpu uses
 *  all four. */
export interface SpectralGridResult {
  /** 32 μ(λ) samples across 380→780 nm (or all-zero when no curve). */
  readonly samples: number[];
  /** Mean of the samples (0 when no curve). pt-webgpu's `spectralAvgMu`. */
  readonly avg: number;
  /** Max of the samples (0 when no curve). pt-webgpu's `spectralMaxMu`. */
  readonly max: number;
  /** Number of samples written (`SPECTRAL_GRID_SAMPLE_COUNT` when a curve is
   *  present, else 0). pt-webgpu's `spectralSampleCount`. */
  readonly sampleCount: number;
}

/**
 * Sample a core `SpectralCurve` onto the canonical 32-sample 380→780 nm uniform
 * grid and fold avg/max/count. A present malformed curve is rejected before any
 * samples are emitted.
 */
export function sampleSpectralGrid(
  curve: SpectralCurveLike | null | undefined,
  curveOptions: SpectralCurveSampleOptions = {},
): SpectralGridResult {
  const samples = new Array<number>(SPECTRAL_GRID_SAMPLE_COUNT).fill(0);
  const minValueCount = curveOptions.minValueCount ?? 1;
  if (!Number.isSafeInteger(minValueCount) || minValueCount < 1) {
    throw new RangeError('sampleSpectralGrid.minValueCount must be a positive safe integer');
  }
  if (curve == null) {
    return { samples, avg: 0, max: 0, sampleCount: 0 };
  }
  const validatedCurve = validateSpectralCurve(curve, minValueCount);
  const denom = SPECTRAL_GRID_SAMPLE_COUNT - 1;
  let sum = 0;
  let maxMu = 0;
  for (let i = 0; i < SPECTRAL_GRID_SAMPLE_COUNT; i += 1) {
    const t = i / denom;
    const lambda = SPECTRAL_GRID_START_NM + t * (SPECTRAL_GRID_END_NM - SPECTRAL_GRID_START_NM);
    const v = sampleValidatedSpectralCurve(validatedCurve, lambda);
    samples[i] = v;
    sum += v;
    maxMu = Math.max(maxMu, v);
  }
  const avg = sum / SPECTRAL_GRID_SAMPLE_COUNT;
  return {
    samples,
    avg,
    max: maxMu,
    sampleCount: SPECTRAL_GRID_SAMPLE_COUNT,
  };
}

/**
 * Dispersion strength from the Abbe number V_d and IOR, evaluated at the
 * Fraunhofer C/F lines — pt-webgl2's `dispersionStrengthFromAbbe`
 * (exact port of the absorbed fork's derivation). Zero Abbe or IOR ≤ 1 means
 * dispersion is disabled; malformed negative/non-finite values are rejected.
 */
export function dispersionStrengthFromAbbe(ior: number, abbe: number): number {
  if (!Number.isFinite(ior) || ior <= 0) {
    throw new RangeError('dispersionStrengthFromAbbe.ior must be finite and positive');
  }
  if (!Number.isFinite(abbe) || abbe < 0) {
    throw new RangeError('dispersionStrengthFromAbbe.abbe must be finite and non-negative');
  }
  if (abbe === 0 || ior <= 1) return 0;
  const denom =
    1 / (FRAUNHOFER_F_NM * FRAUNHOFER_F_NM) - 1 / (FRAUNHOFER_C_NM * FRAUNHOFER_C_NM);
  if (denom === 0) {
    throw new Error('Fraunhofer C/F wavelengths must define a non-zero dispersion denominator');
  }
  return (ior - 1) / (abbe * denom);
}

/** The shared `emissiveIntensity ?? 1` default. Both backends default a missing
 *  `emissiveIntensity` to 1.0 (0.0 would silently black-out emissive maps). */
export function resolveEmissiveIntensity(emissiveIntensity: number | undefined): number {
  if (emissiveIntensity === undefined) return 1;
  if (!Number.isFinite(emissiveIntensity) || emissiveIntensity < 0) {
    throw new RangeError('resolveEmissiveIntensity value must be finite and non-negative');
  }
  return emissiveIntensity;
}
