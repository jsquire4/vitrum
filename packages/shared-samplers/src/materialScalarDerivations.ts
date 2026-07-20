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
//   - `sampleSpectralCurve` — linear-interpolated μ(λ) lookup into a core
//     `SpectralCurve`. The two backends had slightly different edge handling
//     (minimum value count, wavelength-range fallback), so this is parameterized
//     via `SpectralCurveSampleOptions` to reproduce EACH backend byte-for-byte.
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

function isFiniteNumber(v: number | undefined): boolean {
  return Number.isFinite(v);
}

function finiteOr(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? Number(value) : fallback;
}

/** Default transmittance-clamp epsilon shared by both material packers.
 *  `attenuationColor` channels are clamped to `[epsilon, 1]` before the log so
 *  a 0 channel does not produce -Infinity σ_a. */
export const ATTENUATION_TRANSMITTANCE_EPSILON = 1e-4;

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
 *   ⇒ σ_a = -ln(clamp(attenuationColor, epsilon, 1)) / attenuationDistance   (≥ 0)
 *
 * Returns `[0, 0, 0]` when the distance is non-finite or ≤ 0 (no absorption).
 * This reproduces both backends' prior per-channel derivation byte-for-byte.
 */
export function sigmaAFromAttenuation(
  attenuationColor: MaterialScalarVec3,
  attenuationDistance: number,
  epsilon: number = ATTENUATION_TRANSMITTANCE_EPSILON,
): [number, number, number] {
  if (!Number.isFinite(attenuationDistance) || attenuationDistance <= 0.0) {
    return [0.0, 0.0, 0.0];
  }
  const sigmaAChannel = (channel: number): number => {
    const transmittance = Math.min(Math.max(finiteOr(channel, 1.0), epsilon), 1.0);
    return Math.max(-Math.log(transmittance) / attenuationDistance, 0.0);
  };
  return [
    sigmaAChannel(attenuationColor[0]),
    sigmaAChannel(attenuationColor[1]),
    sigmaAChannel(attenuationColor[2]),
  ];
}

/** Edge-handling knobs for {@link sampleSpectralCurve}. The two backends diverge
 *  ONLY in these two aspects — the interpolation math is identical — so this
 *  option object lets one shared function reproduce each backend exactly. */
export interface SpectralCurveSampleOptions {
  /** Minimum number of curve values required to sample (else return 0).
   *  pt-webgpu accepted length ≥ 1; pt-webgl2 requires length ≥ 2. Default 1. */
  readonly minValueCount?: number;
  /** Wavelength-range fallback when `wavelengthStart`/`End` are non-finite.
   *  pt-webgpu used the raw (possibly-non-finite) fields; pt-webgl2 fell back to
   *  the 380/780 nm grid bounds. When omitted, the raw fields are used verbatim
   *  (pt-webgpu behavior). */
  readonly fallbackStartNm?: number;
  readonly fallbackEndNm?: number;
}

/**
 * Sample a core `SpectralCurve` at `lambdaNm` with linear interpolation between
 * the two nearest grid values. Shared by both backends; edge handling is
 * parameterized via {@link SpectralCurveSampleOptions} to preserve each
 * backend's exact prior behavior.
 */
export function sampleSpectralCurve(
  curve: SpectralCurveLike | null | undefined,
  lambdaNm: number,
  options: SpectralCurveSampleOptions = {},
): number {
  if (curve == null) return 0;
  const values = curve.values;
  const minValueCount = options.minValueCount ?? 1;
  if (!values || values.length < minValueCount || values.length === 0) return 0;
  const start =
    options.fallbackStartNm !== undefined
      ? isFiniteNumber(curve.wavelengthStart)
        ? curve.wavelengthStart
        : options.fallbackStartNm
      : curve.wavelengthStart;
  const end =
    options.fallbackEndNm !== undefined
      ? isFiniteNumber(curve.wavelengthEnd)
        ? curve.wavelengthEnd
        : options.fallbackEndNm
      : curve.wavelengthEnd;
  const denom = Math.max(end - start, 1e-6);
  const t = Math.min(1, Math.max(0, (lambdaNm - start) / denom));
  const f = t * (values.length - 1);
  const i0 = Math.floor(f);
  const i1 = Math.min(i0 + 1, values.length - 1);
  const a = Number(values[i0] ?? 0);
  const b = Number(values[i1] ?? a);
  return a + (b - a) * (f - i0);
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
 * grid, clamping each sample to ≥ 0 and folding in avg/max/count. Shared by both
 * backends' spectral-attenuation packing. Edge handling is passed through to
 * {@link sampleSpectralCurve} via `curveOptions` so each backend keeps its exact
 * prior μ(λ) lookup.
 */
export function sampleSpectralGrid(
  curve: SpectralCurveLike | null | undefined,
  curveOptions: SpectralCurveSampleOptions = {},
): SpectralGridResult {
  const samples = new Array<number>(SPECTRAL_GRID_SAMPLE_COUNT).fill(0);
  const hasCurve = curve != null && curve.values.length > 0;
  if (!hasCurve) {
    return { samples, avg: 0, max: 0, sampleCount: 0 };
  }
  const denom = Math.max(SPECTRAL_GRID_SAMPLE_COUNT - 1, 1);
  let sum = 0;
  let maxMu = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < SPECTRAL_GRID_SAMPLE_COUNT; i += 1) {
    const t = i / denom;
    const lambda = SPECTRAL_GRID_START_NM + t * (SPECTRAL_GRID_END_NM - SPECTRAL_GRID_START_NM);
    const v = Math.max(sampleSpectralCurve(curve, lambda, curveOptions), 0);
    samples[i] = v;
    sum += v;
    maxMu = Math.max(maxMu, v);
  }
  const avg = sum / SPECTRAL_GRID_SAMPLE_COUNT;
  return {
    samples,
    avg,
    max: Number.isFinite(maxMu) ? maxMu : 0,
    sampleCount: SPECTRAL_GRID_SAMPLE_COUNT,
  };
}

/**
 * Dispersion strength from the Abbe number V_d and IOR, evaluated at the
 * Fraunhofer C/F lines — pt-webgl2's `dispersionStrengthFromAbbe`
 * (exact port of the absorbed fork's derivation). Returns 0 when abbe ≤ 0 ||
 * ior ≤ 1 (no dispersion) or when the denominator underflows.
 */
export function dispersionStrengthFromAbbe(ior: number, abbe: number): number {
  if (abbe <= 0 || ior <= 1) return 0;
  const denom =
    1 / (FRAUNHOFER_F_NM * FRAUNHOFER_F_NM) - 1 / (FRAUNHOFER_C_NM * FRAUNHOFER_C_NM);
  if (Math.abs(denom) < 1e-12) return 0;
  return Math.max(0, (ior - 1) / (abbe * denom));
}

/** The shared `emissiveIntensity ?? 1` default. Both backends default a missing
 *  `emissiveIntensity` to 1.0 (0.0 would silently black-out emissive maps). */
export function resolveEmissiveIntensity(emissiveIntensity: number | undefined): number {
  return emissiveIntensity ?? 1.0;
}
