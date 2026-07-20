/**
 * materialScalarDerivations.test.ts — T1-1 cross-backend parity.
 *
 * These shared derivations were extracted verbatim from the pt-webgpu
 * (`scene/materialPacking.ts`) and pt-webgl2 (`scene/materialsTexture.ts`)
 * material packers. This suite pins that the shared functions reproduce EACH
 * backend's pre-extraction inline implementation byte-for-byte, using
 * independent reference copies of the original inline math inlined here (so a
 * future edit to the shared module that drifts from either backend's original
 * semantics fails loudly).
 */
import { describe, it, expect } from 'vitest';
import {
  sigmaAFromAttenuation,
  sampleSpectralCurve,
  sampleSpectralGrid,
  dispersionStrengthFromAbbe,
  resolveEmissiveIntensity,
  SPECTRAL_GRID_SAMPLE_COUNT,
  SPECTRAL_GRID_START_NM,
  SPECTRAL_GRID_END_NM,
  type SpectralCurveLike,
  type SpectralCurveSampleOptions,
} from '../materialScalarDerivations.js';

// ── Reference copies of the ORIGINAL inline implementations ─────────────────

/** pt-webgpu materialPacking.ts (pre-T1-1) σ_a per-channel derivation. */
function refWebgpuSigmaA(
  attColor: [number, number, number],
  attDistance: number,
): [number, number, number] {
  const finite = (v: number, fallback = 0): number => (Number.isFinite(v) ? v : fallback);
  if (!(Number.isFinite(attDistance) && attDistance > 0)) return [0, 0, 0];
  const sigmaAChannel = (c: number): number => {
    const t = Math.min(Math.max(finite(c, 1), 1e-4), 1);
    return Math.max(-Math.log(t) / attDistance, 0);
  };
  return [sigmaAChannel(attColor[0]), sigmaAChannel(attColor[1]), sigmaAChannel(attColor[2])];
}

/** pt-webgl2 materialsTexture.ts (pre-T1-1) σ_a derivation. */
function refWebgl2SigmaA(
  attenuationColor: [number, number, number],
  attenuationDistance: number,
): [number, number, number] {
  const finiteOr = (value: number | undefined, fallback: number): number =>
    Number.isFinite(value) ? Number(value) : fallback;
  const ATTENUATION_TRANSMITTANCE_EPSILON = 1e-4;
  if (!Number.isFinite(attenuationDistance) || attenuationDistance <= 0.0) return [0, 0, 0];
  const sigmaAChannel = (channel: number): number => {
    const transmittance = Math.min(
      Math.max(finiteOr(channel, 1.0), ATTENUATION_TRANSMITTANCE_EPSILON),
      1.0,
    );
    return Math.max(-Math.log(transmittance) / attenuationDistance, 0.0);
  };
  return [
    sigmaAChannel(attenuationColor[0]),
    sigmaAChannel(attenuationColor[1]),
    sigmaAChannel(attenuationColor[2]),
  ];
}

/** pt-webgpu materialPacking.ts (pre-T1-1) sampleSpectralCurve. */
function refWebgpuSampleCurve(curve: SpectralCurveLike | null, lambdaNm: number): number {
  if (curve == null || curve.values.length === 0) return 0;
  const start = curve.wavelengthStart;
  const end = curve.wavelengthEnd;
  const denom = Math.max(end - start, 1e-6);
  const t = Math.min(1, Math.max(0, (lambdaNm - start) / denom));
  const f = t * (curve.values.length - 1);
  const i0 = Math.floor(f);
  const i1 = Math.min(i0 + 1, curve.values.length - 1);
  const a = curve.values[i0] ?? 0;
  const b = curve.values[i1] ?? a;
  return a + (b - a) * (f - i0);
}

/** pt-webgl2 materialsTexture.ts (pre-T1-1) sampleSpectralCurve. */
function refWebgl2SampleCurve(curve: SpectralCurveLike | null, lambdaNm: number): number {
  if (!curve) return 0.0;
  const values = curve.values;
  if (!values || values.length < 2) return 0.0;
  const lambdaStart = Number.isFinite(curve.wavelengthStart) ? curve.wavelengthStart : 380.0;
  const lambdaEnd = Number.isFinite(curve.wavelengthEnd) ? curve.wavelengthEnd : 780.0;
  const denom = Math.max(lambdaEnd - lambdaStart, 1e-6);
  const t = Math.min(1.0, Math.max(0.0, (lambdaNm - lambdaStart) / denom));
  const f = t * (values.length - 1);
  const i0 = Math.floor(f);
  const i1 = Math.min(i0 + 1, values.length - 1);
  const a = Number(values[i0] ?? 0.0);
  const b = Number(values[i1] ?? a);
  return a + (b - a) * (f - i0);
}

const WEBGPU_OPTS: SpectralCurveSampleOptions = {};
const WEBGL2_OPTS: SpectralCurveSampleOptions = {
  minValueCount: 2,
  fallbackStartNm: 380.0,
  fallbackEndNm: 780.0,
};

const CURVES: SpectralCurveLike[] = [
  { wavelengthStart: 400, wavelengthEnd: 700, values: [0.1, 0.5, 0.9, 0.3] },
  { wavelengthStart: 380, wavelengthEnd: 780, values: [0.0, 1.0] },
  { wavelengthStart: 450, wavelengthEnd: 650, values: [0.25, 0.75, 0.5, 0.1, 0.9, 0.6] },
  // non-finite bounds — exercises the webgl2 fallback branch
  { wavelengthStart: NaN, wavelengthEnd: NaN, values: [0.2, 0.4, 0.6] },
];

describe('sigmaAFromAttenuation — cross-backend parity', () => {
  const cases: Array<{ color: [number, number, number]; dist: number }> = [
    { color: [0.5, 0.3, 0.9], dist: 1.0 },
    { color: [0.01, 0.5, 1.0], dist: 2.5 },
    { color: [0.0, 0.0, 0.0], dist: 0.5 }, // clamps to epsilon
    { color: [1.0, 1.0, 1.0], dist: 10.0 }, // σ_a = 0
    { color: [0.5, 0.5, 0.5], dist: 0.0 }, // guard → [0,0,0]
    { color: [0.5, 0.5, 0.5], dist: -1.0 }, // guard → [0,0,0]
  ];
  for (const { color, dist } of cases) {
    it(`matches both backends for color=${color} dist=${dist}`, () => {
      const shared = sigmaAFromAttenuation(color, dist);
      expect(shared).toEqual(refWebgpuSigmaA(color, dist));
      expect(shared).toEqual(refWebgl2SigmaA(color, dist));
    });
  }
});

describe('sampleSpectralCurve — reproduces each backend under its options', () => {
  const lambdas = [380, 420, 500, 589.3, 660, 780];
  for (const curve of CURVES) {
    for (const lambda of lambdas) {
      it(`webgpu-parity curve=${JSON.stringify(curve.values)} λ=${lambda}`, () => {
        expect(sampleSpectralCurve(curve, lambda, WEBGPU_OPTS)).toBe(
          refWebgpuSampleCurve(curve, lambda),
        );
      });
      it(`webgl2-parity curve=${JSON.stringify(curve.values)} λ=${lambda}`, () => {
        expect(sampleSpectralCurve(curve, lambda, WEBGL2_OPTS)).toBe(
          refWebgl2SampleCurve(curve, lambda),
        );
      });
    }
  }
  it('null curve → 0 for both option sets', () => {
    expect(sampleSpectralCurve(null, 500, WEBGPU_OPTS)).toBe(0);
    expect(sampleSpectralCurve(undefined, 500, WEBGL2_OPTS)).toBe(0);
  });
  it('single-value curve: webgpu samples it, webgl2 returns 0 (min ≥ 2)', () => {
    const single: SpectralCurveLike = { wavelengthStart: 400, wavelengthEnd: 700, values: [0.7] };
    expect(sampleSpectralCurve(single, 500, WEBGPU_OPTS)).toBe(refWebgpuSampleCurve(single, 500));
    expect(sampleSpectralCurve(single, 500, WEBGL2_OPTS)).toBe(0);
  });
});

describe('sampleSpectralGrid — matches the pt-webgpu 32-sample fold', () => {
  it('grid bounds are the canonical 380/780 nm × 32 samples', () => {
    expect(SPECTRAL_GRID_SAMPLE_COUNT).toBe(32);
    expect(SPECTRAL_GRID_START_NM).toBe(380);
    expect(SPECTRAL_GRID_END_NM).toBe(780);
  });
  for (const curve of CURVES) {
    it(`samples+avg+max+count match webgpu inline for ${JSON.stringify(curve.values)}`, () => {
      const grid = sampleSpectralGrid(curve, WEBGPU_OPTS);
      // Independent reference: webgpu's original inline loop.
      const refSamples = new Array<number>(32).fill(0);
      let sum = 0;
      let maxMu = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < 32; i += 1) {
        const t = i / Math.max(32 - 1, 1);
        const lambda = 380 + t * (780 - 380);
        const v = Math.max(refWebgpuSampleCurve(curve, lambda), 0);
        refSamples[i] = v;
        sum += v;
        maxMu = Math.max(maxMu, v);
      }
      expect(grid.samples).toEqual(refSamples);
      expect(grid.avg).toBe(sum / 32);
      expect(grid.max).toBe(Number.isFinite(maxMu) ? maxMu : 0);
      expect(grid.sampleCount).toBe(32);
    });
  }
  it('no curve → all-zero samples, zero stats', () => {
    const grid = sampleSpectralGrid(null);
    expect(grid.samples).toEqual(new Array<number>(32).fill(0));
    expect(grid.avg).toBe(0);
    expect(grid.max).toBe(0);
    expect(grid.sampleCount).toBe(0);
  });
});

describe('dispersionStrengthFromAbbe — pt-webgl2 fork port', () => {
  const cases: Array<[number, number]> = [
    [1.5, 64],
    [1.62, 36],
    [1.0, 50], // ior ≤ 1 → 0
    [1.5, 0], // abbe ≤ 0 → 0
    [1.8, 20],
  ];
  for (const [ior, abbe] of cases) {
    it(`ior=${ior} abbe=${abbe} matches the fork formula`, () => {
      const F = 486.1;
      const C = 656.3;
      let ref = 0;
      if (!(abbe <= 0 || ior <= 1)) {
        const denom = 1 / (F * F) - 1 / (C * C);
        ref = Math.abs(denom) < 1e-12 ? 0 : Math.max(0, (ior - 1) / (abbe * denom));
      }
      expect(dispersionStrengthFromAbbe(ior, abbe)).toBe(ref);
    });
  }
});

describe('resolveEmissiveIntensity — shared default', () => {
  it('defaults undefined to 1, passes through explicit values', () => {
    expect(resolveEmissiveIntensity(undefined)).toBe(1.0);
    expect(resolveEmissiveIntensity(0)).toBe(0);
    expect(resolveEmissiveIntensity(2.5)).toBe(2.5);
  });
});
