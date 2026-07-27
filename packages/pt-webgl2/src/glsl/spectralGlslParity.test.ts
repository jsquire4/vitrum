import { describe, expect, it } from 'vitest';
import {
  CIE_LAMBDA_MAX,
  CIE_LAMBDA_MIN,
  CIE_LAMBDA_STEP,
  CIE_TABLE_LENGTH,
  CIE_X_TABLE,
  CIE_Y_TABLE,
  CIE_Z_TABLE,
  X_CMF_CDF,
  X_CMF_INTEGRAL,
  Y_CMF_CDF,
  Y_CMF_INTEGRAL,
  Z_CMF_CDF,
  Z_CMF_INTEGRAL,
} from '@vitrum/shared-samplers';
import { _sampleCmfCdfInverseForTest } from '../../../shared-samplers/src/wavelengthSampling.js';
import * as SpectralAccumulatorModule from './shader/bsdf/spectral_accumulator.glsl.js';

function requireSpectralAccumulatorSource(): string {
  const source = (SpectralAccumulatorModule as unknown as Record<string, string>)
    .spectral_accumulator;
  if (source === undefined) throw new Error('spectral_accumulator GLSL export is missing');
  return source;
}

const spectral_accumulator = requireSpectralAccumulatorSource();

interface InverseResult {
  readonly lambdaNm: number;
  readonly pdf: number;
  readonly lo: number;
  readonly t: number;
}

const f32 = Math.fround;

function shaderFloat32Inverse(
  uInput: number,
  table: Float32Array,
  cdf: Float32Array,
  integralInput: number,
): InverseResult {
  const integral = f32(integralInput);
  const u = f32(Math.max(uInput, 0));
  if (u >= 1) {
    return {
      lambdaNm: CIE_LAMBDA_MAX,
      pdf: f32(table[CIE_TABLE_LENGTH - 1]! / integral),
      lo: CIE_TABLE_LENGTH - 2,
      t: 1,
    };
  }

  let lo = 0;
  let hi = CIE_TABLE_LENGTH - 2;
  for (let iter = 0; iter < 7; iter++) {
    const mid = (lo + hi) >> 1;
    if (cdf[mid + 1]! <= u) lo = mid + 1;
    else hi = mid;
  }

  const cdfLo = cdf[lo]!;
  const cdfHi = cdf[lo + 1]!;
  const vLo = table[lo]!;
  const vHi = table[lo + 1]!;
  const segmentFraction = cdfHi > cdfLo
    ? f32(f32(u - cdfLo) / f32(cdfHi - cdfLo))
    : 0;
  const segmentIntegral = f32(0.5 * f32(vLo + vHi));
  const targetIntegral = f32(segmentFraction * segmentIntegral);
  const slope = f32(vHi - vLo);
  let t = 0;
  if (segmentIntegral > 0) {
    const threshold = f32(1e-7 * Math.max(Math.abs(vLo), Math.abs(vHi), 1e-30));
    if (Math.abs(slope) <= threshold) {
      t = f32(targetIntegral / vLo);
    } else {
      const discriminant = f32(Math.max(
        f32(vLo * vLo) + f32(f32(2 * slope) * targetIntegral),
        0,
      ));
      const denominator = f32(vLo + f32(Math.sqrt(discriminant)));
      t = denominator > 0 ? f32(f32(2 * targetIntegral) / denominator) : 0;
    }
  }
  t = f32(Math.min(1, Math.max(0, t)));
  const lambdaNm = f32(
    f32(CIE_LAMBDA_MIN + lo * CIE_LAMBDA_STEP) + f32(t * CIE_LAMBDA_STEP),
  );
  const pdf = f32(f32(vLo + f32(t * slope)) / integral);
  return { lambdaNm, pdf, lo, t };
}

describe('spectral GLSL Float32 parity', () => {
  const distributions = [
    ['X', CIE_X_TABLE, X_CMF_CDF, X_CMF_INTEGRAL],
    ['Y', CIE_Y_TABLE, Y_CMF_CDF, Y_CMF_INTEGRAL],
    ['Z', CIE_Z_TABLE, Z_CMF_CDF, Z_CMF_INTEGRAL],
  ] as const;

  it('matches the production CPU inverse/PDF at uploaded knots, tails, and representable interiors', () => {
    let intervalsWithoutRepresentableInterior = 0;

    for (const [name, tableReadonly, cdfReadonly, integral] of distributions) {
      const table = Float32Array.from(tableReadonly);
      const cdf = Float32Array.from(cdfReadonly);
      const cpuCdf = Float64Array.from(cdf);
      const uCases = new Set<number>([
        0,
        f32(1 - 2 ** -24),
        1,
      ]);

      for (let i = 0; i < CIE_TABLE_LENGTH - 1; i++) {
        const cdfLo = cdf[i]!;
        const cdfHi = cdf[i + 1]!;
        uCases.add(cdfLo);
        uCases.add(cdfHi);
        const midpoint = f32(0.5 * f32(cdfLo + cdfHi));
        if (midpoint > cdfLo && midpoint < cdfHi) uCases.add(midpoint);
        else intervalsWithoutRepresentableInterior++;
      }

      for (const u of uCases) {
        const gpu = shaderFloat32Inverse(u, table, cdf, integral);
        const cpu = _sampleCmfCdfInverseForTest(
          u,
          table,
          cpuCdf,
          f32(integral),
        );
        expect(gpu.lambdaNm, `${name} lambda at u=${u}`).toBeCloseTo(cpu.lambdaNm, 3);
        expect(gpu.pdf, `${name} pdf at u=${u}`).toBeCloseTo(cpu.pdf, 7);

        const expectedLo = u >= 1
          ? CIE_TABLE_LENGTH - 2
          : Array.from({ length: CIE_TABLE_LENGTH - 1 }, (_, i) => i)
              .find((i) => cdf[i + 1]! > f32(u)) ?? CIE_TABLE_LENGTH - 2;
        expect(gpu.lo, `${name} segment at u=${u}`).toBe(expectedLo);
      }
    }

    expect(intervalsWithoutRepresentableInterior).toBeGreaterThan(0);
  });

  it('pins the production GLSL inverse branches used by the numerical mirror', () => {
    const inverse = spectral_accumulator.slice(
      spectral_accumulator.indexOf('float sampleCmfCdfInverse('),
      spectral_accumulator.indexOf('// Mixture pdf evaluated'),
    );
    expect(inverse).toContain('float uClamped = max( u, 0.0 );');
    expect(inverse).toContain('if ( uClamped >= 1.0 )');
    expect(inverse).toContain('outLo = 79;');
    expect(inverse).toContain('outT = 1.0;');
    expect(inverse).toContain('int hi = 79;');
    expect(inverse).toContain('bool nearConstant = abs( slope ) <= 1e-7');
    expect(inverse).toContain('2.0 * targetIntegral / denominator');
    expect(inverse).not.toContain('1.0 - 1e-7');
  });
});
