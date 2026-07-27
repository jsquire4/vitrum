import { describe, expect, it } from 'vitest';
import { bdptHgPhaseCpu } from '../bdpt/bdptMediumTransportCpu.js';
import { PT_WEBGPU_PATH_TRACE_HG_PHASE_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';

const INV_4PI = 1 / (4 * Math.PI);

function clampHg(g: number): number {
  return Math.max(-0.999999, Math.min(0.999999, g));
}

function stableHgPhase(cosThetaRaw: number, gRaw: number): number {
  const g = clampHg(gRaw);
  const a = Math.abs(g);
  const cosTheta = Math.max(-1, Math.min(1, cosThetaRaw));
  const alignedCos = g >= 0 ? cosTheta : -cosTheta;
  const oneMinusA = 1 - a;
  const denom = oneMinusA * oneMinusA + 2 * a * (1 - alignedCos);
  return (
    (INV_4PI * oneMinusA * (1 + a)) /
    (denom * Math.sqrt(denom))
  );
}

function stableHgSampleCos(gRaw: number, u: number): number {
  const g = clampHg(gRaw);
  const a = Math.abs(g);
  const q = 1 - 2 * u;
  let alignedCos: number;
  if (a < 0.125) {
    const d = 1 + a * q;
    const numerator =
      2 * q +
      a * (q * q + 3) +
      2 * a * a * q +
      a * a * a * (q * q - 1);
    alignedCos = numerator / (2 * d * d);
  } else {
    const oneMinusA = 1 - a;
    const ratio =
      (oneMinusA * (1 + a)) / (oneMinusA + 2 * a * (1 - u));
    alignedCos = (1 + a * a - ratio * ratio) / (2 * a);
  }
  const signed = g >= 0 ? alignedCos : -alignedCos;
  return Math.max(-1, Math.min(1, signed));
}

function stableHgNormalization(gRaw: number): number {
  const a = Math.abs(clampHg(gRaw));
  if (a === 0) return 1;
  const oneMinusA = 1 - a;
  const onePlusA = 1 + a;
  // 2π times the analytic integral over cos(theta), evaluated with the same
  // factored endpoint terms as the shader.
  return (
    (0.5 * oneMinusA * onePlusA *
      (1 / oneMinusA - 1 / onePlusA)) /
    a
  );
}

describe('Henyey-Greenstein extreme-anisotropy stability', () => {
  it('keeps the near-delta lobe finite without flattening it', () => {
    for (const g of [-0.999999, -0.9, 0, 0.9, 0.999999]) {
      for (const cosTheta of [-1, -0.5, 0, 0.5, 1]) {
        const phase = stableHgPhase(cosTheta, g);
        expect(Number.isFinite(phase)).toBe(true);
        expect(phase).toBeGreaterThan(0);
      }
    }

    expect(stableHgPhase(1, 0.999999)).toBeGreaterThan(1e10);
    expect(stableHgPhase(-1, -0.999999)).toBeGreaterThan(1e10);
    expect(stableHgPhase(1, 0.999999)).toBeCloseTo(
      stableHgPhase(-1, -0.999999),
      4,
    );
    for (const g of [-0.999999, -0.9, 0, 0.9, 0.999999]) {
      for (const cosTheta of [-1, -0.5, 0, 0.5, 1]) {
        const expected = stableHgPhase(cosTheta, g);
        expect(bdptHgPhaseCpu(cosTheta, g) / expected).toBeCloseTo(1, 12);
      }
    }
  });

  it('normalizes for forward and backward lobes through the supported limit', () => {
    for (const g of [-0.999999, -0.9, -1e-8, 0, 1e-8, 0.9, 0.999999]) {
      expect(stableHgNormalization(g)).toBeCloseTo(1, 8);
    }
  });

  it('preserves every non-zero anisotropy in the exact inverse sampler', () => {
    const sampleCount = 100_000;
    for (const g of [-0.999999, -0.9, -1e-6, 0, 1e-6, 0.9, 0.999999]) {
      let mean = 0;
      let minimum = 1;
      let maximum = -1;
      let allFinite = true;
      for (let i = 0; i < sampleCount; i += 1) {
        const cosTheta = stableHgSampleCos(g, (i + 0.5) / sampleCount);
        allFinite &&= Number.isFinite(cosTheta);
        minimum = Math.min(minimum, cosTheta);
        maximum = Math.max(maximum, cosTheta);
        mean += cosTheta;
      }
      expect(allFinite).toBe(true);
      expect(minimum).toBeGreaterThanOrEqual(-1);
      expect(maximum).toBeLessThanOrEqual(1);
      expect(mean / sampleCount).toBeCloseTo(g, 5);
    }
  });

  it('pins the cancellation-free shader formulas and rejects semantic floors', () => {
    expect(PT_WEBGPU_PATH_TRACE_HG_PHASE_WGSL).toContain(
      'oneMinusA * oneMinusA + 2.0 * a * (1.0 - alignedCos)',
    );
    expect(PT_WEBGPU_PATH_TRACE_HG_PHASE_WGSL).toContain(
      'oneMinusA + 2.0 * a * (1.0 - u1)',
    );
    expect(PT_WEBGPU_PATH_TRACE_HG_PHASE_WGSL).toContain(
      'a * a * a * (q * q - 1.0)',
    );
    expect(PT_WEBGPU_PATH_TRACE_HG_PHASE_WGSL).not.toContain(
      'max(pow(denom, 1.5), 1e-9)',
    );
    expect(PT_WEBGPU_PATH_TRACE_HG_PHASE_WGSL).not.toContain(
      'abs(g) < 1e-3',
    );
  });
});
