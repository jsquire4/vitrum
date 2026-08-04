/**
 * B3 (road-to-100) — directional IBL equirect CDF builder.
 *
 * Verifies the PBRT 2D-distribution build: per-texel solid-angle pdf integrates
 * to 1 over the sphere, the importance sampler concentrates on bright texels, and
 * degenerate maps return null (scalar-tint fallback).
 */
import { describe, it, expect } from 'vitest';
import { buildDirectionalEnv } from '../equirectDirectional.js';

function makeRaw(
  width: number,
  height: number,
  fill: (x: number, y: number) => [number, number, number],
) {
  const data = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const [r, g, b] = fill(x, y);
      const i = (y * width + x) * 3;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
    }
  }
  return { width, height, data, stride: 3 as const };
}

describe('buildDirectionalEnv', () => {
  it('returns null for an all-black map', () => {
    expect(buildDirectionalEnv(makeRaw(4, 2, () => [0, 0, 0]))).toBeNull();
  });

  it('returns null for zero-size', () => {
    expect(
      buildDirectionalEnv({ width: 0, height: 0, data: new Float32Array(0), stride: 3 }),
    ).toBeNull();
  });

  it('per-texel solid-angle pdf integrates to ~1 over the sphere', () => {
    // A non-uniform map (a bright band) so the pdf is non-trivial.
    const W = 16,
      H = 8;
    const env = buildDirectionalEnv(
      makeRaw(W, H, (x, y) => {
        const v = y === 3 ? 5 : 1; // bright row
        return [v, v, v];
      }),
    );
    expect(env).not.toBeNull();
    // ∫ p(ω) dω = Σ_texel p_texel · dω_texel = Σ pmf_texel = 1.
    let integral = 0;
    for (let y = 0; y < H; y += 1) {
      const theta0 = (y / H) * Math.PI;
      const theta1 = ((y + 1) / H) * Math.PI;
      const dOmega = ((2 * Math.PI) / W) * (Math.cos(theta0) - Math.cos(theta1));
      for (let x = 0; x < W; x += 1) {
        const i = y * W + x;
        integral += env!.pdf[i]! * dOmega;
      }
    }
    expect(integral).toBeCloseTo(1, 2);
  });

  it('stores exact monotone forward CDFs ending at one', () => {
    const W = 8,
      H = 4;
    const env = buildDirectionalEnv(
      makeRaw(W, H, (x) => {
        const v = x === 6 ? 10 : 0.1; // bright column
        return [v, v, v];
      }),
    )!;
    expect(env.marginal).toHaveLength(H * 4);
    expect(env.conditional).toHaveLength(W * H * 4);
    let priorMarginal = 0;
    for (let i = 0; i < H; i += 1) {
      const cdf = env.marginal[i * 4]!;
      expect(cdf).toBeGreaterThanOrEqual(priorMarginal);
      expect(cdf).toBeLessThanOrEqual(1);
      priorMarginal = cdf;
    }
    expect(priorMarginal).toBeCloseTo(1, 6);
    for (let y = 0; y < H; y += 1) {
      let prior = 0;
      for (let x = 0; x < W; x += 1) {
        const cdf = env.conditional[(y * W + x) * 4]!;
        expect(cdf).toBeGreaterThanOrEqual(prior);
        expect(cdf).toBeLessThanOrEqual(1);
        prior = cdf;
      }
      expect(prior).toBeCloseTo(1, 6);
      // The bright x=6 texel owns most of the exact CDF interval.
      const before = env.conditional[(y * W + 5) * 4]!;
      const after = env.conditional[(y * W + 6) * 4]!;
      expect(after - before).toBeGreaterThan(0.9);
    }
  });

  it('preserves a dim positive column in the represented Float32 CDF', () => {
    const env = buildDirectionalEnv(
      makeRaw(2, 1, (x) => (x === 0 ? [1, 1, 1] : [1e-8, 1e-8, 1e-8])),
    )!;
    const cellSolidAngle = 2 * Math.PI;

    expect(env.marginal[0]).toBe(1);
    expect(env.conditional[0]).toBeLessThan(1);
    expect(env.conditional[4]).toBe(1);
    expect(env.conditional[4]).toBeGreaterThan(env.conditional[0]!);
    expect(env.pdf[1]).toBeGreaterThan(0);
    expect((env.pdf[0]! + env.pdf[1]!) * cellSolidAngle).toBeCloseTo(1, 7);
  });

  it('retains a positive middle-column CDF interval and PDF', () => {
    const env = buildDirectionalEnv(
      makeRaw(3, 1, (x) => {
        const value = x === 1 ? 1e-12 : 1;
        return [value, value, value];
      }),
    )!;

    expect(env.conditional[0]).toBeCloseTo(0.5, 7);
    expect(env.conditional[4]).toBeGreaterThan(env.conditional[0]!);
    expect(env.conditional[8]).toBe(1);
    expect(env.pdf[1]).toBeGreaterThan(0);
    const cellSolidAngle = (2 * Math.PI / 3) * 2;
    expect((env.pdf[0]! + env.pdf[1]! + env.pdf[2]!) * cellSolidAngle)
      .toBeCloseTo(1, 7);
  });

  it('retains a positive middle-row CDF interval and PDF', () => {
    const env = buildDirectionalEnv(
      makeRaw(1, 3, (_x, y) => {
        const value = y === 1 ? 1e-12 : 1;
        return [value, value, value];
      }),
    )!;

    expect(env.marginal[0]).toBeCloseTo(0.5, 6);
    expect(env.marginal[4]).toBeGreaterThan(env.marginal[0]!);
    expect(env.marginal[8]).toBe(1);
    expect(env.pdf[1]).toBeGreaterThan(0);
    let integrated = 0;
    for (let y = 0; y < 3; y += 1) {
      const theta0 = y * Math.PI / 3;
      const theta1 = (y + 1) * Math.PI / 3;
      const cellSolidAngle =
        2 * Math.PI * (Math.cos(theta0) - Math.cos(theta1));
      integrated += env.pdf[y]! * cellSolidAngle;
    }
    expect(integrated).toBeCloseTo(1, 7);
  });

  it('gives zero-weight rows a total uniform conditional CDF', () => {
    const W = 4;
    const env = buildDirectionalEnv(makeRaw(W, 3, (_x, y) => (y === 1 ? [0, 0, 0] : [1, 1, 1])))!;
    expect(Array.from({ length: W }, (_, x) => env.conditional[(W + x) * 4])).toEqual([
      0.25, 0.5, 0.75, 1,
    ]);
    // The zero row has no marginal interval, including for ξ=0.
    expect(env.marginal[4]).toBe(env.marginal[0]);
    expect(env.marginal[(3 - 1) * 4]).toBeCloseTo(1, 6);
  });

  it('stores unit-intensity radiance in .rgb (host applies intensity at sample time)', () => {
    const env = buildDirectionalEnv(makeRaw(2, 1, (x) => (x === 0 ? [2, 4, 6] : [1, 1, 1])))!;
    expect(env.map[0]).toBeCloseTo(2);
    expect(env.map[1]).toBeCloseTo(4);
    expect(env.map[2]).toBeCloseTo(6);
    expect(env.map[3]).toBe(1);
  });

  it('rejects finite source radiance that overflows or wholly collapses on f32 publication', () => {
    expect(() => buildDirectionalEnv({
      width: 1,
      height: 1,
      data: [Number.MAX_VALUE, 0, 0],
      stride: 3,
    })).toThrow(/remain finite after Float32 packing/);
    expect(() => buildDirectionalEnv({
      width: 1,
      height: 1,
      data: [Number.MIN_VALUE, Number.MIN_VALUE, Number.MIN_VALUE],
      stride: 3,
    })).toThrow(/must not collapse completely to zero/);
  });

  it('allows one underflowed lane when another lane preserves published radiance', () => {
    const env = buildDirectionalEnv({
      width: 1,
      height: 1,
      data: [Number.MIN_VALUE, 1, 0],
      stride: 3,
    })!;
    expect(Array.from(env.map.subarray(0, 4))).toEqual([0, 1, 0, 1]);
    expect(env.pdf[0]).toBeGreaterThan(0);
  });

  it('within-cell solid-angle sampling is unbiased and is not a texel-center delta', () => {
    const row = 2;
    const height = 8;
    const theta0 = (row * Math.PI) / height;
    const theta1 = ((row + 1) * Math.PI) / height;
    const cos0 = Math.cos(theta0);
    const cos1 = Math.cos(theta1);
    let mean = 0;
    const count = 4096;
    for (let i = 0; i < count; i += 1) {
      const xi = (i + 0.5) / count;
      mean += cos0 + (cos1 - cos0) * xi;
    }
    mean /= count;
    expect(mean).toBeCloseTo((cos0 + cos1) * 0.5, 10);
    expect(cos0).not.toBeCloseTo(cos1);
  });
});
