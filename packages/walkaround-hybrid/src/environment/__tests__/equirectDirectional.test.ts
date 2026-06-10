/**
 * B3 (road-to-100) — directional IBL equirect CDF builder.
 *
 * Verifies the PBRT 2D-distribution build: per-texel solid-angle pdf integrates
 * to 1 over the sphere, the importance sampler concentrates on bright texels, and
 * degenerate maps return null (scalar-tint fallback).
 */
import { describe, it, expect } from 'vitest';
import { buildDirectionalEnv } from '../equirectDirectional.js';

function makeRaw(width: number, height: number, fill: (x: number, y: number) => [number, number, number]) {
  const data = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const [r, g, b] = fill(x, y);
      const i = (y * width + x) * 3;
      data[i] = r; data[i + 1] = g; data[i + 2] = b;
    }
  }
  return { width, height, data, stride: 3 as const };
}

describe('buildDirectionalEnv', () => {
  it('returns null for an all-black map', () => {
    expect(buildDirectionalEnv(makeRaw(4, 2, () => [0, 0, 0]))).toBeNull();
  });

  it('returns null for zero-size', () => {
    expect(buildDirectionalEnv({ width: 0, height: 0, data: new Float32Array(0), stride: 3 })).toBeNull();
  });

  it('per-texel solid-angle pdf integrates to ~1 over the sphere', () => {
    // A non-uniform map (a bright band) so the pdf is non-trivial.
    const W = 16, H = 8;
    const env = buildDirectionalEnv(makeRaw(W, H, (x, y) => {
      const v = y === 3 ? 5 : 1; // bright row
      return [v, v, v];
    }));
    expect(env).not.toBeNull();
    // ∫ p(ω) dω = Σ_texel p_texel · dω_texel = Σ pmf_texel = 1.
    let integral = 0;
    for (let y = 0; y < H; y += 1) {
      const theta = ((y + 0.5) / H) * Math.PI;
      const sinTheta = Math.max(Math.sin(theta), 1e-5);
      const dOmega = ((2 * Math.PI) / W) * (Math.PI / H) * sinTheta;
      for (let x = 0; x < W; x += 1) {
        const i = y * W + x;
        integral += env!.map[i * 4 + 3]! * dOmega;
      }
    }
    expect(integral).toBeCloseTo(1, 2);
  });

  it('marginal + conditional are valid inverse-CDF tables (centred coords in [0,1))', () => {
    const W = 8, H = 4;
    const env = buildDirectionalEnv(makeRaw(W, H, (x) => {
      const v = x === 6 ? 10 : 0.1; // bright column
      return [v, v, v];
    }))!;
    expect(env.marginal).toHaveLength(H * 4);
    expect(env.conditional).toHaveLength(W * H * 4);
    for (let i = 0; i < H; i += 1) {
      const v = env.marginal[i * 4]!;
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
    // The bright column (x=6) should be where the conditional concentrates: the
    // sampled column for a mid-random in each row maps near (6+0.5)/8.
    for (let y = 0; y < H; y += 1) {
      const u = env.conditional[(y * W + Math.floor(W / 2)) * 4]!;
      expect(u).toBeCloseTo((6 + 0.5) / W, 5);
    }
  });

  it('stores unit-intensity radiance in .rgb (host applies intensity at sample time)', () => {
    const env = buildDirectionalEnv(makeRaw(2, 1, (x) => (x === 0 ? [2, 4, 6] : [1, 1, 1])))!;
    expect(env.map[0]).toBeCloseTo(2);
    expect(env.map[1]).toBeCloseTo(4);
    expect(env.map[2]).toBeCloseTo(6);
  });
});
