/**
 * tonemap.test.ts — P4 output tonemap operators + WGSL-twin shape pin.
 */
import { describe, it, expect } from 'vitest';
import {
  acesFilmic,
  reinhard,
  agx,
  applyTonemap,
  TONEMAP_MAX_FINITE_F32,
  TONEMAP_MODE_INDEX,
  linearToSrgb,
  srgbToLinear,
} from '../tonemap.js';
import { tonemapWgsl } from '../wgsl/tonemap.wgsl.js';

describe('tonemap operators (P4)', () => {
  it('none vs linear: exposure scales; none is unclamped, linear clamps', () => {
    expect(applyTonemap([0.5, 0.5, 0.5], 'none', 2)).toEqual([1, 1, 1]);
    expect(applyTonemap([0.6, 0.6, 0.6], 'none', 2)).toEqual([1.2, 1.2, 1.2]);
    expect(applyTonemap([0.6, 0.6, 0.6], 'linear', 2)).toEqual([1, 1, 1]);
  });

  it('preserves raw HDR above half-float range and saturates only at finite f32', () => {
    expect(
      applyTonemap([65_504, 32_752, -65_504], 'none', 2),
    ).toEqual([131_008, 65_504, -131_008]);

    expect(
      applyTonemap([2, 1, 0.5], 'none', TONEMAP_MAX_FINITE_F32),
    ).toEqual([
      TONEMAP_MAX_FINITE_F32,
      TONEMAP_MAX_FINITE_F32,
      TONEMAP_MAX_FINITE_F32 * 0.5,
    ]);
    for (const mode of ['aces', 'agx', 'reinhard', 'linear'] as const) {
      expect(
        applyTonemap([2, 1, 0.5], mode, TONEMAP_MAX_FINITE_F32).every(
          Number.isFinite,
        ),
      ).toBe(true);
    }
    expect(() => applyTonemap([1, 1, 1], 'none', -1)).toThrow(
      /exposure must be non-negative/,
    );
  });

  it('rejects inherited object keys as unsupported modes deterministically', () => {
    for (const inheritedKey of ['toString', '__proto__']) {
      expect(() =>
        applyTonemap(
          [0.5, 0.5, 0.5],
          inheritedKey as Parameters<typeof applyTonemap>[1],
        ),
      ).toThrowError(
        new RangeError(`applyTonemap.mode is unsupported: ${inheritedKey}`),
      );
    }
  });

  it('aces: 0→0, bounded ≤1, monotonic', () => {
    expect(acesFilmic(0)).toBeCloseTo(0, 6);
    expect(acesFilmic(1e6)).toBeLessThanOrEqual(1);
    expect(acesFilmic(0.5)).toBeGreaterThan(acesFilmic(0.2));
  });

  it('reinhard: x/(1+x)', () => {
    expect(reinhard(0)).toBe(0);
    expect(reinhard(1)).toBeCloseTo(0.5);
    expect(reinhard(3)).toBeCloseTo(0.75);
    expect(reinhard(-0.5)).toBe(0);
  });

  it('agx: bounded [0,1], increasing across the mid-range', () => {
    expect(agx(0)).toBeGreaterThanOrEqual(0);
    expect(agx(1e6)).toBeLessThanOrEqual(1);
    expect(agx(1)).toBeGreaterThan(agx(0.05));
  });

  it('mode indices match the WGSL selector contract', () => {
    expect(TONEMAP_MODE_INDEX).toEqual({ aces: 0, agx: 1, reinhard: 2, linear: 3, none: 4 });
  });

  it('WGSL twin emits vitrumTonemap with all mode branches', () => {
    const w = tonemapWgsl();
    expect(w).toContain('fn vitrumTonemap(color: vec3f, mode: u32, exposure: f32)');
    expect(w).toContain('fn vt_safeExposure(color: vec3f, exposure: f32)');
    expect(w).toContain('let x = vt_safeExposure(color, exposure);');
    expect(w).toContain('let boundedExposure = min(exposure, VT_MAX_FINITE_F32);');
    expect(w).toContain(
      'let magnitude = min(abs(channel) * boundedExposure, VT_MAX_FINITE_F32);',
    );
    expect(w).not.toContain('VT_MAX_PRESENT_VALUE');
    expect(w).toContain('vt_aces');
    expect(w).toContain('vt_agx');
    expect(w).toMatch(/mode == 1u/);
    expect(w).toMatch(/mode == 4u/);
    expect(w).toContain('vt_linearToSrgb');
    expect(w).toContain('let v = max(x, vec3f(0.0));');
    expect(w).toContain('return v / (1.0 + v);');
    expect(w).not.toContain('return x / (1.0 + max(x');
  });

  it('sRGB OETF: 0→0, 1→1, ~0.735 at 0.5; round-trips with srgbToLinear', () => {
    expect(linearToSrgb(0)).toBeCloseTo(0, 6);
    expect(linearToSrgb(1)).toBeCloseTo(1, 6);
    expect(linearToSrgb(0.5)).toBeCloseTo(0.7353569, 4);
    for (const x of [0.001, 0.05, 0.2, 0.8]) {
      expect(srgbToLinear(linearToSrgb(x))).toBeCloseTo(x, 5);
    }
  });
});
