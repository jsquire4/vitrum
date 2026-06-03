/**
 * tonemap.test.ts — P4 output tonemap operators + WGSL-twin shape pin.
 */
import { describe, it, expect } from 'vitest';
import { acesFilmic, reinhard, agx, applyTonemap, TONEMAP_MODE_INDEX } from '../tonemap.js';
import { tonemapWgsl } from '../wgsl/tonemap.wgsl.js';

describe('tonemap operators (P4)', () => {
  it('none vs linear: exposure scales; none is unclamped, linear clamps', () => {
    expect(applyTonemap([0.5, 0.5, 0.5], 'none', 2)).toEqual([1, 1, 1]);
    expect(applyTonemap([0.6, 0.6, 0.6], 'none', 2)).toEqual([1.2, 1.2, 1.2]);
    expect(applyTonemap([0.6, 0.6, 0.6], 'linear', 2)).toEqual([1, 1, 1]);
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
    expect(w).toContain('vt_aces');
    expect(w).toContain('vt_agx');
    expect(w).toMatch(/mode == 1u/);
    expect(w).toMatch(/mode == 4u/);
  });
});
