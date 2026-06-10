/**
 * Contract-honesty tests for environmentPacking.ts:
 *   1. RGBA input (stride 4) is decoded correctly — RGB-only decode garbles values.
 *   2. All-black HDRI emits an accurate warning rather than the misleading
 *      "lacks CPU pixel data" message.
 */
import { describe, expect, it } from 'vitest';
import { environmentParams } from '../scene/environmentPacking.js';
import type { Scene } from '@vitrum/core';

function makeHdriScene(data: ArrayLike<number>, width: number, height: number): Scene {
  return {
    primitives: [],
    emitters: [],
    environment: {
      kind: 'hdri',
      hdri: { width, height, data } as unknown as Scene['environment'] & object,
    } as Scene['environment'],
  };
}

describe('environmentPacking — RGBA stride detection', () => {
  it('decodes a w×h×3 RGB array correctly (stride 3)', () => {
    const width = 2;
    const height = 2;
    // 4 pixels: red, green, blue, white — tightly packed at stride 3
    const data = new Float32Array([
      1, 0, 0, // px 0 red
      0, 1, 0, // px 1 green
      0, 0, 1, // px 2 blue
      1, 1, 1, // px 3 white
    ]);
    const p = environmentParams(makeHdriScene(data, width, height));
    expect(p.hasHdri).toBe(true);
    // First pixel (red): texels[0..2] should be 1,0,0
    expect(p.hdriTexels[0]).toBeCloseTo(1);
    expect(p.hdriTexels[1]).toBeCloseTo(0);
    expect(p.hdriTexels[2]).toBeCloseTo(0);
    // Second pixel (green): texels[4..6] should be 0,1,0
    expect(p.hdriTexels[4]).toBeCloseTo(0);
    expect(p.hdriTexels[5]).toBeCloseTo(1);
    expect(p.hdriTexels[6]).toBeCloseTo(0);
    // No RGBA stride-ambiguity warning for pure-RGB input
    expect(p.warnings.some((w) => w.includes('RGBA'))).toBe(false);
  });

  it('decodes a w×h×4 RGBA array at stride 4 (not misread at stride 3)', () => {
    const width = 2;
    const height = 1;
    // 2 pixels at stride 4: red + white (with alpha channel)
    // If decoded at stride 3 the second pixel would start at index 3 (alpha lane)
    // and read [alpha=0.5, R_px1=1, G_px1=0] → green channel would be 1, wrong.
    const data = new Float32Array([
      1, 0, 0, 0.5, // px 0 red with alpha 0.5
      0, 1, 0, 1.0, // px 1 green with alpha 1.0
    ]);
    const p = environmentParams(makeHdriScene(data, width, height));
    expect(p.hasHdri).toBe(true);
    // Pixel 0 should decode as red (1,0,0)
    expect(p.hdriTexels[0]).toBeCloseTo(1);
    expect(p.hdriTexels[1]).toBeCloseTo(0);
    expect(p.hdriTexels[2]).toBeCloseTo(0);
    // Pixel 1 should decode as green (0,1,0) — NOT the stride-3 misread (0.5,1,0)
    expect(p.hdriTexels[4]).toBeCloseTo(0);
    expect(p.hdriTexels[5]).toBeCloseTo(1);
    expect(p.hdriTexels[6]).toBeCloseTo(0);
    // Should emit the RGBA stride warning
    expect(p.warnings.some((w) => w.includes('RGBA'))).toBe(true);
  });
});

describe('environmentPacking — all-black HDRI message', () => {
  it('emits a zero-luminance warning (not a "lacks pixel data" message) for an all-black HDRI', () => {
    const width = 2;
    const height = 2;
    // 4 pixels, all zero — totalWeight will be 0 (≤ 1e-12)
    const data = new Float32Array(width * height * 3);
    const p = environmentParams(makeHdriScene(data, width, height));
    expect(p.hasHdri).toBe(false);
    // Must NOT blame missing pixel data
    expect(p.warnings.some((w) => w.includes('lacks CPU pixel data'))).toBe(false);
    // Must accurately report zero luminance
    expect(p.warnings.some((w) => w.includes('zero total luminance') || w.includes('all-black'))).toBe(true);
  });

  it('emits the "lacks CPU pixel data" warning only when data is genuinely absent', () => {
    const scene: Scene = {
      primitives: [],
      emitters: [],
      environment: {
        kind: 'hdri',
        hdri: {} as unknown as Scene['environment'] & object,
      } as Scene['environment'],
    };
    const p = environmentParams(scene);
    expect(p.hasHdri).toBe(false);
    expect(p.warnings.some((w) => w.includes('lacks CPU pixel data'))).toBe(true);
  });
});
