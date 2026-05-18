import { describe, expect, it } from 'vitest';
import { assertAtrousVarianceWebGPUBufferShapes } from '../src/atrousVarianceWebGPU.js';

// ── CPU-mirror of the albedo demodulate/remodulate helpers ────────────────────
// These are plain-JS mirrors of the private helpers in atrousVarianceWebGPU.ts.
// They are tested here independently; the production path is exercised by
// runAtrousVarianceWebGPU when opts.albedoRgb is supplied.

function demodulateAlbedo(
  rgb: Float32Array,
  albedo: Float32Array,
  pixelCount: number,
): Float32Array {
  const out = new Float32Array(rgb.length);
  for (let i = 0; i < pixelCount; i += 1) {
    const si = i * 3;
    const ar = Math.max(albedo[si]!, 1e-3);
    const ag = Math.max(albedo[si + 1]!, 1e-3);
    const ab = Math.max(albedo[si + 2]!, 1e-3);
    out[si] = rgb[si]! / ar;
    out[si + 1] = rgb[si + 1]! / ag;
    out[si + 2] = rgb[si + 2]! / ab;
  }
  return out;
}

function remodulateAlbedo(
  filtered: Float32Array,
  albedo: Float32Array,
  pixelCount: number,
): Float32Array {
  const out = new Float32Array(filtered.length);
  for (let i = 0; i < pixelCount; i += 1) {
    const si = i * 3;
    out[si] = filtered[si]! * albedo[si]!;
    out[si + 1] = filtered[si + 1]! * albedo[si + 1]!;
    out[si + 2] = filtered[si + 2]! * albedo[si + 2]!;
  }
  return out;
}

describe('assertAtrousVarianceWebGPUBufferShapes', () => {
  const minimal = {
    rgb: new Float32Array(12),
    width: 2,
    height: 2,
  };

  it('accepts rgb-only payloads', () => {
    expect(() => assertAtrousVarianceWebGPUBufferShapes(minimal)).not.toThrow();
  });

  it('throws when rgb is undersized', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({ rgb: new Float32Array(8), width: 2, height: 2 }),
    ).toThrow(/rgb/);
  });

  it('throws when prevRadianceRgb is undersized', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...minimal,
        prevRadianceRgb: new Float32Array(11),
      }),
    ).toThrow(/prevRadianceRgb/);
  });

  it('throws when gbufferNormalsRgb is undersized', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...minimal,
        gbufferNormalsRgb: new Float32Array(11),
      }),
    ).toThrow(/gbufferNormalsRgb/);
  });

  it('throws when linearDepth is undersized', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...minimal,
        linearDepth: new Float32Array(3),
      }),
    ).toThrow(/linearDepth/);
  });

  it('throws when motionRg is undersized', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...minimal,
        motionRg: new Float32Array(7),
      }),
    ).toThrow(/motionRg/);
  });

  it('throws when welfordMeanM2 is undersized', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...minimal,
        welfordMeanM2: new Float32Array(7),
      }),
    ).toThrow(/welfordMeanM2/);
  });

  it('accepts fully populated slices', () => {
    const px = 4;
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        rgb: new Float32Array(px * 3),
        width: 2,
        height: 2,
        prevRadianceRgb: new Float32Array(px * 3),
        gbufferNormalsRgb: new Float32Array(px * 3),
        linearDepth: new Float32Array(px),
        motionRg: new Float32Array(px * 2),
        welfordMeanM2: new Float32Array(px * 2),
      }),
    ).not.toThrow();
  });

  it('throws when albedoRgb is undersized', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...{ rgb: new Float32Array(12), width: 2, height: 2 },
        albedoRgb: new Float32Array(11),
      }),
    ).toThrow(/albedoRgb/);
  });

  it('accepts a correctly sized albedoRgb', () => {
    const px = 4;
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        rgb: new Float32Array(px * 3),
        width: 2,
        height: 2,
        albedoRgb: new Float32Array(px * 3),
      }),
    ).not.toThrow();
  });
});

// ── Item 24 — albedo demodulation math (Schied 2017 §4.1) ────────────────────
//
// These tests mirror the private demodulateAlbedo / remodulateAlbedo helpers
// in atrousVarianceWebGPU.ts using local copies (private functions are not
// exported). They verify the demodulation identity and the checkerboard
// invariant described in the sweep plan.

describe('Item 24 — demodulateAlbedo helper', () => {
  it('uniform albedo: demodulation produces L = rgb / albedo per channel', () => {
    const px = 1;
    const rgb = new Float32Array([0.4, 0.2, 0.8]);
    const albedo = new Float32Array([0.8, 0.4, 0.4]);
    const out = demodulateAlbedo(rgb, albedo, px);
    expect(out[0]).toBeCloseTo(0.5, 4); // 0.4 / 0.8
    expect(out[1]).toBeCloseTo(0.5, 4); // 0.2 / 0.4
    expect(out[2]).toBeCloseTo(2.0, 4); // 0.8 / 0.4
  });

  it('clamps denominator to 1e-3 so black albedo does not divide by zero', () => {
    const px = 1;
    const rgb = new Float32Array([0.5, 0.5, 0.5]);
    const albedo = new Float32Array([0.0, 0.0, 0.0]); // fully black surface
    const out = demodulateAlbedo(rgb, albedo, px);
    expect(Number.isFinite(out[0])).toBe(true);
    expect(out[0]).toBeCloseTo(0.5 / 1e-3, 0); // clamped to 1e-3
  });

  it('identity: albedo = (1, 1, 1) → demodulated = rgb unchanged', () => {
    const px = 2;
    const rgb = new Float32Array([0.3, 0.6, 0.9, 0.1, 0.2, 0.3]);
    const albedo = new Float32Array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    const out = demodulateAlbedo(rgb, albedo, px);
    for (let i = 0; i < rgb.length; i += 1) {
      expect(out[i]).toBeCloseTo(rgb[i]!, 5);
    }
  });

  it('checkerboard: alternating albedo 0.1 / 0.9 with uniform L=0.5 → uniform demodulated signal', () => {
    // This is the core Item 24 invariant from the sweep plan:
    // A checkerboard albedo pattern under uniform indirect lighting L=0.5
    // should produce a UNIFORM demodulated signal L/albedo — the albedo
    // variation is completely separated from the lighting.
    //
    // After à-trous filtering (which uses normal/depth stops, not albedo),
    // the demodulated signal stays ~uniform. Re-modulating by albedo restores
    // the checkerboard pattern crisply (no albedo bleeding between tiles).
    const W = 4;
    const H = 1;
    const px = W * H;
    const rgb = new Float32Array(px * 3);
    const albedo = new Float32Array(px * 3);
    for (let i = 0; i < px; i += 1) {
      // Checker: even pixels albedo=0.1, odd pixels albedo=0.9.
      const a = i % 2 === 0 ? 0.1 : 0.9;
      // Full indirect radiance = L * albedo / pi (Lambertian), but for this
      // test we use L * albedo directly (the pi factor cancels in ratio).
      const L = 0.5;
      albedo[i * 3] = albedo[i * 3 + 1] = albedo[i * 3 + 2] = a;
      rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = L * a;
    }
    const demod = demodulateAlbedo(rgb, albedo, px);
    // After demodulation every pixel should equal L = 0.5 (uniform).
    for (let i = 0; i < px; i += 1) {
      expect(demod[i * 3]).toBeCloseTo(0.5, 4);
      expect(demod[i * 3 + 1]).toBeCloseTo(0.5, 4);
      expect(demod[i * 3 + 2]).toBeCloseTo(0.5, 4);
    }
  });
});

describe('Item 24 — remodulateAlbedo helper', () => {
  it('multiplies filtered lighting by albedo to restore outgoing radiance', () => {
    const px = 1;
    const filtered = new Float32Array([0.5, 0.5, 0.5]);
    const albedo = new Float32Array([0.8, 0.4, 0.2]);
    const out = remodulateAlbedo(filtered, albedo, px);
    expect(out[0]).toBeCloseTo(0.4, 5); // 0.5 × 0.8
    expect(out[1]).toBeCloseTo(0.2, 5); // 0.5 × 0.4
    expect(out[2]).toBeCloseTo(0.1, 5); // 0.5 × 0.2
  });

  it('demodulate then remodulate is identity (roundtrip)', () => {
    const px = 3;
    const original = new Float32Array([0.4, 0.8, 0.2, 0.6, 0.3, 0.7, 0.1, 0.5, 0.9]);
    const albedo = new Float32Array([0.9, 0.5, 0.3, 0.7, 0.8, 0.2, 0.4, 0.6, 1.0]);
    const demod = demodulateAlbedo(original, albedo, px);
    const restored = remodulateAlbedo(demod, albedo, px);
    for (let i = 0; i < original.length; i += 1) {
      // Round-trip tolerance is 1e-4 due to the 1e-3 clamp on near-zero albedo.
      expect(restored[i]).toBeCloseTo(original[i]!, 4);
    }
  });

  it('checkerboard round-trip: demod → no-op filter → remod preserves checkerboard edges', () => {
    // After demodulation the atrous filter sees a uniform signal and blurs
    // it (no edges to stop at). Remodulation then restores the checkerboard.
    // In this CPU-side test the "filter" is identity (no GPU); we just
    // verify that demod + remod exactly round-trips the checkerboard.
    const W = 4;
    const H = 1;
    const px = W;
    const albedo = new Float32Array(px * 3);
    const rgb = new Float32Array(px * 3);
    for (let i = 0; i < px; i += 1) {
      const a = i % 2 === 0 ? 0.1 : 0.9;
      const L = 0.5;
      albedo[i * 3] = albedo[i * 3 + 1] = albedo[i * 3 + 2] = a;
      rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = L * a;
    }
    const demod = demodulateAlbedo(rgb, albedo, px);
    const restored = remodulateAlbedo(demod, albedo, px);
    for (let i = 0; i < px; i += 1) {
      const expected = rgb[i * 3]!;
      expect(restored[i * 3]).toBeCloseTo(expected, 4);
      expect(restored[i * 3 + 1]).toBeCloseTo(expected, 4);
      expect(restored[i * 3 + 2]).toBeCloseTo(expected, 4);
    }
  });
});
