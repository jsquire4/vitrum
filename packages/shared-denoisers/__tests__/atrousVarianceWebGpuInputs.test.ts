import { describe, expect, it } from 'vitest';
import {
  assertAtrousVarianceWebGPUBufferShapes,
  runAtrousVarianceWebGPU,
} from '../src/atrousVarianceWebGPU.js';
import { ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT } from '../src/atrousVarianceConstants.js';
import { ATROUS_VARIANCE_WGSL } from '../src/wgsl/atrousVariance.wgsl.js';
import { demodulateAlbedo, remodulateAlbedo } from '../src/albedoModulation.js';

// ── Albedo demodulate/remodulate helpers ─────────────────────────────────────
// These previously were LOCAL COPIES of the helpers, on the premise that they
// were private to atrousVarianceWebGPU.ts. They are not: the single production
// implementation lives in src/albedoModulation.ts and is imported by all three
// host denoiser paths (atrousVarianceWebGPU, svgfRealWebGPU, bmfrWebGPU). Local
// copies made these assertions pass regardless of what production actually did.
// Import the real module so the tests guard the shipped code.
//
// NOTE: remodulateAlbedo mutates its input in place and returns it; the former
// local copy allocated a new array. Callers below clone where they need the
// original preserved.

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
    expect(() => assertAtrousVarianceWebGPUBufferShapes({ rgb: new Float32Array(8), width: 2, height: 2 })).toThrow(/rgb/);
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
    const normals = new Float32Array(px * 3);
    for (let pixel = 0; pixel < px; pixel += 1) {
      normals[pixel * 3 + 1] = 1;
    }
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        rgb: new Float32Array(px * 3),
        width: 2,
        height: 2,
        gbufferNormalsRgb: normals,
        linearDepth: new Float32Array(px),
        welfordMeanM2: new Float32Array(px * 2),
      }),
    ).not.toThrow();
  });

  it('rejects signed normal components outside the shader encoding domain', () => {
    const normals = new Float32Array(12);
    normals[5] = 1.01;
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...minimal,
        gbufferNormalsRgb: normals,
      }),
    ).toThrow(/gbufferNormalsRgb\[5\].*\[-1, 1\]/);
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

  it('rejects an unknown Welford signal domain', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...minimal,
        welfordMeanM2Domain: 'display' as 'radiance',
      }),
    ).toThrow(/welfordMeanM2Domain/);
  });

  it('rejects radiance-domain temporal moments when the filtered signal is albedo-demodulated', async () => {
    const px = 4;
    await expect(runAtrousVarianceWebGPU({
      rgb: new Float32Array(px * 3).fill(0.5),
      width: 2,
      height: 2,
      frameCount: ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT,
      albedoRgb: new Float32Array(px * 3).fill(0.25),
      welfordMeanM2: new Float32Array(px * 2),
      // Omitted means the historical/default radiance domain. The rejection
      // happens before device acquisition, so this is a CPU-only production
      // entry-point regression rather than a source-string pin.
    })).rejects.toThrow(/must be moments of that demodulated signal/);
  });
});

describe('standalone atrous normal edge-stop safety', () => {
  it('bounds the high-exponent dot-product base to the physical cosine domain', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain(
      'let dn = clamp(dot(nCenter, nP), 0.0, 1.0);',
    );
    expect(ATROUS_VARIANCE_WGSL).not.toContain(
      'let dn = max(0.0, dot(nCenter, nP));',
    );
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
    const rgb    = new Float32Array([0.4, 0.2, 0.8]);
    const albedo = new Float32Array([0.8, 0.4, 0.4]);
    const out = demodulateAlbedo(rgb, albedo, px);
    expect(out[0]).toBeCloseTo(0.5, 4);   // 0.4 / 0.8
    expect(out[1]).toBeCloseTo(0.5, 4);   // 0.2 / 0.4
    expect(out[2]).toBeCloseTo(2.0, 4);   // 0.8 / 0.4
  });

  it('clamps denominator to 1e-3 so black albedo does not divide by zero', () => {
    const px = 1;
    const rgb    = new Float32Array([0.5, 0.5, 0.5]);
    const albedo = new Float32Array([0.0, 0.0, 0.0]);  // fully black surface
    const out = demodulateAlbedo(rgb, albedo, px);
    expect(Number.isFinite(out[0])).toBe(true);
    expect(out[0]).toBeCloseTo(0.5 / 1e-3, 0);  // clamped to 1e-3
  });

  it('identity: albedo = (1, 1, 1) → demodulated = rgb unchanged', () => {
    const px = 2;
    const rgb    = new Float32Array([0.3, 0.6, 0.9, 0.1, 0.2, 0.3]);
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
    const W = 4; const H = 1; const px = W * H;
    const rgb    = new Float32Array(px * 3);
    const albedo = new Float32Array(px * 3);
    for (let i = 0; i < px; i += 1) {
      // Checker: even pixels albedo=0.1, odd pixels albedo=0.9.
      const a = i % 2 === 0 ? 0.1 : 0.9;
      // Full indirect radiance = L * albedo / pi (Lambertian), but for this
      // test we use L * albedo directly (the pi factor cancels in ratio).
      const L = 0.5;
      albedo[i * 3] = albedo[i * 3 + 1] = albedo[i * 3 + 2] = a;
      rgb[i * 3]    = rgb[i * 3 + 1]    = rgb[i * 3 + 2]    = L * a;
    }
    const demod = demodulateAlbedo(rgb, albedo, px);
    // After demodulation every pixel should equal L = 0.5 (uniform).
    for (let i = 0; i < px; i += 1) {
      expect(demod[i * 3    ]).toBeCloseTo(0.5, 4);
      expect(demod[i * 3 + 1]).toBeCloseTo(0.5, 4);
      expect(demod[i * 3 + 2]).toBeCloseTo(0.5, 4);
    }
  });
});

describe('Item 24 — remodulateAlbedo helper', () => {
  it('multiplies filtered lighting by albedo to restore outgoing radiance', () => {
    const px = 1;
    const filtered = new Float32Array([0.5, 0.5, 0.5]);
    const albedo   = new Float32Array([0.8, 0.4, 0.2]);
    const out = remodulateAlbedo(filtered, albedo, px);
    expect(out[0]).toBeCloseTo(0.4, 5);   // 0.5 × 0.8
    expect(out[1]).toBeCloseTo(0.2, 5);   // 0.5 × 0.4
    expect(out[2]).toBeCloseTo(0.1, 5);   // 0.5 × 0.2
  });

  it('demodulate then remodulate is identity (roundtrip)', () => {
    const px = 3;
    const original = new Float32Array([0.4, 0.8, 0.2, 0.6, 0.3, 0.7, 0.1, 0.5, 0.9]);
    const albedo   = new Float32Array([0.9, 0.5, 0.3, 0.7, 0.8, 0.2, 0.4, 0.6, 1.0]);
    const demod    = demodulateAlbedo(original, albedo, px);
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
    const W = 4; const px = W;
    const albedo = new Float32Array(px * 3);
    const rgb    = new Float32Array(px * 3);
    for (let i = 0; i < px; i += 1) {
      const a = i % 2 === 0 ? 0.1 : 0.9;
      const L = 0.5;
      albedo[i * 3] = albedo[i * 3 + 1] = albedo[i * 3 + 2] = a;
      rgb[i * 3]    = rgb[i * 3 + 1]    = rgb[i * 3 + 2]    = L * a;
    }
    const demod    = demodulateAlbedo(rgb, albedo, px);
    const restored = remodulateAlbedo(demod, albedo, px);
    for (let i = 0; i < px; i += 1) {
      const expected = rgb[i * 3]!;
      expect(restored[i * 3    ]).toBeCloseTo(expected, 4);
      expect(restored[i * 3 + 1]).toBeCloseTo(expected, 4);
      expect(restored[i * 3 + 2]).toBeCloseTo(expected, 4);
    }
  });

  // REGRESSION: demodulate floored the divisor at 1e-3 while remodulate
  // multiplied by the RAW albedo, so the pair was not an inverse below the
  // floor: channels in (0, 1e-3) were attenuated by up to 1000x and channels at
  // exactly 0 were annihilated. Any radiance that is not purely diffuse-
  // reflected — an emitter, or the environment background, both of which
  // legitimately sit on a zero-albedo g-buffer texel — came out of an
  // albedo-aware denoiser black while rendering correctly with the denoiser off.
  // Both directions now resolve the channel through the same floored accessor.
  it('round-trips exactly at and below the albedo floor (zero-albedo radiance survives)', () => {
    const px = 3;
    //                                    albedo 0        albedo 1e-4     albedo 0.5
    const original = new Float32Array([2.5, 1.0, 0.4,  0.8, 0.2, 0.6,  0.3, 0.9, 0.1]);
    const albedo   = new Float32Array([0.0, 0.0, 0.0,  1e-4, 1e-4, 1e-4, 0.5, 0.5, 0.5]);
    const restored = remodulateAlbedo(demodulateAlbedo(original, albedo, px), albedo, px);
    for (let i = 0; i < original.length; i += 1) {
      expect(restored[i]).toBeCloseTo(original[i]!, 6);
    }
    // The zero-albedo emitter texel specifically must not be blacked out.
    expect(restored[0]).toBeGreaterThan(0);
  });

  it('a short albedo buffer is neutral in both directions, not a 1000x amplifier', () => {
    const px = 2;
    const original = new Float32Array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
    // Only the first pixel has albedo; the second is out of range.
    const albedo = new Float32Array([0.5, 0.5, 0.5]);
    const demod = demodulateAlbedo(original, albedo, px);
    // Out-of-range channels resolve to the neutral 1, so demodulation is a no-op
    // there rather than dividing by the 1e-3 floor.
    expect(demod[3]).toBeCloseTo(0.7, 6);
    const restored = remodulateAlbedo(demod, albedo, px);
    for (let i = 0; i < original.length; i += 1) {
      expect(restored[i]).toBeCloseTo(original[i]!, 6);
    }
  });
});
