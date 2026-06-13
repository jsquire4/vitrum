/**
 * Contract-honesty tests for environmentPacking.ts:
 *   1. RGBA input (stride 4) is decoded correctly — RGB-only decode garbles values.
 *   2. All-black HDRI emits an accurate warning rather than the misleading
 *      "lacks CPU pixel data" message.
 *   3. Preetham procedural-sky bake — numerical invariants.
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
    },
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
    // No shader-side procedural-sky fallback is implied for invalid HDRI payloads.
    expect(p.warnings.some((w) => w.includes('procedural sky model'))).toBe(false);
  });

  it('emits the "lacks CPU pixel data" warning only when data is genuinely absent', () => {
    const scene: Scene = {
      primitives: [],
      emitters: [],
      environment: {
        kind: 'hdri',
        hdri: {} as unknown as Scene['environment'] & object,
      },
    };
    const p = environmentParams(scene);
    expect(p.hasHdri).toBe(false);
    expect(p.warnings.some((w) => w.includes('lacks CPU pixel data'))).toBe(true);
    expect(p.warnings.some((w) => w.includes('procedural sky model'))).toBe(false);
  });
});

describe('environmentPacking — no environment', () => {
  it('packs kind:none as a truly black no-environment slot', () => {
    const scene: Scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' },
    };
    const p = environmentParams(scene);
    expect(p.hasHdri).toBe(false);
    expect(p.hdriWidth).toBe(0);
    expect(p.hdriHeight).toBe(0);
    expect(p.hdriIntensity).toBe(0);
    expect(p.sunStrength).toBe(0);
    expect(p.hdriTexels.length).toBe(0);
    expect(p.hdriCdf.length).toBe(0);
    expect(p.warnings).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Preetham procedural-sky bake tests (CPU, no GPU needed)
// Ref: Preetham, Shirley, Smits, SIGGRAPH 1999.
// ---------------------------------------------------------------------------

function makeProceduralSkyScene(opts: {
  sunDirection?: [number, number, number];
  turbidity?: number;
  rayleigh?: number;
  mieCoefficient?: number;
  mieDirectionalG?: number;
  intensity?: number;
}): Scene {
  return {
    primitives: [],
    emitters: [],
    environment: {
      kind: 'procedural-sky',
      sunDirection: opts.sunDirection ?? [0, 1, 0],
      turbidity: opts.turbidity ?? 2,
      rayleigh: opts.rayleigh ?? 1,
      mieCoefficient: opts.mieCoefficient ?? 0.005,
      mieDirectionalG: opts.mieDirectionalG ?? 0.8,
      ...(opts.intensity !== undefined ? { intensity: opts.intensity } : {}),
    },
  };
}

describe('environmentPacking — Preetham procedural sky bake', () => {
  // ----- 1. Routes through the HDRI path (hasHdri = true) -----
  it('returns hasHdri=true so the HDRI importance-sampling path is used', () => {
    const p = environmentParams(makeProceduralSkyScene({}));
    expect(p.hasHdri).toBe(true);
  });

  // ----- 2. Sun texel is the map maximum (sun-extraction can locate it) -----
  it('sun-direction pixel is the map maximum', () => {
    // Sun at zenith (θ=0, +Y up) — pixel at (py, px) corresponding to θ≈0, any φ
    const p = environmentParams(makeProceduralSkyScene({ sunDirection: [0, 1, 0] }));
    expect(p.hasHdri).toBe(true);
    const W = p.hdriWidth;
    const H = p.hdriHeight;
    expect(W).toBe(256);
    expect(H).toBe(128);

    // Find the map maximum (luminance of RGB channels).
    let maxLum = -Infinity;
    let maxIdx = 0;
    for (let i = 0; i < W * H; i += 1) {
      const r = p.hdriTexels[i * 4] ?? 0;
      const g = p.hdriTexels[i * 4 + 1] ?? 0;
      const b = p.hdriTexels[i * 4 + 2] ?? 0;
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      if (lum > maxLum) { maxLum = lum; maxIdx = i; }
    }

    // The maximum pixel's θ should be near 0 (zenith sun).
    const pyMax = (maxIdx / W) | 0;
    const thetaMax = ((pyMax + 0.5) / H) * Math.PI;
    // Within 10° of zenith (sun at +Y)
    expect(thetaMax).toBeLessThan(Math.PI / 18);
  });

  it('sun-direction pixel is the map maximum for a low-angle (horizon) sun', () => {
    // Sun near the horizon, East (+X, y≈0)
    const sunDir: [number, number, number] = [1, 0.05, 0];
    const len = Math.hypot(...sunDir);
    const normSunDir: [number, number, number] = [sunDir[0] / len, sunDir[1] / len, sunDir[2] / len];
    const p = environmentParams(makeProceduralSkyScene({ sunDirection: normSunDir }));

    let maxLum = -Infinity;
    let maxIdx = 0;
    const W = p.hdriWidth;
    const H = p.hdriHeight;
    for (let i = 0; i < W * H; i += 1) {
      const r = p.hdriTexels[i * 4] ?? 0;
      const g = p.hdriTexels[i * 4 + 1] ?? 0;
      const b = p.hdriTexels[i * 4 + 2] ?? 0;
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      if (lum > maxLum) { maxLum = lum; maxIdx = i; }
    }

    const pyMax = (maxIdx / W) | 0;
    const pxMax = maxIdx % W;
    const thetaMax = ((pyMax + 0.5) / H) * Math.PI;
    const phiMax   = ((pxMax + 0.5) / W) * (2 * Math.PI);
    // Expected: θ ≈ π/2 (horizon), φ ≈ 0 (+X east)
    expect(thetaMax).toBeGreaterThan(Math.PI / 2 - 0.3);
    expect(thetaMax).toBeLessThan(Math.PI / 2 + 0.3);
    // φ=0 corresponds to +X; allow ±15° = ±0.26 rad
    const phiErr = Math.min(phiMax, 2 * Math.PI - phiMax);
    expect(phiErr).toBeLessThan(0.3);
  });

  // ----- 3. Zenith luminance matches Preetham zenith polynomial (T=2, T=5) -----
  // Preetham Eq. A.4: Yz = (4.0453T − 4.9710)·tan((4/9 − T/120)·(π − 2θs)) − 0.2155T + 2.4192
  // At θs ≈ 0 (sun at zenith), we can check the relative zenith brightness via the map.
  it('zenith pixel luminance scales with turbidity (higher T → brighter atmosphere)', () => {
    // Turbidity 2 (clear) vs turbidity 8 (hazy).  Hazy sky has more scattered
    // light at the zenith from Mie, so zenith luminance is HIGHER at T=8.
    const pClear = environmentParams(makeProceduralSkyScene({ sunDirection: [0, 1, 0], turbidity: 2 }));
    const pHazy  = environmentParams(makeProceduralSkyScene({ sunDirection: [0, 1, 0], turbidity: 8 }));

    const W = pClear.hdriWidth;
    // Zenith pixel: py=0, px in the middle
    const pxMid = Math.floor(W / 2);
    const riClear = 4 * pxMid;  // py=0
    const riHazy  = 4 * pxMid;
    const lumClear =
      0.2126 * (pClear.hdriTexels[riClear] ?? 0) +
      0.7152 * (pClear.hdriTexels[riClear + 1] ?? 0) +
      0.0722 * (pClear.hdriTexels[riClear + 2] ?? 0);
    const lumHazy =
      0.2126 * (pHazy.hdriTexels[riHazy] ?? 0) +
      0.7152 * (pHazy.hdriTexels[riHazy + 1] ?? 0) +
      0.0722 * (pHazy.hdriTexels[riHazy + 2] ?? 0);

    // Preetham Yz at θs=0: both are positive; T=8 is brighter than T=2.
    expect(lumHazy).toBeGreaterThan(lumClear);
  });

  // ----- 4. Map is finite and non-negative everywhere -----
  it('all texel values are finite and non-negative', () => {
    const p = environmentParams(makeProceduralSkyScene({ sunDirection: [0.6, 0.8, 0] }));
    const n = p.hdriWidth * p.hdriHeight * 4;
    for (let i = 0; i < n; i += 1) {
      const v = p.hdriTexels[i] ?? 0;
      expect(isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
    }
  });

  // ----- 5. CDF is valid (monotone, starts at 0, ends at 1) -----
  it('CDF is monotone non-decreasing from 0 to 1', () => {
    const p = environmentParams(makeProceduralSkyScene({}));
    const cdf = p.hdriCdf;
    const N = p.hdriWidth * p.hdriHeight;
    expect(cdf.length).toBe(N + 1);
    expect(cdf[0]).toBeCloseTo(0, 10);
    expect(cdf[N]).toBeCloseTo(1, 10);
    for (let i = 0; i < N; i += 1) {
      expect((cdf[i + 1] ?? 0)).toBeGreaterThanOrEqual((cdf[i] ?? 0) - 1e-9);
    }
  });

  // ----- 6. No heuristic red-channel fudge (the 0.2 floor must be gone) -----
  it('tint is white (radiance fully baked into texels; no heuristic tint fudge)', () => {
    const p = environmentParams(makeProceduralSkyScene({ mieCoefficient: 1.0 }));
    // The old heuristic would produce a non-white tint and set hasHdri=false.
    // The new implementation always returns tint=[1,1,1] and hasHdri=true.
    expect(p.tint[0]).toBeCloseTo(1);
    expect(p.tint[1]).toBeCloseTo(1);
    expect(p.tint[2]).toBeCloseTo(1);
    expect(p.hdriIntensity).toBeGreaterThan(0);
  });

  // ----- 7. sunStrength = 0 (sky-NEE branch does NOT double-fire) -----
  it('sunStrength is 0 so the procedural-sky NEE gate does not double-fire', () => {
    const p = environmentParams(makeProceduralSkyScene({ intensity: 3 }));
    expect(p.sunStrength).toBe(0);
  });

  // ----- 8. Horizon–zenith ratio: zenith should be brighter than horizon (clear sky) -----
  it('zenith luminance exceeds average horizon-band luminance (T=2, noon sun)', () => {
    const p = environmentParams(makeProceduralSkyScene({
      sunDirection: [0, 1, 0],   // sun at zenith
      turbidity: 2,
    }));
    const W = p.hdriWidth;
    const H = p.hdriHeight;
    // Zenith row (py=0): average luminance
    let zenithSum = 0;
    for (let px = 0; px < W; px += 1) {
      const i = px;
      zenithSum += 0.2126 * (p.hdriTexels[i * 4] ?? 0) +
                   0.7152 * (p.hdriTexels[i * 4 + 1] ?? 0) +
                   0.0722 * (p.hdriTexels[i * 4 + 2] ?? 0);
    }
    const zenithAvg = zenithSum / W;
    // Horizon band (py near H/2, θ ≈ π/2): rows H*3/8..H*5/8
    const pyLo = Math.floor(H * 3 / 8);
    const pyHi = Math.ceil(H * 5 / 8);
    let horizonSum = 0;
    let horizonCount = 0;
    for (let py = pyLo; py < pyHi; py += 1) {
      for (let px = 0; px < W; px += 1) {
        const i = py * W + px;
        horizonSum += 0.2126 * (p.hdriTexels[i * 4] ?? 0) +
                      0.7152 * (p.hdriTexels[i * 4 + 1] ?? 0) +
                      0.0722 * (p.hdriTexels[i * 4 + 2] ?? 0);
        horizonCount += 1;
      }
    }
    const horizonAvg = horizonSum / horizonCount;
    // At T=2, the zenith under the sun should outshine the mid-sky horizon band.
    // (This tests the sun-at-zenith case; with the sun disk at zenith, zenith ≫ horizon.)
    expect(zenithAvg).toBeGreaterThan(horizonAvg);
  });

  // ----- 9. No warnings emitted for a valid procedural-sky -----
  it('emits no warnings for valid inputs', () => {
    const p = environmentParams(makeProceduralSkyScene({}));
    expect(p.warnings.length).toBe(0);
  });

  // ----- 10. hdriWidth/Height match the expected bake dimensions -----
  it('baked map has the expected 256×128 dimensions', () => {
    const p = environmentParams(makeProceduralSkyScene({}));
    expect(p.hdriWidth).toBe(256);
    expect(p.hdriHeight).toBe(128);
    expect(p.hdriTexels.length).toBe(256 * 128 * 4);
    expect(p.hdriCdf.length).toBe(256 * 128 + 1);
  });

  // ----- 11. Partial-object guard: missing Preetham fields must NOT produce NaN texels -----
  //
  // Root cause (R7a-4): The behavioural gate passes { kind:'procedural-sky', sunDirection }
  // without turbidity/rayleigh/mieCoefficient/mieDirectionalG.  JavaScript's
  //   Math.max(1.5, Math.min(30, undefined)) === NaN   (NOT 1.5)
  // so every Preetham intermediate becomes NaN → the GPU buffer is all-NaN → the
  // sky renders black with zero GPU validation errors (NaN arithmetic is valid WGSL).
  // The fix applies per-field defaults in buildProceduralSkyEnvironmentParams so a
  // partial scene object is equivalent to the documented API defaults.
  it('partial object (sunDirection only, no turbidity/rayleigh/mie fields) produces non-NaN, non-black texels', () => {
    // Exactly the object gate.mjs builds for opts.sky = true.
    const scene: Scene = {
      primitives: [],
      emitters: [],
      // Cast via unknown: the TS type requires all fields, but real-world
      // @ts-nocheck hosts omit them — this pins the runtime NaN guard.
      environment: { kind: 'procedural-sky', sunDirection: [0.5, 1.0, 0.3] } as unknown as Scene['environment'],
    };
    const p = environmentParams(scene);
    expect(p.hasHdri).toBe(true);
    expect(p.hdriIntensity).toBeGreaterThan(0);
    expect(p.hdriWidth).toBe(256);

    // No NaN texels — GPU buffer must be clean.
    let nanCount = 0;
    for (let i = 0; i < p.hdriTexels.length; i++) {
      if (!Number.isFinite(p.hdriTexels[i] ?? NaN)) nanCount++;
    }
    expect(nanCount).toBe(0);

    // Mean luminance must be meaningfully above zero (sky has light).
    let sumLum = 0;
    const N = p.hdriWidth * p.hdriHeight;
    for (let i = 0; i < N; i++) {
      const r = p.hdriTexels[i * 4] ?? 0;
      const g = p.hdriTexels[i * 4 + 1] ?? 0;
      const b = p.hdriTexels[i * 4 + 2] ?? 0;
      sumLum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    expect(sumLum / N).toBeGreaterThan(0.1);
  });

  it('bare { kind: "procedural-sky" } (no other fields) produces non-NaN, non-black texels', () => {
    const scene: Scene = {
      primitives: [],
      emitters: [],
      // All Preetham fields absent; defaults must kick in.
      environment: { kind: 'procedural-sky' } as Scene['environment'],
    };
    const p = environmentParams(scene);
    expect(p.hasHdri).toBe(true);

    let nanCount = 0;
    for (let i = 0; i < p.hdriTexels.length; i++) {
      if (!Number.isFinite(p.hdriTexels[i] ?? NaN)) nanCount++;
    }
    expect(nanCount).toBe(0);

    let sumLum = 0;
    const N = p.hdriWidth * p.hdriHeight;
    for (let i = 0; i < N; i++) {
      const r = p.hdriTexels[i * 4] ?? 0;
      const g = p.hdriTexels[i * 4 + 1] ?? 0;
      const b = p.hdriTexels[i * 4 + 2] ?? 0;
      sumLum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    expect(sumLum / N).toBeGreaterThan(0.1);
  });
});
