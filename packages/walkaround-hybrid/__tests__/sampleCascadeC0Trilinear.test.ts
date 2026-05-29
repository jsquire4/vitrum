/**
 * sampleCascadeC0 — trilinear probe interpolation (Phase-3b, 2026-05-28).
 *
 * `sampleCascadeC0.wgsl.ts` reads Sannikov 2023 Radiance-Cascades level-0
 * storage at a world position + surface normal and returns a Lambertian
 * irradiance estimate. Phase 3 sampled the NEAREST cascade-0 probe only,
 * which produced blocky/discontinuous indirect GI at probe boundaries.
 *
 * Phase-3b interpolates trilinearly over the 8 surrounding cascade-0 probes,
 * weighted by the shading point's fractional position in the probe grid.
 *
 * This test ports the trilinear weight computation + edge/invalid handling
 * to TS, line-for-line with the WGSL, and pins:
 *   (a) trilinear weights over the 8 corners sum to exactly 1;
 *   (b) a point exactly on a probe centre returns THAT probe's value
 *       (degenerate trilinear — single corner carries all the weight);
 *   (c) a midpoint blends the two/eight neighbours (no single corner wins);
 *   (d) edge clamping — a point past the grand boundary clamps its corners
 *       into valid range and never reads out of bounds;
 *   (e) invalid-probe (Wsum ≤ 1e-4) corners are dropped + the surviving
 *       weights renormalise, so zero radiance never leaks into the blend;
 *   (f) the `rcParams.enabled == 0` path returns EXACTLY vec3f(0) — the
 *       disabled path is bit-identical (shade.wgsl's RC-MIS relies on it).
 *
 * Plus structural pins on the WGSL so the shader cannot silently drift
 * back to the nearest-probe form or break the disabled-path contract.
 *
 * References:
 *   Sannikov 2023, "Radiance Cascades" §3 (cascade construction).
 *   probeRayCast.wgsl.ts:220 — probes centred at (p + 0.5)/count in grid-UV.
 */

import { describe, expect, it } from 'vitest';
import { SAMPLE_CASCADE_C0_WGSL } from '../src/shaders/sampleCascadeC0.wgsl.js';

type Vec3 = [number, number, number];

// ── CPU port of the WGSL trilinear blend (sampleCascadeC0) ──────────────
// Mirrors the WGSL body so the test fails if the shader math drifts.

interface RCGrid {
  probeOriginWorld: Vec3;
  roomSize: Vec3;
  probeCount: [number, number, number];
}

/** Probe linear index — matches the WGSL + producer
 *  `pi.z·PX·PY + pi.y·PX + pi.x`. */
function probeLinearIdx(pi: [number, number, number], count: [number, number, number]): number {
  return pi[2] * count[0] * count[1] + pi[1] * count[0] + pi[0];
}

/** The 8 trilinear corner weights, keyed by corner bit (x=bit0, y=bit1, z=bit2).
 *  Mirrors `mix(1-f, f, d)` per axis. */
function cornerWeights(f: Vec3): number[] {
  const w: number[] = [];
  for (let corner = 0; corner < 8; corner++) {
    const dx = corner & 1;
    const dy = (corner >> 1) & 1;
    const dz = (corner >> 2) & 1;
    const wx = dx ? f[0] : 1 - f[0];
    const wy = dy ? f[1] : 1 - f[1];
    const wz = dz ? f[2] : 1 - f[2];
    w.push(wx * wy * wz);
  }
  return w;
}

/** clamp helper (per-axis), matching WGSL `clamp(v, 0, count-1)`. */
function clampCorner(
  g0: Vec3,
  d: [number, number, number],
  count: [number, number, number],
): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const v = g0[a]! + d[a]!;
    out[a] = Math.min(Math.max(v, 0), count[a]! - 1);
  }
  return out;
}

interface BlendTrace {
  /** Sum of trilinear weights over all 8 corners (before any drop). */
  weightSumAll: number;
  /** Sum of weights over corners that survived the validity filter. */
  weightSumSurviving: number;
  /** Final renormalised, INV_PI-scaled radiance. */
  lo: Vec3;
  /** Per-corner: clamped probe index + weight (for edge-clamp assertions). */
  corners: { probeIdx: number; wTri: number; valid: boolean }[];
}

const INV_PI = 1 / Math.PI;

/**
 * CPU port of `sampleCascadeC0`.
 * `probeIrradiance(probeIdx)` returns the probe's `(rgb, Wsum)` estimate;
 * a probe with `Wsum ≤ 1e-4` is treated as invalid (dropped + renormalised),
 * exactly like the WGSL `rcProbeIrradiance` guard.
 */
function sampleCascadeC0CPU(
  worldPos: Vec3,
  grid: RCGrid,
  probeIrradiance: (idx: number) => { rgb: Vec3; wsum: number },
  enabled = true,
): BlendTrace {
  // Disabled path is bit-identical to the WGSL early return.
  if (!enabled) {
    return { weightSumAll: 0, weightSumSurviving: 0, lo: [0, 0, 0], corners: [] };
  }

  const count = grid.probeCount;
  const probeUV: Vec3 = [
    (worldPos[0] - grid.probeOriginWorld[0]) / grid.roomSize[0],
    (worldPos[1] - grid.probeOriginWorld[1]) / grid.roomSize[1],
    (worldPos[2] - grid.probeOriginWorld[2]) / grid.roomSize[2],
  ];
  // Probe-centre-relative grid coord: g = probeUV·count − 0.5.
  const g: Vec3 = [
    probeUV[0] * count[0] - 0.5,
    probeUV[1] * count[1] - 0.5,
    probeUV[2] * count[2] - 0.5,
  ];
  const g0: Vec3 = [Math.floor(g[0]), Math.floor(g[1]), Math.floor(g[2])];
  const f: Vec3 = [g[0] - g0[0], g[1] - g0[1], g[2] - g0[2]];

  const weights = cornerWeights(f);
  const blendL: Vec3 = [0, 0, 0];
  let weightSumAll = 0;
  let weightSumSurviving = 0;
  const corners: BlendTrace['corners'] = [];

  for (let corner = 0; corner < 8; corner++) {
    const wTri = weights[corner]!;
    weightSumAll += wTri;
    if (wTri <= 0) continue;
    const d: [number, number, number] = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
    const pi = clampCorner(g0, d, count);
    const probeIdx = probeLinearIdx(pi, count);
    const probe = probeIrradiance(probeIdx);
    const valid = probe.wsum > 1e-4;
    corners.push({ probeIdx, wTri, valid });
    if (!valid) continue; // drop + renormalise
    blendL[0] += probe.rgb[0] * wTri;
    blendL[1] += probe.rgb[1] * wTri;
    blendL[2] += probe.rgb[2] * wTri;
    weightSumSurviving += wTri;
  }

  if (weightSumSurviving > 1e-4) {
    return {
      weightSumAll,
      weightSumSurviving,
      lo: [
        (blendL[0] * INV_PI) / weightSumSurviving,
        (blendL[1] * INV_PI) / weightSumSurviving,
        (blendL[2] * INV_PI) / weightSumSurviving,
      ],
      corners,
    };
  }
  return { weightSumAll, weightSumSurviving, lo: [0, 0, 0], corners };
}

// A 4×4×4 unit-cube grid for the math tests.
const grid: RCGrid = {
  probeOriginWorld: [0, 0, 0],
  roomSize: [1, 1, 1],
  probeCount: [4, 4, 4],
};

/** Every probe valid, all radiance = `value`. */
function uniformProbes(value: Vec3, wsum = 8): (idx: number) => { rgb: Vec3; wsum: number } {
  return () => ({ rgb: [...value] as Vec3, wsum });
}

describe('sampleCascadeC0 — trilinear weights (CPU port)', () => {
  // (a) Weights over the 8 corners are a partition of unity for any fraction.
  it('(a) the 8 trilinear corner weights sum to exactly 1', () => {
    for (const fx of [0, 0.1, 0.5, 0.73, 0.999]) {
      for (const fy of [0, 0.25, 0.5, 0.9]) {
        for (const fz of [0, 0.5, 0.6]) {
          const w = cornerWeights([fx, fy, fz]);
          const sum = w.reduce((a, b) => a + b, 0);
          expect(sum, `f=(${fx},${fy},${fz})`).toBeCloseTo(1, 12);
          // Each weight is a valid [0,1] contribution.
          for (const wi of w) {
            expect(wi).toBeGreaterThanOrEqual(0);
            expect(wi).toBeLessThanOrEqual(1);
          }
        }
      }
    }
  });

  // (b) On a probe centre, the trilinear blend returns that probe's value.
  // Probe (i,j,k) centre is at world ((i+0.5)/4, (j+0.5)/4, (k+0.5)/4).
  it('(b) a point on a probe centre returns that probe value (degenerate)', () => {
    // Give probe #21 a distinct colour; everything else is gray.
    const targetIdx = probeLinearIdx([1, 1, 1], grid.probeCount);
    const targetColor: Vec3 = [0.4, 0.5, 0.6];
    const probeFn = (idx: number) => ({
      rgb: (idx === targetIdx ? targetColor : [0.1, 0.1, 0.1]) as Vec3,
      wsum: 8,
    });
    // World pos exactly at probe (1,1,1) centre = (1.5/4, 1.5/4, 1.5/4).
    const pos: Vec3 = [1.5 / 4, 1.5 / 4, 1.5 / 4];
    const r = sampleCascadeC0CPU(pos, grid, probeFn);
    // f = 0 ⇒ all weight on the lower corner = probe (1,1,1).
    expect(r.weightSumSurviving).toBeCloseTo(1, 12);
    // Result = targetColor · INV_PI (the per-probe nearest-path formula).
    expect(r.lo[0]).toBeCloseTo(targetColor[0] * INV_PI, 10);
    expect(r.lo[1]).toBeCloseTo(targetColor[1] * INV_PI, 10);
    expect(r.lo[2]).toBeCloseTo(targetColor[2] * INV_PI, 10);
  });

  // (c) A midpoint between two probes blends them 50/50.
  it('(c) a midpoint blends neighbouring probes', () => {
    // Probes (1,1,1) and (2,1,1) carry different colours; sample halfway in x.
    const idxA = probeLinearIdx([1, 1, 1], grid.probeCount);
    const idxB = probeLinearIdx([2, 1, 1], grid.probeCount);
    const colA: Vec3 = [1, 0, 0];
    const colB: Vec3 = [0, 0, 1];
    const probeFn = (idx: number) => {
      if (idx === idxA) return { rgb: [...colA] as Vec3, wsum: 8 };
      if (idx === idxB) return { rgb: [...colB] as Vec3, wsum: 8 };
      return { rgb: [0, 0, 0] as Vec3, wsum: 8 };
    };
    // Halfway between probe-1 centre (1.5/4) and probe-2 centre (2.5/4) in x:
    // world x = 2.0/4 ; y,z on the (1,1) centre line.
    const pos: Vec3 = [2.0 / 4, 1.5 / 4, 1.5 / 4];
    const r = sampleCascadeC0CPU(pos, grid, probeFn);
    // fx = 0.5, fy = fz = 0 ⇒ exactly 0.5 weight on each of A and B.
    // Blend = (0.5·colA + 0.5·colB)·INV_PI.
    expect(r.lo[0]).toBeCloseTo(0.5 * colA[0] * INV_PI, 10);
    expect(r.lo[1]).toBeCloseTo(0, 10);
    expect(r.lo[2]).toBeCloseTo(0.5 * colB[2] * INV_PI, 10);
    // The result is NOT equal to either probe alone (it blended).
    expect(r.lo[0]).not.toBeCloseTo(colA[0] * INV_PI, 6);
    expect(r.lo[2]).not.toBeCloseTo(colB[2] * INV_PI, 6);
  });

  // (d) Edge clamping — a point past the far boundary clamps all corners into
  // valid range and never produces an out-of-bounds probe index.
  it('(d) edge clamping keeps every corner probe in [0, count-1]', () => {
    const maxIdx = grid.probeCount[0] * grid.probeCount[1] * grid.probeCount[2] - 1;
    // Far corner of the room, beyond the last probe centre (g > count-1).
    const pos: Vec3 = [0.99, 0.99, 0.99];
    const r = sampleCascadeC0CPU(pos, grid, uniformProbes([0.3, 0.3, 0.3]));
    for (const c of r.corners) {
      expect(c.probeIdx).toBeGreaterThanOrEqual(0);
      expect(c.probeIdx).toBeLessThanOrEqual(maxIdx);
    }
    // Uniform field ⇒ blend equals the uniform value (clamped corners all
    // read the same colour), proving no out-of-range darkness leaks in.
    expect(r.lo[0]).toBeCloseTo(0.3 * INV_PI, 10);

    // A point at the exact min boundary too.
    const posMin: Vec3 = [0, 0, 0];
    const rMin = sampleCascadeC0CPU(posMin, grid, uniformProbes([0.3, 0.3, 0.3]));
    for (const c of rMin.corners) {
      expect(c.probeIdx).toBeGreaterThanOrEqual(0);
      expect(c.probeIdx).toBeLessThanOrEqual(maxIdx);
    }
    expect(rMin.lo[0]).toBeCloseTo(0.3 * INV_PI, 10);
  });

  // (e) Invalid probes (Wsum ≤ 1e-4) are dropped and the surviving weights
  // renormalise — a zero/uninitialised probe must NOT darken the blend.
  it('(e) invalid corners are dropped + surviving weights renormalise', () => {
    // Two-probe blend in x: probe A valid (colour C), probe B invalid (wsum 0).
    const idxA = probeLinearIdx([1, 1, 1], grid.probeCount);
    const idxB = probeLinearIdx([2, 1, 1], grid.probeCount);
    const C: Vec3 = [0.8, 0.4, 0.2];
    const probeFn = (idx: number) => {
      if (idx === idxA) return { rgb: [...C] as Vec3, wsum: 8 };
      if (idx === idxB) return { rgb: [9, 9, 9] as Vec3, wsum: 0 }; // invalid: huge rgb, zero mass
      return { rgb: [0, 0, 0] as Vec3, wsum: 8 };
    };
    // 75% toward the invalid probe B in x (fx = 0.75), on the (1,1) line.
    const pos: Vec3 = [(1.5 + 0.75) / 4, 1.5 / 4, 1.5 / 4];
    const r = sampleCascadeC0CPU(pos, grid, probeFn);
    // Only A survives ⇒ renormalised result is EXACTLY C·INV_PI, not a
    // weighted blend toward zero (and certainly not toward B's bogus 9s).
    expect(r.lo[0]).toBeCloseTo(C[0] * INV_PI, 10);
    expect(r.lo[1]).toBeCloseTo(C[1] * INV_PI, 10);
    expect(r.lo[2]).toBeCloseTo(C[2] * INV_PI, 10);
    // Surviving weight mass is the dropped-renormalised subset (< full 1).
    expect(r.weightSumSurviving).toBeGreaterThan(0);
    expect(r.weightSumSurviving).toBeLessThan(r.weightSumAll - 1e-9);
  });

  it('(e2) all-invalid corners ⇒ vec3f(0) (matches the old Wsum guard)', () => {
    const r = sampleCascadeC0CPU(
      [0.5, 0.5, 0.5],
      grid,
      () => ({ rgb: [5, 5, 5], wsum: 0 }), // every probe invalid
    );
    expect(r.lo).toEqual([0, 0, 0]);
    expect(Number.isNaN(r.lo[0])).toBe(false);
  });

  // (f) Disabled path: bit-identical vec3f(0) regardless of probe data.
  it('(f) enabled==0 returns EXACTLY vec3f(0)', () => {
    const r = sampleCascadeC0CPU(
      [0.5, 0.5, 0.5],
      grid,
      uniformProbes([0.9, 0.9, 0.9]),
      /* enabled */ false,
    );
    expect(r.lo[0]).toBe(0);
    expect(r.lo[1]).toBe(0);
    expect(r.lo[2]).toBe(0);
  });
});

describe('sampleCascadeC0 — WGSL structural pins', () => {
  it('keeps the (vec3f, vec3f) -> vec3f signature shade.wgsl depends on', () => {
    expect(SAMPLE_CASCADE_C0_WGSL).toContain(
      'fn sampleCascadeC0(worldPos: vec3f, normal: vec3f) -> vec3f {',
    );
  });

  it('the disabled path is the bit-identical early return vec3f(0)', () => {
    expect(SAMPLE_CASCADE_C0_WGSL).toMatch(
      /if \(rcParams\.enabled == 0u\) \{ return vec3f\(0\.0\); \}/,
    );
  });

  it('maps to probe-centre-relative grid coords (g = probeUV·count − 0.5)', () => {
    expect(SAMPLE_CASCADE_C0_WGSL).toContain('let g = probeUV * count - vec3f(0.5);');
    expect(SAMPLE_CASCADE_C0_WGSL).toContain('let g0 = floor(g);');
  });

  it('blends 8 corners with a trilinear product weight', () => {
    expect(SAMPLE_CASCADE_C0_WGSL).toMatch(/for \(var corner: u32 = 0u; corner < 8u;/);
    expect(SAMPLE_CASCADE_C0_WGSL).toContain('let wTri = mix(1.0 - f.x, f.x, dx)');
    expect(SAMPLE_CASCADE_C0_WGSL).toContain('* mix(1.0 - f.y, f.y, dy)');
    expect(SAMPLE_CASCADE_C0_WGSL).toContain('* mix(1.0 - f.z, f.z, dz);');
  });

  it('clamps each corner into valid grid bounds (edge handling)', () => {
    expect(SAMPLE_CASCADE_C0_WGSL).toContain(
      'let pi = vec3u(clamp(g0 + vec3f(dx, dy, dz), vec3f(0.0), cmax));',
    );
  });

  it('drops degenerate corners + renormalises by the surviving weight mass', () => {
    expect(SAMPLE_CASCADE_C0_WGSL).toContain('if (probe.w <= 1e-4) { continue; }');
    expect(SAMPLE_CASCADE_C0_WGSL).toContain('weightSum = weightSum + wTri;');
    expect(SAMPLE_CASCADE_C0_WGSL).toContain('return blendL * INV_PI / weightSum;');
  });

  it('no longer quantises to the nearest probe', () => {
    // The old nearest-probe quantisation form must be gone.
    expect(SAMPLE_CASCADE_C0_WGSL).not.toContain('let pi = vec3u(pf);');
    expect(SAMPLE_CASCADE_C0_WGSL).not.toMatch(/Quantise to nearest probe/);
  });
});
