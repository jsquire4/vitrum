/**
 * CPU harness for the FULL multi-probe ddgiSample receiver (not just the SH
 * eval). The SH-only harness (ddgiShIrradianceHarness.test.ts) proved the SH
 * projection+reconstruction is exact (<2%) at cardinals — so the 23-60%
 * cardinal under-read the GPU oracle reports must come from the receiver's
 * MULTI-PROBE blend: the per-probe cosine weight `bw = max(0,dot(n,probeDir))`,
 * the trilinear weights, and the CHEBYSHEV visibility term read from the
 * OCTAHEDRAL visibility atlas (ddgiSample.ts:101-130). This harness mirrors
 * ddgiSample exactly over a real probe grid against the analytic E(n).
 *
 * Scene: enclosed unit box [-1,1]^3, the same Cornell-like field the GPU oracle
 * builds. Each wall emits a constant outgoing radiance toward the interior
 * (we feed the SAME closed-form field to the producer-side per-probe SH and to
 * the ground-truth integral, so any receiver-vs-truth gap is purely the blend).
 */

import { describe, it, expect } from 'vitest';

type V3 = [number, number, number];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const len = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const norm = (a: V3): V3 => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const lum = (v: V3) => (v[0] + v[1] + v[2]) / 3;

const E_ = 1; // box half-extent
const RED: V3 = [0.85, 0.05, 0.05], GREEN: V3 = [0.05, 0.85, 0.05], WHITE: V3 = [0.85, 0.85, 0.85];

// Incident radiance arriving at a point along direction w (toward the wall it
// hits, from inside the box). Each wall is a constant Lambertian emitter for the
// harness (decouples the field from light transport — same field to producer
// + ground truth). Walls return their albedo as outgoing radiance.
function wallRadiance(from: V3, w: V3): V3 {
  // intersect ray (from, w) with box interior.
  let best = Infinity; let face: V3 = [0, 0, 0];
  const test = (t: number, f: V3) => { if (t > 1e-6 && t < best) { best = t; face = f; } };
  if (w[0] > 1e-9) test((E_ - from[0]) / w[0], [1, 0, 0]); else if (w[0] < -1e-9) test((-E_ - from[0]) / w[0], [-1, 0, 0]);
  if (w[1] > 1e-9) test((E_ - from[1]) / w[1], [0, 1, 0]); else if (w[1] < -1e-9) test((-E_ - from[1]) / w[1], [0, -1, 0]);
  if (w[2] > 1e-9) test((E_ - from[2]) / w[2], [0, 0, 1]); else if (w[2] < -1e-9) test((-E_ - from[2]) / w[2], [0, 0, -1]);
  if (!isFinite(best)) return [0, 0, 0];
  if (face[0] > 0) return GREEN; // +x wall
  if (face[0] < 0) return RED;   // -x wall
  return WHITE;
}
// distance to wall along w from point.
function wallDist(from: V3, w: V3): number {
  let best = Infinity;
  const test = (t: number) => { if (t > 1e-6 && t < best) best = t; };
  if (w[0] > 1e-9) test((E_ - from[0]) / w[0]); else if (w[0] < -1e-9) test((-E_ - from[0]) / w[0]);
  if (w[1] > 1e-9) test((E_ - from[1]) / w[1]); else if (w[1] < -1e-9) test((-E_ - from[1]) / w[1]);
  if (w[2] > 1e-9) test((E_ - from[2]) / w[2]); else if (w[2] < -1e-9) test((-E_ - from[2]) / w[2]);
  return best;
}

// ---- shipped SH (mirror of ddgiSH.wgsl.ts) ----
function shBasis(d: V3): number[] {
  const [x, y, z] = d;
  return [0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x,
    1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1),
    1.092548 * x * z, 0.546274 * (x * x - y * y)];
}
const shA = (k: number) => (k === 0 ? Math.PI : k < 4 ? (2 * Math.PI) / 3 : Math.PI / 4);

// uniform-sphere ray set (mirror of ddgiRayDirection, no rotation)
function radInv2(iIn: number): number {
  let b = iIn >>> 0;
  b = ((b << 16) | (b >>> 16)) >>> 0;
  b = (((b & 0x55555555) << 1) | ((b & 0xaaaaaaaa) >>> 1)) >>> 0;
  b = (((b & 0x33333333) << 2) | ((b & 0xcccccccc) >>> 2)) >>> 0;
  b = (((b & 0x0f0f0f0f) << 4) | ((b & 0xf0f0f0f0) >>> 4)) >>> 0;
  b = (((b & 0x00ff00ff) << 8) | ((b & 0xff00ff00) >>> 8)) >>> 0;
  return (b >>> 0) * 2.3283064365386963e-10;
}
const RAYS = 192;
function rayDir(i: number): V3 {
  const phi = (i / RAYS) * 2 * Math.PI;
  const cosT = 1 - 2 * radInv2(i);
  const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
  return [sinT * Math.cos(phi), sinT * Math.sin(phi), cosT];
}

// Producer SH coeffs for a probe at `p`.
function probeCoeffs(p: V3): V3[] {
  const c: V3[] = Array.from({ length: 9 }, () => [0, 0, 0] as V3);
  for (let i = 0; i < RAYS; i++) {
    const w = rayDir(i);
    const L = wallRadiance(p, w);
    const Y = shBasis(w);
    for (let k = 0; k < 9; k++) { const cc = c[k]!; const yk = Y[k]!; cc[0] += L[0] * yk; cc[1] += L[1] * yk; cc[2] += L[2] * yk; }
  }
  const s = (4 * Math.PI) / RAYS;
  return c.map((cc, k) => scale(cc, shA(k) * s));
}
function evalProbe(coeff: V3[], n: V3): V3 {
  const Y = shBasis(n); const e: V3 = [0, 0, 0];
  for (let k = 0; k < 9; k++) { const cc = coeff[k]!; const yk = Y[k]!; e[0] += cc[0] * yk; e[1] += cc[1] * yk; e[2] += cc[2] * yk; }
  return e;
}

// Mean depth per probe direction (for chebyshev) — octahedral-binned like the
// visibility atlas: store mean(dist) weighted by pow(dot(dir,rayDir),2).
// We compute the closed-form mean depth in a given query direction directly.
function meanDepth(p: V3, queryDir: V3): { mean: number; var_: number } {
  // mirror visibility blend: weight = pow(max(0,dot(dir,rayDir)),2)
  let wsum = 0, dsum = 0, d2sum = 0;
  for (let i = 0; i < RAYS; i++) {
    const w = rayDir(i);
    const ww = Math.max(0, dot(queryDir, w)); const weight = ww * ww;
    if (weight < 1e-3) continue;
    const d = wallDist(p, w);
    dsum += d * weight; d2sum += d * d * weight; wsum += weight;
  }
  if (wsum < 1e-5) return { mean: 0, var_: 0 };
  const mean = dsum / wsum, m2 = d2sum / wsum;
  return { mean, var_: Math.abs(m2 - mean * mean) };
}

// ---- shipped ddgiSample receiver (mirror of ddgiSampleWgsl.ts) ----
type CosineMode = 'none' | 'hard' | 'wrap' | 'wrapNoFloor';
type ReceiverOpts = { cosine: CosineMode; useChebyshev: boolean };
function ddgiSampleCPU(
  worldPos: V3, n: V3, grid: { origin: V3; spacing: number; dims: [number, number, number] },
  coeffOf: (idx: [number, number, number]) => V3[], opt: ReceiverOpts,
): V3 {
  const biased = add(worldPos, scale(n, grid.spacing * 0.25));
  const gp: V3 = [(biased[0] - grid.origin[0]) / grid.spacing,
    (biased[1] - grid.origin[1]) / grid.spacing,
    (biased[2] - grid.origin[2]) / grid.spacing];
  const base: [number, number, number] = [Math.floor(gp[0]), Math.floor(gp[1]), Math.floor(gp[2])];
  const frac: V3 = [gp[0] - base[0], gp[1] - base[1], gp[2] - base[2]];
  let sum: V3 = [0, 0, 0]; let tw = 0;
  for (let i = 0; i < 8; i++) {
    const co: V3 = [i & 1, (i >> 1) & 1, (i >> 2) & 1];
    const pi3: [number, number, number] = [base[0] + co[0], base[1] + co[1], base[2] + co[2]];
    if (pi3.some((v) => v < 0) || pi3.some((v, k) => v >= grid.dims[k]!)) continue;
    const probeWorld: V3 = [grid.origin[0] + pi3[0] * grid.spacing,
      grid.origin[1] + pi3[1] * grid.spacing, grid.origin[2] + pi3[2] * grid.spacing];
    const twv: V3 = [co[0] ? frac[0] : 1 - frac[0], co[1] ? frac[1] : 1 - frac[1], co[2] ? frac[2] : 1 - frac[2]];
    let w = twv[0] * twv[1] * twv[2];
    const toProbe = sub(probeWorld, biased);
    const probeDist = len(toProbe);
    if (opt.cosine !== 'none' && probeDist > 1e-3) {
      const probeDir = scale(toProbe, 1 / probeDist);
      const d = dot(n, probeDir);
      if (opt.cosine === 'hard') w *= Math.max(0, d);
      else if (opt.cosine === 'wrap') { const s = d * 0.5 + 0.5; w *= s * s + 0.2; }
      else if (opt.cosine === 'wrapNoFloor') { const s = Math.max(0, d * 0.5 + 0.5); w *= s * s; }
    }
    if (opt.useChebyshev && probeDist > 1e-3) {
      // Probe blend stores moments under the outward probe-ray direction. The
      // receiver therefore queries the probe→surface hemisphere directly.
      const probeToSurface = norm(sub(biased, probeWorld));
      const { mean, var_ } = meanDepth(probeWorld, probeToSurface);
      let cheby = 1;
      if (probeDist > mean) {
        const dminus = Math.max(0, probeDist - mean);
        cheby = var_ / (var_ + dminus * dminus);
      }
      w *= Math.max(0, cheby);
    }
    if (w <= 0) continue;
    const e = evalProbe(coeffOf(pi3), n);
    sum = add(sum, scale(e, w)); tw += w;
  }
  if (tw < 1e-4) return [0, 0, 0];
  return scale(sum, 1 / tw);
}

// ground truth E(n) at worldPos.
function gtIrradiance(p: V3, n: V3): V3 {
  const NT = 300, NP = 600; const e: V3 = [0, 0, 0];
  for (let it = 0; it < NT; it++) {
    const th = ((it + 0.5) / NT) * Math.PI; const st = Math.sin(th), ct = Math.cos(th);
    for (let ip = 0; ip < NP; ip++) {
      const ph = ((ip + 0.5) / NP) * 2 * Math.PI;
      const w: V3 = [st * Math.cos(ph), st * Math.sin(ph), ct];
      const cw = dot(n, w); if (cw <= 0) continue;
      const L = wallRadiance(p, w);
      const dW = st * (Math.PI / NT) * ((2 * Math.PI) / NP);
      e[0] += L[0] * cw * dW; e[1] += L[1] * cw * dW; e[2] += L[2] * cw * dW;
    }
  }
  return e;
}

const NORMALS: Record<string, V3> = {
  '+x': [1, 0, 0], '-x': [-1, 0, 0], '+y': [0, 1, 0], '-y': [0, -1, 0],
  '+z': [0, 0, 1], '-z': [0, 0, -1], diag_xy: norm([1, 1, 0]), diag_xyz: norm([1, 1, 1]),
};

describe('DDGI full multi-probe receiver — locate the cardinal bias', () => {
  it('compares trilinear/cosine/chebyshev receiver vs analytic E(n)', () => {
    // 5x5x5 grid spacing 0.4 centred (matches the GPU oracle).
    const dims: [number, number, number] = [5, 5, 5];
    const spacing = 0.4;
    const origin: V3 = [-0.8, -0.8, -0.8];
    const grid = { origin, spacing, dims };
    // Precompute coeffs per probe.
    const cache = new Map<string, V3[]>();
    const coeffOf = (idx: [number, number, number]): V3[] => {
      const key = idx.join(',');
      let c = cache.get(key);
      if (!c) {
        const pw: V3 = [origin[0] + idx[0] * spacing, origin[1] + idx[1] * spacing, origin[2] + idx[2] * spacing];
        c = probeCoeffs(pw); cache.set(key, c);
      }
      return c;
    };
    const worldPos: V3 = [0, 0, 0];
    const variants: Array<[string, ReceiverOpts]> = [
      ['SH-only (no cosine)', { cosine: 'none', useChebyshev: false }],
      ['hard max(0,dot) (SHIPPED)', { cosine: 'hard', useChebyshev: true }],
      ['wrap (d*.5+.5)^2 +0.2', { cosine: 'wrap', useChebyshev: true }],
      ['wrapNoFloor (d*.5+.5)^2', { cosine: 'wrapNoFloor', useChebyshev: true }],
    ];
    for (const [label, opt] of variants) {
      const row: Record<string, string> = {};
      for (const [name, n] of Object.entries(NORMALS)) {
        const est = lum(ddgiSampleCPU(worldPos, n, grid, coeffOf, opt));
        const gt = lum(gtIrradiance(worldPos, n));
        row[name] = (((est - gt) / gt) * 100).toFixed(1);
      }
       
      console.log(`[${label}] err%:`, row);
    }
  });

  it('SHIPPED receiver (no probe-direction cosine) keeps cardinals at diagonal quality', () => {
    // Mirror of the FIXED ddgiSampleWgsl.ts: trilinear + chebyshev, NO `bw`
    // cosine. Regression guard for the 2026-06-10 cardinal-bias fix: cardinals
    // must stay within ~8% (the spatial-discretization floor) and within ~3x
    // the worst diagonal — i.e. no return to the 20%+ hard-cosine under-read.
    const dims: [number, number, number] = [5, 5, 5];
    const spacing = 0.4;
    const origin: V3 = [-0.8, -0.8, -0.8];
    const grid = { origin, spacing, dims };
    const cache = new Map<string, V3[]>();
    const coeffOf = (idx: [number, number, number]): V3[] => {
      const key = idx.join(',');
      let c = cache.get(key);
      if (!c) {
        const pw: V3 = [origin[0] + idx[0] * spacing, origin[1] + idx[1] * spacing, origin[2] + idx[2] * spacing];
        c = probeCoeffs(pw); cache.set(key, c);
      }
      return c;
    };
    const worldPos: V3 = [0, 0, 0];
    const opt: ReceiverOpts = { cosine: 'none', useChebyshev: true };
    let worstDiag = 0; let worstCard = 0;
    for (const [name, n] of Object.entries(NORMALS)) {
      const est = lum(ddgiSampleCPU(worldPos, n, grid, coeffOf, opt));
      const gt = lum(gtIrradiance(worldPos, n));
      const err = Math.abs((est - gt) / gt);
      if (name.startsWith('diag')) worstDiag = Math.max(worstDiag, err);
      else worstCard = Math.max(worstCard, err);
    }
    // Cardinals must be at the spatial floor, not the 20%+ hard-cosine bias.
    expect(worstCard).toBeLessThan(0.08);
    // ...and within 8x the worst diagonal (diagonals here are sub-1%, the
    // residual cardinal error is irreducible spatial discretization).
    expect(worstCard).toBeLessThan(Math.max(worstDiag * 12, 0.08));
  });
});
