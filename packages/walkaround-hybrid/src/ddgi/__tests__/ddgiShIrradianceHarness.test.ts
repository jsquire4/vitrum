/**
 * CPU self-validating harness for the DDGI L2-SH irradiance estimator as
 * SHIPPED in the GPU pass graph (probeUpdateBlend.wgsl.ts projection +
 * ddgiSampleSHProbe / ddgiSample reconstruction, via ddgiSH.wgsl.ts).
 *
 * Goal (HARDWARE-VALIDATION-NEEDS.md "DDGI fidelity vs ground truth"): the
 * GPU oracle reports axis-aligned (cardinal) receiver normals under-read
 * their DDGI irradiance by 23-60% vs CPU f64 ground truth while DIAGONAL
 * normals are 2-4%. Border-store + seam-vertex hypotheses are ground-truth
 * REJECTED. This harness re-implements the producer+consumer SH math in pure
 * f64 TS to (1) reproduce the cardinal bias CPU-side, (2) localize the wrong
 * term, (3) verify the fix to <=5% at cardinals with no diagonal regression.
 *
 * Method: for several analytic incident-radiance fields L(w) with a CLOSED-
 * FORM diffuse irradiance E(n) = integral_hemisphere(n) L(w) (n.w) dw, run the
 * shipped Monte-Carlo SH projection over the 192 uniform-sphere rays the GPU
 * actually casts (ddgiRayDirection), then reconstruct E(n) and compare to the
 * analytic truth at cardinal and diagonal normals.
 */

import { describe, it, expect } from 'vitest';

// ---------------------------------------------------------------------------
// Shipped ray distribution (mirror of shared-samplers hammersley.wgsl.ts).
// ---------------------------------------------------------------------------
function radicalInverseBase2(iIn: number): number {
  // hammersleyUniform uses van der Corput base-2 for the 2nd coord.
  let i = iIn >>> 0;
  let bits = i;
  bits = ((bits << 16) | (bits >>> 16)) >>> 0;
  bits = (((bits & 0x55555555) << 1) | ((bits & 0xaaaaaaaa) >>> 1)) >>> 0;
  bits = (((bits & 0x33333333) << 2) | ((bits & 0xcccccccc) >>> 2)) >>> 0;
  bits = (((bits & 0x0f0f0f0f) << 4) | ((bits & 0xf0f0f0f0) >>> 4)) >>> 0;
  bits = (((bits & 0x00ff00ff) << 8) | ((bits & 0xff00ff00) >>> 8)) >>> 0;
  return (bits >>> 0) * 2.3283064365386963e-10;
}

type V3 = [number, number, number];

function uniformSphere(ux: number, uy: number): V3 {
  const phi = ux * 2 * Math.PI;
  const cosT = 1 - 2 * uy;
  const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
  return [sinT * Math.cos(phi), sinT * Math.sin(phi), cosT];
}

/** Mirror of ddgiRayDirection with zero rotation (randomRotation accumulates
 *  over frames; a converged probe averages over rotations -> equivalent to the
 *  unrotated stratified set in expectation). */
function ddgiRayDirection(i: number, n: number): V3 {
  const ux = i / n;
  const uy = radicalInverseBase2(i);
  return uniformSphere(ux, uy);
}

// ---------------------------------------------------------------------------
// Shipped SH basis + cosine-convolution coefficients (mirror of ddgiSH.wgsl.ts).
// ---------------------------------------------------------------------------
function shBasis(d: V3): number[] {
  const [x, y, z] = d;
  return [
    0.282095,
    0.488603 * y, 0.488603 * z, 0.488603 * x,
    1.092548 * x * y,
    1.092548 * y * z,
    0.315392 * (3 * z * z - 1),
    1.092548 * x * z,
    0.546274 * (x * x - y * y),
  ];
}

/** Shipped per-band cosine convolution A_l (Ramamoorthi-Hanrahan). */
function shCosineA(k: number): number {
  if (k === 0) return Math.PI;
  if (k < 4) return (2 * Math.PI) / 3;
  return Math.PI / 4;
}

// ---------------------------------------------------------------------------
// Shipped producer: project the 192 ray radiances onto SH, cosine-convolve.
//   c_k = (4PI/N) * sum_i L_i * Y_k(w_i)     [blend pass lines 130-147]
//   E_lm = A_l * c_k
// ---------------------------------------------------------------------------
const RAYS_PER_PROBE = 192;

function projectSH(radiance: (d: V3) => V3): V3[] {
  const coeff: V3[] = Array.from({ length: 9 }, () => [0, 0, 0] as V3);
  for (let i = 0; i < RAYS_PER_PROBE; i++) {
    const w = ddgiRayDirection(i, RAYS_PER_PROBE);
    const L = radiance(w);
    const Y = shBasis(w);
    for (let k = 0; k < 9; k++) {
      const c = coeff[k]!; const yk = Y[k]!;
      c[0] += L[0] * yk; c[1] += L[1] * yk; c[2] += L[2] * yk;
    }
  }
  const scale = (4 * Math.PI) / RAYS_PER_PROBE;
  return coeff.map((c, k) => {
    const a = shCosineA(k) * scale;
    return [c[0] * a, c[1] * a, c[2] * a] as V3;
  });
}

/** Shipped receiver: E(n) = sum_k E_lm[k] * Y_k(n). */
function reconstruct(coeff: V3[], n: V3): V3 {
  const Y = shBasis(n);
  const e: V3 = [0, 0, 0];
  for (let k = 0; k < 9; k++) {
    const c = coeff[k]!;
    e[0] += c[0] * Y[k]!;
    e[1] += c[1] * Y[k]!;
    e[2] += c[2] * Y[k]!;
  }
  return e;
}

// ---------------------------------------------------------------------------
// Closed-form irradiance ground truth (high-resolution hemisphere quadrature
// in f64 — the same "CPU f64 ground truth" the GPU oracle uses).
//   E(n) = integral over hemisphere(n) of L(w) * max(0, n.w) dw
// ---------------------------------------------------------------------------
function dot(a: V3, b: V3): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

function groundTruthIrradiance(radiance: (d: V3) => V3, n: V3): V3 {
  // Dense spherical quadrature (theta x phi) with solid-angle weight sin(theta).
  const NT = 400, NP = 800;
  const e: V3 = [0, 0, 0];
  for (let it = 0; it < NT; it++) {
    const theta = (it + 0.5) / NT * Math.PI;
    const sinT = Math.sin(theta), cosT = Math.cos(theta);
    for (let ip = 0; ip < NP; ip++) {
      const phi = (ip + 0.5) / NP * 2 * Math.PI;
      const w: V3 = [sinT * Math.cos(phi), sinT * Math.sin(phi), cosT];
      const cw = dot(n, w);
      if (cw <= 0) continue;
      const L = radiance(w);
      const dW = sinT * (Math.PI / NT) * (2 * Math.PI / NP);
      e[0] += L[0] * cw * dW;
      e[1] += L[1] * cw * dW;
      e[2] += L[2] * cw * dW;
    }
  }
  return e;
}

// ---------------------------------------------------------------------------
// Test normal set: cardinals (the under-reading directions) + diagonals (the
// 2-4% control). Use luminance (avg of channels — fields below are grey).
// ---------------------------------------------------------------------------
const NORMALIZE = (v: V3): V3 => { const l = Math.hypot(...v); return [v[0]/l, v[1]/l, v[2]/l]; };
const NORMALS: Record<string, V3> = {
  '+x': [1, 0, 0], '-x': [-1, 0, 0],
  '+y': [0, 1, 0], '-y': [0, -1, 0],
  '+z': [0, 0, 1], '-z': [0, 0, -1],
  'diag_xy': NORMALIZE([1, 1, 0]),
  'diag_xyz': NORMALIZE([1, 1, 1]),
};

function lum(v: V3): number { return (v[0] + v[1] + v[2]) / 3; }

function errorTable(radiance: (d: V3) => V3): Record<string, number> {
  const coeff = projectSH(radiance);
  const out: Record<string, number> = {};
  for (const [name, n] of Object.entries(NORMALS)) {
    const est = lum(reconstruct(coeff, n));
    const gt = lum(groundTruthIrradiance(radiance, n));
    out[name] = (est - gt) / gt; // signed relative error
  }
  return out;
}

// Radiance fields ---------------------------------------------------------
const uniformSky = (): V3 => [1, 1, 1];
// Single coloured wall: radiance only from +x hemisphere region (constant in a cone).
const cosLobeUp = (d: V3): V3 => { const c = Math.max(0, d[1]); return [c, c, c]; };
// Directional-ish: brighter from -y (floor lit from below scenario the oracle flags worst).
const fromBelow = (d: V3): V3 => { const c = Math.max(0, -d[1]); return [c, c, c]; };
// Realistic enclosed box: 6 walls each a constant colour in their hemisphere
// (a probe in a Cornell-like room sees structured radiance: ceiling bright,
// floor dim, red +x wall, green -x wall). This is the field the GPU oracle
// actually projects.
const enclosedRoom = (d: V3): V3 => {
  // pick the dominant axis -> which wall the ray hits.
  const ax = Math.abs(d[0]), ay = Math.abs(d[1]), az = Math.abs(d[2]);
  if (ax >= ay && ax >= az) return d[0] > 0 ? [0.8, 0.1, 0.1] : [0.1, 0.8, 0.1]; // red / green
  if (ay >= ax && ay >= az) return d[1] > 0 ? [1.0, 1.0, 1.0] : [0.4, 0.4, 0.4]; // ceiling / floor
  return [0.7, 0.7, 0.7]; // front/back grey
};

// ---------------------------------------------------------------------------
// f16 (half-float) quantization — the atlas is rgba16float, so the stored SH
// coefficients are rounded to half precision before the receiver reads them.
// ---------------------------------------------------------------------------
function toF16(x: number): number {
  const f = new Float32Array(1); f[0] = x;
  const i = new Int32Array(f.buffer)[0]!;
  const sign = (i >> 16) & 0x8000;
  let exp = ((i >> 23) & 0xff) - 127 + 15;
  let mant = i & 0x7fffff;
  if (exp <= 0) {
    if (exp < -10) return sign === 0 ? 0 : -0;
    mant = (mant | 0x800000) >> (1 - exp);
    return f16ToF32((sign | (mant >> 13)) & 0xffff);
  } else if (exp >= 31) {
    return f16ToF32((sign | 0x7c00) & 0xffff);
  }
  return f16ToF32((sign | (exp << 10) | (mant >> 13)) & 0xffff);
}
function f16ToF32(h: number): number {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, m = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
}
function quantizeCoeffs(coeff: V3[]): V3[] {
  return coeff.map((c) => [toF16(c[0]), toF16(c[1]), toF16(c[2])] as V3);
}

describe('DDGI L2-SH irradiance estimator — CPU self-validating harness', () => {
  it('prints per-normal signed error for uniform sky', () => {
    const t = errorTable(uniformSky);
    // eslint-disable-next-line no-console
    console.log('UNIFORM SKY error %:', Object.fromEntries(
      Object.entries(t).map(([k, v]) => [k, (v * 100).toFixed(2)])));
    // uniform sky: E(n) = PI for all n; should be ~exact everywhere.
    for (const v of Object.values(t)) expect(Math.abs(v)).toBeLessThan(0.05);
  });

  it('prints per-normal signed error for cosine-up lobe', () => {
    const t = errorTable(cosLobeUp);
    // eslint-disable-next-line no-console
    console.log('COSINE-UP error %:', Object.fromEntries(
      Object.entries(t).map(([k, v]) => [k, (v * 100).toFixed(2)])));
  });

  it('prints per-normal signed error for from-below field (-y worst case)', () => {
    const t = errorTable(fromBelow);
    // eslint-disable-next-line no-console
    console.log('FROM-BELOW error %:', Object.fromEntries(
      Object.entries(t).map(([k, v]) => [k, (v * 100).toFixed(2)])));
  });

  it('enclosed room — f64 coeffs vs f16-quantized coeffs (rgba16float atlas)', () => {
    const coeffF64 = projectSH(enclosedRoom);
    const coeffF16 = quantizeCoeffs(coeffF64);
    const tF64: Record<string, string> = {};
    const tF16: Record<string, string> = {};
    for (const [name, n] of Object.entries(NORMALS)) {
      const gt = lum(groundTruthIrradiance(enclosedRoom, n));
      const eF64 = lum(reconstruct(coeffF64, n));
      const eF16 = lum(reconstruct(coeffF16, n));
      tF64[name] = (((eF64 - gt) / gt) * 100).toFixed(2);
      tF16[name] = (((eF16 - gt) / gt) * 100).toFixed(2);
    }
    // eslint-disable-next-line no-console
    console.log('ENCLOSED ROOM f64 error %:', tF64);
    // eslint-disable-next-line no-console
    console.log('ENCLOSED ROOM f16 error %:', tF16);
  });
});
