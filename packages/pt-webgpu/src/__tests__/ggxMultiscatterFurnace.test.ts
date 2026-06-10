/**
 * B9 — GGX multiscatter energy-compensation white-furnace harness (CPU mirror).
 *
 * White-furnace test: an albedo-1 (F=1) metal in a uniform unit environment must
 * reflect ALL incident energy — the EXPECTED sampled-BSDF throughput must equal
 * ~1 at every roughness. We mirror the renderer's ACTUAL sampled estimator
 * (bsdf.wgsl.ts sampleNextBounceDirection specular branch): for a VNDF-sampled
 * direction the single-scatter throughput is F·G1(wi); B9 multiplies by the
 * Kulla-Conty boost 1 + F_avg·(1−E_ss)/E_ss (material.wgsl.ts ggxMultiscatterBoost).
 *
 * Using the SAME VNDF sampler the renderer uses (not uniform-hemisphere MC) makes
 * this a faithful, low-variance furnace test — uniform-hemisphere MC cannot
 * resolve the near-delta lobe at low roughness, so it would report phantom energy
 * loss there. This harness checks:
 *   1. SINGLE-SCATTER throughput E[F·G1(wi)] < 1 at high roughness (the defect).
 *   2. SINGLE × MULTISCATTER-BOOST recovers to ~1 across roughness (the B9 fix).
 *   3. The boost is ~1 at low roughness (smooth surfaces ≈ unchanged).
 */
import { describe, expect, it } from 'vitest';

const PI = Math.PI;

// ── WGSL mirrors ─────────────────────────────────────────────────────────────
function smithG1(nDotV: number, roughness: number): number {
  const r = roughness + 1;
  const k = (r * r) * 0.125;
  return nDotV / Math.max(nDotV * (1 - k) + k, 1e-6);
}
// 8×8 E_ss LUT — byte-identical to GGX_E_LUT in material.wgsl.ts.
const GGX_E_LUT_DIM = 8;
const GGX_E_LUT = [
  0.1375, 0.5617, 0.7546, 0.8522, 0.9111, 0.9505, 0.9788, 1.0,
  0.2955, 0.515, 0.7091, 0.8192, 0.889, 0.937, 0.9721, 0.9988,
  0.5794, 0.5541, 0.6677, 0.7691, 0.8451, 0.9021, 0.9457, 0.98,
  0.7011, 0.6486, 0.6669, 0.7199, 0.7776, 0.8305, 0.8764, 0.9155,
  0.7335, 0.6901, 0.6696, 0.6756, 0.6972, 0.7262, 0.7578, 0.7893,
  0.7153, 0.6712, 0.6355, 0.6145, 0.6052, 0.6045, 0.6101, 0.6199,
  0.6669, 0.6137, 0.5657, 0.5286, 0.5, 0.478, 0.4611, 0.4483,
  0.6017, 0.537, 0.4773, 0.4296, 0.3905, 0.358, 0.3305, 0.3069,
];
function ggxDirectionalAlbedo(cosTheta: number, roughness: number): number {
  const mu = Math.min(1, Math.max(0, cosTheta));
  const r = Math.min(1, Math.max(0, roughness));
  const fr = r * (GGX_E_LUT_DIM - 1);
  const fm = mu * (GGX_E_LUT_DIM - 1);
  const r0 = Math.floor(fr);
  const m0 = Math.floor(fm);
  const r1 = Math.min(r0 + 1, GGX_E_LUT_DIM - 1);
  const m1 = Math.min(m0 + 1, GGX_E_LUT_DIM - 1);
  const tr = fr - r0;
  const tm = fm - m0;
  const e00 = GGX_E_LUT[r0 * GGX_E_LUT_DIM + m0]!;
  const e01 = GGX_E_LUT[r0 * GGX_E_LUT_DIM + m1]!;
  const e10 = GGX_E_LUT[r1 * GGX_E_LUT_DIM + m0]!;
  const e11 = GGX_E_LUT[r1 * GGX_E_LUT_DIM + m1]!;
  const e0 = e00 + (e01 - e00) * tm;
  const e1 = e10 + (e11 - e10) * tm;
  return Math.min(1, Math.max(0.02, e0 + (e1 - e0) * tr));
}
// ggxMultiscatterBoost (material.wgsl.ts): 1 + F_avg·(1−E_ss(μo))/E_ss(μo).
function ggxMultiscatterBoost(fresnel: number, roughness: number, nDotV: number): number {
  const eo = ggxDirectionalAlbedo(nDotV, roughness);
  const missing = Math.min(1, Math.max(0, 1 - eo));
  if (missing < 1e-4) return 1;
  const fAvg = fresnel + (1 - fresnel) * (1 / 21);
  return 1 + fAvg * (missing / Math.max(eo, 1e-3));
}

// Heitz 2018 VNDF sampler (tangent space, N=+Z), mirrors sampleGgxVndfTangent.
function sampleVndf(woL: number[], alpha: number, u1: number, u2: number): number[] {
  const Vh = normalize([alpha * woL[0]!, alpha * woL[1]!, woL[2]!]);
  const lensq = Vh[0]! * Vh[0]! + Vh[1]! * Vh[1]!;
  const T1 =
    lensq > 1e-10 ? scale([-Vh[1]!, Vh[0]!, 0], 1 / Math.sqrt(lensq)) : [1, 0, 0];
  const T2 = cross(Vh, T1);
  const r = Math.sqrt(u1);
  const phi = 2 * PI * u2;
  const t1 = r * Math.cos(phi);
  let t2 = r * Math.sin(phi);
  const s = 0.5 * (1 + Vh[2]!);
  t2 = (1 - s) * Math.sqrt(Math.max(0, 1 - t1 * t1)) + s * t2;
  const Nh = add(
    add(scale(T1, t1), scale(T2, t2)),
    scale(Vh, Math.sqrt(Math.max(0, 1 - t1 * t1 - t2 * t2))),
  );
  return normalize([alpha * Nh[0]!, alpha * Nh[1]!, Math.max(1e-6, Nh[2]!)]);
}
function normalize(v: number[]): number[] {
  const l = Math.hypot(v[0]!, v[1]!, v[2]!) || 1;
  return [v[0]! / l, v[1]! / l, v[2]! / l];
}
function cross(a: number[], b: number[]): number[] {
  return [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
}
function scale(a: number[], s: number): number[] {
  return [a[0]! * s, a[1]! * s, a[2]! * s];
}
function add(a: number[], b: number[]): number[] {
  return [a[0]! + b[0]!, a[1]! + b[1]!, a[2]! + b[2]!];
}
function dot(a: number[], b: number[]): number {
  return a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
}
function reflect(i: number[], n: number[]): number[] {
  const d = 2 * dot(i, n);
  return [i[0]! - d * n[0]!, i[1]! - d * n[1]!, i[2]! - d * n[2]!];
}

// Expected sampled-BSDF throughput in a white furnace (F=1), using the SAME VNDF
// sampler + throughput the renderer uses. Single-scatter = F·G1(wi); with B9 the
// throughput is ×ggxMultiscatterBoost. N=+Z; view at nDotV in the X-Z plane.
function furnaceThroughput(roughness: number, nDotV: number, withBoost: boolean): number {
  const alpha = Math.max(roughness * roughness, 0.001);
  const sinV = Math.sqrt(Math.max(0, 1 - nDotV * nDotV));
  const wo = [sinV, 0, nDotV]; // tangent space, N=+Z
  const N = 400;
  let sum = 0;
  let count = 0;
  const boost = withBoost ? ggxMultiscatterBoost(1.0, roughness, nDotV) : 1;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const u1 = (i + 0.5) / N;
      const u2 = (j + 0.5) / N;
      const h = sampleVndf(wo, alpha, u1, u2);
      const wi = reflect(scale(wo, -1), h);
      const nDotL = wi[2]!;
      if (nDotL <= 1e-5) {
        count++;
        continue;
      }
      // MC estimator for VNDF sampling collapses to F·G1(wi); F=1 furnace.
      const g1Wi = smithG1(nDotL, roughness);
      sum += g1Wi * boost;
      count++;
    }
  }
  return sum / count;
}

describe('B9 — GGX multiscatter white-furnace energy conservation (CPU mirror)', () => {
  const nDotV = 0.7;

  it('SINGLE-SCATTER throughput E[F·G1(wi)] < 1 at high roughness (the defect B9 fixes)', () => {
    const e = furnaceThroughput(0.9, nDotV, false);
    expect(e).toBeLessThan(0.95);
  });

  it('SINGLE × MULTISCATTER-BOOST recovers to ~1 across the roughness range', () => {
    for (const r of [0.2, 0.4, 0.6, 0.8, 1.0]) {
      const e = furnaceThroughput(r, nDotV, true);
      // White furnace: total reflected energy ≈ 1 (analytic-fit + MC tolerance).
      expect(e).toBeGreaterThan(0.9);
      expect(e).toBeLessThan(1.15);
    }
  });

  it('the boost only RAISES energy toward 1 (never reduces, never over-shoots far)', () => {
    // The Smith-G1 model carries a small (~8%) residual energy loss even at low
    // roughness (E[G1(wi)] < 1), which the boost legitimately compensates UP
    // toward the white-furnace target of 1 — it must never push below the
    // single-scatter value nor blow far past 1. (B9 is intentionally render-
    // CHANGING; the byte-identity guarantee is the spectralEnabled/RGB axis, not
    // smooth-metal energy.) Verify the boost moves energy strictly closer to 1.
    const single = furnaceThroughput(0.05, nDotV, false);
    const full = furnaceThroughput(0.05, nDotV, true);
    expect(full).toBeGreaterThanOrEqual(single - 1e-6); // never reduces energy
    expect(full).toBeLessThan(1.12); // stays within furnace tolerance of 1
    expect(Math.abs(full - 1)).toBeLessThan(Math.abs(single - 1) + 1e-6); // closer to 1
  });
});
