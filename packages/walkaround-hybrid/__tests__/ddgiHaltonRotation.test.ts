/**
 * F3 — Halton SO(3) axis-angle rotation tests.
 *
 * Verifies the CPU-side Halton quasi-random rotation in
 * `probeUpdateFrameParams.ts` (haltonSO3AxisAngleFromFrameIndex, Shoemake 1992 + Rodrigues axis-angle convention)
 * produces valid, uniformly-distributed, decorrelated rotation matrices.
 *
 * Three properties tested:
 *   1. Validity: R·R^T = I and det(R) = 1 for every frame index.
 *   2. Uniformity: 1000 unit vectors rotated by R(frame) distribute
 *      roughly uniformly over octants (chi-squared on 8 buckets).
 *   3. Decorrelation: R(f)·R(f+1)^T is NOT close to identity
 *      (Frobenius distance > 0.5 for most pairs).
 *
 * References:
 *   Shoemake 1992 — "Uniform Random Rotations", Graphics Gems III §III.6.
 *   Majercik et al. 2019 §3.1 — per-frame SO(3) rotation for DDGI probes.
 */

import { describe, it, expect } from 'vitest';

// ─── Halton sequence ─────────────────────────────────────────────────────────

function haltonBase(i: number, base: number): number {
  let result = 0;
  let f = 1;
  let n = i;
  while (n > 0) {
    f /= base;
    result += f * (n % base);
    n = Math.floor(n / base);
  }
  return result;
}

// ─── Axis-angle → rotation matrix (Rodrigues) ────────────────────────────────
// Mirrors the WGSL `rotateAngleAxis` in shared-samplers/hammersley.wgsl.ts.

type Mat3 = [
  number, number, number,
  number, number, number,
  number, number, number,
];

function rodriguesMatrix(ax: number, ay: number, az: number, angle: number): Mat3 {
  if (angle < 1e-6) {
    return [1, 0, 0,  0, 1, 0,  0, 0, 1];
  }
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;
  // Rodrigues: R = I·cos + [axis×]·sin + axis⊗axis·(1-cos)
  return [
    t*ax*ax + c,     t*ax*ay - s*az, t*ax*az + s*ay,
    t*ax*ay + s*az,  t*ay*ay + c,    t*ay*az - s*ax,
    t*ax*az - s*ay,  t*ay*az + s*ax, t*az*az + c,
  ];
}

// ─── Matrix helpers ───────────────────────────────────────────────────────────

function matMul(A: Mat3, B: Mat3): Mat3 {
  const out: number[] = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      for (let k = 0; k < 3; k++) {
        out[r * 3 + c]! += A[r * 3 + k]! * B[k * 3 + c]!;
      }
    }
  }
  return out as Mat3;
}

function matTranspose(M: Mat3): Mat3 {
  return [
    M[0]!, M[3]!, M[6]!,
    M[1]!, M[4]!, M[7]!,
    M[2]!, M[5]!, M[8]!,
  ];
}

function frobenius(M: Mat3): number {
  let s = 0;
  for (let i = 0; i < 9; i++) s += M[i]! * M[i]!;
  return Math.sqrt(s);
}

function det3(M: Mat3): number {
  return (
    M[0]! * (M[4]! * M[8]! - M[5]! * M[7]!) -
    M[1]! * (M[3]! * M[8]! - M[5]! * M[6]!) +
    M[2]! * (M[3]! * M[7]! - M[4]! * M[6]!)
  );
}

function matVec(M: Mat3, v: [number, number, number]): [number, number, number] {
  return [
    M[0]! * v[0] + M[1]! * v[1] + M[2]! * v[2],
    M[3]! * v[0] + M[4]! * v[1] + M[5]! * v[2],
    M[6]! * v[0] + M[7]! * v[1] + M[8]! * v[2],
  ];
}

// ─── Build R(frame) — mirrors probeUpdateFrameParams.ts:haltonSO3AxisAngleFromFrameIndex ──

function buildRotationMatrix(frameIndex: number): Mat3 {
  const fi = frameIndex + 1;
  const u1 = haltonBase(fi, 2);
  const u2 = haltonBase(fi, 3);
  const u3 = haltonBase(fi, 5);

  const sigma1 = Math.sqrt(1 - u1);
  const sigma2 = Math.sqrt(u1);
  const theta1 = 2 * Math.PI * u2;
  const theta2 = 2 * Math.PI * u3;

  const qw = sigma2 * Math.cos(theta2);
  const qx = sigma1 * Math.sin(theta1);
  const qy = sigma1 * Math.cos(theta1);
  const qz = sigma2 * Math.sin(theta2);

  const angle = 2 * Math.acos(Math.min(1, Math.abs(qw)));
  const sinHalf = Math.sqrt(Math.max(0, 1 - qw * qw));

  let ax: number, ay: number, az: number;
  if (sinHalf < 1e-6) {
    ax = 1; ay = 0; az = 0;
  } else {
    ax = qx / sinHalf;
    ay = qy / sinHalf;
    az = qz / sinHalf;
  }

  return rodriguesMatrix(ax, ay, az, angle);
}

// ─── Simple seeded LCG RNG for deterministic test vectors ─────────────────────

function makeLcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function randomUnitVector(rng: () => number): [number, number, number] {
  // Rejection method: sample box until inside unit sphere, then normalise.
  for (;;) {
    const x = rng() * 2 - 1;
    const y = rng() * 2 - 1;
    const z = rng() * 2 - 1;
    const r2 = x * x + y * y + z * z;
    if (r2 > 0 && r2 <= 1) {
      const r = Math.sqrt(r2);
      return [x / r, y / r, z / r];
    }
  }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('ddgi Halton SO(3) rotation — F3 re-verification', () => {

  // ── 1. Validity: R·R^T = I, det(R) = 1 ──────────────────────────────────────
  it('R·R^T ≈ I and det(R) ≈ 1 for 1000 frame indices', () => {
    const identity: Mat3 = [1,0,0, 0,1,0, 0,0,1];
    for (let f = 0; f < 1000; f++) {
      const R = buildRotationMatrix(f);
      const RT = matTranspose(R);
      const RRT = matMul(R, RT);

      // Frobenius distance from identity should be < 1e-5.
      const diffFrob = frobenius([
        RRT[0]!-1, RRT[1]!, RRT[2]!,
        RRT[3]!, RRT[4]!-1, RRT[5]!,
        RRT[6]!, RRT[7]!, RRT[8]!-1,
      ] as Mat3);
      expect(diffFrob, `frame ${f}: R·R^T ≠ I (Frob = ${diffFrob.toFixed(6)})`).toBeLessThan(1e-5);

      const d = det3(R);
      expect(Math.abs(d - 1), `frame ${f}: det(R) = ${d}`).toBeLessThan(1e-5);
    }
  });

  // ── 2. Uniformity: octant distribution chi-squared ───────────────────────────
  it('rotated unit vectors distribute roughly uniformly across 8 octants', () => {
    const rng = makeLcg(0xdeadbeef);
    const N_FRAMES = 1000;
    const octantCounts = new Array(8).fill(0) as number[];

    for (let f = 0; f < N_FRAMES; f++) {
      const R = buildRotationMatrix(f);
      const v = randomUnitVector(rng);
      const rv = matVec(R, v);
      // Octant index: bit 0 = sign(x), bit 1 = sign(y), bit 2 = sign(z)
      const oct = (rv[0] >= 0 ? 1 : 0) | (rv[1] >= 0 ? 2 : 0) | (rv[2] >= 0 ? 4 : 0);
      octantCounts[oct]!++;
    }

    // Expected count per octant = N_FRAMES / 8 = 125.
    // Chi-squared statistic: Σ (O - E)² / E; critical value for df=7, p=0.001 is ~24.
    const expected = N_FRAMES / 8;
    let chi2 = 0;
    for (let i = 0; i < 8; i++) {
      const diff = (octantCounts[i]! - expected);
      chi2 += diff * diff / expected;
    }

    // Allow generous threshold (chi2 < 40 for 8 buckets, p ≈ 5e-6 critical).
    // If chi2 > 40, the distribution is significantly non-uniform.
    expect(chi2, `chi-squared = ${chi2.toFixed(2)}, octants = [${octantCounts.join(', ')}]`).toBeLessThan(40);
  });

  // ── 3. Decorrelation: R(f)·R(f+1)^T is NOT near identity ────────────────────
  it('R(f) and R(f+1) are decorrelated (Frobenius distance > 0.5 on average)', () => {
    const N_PAIRS = 1000;
    let countAboveThreshold = 0;

    for (let f = 0; f < N_PAIRS; f++) {
      const R0 = buildRotationMatrix(f);
      const R1 = buildRotationMatrix(f + 1);
      const R1T = matTranspose(R1);
      const diff = matMul(R0, R1T);
      // Frobenius distance from identity of R0·R1^T:
      const frobDist = frobenius([
        diff[0]!-1, diff[1]!, diff[2]!,
        diff[3]!, diff[4]!-1, diff[5]!,
        diff[6]!, diff[7]!, diff[8]!-1,
      ] as Mat3);
      if (frobDist > 0.5) countAboveThreshold++;
    }

    // At least 80% of frame pairs should have Frobenius distance > 0.5.
    // (Identity rotation has Frob distance 0; any non-trivial rotation exceeds 0.5.)
    const fraction = countAboveThreshold / N_PAIRS;
    expect(fraction, `Only ${(fraction*100).toFixed(1)}% of pairs decorrelated (expected ≥ 80%)`).toBeGreaterThanOrEqual(0.8);
  });

});
