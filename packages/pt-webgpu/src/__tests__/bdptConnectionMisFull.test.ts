import { describe, expect, it } from 'vitest';
import {
  buildBDPTStrategyPDFs_full,
  bdptConnectionMIS_full,
  type BDPTFullVertex,
} from '@vitrum/shared-samplers';
import {
  assembleMergedConnectionPath,
  buildStrategyPdfs,
  powerHeuristicWeight,
  convertDensitySAtoArea,
  bdptConnectionMisFull,
  type EyeStackVertex,
  type LightStackVertex,
  type MergedVertex,
  type Vec3,
} from '../bdpt/bdptConnectionMisFull.js';

// ─────────────────────────────────────────────────────────────────────────────
// Full Veach §10.3 BDPT connection MIS — machine-epsilon match to the read-only
// oracle in @vitrum/shared-samplers (bdptMIS.ts). The pt-webgpu port
// (bdptConnectionMisFull.ts) is an INDEPENDENT reimplementation of the same
// recurrence; this suite proves the two agree to ~1e-12 on a varying-G,
// multi-bounce fixture and that the per-strategy weights sum to 1.
// ─────────────────────────────────────────────────────────────────────────────

function toOracleVertices(v: ReadonlyArray<MergedVertex>): BDPTFullVertex[] {
  return v.map((x) => ({
    position: x.position,
    normal: x.normal,
    pdfFwd: x.pdfFwd,
    pdfRev: x.pdfRev,
    isSpecular: x.isSpecular,
  }));
}

/**
 * A deterministic varying-G multi-bounce fixture: 2 light vertices (emitter +
 * one bounce) connected to an eye vertex with a 3-deep eye chain (E_2 = current
 * bounce, E_1, E_0, camera). Distances and normals are chosen so every
 * geometry term G differs (no degenerate equal-distance shortcut), exercising
 * the destination-cosine ConvertDensity Jacobian at every edge.
 */
function makeFixture(): {
  lightChain: LightStackVertex[];
  eyeChain: EyeStackVertex[];
  camera: { position: Vec3; normal: Vec3 };
  eyeBrdfPdf: (wo: Vec3, wi: Vec3) => number;
} {
  const lightChain: LightStackVertex[] = [
    {
      position: [0, 5, 0],
      normal: [0, -1, 0],
      pdfFwd: 0.37, // joint emitter area×dir density (endpoint, area-measure)
      pdfRev: 0.19,
      isSpecular: false,
    },
    {
      position: [1.3, 3.1, 0.4],
      normal: normalize([0.2, 1, 0.1]),
      pdfFwd: 0.51,
      pdfRev: 0.27,
      isSpecular: false,
    },
  ];
  const eyeChain: EyeStackVertex[] = [
    // E_0 = primary hit
    {
      position: [-2, 0.2, 1],
      normal: normalize([0.1, 1, 0.05]),
      pdfFwd: 0.44,
      pdfRev: 0.61,
      isSpecular: false,
    },
    // E_1
    {
      position: [-0.8, 1.0, 0.7],
      normal: normalize([0.3, 0.9, -0.2]),
      pdfFwd: 0.29,
      pdfRev: 0.48,
      isSpecular: false,
    },
    // E_2 = current bounce (eye connection vertex)
    {
      position: [0.4, 2.0, 0.55],
      normal: normalize([-0.15, 0.95, 0.27]),
      pdfFwd: 0.33,
      pdfRev: 0.42,
      isSpecular: false,
    },
  ];
  const camera = { position: [-4, -1.5, 2.5] as Vec3, normal: [0, 0, 1] as Vec3 };
  // Non-symmetric eye BSDF directional pdf (asymmetric in wo/wi) to exercise D1.
  const eyeBrdfPdf = (wo: Vec3, wi: Vec3): number => {
    const a = Math.abs(wo[0] * 0.7 + wo[1] * 0.2 + wo[2] * 0.1);
    const b = Math.abs(wi[0] * 0.1 + wi[1] * 0.6 + wi[2] * 0.3);
    return 0.05 + 0.4 * a + 0.25 * b + 0.1 * (a * b);
  };
  return { lightChain, eyeChain, camera, eyeBrdfPdf };
}

function normalize(a: Vec3): Vec3 {
  const l = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / l, a[1] / l, a[2] / l];
}

describe('bdptConnectionMisFull — §10.3 port vs shared-samplers oracle', () => {
  it('ConvertDensity matches the PBRT destination-cosine Jacobian', () => {
    const from: Vec3 = [0, 0, 0];
    const dest: Vec3 = [3, 4, 0]; // dist² = 25
    const destNorm: Vec3 = normalize([1, 1, 0]);
    const pdfSA = 0.73;
    const cosDest = Math.abs((3 / 5) * destNorm[0] + (4 / 5) * destNorm[1]);
    const expected = (pdfSA * cosDest) / 25;
    expect(convertDensitySAtoArea(pdfSA, from, dest, destNorm)).toBeCloseTo(expected, 14);
  });

  it('assembled merged path has the expected length and selectedS', () => {
    const f = makeFixture();
    const { vertices, selectedS } = assembleMergedConnectionPath(f);
    // n = c + e + 3 = 1 + 2 + 3 = 6 ; selectedS = c+1 = 2
    expect(vertices.length).toBe(6);
    expect(selectedS).toBe(2);
  });

  it('strategy pdf vector matches the oracle to ~1e-12 on a varying-G fixture', () => {
    const f = makeFixture();
    const { vertices, selectedS } = assembleMergedConnectionPath(f);
    const pRef = 0.018; // joint forward density of the chosen strategy
    const mine = buildStrategyPdfs(vertices, selectedS, pRef);
    const oracle = buildBDPTStrategyPDFs_full(toOracleVertices(vertices), selectedS, pRef);
    expect(mine.length).toBe(oracle.length);
    for (let i = 0; i < mine.length; i += 1) {
      const a = mine[i]!;
      const b = oracle[i]!;
      const tol = 1e-12 * Math.max(1, Math.abs(b));
      expect(Math.abs(a - b)).toBeLessThanOrEqual(tol);
    }
  });

  it('MIS weight matches the oracle to ~1e-12 for every strategy', () => {
    const f = makeFixture();
    const { vertices, selectedS } = assembleMergedConnectionPath(f);
    const pRef = 0.018;
    const mine = buildStrategyPdfs(vertices, selectedS, pRef);
    const oracle = buildBDPTStrategyPDFs_full(toOracleVertices(vertices), selectedS, pRef);
    for (let s = 0; s < vertices.length; s += 1) {
      const wMine = powerHeuristicWeight(mine, s, 2);
      const wOracle = bdptConnectionMIS_full(oracle, s, 2);
      expect(Math.abs(wMine - wOracle)).toBeLessThanOrEqual(1e-12);
    }
  });

  it('per-strategy MIS weights sum to 1 (partition of unity)', () => {
    const f = makeFixture();
    const { vertices, selectedS } = assembleMergedConnectionPath(f);
    const pRef = 0.018;
    const pdfs = buildStrategyPdfs(vertices, selectedS, pRef);
    let sum = 0;
    for (let s = 0; s < vertices.length; s += 1) {
      sum += powerHeuristicWeight(pdfs, s, 2);
    }
    expect(sum).toBeCloseTo(1, 12);
  });

  it('end-to-end bdptConnectionMisFull equals the oracle on the chosen strategy', () => {
    const f = makeFixture();
    const pRef = 0.018;
    const { vertices, selectedS } = assembleMergedConnectionPath(f);
    const oracle = bdptConnectionMIS_full(
      buildBDPTStrategyPDFs_full(toOracleVertices(vertices), selectedS, pRef),
      selectedS,
      2,
    );
    const mine = bdptConnectionMisFull({ ...f, pRef, beta: 2 });
    expect(Math.abs(mine - oracle)).toBeLessThanOrEqual(1e-12);
  });

  it('specular connection-edge endpoint zeroes hypothetical strategies (guard parity)', () => {
    const f = makeFixture();
    // Mark the light connection vertex specular: left sweep must break at once.
    const lc = { ...f.lightChain[1]!, isSpecular: true };
    const lightChain = [f.lightChain[0]!, lc];
    const { vertices, selectedS } = assembleMergedConnectionPath({ ...f, lightChain });
    const pRef = 0.02;
    const mine = buildStrategyPdfs(vertices, selectedS, pRef);
    const oracle = buildBDPTStrategyPDFs_full(toOracleVertices(vertices), selectedS, pRef);
    for (let i = 0; i < mine.length; i += 1) {
      expect(Math.abs(mine[i]! - oracle[i]!)).toBeLessThanOrEqual(1e-12);
    }
  });

  it('MIS weight is invariant to the absolute scale of pRef (GPU pRef choice is valid)', () => {
    // The power-heuristic weight pₛ²/Σpᵢ² is scale-free in pRef: every pdfs[i] is
    // derived from pRef by the ratio sweep, so a global scale k cancels (k² in
    // numerator and denominator). The GPU shader picks pRef = lightPdfFwd·fwdEe
    // (≠ the CPU test's pRef); this proves that choice yields the same weight.
    const f = makeFixture();
    const { vertices, selectedS } = assembleMergedConnectionPath(f);
    const wA = powerHeuristicWeight(buildStrategyPdfs(vertices, selectedS, 0.013), selectedS, 2);
    const wB = powerHeuristicWeight(buildStrategyPdfs(vertices, selectedS, 7.9), selectedS, 2);
    const wC = powerHeuristicWeight(buildStrategyPdfs(vertices, selectedS, 1e-6), selectedS, 2);
    expect(Math.abs(wA - wB)).toBeLessThanOrEqual(1e-12);
    expect(Math.abs(wA - wC)).toBeLessThanOrEqual(1e-12);
  });

  it('single-light-vertex (c=0 emitter connection) matches the oracle', () => {
    const f = makeFixture();
    const lightChain = [f.lightChain[0]!]; // emitter only
    const eyeChain = [f.eyeChain[0]!, f.eyeChain[1]!]; // E_0, E_1
    const { vertices, selectedS } = assembleMergedConnectionPath({ ...f, lightChain, eyeChain });
    // n = 0 + 1 + 3 = 4 ; selectedS = 1
    expect(vertices.length).toBe(4);
    expect(selectedS).toBe(1);
    const pRef = 0.05;
    const mine = buildStrategyPdfs(vertices, selectedS, pRef);
    const oracle = buildBDPTStrategyPDFs_full(toOracleVertices(vertices), selectedS, pRef);
    for (let i = 0; i < mine.length; i += 1) {
      expect(Math.abs(mine[i]! - oracle[i]!)).toBeLessThanOrEqual(1e-12);
    }
  });
});
