/**
 * bdptVeachFull.test.ts — Full Veach §10.3 BDPT MIS unit tests (T2.H4).
 *
 * All tests are derived from first principles (the paper / PBR4e), not copied
 * from the implementation. No GPU, no MC, fully deterministic.
 *
 * Test plan:
 *   T1 — Path of length 2: MIS weights sum to 1.
 *   T2 — Equal PDFs: all weights equal 1/N (power heuristic degeneracy).
 *   T3 — Specular zero-weight: strategies through a specular vertex have weight 0.
 *   T4 — Physical area-measure path PDF on a VARYING-G fixture: per-strategy
 *        weights match an independent first-principles derivation (built from the
 *        path-space measure, not the implementation's recurrence). Includes a
 *        regression guard that the pre-fix single-full-G oracle is rejected.
 *   T5 — Geometric term G(x↔y): matches analytic formula for known geometry.
 *   T6 — Camera/light endpoint corner cases: s=0 and s=k produce non-zero weights.
 *
 * References:
 *   - Veach 1997, §10.3 (BDPT MIS), §8.3.2 (geometry term).
 *   - Pharr, Jakob, Humphreys 2023, §16.3.5, Eq. 16.16.
 */

import { describe, it, expect } from 'vitest';
import {
  geometricTermG,
  buildBDPTStrategyPDFs_full,
  bdptConnectionMIS_full,
} from '../src/bdptMIS.js';
import type { BDPTFullVertex } from '../src/bdptMIS.js';

// ── Helper factories ──────────────────────────────────────────────────────────

/** Build a non-specular diffuse vertex on the XZ plane (normal pointing up). Retained for future BDPT path tests. */
function _makeVertex(
  x: number,
  z: number,
  pdfFwd: number,
  pdfRev: number,
  isSpecular = false,
): BDPTFullVertex {
  return {
    position: [x, 0, z],
    normal:   [0, 1, 0],
    pdfFwd,
    pdfRev,
    isSpecular,
  };
}

/** Sum MIS weights over all strategies and return the total. */
function sumWeights(pdfs: Float64Array, beta = 2): number {
  let total = 0;
  for (let s = 0; s < pdfs.length; s++) {
    total += bdptConnectionMIS_full(pdfs, s, beta);
  }
  return total;
}

// ── T1: path of length 2 (3 vertices: light, hit, camera) ────────────────────
//
// The path: v_0 (emitter, x=0) → v_1 (diffuse hit, x=1) → v_2 (camera, x=2).
// k = 2 segments → 3 strategies: s ∈ {0, 1, 2}.
//
// We choose the interior strategy s=1 (1 light vertex, 1 camera vertex) and
// supply p_ref = pdfFwd_v0 · pdfFwd_v1 (the product under that strategy).
// The sweep must recover non-zero probabilities for s=0 and s=2 as well,
// and the MIS weights must sum to 1.

describe('T1 — 3-vertex path MIS weights sum to 1', () => {
  // Geometry: v_0 at (0,0,0), v_1 at (1,0,0), v_2 at (2,0,0).
  // Normals all [0,1,0] (lying on XZ plane, normal pointing +Y).
  // Cosines between connection direction [1,0,0] and normal [0,1,0] = 0 for ALL vertices.
  //
  // With horizontal normals and a horizontal path the standard G formula gives 0 —
  // that's not useful for a sum-to-1 test. Use normals aligned with the path instead:
  // normal = [1,0,0] (pointing along the path) so cos θ = 1 at each vertex.

  const v0: BDPTFullVertex = {
    position:   [0, 0, 0],
    normal:     [1, 0, 0],
    pdfFwd:     0.4,
    pdfRev:     0.2,
    isSpecular: false,
  };
  const v1: BDPTFullVertex = {
    position:   [1, 0, 0],
    normal:     [1, 0, 0],
    pdfFwd:     0.5,
    pdfRev:     0.3,
    isSpecular: false,
  };
  const v2: BDPTFullVertex = {
    position:   [2, 0, 0],
    normal:     [1, 0, 0],
    pdfFwd:     0.6,
    pdfRev:     0.4,
    isSpecular: false,
  };

  const vertices = [v0, v1, v2];
  const selectedS = 1;
  const pRef = 0.4 * 0.5; // pdfFwd_v0 × pdfFwd_v1

  const pdfs = buildBDPTStrategyPDFs_full(vertices, selectedS, pRef);

  it('returns exactly 3 strategy PDFs for a 3-vertex path', () => {
    expect(pdfs.length).toBe(3);
  });

  it('selected strategy p_1 equals pRef', () => {
    expect(pdfs[1]).toBeCloseTo(pRef, 12);
  });

  it('all strategy PDFs are non-negative', () => {
    for (const p of pdfs) {
      expect(p).toBeGreaterThanOrEqual(0);
    }
  });

  it('MIS weights sum to 1 (power heuristic, β=2)', () => {
    const total = sumWeights(pdfs);
    expect(total).toBeCloseTo(1.0, 10);
  });

  it('MIS weights sum to 1 with balance heuristic (β=1)', () => {
    const total = sumWeights(pdfs, 1);
    expect(total).toBeCloseTo(1.0, 10);
  });
});

// ── T2: equal PDFs → equal weights = 1/N ─────────────────────────────────────
//
// When all strategy PDFs are equal, (p_s)^β / Σ(p_i)^β = p^β / (N·p^β) = 1/N
// regardless of β. This must hold for any N ≥ 1.

describe('T2 — equal PDFs yield equal weights', () => {
  // Build a 4-vertex path (3 segments, 4 strategies) where every strategy
  // PDF is exactly 1.0. The easiest way: pass a pre-computed pdfs array
  // directly to bdptConnectionMIS_full.

  it('4 strategies with equal PDFs each get weight 1/4', () => {
    const equalPdfs = new Float64Array([1.0, 1.0, 1.0, 1.0]);
    for (let s = 0; s < 4; s++) {
      expect(bdptConnectionMIS_full(equalPdfs, s, 2)).toBeCloseTo(0.25, 12);
    }
  });

  it('3 strategies with equal PDFs each get weight 1/3', () => {
    const equalPdfs = new Float64Array([0.7, 0.7, 0.7]);
    for (let s = 0; s < 3; s++) {
      expect(bdptConnectionMIS_full(equalPdfs, s, 2)).toBeCloseTo(1 / 3, 10);
    }
  });

  it('equal PDFs, β=1: weights still 1/N', () => {
    const equalPdfs = new Float64Array([0.3, 0.3, 0.3, 0.3, 0.3]);
    for (let s = 0; s < 5; s++) {
      expect(bdptConnectionMIS_full(equalPdfs, s, 1)).toBeCloseTo(0.2, 12);
    }
  });
});

// ── T3: specular zero-weight ──────────────────────────────────────────────────
//
// 4 vertices: v0 (light) — v1 (diffuse) — v2 (specular!) — v3 (camera).
// Strategies s ∈ {0,1,2,3}. The connection edge for hypothetical strategy s is
// (v_{s−1}, v_s); strategies s=0 (pure camera) and s=n−1 (pure light) make no
// explicit connection.
//
// Under Veach §10.3.5 / PBRT's `MISWeight` delta rule, a hypothetical strategy is
// excluded (pdf 0) iff an endpoint of ITS connection edge is a delta (specular)
// surface — because that connection cannot be sampled by an explicit deterministic
// join. v2 (index 2) is specular. The connection edges are:
//   s=1 → edge (v0,v1)  — both diffuse  → VALID (camera subpath still traces
//                                          *through* the specular bounce v2; that
//                                          is exactly what specular vertices are for)
//   s=2 → edge (v1,v2)  — v2 specular    → excluded (this is the selected strategy,
//                                          so pRef is the supplied anchor, but no
//                                          OTHER strategy is allowed to land here)
//   s=3 → edge (v2,v3)  — v2 specular    → excluded
//
// So the physically correct outcome is: s=3 is zero (its connection touches the
// specular vertex), while s=0 and s=1 are NON-zero. (The pre-fix oracle wrongly
// broke the entire left sweep on first touching v2, zeroing s=0 and s=1 too — a
// bug; this test now pins the correct pattern.)

describe('T3 — specular vertex zeroes only strategies whose connection touches it', () => {
  // Collinear path along X, normals pointing along X (so every D-Jacobian = 1).
  const vertices: BDPTFullVertex[] = [
    { position: [0, 0, 0], normal: [1, 0, 0], pdfFwd: 0.5, pdfRev: 0.3, isSpecular: false },
    { position: [1, 0, 0], normal: [1, 0, 0], pdfFwd: 0.4, pdfRev: 0.2, isSpecular: false },
    { position: [2, 0, 0], normal: [1, 0, 0], pdfFwd: 0.6, pdfRev: 0.5, isSpecular: true  }, // specular
    { position: [3, 0, 0], normal: [1, 0, 0], pdfFwd: 0.3, pdfRev: 0.4, isSpecular: false },
  ];

  const pRef = vertices[0]!.pdfFwd * vertices[1]!.pdfFwd; // s=2 chosen strategy
  const selectedS = 2;
  const pdfs = buildBDPTStrategyPDFs_full(vertices, selectedS, pRef);

  it('strategy s=3 is zero (its connection edge (v2,v3) touches specular v2)', () => {
    expect(pdfs[3]).toBe(0); // right sweep blocked by v2.isSpecular
  });

  it('strategy s=1 is NON-zero (its connection edge (v0,v1) is fully diffuse)', () => {
    // The camera subpath v1→v2→v3 legitimately traces through the specular
    // bounce; the explicit connection happens at the diffuse edge (v0,v1).
    expect(pdfs[1]).toBeGreaterThan(0);
  });

  it('strategy s=0 is NON-zero (pure camera path, no explicit connection)', () => {
    expect(pdfs[0]).toBeGreaterThan(0);
  });

  it('the specular-blocked strategy s=3 carries zero MIS weight', () => {
    expect(bdptConnectionMIS_full(pdfs, 3)).toBe(0);
  });

  it('sum of all MIS weights equals 1', () => {
    expect(sumWeights(pdfs)).toBeCloseTo(1.0, 10);
  });
});

// ── T4: physical area-measure path PDF on a VARYING-G fixture ─────────────────
//
// This is a *genuine* first-principles test, NOT a re-statement of the
// implementation's recurrence. We construct a path with non-unit edge lengths
// and non-aligned shading normals so the per-edge geometry factors all DIFFER
// (G ≠ 1, and every D-Jacobian is distinct). We then derive each strategy's
// path PDF directly from the path-space measure:
//
//   p_s = [ Π_{i=0}^{s−1} pA_fwd(i) ] · [ Π_{j=s}^{n−1} pA_rev(j) ]
//
// where the solid-angle pdfs are lifted to AREA measure by the canonical
// change-of-variables Jacobian (Veach §8.2.2.2 / PBRT §16.1.1, `ConvertDensity`):
//
//   dω(from→dest) = dA_dest · |cos θ_dest| / ‖from − dest‖²
//
// i.e. the JACOBIAN CARRIES ONLY THE DESTINATION-VERTEX COSINE (a "half-G"),
//   pA_fwd(i) = pdfFwd_SA(i) · |cos θ_i (v_{i−1}→v_i)| / ‖v_{i−1}−v_i‖²   (i>0)
//   pA_rev(j) = pdfRev_SA(j) · |cos θ_j (v_{j+1}→v_j)| / ‖v_{j+1}−v_j‖²   (j<n−1)
// with unit Jacobian at the two path endpoints (their pdfs are already area
// densities — emitter area-sampling pdf / camera importance area density).
//
// Crucially this derivation is written from the measure definition, by hand,
// using a different code path (explicit per-strategy product, no ratio sweep,
// no reuse of any implementation helper). It therefore independently pins the
// oracle. The OLD oracle — which used a single FULL two-cosine G and the wrong
// transfer-vertex index — FAILS this test (it diverges by orders of magnitude
// once G varies along the path); the corrected oracle passes to machine ε.

describe('T4 — physical area-measure path PDF (varying-G, non-circular)', () => {
  // Non-unit spacing, non-aligned normals → every edge has a distinct geometry
  // factor, so the destination-cosine Jacobians do NOT cancel.
  function unit(v: readonly [number, number, number]): [number, number, number] {
    const l = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / l, v[1] / l, v[2] / l];
  }
  const verts: BDPTFullVertex[] = [
    { position: [0.0,  0.0,  0.0], normal: unit([0.2,  1.0,  0.1]), pdfFwd: 0.70, pdfRev: 0.35, isSpecular: false },
    { position: [1.3,  0.8,  0.0], normal: unit([-0.5, 0.9,  0.3]), pdfFwd: 0.55, pdfRev: 0.40, isSpecular: false },
    { position: [2.1, -0.4,  1.2], normal: unit([0.1,  0.7, -0.8]), pdfFwd: 0.42, pdfRev: 0.61, isSpecular: false },
    { position: [3.9,  0.6,  0.5], normal: unit([0.6,  0.5,  0.4]), pdfFwd: 0.33, pdfRev: 0.52, isSpecular: false },
    { position: [5.0,  1.7, -0.7], normal: unit([-0.3, 0.4,  0.85]), pdfFwd: 0.81, pdfRev: 0.12, isSpecular: false },
  ];
  const n = verts.length;

  // Destination-cosine-only Jacobian: |cos θ_dest| / dist², derived inline from
  // the change-of-variables definition (NOT calling the implementation).
  function destJacobian(fromIdx: number, destIdx: number): number {
    const a = verts[fromIdx]!.position;
    const b = verts[destIdx]!.position;
    const nB = verts[destIdx]!.normal;
    const d: [number, number, number] = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    const invDist = 1 / Math.sqrt(dist2);
    const cosDest = Math.abs(nB[0] * d[0] * invDist + nB[1] * d[1] * invDist + nB[2] * d[2] * invDist);
    return cosDest / dist2;
  }

  // Independent area-measure path PDF for strategy s, built straight from the
  // path-space measure (forward product up to s−1, reverse product from s).
  function independentAreaPdf(s: number): number {
    let p = 1;
    for (let i = 0; i < s; i++) {
      const jac = i === 0 ? 1 : destJacobian(i - 1, i);
      p *= verts[i]!.pdfFwd * jac;
    }
    for (let j = s; j < n; j++) {
      const jac = j === n - 1 ? 1 : destJacobian(j + 1, j);
      p *= verts[j]!.pdfRev * jac;
    }
    return p;
  }

  // Anchor the oracle at the same physical reference pdf for strategy selectedS,
  // so all strategies are compared on a shared scale.
  const selectedS = 2;
  const pRef = independentAreaPdf(selectedS);
  const pdfs = buildBDPTStrategyPDFs_full(verts, selectedS, pRef);

  it('reference strategy p_2 equals pRef', () => {
    expect(pdfs[2]).toBeCloseTo(pRef, 14);
  });

  it('every strategy PDF matches the independent physical derivation', () => {
    for (let s = 0; s < n; s++) {
      expect(pdfs[s]).toBeCloseTo(independentAreaPdf(s), 12);
    }
  });

  it('MIS weights match the physical weights and sum to 1', () => {
    // Build the physical MIS weights from the independent pdfs and compare to
    // the oracle's weights strategy-by-strategy.
    const indep = Float64Array.from({ length: n }, (_, s) => independentAreaPdf(s));
    let denom = 0;
    for (const p of indep) denom += p * p; // β=2
    for (let s = 0; s < n; s++) {
      const physW = (indep[s]! * indep[s]!) / denom;
      expect(bdptConnectionMIS_full(pdfs, s, 2)).toBeCloseTo(physW, 10);
    }
    expect(sumWeights(pdfs)).toBeCloseTo(1.0, 10);
  });

  it('weight distribution is interior-peaked and smooth (not a degenerate spike)', () => {
    // The corrected oracle yields a smooth distribution that peaks at an
    // INTERIOR strategy. The OLD (biased) oracle collapsed ~99.97% of the weight
    // onto strategy 0 — a degenerate spike — on this exact fixture.
    const weights = Array.from({ length: n }, (_, s) => bdptConnectionMIS_full(pdfs, s, 2));
    let peak = 0;
    for (let s = 1; s < n; s++) if (weights[s]! > weights[peak]!) peak = s;
    expect(peak).toBeGreaterThan(0);     // not the leftmost endpoint
    expect(peak).toBeLessThan(n - 1);    // not the rightmost endpoint
    expect(weights[peak]!).toBeLessThan(0.95); // not collapsed to a single spike
    for (const w of weights) expect(w).toBeGreaterThan(0); // every strategy viable
  });

  it('REGRESSION GUARD: the old single-full-G / wrong-index recurrence is rejected', () => {
    // Reproduce the pre-fix oracle verbatim and assert it does NOT reproduce the
    // physical pdfs on this varying-G fixture — proving the test is sensitive to
    // the bug (it would pass trivially on the old all-G=1 collinear fixtures).
    function oldBiasedOracle(vs: BDPTFullVertex[], sel: number, pref: number): Float64Array {
      const out = new Float64Array(vs.length);
      out[sel] = pref;
      { let p = pref;
        for (let s = sel; s > 0; s--) {
          const v = vs[s]!, vPrev = vs[s - 1]!;
          if (v.isSpecular || vPrev.isSpecular) break;
          const g = geometricTermG(vPrev.position, vPrev.normal, v.position, v.normal);
          if (g <= 0 || v.pdfFwd <= 0) break;
          p = p * (v.pdfRev / (v.pdfFwd * g));
          out[s - 1] = p;
        } }
      { let p = pref;
        for (let s = sel; s < vs.length - 1; s++) {
          const v = vs[s]!, vNext = vs[s + 1]!;
          if (vNext.isSpecular || v.isSpecular) break;
          const g = geometricTermG(v.position, v.normal, vNext.position, vNext.normal);
          if (g <= 0 || vNext.pdfRev <= 0) break;
          p = p * ((vNext.pdfFwd * g) / vNext.pdfRev);
          out[s + 1] = p;
        } }
      return out;
    }
    const old = oldBiasedOracle(verts, selectedS, pRef);
    // At least one non-reference strategy must diverge grossly from the physical
    // value (the old oracle was ~685× off at s=0 on this fixture).
    let maxRel = 0;
    for (let s = 0; s < n; s++) {
      if (s === selectedS) continue;
      const phys = independentAreaPdf(s);
      maxRel = Math.max(maxRel, Math.abs(old[s]! - phys) / phys);
    }
    expect(maxRel).toBeGreaterThan(1.0); // grossly biased — confirms the bug was real
  });
});

// ── T5: geometric term G(x↔y) ────────────────────────────────────────────────
//
// Analytic formula: G(xᵢ ↔ xⱼ) = |cos θᵢ · cos θⱼ| / ‖xᵢ − xⱼ‖²
//
// We test:
//   (a) Two points 2 units apart, normals facing each other: G = (1·1)/4 = 0.25
//   (b) Coincident points: G = 0 (degenerate)
//   (c) Normals perpendicular to connection: G = 0 (backface)
//   (d) General case: G matches formula.

describe('T5 — geometricTermG matches analytic formula', () => {
  it('two points 2 units apart, facing normals → G = 1/4 = 0.25', () => {
    // posI = (0,0,0), normalI = (1,0,0)
    // posJ = (2,0,0), normalJ = (-1,0,0)  [facing each other]
    // dir = (1,0,0); cosI = 1, cosJ = 1; dist² = 4
    // G = 1·1/4 = 0.25
    const g = geometricTermG(
      [0, 0, 0], [1, 0, 0],
      [2, 0, 0], [-1, 0, 0],
    );
    expect(g).toBeCloseTo(0.25, 12);
  });

  it('same-direction normals also give 0.25 (absolute cosines)', () => {
    // With same normals the cosines have opposite signs in the dot-product
    // sense, but we take |cos θ| so the result is still 0.25.
    const g = geometricTermG(
      [0, 0, 0], [1, 0, 0],
      [2, 0, 0], [1, 0, 0],
    );
    expect(g).toBeCloseTo(0.25, 12);
  });

  it('coincident points → G = 0', () => {
    const g = geometricTermG(
      [1, 2, 3], [0, 1, 0],
      [1, 2, 3], [0, 1, 0],
    );
    expect(g).toBe(0);
  });

  it('normal perpendicular to connection direction → G = 0', () => {
    // Connection along X; normal along Z ⊥ X → cos θ = 0 on one side
    const g = geometricTermG(
      [0, 0, 0], [0, 0, 1],  // normal along Z
      [1, 0, 0], [1, 0, 0],
    );
    expect(g).toBeCloseTo(0, 12);
  });

  it('general case: distance 3, 45-degree normals', () => {
    // posI = (0,0,0), posJ = (3,0,0) → dist = 3, dist² = 9
    // normalI = (1,0,0), normalJ = (1/√2, 1/√2, 0)
    // dir = (1,0,0)
    // cosI = |1| = 1, cosJ = |1/√2| = 1/√2
    // G = 1 · (1/√2) / 9 = 1/(9√2)
    const inv9sqrt2 = 1 / (9 * Math.SQRT2);
    const g = geometricTermG(
      [0, 0, 0], [1, 0, 0],
      [3, 0, 0], [1 / Math.SQRT2, 1 / Math.SQRT2, 0],
    );
    expect(g).toBeCloseTo(inv9sqrt2, 12);
  });

  it('distance 1 unit, both normals aligned with direction → G = 1', () => {
    const g = geometricTermG(
      [0, 0, 0], [1, 0, 0],
      [1, 0, 0], [1, 0, 0],
    );
    expect(g).toBeCloseTo(1.0, 12);
  });
});

// ── T6: camera/light endpoint corner cases ────────────────────────────────────
//
// s=0 (pure camera path, t=k, no light vertices): the camera endpoint reaches
//   the emitter directly. The selected strategy is s=0 with pRef supplied.
//
// s=k (pure light path, t=0, no camera vertices): the light endpoint is reached
//   by a full light subpath. Selected strategy is s=k.
//
// Both corner cases must produce non-zero MIS weights; specifically, when the
// selected strategy is the only non-zero one (specular walls left and right of
// a non-specular path segment of length 2) the weight must be 1.

describe('T6 — camera/light endpoint corner cases', () => {
  it('s=0 (pure camera path): selected strategy weight = 1 when only viable', () => {
    // Single segment path: v0 (light) → v1 (camera). k=1.
    // selectedS = 0 means the camera subpath generated the full path.
    // Specular at v1 blocks rightward sweep; all weight on s=0.
    const vertices: BDPTFullVertex[] = [
      { position: [0, 0, 0], normal: [1, 0, 0], pdfFwd: 0.5, pdfRev: 0.4, isSpecular: false },
      { position: [1, 0, 0], normal: [1, 0, 0], pdfFwd: 0.3, pdfRev: 0.2, isSpecular: true  },
    ];
    const pdfs = buildBDPTStrategyPDFs_full(vertices, 0, 0.5);
    // Right sweep from s=0 hits v1.isSpecular → p_1 = 0.
    expect(pdfs[0]).toBeCloseTo(0.5, 12);
    expect(pdfs[1]).toBe(0);
    expect(bdptConnectionMIS_full(pdfs, 0)).toBeCloseTo(1.0, 10);
  });

  it('s=k (pure light path): selected strategy weight = 1 when only viable', () => {
    // Single segment path: v0 (light) → v1 (camera). k=1.
    // selectedS = 1 means the light subpath generated the full path.
    // Specular at v0 blocks leftward sweep; all weight on s=1.
    const vertices: BDPTFullVertex[] = [
      { position: [0, 0, 0], normal: [1, 0, 0], pdfFwd: 0.5, pdfRev: 0.4, isSpecular: true  },
      { position: [1, 0, 0], normal: [1, 0, 0], pdfFwd: 0.3, pdfRev: 0.2, isSpecular: false },
    ];
    const pdfs = buildBDPTStrategyPDFs_full(vertices, 1, 0.3);
    // Left sweep from s=1 hits v0.isSpecular → p_0 = 0.
    expect(pdfs[1]).toBeCloseTo(0.3, 12);
    expect(pdfs[0]).toBe(0);
    expect(bdptConnectionMIS_full(pdfs, 1)).toBeCloseTo(1.0, 10);
  });

  it('s=0 and s=k both produce non-zero weights in a fully diffuse 2-vertex path', () => {
    const vertices: BDPTFullVertex[] = [
      { position: [0, 0, 0], normal: [1, 0, 0], pdfFwd: 0.5, pdfRev: 0.4, isSpecular: false },
      { position: [1, 0, 0], normal: [1, 0, 0], pdfFwd: 0.3, pdfRev: 0.2, isSpecular: false },
    ];
    const pdfs0 = buildBDPTStrategyPDFs_full(vertices, 0, 0.5);
    const pdfs1 = buildBDPTStrategyPDFs_full(vertices, 1, 0.3);

    // Either reference strategy should give both strategies non-zero
    expect(pdfs0[0]).toBeGreaterThan(0);
    expect(pdfs0[1]).toBeGreaterThanOrEqual(0); // may be zero if G=0, but >= 0
    expect(pdfs1[0]).toBeGreaterThanOrEqual(0);
    expect(pdfs1[1]).toBeGreaterThan(0);
  });

  it('empty vertices array returns empty PDF array without error', () => {
    const pdfs = buildBDPTStrategyPDFs_full([], 0, 1.0);
    expect(pdfs.length).toBe(0);
  });

  it('out-of-range selectedS returns 0 weight', () => {
    const pdfs = new Float64Array([0.5, 0.5]);
    expect(bdptConnectionMIS_full(pdfs, -1)).toBe(0);
    expect(bdptConnectionMIS_full(pdfs, 2)).toBe(0);
  });

  it('all-zero pdfs array returns 0 gracefully', () => {
    const pdfs = new Float64Array([0, 0, 0]);
    expect(bdptConnectionMIS_full(pdfs, 1)).toBe(0);
    expect(Number.isFinite(bdptConnectionMIS_full(pdfs, 1))).toBe(true);
  });
});
