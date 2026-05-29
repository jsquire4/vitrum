/**
 * grisReuseMis.test.ts — GRIS reconnection-shift + pairwise-MIS mirror tests.
 *
 * Two things are pinned, in the same rigor as bdptConnectionMisFull.test.ts:
 *
 *   1. ORACLE PARITY — the GPU-side arithmetic (`grisReuseMis.ts`, which the
 *      WGSL `grisReuse.wgsl.ts` ports verbatim) reproduces the first-principles
 *      `@vitrum/shared-samplers/reconnectionShift.ts` oracle on shared fixtures
 *      to machine ε: the geometry term, the Jacobian-from-cached-half-G, and the
 *      degenerate guards. This is the GPU-vs-oracle agreement the Phase-1 shift
 *      must satisfy.
 *
 *   2. PARTITION OF UNITY — the GRIS generalized-balance (pairwise) MIS weights
 *      `m_i` sum to exactly 1 over any set of domains whose samples have a
 *      non-degenerate target. This is the unbiasedness invariant of the GRIS
 *      MIS: Σ_i m_i = 1 (Lin 2022, generalized balance heuristic).
 *
 * No GPU; fully deterministic.
 *
 * References:
 *   - Lin et al. 2022 (GRIS), §5 (reconnection shift), Eq. 12 (Jacobian),
 *     §"pairwise MIS" (generalized balance heuristic).
 */

import { describe, it, expect } from 'vitest';
import {
  reconnectionGeometryTerm as oracleGeometryTerm,
  reconnectionShift,
  reconnectionJacobian as oracleJacobian,
  type ReconnectionPath,
} from '@vitrum/shared-samplers';
import {
  reconnectionGeometryTerm,
  shiftJacobian,
  cachedBaseHalfG,
  giTargetAt,
  grisGeneralizedBalanceWeights,
  pairwiseDenomNeighbor,
  pairwiseDenomCanonical,
  type Vec3,
  type GrisDomain,
  type GrisSample,
} from '../src/pipeline/grisReuseMis.js';

function unit(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
}

// ── ORACLE PARITY ─────────────────────────────────────────────────────────────

describe('GRIS mirror — geometry term matches the shared-samplers oracle', () => {
  const fixtures: { x1: Vec3; x2: Vec3; n2: Vec3 }[] = [
    { x1: [0, 0, 0], x2: [2, 0, 0], n2: [-1, 0, 0] },
    { x1: [0.2, -0.5, 1.1], x2: [2.7, 1.3, -0.4], n2: unit([0.3, -0.9, 0.5]) },
    { x1: [1, 2, -1], x2: [-3, 0.5, 2.2], n2: unit([-0.3, 0.4, 0.85]) },
  ];

  it('reproduces reconnectionGeometryTerm to ~1e-12 on every fixture', () => {
    for (const { x1, x2, n2 } of fixtures) {
      expect(reconnectionGeometryTerm(x1, x2, n2)).toBeCloseTo(
        oracleGeometryTerm(x1, x2, n2),
        12,
      );
    }
  });

  it('degenerate / tangent edges → 0 (matches oracle guards)', () => {
    // Coincident vertices.
    expect(reconnectionGeometryTerm([1, 2, 3], [1, 2, 3], [0, 1, 0])).toBe(0);
    expect(oracleGeometryTerm([1, 2, 3], [1, 2, 3], [0, 1, 0])).toBe(0);
    // Tangent connection (edge ⊥ normal).
    expect(reconnectionGeometryTerm([0, 0, 0], [1, 0, 0], [0, 0, 1])).toBeCloseTo(0, 12);
  });
});

describe('GRIS mirror — shift Jacobian matches the oracle reconnectionJacobian', () => {
  // Base path (the neighbour's reconnection sample): x1 = q.xv, x2 = q.xs.
  const base: ReconnectionPath = {
    x1: [0.0, 0.0, 0.0],
    x2: [2.1, -0.4, 1.2],
    n2: unit([0.1, 0.7, -0.8]),
  };
  // Offset primary vertex (the current pixel's visible vertex).
  const offsets: Vec3[] = [
    [-0.6, 0.9, 0.3],
    [1.0, 0.0, 0.0],
    [0.4, -0.9, 1.7],
    [2.5, 2.5, -1.1],
  ];

  it('Jacobian-from-cached-half-G equals G(shifted)/G(base) to ~1e-12', () => {
    // The GPU recovers the base half-G from the Phase-0 cache: the cache stores
    // cosReconOut = |cosθ_out(x2)| and distRecon = ‖x1 − x2‖, so
    // cachedBaseHalfG === reconnectionGeometryTerm(base.x1, base.x2, base.n2).
    const dBase: Vec3 = [base.x2[0] - base.x1[0], base.x2[1] - base.x1[1], base.x2[2] - base.x1[2]];
    const distRecon = Math.hypot(dBase[0], dBase[1], dBase[2]);
    const cosReconOut = Math.abs(
      (base.n2[0] * dBase[0] + base.n2[1] * dBase[1] + base.n2[2] * dBase[2]) / distRecon,
    );
    const gBase = cachedBaseHalfG(cosReconOut, distRecon);

    // The cached half-G must equal the oracle geometry term at the base edge.
    expect(gBase).toBeCloseTo(oracleGeometryTerm(base.x1, base.x2, base.n2), 12);

    for (const offsetX1 of offsets) {
      const gpuJ = shiftJacobian(gBase, offsetX1, base.x2, base.n2);
      const shifted = reconnectionShift(base, offsetX1);
      const oracleJ = oracleJacobian(base, shifted);
      expect(gpuJ).toBeCloseTo(oracleJ, 12);
    }
  });

  it('self-shift (offset == base primary) has unit Jacobian', () => {
    const dBase: Vec3 = [base.x2[0] - base.x1[0], base.x2[1] - base.x1[1], base.x2[2] - base.x1[2]];
    const distRecon = Math.hypot(dBase[0], dBase[1], dBase[2]);
    const cosReconOut = Math.abs(
      (base.n2[0] * dBase[0] + base.n2[1] * dBase[1] + base.n2[2] * dBase[2]) / distRecon,
    );
    const gBase = cachedBaseHalfG(cosReconOut, distRecon);
    expect(shiftJacobian(gBase, base.x1, base.x2, base.n2)).toBeCloseTo(1.0, 12);
  });

  it('degenerate base half-G (gBase == 0) → Jacobian 0, not NaN/Infinity', () => {
    const j = shiftJacobian(0, [1, 2, 3], base.x2, base.n2);
    expect(j).toBe(0);
    expect(Number.isFinite(j)).toBe(true);
  });

  it('degenerate offset edge (offset == reconnection vertex) → Jacobian 0', () => {
    const dBase: Vec3 = [base.x2[0] - base.x1[0], base.x2[1] - base.x1[1], base.x2[2] - base.x1[2]];
    const distRecon = Math.hypot(dBase[0], dBase[1], dBase[2]);
    const cosReconOut = Math.abs(
      (base.n2[0] * dBase[0] + base.n2[1] * dBase[1] + base.n2[2] * dBase[2]) / distRecon,
    );
    const gBase = cachedBaseHalfG(cosReconOut, distRecon);
    // Offset primary coincident with the reconnection vertex ⇒ G(shifted)=0.
    const j = shiftJacobian(gBase, base.x2, base.x2, base.n2);
    expect(j).toBe(0);
    expect(Number.isFinite(j)).toBe(true);
  });
});

// ── PARTITION OF UNITY (the GRIS MIS unbiasedness invariant) ──────────────────

describe('GRIS generalized-balance MIS — weights for one fixed sample partition unity', () => {
  function sum(a: readonly number[]): number {
    return a.reduce((s, x) => s + x, 0);
  }

  // The generalized balance heuristic partitions unity for ONE FIXED sample y
  // across the techniques (domains) that could have produced it: Σ_i m_i(y) = 1.
  // (NOT a sum over the domains' DIFFERENT own samples — that does not partition
  // unity; it is the per-output-sample technique weighting that must.)

  it('Σ m_i(y) = 1 on a canonical + 2-neighbour fixture', () => {
    const domains: GrisDomain[] = [
      { xv: [0, 0, 0],    nv: [0, 1, 0], xs: [0.3, 1.0, 0.2], ns: [0, -1, 0], Lo: [2.0, 1.5, 1.0], c: 8 },
      { xv: [0.5, 0, 0.1], nv: [0, 1, 0], xs: [0.9, 1.1, 0.0], ns: [0, -1, 0], Lo: [1.0, 2.0, 0.5], c: 20 },
      { xv: [-0.4, 0, 0.3], nv: [0, 1, 0], xs: [-0.2, 0.9, 0.4], ns: [0, -1, 0], Lo: [0.5, 0.5, 3.0], c: 12 },
    ];
    // Evaluate the partition for EACH domain's reconnection sample as the fixed y.
    for (const src of domains) {
      const y: GrisSample = { xs: src.xs, ns: src.ns, Lo: src.Lo };
      const m = grisGeneralizedBalanceWeights(domains, y);
      expect(sum(m)).toBeCloseTo(1.0, 12);
      for (const w of m) expect(w).toBeGreaterThanOrEqual(0);
    }
  });

  it('Σ m_i(y) = 1 across a sweep of multi-neighbour fixtures', () => {
    const sweeps: GrisDomain[][] = [
      [
        { xv: [0, 0, 0], nv: unit([0, 1, 0.1]), xs: [0.2, 1.0, 0.0], ns: unit([0.1, -1, 0]), Lo: [3, 1, 1], c: 5 },
        { xv: [1, 0, 0], nv: unit([0.1, 1, 0]), xs: [1.1, 1.2, 0.1], ns: unit([0, -1, 0.2]), Lo: [1, 3, 1], c: 30 },
        { xv: [0, 0, 1], nv: unit([0, 1, -0.1]), xs: [0.0, 0.8, 1.2], ns: unit([0, -1, 0]), Lo: [1, 1, 3], c: 15 },
        { xv: [-1, 0, -1], nv: unit([0.05, 1, 0.05]), xs: [-0.9, 1.1, -0.8], ns: unit([0, -1, 0]), Lo: [2, 2, 0.3], c: 9 },
      ],
      [
        { xv: [0, 0, 0], nv: [0, 1, 0], xs: [0.5, 2.0, 0.5], ns: [0, -1, 0], Lo: [4, 0.5, 0.5], c: 1 },
        { xv: [2, 0, 0], nv: [0, 1, 0], xs: [2.3, 1.5, -0.3], ns: [0, -1, 0], Lo: [0.5, 4, 0.5], c: 50 },
        { xv: [0, 0, 2], nv: [0, 1, 0], xs: [-0.2, 1.8, 2.2], ns: [0, -1, 0], Lo: [0.5, 0.5, 4], c: 7 },
        { xv: [-2, 0, 0], nv: [0, 1, 0], xs: [-2.1, 1.3, 0.4], ns: [0, -1, 0], Lo: [2, 2, 2], c: 22 },
        { xv: [0, 0, -2], nv: [0, 1, 0], xs: [0.4, 1.6, -1.7], ns: [0, -1, 0], Lo: [3, 3, 0.2], c: 13 },
      ],
    ];
    for (const domains of sweeps) {
      for (const src of domains) {
        const y: GrisSample = { xs: src.xs, ns: src.ns, Lo: src.Lo };
        expect(sum(grisGeneralizedBalanceWeights(domains, y))).toBeCloseTo(1.0, 12);
      }
    }
  });

  it('single domain → m(y) = 1 (trivial partition)', () => {
    const domains: GrisDomain[] = [
      { xv: [0, 0, 0], nv: [0, 1, 0], xs: [0.2, 1.0, 0.0], ns: [0, -1, 0], Lo: [1, 1, 1], c: 8 },
    ];
    const y: GrisSample = { xs: domains[0]!.xs, ns: domains[0]!.ns, Lo: domains[0]!.Lo };
    const m = grisGeneralizedBalanceWeights(domains, y);
    expect(m).toHaveLength(1);
    expect(m[0]).toBeCloseTo(1.0, 12);
  });

  it('a domain that shifts y to a zero target gets weight 0; the rest still sum to 1', () => {
    // Domain 0's primary surface faces AWAY from y (cosθ ≤ 0 when y is shifted
    // onto it) → its technique weight for y is 0. The other two share the unit.
    const domains: GrisDomain[] = [
      { xv: [0, 2, 0], nv: [0, 1, 0], xs: [0.2, 1.0, 0.0], ns: [0, -1, 0], Lo: [1, 1, 1], c: 8 },
      { xv: [0.5, 0, 0], nv: [0, 1, 0], xs: [0.6, 1.0, 0.0], ns: [0, -1, 0], Lo: [2, 1, 1], c: 10 },
      { xv: [-0.5, 0, 0], nv: [0, 1, 0], xs: [-0.4, 1.1, 0.0], ns: [0, -1, 0], Lo: [1, 2, 1], c: 12 },
    ];
    // Fixed sample y sits at y=1.0 in world; domain 0's primary vertex is ABOVE
    // it (y=2.0) so the shifted direction xv0→y points downward, cos(nv0,·) ≤ 0.
    const y: GrisSample = { xs: [0.0, 1.0, 0.0], ns: [0, -1, 0], Lo: [1, 1, 1] };
    const m = grisGeneralizedBalanceWeights(domains, y);
    expect(m[0]).toBe(0);
    expect(sum(m)).toBeCloseTo(1.0, 12);
  });
});

// ── Streaming pairwise denominators (the GPU reuse loop's per-neighbour form) ──

describe('GRIS streaming pairwise — GPU denominators reproduce the 2-domain generalized balance', () => {
  // The GPU reuse loop folds neighbour q into canonical r ONE pair at a time. For
  // the NEIGHBOUR's sample y_q the GPU computes
  //   m_q(y_q) = c_q·p̂_q(y_q) / (c_r·p̂_r(y_q) + c_q·p̂_q(y_q))
  // via `pairwiseDenomNeighbor`. For the CANONICAL's sample y_r it computes the
  // symmetric m_r(y_r) via `pairwiseDenomCanonical`. Each is the 2-domain
  // generalized-balance weight for ITS OWN sample — which partitions unity
  // across the pair for a FIXED sample (NOT across the two different samples).
  const r: GrisDomain = { xv: [0, 0, 0], nv: [0, 1, 0], xs: [0.3, 1.0, 0.2], ns: [0, -1, 0], Lo: [2, 1.5, 1], c: 8 };
  const q: GrisDomain = { xv: [0.5, 0, 0.1], nv: [0, 1, 0], xs: [0.9, 1.1, 0.0], ns: [0, -1, 0], Lo: [1, 2, 0.5], c: 20 };

  it('m_q(y_q) from pairwiseDenomNeighbor matches the 2-domain balance for q\'s sample', () => {
    const pHatQ_native = giTargetAt(q.xv, q.nv, q.xs, q.Lo);
    const pHatR_atQsample = giTargetAt(r.xv, r.nv, q.xs, q.Lo);
    const denomQ = pairwiseDenomNeighbor(r.c, pHatR_atQsample, q.c, pHatQ_native);
    const m_q = (q.c * pHatQ_native) / denomQ;

    // Independent 2-domain generalized balance for the FIXED sample y = q's.
    const yq: GrisSample = { xs: q.xs, ns: q.ns, Lo: q.Lo };
    const full = grisGeneralizedBalanceWeights([r, q], yq);
    expect(m_q).toBeCloseTo(full[1]!, 12);
    // Both techniques' weights for y_q partition unity.
    expect(full[0]! + full[1]!).toBeCloseTo(1.0, 12);
  });

  it('m_r(y_r) from pairwiseDenomCanonical matches the 2-domain balance for r\'s sample', () => {
    const pHatR_native = giTargetAt(r.xv, r.nv, r.xs, r.Lo);
    const pHatQ_atRsample = giTargetAt(q.xv, q.nv, r.xs, r.Lo);
    const denomR = pairwiseDenomCanonical(r.c, pHatR_native, q.c, pHatQ_atRsample);
    const m_r = (r.c * pHatR_native) / denomR;

    const yr: GrisSample = { xs: r.xs, ns: r.ns, Lo: r.Lo };
    const full = grisGeneralizedBalanceWeights([r, q], yr);
    expect(m_r).toBeCloseTo(full[0]!, 12);
    expect(full[0]! + full[1]!).toBeCloseTo(1.0, 12);
  });
});
