/**
 * reconnectionShift.test.ts — GRIS / ReSTIR-PT reconnection-shift oracle tests.
 *
 * All expectations are derived from first principles (the path-space measure /
 * the change-of-variables definition), NOT copied from the implementation — the
 * same rigor as bdptVeachFull.test.ts. No GPU, no Monte-Carlo, fully
 * deterministic.
 *
 * Test plan:
 *   J1 — Jacobian equals the analytic (cosθ'_out/dist'²)·(dist²/cosθ_out) ratio
 *        on hand-built varying-geometry fixtures, to ~1e-12.
 *   J2 — Invertibility: T ∘ T⁻¹ = identity (and T⁻¹ ∘ T = identity).
 *   J3 — Reciprocal relation: J(T) · J(T⁻¹) = 1 to ~1e-12.
 *   J4 — Geometry term G(x1↔x2) = |cos θ_out| / dist² matches the analytic
 *        value, including the degenerate/tangent guards.
 *   J5 — Self-shift (x1' == x1) has unit Jacobian.
 *
 * References:
 *   - Lin et al. 2022 (GRIS), §5 (reconnection shift), Eq. 12 (shift Jacobian).
 */

import { describe, it, expect } from 'vitest';
import {
  reconnectionGeometryTerm,
  reconnectionShift,
  reconnectionShiftInverse,
  reconnectionJacobian,
} from '../src/reconnectionShift.js';
import type { ReconnectionPath } from '../src/reconnectionShift.js';

type V3 = readonly [number, number, number];

// ── independent (non-implementation) analytic helpers ─────────────────────────

function unit(v: V3): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
}

/**
 * Analytic destination-cosine geometry term, written straight from the
 * change-of-variables definition (NOT calling the implementation):
 *   G = |cos θ_out| / dist² ,  cos θ_out = n2 · (x2 − x1)/‖x2 − x1‖
 */
function analyticG(x1: V3, x2: V3, n2: V3): number {
  const d: V3 = [x2[0] - x1[0], x2[1] - x1[1], x2[2] - x1[2]];
  const dist = Math.hypot(d[0], d[1], d[2]);
  const cosOut = Math.abs((n2[0] * d[0] + n2[1] * d[1] + n2[2] * d[2]) / dist);
  return cosOut / (dist * dist);
}

/** Analytic shift Jacobian = G(offset)/G(base), built independently. */
function analyticJacobian(base: ReconnectionPath, offsetX1: V3): number {
  const gBase = analyticG(base.x1, base.x2, base.n2);
  const gOff = analyticG(offsetX1, base.x2, base.n2);
  return gOff / gBase;
}

// ── J1: Jacobian matches the analytic geometry ratio on varying fixtures ──────

describe('J1 — reconnectionJacobian equals the analytic geometry-term ratio', () => {
  // Non-aligned normal + non-unit, non-symmetric edge lengths so the base and
  // offset half-G terms genuinely differ (no accidental cancellation).
  const base: ReconnectionPath = {
    x1: [0.0, 0.0, 0.0],
    x2: [2.1, -0.4, 1.2],
    n2: unit([0.1, 0.7, -0.8]),
  };
  const offsetX1: V3 = [-0.6, 0.9, 0.3];

  it('matches the analytic (cosθ\'/dist\'²)·(dist²/cosθ) ratio to ~1e-12', () => {
    const shifted = reconnectionShift(base, offsetX1);
    const j = reconnectionJacobian(base, shifted);

    // Fully independent analytic derivation of the same ratio.
    const expected = analyticJacobian(base, offsetX1);
    expect(j).toBeCloseTo(expected, 12);
  });

  it('matches the explicit (|cosθ\'|·dist²)/(|cosθ|·dist\'²) closed form', () => {
    const shifted = reconnectionShift(base, offsetX1);
    const j = reconnectionJacobian(base, shifted);

    const dB: V3 = [base.x2[0] - base.x1[0], base.x2[1] - base.x1[1], base.x2[2] - base.x1[2]];
    const dO: V3 = [base.x2[0] - offsetX1[0], base.x2[1] - offsetX1[1], base.x2[2] - offsetX1[2]];
    const distB = Math.hypot(dB[0], dB[1], dB[2]);
    const distO = Math.hypot(dO[0], dO[1], dO[2]);
    const cosB = Math.abs((base.n2[0] * dB[0] + base.n2[1] * dB[1] + base.n2[2] * dB[2]) / distB);
    const cosO = Math.abs((base.n2[0] * dO[0] + base.n2[1] * dO[1] + base.n2[2] * dO[2]) / distO);
    const closedForm = (cosO * distB * distB) / (cosB * distO * distO);

    expect(j).toBeCloseTo(closedForm, 12);
  });

  it('holds across a sweep of distinct offset primary vertices', () => {
    const offsets: V3[] = [
      [1.0, 0.0, 0.0],
      [-1.3, 0.5, 0.2],
      [0.4, -0.9, 1.7],
      [2.5, 2.5, -1.1],
      [-0.05, -0.05, -0.05],
    ];
    for (const o of offsets) {
      const shifted = reconnectionShift(base, o);
      expect(reconnectionJacobian(base, shifted)).toBeCloseTo(analyticJacobian(base, o), 12);
    }
  });
});

// ── J2: invertibility T ∘ T⁻¹ = identity ──────────────────────────────────────

describe('J2 — invertibility (T ∘ T⁻¹ = identity)', () => {
  const base: ReconnectionPath = {
    x1: [0.3, -0.2, 0.5],
    x2: [3.9, 0.6, 0.5],
    n2: unit([0.6, 0.5, 0.4]),
  };
  const offsetX1: V3 = [1.3, 0.8, 0.0];

  it('T then T⁻¹ recovers the base path exactly', () => {
    const shifted = reconnectionShift(base, offsetX1);
    const back = reconnectionShiftInverse(shifted, base.x1);
    expect(back.x1).toEqual(base.x1);
    expect(back.x2).toEqual(base.x2);
    expect(back.n2).toEqual(base.n2);
  });

  it('T⁻¹ then T recovers the shifted path exactly', () => {
    const shifted = reconnectionShift(base, offsetX1);
    const unshifted = reconnectionShiftInverse(shifted, base.x1);
    const reshifted = reconnectionShift(unshifted, offsetX1);
    expect(reshifted.x1).toEqual(shifted.x1);
    expect(reshifted.x2).toEqual(shifted.x2);
    expect(reshifted.n2).toEqual(shifted.n2);
  });

  it('reconnection vertex (x2, n2) is invariant under both directions', () => {
    const shifted = reconnectionShift(base, offsetX1);
    expect(shifted.x2).toEqual(base.x2);
    expect(shifted.n2).toEqual(base.n2);
  });
});

// ── J3: reciprocal relation J(T)·J(T⁻¹) = 1 ───────────────────────────────────

describe('J3 — reciprocal Jacobian relation J(T)·J(T⁻¹) = 1', () => {
  const fixtures: { base: ReconnectionPath; offsetX1: V3 }[] = [
    {
      base: { x1: [0, 0, 0], x2: [2.1, -0.4, 1.2], n2: unit([0.1, 0.7, -0.8]) },
      offsetX1: [-0.6, 0.9, 0.3],
    },
    {
      base: { x1: [1.0, 2.0, -1.0], x2: [-3.0, 0.5, 2.2], n2: unit([-0.3, 0.4, 0.85]) },
      offsetX1: [0.2, -1.4, 0.7],
    },
    {
      base: { x1: [-2.5, -2.5, -2.5], x2: [0.0, 0.0, 0.0], n2: unit([1, 1, 1]) },
      offsetX1: [4.0, -1.0, 3.0],
    },
  ];

  it('forward × inverse Jacobian equals 1 to ~1e-12 on every fixture', () => {
    for (const { base, offsetX1 } of fixtures) {
      const shifted = reconnectionShift(base, offsetX1);

      const jFwd = reconnectionJacobian(base, shifted);
      // Inverse shift maps shifted → base, so its Jacobian is computed with the
      // shifted path as the (denominator) base.
      const jInv = reconnectionJacobian(shifted, base);

      expect(jFwd * jInv).toBeCloseTo(1.0, 12);
      // And J(T⁻¹) must equal 1/J(T) independently.
      expect(jInv).toBeCloseTo(1 / jFwd, 12);
    }
  });
});

// ── J4: geometry term matches analytic formula + guards ───────────────────────

describe('J4 — reconnectionGeometryTerm matches |cos θ_out| / dist²', () => {
  it('dist 2, normal facing the edge → G = 1/4', () => {
    // x1=(0,0,0), x2=(2,0,0), n2=(-1,0,0): dir x1→x2 = (1,0,0),
    // cos θ_out = |(-1,0,0)·(1,0,0)| = 1, dist²=4 → G = 1/4.
    expect(reconnectionGeometryTerm([0, 0, 0], [2, 0, 0], [-1, 0, 0])).toBeCloseTo(0.25, 12);
  });

  it('absolute cosine: same-direction normal also gives 1/4', () => {
    expect(reconnectionGeometryTerm([0, 0, 0], [2, 0, 0], [1, 0, 0])).toBeCloseTo(0.25, 12);
  });

  it('general case matches the analytic formula', () => {
    const x1: V3 = [0.2, -0.5, 1.1];
    const x2: V3 = [2.7, 1.3, -0.4];
    const n2 = unit([0.3, -0.9, 0.5]);
    expect(reconnectionGeometryTerm(x1, x2, n2)).toBeCloseTo(analyticG(x1, x2, n2), 12);
  });

  it('coincident vertices → G = 0 (degenerate edge guard)', () => {
    expect(reconnectionGeometryTerm([1, 2, 3], [1, 2, 3], [0, 1, 0])).toBe(0);
  });

  it('normal tangent to the connection direction → G = 0', () => {
    // Edge along X, normal along Z ⊥ X → cos θ_out = 0.
    expect(reconnectionGeometryTerm([0, 0, 0], [1, 0, 0], [0, 0, 1])).toBeCloseTo(0, 12);
  });
});

// ── J5: degenerate-Jacobian behaviour + self-shift ────────────────────────────

describe('J5 — self-shift and degenerate guards', () => {
  const base: ReconnectionPath = {
    x1: [0.5, 0.5, 0.5],
    x2: [2.0, -1.0, 1.5],
    n2: unit([0.2, 0.9, -0.1]),
  };

  it('self-shift (offset == base primary vertex) has unit Jacobian', () => {
    const shifted = reconnectionShift(base, base.x1);
    expect(reconnectionJacobian(base, shifted)).toBeCloseTo(1.0, 12);
  });

  it('degenerate base edge (x1 == x2) yields Jacobian 0, not NaN/Infinity', () => {
    const degenerate: ReconnectionPath = { x1: base.x2, x2: base.x2, n2: base.n2 };
    const shifted = reconnectionShift(degenerate, [1, 2, 3]);
    const j = reconnectionJacobian(degenerate, shifted);
    expect(j).toBe(0);
    expect(Number.isFinite(j)).toBe(true);
  });

  it('degenerate offset edge yields Jacobian 0 (non-invertible shift)', () => {
    const shifted = reconnectionShift(base, base.x2); // offset primary == reconnection vertex
    const j = reconnectionJacobian(base, shifted);
    expect(j).toBe(0);
    expect(Number.isFinite(j)).toBe(true);
  });
});
