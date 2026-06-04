// restirPtShift.test.ts — the ReSTIR-PT / GRIS reconnection-shift Jacobian for a
// general reconnection vertex (the hero-stack, arbitrary-path-length form of the
// 1-bounce GI special case in walkaround-hybrid's jacobianShift.wgsl.ts). The
// EXECUTED correctness check runs on GPU (wsl-gpu scripts/restir-pt-shift-
// validate.ts, lavapipe: the analytic ratio == the finite-difference of the
// actual shift-map measure change, perturbing x_s over its area params). Here we
// pin the host-side packing + the kernel composition + a few analytic invariants
// computed on the CPU (the same closed form the WGSL emits + the oracle holds).
import { describe, it, expect } from 'vitest';
import {
  RESTIR_PT_SHIFT_WGSL,
  RESTIR_PT_SHIFT_HARNESS_WGSL,
  packRestirPtShiftInput,
  RESTIR_PT_SHIFT_INPUT_FLOATS,
} from '../wgsl/pathTrace/restirPtShift.wgsl.js';

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3): number => Math.hypot(a[0], a[1], a[2]);
// Reference half-G + Jacobian (the SAME closed form the WGSL emits and the CPU
// oracle @vitrum/shared-samplers/reconnectionShift.ts holds), n_s = +z.
const halfG = (xa: V3, xs: V3, ns: V3): number => {
  const d = sub(xa, xs);
  const dist = len(d);
  if (dist <= 0) return 0;
  return Math.abs(dot(ns, d) / dist) / (dist * dist);
};
const shiftJ = (xq: V3, xr: V3, xs: V3, ns: V3): number => {
  const g = halfG(xq, xs, ns);
  return g <= 0 ? 0 : halfG(xr, xs, ns) / g;
};

describe('ReSTIR-PT / GRIS reconnection-shift Jacobian (general reconnection vertex)', () => {
  it('packs a config into the 12-float vec4-aligned record', () => {
    const r = packRestirPtShiftInput([1, 0, 1], [-1, 0, 1], [0, 0, 0]);
    expect(r).toHaveLength(RESTIR_PT_SHIFT_INPUT_FLOATS);
    expect(r.slice(0, 4)).toEqual([1, 0, 1, 0]);   // x_q.xyz, pad
    expect(r.slice(4, 8)).toEqual([-1, 0, 1, 0]);  // x_r.xyz, pad
    expect(r.slice(8, 12)).toEqual([0, 0, 0, 0]);  // x_s.xyz, pad
  });

  it('exports the destination-cosine half-G + the unclamped shift Jacobian ratio', () => {
    // G(x_a ↔ x_s) = |cos θ_s(a)| / ‖x_a − x_s‖² — cosine at the SHARED x_s normal.
    expect(RESTIR_PT_SHIFT_WGSL).toContain('fn restirPtReconnectionGeometryTerm(');
    expect(RESTIR_PT_SHIFT_WGSL).toContain('let cosOut = abs(dot(ns, d) / dist)');
    expect(RESTIR_PT_SHIFT_WGSL).toContain('return cosOut / dist2');
    // |∂T/∂·| = G(target)/G(source); UNCLAMPED (hero-stack form) — the GI special
    // case clamps [0.1,10], this returns the true ratio.
    expect(RESTIR_PT_SHIFT_WGSL).toContain('fn restirPtShiftJacobian(');
    expect(RESTIR_PT_SHIFT_WGSL).toContain('return gTarget / gSource');
    expect(RESTIR_PT_SHIFT_WGSL).not.toContain('clamp('); // unclamped, unlike the GI version
  });

  it('exposes the geometry-MEASURED |dω/dA_s| the harness finite-differences against', () => {
    // The basis-free solid-angle⇄area determinant at a pre-reconnection vertex —
    // the actual measure-change the validator FD-confirms (mnee-pdf pattern).
    expect(RESTIR_PT_SHIFT_WGSL).toContain('fn restirPtSolidAngleAreaDeriv(');
    expect(RESTIR_PT_SHIFT_WGSL).toContain('let dw_ds = (ts - w * dot(w, ts)) / dist');
    expect(RESTIR_PT_SHIFT_WGSL).toContain('return length(cross(dw_ds, dw_dt))');
  });

  it('harness composes the core byte-identically + writes [J, gSource, gTarget] and the SA derivs', () => {
    expect(RESTIR_PT_SHIFT_HARNESS_WGSL).toContain(RESTIR_PT_SHIFT_WGSL); // byte-identical core
    expect(RESTIR_PT_SHIFT_HARNESS_WGSL).toContain('let ns = vec3f(0.0, 0.0, 1.0)');
    expect(RESTIR_PT_SHIFT_HARNESS_WGSL).toContain('let J       = restirPtShiftJacobian(c.xq, c.xr, c.xs, ns)');
    expect(RESTIR_PT_SHIFT_HARNESS_WGSL).toContain('let saSource = restirPtSolidAngleAreaDeriv(c.xq, c.xs, ts, tt)');
    expect(RESTIR_PT_SHIFT_HARNESS_WGSL).toContain('hOut[i * 2u + 0u] = vec4f(J, gSource, gTarget, 0.0)');
  });

  it('analytic CPU invariants: J == G_target/G_source, and the shift is its own inverse (reciprocal)', () => {
    const ns: V3 = [0, 0, 1];
    const cases: { xq: V3; xr: V3; xs: V3 }[] = [
      { xq: [0.5, 0.3, 1.0], xr: [-0.4, 0.2, 0.8], xs: [0, 0, 0] },
      { xq: [1, -1, 1.3], xr: [-1, 1, 0.9], xs: [0.1, -0.2, 0] },
      { xq: [0.2, 0.0, 2.0], xr: [0.6, 0.1, 0.5], xs: [-0.1, 0.05, 0] },
    ];
    for (const { xq, xr, xs } of cases) {
      const j = shiftJ(xq, xr, xs, ns);
      // Definition: J equals the ratio of the two half-G terms.
      expect(j).toBeCloseTo(halfG(xr, xs, ns) / halfG(xq, xs, ns), 10);
      // Reciprocity: T⁻¹ swaps x_q ↔ x_r, so the inverse Jacobian is 1/J and the
      // round trip is 1 (Lin 2022 — the reconnection shift is its own inverse form).
      const jInv = shiftJ(xr, xq, xs, ns);
      expect(j * jInv).toBeCloseTo(1, 10);
    }
  });

  it('degenerate edges return 0 (non-invertible shift): coincident or tangent connection', () => {
    const ns: V3 = [0, 0, 1];
    // x_q == x_s ⇒ source half-G is 0 ⇒ nothing to remap from ⇒ J = 0.
    expect(shiftJ([0, 0, 0], [1, 0, 1], [0, 0, 0], ns)).toBe(0);
    // x_r tangent to n_s (x_r − x_s ⟂ n_s, i.e. same z as x_s) ⇒ target half-G 0 ⇒ J = 0.
    expect(shiftJ([0.5, 0.3, 1.0], [1, 0, 0], [0, 0, 0], ns)).toBe(0);
  });
});
