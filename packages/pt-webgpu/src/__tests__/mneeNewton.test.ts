// mneeNewton.test.ts — the real-MNEE core: the half-vector Newton solve. The
// EXECUTED correctness check runs on GPU (wsl-gpu scripts/mnee-newton-validate.ts,
// lavapipe: residual→0 + the converged vertex == the analytic mirror reflection
// point to ~1e-5). Here we pin the host-side packing + the kernel composition.
import { describe, it, expect } from 'vitest';
import {
  MNEE_NEWTON_WGSL,
  MNEE_NEWTON_HARNESS_WGSL,
  MNEE_NEWTON_JAC_HARNESS_WGSL,
  MNEE_JACOBIAN_HARNESS_WGSL,
  MNEE_PDF_HARNESS_WGSL,
  packMneeHarnessInput,
  MNEE_HARNESS_INPUT_FLOATS,
  MNEE_NEWTON_MAX_ITERS,
} from '../wgsl/pathTrace/mneeNewton.wgsl.js';

describe('MNEE half-vector Newton solve (real-MNEE core)', () => {
  it('packs a config into the 12-float vec4-aligned record', () => {
    const r = packMneeHarnessInput([0, 0, 1], [1, 0, 1], [0, 0, 0]); // etaI/etaT default 1
    expect(r).toHaveLength(MNEE_HARNESS_INPUT_FLOATS);
    expect(r.slice(0, 4)).toEqual([0, 0, 1, 1]);   // receiver.xyz, etaI
    expect(r.slice(4, 8)).toEqual([1, 0, 1, 1]);   // light.xyz, etaT
    expect(r.slice(8, 12)).toEqual([0, 0, 0, 0]);  // planePoint.xyz, pad
    // Refraction: distinct etas land in the .w slots.
    expect(packMneeHarnessInput([0, 0, 1], [0, 0, -1], [0, 0, 0], 1.5, 1).slice(0, 8))
      .toEqual([0, 0, 1, 1.5, 0, 0, -1, 1]);
  });

  it('the solve is an ETA-GENERALIZED half-vector Newton iteration (analytic Jacobian)', () => {
    // The eta-generalized half-vector (reflection when etaI==etaT, Snell otherwise).
    expect(MNEE_NEWTON_WGSL).toContain('fn mneeHalfVectorResidual2d(');
    expect(MNEE_NEWTON_WGSL).toContain('let h = mnee_safe_normalize(etaI * wi + etaT * wo)');
    expect(MNEE_NEWTON_WGSL).toContain('let hTan = h - dot(h, nm) * nm');
    // The Newton step: ANALYTIC Jacobian (mneeResidualJacobian) + J·δ = −r.
    expect(MNEE_NEWTON_WGSL).toContain('fn mneeNewtonSolve(');
    expect(MNEE_NEWTON_WGSL).toContain('let jac = mneeResidualJacobian(v, recv, light, nm, tu, tv, etaI, etaT)');
    expect(MNEE_NEWTON_WGSL).toContain('let det = j00 * j11 - j01 * j10');
    expect(MNEE_NEWTON_WGSL).toContain('if (rmag < 1e-5) { return out; }'); // convergence exit
  });

  it('analytic residual Jacobian ∂r/∂(a,b) — validated analytic == FD on GPU', () => {
    // GPU-validated against finite difference at a generic test vertex
    // (mnee-newton-jac-validate.ts, lavapipe: analytic == FD, reflect + refract).
    // The exact derivative replaced the FD columns that drove the Newton step.
    expect(MNEE_NEWTON_WGSL).toContain('fn mneeDNormalize(');                 // (I − x̂x̂ᵀ)/|x| projector
    expect(MNEE_NEWTON_WGSL).toContain('return (dx - xh * dot(xh, dx)) / len');
    expect(MNEE_NEWTON_WGSL).toContain('fn mneeResidualJacobian(');
    expect(MNEE_NEWTON_WGSL).toContain('let dwi_a = mneeDNormalize(wiVec, -tu)');
    expect(MNEE_NEWTON_JAC_HARNESS_WGSL).toContain(MNEE_NEWTON_WGSL);          // byte-identical core
    expect(MNEE_NEWTON_JAC_HARNESS_WGSL).toContain('let jac = mneeResidualJacobian(v, c.recv, c.light, nm, tu, tv, c.etaI, c.etaT)');
  });

  it('harness kernel runs the solve over a flat surface (nm=+z) with per-config etas', () => {
    expect(MNEE_NEWTON_HARNESS_WGSL).toContain(MNEE_NEWTON_WGSL); // byte-identical core
    expect(MNEE_NEWTON_HARNESS_WGSL).toContain('let nm = vec3f(0.0, 0.0, 1.0)');
    expect(MNEE_NEWTON_HARNESS_WGSL).toContain(`mneeNewtonSolve(c.planePoint, nm, tu, tv, c.recv, c.light, c.etaI, c.etaT, ${MNEE_NEWTON_MAX_ITERS}u)`);
    expect(MNEE_NEWTON_MAX_ITERS).toBe(16);
  });

  it('manifold Jacobian d(vertex)/d(light) via the implicit function theorem', () => {
    // GPU-validated against brute-force FD re-solve (mnee-jacobian-validate.ts,
    // lavapipe: analytic == FD to ~1e-3, reflection + refraction). Pin the structure.
    expect(MNEE_NEWTON_WGSL).toContain('fn mneeManifoldJacobian(');
    expect(MNEE_NEWTON_WGSL).toContain('d(a,b)/d(light) = −J_vertex⁻¹ · J_light'); // the IFT formula
    expect(MNEE_JACOBIAN_HARNESS_WGSL).toContain(MNEE_NEWTON_WGSL); // byte-identical core
    expect(MNEE_JACOBIAN_HARNESS_WGSL).toContain('let jac = mneeManifoldJacobian(r.vertex, nm, tu, tv, c.recv, c.light, c.etaI, c.etaT)');
    expect(MNEE_JACOBIAN_HARNESS_WGSL).toContain('hOut[i * 3u + 1u] = vec4f(jac.dadL, jac.dbdL.x)');
  });

  it('connection-PDF factor |dω_recv/dA_light| (basis-free determinant)', () => {
    // GPU-validated against brute-force FD over the light area params
    // (mnee-pdf-validate.ts, lavapipe: analytic == FD to ~1e-3, reflect + refract).
    expect(MNEE_NEWTON_WGSL).toContain('fn mneePdfJacobianDet(');
    // The solid-angle projection + the basis-free 2×2 determinant (cross product).
    expect(MNEE_NEWTON_WGSL).toContain('let dw_ds = (dv_ds - w * dot(w, dv_ds)) / dist');
    expect(MNEE_NEWTON_WGSL).toContain('return length(cross(dw_ds, dw_dt))');
    expect(MNEE_PDF_HARNESS_WGSL).toContain(MNEE_NEWTON_WGSL); // byte-identical core
    expect(MNEE_PDF_HARNESS_WGSL).toContain('let det = mneePdfJacobianDet(r.vertex, c.recv, jac.dadL, jac.dbdL, tu, tv)');
  });
});
