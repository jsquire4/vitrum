// mneeNewton.test.ts — the real-MNEE core: the half-vector Newton solve. The
// EXECUTED correctness check runs on GPU (wsl-gpu scripts/mnee-newton-validate.ts,
// lavapipe: residual→0 + the converged vertex == the analytic mirror reflection
// point to ~1e-5). Here we pin the host-side packing + the kernel composition.
import { describe, it, expect } from 'vitest';
import {
  MNEE_NEWTON_WGSL,
  MNEE_CHAIN_WGSL,
  MNEE_CONNECTION_WGSL,
  MNEE_NEWTON_MAX_ITERS,
  MNEE_CHAIN_MAX_ITERS,
} from '../wgsl/pathTrace/mneeNewton.wgsl.js';
import {
  MNEE_NEWTON_HARNESS_WGSL,
  MNEE_NEWTON_JAC_HARNESS_WGSL,
  MNEE_JACOBIAN_HARNESS_WGSL,
  MNEE_PDF_HARNESS_WGSL,
  MNEE_CHAIN_HARNESS_WGSL,
  MNEE_CHAIN_PDF_HARNESS_WGSL,
  MNEE_REFLECTION_HARNESS_WGSL,
  packMneeHarnessInput,
  MNEE_HARNESS_INPUT_FLOATS,
} from '../wgsl/pathTrace/mneeNewton.harness.wgsl.js';

type Vec3 = readonly [number, number, number];

const VEC3_ZERO: Vec3 = [0, 0, 0];
const VEC3_Z: Vec3 = [0, 0, 1];
const VEC3_X: Vec3 = [1, 0, 0];
const VEC3_Y: Vec3 = [0, 1, 0];

function add3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale3(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function length3(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize3(v: Vec3): Vec3 {
  const len = length3(v);
  if (len < 1e-12) return VEC3_ZERO;
  return scale3(v, 1 / len);
}

function mneeHalfVectorResidualReference(
  v: Vec3,
  recv: Vec3,
  light: Vec3,
  nm: Vec3,
  tu: Vec3,
  tv: Vec3,
  etaI: number,
  etaT: number,
): readonly [number, number] {
  const wi = normalize3(sub3(light, v));
  const wo = normalize3(sub3(recv, v));
  const h = normalize3(add3(scale3(wi, etaI), scale3(wo, etaT)));
  const hTan = sub3(h, scale3(nm, dot3(h, nm)));
  return [dot3(hTan, tu), dot3(hTan, tv)];
}

function residualLength2(r: readonly [number, number]): number {
  return Math.hypot(r[0], r[1]);
}

function analyticMirrorVertexOnZPlane(recv: Vec3, light: Vec3): Vec3 {
  const mirroredLight: Vec3 = [light[0], light[1], -light[2]];
  const line = sub3(mirroredLight, recv);
  const t = -recv[2] / line[2];
  return add3(recv, scale3(line, t));
}

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

  it('CPU oracle: the analytic flat-mirror vertex zeros the reflection half-vector residual', () => {
    for (const [recv, light] of [
      [[0, 0, 1], [1, 0, 1]],
      [[-0.4, 0.25, 1.6], [0.8, -0.35, 0.9]],
      [[0.2, -0.7, 2.1], [-1.1, 0.4, 0.6]],
    ] as const satisfies readonly (readonly [Vec3, Vec3])[]) {
      const v = analyticMirrorVertexOnZPlane(recv, light);
      expect(v[2]).toBeCloseTo(0, 12);
      expect(residualLength2(
        mneeHalfVectorResidualReference(v, recv, light, VEC3_Z, VEC3_X, VEC3_Y, 1, 1),
      )).toBeLessThan(1e-12);

      const shifted = add3(v, [0.125, -0.075, 0]);
      expect(residualLength2(
        mneeHalfVectorResidualReference(shifted, recv, light, VEC3_Z, VEC3_X, VEC3_Y, 1, 1),
      )).toBeGreaterThan(1e-3);
    }
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
    expect(MNEE_NEWTON_WGSL).toContain(
      'if (rmag < mneeResidualToleranceFromScales(solverScales))',
    );
    expect(MNEE_NEWTON_WGSL).toContain(
      'if (!mneeScalesRepresentable(solverScales)) { return out; }',
    );
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
    expect(MNEE_NEWTON_WGSL).toContain(
      'let determinant = length(cross(dw_ds, dw_dt))',
    );
    expect(MNEE_NEWTON_WGSL).toContain(
      'if (!(determinant >= 0.0) || !(determinant < INFINITY))',
    );
    expect(MNEE_PDF_HARNESS_WGSL).toContain(MNEE_NEWTON_WGSL); // byte-identical core
    expect(MNEE_PDF_HARNESS_WGSL).toContain('let det = mneePdfJacobianDet(r.vertex, c.recv, jac.dadL, jac.dbdL, tu, tv)');
  });

  it('does not ship the uncalled finite-emitter-axis Jacobian variant', () => {
    expect(MNEE_NEWTON_WGSL).not.toContain('fn mneePdfJacobianDetAxes(');
  });

  it('2-vertex chain solve (glass enter+exit) — block-tridiagonal Newton', () => {
    // GPU-validated on lavapipe (mnee-chain-validate.ts): both tangential
    // half-vector residuals → 0 AND the converged vertices satisfy Snell's ratio
    // at each interface (independent of the half-vector residual formula).
    expect(MNEE_CHAIN_WGSL).toContain('fn mneeChainResidual4d(');         // 4D coupled residual
    expect(MNEE_CHAIN_WGSL).toContain('fn mneeNewtonSolveChain2(');
    expect(MNEE_CHAIN_WGSL).toContain('fn mnee_inv2x2(');                 // 2×2 block inverse
    expect(MNEE_CHAIN_WGSL).toContain('let S = D - CAinv * B;');          // Schur complement
    expect(MNEE_CHAIN_MAX_ITERS).toBe(32);
    // Harness composes BOTH the single-vertex core (for mnee_safe_normalize) and
    // the chain module, then runs a glass-slab config.
    expect(MNEE_CHAIN_HARNESS_WGSL).toContain(MNEE_NEWTON_WGSL);
    expect(MNEE_CHAIN_HARNESS_WGSL).toContain(MNEE_CHAIN_WGSL);
    expect(MNEE_CHAIN_HARNESS_WGSL).toContain(`mneeNewtonSolveChain2(p1, n, tu, tv, p2, n, tu, tv, c.lightP, c.recv, 1.0, c.etaGlass, c.etaGlass, 1.0, ${MNEE_CHAIN_MAX_ITERS}u)`);
  });

  it('chain connection-PDF |dω_recv/dA_light| via the 4-DOF IFT — validated analytic == FD', () => {
    // GPU-validated against a brute-force FD re-solve over the area light's (x,y)
    // params (mnee-chain-pdf-validate.ts, lavapipe). The (a2,b2) IFT rows reuse the
    // solve's Schur block form S⁻¹(C·A⁻¹·r_top − r_bot).
    expect(MNEE_CHAIN_WGSL).toContain('fn mneeChainPdfJacobianDet(');
    expect(MNEE_CHAIN_WGSL).toContain('let dab2_ds = Sinv * (CAinv * r_s.xy - r_s.zw);'); // the IFT (a2,b2) rows
    expect(MNEE_CHAIN_WGSL).toContain('let determinant = length(cross(dw_ds, dw_dt));'); // basis-free determinant
    expect(MNEE_CHAIN_WGSL).toContain('if (!mneeMat2Invertible(S)) { return 0.0; }');
    expect(MNEE_CHAIN_PDF_HARNESS_WGSL).toContain(MNEE_CHAIN_WGSL);                       // byte-identical core
    expect(MNEE_CHAIN_PDF_HARNESS_WGSL).toContain('let det = mneeChainPdfJacobianDet(res.v1, res.v2');
  });

  it('reflection contribution core (Phase I.1 integration) — irradiance vs analytic mirror image', () => {
    // GPU-validated against the EXACT analytic mirror-image point-light irradiance
    // (mnee-reflection-validate.ts, lavapipe — deterministic, non-noisy). This is
    // the kernel-ready contribution piece (E = I·cosθ/d_unfolded²); the kernel
    // multiplies it by the receiver BRDF + visibility.
    expect(MNEE_CONNECTION_WGSL).toContain('fn mneeReflectionIrradiance(');
    expect(MNEE_CONNECTION_WGSL).toContain('let dTotal = length(lightPos - v) + length(recv - v);'); // unfolded path = dist(image,recv)
    expect(MNEE_CONNECTION_WGSL).toContain(
      'if (dTotal <= mneeLengthFloorFromScales(solverScales))',
    );
    expect(MNEE_CONNECTION_WGSL).toContain(
      'return lightIntensity * nDotL / (dTotal * dTotal);',
    );
    expect(MNEE_REFLECTION_HARNESS_WGSL).toContain(MNEE_NEWTON_WGSL);  // composes the validated solve
    expect(MNEE_REFLECTION_HARNESS_WGSL).toContain(MNEE_CONNECTION_WGSL);
  });
});
