/**
 * mneeNewton.harness.wgsl.ts — GPU validation harness shaders for the MNEE
 * Newton solver and chain solve. These are STANDALONE compute shaders used by
 * wsl-gpu validation scripts (mnee-*-validate.ts) and NOT composed into any
 * production pipeline.
 *
 * Production WGSL exports (MNEE_NEWTON_WGSL, MNEE_CHAIN_WGSL,
 * MNEE_CONNECTION_WGSL) remain in mneeNewton.wgsl.ts; this file only holds the
 * harness shaders and the TypeScript helpers for packing harness inputs.
 *
 * Consumers (wsl-gpu scripts — repointed from mneeNewton.wgsl.ts):
 *   scripts/mnee-newton-validate.ts
 *   scripts/mnee-newton-jac-validate.ts
 *   scripts/mnee-jacobian-validate.ts
 *   scripts/mnee-pdf-validate.ts
 *   scripts/mnee-chain-validate.ts
 *   scripts/mnee-chain-pdf-validate.ts
 *   scripts/mnee-reflection-validate.ts  (imports MNEE_REFLECTION_HARNESS_WGSL)
 */

import {
  MNEE_NEWTON_WGSL,
  MNEE_CHAIN_WGSL,
  MNEE_CONNECTION_WGSL,
  MNEE_NEWTON_MAX_ITERS,
  MNEE_CHAIN_MAX_ITERS,
} from './mneeNewton.wgsl.js';

/** @public — oracle-referenced: wsl-gpu/scripts/mnee-*-validate.ts harnesses embed these as WGSL literal constants. */
export { MNEE_NEWTON_MAX_ITERS, MNEE_CHAIN_MAX_ITERS };

/** Floats per harness input record (vec4-aligned: 3 × vec4 = 12 floats). */
export const MNEE_HARNESS_INPUT_FLOATS = 12;

/** Pack one harness config: receiver, light, and the mirror plane point (the
 *  plane is z=0 with normal +z in harness space, so only the point varies). */
export function packMneeHarnessInput(
  receiver: readonly [number, number, number],
  light: readonly [number, number, number],
  planePoint: readonly [number, number, number],
  etaI = 1,
  etaT = 1,
): number[] {
  return [
    receiver[0], receiver[1], receiver[2], etaI,
    light[0], light[1], light[2], etaT,
    planePoint[0], planePoint[1], planePoint[2], 0,
  ];
}

/** Harness kernel: runs the Newton solve per config (mirror plane z = planePoint.z,
 *  normal +z, tangents +x/+y) and writes the converged vertex + final residual. */
export const MNEE_NEWTON_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}

// recv.w = etaI (IOR on the light side), light.w = etaT (IOR on the receiver side).
struct MneeIn { recv: vec3f, etaI: f32, light: vec3f, etaT: f32, planePoint: vec3f, _p2: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<MneeIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>; // xyz = vertex, w = residual

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let nm = vec3f(0.0, 0.0, 1.0);
  let tu = vec3f(1.0, 0.0, 0.0);
  let tv = vec3f(0.0, 1.0, 0.0);
  let r = mneeNewtonSolve(c.planePoint, nm, tu, tv, c.recv, c.light, c.etaI, c.etaT, ${MNEE_NEWTON_MAX_ITERS}u);
  hOut[i] = vec4f(r.vertex, r.residual);
}
`;

/** Newton-Jacobian harness: writes the ANALYTIC residual Jacobian ∂r/∂(a,b) and
 *  the finite-difference reference at a generic test vertex (2 vec4 per config:
 *  [analytic j00,j10,j01,j11], [FD j00,j10,j01,j11]). The validation asserts
 *  analytic == FD — proving the exact Jacobian that drives the Newton step. */
export const MNEE_NEWTON_JAC_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}

struct MneeIn { recv: vec3f, etaI: f32, light: vec3f, etaT: f32, planePoint: vec3f, _p2: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<MneeIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let nm = vec3f(0.0, 0.0, 1.0);
  let tu = vec3f(1.0, 0.0, 0.0);
  let tv = vec3f(0.0, 1.0, 0.0);
  // A generic (non-degenerate) test vertex on the plane — the Jacobian formula
  // is point-independent, so analytic == FD here proves it everywhere.
  let v = c.planePoint + 0.1 * tu + 0.05 * tv;
  let jac = mneeResidualJacobian(v, c.recv, c.light, nm, tu, tv, c.etaI, c.etaT);
  // CENTRAL-difference reference (O(eps²)). Forward difference's O(eps) truncation
  // is ~1e-2 on the highly-nonlinear refraction residual — enough to spuriously
  // diverge from the EXACT analytic on small components.
  let eps = 1e-3;
  let ra_p = mneeHalfVectorResidual2d(v + eps * tu, c.recv, c.light, nm, tu, tv, c.etaI, c.etaT);
  let ra_m = mneeHalfVectorResidual2d(v - eps * tu, c.recv, c.light, nm, tu, tv, c.etaI, c.etaT);
  let rb_p = mneeHalfVectorResidual2d(v + eps * tv, c.recv, c.light, nm, tu, tv, c.etaI, c.etaT);
  let rb_m = mneeHalfVectorResidual2d(v - eps * tv, c.recv, c.light, nm, tu, tv, c.etaI, c.etaT);
  hOut[i * 2u + 0u] = vec4f(jac.cA.x, jac.cA.y, jac.cB.x, jac.cB.y);
  hOut[i * 2u + 1u] = vec4f((ra_p.x - ra_m.x) / (2.0 * eps), (ra_p.y - ra_m.y) / (2.0 * eps), (rb_p.x - rb_m.x) / (2.0 * eps), (rb_p.y - rb_m.y) / (2.0 * eps));
}
`;

/** Jacobian harness: solves, then writes the manifold derivative d(a,b)/d(light)
 *  (3 vec4 per config: [vertex, residual], [dadL.xyz, dbdL.x], [dbdL.yz, _, _]).
 *  The validation FD-re-solves to confirm the analytic Jacobian == finite diff. */
export const MNEE_JACOBIAN_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}

struct MneeIn { recv: vec3f, etaI: f32, light: vec3f, etaT: f32, planePoint: vec3f, _p2: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<MneeIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let nm = vec3f(0.0, 0.0, 1.0);
  let tu = vec3f(1.0, 0.0, 0.0);
  let tv = vec3f(0.0, 1.0, 0.0);
  let r = mneeNewtonSolve(c.planePoint, nm, tu, tv, c.recv, c.light, c.etaI, c.etaT, ${MNEE_NEWTON_MAX_ITERS}u);
  let jac = mneeManifoldJacobian(r.vertex, nm, tu, tv, c.recv, c.light, c.etaI, c.etaT);
  hOut[i * 3u + 0u] = vec4f(r.vertex, r.residual);
  hOut[i * 3u + 1u] = vec4f(jac.dadL, jac.dbdL.x);
  hOut[i * 3u + 2u] = vec4f(jac.dbdL.y, jac.dbdL.z, 0.0, 0.0);
}
`;

/** PDF harness: solves, then writes the connection-PDF geometric factor
 *  |dω_recv/dA_light| (per config: vec4[vertex.xyz, |det|]). The validation
 *  FD-re-solves over the light's (x,y) area params to confirm analytic == FD. */
export const MNEE_PDF_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}

struct MneeIn { recv: vec3f, etaI: f32, light: vec3f, etaT: f32, planePoint: vec3f, _p2: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<MneeIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>; // xyz = vertex, w = |dω/dA|

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let nm = vec3f(0.0, 0.0, 1.0);
  let tu = vec3f(1.0, 0.0, 0.0);
  let tv = vec3f(0.0, 1.0, 0.0);
  let r = mneeNewtonSolve(c.planePoint, nm, tu, tv, c.recv, c.light, c.etaI, c.etaT, ${MNEE_NEWTON_MAX_ITERS}u);
  let jac = mneeManifoldJacobian(r.vertex, nm, tu, tv, c.recv, c.light, c.etaI, c.etaT);
  let det = mneePdfJacobianDet(r.vertex, c.recv, jac.dadL, jac.dbdL, tu, tv);
  hOut[i] = vec4f(r.vertex, det);
}
`;

/** Chain harness: a glass slab — plane 1 at z=0, plane 2 at z=−slabD (both +z
 *  normal, +x/+y tangents); air→glass→air (eta 1 / etaGlass / 1). Writes the two
 *  converged vertices + the final 4D residual. */
export const MNEE_CHAIN_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}
${MNEE_CHAIN_WGSL}

struct ChainIn { lightP: vec3f, slabD: f32, recv: vec3f, etaGlass: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<ChainIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>; // [v1.xyz, residual], [v2.xyz, iters]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let n = vec3f(0.0, 0.0, 1.0);
  let tu = vec3f(1.0, 0.0, 0.0);
  let tv = vec3f(0.0, 1.0, 0.0);
  let p1 = vec3f(0.0, 0.0, 0.0);
  let p2 = vec3f(0.0, 0.0, -c.slabD);
  let res = mneeNewtonSolveChain2(p1, n, tu, tv, p2, n, tu, tv, c.lightP, c.recv, 1.0, c.etaGlass, c.etaGlass, 1.0, ${MNEE_CHAIN_MAX_ITERS}u);
  hOut[i * 2u + 0u] = vec4f(res.v1, res.residual);
  hOut[i * 2u + 1u] = vec4f(res.v2, f32(res.iters));
}
`;

/** Chain-PDF harness: solve the glass-slab chain, then write the chain connection-
 *  PDF factor |dω_recv/dA_light| (light area axes +x/+y). Validated analytic == FD
 *  re-solve over the light's area params. Per config: [v1.xyz, residual], [v2.xyz, det]. */
export const MNEE_CHAIN_PDF_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}
${MNEE_CHAIN_WGSL}

struct ChainIn { lightP: vec3f, slabD: f32, recv: vec3f, etaGlass: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<ChainIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let n = vec3f(0.0, 0.0, 1.0);
  let tu = vec3f(1.0, 0.0, 0.0);
  let tv = vec3f(0.0, 1.0, 0.0);
  let p1 = vec3f(0.0, 0.0, 0.0);
  let p2 = vec3f(0.0, 0.0, -c.slabD);
  let res = mneeNewtonSolveChain2(p1, n, tu, tv, p2, n, tu, tv, c.lightP, c.recv, 1.0, c.etaGlass, c.etaGlass, 1.0, ${MNEE_CHAIN_MAX_ITERS}u);
  let det = mneeChainPdfJacobianDet(res.v1, res.v2, n, tu, tv, n, tu, tv, c.lightP, c.recv, 1.0, c.etaGlass, c.etaGlass, 1.0, vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0));
  hOut[i * 2u + 0u] = vec4f(res.v1, res.residual);
  hOut[i * 2u + 1u] = vec4f(res.v2, det);
}
`;

/** Reflection harness: a mirror at z=0 (+z normal, +x/+y tangents); writes the MNEE
 *  reflection irradiance.rgb per config. Validated against the analytic mirror-image
 *  point-light irradiance (deterministic ground truth). */
export const MNEE_REFLECTION_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}
${MNEE_CONNECTION_WGSL}

struct ReflIn { recv: vec3f, _p0: f32, recvNormal: vec3f, _p1: f32, lightPos: vec3f, _p2: f32, intensity: vec3f, _p3: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<ReflIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let mP = vec3f(0.0, 0.0, 0.0);
  let mN = vec3f(0.0, 0.0, 1.0);
  let mTu = vec3f(1.0, 0.0, 0.0);
  let mTv = vec3f(0.0, 1.0, 0.0);
  let e = mneeReflectionIrradiance(c.recv, c.recvNormal, mP, mN, mTu, mTv, c.lightPos, c.intensity);
  hOut[i] = vec4f(e, 0.0);
}
`;
