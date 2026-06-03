/**
 * adjointHarness.wgsl.ts — a standalone compute kernel that EXECUTES the WS5
 * path-replay BRDF adjoint partials (`pathTraceAdjoint.wgsl.ts`) on a flat list
 * of inputs and writes their results, so a GPU run can be A/B'd numerically
 * against the FD-validated CPU oracle (`brdfAdjoint.ts`). This is the V24
 * foundation: it upgrades the codegen-shape string-pin to an executed
 * GPU == CPU-oracle check, proving the GPU partials' arithmetic on real hardware
 * before the full single-bounce adjoint tracer wires them into inverseSession.
 *
 * It composes the SAME `PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL` the real pass uses (so
 * the partials under test are byte-identical), plus the minimal BRDF primitives
 * the adjoint references (matching `material.wgsl.ts` / the CPU oracle exactly)
 * and a `gradAccum` binding so the bundled `adjointScatter` compiles. The kernel
 * itself only evaluates the two partials per input.
 *
 * Mirrors the NRC `fusedMlpHarness` pattern (GPU == CPU oracle on lavapipe).
 */
import { PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL } from '../wgsl/pathTrace/pathTraceAdjoint.wgsl.js';

/** Floats per harness input record (vec4-aligned: 4 × vec4f = 16 floats). */
export const ADJOINT_HARNESS_INPUT_FLOATS = 16;

/** Floats per shading-adjoint input record (vec4-aligned: 6 × vec4f = 24 floats). */
export const ADJOINT_SHADING_INPUT_FLOATS = 24;

/** Fixed-point scale the adjoint atomics use (mirror of ADJOINT_GRAD_FP = 2^20). */
export const ADJOINT_GRAD_FP_TS = 1048576;

/**
 * Pack one single-bounce shading-adjoint input (24 floats), matching the WGSL
 * `ShIn` std430 layout: baseColor+roughness, normal+metallic, wo, wi, Li, target.
 */
export function packShadingAdjointInput(
  baseColor: readonly [number, number, number],
  roughness: number,
  metallic: number,
  normal: readonly [number, number, number],
  wo: readonly [number, number, number],
  wi: readonly [number, number, number],
  Li: readonly [number, number, number],
  target: readonly [number, number, number],
): number[] {
  return [
    baseColor[0], baseColor[1], baseColor[2], roughness,
    normal[0], normal[1], normal[2], metallic,
    wo[0], wo[1], wo[2], 0,
    wi[0], wi[1], wi[2], 0,
    Li[0], Li[1], Li[2], 0,
    target[0], target[1], target[2], 0,
  ];
}

/**
 * Pack one harness input into its 16-float (64-byte) record, matching the WGSL
 * `AdjIn` std430 layout below:
 *   vec4 0: baseColor.xyz, roughness
 *   vec4 1: normal.xyz,    metallic
 *   vec4 2: wo.xyz,        _pad
 *   vec4 3: wi.xyz,        _pad
 */
export function packAdjointHarnessInput(
  baseColor: readonly [number, number, number],
  roughness: number,
  metallic: number,
  normal: readonly [number, number, number],
  wo: readonly [number, number, number],
  wi: readonly [number, number, number],
): number[] {
  return [
    baseColor[0], baseColor[1], baseColor[2], roughness,
    normal[0], normal[1], normal[2], metallic,
    wo[0], wo[1], wo[2], 0,
    wi[0], wi[1], wi[2], 0,
  ];
}

export const ADJOINT_HARNESS_WGSL = /* wgsl */ `
const PI = 3.14159265358979;
const INV_PI = 0.31830988618;

// BRDF primitives — exact mirrors of material.wgsl.ts (and the CPU oracle).
fn safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  if (l < 1e-8) { return vec3f(0.0); }
  return v / l;
}
fn ggxD(nDotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-6);
}
fn smithG1(nDotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) * 0.125;
  return nDotV / max(nDotV * (1.0 - k) + k, 1e-6);
}
fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  let m5 = m2 * m2 * m;
  return f0 + (vec3f(1.0) - f0) * m5;
}

struct AdjIn {
  baseColor: vec3f, roughness: f32,
  normal:    vec3f, metallic:  f32,
  wo:        vec3f, _pad0:     f32,
  wi:        vec3f, _pad1:     f32,
}

@group(0) @binding(0) var<storage, read>       hIn:    array<AdjIn>;
@group(0) @binding(1) var<storage, read_write> hOutBC: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> hOutR:  array<vec4f>;
// gradAccum exists only so the bundled adjointScatter compiles (the harness does
// not differentiate the loss, just the BRDF partials).
@group(0) @binding(3) var<storage, read_write> gradAccum: array<atomic<i32>>;

${PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let a = hIn[i];
  hOutBC[i] = vec4f(dBrdf_dBaseColor(a.baseColor, a.roughness, a.metallic, a.normal, a.wo, a.wi), 0.0);
  hOutR[i]  = vec4f(dBrdf_dRoughness(a.baseColor, a.roughness, a.metallic, a.normal, a.wo, a.wi), 0.0);
  // Keep gradAccum + adjointScatter referenced so the bundle is the SAME string
  // the real pass uses (never executed: arrayLength is always > the never-index).
  if (i >= 0xfffffff0u) { adjointScatter(0u, hOutR[i].x); }
}
`;

/**
 * ADJOINT_SHADING_FD_WGSL — proves the path-replay adjoint's CHAIN RULE +
 * GRADIENT ACCUMULATION end-to-end against finite-difference, on-device. For a
 * single-bounce direct-lighting shading model `rendered = brdf·NdotL·Li` with an
 * L2 image loss, it computes the analytic gradient (`dLoss/dRendered · dBrdf/dθ ·
 * NdotL · Li`, scattered via fixed-point atomics — the SAME machinery the real
 * adjoint pass uses) AND a central finite-difference of the SAME f32 forward, into
 * two separate accumulators. They must agree (same forward, exact derivative) —
 * a wrong chain rule or accumulation gives an O(1) relative error.
 *
 * This is the second V24 increment: the partials are GPU-proven (adjointHarness);
 * this proves the loss→param accumulation that wraps them. What remains for the
 * full inverseSession wire is only the primary-ray RE-TRACE (the forward tracer's
 * already-validated intersection + NEE) feeding these same per-hit inputs.
 */
export const ADJOINT_SHADING_FD_WGSL = /* wgsl */ `
const PI = 3.14159265358979;
const INV_PI = 0.31830988618;
// ADJOINT_GRAD_FP (2^20) comes from the bundled PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL.
const FD_EPS = 3e-3;

fn safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  if (l < 1e-8) { return vec3f(0.0); }
  return v / l;
}
fn ggxD(nDotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-6);
}
fn smithG1(nDotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) * 0.125;
  return nDotV / max(nDotV * (1.0 - k) + k, 1e-6);
}
fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  let m5 = m2 * m2 * m;
  return f0 + (vec3f(1.0) - f0) * m5;
}

// Forward Cook-Torrance BRDF — exact mirror of bsdf.wgsl.ts / brdfAdjoint.ts.
fn evaluateBrdf(baseColor: vec3f, roughness: f32, metallic: f32, normal: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  let f0 = vec3f(0.04) + (baseColor - vec3f(0.04)) * metallic;
  let f = fresnelSchlick(vDotH, f0);
  let alpha = max(roughness * roughness, 1e-3);
  let d = ggxD(nDotH, alpha);
  let g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  let specScale = (d * g) / max(4.0 * nDotV * nDotL, 1e-6);
  let kd0 = 1.0 - metallic;
  var outv = vec3f(0.0);
  for (var c: u32 = 0u; c < 3u; c = c + 1u) {
    let spec = specScale * f[c];
    let kd = (1.0 - f[c]) * kd0;
    let diff = kd * baseColor[c] * INV_PI;
    outv[c] = diff + spec;
  }
  return outv;
}

struct ShIn {
  baseColor: vec3f, roughness: f32,
  normal:    vec3f, metallic:  f32,
  wo:        vec3f, _p0:       f32,
  wi:        vec3f, _p1:       f32,
  Li:        vec3f, _p2:       f32,
  tgt:       vec3f, _p3:       f32,
}

@group(0) @binding(0) var<storage, read>       shIn:    array<ShIn>;
@group(0) @binding(1) var<storage, read_write> gradAdj: array<atomic<i32>>; // [0..3] analytic
@group(0) @binding(2) var<storage, read_write> gradFd:  array<atomic<i32>>; // [0..3] finite-diff
// gradAccum exists only so the bundled adjointScatter compiles (we scatter via
// scatterTo into gradAdj/gradFd; the bundle's own scatter target goes unused).
@group(0) @binding(3) var<storage, read_write> gradAccum: array<atomic<i32>>;

${PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL}

fn scatterTo(slot: u32, g: f32, isFd: bool) {
  let q = i32(round(g * ADJOINT_GRAD_FP));
  if (isFd) { atomicAdd(&gradFd[slot], q); } else { atomicAdd(&gradAdj[slot], q); }
}

// rendered = brdf·NdotL·Li ; loss = |rendered − target|².
fn shLoss(s: ShIn, bc: vec3f, rough: f32) -> f32 {
  let nDotL = max(dot(s.normal, s.wi), 0.0);
  let rendered = evaluateBrdf(bc, rough, s.metallic, s.normal, s.wo, s.wi) * nDotL * s.Li;
  let dlt = rendered - s.tgt;
  return dot(dlt, dlt);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&shIn)) { return; }
  let s = shIn[i];
  let nDotL = max(dot(s.normal, s.wi), 0.0);
  let rendered = evaluateBrdf(s.baseColor, s.roughness, s.metallic, s.normal, s.wo, s.wi) * nDotL * s.Li;
  let dLoss_dR = 2.0 * (rendered - s.tgt);               // ∂loss/∂rendered (vec3)

  // ── analytic adjoint: ∂loss/∂θ = ∂loss/∂rendered · ∂rendered/∂θ ──
  // baseColor is diagonal: ∂rendered_c/∂baseColor_c = dBrdf_dBaseColor_c·NdotL·Li_c.
  let dBC = dBrdf_dBaseColor(s.baseColor, s.roughness, s.metallic, s.normal, s.wo, s.wi);
  let gBC = dLoss_dR * dBC * nDotL * s.Li;               // vec3, channel-diagonal
  scatterTo(0u, gBC.x, false);
  scatterTo(1u, gBC.y, false);
  scatterTo(2u, gBC.z, false);
  // roughness couples all channels: ∂loss/∂rough = Σ_c dLoss_dR_c·dBrdf_dRough_c·NdotL·Li_c.
  let dR = dBrdf_dRoughness(s.baseColor, s.roughness, s.metallic, s.normal, s.wo, s.wi);
  let gR = dot(dLoss_dR, dR * nDotL * s.Li);
  scatterTo(3u, gR, false);

  // ── central finite-difference of the SAME f32 forward ──
  for (var j: u32 = 0u; j < 3u; j = j + 1u) {
    var bcP = s.baseColor; bcP[j] = bcP[j] + FD_EPS;
    var bcM = s.baseColor; bcM[j] = bcM[j] - FD_EPS;
    scatterTo(j, (shLoss(s, bcP, s.roughness) - shLoss(s, bcM, s.roughness)) / (2.0 * FD_EPS), true);
  }
  scatterTo(3u, (shLoss(s, s.baseColor, s.roughness + FD_EPS) - shLoss(s, s.baseColor, s.roughness - FD_EPS)) / (2.0 * FD_EPS), true);

  // Keep the bundled adjointScatter/gradAccum live (never executed) so layout:auto
  // retains binding 3 and the partials bundle stays byte-identical to production.
  if (i >= 0xfffffff0u) { adjointScatter(0u, gR); }
}
`;
