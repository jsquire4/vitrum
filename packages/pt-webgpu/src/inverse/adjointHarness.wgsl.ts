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
