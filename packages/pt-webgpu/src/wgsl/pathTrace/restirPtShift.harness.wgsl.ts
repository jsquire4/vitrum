/**
 * restirPtShift.harness.wgsl.ts — GPU validation harness shader for the
 * ReSTIR-PT reconnection-shift Jacobian. NOT composed into any production
 * pipeline; used by wsl-gpu/scripts/restir-pt-shift-validate.ts.
 *
 * Production WGSL (RESTIR_PT_SHIFT_WGSL) remains in restirPtShift.wgsl.ts.
 *
 * Consumers (wsl-gpu scripts — repointed from restirPtShift.wgsl.ts):
 *   scripts/restir-pt-shift-validate.ts
 */

import {
  RESTIR_PT_SHIFT_WGSL,
} from './restirPtShift.wgsl.js';

/** @public — oracle-referenced: wsl-gpu/scripts/restir-pt-shift-validate.ts imports this harness for GPU validation. */
export { RESTIR_PT_SHIFT_WGSL };

/** Floats per harness config record (vec4-aligned: 3 × vec4 = 12 floats):
 *  [xq.xyz, _], [xr.xyz, _], [xs.xyz, _]. The reconnection-vertex normal n_s and
 *  its tangents are fixed in harness space (+z normal, +x/+y tangents) exactly as
 *  the MNEE harnesses fix their plane. */
export const RESTIR_PT_SHIFT_INPUT_FLOATS = 12;

/** Pack one harness config: the source pre-reconnection vertex x_q, the target
 *  pre-reconnection vertex x_r, and the shared reconnection vertex x_s. */
export function packRestirPtShiftInput(
  xq: readonly [number, number, number],
  xr: readonly [number, number, number],
  xs: readonly [number, number, number],
): number[] {
  return [
    xq[0], xq[1], xq[2], 0,
    xr[0], xr[1], xr[2], 0,
    xs[0], xs[1], xs[2], 0,
  ];
}

/**
 * Harness kernel. The reconnection vertex x_s is on a plane with normal +z and
 * tangents +x/+y in harness space (same convention as the MNEE harnesses). Per
 * config it writes, in TWO vec4 records:
 *   record 0 = [ analyticJ,   gSource, gTarget,    0 ]
 *   record 1 = [ saSource,    saTarget, 0,         0 ]
 * where saSource/saTarget are the geometry-MEASURED |dω/dA_s| at x_q / x_r (the
 * basis-free area-determinants). The validator finite-differences x_s over its
 * (s,t) area params, forms FD |dω/dA_s| at each endpoint, and asserts
 *   analyticJ == saTarget/saSource  AND  saSource==gSource, saTarget==gTarget
 * — i.e. the analytic ratio equals the ratio of the directly-measured (and
 * FD-confirmed) solid-angle-to-area measure changes. (n_s = +z, ts = +x, tt = +y.)
 */
export const RESTIR_PT_SHIFT_HARNESS_WGSL = /* wgsl */ `
${RESTIR_PT_SHIFT_WGSL}

struct ShiftIn { xq: vec3f, _p0: f32, xr: vec3f, _p1: f32, xs: vec3f, _p2: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<ShiftIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>; // 2 vec4 / config

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let ns = vec3f(0.0, 0.0, 1.0);
  let ts = vec3f(1.0, 0.0, 0.0);
  let tt = vec3f(0.0, 1.0, 0.0);
  let gSource = restirPtReconnectionGeometryTerm(c.xq, c.xs, ns);
  let gTarget = restirPtReconnectionGeometryTerm(c.xr, c.xs, ns);
  let J       = restirPtShiftJacobian(c.xq, c.xr, c.xs, ns);
  let saSource = restirPtSolidAngleAreaDeriv(c.xq, c.xs, ts, tt);
  let saTarget = restirPtSolidAngleAreaDeriv(c.xr, c.xs, ts, tt);
  hOut[i * 2u + 0u] = vec4f(J, gSource, gTarget, 0.0);
  hOut[i * 2u + 1u] = vec4f(saSource, saTarget, 0.0, 0.0);
}
`;
