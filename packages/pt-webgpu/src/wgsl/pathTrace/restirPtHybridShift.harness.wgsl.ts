/**
 * restirPtHybridShift.harness.wgsl.ts — GPU validation harness shader for the
 * ReSTIR-PT hybrid-shift Jacobian. NOT composed into any production pipeline;
 * used by wsl-gpu/scripts/restir-pt-hybrid-shift-validate.ts.
 *
 * Production WGSL (RESTIR_PT_HYBRID_SHIFT_WGSL) remains in
 * restirPtHybridShift.wgsl.ts.
 *
 * Consumers (wsl-gpu scripts — repointed from restirPtHybridShift.wgsl.ts):
 *   scripts/restir-pt-hybrid-shift-validate.ts
 */

import {
  RESTIR_PT_HYBRID_SHIFT_WGSL,
} from './restirPtHybridShift.wgsl.js';

/** @public — oracle-referenced: wsl-gpu/scripts/restir-pt-hybrid-shift-validate.ts imports this harness for GPU validation. */
export { RESTIR_PT_HYBRID_SHIFT_WGSL };

/** Floats per harness config record. A single-replayed-segment hybrid-shift config
 *  needs, for BOTH the source (q) and the offset (r) domain, the replayed bounce's
 *  geometry (vertex, shading normal, wo, wi) + material (baseColor, roughness,
 *  metallic, transmission, etaTOverI), PLUS the shared reconnection vertex x_s.
 *
 *  Layout (std430 vec4-aligned; 12 × vec4 = 48 floats):
 *   [ 0] xq.xyz, _            (source replayed-bounce vertex = pre-reconnection vtx)
 *   [ 1] nq.xyz, _            (source shading normal at xq)
 *   [ 2] woq.xyz, _           (source outgoing dir at xq, toward the previous vtx)
 *   [ 3] wiq.xyz, _           (source sampled prefix dir at xq, toward x_s)
 *   [ 4] xr.xyz, _            (offset replayed-bounce vertex)
 *   [ 5] nr.xyz, _            (offset shading normal at xr)
 *   [ 6] wor.xyz, _           (offset outgoing dir at xr)
 *   [ 7] wir.xyz, _           (offset sampled prefix dir at xr, toward x_s)
 *   [ 8] xs.xyz, _            (shared reconnection vertex)
 *   [ 9] baseColor.rgb, rough (q-domain material; rough in .w)
 *   [10] metallic, transmission, etaTOverI, _   (q-domain interface scalars)
 *   [11] baseColorR.rgb, roughR           (r-domain material; roughR in .w)
 *  (r-domain metallic/trans/etaTOverI reuse the q scalars in this harness — same interface
 *   under the shift in the canonical case; the validator sets them equal. The WGSL
 *   fn takes them as explicit params so a future heterogeneous-surface replay can
 *   pass distinct values.) */
export const RESTIR_PT_HYBRID_SHIFT_INPUT_FLOATS = 48;

/** Pack one single-segment hybrid-shift harness config. */
export function packRestirPtHybridShiftInput(cfg: {
  xq: readonly [number, number, number];
  nq: readonly [number, number, number];
  woq: readonly [number, number, number];
  wiq: readonly [number, number, number];
  xr: readonly [number, number, number];
  nr: readonly [number, number, number];
  wor: readonly [number, number, number];
  wir: readonly [number, number, number];
  xs: readonly [number, number, number];
  baseColor: readonly [number, number, number];
  roughness: number;
  metallic: number;
  transmission: number;
  etaTOverI: number;
}): number[] {
  const { xq, nq, woq, wiq, xr, nr, wor, wir, xs, baseColor, roughness, metallic, transmission, etaTOverI } = cfg;
  return [
    xq[0], xq[1], xq[2], 0,
    nq[0], nq[1], nq[2], 0,
    woq[0], woq[1], woq[2], 0,
    wiq[0], wiq[1], wiq[2], 0,
    xr[0], xr[1], xr[2], 0,
    nr[0], nr[1], nr[2], 0,
    wor[0], wor[1], wor[2], 0,
    wir[0], wir[1], wir[2], 0,
    xs[0], xs[1], xs[2], 0,
    baseColor[0], baseColor[1], baseColor[2], roughness,
    metallic, transmission, etaTOverI, 0,
    baseColor[0], baseColor[1], baseColor[2], roughness,
  ];
}

/**
 * Harness kernel. Per single-segment hybrid-shift config it writes, in THREE vec4
 * records, every quantity the validator needs to check (★) against a finite
 * difference of the ACTUAL hybrid-shift map:
 *
 *   record 0 = [ J_hybrid,  J_geom,   J_replay,  0 ]
 *   record 1 = [ pq,        pr,       gSource,   gTarget ]
 *   record 2 = [ saSource,  saTarget, 0,         0 ]
 *
 * where:
 *   • J_hybrid = J_geom · (pq/pr) is the analytic hybrid Jacobian,
 *   • pq / pr are the source / target replayed-segment BSDF pdfs,
 *   • gSource / gTarget are the reconnection half-G terms,
 *   • saSource / saTarget are the geometry-MEASURED |dω/dA_s| at x_q / x_r (the
 *     basis-free solid-angle⇄area determinants — the FD-able geometric measure,
 *     reused from the restirPtShift discipline).
 *
 * The validator finite-differences x_s over its (s,t) area params (to FD the
 * geometric factor) AND finite-differences the canonical random numbers u through
 * the actual replay sampler (to FD the BSDF-pdf factor), forms the product, and
 * asserts analytic J_hybrid == FD J_hybrid. (n_s = +z, ts = +x, tt = +y, matching
 * the restirPtShift harness.)
 */
export const RESTIR_PT_HYBRID_SHIFT_HARNESS_WGSL = /* wgsl */ `
${RESTIR_PT_HYBRID_SHIFT_WGSL}

// Geometry-measured |dω_a/dA_s| at a pre-reconnection vertex (the FD-able measure,
// identical to restirPtShift.wgsl.ts::restirPtSolidAngleAreaDeriv).
fn rptHybridSolidAngleAreaDeriv(xa: vec3f, xs: vec3f, ts: vec3f, tt: vec3f) -> f32 {
  let d = xs - xa;
  let dist = length(d);
  if (dist < 1e-8) { return 0.0; }
  let w = d / dist;
  let dw_ds = (ts - w * dot(w, ts)) / dist;
  let dw_dt = (tt - w * dot(w, tt)) / dist;
  return length(cross(dw_ds, dw_dt));
}

struct HybridIn {
  xq: vec3f,  _0: f32,
  nq: vec3f,  _1: f32,
  woq: vec3f, _2: f32,
  wiq: vec3f, _3: f32,
  xr: vec3f,  _4: f32,
  nr: vec3f,  _5: f32,
  wor: vec3f, _6: f32,
  wir: vec3f, _7: f32,
  xs: vec3f,  _8: f32,
  matQ: vec4f,   // baseColor.rgb, roughness
  matQ2: vec4f,  // metallic, transmission, etaTOverI, _
  matR: vec4f,   // baseColorR.rgb, roughnessR
}
@group(0) @binding(0) var<storage, read>       hIn:  array<HybridIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>; // 3 vec4 / config

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let ns = vec3f(0.0, 0.0, 1.0);
  let ts = vec3f(1.0, 0.0, 0.0);
  let tt = vec3f(0.0, 1.0, 0.0);

  let baseColorQ = c.matQ.xyz;
  let roughnessQ = c.matQ.w;
  let metallicQ = c.matQ2.x;
  let transmissionQ = c.matQ2.y;
  let etaTOverIQ = c.matQ2.z;
  let baseColorR = c.matR.xyz;
  let roughnessR = c.matR.w;
  // r-domain metallic/trans/etaTOverI reuse the q scalars (same interface under the shift).

  let gSource = rptHybridReconnectionGeometryTerm(c.xq, c.xs, ns);
  let gTarget = rptHybridReconnectionGeometryTerm(c.xr, c.xs, ns);
  let jGeom   = rptHybridGeomJacobian(c.xq, c.xr, c.xs, ns);
  let pq = rptHybridBsdfReplayPdf(baseColorQ, roughnessQ, metallicQ, transmissionQ, etaTOverIQ, c.nq, c.woq, c.wiq);
  let pr = rptHybridBsdfReplayPdf(baseColorR, roughnessR, metallicQ, transmissionQ, etaTOverIQ, c.nr, c.wor, c.wir);
  let jReplay = rptHybridReplaySegmentJacobian(
    baseColorQ, roughnessQ, metallicQ, transmissionQ, etaTOverIQ, c.nq, c.woq, c.wiq,
    baseColorR, roughnessR, metallicQ, transmissionQ, etaTOverIQ, c.nr, c.wor, c.wir);
  let jHybrid = rptHybridShiftJacobian(
    c.xq, c.xr, c.xs, ns,
    baseColorQ, roughnessQ, metallicQ, transmissionQ, etaTOverIQ, c.nq, c.woq, c.wiq,
    baseColorR, roughnessR, metallicQ, transmissionQ, etaTOverIQ, c.nr, c.wor, c.wir);
  let saSource = rptHybridSolidAngleAreaDeriv(c.xq, c.xs, ts, tt);
  let saTarget = rptHybridSolidAngleAreaDeriv(c.xr, c.xs, ts, tt);

  hOut[i * 3u + 0u] = vec4f(jHybrid, jGeom, jReplay, 0.0);
  hOut[i * 3u + 1u] = vec4f(pq, pr, gSource, gTarget);
  hOut[i * 3u + 2u] = vec4f(saSource, saTarget, 0.0, 0.0);
}
`;
