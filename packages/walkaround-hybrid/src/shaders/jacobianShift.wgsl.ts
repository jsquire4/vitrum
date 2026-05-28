/**
 * ReSTIR-GI Jacobian reconnection shift.
 *
 * Split out of common.wgsl.ts (T9-stepA): `jacobianReconnectionShift`
 * (Eq. 11 — cosine ratio × inverse-square distance ratio, clamped) used by
 * the GI spatial-reuse pass.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const JACOBIAN_SHIFT_WGSL = /* wgsl */ `// ============================================================
// Jacobian reconnection shift
// ============================================================
fn jacobianReconnectionShift(
  xv_r: vec3f, nv_r: vec3f,  // current pixel primary hit + normal
  xv_q: vec3f,               // neighbor pixel primary hit (source)
  xs:   vec3f, ns: vec3f,    // reconnection vertex + normal (invariant)
) -> f32 {
  let dq = xv_q - xs;
  let dr = xv_r - xs;
  let dq_len2 = dot(dq, dq);
  let dr_len2 = dot(dr, dr);

  if (dr_len2 < 1e-8 || dq_len2 < 1e-8) { return 0.0; }

  let inv_dq_len = inverseSqrt(dq_len2);
  let inv_dr_len = inverseSqrt(dr_len2);

  let cos_theta_q = dot(ns, dq * inv_dq_len);
  let cos_theta_r = dot(ns, dr * inv_dr_len);

  if (cos_theta_q <= 1e-4 || cos_theta_r <= 1e-4) { return 0.0; }

  // Eq. 11 reconnection shift: cosine ratio x inverse-square distance ratio.
  let J = (cos_theta_r / cos_theta_q) * (dq_len2 / dr_len2);
  return clamp(J, 0.1, 10.0);
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const JACOBIAN_SHIFT_MODULE: WgslModule = {
  name: "jacobianShift",
  source: JACOBIAN_SHIFT_WGSL,
  requires: [],
};
