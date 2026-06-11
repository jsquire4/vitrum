/**
 * restirPtShift.wgsl.ts — the ReSTIR-PT / GRIS RECONNECTION-SHIFT Jacobian for a
 * GENERAL reconnection vertex (the hero-stack, arbitrary-path-length form of the
 * 1-bounce GI special case in walkaround-hybrid's `jacobianShift.wgsl.ts`).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * What the reconnection shift is
 * ════════════════════════════════════════════════════════════════════════════
 * GRIS (Lin et al. 2022) resamples PATHS across pixels. A path generated for a
 * source pixel q can be reused at a target pixel r only after a SHIFT map T
 * re-roots it into r's integration domain, and its contribution must be
 * re-weighted by the change-of-variables Jacobian |∂T/∂·|.
 *
 * The RECONNECTION shift is the workhorse shift. A reconnection path is split at
 * a reconnection vertex x_s (the first vertex "rough enough" / far enough to
 * reconnect to). The shift:
 *   • keeps the SUFFIX (everything from x_s onward, including x_s's cached
 *     outgoing radiance L_o) FIXED in world space,
 *   • keeps each pixel's own PREFIX (camera→…→the vertex BEFORE x_s; for the
 *     canonical single-reconnection case this is just the primary/visible vertex
 *     x_q for the source, x_r for the target),
 *   • swaps the reconnection EDGE  x_q ↔ x_s  for  x_r ↔ x_s  (a fresh edge).
 *
 *   T : (x_q, x_s, n_s)  ↦  (x_r, x_s, n_s)        [x_s, n_s, suffix held fixed]
 *
 * This module is general over the reconnection vertex: x_q and x_r are the
 * pre-reconnection vertices of the source and target paths (NOT restricted to a
 * 1-bounce GI primary hit — for a length-k reuse they are the (k−1)-th vertices
 * of each path, the suffix from x_s onward being the shared tail). The geometry
 * of the shift's Jacobian depends ONLY on this last pre-reconnection edge, which
 * is exactly why a single reconnection-edge Jacobian generalizes to arbitrary
 * path length (Lin 2022 §5; the prefix is replayed deterministically, the suffix
 * is invariant, so neither contributes a Jacobian factor — only the reconnection
 * edge's solid-angle⇄area conversion changes).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * The Jacobian (geometry-only reconnection shift) and WHY it is correct
 * ════════════════════════════════════════════════════════════════════════════
 * The reconnection sample's shared resampling coordinate is the AREA of x_s (its
 * world-space position is what both domains integrate over). The path integrand
 * across the reconnection edge is written in SOLID-ANGLE measure about the
 * pre-reconnection vertex. The solid-angle ⇄ area change of variables AT x_s is
 * the half-G geometry term (destination-cosine only):
 *
 *     G(x_a ↔ x_s) = |cos θ_s(a)| / ‖x_a − x_s‖²,
 *        cos θ_s(a) = n_s · (x_a − x_s)/‖x_a − x_s‖      [the x_s-side cosine]
 *
 * i.e.  dA_s = (‖x_a − x_s‖² / |cos θ_s(a)|) · dω_a = (1/G(x_a↔x_s)) · dω_a,
 * so    dω_a/dA_s = G(x_a ↔ x_s).
 *
 * The shift carries the SOURCE solid-angle sample (at x_q) to the TARGET
 * solid-angle sample (at x_r) THROUGH the shared area element dA_s. By the chain
 * rule the composite Jacobian is therefore the RATIO of the two edges' half-G:
 *
 *     |∂T/∂·| = dω_r/dω_q = (dω_r/dA_s)·(dA_s/dω_q)
 *             = G(x_r ↔ x_s) / G(x_q ↔ x_s)
 *             = ( |cos θ_s(r)| · ‖x_q − x_s‖² )
 *             / ( |cos θ_s(q)| · ‖x_r − x_s‖² )          (Lin 2022 Eq. 11/12)
 *
 * The PRE-RECONNECTION-vertex cosines (at x_q, x_r) do NOT appear: they belong to
 * the integrand's directional density (the BSDF·cosine target factor), which is
 * carried by the resampling TARGET function p̂ in each native domain, not by this
 * geometric measure-conversion Jacobian. (Same destination-cosine-only "half-G"
 * distinction as `bdptMIS.ts::convertDensitySAtoArea` and the GI oracle
 * `@vitrum/shared-samplers/reconnectionShift.ts`, which this module's pure
 * geometry mirrors EXACTLY — see `restirPtReconnectionGeometryTerm` ==
 * `reconnectionGeometryTerm`, `restirPtShiftJacobian` == `reconnectionJacobian`.)
 *
 * Reciprocity: T⁻¹ swaps the roles of x_q and x_r, so |∂T⁻¹/∂·| = 1/|∂T/∂·| and
 * |∂T/∂·|·|∂T⁻¹/∂·| = 1 (pinned by the unit test + the CPU oracle's test).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Validation (self-validating MATH HARNESS — no radiometric A/B)
 * ════════════════════════════════════════════════════════════════════════════
 * `restirPtShiftJacobianFromConfig` (the harness entry) computes BOTH the
 * analytic ratio above AND the ACTUAL measure-change finite difference: it
 * parameterizes x_s by 2 AREA coords (s,t) on its tangent plane and, via
 * `restirPtSolidAngleAreaDeriv`, finite-differences the solid-angle subtended at
 * each pre-reconnection vertex as x_s moves — |∂ω_a/∂s × ∂ω_a/∂t| = the per-edge
 * |dω_a/dA_s| = G(x_a↔x_s) measured directly off the shift geometry. The FD
 * Jacobian is the ratio of those two basis-free area-determinants, and the
 * harness asserts analytic == FD (wsl-gpu scripts/restir-pt-shift-validate.ts,
 * lavapipe). This is the genuine change-of-variables FD of the shift map (the
 * same discipline as mnee-pdf-validate.ts), NOT a re-print of the closed form.
 *
 * The GI version (`walkaround-hybrid/.../jacobianShift.wgsl.ts`) is the clamped,
 * real-time 1-bounce special case of THIS; the unclamped general form here is the
 * foundation a future hero-stack path-space reservoir integrator (a pt-webgpu
 * ReSTIR-PT) needs. The HYBRID shift (a BSDF-pdf ratio for a replayed-prefix
 * segment) is a follow-up — see the note at the bottom; this module proves the
 * pure RECONNECTION (geometry) Jacobian rigorously.
 *
 * Ref: Lin, Kettunen, Bitterli, Pantaleoni, Jakob, Nowrouzezahrai,
 *      "Generalized Resampled Importance Sampling: Foundations of ReSTIR,"
 *      ACM TOG 41(4) / SIGGRAPH 2022, §5 (reconnection shift), Eq. 11/12.
 */

// NOTE: Harness exports (RESTIR_PT_SHIFT_HARNESS_WGSL, packRestirPtShiftInput,
// RESTIR_PT_SHIFT_INPUT_FLOATS) live in restirPtShift.harness.wgsl.ts.

export const RESTIR_PT_SHIFT_WGSL = /* wgsl */ `
fn restirpt_safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  if (l < 1e-12) { return vec3f(0.0); }
  return v / l;
}

// G(x_a ↔ x_s) = |cos θ_s(a)| / ‖x_a − x_s‖² — the destination-cosine ("half-G")
// reconnection-edge geometry term (cosine taken at the SHARED reconnection vertex
// x_s, via its normal n_s). Mirrors @vitrum/shared-samplers reconnectionGeometry-
// Term EXACTLY. Returns 0 on a degenerate (coincident) edge or a tangent
// connection (cos θ_s = 0) so the caller treats the shift as non-invertible there.
fn restirPtReconnectionGeometryTerm(xa: vec3f, xs: vec3f, ns: vec3f) -> f32 {
  let d = xa - xs;
  let dist2 = dot(d, d);
  if (dist2 <= 0.0) { return 0.0; }
  let dist = sqrt(dist2);
  let cosOut = abs(dot(ns, d) / dist);
  return cosOut / dist2;
}

// The reconnection-shift Jacobian |∂T/∂·| = G(target edge) / G(source edge)
// (Lin 2022 Eq. 11/12). The SOURCE path (through x_q) is the DENOMINATOR (the
// density we re-map FROM); the TARGET (through x_r) is the NUMERATOR. Returns 0
// when the source half-G is 0 (nothing to re-map from — guards the divide) or
// when the target half-G is 0 (the offset edge is degenerate/tangent → the shift
// carries no weight there). Mirrors the CPU oracle reconnectionJacobian EXACTLY;
// UNCLAMPED (the hero-stack form — the GI jacobianReconnectionShift clamps to
// [0.1,10] for real-time temporal stability; the integrator clamps at the call
// site if it wants that, this returns the true ratio).
fn restirPtShiftJacobian(xq: vec3f, xr: vec3f, xs: vec3f, ns: vec3f) -> f32 {
  let gSource = restirPtReconnectionGeometryTerm(xq, xs, ns);
  if (gSource <= 0.0) { return 0.0; }
  let gTarget = restirPtReconnectionGeometryTerm(xr, xs, ns);
  return gTarget / gSource;
}

// |dω_a / dA_s| at a pre-reconnection vertex x_a, MEASURED off the shift geometry
// (not the closed form): as the reconnection vertex x_s sweeps its tangent plane
// (area params s = ts, t = tt), the direction ω_a = normalize(x_s − x_a) the path
// takes from x_a sweeps a solid angle. The solid-angle projection of ∂x_s/∂* is
//   ∂ω_a/∂* = (∂x_s/∂* − ω_a(ω_a·∂x_s/∂*)) / ‖x_s − x_a‖,
// and the basis-free 2×2 area-determinant on ω_a's tangent plane is
//   |dω_a/dA_s| = |∂ω_a/∂s × ∂ω_a/∂t|.
// (∂x_s/∂s = ts, ∂x_s/∂t = tt are unit & orthonormal, so dA_s = ds·dt.) This is
// exactly G(x_a↔x_s) by construction — computing it from the geometry is the
// FD-able measure that the harness diffs the analytic ratio against (the
// mnee-pdf-validate.ts pattern, applied to the reconnection edge). Returns 0 on a
// degenerate edge.
fn restirPtSolidAngleAreaDeriv(xa: vec3f, xs: vec3f, ts: vec3f, tt: vec3f) -> f32 {
  let d = xs - xa;
  let dist = length(d);
  if (dist < 1e-8) { return 0.0; }
  let w = d / dist;
  let dw_ds = (ts - w * dot(w, ts)) / dist;
  let dw_dt = (tt - w * dot(w, tt)) / dist;
  return length(cross(dw_ds, dw_dt));
}
`;

// NOTE: Harness shader (RESTIR_PT_SHIFT_HARNESS_WGSL) and TypeScript helpers
// (packRestirPtShiftInput, RESTIR_PT_SHIFT_INPUT_FLOATS) live in
// restirPtShift.harness.wgsl.ts — kept separate from this production module.
