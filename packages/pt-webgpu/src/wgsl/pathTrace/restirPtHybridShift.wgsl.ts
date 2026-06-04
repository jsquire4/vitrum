/**
 * restirPtHybridShift.wgsl.ts — the ReSTIR-PT / GRIS HYBRID-SHIFT Jacobian for the
 * GGX/Cook-Torrance BSDF the pt-webgpu kernel uses. This is the follow-up to the
 * pure-geometry RECONNECTION-shift Jacobian in `restirPtShift.wgsl.ts`: it adds the
 * BSDF-sampling-pdf ratio of the RANDOM-REPLAYED prefix segment(s) that the hybrid
 * shift introduces ahead of the reconnection edge.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * What the HYBRID shift is (and why the reconnection shift alone is insufficient)
 * ════════════════════════════════════════════════════════════════════════════
 * The reconnection shift (`restirPtShift.wgsl.ts`) reconnects to the suffix at the
 * FIRST vertex that is "smooth enough / far enough" — call it x_s. But the vertices
 * BEFORE x_s may be too specular/peaked to reconnect to deterministically (a fresh
 * deterministic edge into a near-mirror vertex would land off the BSDF lobe and
 * contribute ~0, wrecking the shift's invertibility / efficiency). The HYBRID shift
 * (Lin et al. 2022 §5.2; the random-replay shift of Kettunen et al. 2015) handles
 * that prefix by RANDOM REPLAY:
 *   • it re-uses the path's CANONICAL random numbers u_1..u_k and re-runs the
 *     engine's OWN direction sampler (sampleNextBounceDirection / brdfDirectional-
 *     Pdf's lobe partition) from the OFFSET domain's vertices, regenerating the
 *     prefix directions ω_1^r..ω_k^r (deterministic given the shared u's), THEN
 *   • RECONNECTS at the first sufficiently-smooth vertex x_s by the same fresh
 *     deterministic edge the reconnection shift uses.
 *
 *   T_hybrid = (random-replay of the rough prefix)  ∘  (reconnection at x_s)
 *
 * The suffix from x_s onward is held FIXED in world space (invariant, no Jacobian);
 * the rough prefix is REPLAYED (a BSDF-pdf change of variables); the reconnection
 * edge is SWAPPED (the geometric half-G change of variables).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * The hybrid Jacobian = (reconnection geometry) × (replayed-prefix BSDF-pdf ratio)
 * ════════════════════════════════════════════════════════════════════════════
 * The total change of variables |∂T_hybrid/∂·| = |∂(target path coords)/∂(source
 * path coords)| factors over the path's free variables, which the hybrid shift
 * parameterizes in TWO commensurable groups:
 *
 *   (A) The reconnection EDGE, shared resampling coordinate = the AREA of x_s.
 *       Its factor is the destination-cosine half-G RATIO (target / source),
 *       IDENTICAL to `restirPtShiftJacobian` (Lin 2022 Eq. 11/12):
 *
 *         J_geom = G(x_r ↔ x_s) / G(x_q ↔ x_s),
 *                  G(x_a ↔ x_s) = |cos θ_s(a)| / ‖x_a − x_s‖².
 *
 *       Here x_q / x_r are the LAST pre-reconnection vertices of the source /
 *       target prefixes — i.e. the END of the replayed prefix, NOT the camera
 *       vertex (for a 1-segment replay they ARE the camera-adjacent vertex; for a
 *       k-segment replay they are the k-th replayed vertex).
 *
 *   (B) Each RANDOM-REPLAYED prefix bounce j, shared resampling coordinate = the
 *       CANONICAL random numbers u_j. Both domains push the SAME u_j through the
 *       SAME sampler, so the per-bounce change of variables ω_j^q → ω_j^r is
 *
 *         dω_j^r/dω_j^q = (dω_j^r/du_j)·(du_j/dω_j^q) = (1/p_j^r)·p_j^q
 *                       = p_BSDF(x_q^{(j)}; wo_j^q → wi_j^q)
 *                       / p_BSDF(x_r^{(j)}; wo_j^r → wi_j^r).
 *
 *       The SOURCE forward-sampling pdf is the NUMERATOR, the TARGET pdf the
 *       DENOMINATOR — the OPPOSITE orientation to the geometric term, because the
 *       sampler maps u→ω with Jacobian 1/p (so du/dω = p contributes the source
 *       pdf as a NUMERATOR, dω/du = 1/p contributes the target pdf as a
 *       DENOMINATOR). This is exactly the gradient-domain random-replay shift
 *       Jacobian (Kettunen 2015), generalized by Lin 2022.
 *
 *   |∂T_hybrid/∂·| = J_geom · ∏_{j ∈ replayed prefix}  p_j^q / p_j^r.            (★)
 *
 * For the canonical SINGLE replayed segment (one rough bounce ahead of x_s) this is
 *
 *   |∂T_hybrid/∂·| = [G(x_r↔x_s)/G(x_q↔x_s)] · [p_BSDF^q / p_BSDF^r].
 *
 * ════════════════════════════════════════════════════════════════════════════
 * WHY p_BSDF here is exactly `brdfDirectionalPdf` (the engine's forward pdf)
 * ════════════════════════════════════════════════════════════════════════════
 * The replay re-runs the engine's own sampler, so the per-segment density p_j is
 * the sampler's OWN solid-angle pdf — which is precisely `brdfDirectionalPdf`
 * (bsdf.wgsl.ts:40). For the GGX specular lobe the sampler is Heitz-2018 VNDF
 * (`glossyReflectionSample`), whose construction GUARANTEES the pushforward of
 * uniform [0,1)² has solid-angle density
 *
 *     p_VNDF(wi | wo) = D(h)·G1(wo) / (4·(N·wo)),   h = normalize(wi+wo),
 *
 * i.e. the sampler's |∂wi/∂(u1,u2)| = 1/p_VNDF(wi|wo) BY CONSTRUCTION. That is
 * `pdfSpec` in `brdfDirectionalPdf` (bsdf.wgsl.ts:90-91). Replaying the SAME
 * (u1,u2) at the offset vertex x_r (its own wo_r, roughness_r) yields
 * |∂wi_r/∂(u1,u2)| = 1/p_VNDF(wi_r|wo_r), so the composite replay Jacobian for that
 * bounce is p_VNDF(wi_q|wo_q)/p_VNDF(wi_r|wo_r). The FULL three-lobe pdf
 * (diffProb·(N·wi)/π  +  specProb·pdfSpec  +  the opposite-hemisphere refraction
 * lobe) is the correct p_j WHENEVER the replay re-runs the SAME stochastic lobe
 * partition with the shared random stream — which is the standard hybrid-shift
 * replay assumption (re-use canonical randoms through the identical routine). So
 * this module's `restirPtBsdfReplayPdf` mirrors `brdfDirectionalPdf` EXACTLY (same
 * three-lobe MIS mixture, same VNDF specular branch, same η² refraction branch).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Reciprocity, degeneracy, model-dependence
 * ════════════════════════════════════════════════════════════════════════════
 * • Reciprocity. T_hybrid⁻¹ swaps the source/target roles in BOTH factors: J_geom
 *   inverts (target↔source) AND every p_j^q/p_j^r inverts, so
 *   |∂T_hybrid/∂·|·|∂T_hybrid⁻¹/∂·| = 1 (pinned by the unit test).
 * • Degeneracy. If the geometric term is 0 (coincident / tangent reconnection
 *   edge) OR any source replay pdf p_j^q is 0 (the source could not have sampled
 *   that prefix direction — nothing to re-map from) the hybrid Jacobian is 0 (the
 *   shift carries no weight / is non-invertible there). A 0 TARGET replay pdf
 *   p_j^r ALSO forces 0: the offset domain assigns zero density to the replayed
 *   direction, so the shifted path is unreachable.
 * • Model-dependence (proven-vs-assumed, stated honestly):
 *     – The GGX/Cook-Torrance case is derived RIGOROUSLY: the VNDF specular pdf is
 *       the exact pushforward of `glossyReflectionSample`, and the diffuse cosine
 *       lobe is the exact pushforward of `cosineHemisphereSample`. The replay
 *       ratio p_q/p_r for those lobes is GPU-validated (analytic == FD of the
 *       actual replay map) in restir-pt-hybrid-shift-validate.ts.
 *     – smithG1 here is the Schlick-GGX/UE4 approximation k=(roughness+1)²/8 (NOT
 *       the exact Smith Λ), because that is what the engine's sampler+pdf use; the
 *       Jacobian must match the sampler that GENERATED the path, so using the
 *       engine's own approximate G1 is CORRECT for this engine (the Jacobian is a
 *       property of the sampler, not of an idealized BSDF). A different engine with
 *       a different sampler would need its own p_j.
 *     – The lobe-MIXING (the discrete diffuse/spec/trans partition) is assumed to
 *       be replayed identically across domains (same random stream → same lobe
 *       choice). If a future replay re-rolled the lobe choice from a re-evaluated
 *       partition the per-bounce factor would gain the partition-probability ratio;
 *       that is out of scope here (and is the standard hybrid-shift assumption).
 *
 * Refs:
 *   - Lin, Kettunen, Bitterli, Pantaleoni, Jakob, Nowrouzezahrai 2022,
 *     "Generalized Resampled Importance Sampling: Foundations of ReSTIR,"
 *     ACM TOG 41(4) / SIGGRAPH 2022, §5.2 (hybrid shift = random replay +
 *     reconnection).
 *   - Kettunen, Manzi, Aittala, Lehtinen, Durand, Zwicker 2015, "Gradient-Domain
 *     Path Tracing," ACM TOG 34(4) — the random-replay shift Jacobian = the
 *     per-segment BSDF-pdf ratio.
 *   - Heitz 2018, "Sampling the GGX Distribution of Visible Normals," JCGT 7(4) —
 *     the VNDF sampler whose pushforward density IS the replay pdf used here.
 */

/* NOTE: this module is intentionally self-contained — its WGSL inlines the tiny
 * set of microfacet primitives it needs (so the GPU validator compiles ONE
 * standalone translation unit, exactly like restirPtShift.wgsl.ts) and re-states
 * the reconnection half-G ratio BY VALUE (it equals restirPtShift.wgsl.ts's
 * restirPtShiftJacobian) rather than importing that module's WGSL string. This
 * keeps the composition explicit and avoids editing restirPtShift.wgsl.ts. There
 * are deliberately no cross-module imports. */

/** Floats per harness config record. A single-replayed-segment hybrid-shift config
 *  needs, for BOTH the source (q) and the offset (r) domain, the replayed bounce's
 *  geometry (vertex, shading normal, wo, wi) + material (baseColor, roughness,
 *  metallic, transmission, ior), PLUS the shared reconnection vertex x_s.
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
 *   [10] metallic, transmission, ior, _   (q-domain material scalars)
 *   [11] baseColorR.rgb, roughR           (r-domain material; roughR in .w)
 *  (r-domain metallic/trans/ior reuse the q scalars in this harness — same surface
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
  ior: number;
}): number[] {
  const { xq, nq, woq, wiq, xr, nr, wor, wir, xs, baseColor, roughness, metallic, transmission, ior } = cfg;
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
    metallic, transmission, ior, 0,
    baseColor[0], baseColor[1], baseColor[2], roughness,
  ];
}

/**
 * Core hybrid-shift WGSL. Self-contained: inlines the few microfacet primitives it
 * needs (ggxD / smithG1 / fresnelSchlick / luminance) so the harness is a single
 * translation unit AND so the BSDF-replay pdf here is BYTE-FOR-BYTE the same closed
 * form as bsdf.wgsl.ts:brdfDirectionalPdf (the engine's forward sampling pdf). The
 * geometric reconnection half-G ratio is re-stated by value (NOT imported) to keep
 * the composition explicit; it equals restirPtShift.wgsl.ts's restirPtShiftJacobian.
 */
export const RESTIR_PT_HYBRID_SHIFT_WGSL = /* wgsl */ `
const RPT_PI = 3.14159265358979;
const RPT_INV_PI = 0.31830988618;

fn rptHybrid_safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  if (l < 1e-12) { return vec3f(0.0); }
  return v / l;
}

fn rptHybrid_luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Schlick GGX / UE4 G1 (the engine's smithG1: k = (roughness+1)²/8). The Jacobian
// MUST use the SAME G1 the engine's sampler+pdf use — the replay pdf is a property
// of the sampler that generated the path, not of an idealized Smith Λ.
fn rptHybrid_smithG1(nDotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) * 0.125;
  return nDotV / max(nDotV * (1.0 - k) + k, 1e-6);
}

// Trowbridge-Reitz / GGX NDF (identical to bsdf.wgsl.ts ggxD).
fn rptHybrid_ggxD(nDotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(RPT_PI * d * d, 1e-6);
}

fn rptHybrid_fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  let m5 = m2 * m2 * m;
  return f0 + (vec3f(1.0) - f0) * m5;
}

// ── (A) reconnection-edge geometry: the destination-cosine half-G + its ratio ──
// IDENTICAL to restirPtShift.wgsl.ts (re-stated by value; we do not edit that
// module). G(x_a ↔ x_s) = |cos θ_s(a)| / ‖x_a − x_s‖², cosine at the SHARED x_s.
fn rptHybridReconnectionGeometryTerm(xa: vec3f, xs: vec3f, ns: vec3f) -> f32 {
  let d = xa - xs;
  let dist2 = dot(d, d);
  if (dist2 <= 0.0) { return 0.0; }
  let dist = sqrt(dist2);
  let cosOut = abs(dot(ns, d) / dist);
  return cosOut / dist2;
}

// J_geom = G(target edge x_r↔x_s) / G(source edge x_q↔x_s) (Lin 2022 Eq. 11/12).
// Returns 0 on a degenerate source edge (nothing to remap from) or a degenerate
// target edge. Equals restirPtShift.wgsl.ts::restirPtShiftJacobian.
fn rptHybridGeomJacobian(xq: vec3f, xr: vec3f, xs: vec3f, ns: vec3f) -> f32 {
  let gSource = rptHybridReconnectionGeometryTerm(xq, xs, ns);
  if (gSource <= 0.0) { return 0.0; }
  let gTarget = rptHybridReconnectionGeometryTerm(xr, xs, ns);
  return gTarget / gSource;
}

// ── (B) replayed-segment BSDF sampling pdf — BYTE-IDENTICAL to brdfDirectionalPdf ─
// The engine's forward solid-angle sampling density at one replayed bounce: the
// three-lobe MIS mixture (diffuse cosine + VNDF specular + η² refraction) with the
// SAME lobe-selection probabilities the sampler partitions on. This is the p_j the
// random-replay change of variables uses. (Mirrors bsdf.wgsl.ts:40-94 EXACTLY,
// with the local _-prefixed primitives.)
fn rptHybridBsdfReplayPdf(
  baseColor: vec3f, roughness: f32, metallic: f32, transmission: f32, ior: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
) -> f32 {
  let wiDotN = dot(normal, wi);
  let woDotN = dot(normal, wo);
  let nDotV = max(woDotN, 0.0);
  if (nDotV <= 1e-5) { return 0.0; }
  let h = rptHybrid_safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 1e-6);
  let f0 = mix(vec3f(0.04), baseColor, metallic);
  let fresnel = rptHybrid_fresnelSchlick(vDotH, f0);
  let baseSpecProb = clamp(mix(0.04, 0.96, max(rptHybrid_luminance(fresnel), metallic)), 0.04, 0.96);
  let baseTransProb = clamp(transmission * (1.0 - metallic), 0.0, 0.95);
  let baseDiffProb = max(0.0, (1.0 - metallic) * (1.0 - transmission));
  let sumProb = max(baseSpecProb + baseTransProb + baseDiffProb, 1e-4);
  let specProb = baseSpecProb / sumProb;
  let transProb = baseTransProb / sumProb;
  let diffProb = baseDiffProb / sumProb;
  let sameHemisphere = wiDotN * woDotN > 0.0;
  if (!sameHemisphere) {
    // Refraction lobe (opposite hemispheres): cosine on transmitted side · η²
    // (PBRT Dielectric / Walter 2007 hemisphere Jacobian).
    let eta = select(ior, 1.0 / max(ior, 1.0), woDotN > 0.0);
    let nDotT = max(abs(wiDotN), 1e-5);
    let pdfTransApprox = nDotT * eta * eta * RPT_INV_PI;
    return max(transProb * pdfTransApprox, 1e-8);
  }
  let nDotL = max(wiDotN, 0.0);
  if (nDotL <= 1e-5) { return 0.0; }
  let alpha = max(roughness * roughness, 1e-3);
  let d = rptHybrid_ggxD(nDotH, alpha);
  // VNDF reflection pdf (Heitz 2018 Eq. 17 with the 1/(4·wo·h) reflection Jacobian):
  //   p_VNDF(wi|wo) = D(h)·G1(wo) / (4·N·wo) — matches glossyReflectionSample.
  let g1Wo = rptHybrid_smithG1(nDotV, roughness);
  let pdfSpec = (d * g1Wo) / max(4.0 * nDotV, 1e-6);
  let pdfDiff = nDotL * RPT_INV_PI;
  return diffProb * pdfDiff + specProb * pdfSpec;
}

// Replay factor for ONE replayed prefix segment: p_BSDF^q / p_BSDF^r (SOURCE pdf in
// the NUMERATOR — the inverse-sampler Jacobian du/dω_q = p_q; TARGET pdf in the
// denominator — the forward-sampler Jacobian dω_r/du = 1/p_r). Returns 0 if the
// SOURCE pdf is 0 (unreachable source prefix — nothing to remap) or the TARGET pdf
// is 0 (the offset domain cannot reach the replayed direction → shifted path
// unreachable). This is the gradient-domain random-replay shift Jacobian (Kettunen
// 2015) for one bounce.
fn rptHybridReplaySegmentJacobian(
  baseColorQ: vec3f, roughnessQ: f32, metallicQ: f32, transmissionQ: f32, iorQ: f32,
  nq: vec3f, woq: vec3f, wiq: vec3f,
  baseColorR: vec3f, roughnessR: f32, metallicR: f32, transmissionR: f32, iorR: f32,
  nr: vec3f, wor: vec3f, wir: vec3f,
) -> f32 {
  let pq = rptHybridBsdfReplayPdf(baseColorQ, roughnessQ, metallicQ, transmissionQ, iorQ, nq, woq, wiq);
  if (pq <= 0.0) { return 0.0; }
  let pr = rptHybridBsdfReplayPdf(baseColorR, roughnessR, metallicR, transmissionR, iorR, nr, wor, wir);
  if (pr <= 0.0) { return 0.0; }
  return pq / pr;
}

// ── (★) the full SINGLE-segment hybrid-shift Jacobian = J_geom · (p_q/p_r) ──────
// |∂T_hybrid/∂·| for a hybrid shift whose rough prefix is ONE replayed bounce ahead
// of the reconnection vertex x_s. For a k-segment prefix, multiply additional
// rptHybridReplaySegmentJacobian factors (the geometric J_geom is unchanged — it
// depends only on the LAST pre-reconnection edge x_{q,r}↔x_s). Returns 0 if either
// factor is 0 (degenerate edge OR unreachable replayed prefix in either domain).
fn rptHybridShiftJacobian(
  // reconnection edge: last pre-reconnection vertices + shared x_s/n_s
  xq: vec3f, xr: vec3f, xs: vec3f, ns: vec3f,
  // the single replayed bounce in each domain (at xq / xr)
  baseColorQ: vec3f, roughnessQ: f32, metallicQ: f32, transmissionQ: f32, iorQ: f32,
  nq: vec3f, woq: vec3f, wiq: vec3f,
  baseColorR: vec3f, roughnessR: f32, metallicR: f32, transmissionR: f32, iorR: f32,
  nr: vec3f, wor: vec3f, wir: vec3f,
) -> f32 {
  let jGeom = rptHybridGeomJacobian(xq, xr, xs, ns);
  if (jGeom <= 0.0) { return 0.0; }
  let jReplay = rptHybridReplaySegmentJacobian(
    baseColorQ, roughnessQ, metallicQ, transmissionQ, iorQ, nq, woq, wiq,
    baseColorR, roughnessR, metallicR, transmissionR, iorR, nr, wor, wir);
  if (jReplay <= 0.0) { return 0.0; }
  return jGeom * jReplay;
}
`;

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
  matQ2: vec4f,  // metallic, transmission, ior, _
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
  let iorQ = c.matQ2.z;
  let baseColorR = c.matR.xyz;
  let roughnessR = c.matR.w;
  // r-domain metallic/trans/ior reuse the q scalars (same surface under the shift).

  let gSource = rptHybridReconnectionGeometryTerm(c.xq, c.xs, ns);
  let gTarget = rptHybridReconnectionGeometryTerm(c.xr, c.xs, ns);
  let jGeom   = rptHybridGeomJacobian(c.xq, c.xr, c.xs, ns);
  let pq = rptHybridBsdfReplayPdf(baseColorQ, roughnessQ, metallicQ, transmissionQ, iorQ, c.nq, c.woq, c.wiq);
  let pr = rptHybridBsdfReplayPdf(baseColorR, roughnessR, metallicQ, transmissionQ, iorQ, c.nr, c.wor, c.wir);
  let jReplay = rptHybridReplaySegmentJacobian(
    baseColorQ, roughnessQ, metallicQ, transmissionQ, iorQ, c.nq, c.woq, c.wiq,
    baseColorR, roughnessR, metallicQ, transmissionQ, iorQ, c.nr, c.wor, c.wir);
  let jHybrid = rptHybridShiftJacobian(
    c.xq, c.xr, c.xs, ns,
    baseColorQ, roughnessQ, metallicQ, transmissionQ, iorQ, c.nq, c.woq, c.wiq,
    baseColorR, roughnessR, metallicQ, transmissionQ, iorQ, c.nr, c.wor, c.wir);
  let saSource = rptHybridSolidAngleAreaDeriv(c.xq, c.xs, ts, tt);
  let saTarget = rptHybridSolidAngleAreaDeriv(c.xr, c.xs, ts, tt);

  hOut[i * 3u + 0u] = vec4f(jHybrid, jGeom, jReplay, 0.0);
  hOut[i * 3u + 1u] = vec4f(pq, pr, gSource, gTarget);
  hOut[i * 3u + 2u] = vec4f(saSource, saTarget, 0.0, 0.0);
}
`;
