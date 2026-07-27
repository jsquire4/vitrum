/**
 * restirPtResolve.wgsl.ts — the ReSTIR-PT RESOLVE pass.
 *
 * A SEPARATE `@compute` entry point (`restirPtResolve`) — reconstructs the path
 * contribution from the resolved (temporally-reused) reservoir and writes it to a
 * result storage buffer. This is the hero-stack analogue of walkaround-hybrid's
 * GI shade step (`applyDDGIShading` / the shade.wgsl GI-reservoir read), which
 * injects `materialColor · Lo · cos · W` (M7 receiver: albedo·PI_INV applied once
 * at injection).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * The reconstruction (and WHY it uses the FULL BRDF, not the proxy target)
 * ════════════════════════════════════════════════════════════════════════════
 * The reservoir's W was finalised as W = w_sum/p̂ where p̂ is the
 * INTEGRAND-MATCHING target (restirPtTargetAt: luminance of the real unshadowed
 * reconnection contribution — evaluateBrdf·cos·Lo; B3). For an initial
 * one-candidate producer reservoir, w = p̂/p_src and this reduces to W=1/p_src.
 * After temporal/spatial reuse, however, w_sum contains generalized-balance
 * contributions from multiple source reservoirs, so W is NOT generally the
 * reciprocal of the selected sample's rCur.pdfSrc. The GRIS contribution is:
 *
 *   L = f_bsdf(xv; wo → wi) · cos(nv, wi) · Lo · W
 *
 * with W carrying the complete resampling normalization. pdfSrc remains source
 * proposal metadata for the selected shifted sample; resolve must not divide by
 * it again. In the one-candidate producer case the expression reduces to the
 * standard f_bsdf·cos·Lo/p_src Monte Carlo estimator.
 *
 * The scalar target only controls which candidate survives resampling (variance,
 * not mean). Resolve therefore evaluates the FULL visible-vertex BRDF
 * (evaluateBrdf with the cached
 * albV/roughnessV/metalV), making a GLOSSY visible vertex unbiased — the diffuse
 * proxy is NEVER used in the reconstruction. (See reservoirPtHero.wgsl.ts
 * unbiasedness note.)
 *
 * wo at xv is the native producer direction stored in r.woV; wi is recomputed
 * from xv → xs. W carries the finalized GRIS normalization.
 *
 * HONEST NOTE: this contribution is the INDIRECT (one-bounce-reconnection) term
 * ONLY. The DIRECT lighting at the visible vertex xv (NEE at xv) and the
 * camera-visible emission of xv are NOT part of the reservoir (the reservoir's Lo
 * is the suffix radiance leaving xs); the wiring/compositing step is responsible
 * for adding xv's own direct + emissive (e.g. from the megakernel's first-bounce
 * terms or a companion direct pass) to this indirect result. Writing ONLY the
 * reconnection indirect here keeps the reservoir contribution unambiguous.
 *
 * ── Result buffer layout (defined here; the wiring step binds it) ───────────
 *   rpt_result : array<vec4f>, one vec4 per full-res pixel (row-major,
 *                index = y·width + x). .rgb = the reconnection indirect radiance
 *                (linear HDR); .a = 1.0 on a contributing pixel, 0.0 on an empty
 *                reservoir (so the compositor can distinguish "no reuse" from
 *                "reuse produced black"). The wiring step composites .rgb into the
 *                beauty buffer (add to the direct/emissive at xv).
 *
 * ── Bind groups ─────────────────────────────────────────────────────────────
 * Composes the SHARED pt-webgpu modules (for evaluateBrdf); the ReSTIR-PT
 * reservoir + result + params live in @group(4):
 *   @binding(5) rpt_resResolved  (read)       — the resolved reservoir (the SPATIAL
 *                                                pass output; temporal→spatial→resolve)
 *   @binding(3) rpt_result       (read_write) — the reconnection-indirect output
 *   @binding(4) rptParams        (uniform)
 */

export const RESTIR_PT_RESOLVE_WGSL = /* wgsl */ `
@group(4) @binding(5) var<storage, read>       rpt_resResolved: array<u32>;
@group(4) @binding(3) var<storage, read_write> rpt_result:      array<vec4f>;
@group(4) @binding(4) var<uniform>             rptParams:       RestirPtParams;

@compute @workgroup_size(8, 8, 1)
fn restirPtResolve(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  let pixelIdx = gid.y * params.width + gid.x;
  let r = loadReservoirPTHero_ro(&rpt_resResolved, pixelIdx);

  // Empty reservoir → no reconnection indirect (alpha 0 so the compositor can
  // tell "no reuse" apart from "reuse is black").
  if (r.M == 0u || !rptFinitePositive(r.W)
   || !rptFiniteVec3(r.xv) || !rptFiniteVec3(r.xs)
   || !rptFiniteVec3(r.nv) || !rptFiniteVec3(r.woV)
   || !rptFiniteVec3(r.Lo)) {
    rpt_result[pixelIdx] = vec4f(0.0, 0.0, 0.0, 0.0);
    return;
  }

  // Reconstruct the reconnection edge xv → xs.
  let toS = r.xs - r.xv;
  let dist2 = dot(toS, toS);
  if (!rptFinitePositive(dist2) || dist2 < 1e-8) {
    rpt_result[pixelIdx] = vec4f(0.0, 0.0, 0.0, 0.0);
    return;
  }
  let wiRecon = toS * inverseSqrt(dist2);
  let cosTheta = max(0.0, dot(r.nv, wiRecon));
  if (!rptFinitePositive(cosTheta) || cosTheta <= 1e-6) {
    rpt_result[pixelIdx] = vec4f(0.0, 0.0, 0.0, 0.0);
    return;
  }

  // Native eye direction is part of the visible domain and survives camera motion.
  let wo = r.woV;

  // FULL visible-vertex BRDF. Multiply the vector integrand by the finalized
  // scalar GRIS weight exactly once; reused reservoirs must not add /pdfSrc.
  let fBsdf = evaluateBrdfFullWithClearcoatNormal(
    r.albV, r.roughnessV, r.metalV, r.nv, r.clearcoatNormalV, wo, wiRecon,
    r.clearcoatV, r.clearcoatRoughnessV, r.sheenV, r.sheenRoughnessV, r.sheenColorV,
    r.iridescenceV, r.iridescenceIorV, r.iridescenceThicknessMinV, r.iridescenceThicknessMaxV,
    r.specularColorV, r.specularIntensityV,
    r.anisotropyV, r.anisotropyRotationV,
  );
  let integrand = fBsdf * cosTheta * r.Lo;
  if (!rptFiniteVec3(fBsdf) || !rptFiniteVec3(integrand)) {
    rpt_result[pixelIdx] = vec4f(0.0, 0.0, 0.0, 0.0);
    return;
  }
  let indirect = integrand * r.W;
  if (!rptFiniteVec3(indirect)) {
    rpt_result[pixelIdx] = vec4f(0.0, 0.0, 0.0, 0.0);
    return;
  }

  // Lo and the visible-domain BRDF are monochromatic replicated scalars in
  // spectral mode.  Reconstruct this reservoir with THE RESERVOIR'S wavelength
  // before publication: temporal/spatial reuse may select a sample drawn at a
  // different lambda than the megakernel's current path.  Publishing the scalar
  // and converting it later with the current path's lambda changes the estimator.
  var indirectOut = max(indirect, vec3f(0.0));
  if (params.spectralEnabled != 0u) {
    let reservoirHeroPdf = heroMisMixturePdf(r.heroLambdaV);
    if (!rptFinitePositive(reservoirHeroPdf)) {
      rpt_result[pixelIdx] = vec4f(0.0, 0.0, 0.0, 0.0);
      return;
    }
    indirectOut = heroWavelengthToRgb(
      r.heroLambdaV, luminance(indirect), reservoirHeroPdf,
    );
    if (!rptFiniteVec3(indirectOut)) {
      rpt_result[pixelIdx] = vec4f(0.0, 0.0, 0.0, 0.0);
      return;
    }
  }

  rpt_result[pixelIdx] = vec4f(indirectOut, 1.0);
}
`;
