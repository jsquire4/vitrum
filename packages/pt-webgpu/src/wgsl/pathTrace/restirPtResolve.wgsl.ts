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
 * reconnection contribution — evaluateBrdf·cos·Lo; B3). The producer's candidate weight was
 * w = p̂/p_src (p_src = the REAL visible-vertex BSDF sampling pdf, stored as
 * rCur.pdfSrc). So W = 1/p_src for the chosen sample, and the unbiased
 * single-bounce contribution is:
 *
 *   L = f_bsdf(xv; wo → wi_recon) · cos(nv, wi_recon) · Lo · W
 *     = f_bsdf · cos · Lo / p_src
 *
 * which is the standard MC estimator of the indirect radiance from the
 * BSDF-sampled reconnection direction. The diffuse-cosine proxy p̂ CANCELS (it
 * only chose which candidate survived resampling — variance, not mean). Resolve
 * therefore evaluates the FULL visible-vertex BRDF (evaluateBrdf with the cached
 * albV/roughnessV/metalV), making a GLOSSY visible vertex unbiased — the diffuse
 * proxy is NEVER used in the reconstruction. (See reservoirPtHero.wgsl.ts
 * unbiasedness note.)
 *
 * wo at xv is the camera direction: wo = normalize(cameraPos − xv). wi_recon is
 * the cached reconnection-edge direction. W bakes 1/p_src for the chosen sample.
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
  if (r.M == 0u || r.W <= 0.0) {
    rpt_result[pixelIdx] = vec4f(0.0, 0.0, 0.0, 0.0);
    return;
  }

  // Reconstruct the reconnection edge xv → xs.
  let toS = r.xs - r.xv;
  let dist2 = dot(toS, toS);
  if (dist2 < 1e-8) {
    rpt_result[pixelIdx] = vec4f(0.0, 0.0, 0.0, 0.0);
    return;
  }
  let wiRecon = toS * inverseSqrt(dist2);
  let cosTheta = max(0.0, dot(r.nv, wiRecon));
  if (cosTheta <= 1e-6) {
    rpt_result[pixelIdx] = vec4f(0.0, 0.0, 0.0, 0.0);
    return;
  }

  // wo at the visible vertex is the camera direction.
  let wo = safe_normalize(params.cameraPos.xyz - r.xv);

  // FULL visible-vertex BRDF (NOT the diffuse proxy target). With W = 1/p_src for
  // the chosen sample, f·cos·Lo·W = f·cos·Lo/p_src = the unbiased single-bounce
  // estimator (see the file header).
  let fBsdf = evaluateBrdfFull(
    r.albV, r.roughnessV, r.metalV, r.nv, wo, wiRecon,
    r.clearcoatV, r.clearcoatRoughnessV, r.sheenV, r.sheenRoughnessV, r.sheenColorV,
    r.iridescenceV, r.iridescenceIorV, r.iridescenceThicknessMinV, r.iridescenceThicknessMaxV,
    vec3f(1.0), 1.0,
    r.anisotropyV, r.anisotropyRotationV,
  );
  let indirect = fBsdf * cosTheta * r.Lo * r.W;

  rpt_result[pixelIdx] = vec4f(max(indirect, vec3f(0.0)), 1.0);
}
`;
