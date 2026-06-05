/**
 * reservoirPtHero.wgsl.ts — the FULL-RES (hero-stack) ReSTIR-PT / GRIS path
 * reservoir ADT for `@vitrum/pt-webgpu`, plus the hero target function and the
 * reconnection-shift module it consumes.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * What this is (and what it is a port OF)
 * ════════════════════════════════════════════════════════════════════════════
 * This is the hero-stack generalization of the SHIPPING walkaround-hybrid
 * GRIS-GI reservoir (`@vitrum/walkaround-hybrid/src/shaders/reservoirGi.wgsl.ts`
 * — `struct ReservoirPT` + `RESERVOIR_GI_STRIDE` + the strided bitcast
 * load/store helpers + `updateReservoirGI` (streaming RIS) +
 * `finaliseGIReservoirWGris` (GRIS W = w_sum/p̂, NO /M)). The struct field set,
 * the bitcast-into-`array<u32>` serialization, the streaming-RIS update, and the
 * two finalize forms are MIRRORED from that proven module. This file widens the
 * struct for the hero stack (full-res, arbitrary visible-vertex material) and
 * single-homes the hero target `p̂` so the producer / temporal / resolve passes
 * all read the SAME definition.
 *
 * Ref: Lin, Kettunen, Bitterli, Pantaleoni, Jakob, Nowrouzezahrai — "Generalized
 * Resampled Importance Sampling: Foundations of ReSTIR", SIGGRAPH 2022 (GRIS);
 * Bitterli et al. 2020/2021 (ReSTIR DI/GI base reservoir + reconnection).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Increment scope (temporal-only, reconnection-shift-only, prefix-length-1)
 * ════════════════════════════════════════════════════════════════════════════
 * The reservoir stores a SINGLE-bounce reconnection sample:
 *   • xv / nv  — the VISIBLE vertex (the camera's first surface hit) and its
 *                shading normal. This is the path PREFIX (prefix length 1: the
 *                only pre-reconnection vertex is the primary hit).
 *   • xs / ns  — the RECONNECTION vertex: the surface hit by the ray bounced off
 *                xv, held FIXED in world space by the reconnection shift.
 *   • Lo       — the outgoing radiance LEAVING xs back toward xv (everything the
 *                suffix path gathers from xs onward; see the producer for the
 *                exact, energy-critical definition).
 *   • wi_recon — the unit reconnection-edge direction xv → xs along the BASE
 *                path (cached so the shift / resolve need not recompute it).
 *   • pdfSrc   — the ACTUAL source directional pdf that generated wi_recon at xv
 *                (the visible-vertex BSDF sampling pdf). UNBIASEDNESS-CRITICAL —
 *                see the unbiasedness note below; this is the hero-stack field
 *                the diffuse-only GI version does not need (it cosine-samples).
 *
 * The hero target p̂ (the resampling heuristic) is the INTEGRAND-MATCHING target —
 * the luminance of the real unshadowed reconnection contribution:
 *   p̂(z) = luminance( f_bsdf(xv; wo→wi) · max(0, cos(nv, wi)) · Lo ),  wi = xv→xs
 * (see `restirPtTargetAt`, using the visible-vertex BRDF — B3). It is a scalar
 * resampling heuristic; the W finalize divides it OUT (W = w_sum/p̂), so the
 * converged mean does NOT depend on it — only the resampling VARIANCE does, which
 * is exactly why matching the integrand (vs the old diffuse-cosine proxy) reduces
 * variance decisively for a glossy xv. The RESOLVE pass reconstructs the path
 * contribution with the SAME real BRDF (`evaluateBrdf`). (See the unbiasedness note.)
 *
 * ── Unbiasedness note (the load-bearing energy-consistency argument) ─────────
 * Let the producer sample wi_recon from the visible-vertex BSDF with the true
 * directional pdf p_src, hit xs, and cache Lo. The 1-sample RIS candidate weight
 * is w = p̂ / p_src. After finalize, W = w_sum / p̂ = (p̂/p_src) / p̂ = 1/p_src.
 * The RESOLVE pass forms  f_bsdf(xv; wo→wi_recon) · cos(nv,wi_recon) · Lo · W
 *                       = f_bsdf · cos · Lo / p_src,
 * which is EXACTLY the unbiased single-bounce MC estimator of the indirect
 * radiance from the cosine/BSDF-sampled direction. The diffuse-cosine proxy p̂
 * CANCELS (it set which candidate survived resampling — variance only). This
 * holds for a DIFFUSE *and* a GLOSSY visible vertex, PROVIDED:
 *   (1) the producer's candidate denominator is the REAL p_src (the
 *       visible-vertex BSDF directional pdf), not the cosine proxy, AND
 *   (2) the resolve uses the REAL evaluateBrdf at xv.
 * Both hold in this increment. The diffuse GI version gets away with the cosine
 * proxy as p_src because its visible vertex is ALWAYS Lambertian (cosine sampling
 * ⇒ p_src = cos·INV_PI ⇒ w = luminance(Lo)); the hero stack stores p_src so the
 * SAME cancellation works for non-Lambertian xv.
 *
 * The HONEST residual bias is the REUSE (the shift), not the producer:
 *   • A NEAR-SPECULAR / transmissive visible vertex makes the prefix BSDF nearly
 *     singular, so reusing a neighbour's Lo through a DIFFERENT wi_recon is
 *     invalid (the reconnection shift assumes the reconnection vertex is reached
 *     by a non-singular prefix BSDF). The PRODUCER therefore writes an EMPTY
 *     reservoir (M = 0) for specular/transmissive xv — that pixel does not reuse.
 *   • A MODERATELY-GLOSSY xv: the geometric reconnection shift holds xs fixed and
 *     re-roots the edge; the glossy BRDF at xv is direction-sensitive. The
 *     INTEGRAND-MATCHING target (B3) now weights such candidates by the REAL BRDF
 *     in the temporal MIS (was a diffuse-cosine proxy that under-weighted them), so
 *     the prefix-1 glossy reuse is resampled correctly — unbiased by construction
 *     (W cancels p̂) and variance-reduced. (A full random-replay shift for
 *     multi-vertex glossy PREFIXES — which prefix-1 lacks — remains a later item.)
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Phase-0 written-but-unread fields (the hybrid-shift + rngSeed headroom)
 * ════════════════════════════════════════════════════════════════════════════
 * The hybrid shift (a BSDF-pdf-ratio replayed-prefix shift — `restirPtHybridShift
 * .wgsl.ts`) and a per-reservoir rng seed are NOT consumed in this increment.
 * Per the Phase-0 discipline of the GI reservoir's GRIS cache, the struct
 * RESERVES + ZERO-INITIALISES + SERIALISES them now (so a later hybrid-shift /
 * random-replay follow-up needs no stride bump and no reorder of the live
 * fields), but NO pass in this increment READS them:
 *   • hybridJacCache / hybridShiftPdf — the replayed-prefix shift's cached
 *     Jacobian + reverse pdf (filled by the future hybrid shift).
 *   • rngSeed — the per-reservoir decorrelated seed a future random-replay shift
 *     reuses to re-trace the offset prefix deterministically.
 */

import { RESTIR_PT_SHIFT_WGSL } from './restirPtShift.wgsl.js';

/**
 * ReSTIR-PT runtime params UBO (the reuse unit's OWN tunables) + the reservoir
 * bind-group layout convention.
 *
 * The shared `FrameParams` (material.wgsl.ts @group(0) @binding(1)) is OWNED by
 * the megakernel and MUST NOT grow a ReSTIR-PT field for this increment (a
 * parallel agent owns that file). So the reuse passes declare their OWN small
 * uniform — `RestirPtParams` — in the ReSTIR-PT bind group (@group(4)). It
 * carries the GRIS W-cap (the temporal-feedback gain bound — the GI version's
 * `restirGiWCap`) and the temporal M-clamp (the GI `restirGiMClamp`). The WIRING
 * step allocates + writes this buffer.
 *
 * @group(4) layout (the ReSTIR-PT-specific resources, separate from the
 * inherited @group(0..3) the shared modules own):
 *   @binding(0) rpt_reservoirOut  (storage, read_write) — producer output
 *   @binding(1) rpt_reservoirCur  (storage, read_write) — temporal in/out
 *   @binding(2) rpt_reservoirPrev (storage, read)        — last frame's output
 *   @binding(3) rpt_result        (storage, read_write) — resolve output
 *   @binding(4) rptParams         (uniform)              — RestirPtParams
 * Each pass declares only the subset of @group(4) it uses (WGSL permits sparse
 * bindings); the wiring step builds the matching per-pass layout.
 */
export const RESTIR_PT_PARAMS_WGSL = /* wgsl */ `
struct RestirPtParams {
  width:    u32,   // full-res width  (mirrors params.width; lets a reuse pass
  height:   u32,   // full-res height  run without re-reading FrameParams dims)
  mClamp:   u32,   // temporal M-clamp (GI restirGiMClamp analogue)
  _padA:    u32,
  wCap:     f32,   // GRIS W-cap (temporal-feedback gain bound)
  _padB:    f32,
  _padC:    f32,
  _padD:    f32,
};
`;

export const RESERVOIR_PT_HERO_WGSL = /* wgsl */ `// ============================================================
// ReSTIR-PT / GRIS hero reservoir (ReservoirPTHero, 144 bytes = 36 × u32).
// Full-res, arbitrary visible-vertex material. Mirrors the walkaround-hybrid
// ReservoirPT field set + bitcast serialization, widened for the hero stack:
//   - pdfSrc (the REAL visible-vertex BSDF sampling pdf) for unbiased glossy
//     reconstruction (the diffuse GI version cosine-samples and does not need it),
//   - the visible-vertex material (roughness/metallic/albedo) so the RESOLVE pass
//     can evaluate the FULL BRDF (not a diffuse proxy),
//   - Phase-0 hybrid-shift + rngSeed headroom (written-but-UNREAD this increment).
// ============================================================
struct ReservoirPTHero {
  // ── reconnection sample (prefix length 1) ──
  xv:      vec3f,   // visible vertex (primary hit / path prefix)   idx 0..2
  _pad0:   f32,     //                                              idx 3
  nv:      vec3f,   // shading normal at xv                         idx 4..6
  W:       f32,     // RIS unbiased contribution weight (UCW)       idx 7
  xs:      vec3f,   // reconnection vertex (held fixed by shift)    idx 8..10
  w_sum:   f32,     // running RIS weight sum                       idx 11
  ns:      vec3f,   // shading normal at xs                         idx 12..14
  M:       u32,     // confidence (candidate count)                idx 15
  Lo:      vec3f,   // outgoing radiance LEAVING xs toward xv       idx 16..18
  pdfSrc:  f32,     // REAL source BSDF directional pdf at xv       idx 19
  // ── reconnection-shift cache (consumed by the temporal shift + resolve) ──
  wi_recon:          vec3f, // unit reconnection dir xv→xs (base)   idx 20..22
  distRecon:         f32,   // ‖xv − xs‖ along the base edge        idx 23
  cosReconOut:       f32,   // |cos θ_out| at xs (ns · −wi_recon)   idx 24
  prefixVertexCount: u32,   // path-prefix vertex count (1 here)    idx 25
  roughnessV:        f32,   // visible-vertex roughness (resolve)   idx 26
  metalV:            f32,   // visible-vertex metallic (resolve)    idx 27
  // ── visible-vertex albedo (resolve evaluates the FULL BRDF) ──
  albV:              vec3f, // visible-vertex baseColor             idx 28..30
  _pad1:             f32,   //                                      idx 31
  // ── Phase-0 hybrid-shift + rngSeed headroom (WRITTEN-but-UNREAD here) ──
  hybridJacCache:    f32,   // future replayed-prefix shift Jacobian idx 32
  hybridShiftPdf:    f32,   // future replayed-prefix reverse pdf    idx 33
  rngSeed:           u32,   // future random-replay decorrelated seed idx 34
  _padHybrid:        u32,   //                                      idx 35
};

// ReservoirPTHero byte layout (144 bytes = 36 × u32):
//   [0..2]   xv.xyz        [3]    _pad0
//   [4..6]   nv.xyz        [7]    W
//   [8..10]  xs.xyz        [11]   w_sum
//   [12..14] ns.xyz        [15]   M
//   [16..18] Lo.xyz        [19]   pdfSrc
//   [20..22] wi_recon.xyz  [23]   distRecon
//   [24]     cosReconOut   [25]   prefixVertexCount
//   [26]     roughnessV    [27]   metalV
//   [28..30] albV.xyz      [31]   _pad1
//   [32]     hybridJacCache  (Phase-0, unread)
//   [33]     hybridShiftPdf  (Phase-0, unread)
//   [34]     rngSeed         (Phase-0, unread)
//   [35]     _padHybrid
// Strided storage in array<u32> (4-byte elements) — stride = 36 u32.
const RESERVOIR_PT_HERO_STRIDE: u32 = 36u;

fn emptyReservoirPTHero() -> ReservoirPTHero {
  var r: ReservoirPTHero;
  r.xv = vec3f(0.0); r.nv = vec3f(0,1,0);
  r.xs = vec3f(0.0); r.ns = vec3f(0,1,0);
  r.Lo = vec3f(0.0); r.W = 0.0; r.w_sum = 0.0; r.M = 0u;
  r.pdfSrc = 0.0; r._pad0 = 0.0;
  r.wi_recon = vec3f(0.0);
  r.distRecon = 0.0;
  r.cosReconOut = 0.0;
  r.prefixVertexCount = 0u;
  r.roughnessV = 0.0;
  r.metalV = 0.0;
  r.albV = vec3f(0.0);
  r._pad1 = 0.0;
  // Phase-0 headroom — zero-initialised, READ BY NO PASS in this increment.
  r.hybridJacCache = 0.0;
  r.hybridShiftPdf = 0.0;
  r.rngSeed = 0u;
  r._padHybrid = 0u;
  return r;
}

fn loadReservoirPTHero_ro(buf: ptr<storage, array<u32>, read>, pixelIdx: u32) -> ReservoirPTHero {
  let b = pixelIdx * RESERVOIR_PT_HERO_STRIDE;
  var r: ReservoirPTHero;
  r.xv      = vec3f(bitcast<f32>(buf[b + 0u]), bitcast<f32>(buf[b + 1u]), bitcast<f32>(buf[b + 2u]));
  r._pad0   = bitcast<f32>(buf[b + 3u]);
  r.nv      = vec3f(bitcast<f32>(buf[b + 4u]), bitcast<f32>(buf[b + 5u]), bitcast<f32>(buf[b + 6u]));
  r.W       = bitcast<f32>(buf[b + 7u]);
  r.xs      = vec3f(bitcast<f32>(buf[b + 8u]), bitcast<f32>(buf[b + 9u]), bitcast<f32>(buf[b + 10u]));
  r.w_sum   = bitcast<f32>(buf[b + 11u]);
  r.ns      = vec3f(bitcast<f32>(buf[b + 12u]), bitcast<f32>(buf[b + 13u]), bitcast<f32>(buf[b + 14u]));
  r.M       = buf[b + 15u];
  r.Lo      = vec3f(bitcast<f32>(buf[b + 16u]), bitcast<f32>(buf[b + 17u]), bitcast<f32>(buf[b + 18u]));
  r.pdfSrc  = bitcast<f32>(buf[b + 19u]);
  r.wi_recon          = vec3f(bitcast<f32>(buf[b + 20u]), bitcast<f32>(buf[b + 21u]), bitcast<f32>(buf[b + 22u]));
  r.distRecon         = bitcast<f32>(buf[b + 23u]);
  r.cosReconOut       = bitcast<f32>(buf[b + 24u]);
  r.prefixVertexCount = buf[b + 25u];
  r.roughnessV        = bitcast<f32>(buf[b + 26u]);
  r.metalV            = bitcast<f32>(buf[b + 27u]);
  r.albV              = vec3f(bitcast<f32>(buf[b + 28u]), bitcast<f32>(buf[b + 29u]), bitcast<f32>(buf[b + 30u]));
  r._pad1             = bitcast<f32>(buf[b + 31u]);
  r.hybridJacCache    = bitcast<f32>(buf[b + 32u]);
  r.hybridShiftPdf    = bitcast<f32>(buf[b + 33u]);
  r.rngSeed           = buf[b + 34u];
  r._padHybrid        = buf[b + 35u];
  return r;
}

fn loadReservoirPTHero_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32) -> ReservoirPTHero {
  let b = pixelIdx * RESERVOIR_PT_HERO_STRIDE;
  var r: ReservoirPTHero;
  r.xv      = vec3f(bitcast<f32>(buf[b + 0u]), bitcast<f32>(buf[b + 1u]), bitcast<f32>(buf[b + 2u]));
  r._pad0   = bitcast<f32>(buf[b + 3u]);
  r.nv      = vec3f(bitcast<f32>(buf[b + 4u]), bitcast<f32>(buf[b + 5u]), bitcast<f32>(buf[b + 6u]));
  r.W       = bitcast<f32>(buf[b + 7u]);
  r.xs      = vec3f(bitcast<f32>(buf[b + 8u]), bitcast<f32>(buf[b + 9u]), bitcast<f32>(buf[b + 10u]));
  r.w_sum   = bitcast<f32>(buf[b + 11u]);
  r.ns      = vec3f(bitcast<f32>(buf[b + 12u]), bitcast<f32>(buf[b + 13u]), bitcast<f32>(buf[b + 14u]));
  r.M       = buf[b + 15u];
  r.Lo      = vec3f(bitcast<f32>(buf[b + 16u]), bitcast<f32>(buf[b + 17u]), bitcast<f32>(buf[b + 18u]));
  r.pdfSrc  = bitcast<f32>(buf[b + 19u]);
  r.wi_recon          = vec3f(bitcast<f32>(buf[b + 20u]), bitcast<f32>(buf[b + 21u]), bitcast<f32>(buf[b + 22u]));
  r.distRecon         = bitcast<f32>(buf[b + 23u]);
  r.cosReconOut       = bitcast<f32>(buf[b + 24u]);
  r.prefixVertexCount = buf[b + 25u];
  r.roughnessV        = bitcast<f32>(buf[b + 26u]);
  r.metalV            = bitcast<f32>(buf[b + 27u]);
  r.albV              = vec3f(bitcast<f32>(buf[b + 28u]), bitcast<f32>(buf[b + 29u]), bitcast<f32>(buf[b + 30u]));
  r._pad1             = bitcast<f32>(buf[b + 31u]);
  r.hybridJacCache    = bitcast<f32>(buf[b + 32u]);
  r.hybridShiftPdf    = bitcast<f32>(buf[b + 33u]);
  r.rngSeed           = buf[b + 34u];
  r._padHybrid        = buf[b + 35u];
  return r;
}

fn storeReservoirPTHero_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32, r: ReservoirPTHero) {
  let b = pixelIdx * RESERVOIR_PT_HERO_STRIDE;
  buf[b + 0u]  = bitcast<u32>(r.xv.x);
  buf[b + 1u]  = bitcast<u32>(r.xv.y);
  buf[b + 2u]  = bitcast<u32>(r.xv.z);
  buf[b + 3u]  = bitcast<u32>(r._pad0);
  buf[b + 4u]  = bitcast<u32>(r.nv.x);
  buf[b + 5u]  = bitcast<u32>(r.nv.y);
  buf[b + 6u]  = bitcast<u32>(r.nv.z);
  buf[b + 7u]  = bitcast<u32>(r.W);
  buf[b + 8u]  = bitcast<u32>(r.xs.x);
  buf[b + 9u]  = bitcast<u32>(r.xs.y);
  buf[b + 10u] = bitcast<u32>(r.xs.z);
  buf[b + 11u] = bitcast<u32>(r.w_sum);
  buf[b + 12u] = bitcast<u32>(r.ns.x);
  buf[b + 13u] = bitcast<u32>(r.ns.y);
  buf[b + 14u] = bitcast<u32>(r.ns.z);
  buf[b + 15u] = r.M;
  buf[b + 16u] = bitcast<u32>(r.Lo.x);
  buf[b + 17u] = bitcast<u32>(r.Lo.y);
  buf[b + 18u] = bitcast<u32>(r.Lo.z);
  buf[b + 19u] = bitcast<u32>(r.pdfSrc);
  buf[b + 20u] = bitcast<u32>(r.wi_recon.x);
  buf[b + 21u] = bitcast<u32>(r.wi_recon.y);
  buf[b + 22u] = bitcast<u32>(r.wi_recon.z);
  buf[b + 23u] = bitcast<u32>(r.distRecon);
  buf[b + 24u] = bitcast<u32>(r.cosReconOut);
  buf[b + 25u] = r.prefixVertexCount;
  buf[b + 26u] = bitcast<u32>(r.roughnessV);
  buf[b + 27u] = bitcast<u32>(r.metalV);
  buf[b + 28u] = bitcast<u32>(r.albV.x);
  buf[b + 29u] = bitcast<u32>(r.albV.y);
  buf[b + 30u] = bitcast<u32>(r.albV.z);
  buf[b + 31u] = bitcast<u32>(r._pad1);
  // Phase-0 headroom (written-but-unread this increment).
  buf[b + 32u] = bitcast<u32>(r.hybridJacCache);
  buf[b + 33u] = bitcast<u32>(r.hybridShiftPdf);
  buf[b + 34u] = r.rngSeed;
  buf[b + 35u] = r._padHybrid;
}

// Streaming RIS reservoir update (Talbot 2005 / Bitterli 2020). Mirrors
// walkaround-hybrid's updateReservoirGI EXACTLY but carries the hero-specific
// pdfSrc alongside the chosen sample (so the resolve can divide by the REAL
// source pdf). \`rand_f32\` is forward-referenced (composed earlier from PCG_WGSL).
fn updateReservoirPT(
  r: ptr<function, ReservoirPTHero>,
  xs: vec3f, ns: vec3f, Lo: vec3f, pdfSrc: f32,
  w: f32,
  rng: ptr<function, u32>,
) {
  (*r).M = (*r).M + 1u;
  (*r).w_sum = (*r).w_sum + w;
  if (rand_f32(rng) * (*r).w_sum < w) {
    (*r).xs = xs;
    (*r).ns = ns;
    (*r).Lo = Lo;
    (*r).pdfSrc = pdfSrc;
  }
}

// The hero target function p̂ in the domain whose visible vertex is xv:
//   p̂(z) = luminance( f_bsdf(xv; wo→wi) · max(0, cos(nv, wi)) · Lo ),  wi = xv→xs
// the INTEGRAND-MATCHING target (the luminance of the real unshadowed reconnection
// contribution, using the visible-vertex BRDF) — NOT the diffuse-cosine proxy the
// diffuse-only GI version uses. It is a SCALAR resampling heuristic only — W =
// w_sum/p̂ divides it out, so the converged mean is INDEPENDENT of it (the producer
// W = 1/p_src cancellation holds for ANY p̂ — see the unbiasedness note); the RESOLVE
// pass likewise reconstructs with the REAL evaluateBrdf at xv. Matching the integrand
// only reduces RESAMPLING VARIANCE — decisively for a GLOSSY visible vertex whose
// direction-sensitive BRDF the old cosine proxy mis-weighted (the documented prefix-1
// glossy drift, fixed here). Returns 0 on a degenerate / back-facing edge.
fn restirPtTargetAt(xv: vec3f, nv: vec3f, wo: vec3f, albV: vec3f, roughnessV: f32, metalV: f32, xs: vec3f, Lo: vec3f) -> f32 {
  let d = xs - xv;
  let dist2 = dot(d, d);
  if (dist2 < 1e-8) { return 0.0; }
  let wi = d * inverseSqrt(dist2);
  let cosTheta = max(0.0, dot(nv, wi));
  if (cosTheta <= 0.0) { return 0.0; }
  let f = evaluateBrdf(albV, roughnessV, metalV, nv, wo, wi);
  return luminance(f * cosTheta * Lo);
}

// ── GRIS pairwise MIS (Lin 2022 §"pairwise MIS") — mirrors walkaround-hybrid's
// grisPairwiseDenomNeighbor / grisPairwiseDenomCanonical EXACTLY. The streaming
// temporal reuse folds the canonical (current) + the reprojected (previous)
// sample with the two-domain generalized balance heuristic. The shift Jacobian
// re-weights the RESAMPLING weight (multiplied in by the caller), NOT the target
// ratio — the target p̂ is evaluated in each term's NATIVE domain (commensurable
// densities). Returns the denominators; the caller forms m = numer/denom.
fn restirPtPairwiseDenomNeighbor(
  cR: f32, pHatR_atQsample: f32,
  cQ: f32, pHatQ_native: f32,
) -> f32 {
  return cR * pHatR_atQsample + cQ * pHatQ_native;
}

fn restirPtPairwiseDenomCanonical(
  cR: f32, pHatR_native: f32,
  cQ: f32, pHatQ_atRsample: f32,
) -> f32 {
  return cR * pHatR_native + cQ * pHatQ_atRsample;
}

// Finalise W for a GRIS hero reservoir (temporal reuse ON). GRIS folds each
// sample with a pairwise-MIS weight m_i where Σ m_i = 1, so the M-count does NOT
// normalise the sum — dividing by M again would under-energise the estimate.
//   W = w_sum / p̂(chosen sample)    — Lin 2022 §generalized RIS, NO /M.
// Mirrors walkaround-hybrid finaliseGIReservoirWGris EXACTLY (the GRIS form),
// with the hero target p̂ via restirPtTargetAt. wCap bounds the temporal-feedback
// gain (the V19 grison-divergence guard).
fn finaliseReservoirPTWGris(r: ptr<function, ReservoirPTHero>, wCap: f32, cameraPos: vec3f) {
  if ((*r).M > 0u) {
    let wo = restirpt_safe_normalize(cameraPos - (*r).xv);
    let pHatF = restirPtTargetAt((*r).xv, (*r).nv, wo, (*r).albV, (*r).roughnessV, (*r).metalV, (*r).xs, (*r).Lo);
    let W_raw = select(0.0, (*r).w_sum / pHatF, pHatF > 1e-9);
    (*r).W = min(W_raw, wCap);
  }
}

// Refresh the reconnection-shift cache (wi_recon / distRecon / cosReconOut /
// prefixVertexCount) from the CHOSEN base edge xv → xs, so downstream reuse sees
// a base half-G rooted at THIS pixel's visible vertex. Mirrors the proven GI
// refreshPhase0Cache + the temporalGi GRIS cache-refresh block. Leaves the cache
// zeroed + prefixVertexCount = 0 on an empty / degenerate reservoir.
fn refreshReconnectionCachePT(r: ptr<function, ReservoirPTHero>) {
  let toRecon = (*r).xs - (*r).xv;
  let dRecon = length(toRecon);
  if (dRecon > 1e-6 && (*r).M > 0u) {
    let wiR = toRecon / dRecon;
    (*r).wi_recon    = wiR;
    (*r).distRecon   = dRecon;
    (*r).cosReconOut = abs(dot((*r).ns, -wiR));
    (*r).prefixVertexCount = 1u;
  } else {
    (*r).wi_recon    = vec3f(0.0);
    (*r).distRecon   = 0.0;
    (*r).cosReconOut = 0.0;
    (*r).prefixVertexCount = 0u;
  }
}
`;

/**
 * The reservoir module composed with the reconnection-shift Jacobian it sits
 * next to. `RESTIR_PT_SHIFT_WGSL` (`restirPtReconnectionGeometryTerm` +
 * `restirPtShiftJacobian`) is the FD-validated shift this increment consumes;
 * emitting it HERE (before the reservoir helpers) keeps the two together for the
 * pass strings that reference both. `composePtWebgpuReuseWgsl` includes this once.
 */
export const RESERVOIR_PT_HERO_WITH_SHIFT_WGSL = /* wgsl */ `
${RESTIR_PT_SHIFT_WGSL}
${RESTIR_PT_PARAMS_WGSL}
${RESERVOIR_PT_HERO_WGSL}
`;
