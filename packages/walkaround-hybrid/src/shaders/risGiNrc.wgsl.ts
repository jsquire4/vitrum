/**
 * risGiNrc.wgsl.ts — NRC (Müller et al. 2021) variant of the Sprint-16
 * ReSTIR-GI RIS pass.
 *
 * Reference: Müller, Rousselle, Novák, Keller 2021, "Real-time Neural Radiance
 * Caching for Path Tracing", ACM TOG 40(4) §4 (encoding) + §5 (spread
 * termination + self-training).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * COMPILE-TIME NRC gate (nrcEnabled) — why this file exists alongside risGi.wgsl
 * ════════════════════════════════════════════════════════════════════════════
 * Live NRC is an OPT-IN feature gated by `HybridEngineOptions.nrcEnabled`. The
 * gate is resolved at PIPELINE-COMPILE time (fixed at engine creation), NOT at
 * runtime, because turning it on STRUCTURALLY changes the gi-ris pass: the ON
 * shader declares a `@group(4)` NRC bind group (MLP weights/biases, hash-grid
 * tables + level descriptors, the record-gather buffer, the encoding-config
 * uniform) so the inline MLP forward can run at the suffix vertex. The OFF
 * (default) shader is the verbatim Sprint-16 single-DDGI-estimate pass with NO
 * @group(4) and NONE of the NRC symbols.
 *
 * A runtime `ubo.nrcEnabled` flag is NOT sufficient: binding a fifth group /
 * changing the pipeline layout on the DEFAULT path alters the default pipeline
 * structure — exactly the class of change that regressed the default render to
 * an all-black frame for GRIS (f8df9a4). An opt-in feature must not change the
 * default pipeline at all, so the structure is gated at compile time:
 *   - {@link RIS_GI_MODULE}        (risGi.wgsl.ts) — OFF (default). 4 groups
 *                                   (frame/scene/ubo/hybrid), DDGI Lo estimate.
 *   - {@link buildRisGiNrcModule}  — ON. Adds @group(4) NRC + the inline MLP
 *                                   forward + spread-gated cache query + record
 *                                   gather. Composed ONLY when nrcEnabled.
 * {@link compilePipelines} composes whichever module matches the host flag and
 * builds the matching pipeline layout (4 groups OFF, 5 ON); {@link RISGIPass}
 * only calls `setBindGroup(4, …)` when ON.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * The NRC suffix replacement (the only behavioural delta vs the OFF pass)
 * ════════════════════════════════════════════════════════════════════════════
 * The OFF pass computes the reconnection-vertex outgoing radiance Lo by sampling
 * the DDGI irradiance atlas (`Lo = irrAtXs · albedo · INV_PI`). The NRC pass
 * tracks Müller's path SPREAD along the single bounce (primary→suffix edge); at
 * the suffix vertex, when the spread heuristic fires (a(x) > c·a₀), it REPLACES
 * that DDGI estimate with the MLP's predicted outgoing radiance (the cache
 * query). Below the spread threshold it keeps the DDGI estimate verbatim, so a
 * scene/region where spread never exceeds c·a₀ is bit-identical to the OFF pass.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * A6 — Training target: ReSTIR-GI reservoir Lo (not DDGI distillation)
 * ════════════════════════════════════════════════════════════════════════════
 * Previously, training wrote the DDGI irradiance (or sun+DDGI) as the target —
 * this was DISTILLATION: the cache learned to approximate the same estimate it
 * was replacing, with no fidelity upside.
 *
 * The correct target is the ReSTIR-GI reconnection radiance r.Lo — the Lo that
 * the full RIS process SELECTED as the best estimate for the visible pixel. This
 * Lo is computed post-loop (after the visibility test and W normalisation) and
 * stored in the reservoir. Training on r.Lo means the NRC converges to the same
 * quantity the ReSTIR-GI estimator produces — strictly more informative than
 * DDGI distillation because: (1) it incorporates RIS importance weighting across
 * M_GI candidates, (2) it includes the post-visibility correction (r.w_sum = 0
 * for occluded paths), and (3) it naturally accounts for sky-miss paths (Lo_env).
 *
 * BIAS BOUND: r.Lo is itself the biased-default ReSTIR-GI estimate (clamped
 * Jacobian [0.1,10], no reuse-visibility, centroid-p̂). The NRC converges to
 * THAT estimate, which is stricter than the previous DDGI-distillation bound
 * but still biased relative to the true path integral. See HARDWARE-VALIDATION-
 * NEEDS V20.
 *
 * IMPLEMENTATION: The NRC spread heuristic fires on the FIRST RIS candidate that
 * exceeds the threshold. The surface data at that candidate (xs, ns, wi, rough,
 * albedo) is saved in per-pixel tracking vars. After the loop, ONE training
 * record is written for the NRC-fired candidate using r.Lo (the final selected
 * reservoir Lo) as the target. If the NRC never fired this pixel, no record is
 * written. If a non-NRC candidate won the reservoir, r.Lo is a DDGI estimate for
 * that candidate — still a better signal than the NRC-candidate's own DDGI
 * (because it was importance-weighted selected). If the NRC candidate won, r.Lo
 * is nrcQueryRadiance(…) — which is circular but self-consistent (the NRC
 * gradient is computed against its own output, regularised by other-candidate
 * records; this is standard self-distillation that converges over frames).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * A6 — xsRough: real per-tri roughness from bvh_material (not hardcoded 1.0)
 * ════════════════════════════════════════════════════════════════════════════
 * The NRC now binds bvh_material at @group(1) @binding(14) (same as ris.wgsl /
 * restirCastPrimary.wgsl). At the bounce vertex xs, decodeRoughMetal returns the
 * authored roughness packed by BvhBufferHost. The QUERY and the RECORD use the
 * same real xsRough, making the MLP input consistent with the surface being
 * approximated. Default-diffuse scenes (rough=0.85) are numerically unchanged
 * relative to the old xsRough=1.0; only authored glossy/metal bounce vertices
 * (rare in a GI RIS pass whose candidates are cosine-hemisphere-sampled diffuse
 * targets) change.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * A6 — spreadC derivation + selectivity arithmetic
 * ════════════════════════════════════════════════════════════════════════════
 * The default spreadC = 0.01 is Müller §5's published value. With the H26
 * camera-pdf fix the a0 footprint is now physically correct — but this changes
 * WHEN the heuristic fires:
 *
 *   For a 1280×720 frame at 60° vFOV:
 *     cameraPixelPdf = cot²(30°) × 1280 × 720 / 4 = 3 × 921600 / 4 ≈ 691 200
 *     a0term = sqrt(d0² / (camPdf × cos)) = sqrt(25 / (691200 × 1)) ≈ 0.006
 *     a0 = 0.006² ≈ 3.6×10⁻⁵
 *     threshold = c × a0 = 0.01 × 3.6×10⁻⁵ ≈ 3.6×10⁻⁷
 *
 *   Bounce-1 spread (dist=2, pdf=0.7/π≈0.222, cosArrive=0.7):
 *     bounceSpread1 = sqrt(4 / (0.222 × 0.7)) = sqrt(25.7) ≈ 5.07
 *     aX1 = 5.07² ≈ 25.7  >>  threshold 3.6×10⁻⁷  →  fires at bounce 1
 *
 * With the correct camera pdf, c=0.01 ALWAYS fires at bounce 1 for typical
 * scene geometry. This is the CORRECT behaviour: Müller's heuristic is meant
 * to terminate "once the path footprint significantly exceeds the camera pixel
 * footprint." After a single GI bounce the path-footprint (∝ bounce-distance²/
 * bounce-pdf) is many orders of magnitude larger than the camera pixel footprint
 * (∝ 1/camPdf). Firing at bounce 1 is physically correct at high resolution.
 *
 * To fire at bounce 2-3 instead (less aggressive, lower bias):
 *   Require aX1 ≤ c × a0, i.e. c ≥ aX1/a0 = 25.7 / 3.6×10⁻⁵ ≈ 714 000.
 *   To fire at bounce 2: c ∈ [714 000, 4 000 000].  Choose c ≈ 1e6.
 *   BUT: these values are scene-scale-dependent (bounce distances, pdfs vary).
 *   For low-res scenes (320×240 @ 90° fovY): camPdf ≈ 19 200, a0 ≈ 0.0013,
 *   aX1 ≈ 25.7, ratio ≈ 19 750 → c=1e6 would never fire in 3 bounces.
 *
 * DECISION (A6): c=0.01 is retained. It fires at bounce 1 for typical
 * high-resolution scenes (physically correct, aggressive cache use) and is
 * still a no-op for sky-miss pixels (no bounceHit → NRC block not reached).
 * Users who want bounce-2 termination for a specific scene should set spreadC
 * via HybridEngineOptions (it is exposed as a first-class option). The default
 * is documented here; the new tests in spreadTermination.test.ts enforce it.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { nrcEncodeHelpersWgsl } from '../neural/nrc/wgsl/nrcEncoding.wgsl.js';
import { nrcSpreadTerminationWgsl } from '../neural/nrc/wgsl/spreadTermination.wgsl.js';
import { nrcQueryWgsl, type NrcQueryWgslOptions } from '../neural/nrc/wgsl/nrcQuery.wgsl.js';

/** Config the NRC gi-ris module bakes its sizes from. Must agree with the host
 *  `NrcSubsystem` config (same hash-grid L/F, one-blob bins, MLP W/OUT/hidden). */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type -- intentional branded alias; named type aids type errors and host visibility
export interface RisGiNrcConfig extends NrcQueryWgslOptions {}

// The NRC gi-ris compute body. Identical to RIS_GI_WGSL EXCEPT:
//   • it tracks the primary-edge footprint a0 + the bounce-edge spread a(x),
//   • at the suffix vertex it cache-queries the MLP when the spread fires,
//   • it writes a self-training record per pixel.
// The @group(0..3) bindings are byte-identical to risGi.wgsl; @group(4) is the
// NRC group declared by the nrcQueryWgsl helpers prepended to this body.
export const RIS_GI_NRC_BODY = /* wgsl */ `

@group(0) @binding(10) var gi_gNormalDepth: texture_2d<f32>;
@group(0) @binding(11) var<storage, read_write> reservoirGiCurrent: array<u32>;

@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(6) var<storage, read> tlasNodes: array<BVHNode>;
@group(1) @binding(7) var<storage, read> tlasInstanceIndices: array<u32>;
@group(1) @binding(8) var<storage, read> tlasBlasRoots: array<u32>;
@group(1) @binding(9) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(1) @binding(10) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
// WS1 (2026-05-29) — bvh_normal is declared by materialAtlas.wgsl so alpha
// cutout traversal and NRC GI shading share the same UV1/normal source.
// A6 — per-triangle roughness+metalness (r32uint texture, binding 14). Decoded
// into the real authored roughness at the bounce vertex xs, replacing the
// old hardcoded xsRough=1.0. Same binding as ris.wgsl / restirCastPrimary.wgsl;
// decodeRoughMetal comes from the materialDecode module (BVH_MATERIAL_TEX_WIDTH
// constant + fn decodeRoughMetal). Default-diffuse invariant: no authored
// roughness → packed as 0.85 (see packingHelpers.packBVHRoughMetal).
@group(1) @binding(14) var bvh_material: texture_2d<u32>;

@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;
@group(2) @binding(2) var gi_tier: texture_2d<u32>;

// D5.1+D5.2: DDGIGridUBO struct, @group(3) @binding(3) ddgiGrid UBO, and
// sampleDDGIAtPoint are now provided by the shared ddgiGridUbo module.
@group(3) @binding(0) var ddgiIrradiance: texture_2d<f32>;
@group(3) @binding(1) var ddgiVisibility: texture_2d<f32>;
@group(3) @binding(2) var ddgiSampler:    sampler;

const M_GI_BASE: u32 = 8u;
const RECONNECT_MAX_DIST: f32 = 100.0;
const NORMAL_BIAS_GI: f32 = 1e-3;

@compute @workgroup_size(8, 8, 1)
fn risGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }

  let pixelIdxGi = gid.y * halfDims.x + gid.x;
  let fullPx = gid.xy * 2u + 1u;

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0xA5A5u),
    gid.y ^ (ubo.frameSeed * 0x5A5Au),
    ubo.frameSeed ^ 0xC1A2u,
  );

  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let primaryRay = generatePrimaryRay_common(
    fullPx.x, fullPx.y, fullDims.x, fullDims.y, ubo.cameraPos, invVP,
  );
  let hit = traceSceneFirstHitAlphaMaskTextured(
    ubo.bvhMode, ubo.tlasNodeCount,
    &bvh_index, &bvh_position, &bvh,
    &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
    &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
    primaryRay, ubo.triIntersectEpsilon,
    bvh_material, BVH_MATERIAL_TEX_WIDTH);
  if (!hit.didHit) {
    storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, emptyReservoirGI());
    return;
  }

  let pos = primaryRay.origin + primaryRay.direction * hit.dist;
  // WS1 — smooth shading normal; geometric normal kept for the bounce offset.
  // V21 — applies in TLAS too (LOCAL blend → world by the instance inverse-transpose).
  let geoNormal = hit.normal;
  let n_isTlas = ubo.bvhMode == 1u;
  let n_base = hit.instanceIndex * 4u;
  let n_ok = n_isTlas && n_base + 2u < arrayLength(&tlasInstanceWorldToLocal);
  let n_i = select(0u, n_base, n_ok);
  let normal = smoothShadingNormal(
    hit, geoNormal,
    bvh_normal[hit.indices.x].xyz, bvh_normal[hit.indices.y].xyz, bvh_normal[hit.indices.z].xyz,
    n_ok,
    tlasInstanceWorldToLocal[n_i], tlasInstanceWorldToLocal[n_i + 1u], tlasInstanceWorldToLocal[n_i + 2u],
  );
  // B1 — metals/glossy now get a (diffuse-target) GI reservoir; shade reflects
  // it via the GGX specular lobe. Glass still punts (refracted GI out of scope).
  // Mirrors risGi.wgsl. The Lambertian target p̂ is unchanged.
  let scalarMatColor = decodeMaterialColor(hit.matColorPacked);
  let matColor = vec4f(
    scalarMatColor.rgb,
    sampleTransmissionMapForHit(hit, scalarMatColor.a),
  );
  let isGlass = matColor.a > 0.3;
  if (isGlass) {
    storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, emptyReservoirGI());
    return;
  }

  var r: ReservoirGI = emptyReservoirGI();
  r.xv = pos;
  r.nv = normal;

  let tier_raw = textureLoad(gi_tier, vec2i(fullPx), 0).r;
  let tier = clamp(tier_raw, 1u, 4u);
  let M_GI = M_GI_BASE * tier / 2u;

  let ppgGuidedOn = (ubo.ppgEnabled == 1u);
  let alpha = select(0.0, ubo.ppgMixAlpha, ppgGuidedOn);

  // ── NRC: primary-vertex footprint a0 (Müller §5). The primary edge is
  // camera→primary hit; its spread term uses the camera's per-pixel
  // solid-angle pdf (nrcCfg.cameraPixelPdf — host-computed each frame from
  // the projection matrix and render resolution; see NrcSubsystem.updateCameraPixelPdf)
  // and the primary-hit |cosθ|. a0 = (first-segment term)². The bounce edge's
  // spread is compared against c·a0 to decide cache termination.
  //
  // H26 camera-pdf fix: previously this used pdf=1.0 (pinhole unit-resolution
  // fallback). The Müller-correct value is the camera's per-pixel solid-angle
  // pdf = cot²(fovY/2)·W·H/4, which grows with resolution and FOV narrowing.
  // Higher camPdf → smaller a0term → smaller a0 → smaller threshold c·a0 →
  // the heuristic fires SOONER (more aggressive cache use at high resolution).
  // This is physically correct: a high-resolution camera has a narrow per-pixel
  // footprint; after one GI bounce the path has spread far beyond it → terminate
  // into the cache. See the A6 spreadC derivation in the file-level docblock.
  let cosThetaPrimary = max(1e-4, abs(dot(normal, primaryRay.direction)));
  let a0term = nrcSegmentSpreadTerm(hit.dist, nrcCfg.cameraPixelPdf, cosThetaPrimary);
  let a0 = a0term * a0term;

  // A6 — NRC candidate tracking: save surface data for the FIRST candidate
  // that triggers the spread heuristic. The training record is written once
  // after the loop using r.Lo (the final ReSTIR-GI selected radiance) as the
  // target — not the per-candidate DDGI estimate (which was DDGI distillation).
  var nrcFired: bool = false;
  var nrcTrackXs: vec3f = vec3f(0.0);
  var nrcTrackNs: vec3f = vec3f(0.0, 1.0, 0.0);
  var nrcTrackWi: vec3f = vec3f(0.0);
  var nrcTrackRough: f32 = 1.0;
  var nrcTrackAlbedo: vec3f = vec3f(0.0);

  for (var i: u32 = 0u; i < M_GI; i = i + 1u) {
    var wi: vec3f;
    if (alpha > 0.0) {
      let bern = rand_f32(&rng);
      if (bern < alpha) {
        wi = ppgSampleGuidedDir(pos, &rng);
      } else {
        wi = sampleCosineHemisphere(normal, &rng);
      }
    } else {
      wi = sampleCosineHemisphere(normal, &rng);
    }
    let cosTheta = max(0.0, dot(normal, wi));
    if (cosTheta < 1e-4) { continue; }

    // WS1 — offset the bounce-ray origin along the GEOMETRIC normal.
    let bounceRay = Ray(pos + geoNormal * NORMAL_BIAS_GI, wi);
    let bounceHit = traceSceneFirstHitAlphaMaskTextured(
      ubo.bvhMode, ubo.tlasNodeCount,
      &bvh_index, &bvh_position, &bvh,
      &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
      &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
      bounceRay, ubo.triIntersectEpsilon,
      bvh_material, BVH_MATERIAL_TEX_WIDTH,
    );

    var xs:  vec3f;
    var ns:  vec3f;
    var Lo:  vec3f;

    if (bounceHit.didHit) {
      xs = bounceRay.origin + wi * bounceHit.dist;
      ns = bounceHit.normal;
      let irrAtXs = min(sampleDDGIAtPoint(xs, ns), vec3f(ubo.restirGiIrrClamp));
      let xsMat = decodeMaterialColor(bounceHit.matColorPacked);
      let ddgiLo = irrAtXs * xsMat.rgb * INV_PI;

      // ── NRC cache termination (Müller §5) ──
      // The bounce edge pos→xs accumulates spread. The candidate's source pdf
      // is the cosine-hemisphere pdf cosθ/π (the always-present component; the
      // guided dTree term only narrows it). When a(x) > c·a0 the suffix is
      // TERMINATED into the cache: the MLP prediction REPLACES the DDGI suffix
      // estimate. Below threshold the DDGI estimate is kept verbatim, so a
      // sub-threshold region is bit-identical to the OFF pass.
      //
      // H26 seeding fix: runningSum starts at 0.0 (Müller 2021 §5). The
      // previous seed of a0term was a tautology: nrcAccumulateSpread adds
      // the bounce-edge segment term to runningSum, then squares it for a(x).
      // Seeding with a0term made a(x) depend on a0term even before the first
      // bounce edge was measured — the threshold a(x) > c·a0 fired
      // immediately on the primary hit (k=0), turning every pixel into a
      // cache query. The correct seeding (0.0) means only the bounce edge's
      // spread term matters, matching the oracle in spreadTermination.ts.
      var runningSum: f32 = 0.0;
      let cosArrive = max(1e-4, abs(dot(ns, -wi)));
      let pSrcBounce = max(cosTheta * INV_PI, 1e-12);
      let aX = nrcAccumulateSpread(&runningSum, bounceHit.dist, pSrcBounce, cosArrive);
      if (nrcShouldTerminateIntoCache(aX, a0, nrcCfg.spreadC)) {
        let xsAlbedo = xsMat.rgb;
        // A6 — real per-tri roughness from bvh_material (was hardcoded 1.0).
        // decodeRoughMetal returns vec2f(roughness, metalness); we only need
        // roughness for the NRC encoding (xsRough). The diffuse-default invariant
        // packs 0.85 for materials without authored roughness, so default-diffuse
        // scenes are numerically close to the old xsRough=1.0.
        let xsRmCoord = vec2u(
          bounceHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
          bounceHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
        );
        let xsRough = decodeRoughMetal(
          textureLoad(bvh_material, vec2i(xsRmCoord), 0).r
        ).x;
        // Query the cache for outgoing radiance toward the visible point
        // (view dir at xs is −wi, the incident bounce direction reversed).
        Lo = nrcQueryRadiance(xs, ns, -wi, xsRough, xsAlbedo);
        // A6 — save the NRC candidate surface data for the post-loop record.
        // We only save the FIRST fired candidate (the one most likely to be
        // importance-selected into the reservoir). After the loop, we write
        // ONE record with r.Lo as the training target (see below).
        if (!nrcFired) {
          nrcFired = true;
          nrcTrackXs = xs;
          nrcTrackNs = ns;
          nrcTrackWi = wi;
          nrcTrackRough = xsRough;
          nrcTrackAlbedo = xsAlbedo;
        }
      } else {
        Lo = ddgiLo;
      }
    } else {
      xs = pos + wi * RECONNECT_MAX_DIST;
      ns = -wi;
      // Wave 4 parity — directional IBL: sample the actual map along wi (same
      // as risGi.wgsl:264). envRadiance falls back to skyTint×skyIrradiance when
      // no HDRI is bound, preserving byte-identity with the scalar-sky path.
      Lo = envRadiance(wi);
    }

    let pHat = luminance(Lo) * cosTheta * INV_PI;
    if (pHat < 1e-9) { continue; }

    var w: f32;
    if (alpha > 0.0) {
      let pCos = cosTheta * INV_PI;
      let pGuide = ppgEvalPdf(pos, wi);
      let pSrc = alpha * pGuide + (1.0 - alpha) * pCos;
      w = select(0.0, pHat / pSrc, pSrc > 1e-12);
    } else {
      w = luminance(Lo);
    }
    updateReservoirGI(&r, xs, ns, Lo, w, &rng);
  }

  if (r.M > 0u && r.w_sum > 0.0) {
    let toS = r.xs - r.xv;
    let distS = length(toS);
    if (distS > 1e-4) {
      let wiZ = toS / distS;
      let shadowOrig = r.xv + r.nv * NORMAL_BIAS_GI;
      let occ = traceSceneAnyCastMask(
        ubo.bvhMode, ubo.tlasNodeCount,
        &bvh_index, &bvh_position, &bvh,
        &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
        &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
        shadowOrig, wiZ, distS - 2e-3, ubo.triIntersectEpsilon, true,
        bvh_material, BVH_MATERIAL_TEX_WIDTH,
      );
      if (occ) {
        r.w_sum = 0.0;
        r.W = 0.0;
      } else {
        let cosThetaZ = max(0.0, dot(r.nv, wiZ));
        let pHatZ = luminance(r.Lo) * cosThetaZ * INV_PI;
        let W_raw = select(0.0, r.w_sum / (f32(r.M) * pHatZ), pHatZ > 1e-9);
        r.W = min(W_raw, ubo.restirGiWCap);
      }
    } else {
      r.W = 0.0;
      r.w_sum = 0.0;
    }
  }

  // A6 — post-loop NRC training record (ReSTIR-GI Lo as target).
  // Write ONE record if the spread heuristic fired on at least one candidate.
  // r.Lo is the final selected reservoir radiance — the ReSTIR-GI reconnection
  // estimate for this pixel. Training on r.Lo means the NRC converges to the
  // same quantity the RIS estimator produces (vs the old DDGI distillation).
  //
  // BIAS BOUND (documented): r.Lo is itself the biased ReSTIR-GI estimate
  // (clamped Jacobian, no reuse-visibility, centroid-p̂). The NRC converges to
  // THAT, which is strictly more informative than DDGI-only distillation.
  //
  // TAIL-PADDING GUARD: records where r.Lo == 0 (occluded sample, W=0) are
  // valid zero-radiance training data — the NRC should predict 0 for dark
  // surfaces. The host skips only all-zero ENCODED-INPUT slots (indicating the
  // slot was never written), not zero-radiance slots. Zero Lo trains the MLP
  // toward 0 for occlusion, which is correct.
  if (nrcFired) {
    // Slot: deterministic per-pixel assignment (pixelIdxGi % recordCap) with
    // first-writer-wins atomic claim in nrcWriteRecord (H27 torn-record fix).
    nrcWriteRecord(
      pixelIdxGi % nrcCfg.recordCap,
      nrcTrackXs, nrcTrackNs, -nrcTrackWi,
      nrcTrackRough, nrcTrackAlbedo,
      r.Lo,
    );
  }

  // GRIS Phase-0 reconnection-shift cache (Lin et al. 2022 §5) — written, read
  // by no pass in Phase 0.  Shared with risGi.wgsl via refreshPhase0Cache.
  refreshPhase0Cache(&r);

  storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, r);
}
`;

/**
 * Build the NRC gi-ris module for a given encoding/MLP config. The source is:
 *   nrcEncodeHelpers (hash + one-blob + normalise)
 *   + nrcHashGridForward (NrcLevelDesc + trilinear forward)
 *   + nrcSpreadTermination (segment-spread + accumulate + predicate)
 *   + nrcQuery (@group(4) bindings + one-blob/oct/assemble/MLP-forward/record)
 *   + RIS_GI_NRC_BODY (the gi-ris compute body)
 * Its `requires` MATCH RIS_GI_MODULE exactly (walkaroundUbo / sceneTraversal /
 * reservoirGi / sharedPrimitives / materialDecode / cameraRays / ddgiSample /
 * ppgPdf) so the @group(0..3) closure is identical; the only structural delta
 * is the prepended NRC helpers + the @group(4) NRC bindings.
 *
 * Composed ONLY when nrcEnabled is compile-time true — see compilePipelines.
 */
export function buildRisGiNrcModule(cfg: RisGiNrcConfig): WgslModule {
  // nrcEncodeHelpers provides nrcSpatialHash3D / nrcNormalizeToAabb /
  // nrcOneBlobScalar (value/function-space args only — WGSL-portable). The
  // hash-grid trilinear forward is INLINED inside nrcQueryWgsl (it reads the
  // storage `nrcTables` directly; a helper taking a storage pointer is
  // WGSL-illegal without unrestricted_pointer_parameters). So we do NOT include
  // nrcHashGridForwardWgsl here.
  const source =
    nrcEncodeHelpersWgsl() +
    nrcSpreadTerminationWgsl() +
    nrcQueryWgsl(cfg) +
    RIS_GI_NRC_BODY;
  return {
    name: 'risGiNrc',
    source,
    // Wave 4 parity: `environmentSample` added so envRadiance() is in scope for
    // the GI-escape sky-miss branch (matching risGi.wgsl:264). The @group(1)
    // env bindings 15-19 are already present in the scene BGL for NRC passes.
    // D5.1+D5.2: ddgiSample replaced by ddgiGridUbo (which requires ddgiSample
    // transitively, and adds the DDGIGridUBO struct + binding + sampleDDGIAtPoint).
    requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'materialDecode', 'materialAtlas', 'cameraRays', 'ddgiGridUbo', 'ppgPdf', 'environmentSample'],
  };
}
