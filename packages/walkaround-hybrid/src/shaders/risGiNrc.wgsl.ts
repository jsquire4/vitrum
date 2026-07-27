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
 *
 * The cache substitution is additionally warm-up gated: the spread heuristic
 * may write training records from frame 0, but the reservoir keeps the DDGI
 * suffix estimate until the host reports enough completed trainer windows in
 * `nrcCfg.trainedSteps`. This avoids replacing radiance with cold random MLP
 * predictions while preserving self-training cadence.
 * {@link compilePipelines} composes whichever module matches the host flag and
 * builds the matching pipeline layout (4 groups OFF, 5 ON); {@link RISGIPass}
 * only calls `setBindGroup(4, …)` when ON.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * The NRC suffix replacement (the only behavioural delta vs the OFF pass)
 * ════════════════════════════════════════════════════════════════════════════
 * The OFF pass computes the reconnection-vertex outgoing radiance Lo by sampling
 * the DDGI irradiance atlas and applying the same material-aware suffix response
 * as risGi.wgsl (`albedo / π` for ordinary suffixes; extension-aware
 * GGX/clearcoat/sheen proxy for rich suffixes). The NRC pass
 * tracks Müller's path SPREAD along the single bounce (primary→suffix edge); at
 * the suffix vertex, when the spread heuristic fires (a(x) > c·a₀), it REPLACES
 * that material-shaded DDGI suffix estimate with the MLP's predicted outgoing
 * radiance (the cache query). Below the spread threshold it keeps the same
 * material-shaded suffix estimate verbatim, so a scene/region where spread never
 * exceeds c·a₀ is bit-identical to the OFF pass.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * NRC training target: independent bounded path suffix
 * ════════════════════════════════════════════════════════════════════════════
 * The first qualifying opaque candidate launches a private four-vertex path
 * suffix. Its target contains mapped self-emission/light maps, exact-CDF finite
 * emitter NEE, analytic point/spot and directional-sun NEE, and BSDF continuation
 * to the environment. A defensive cosine/GGX-VNDF mixture evaluates the full
 * mapped material response with its matching mixture PDF; Russian roulette starts
 * after two vertices. The target reads neither DDGI nor the cache prediction.
 *
 * Cache substitution is still receiver-lobe/material-aware when the candidate
 * enters the ReSTIR-GI reservoir; the independent target is matched to its key.
 * The teacher stream is seeded independently from RIS, so training cannot change
 * later candidates or reservoir selection. Glass candidates are not cached because
 * the current NRC key has no IOR/transmission coordinate; they retain the dedicated
 * bounded glass-GI path instead of aliasing optically distinct states.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * A6 — xsRough: real material-payload roughness (not hardcoded 1.0)
 * ════════════════════════════════════════════════════════════════════════════
 * The NRC GI-RIS variant now samples the same mapped material payload as the
 * OFF pass at the bounce vertex xs. The QUERY and the RECORD use
 * `xsPayload.rough` and `xsPayload.albedo`, making the MLP input consistent
 * with the surface whose outgoing radiance is being approximated. Default
 * diffuse scenes (rough≈0.85) remain close to the old xsRough=1.0 path; authored
 * glossy/metal/mapped bounce vertices now train/query with their real payload.
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
import {
  RIS_GI_GLASS_TRANSPORT_PREFIX_WGSL,
  RIS_GI_GLASS_RESERVOIR_LOOP_WGSL,
  RIS_GI_GLASS_VISIBILITY_TAIL_WGSL,
} from './risGiGlassWalk.wgsl.js';
import { NRC_INDEPENDENT_SUFFIX_WGSL } from './nrcIndependentSuffix.wgsl.js';
import { reservoirGiAccessorsWgsl } from './reservoirGi.wgsl.js';

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

@group(0) @binding(11) var<storage, read_write> reservoirGiCurrent: array<u32>;

${reservoirGiAccessorsWgsl({ storeReadWriteBinding: 'reservoirGiCurrent' })}

@group(1) @binding(5) var bvh_beer: texture_2d<u32>;
// WS1 (2026-05-29) — bvh_normal is declared by materialAtlas.wgsl so alpha
// cutout traversal and NRC GI shading share the same UV1/normal source.
// A6 — per-triangle roughness+metalness (r32uint texture, binding 14). Decoded
// into the real authored roughness at the bounce vertex xs, replacing the
// old hardcoded xsRough=1.0. Same binding as ris.wgsl / restirCastPrimary.wgsl;
// decodeRoughMetal comes from the materialDecode module (BVH_MATERIAL_TEX_WIDTH
// constant + fn decodeRoughMetal). Default-diffuse invariant: no authored
// roughness → packed as 0.85 (see packingHelpers.packBVHRoughMetal).
@group(1) @binding(13) var analytic_lights: texture_2d<f32>;
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

${NRC_INDEPENDENT_SUFFIX_WGSL}
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
  let hit = traceSceneFirstHitAlphaMaskTexturedOpaqueOnly(
    ubo.bvhMode, ubo.tlasNodeCount,

    primaryRay, ubo.triIntersectEpsilon,
    bvh_material, BVH_MATERIAL_TEX_WIDTH, 0u);
  if (!hit.didHit) {
    storeReservoirGI_rw(pixelIdxGi, emptyReservoirGI());
    return;
  }

  let pos = primaryRay.origin + primaryRay.direction * hit.dist;
  // WS1 — smooth shading normal; geometric normal kept for the bounce offset.
  // V21 — applies in TLAS too (LOCAL blend → world by the instance inverse-transpose).
  let geoNormal = hit.normal;
  let n_isTlas = ubo.bvhMode == 1u;
  let n_base = hit.instanceIndex * 4u;
  let n_ok = n_isTlas && n_base + 2u < tlasWorldToLocalColumnCount();
  let n_i = select(0u, n_base, n_ok);
  let smoothNormal = smoothShadingNormal(
    hit, geoNormal,
    sceneLoadBvhNormal(hit.indices.x).xyz, sceneLoadBvhNormal(hit.indices.y).xyz, sceneLoadBvhNormal(hit.indices.z).xyz,
    n_ok,
    tlasLoadWorldToLocalColumn(n_i), tlasLoadWorldToLocalColumn(n_i + 1u), tlasLoadWorldToLocalColumn(n_i + 2u),
  );
  let normalMapped = applyNormalMapForHit(hit, smoothNormal);
  let normal = applyBumpMapForHit(hit, normalMapped);
  // B1 — metals/glossy now get a GI reservoir; shade reflects it via the GGX
  // specular lobe. Glass primaries mirror risGi.wgsl's bounded multi-interface
  // refracted GI walk, then return before NRC cache substitution/training. That
  // keeps the learned cache scope honest without dropping glass indirect light.
  let scalarMatColor = decodeMaterialColor(hit.matColorPacked);
  let matColor = vec4f(
    scalarMatColor.rgb,
    sampleTransmissionMapForHit(hit, scalarMatColor.a),
  );
  let receiverMaterialWordCoord = vec2u(
    hit.indices.w % BVH_MATERIAL_TEX_WIDTH,
    hit.indices.w / BVH_MATERIAL_TEX_WIDTH,
  );
  let receiverMaterialWord = textureLoad(bvh_material, vec2i(receiverMaterialWordCoord), 0).r;
  let receiverPayload = sampleRestirDIMaterialPayloadForHit(
    hit,
    smoothNormal,
    normal,
    scalarMatColor.rgb,
    receiverMaterialWord,
    -primaryRay.direction,
  );
  let receiverClearcoatNormal = receiverPayload.clearcoatNormal;
  let receiverWo = -primaryRay.direction;
  let isGlass = materialHasTransmission(matColor.a);
  let grisOn = ubo.grisReuse == 1u;
  let currentGrisEpoch = bitcast<u32>(ubo.sunAngular.y);

${RIS_GI_GLASS_TRANSPORT_PREFIX_WGSL}

${RIS_GI_GLASS_RESERVOIR_LOOP_WGSL}

${RIS_GI_GLASS_VISIBILITY_TAIL_WGSL}
  }

  var r: ReservoirGI = emptyReservoirGI();
  r.xv = pos;
  r.nv = normal;
  if (grisOn) { r.historyEpoch = currentGrisEpoch; }

  let tier_raw = textureLoad(gi_tier, vec2i(fullPx), 0).r;
  let tier = clamp(tier_raw, 1u, 4u);
  let M_GI = M_GI_BASE * tier / 2u;

  let ppgGuidedOn = (ubo.ppgEnabled == 1u) && !grisOn;
  let alpha = select(0.0, ubo.ppgMixAlpha, ppgGuidedOn);

  // ── NRC: primary-vertex footprint a0 (Müller §5). The primary edge is
  // camera→primary hit; its spread term uses the camera's per-pixel
  // solid-angle pdf (nrcCfg.cameraPixelPdf — host-computed each frame from
  // the projection matrix and render resolution; see NrcSubsystem.updateCameraPixelPdf)
  // and the primary-hit |cosθ|. a0 = (first-segment term)². The bounce edge's
  // spread is compared against c·a0 to decide cache termination.
  //
  // H26 camera-pdf fix: previously this used pdf=1.0 (pinhole unit-resolution
  // fallback). The centre-pixel pinhole approximation is
  // pdf = |fx·fy|·W·H/4, which grows with resolution and FOV narrowing.
  // Higher camPdf → smaller a0term → smaller a0 → smaller threshold c·a0 →
  // the heuristic fires SOONER (more aggressive cache use at high resolution).
  // This is physically correct: a high-resolution camera has a narrow per-pixel
  // footprint; after one GI bounce the path has spread far beyond it → terminate
  // into the cache. See the A6 spreadC derivation in the file-level docblock.
  let cosThetaPrimary = abs(dot(normal, primaryRay.direction));
  let a0term = nrcSegmentSpreadTerm(hit.dist, nrcCfg.cameraPixelPdf, cosThetaPrimary);
  let a0 = a0term * a0term;

  let nrcCanSubstitute = nrcCfg.trainedSteps >= nrcCfg.warmupSteps
    && nrcInferenceArenaValid() && nrcRuntimeArenaValid();


  // Bound teacher work to at most recordCap suffixes per frame. Each record
  // slot owns one contiguous pixel block and selects one frame-varying pixel
  // from that block, so eligible invocations never collide and the sample set
  // changes with frameSeed without perturbing the RIS RNG stream.
  let nrcPixelCount = halfDims.x * halfDims.y;
  let nrcTeacherCap = max(nrcCfg.recordCap, 1u);
  let nrcTeacherStride = max(1u, 1u + (nrcPixelCount - 1u) / nrcTeacherCap);
  let nrcTeacherSlot = pixelIdxGi / nrcTeacherStride;
  let nrcTeacherBlockStart = nrcTeacherSlot * nrcTeacherStride;
  let nrcTeacherBlockLength = min(
    nrcTeacherStride,
    nrcPixelCount - nrcTeacherBlockStart,
  );
  var nrcTeacherSelectRng = pcgInit(
    nrcTeacherSlot,
    ubo.frameSeed ^ 0x74656163u,
    0x68657221u,
  );
  let nrcTeacherSelectedLocal = min(
    u32(rand_f32(&nrcTeacherSelectRng) * f32(nrcTeacherBlockLength)),
    nrcTeacherBlockLength - 1u,
  );
  let nrcTeacherEligible =
    nrcCfg.recordCap > 0u &&
    nrcTeacherSlot < nrcCfg.recordCap &&
    pixelIdxGi - nrcTeacherBlockStart == nrcTeacherSelectedLocal;

  // NRC candidate tracking: preserve a matched input/teacher pair for the first
  // candidate that triggers the spread heuristic. The target is captured from
  // this candidate before cache substitution, preventing a cross-candidate or
  // self-prediction label after reservoir selection.
  var nrcFired: bool = false;
  var nrcTrackXs: vec3f = vec3f(0.0);
  var nrcTrackNs: vec3f = vec3f(0.0, 1.0, 0.0);
  var nrcTrackWi: vec3f = vec3f(0.0);
  var nrcTrackRough: f32 = 1.0;
  var nrcTrackAlbedo: vec3f = vec3f(0.0);

  var nrcTrackTarget: vec3f = vec3f(0.0);
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
    if (!nrcFinite(cosTheta) || !(cosTheta > 0.0)) {
      if (grisOn) {
        recordInvalidReservoirGICandidate(&r, GI_SAMPLE_SURFACE, currentGrisEpoch);
      } else {
        r.M = r.M + 1u;
      }
      continue;
    }

    // WS1 — offset the bounce-ray origin along the GEOMETRIC normal.
    let bounceRay = Ray(pos + geoNormal * NORMAL_BIAS_GI, wi);
    let bounceHit = traceSceneFirstHitAlphaMaskTextured(
      ubo.bvhMode, ubo.tlasNodeCount,

      bounceRay, ubo.triIntersectEpsilon,
      bvh_material, BVH_MATERIAL_TEX_WIDTH,
      ubo.frameSeed ^ (i * 0x85ebca6bu) ^ 0x4e474942u,
    );

    var xs:  vec3f;
    var ns:  vec3f;
    var Lo:  vec3f;
    var sampleKind: u32 = GI_SAMPLE_ENVIRONMENT;

    if (bounceHit.didHit) {
      sampleKind = GI_SAMPLE_SURFACE;
      xs = bounceRay.origin + wi * bounceHit.dist;
      let smoothNs = restir_gi_smooth_normal_for_hit(bounceHit, bounceHit.normal);
      ns = applyBumpMapForHit(bounceHit, applyNormalMapForHit(bounceHit, smoothNs));
      let irrAtXs = min(sampleDDGIAtPoint(xs, ns), vec3f(ubo.restirGiIrrClamp));
      let xsRmCoord = vec2u(
        bounceHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
        bounceHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
      );
      let xsMaterialWord = textureLoad(bvh_material, vec2i(xsRmCoord), 0).r;
      let xsPayload = sampleRestirGIHitMaterialForHit(
        bounceHit,
        smoothNs,
        ns,
        irrAtXs,
        wi,
        xsMaterialWord,
      );
      let ddgiProxyLo = restir_gi_surface_source_for_hit(bounceHit, xsPayload.albedo)
        + irrAtXs * xsPayload.albedo * INV_PI;
      let ddgiLo = select(xsPayload.Lo, ddgiProxyLo, grisOn);
      let xsTransmission = sampleTransmissionMapForHit(
        bounceHit,
        decodeMaterialColor(bounceHit.matColorPacked).a,
      );

      // ── NRC cache termination (Müller §5) ──
      // The bounce edge pos→xs accumulates spread. The candidate's source pdf
      // is the cosine-hemisphere pdf cosθ/π (the always-present component; the
      // guided dTree term only narrows it). When a(x) > c·a0 the suffix is
      // TERMINATED into the cache: once warm, the MLP prediction REPLACES the
      // material-shaded DDGI suffix estimate. Before warm-up, the same fired
      // candidate still writes a training record but keeps ddgiLo in the
      // reservoir so cold random predictions never drive visible GI.
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
      let cosArrive = abs(dot(ns, -wi));
      let pCosBounce = cosTheta * INV_PI;
      var pSrcBounce = pCosBounce;
      if (alpha > 0.0) {
        pSrcBounce = alpha * ppgEvalPdf(pos, wi) + (1.0 - alpha) * pCosBounce;
      }
      if (!nrcFinite(pSrcBounce) || !(pSrcBounce > 0.0)) {
        nrcRecordInvalidPdf();
        if (grisOn) {
          recordInvalidReservoirGICandidate(&r, GI_SAMPLE_SURFACE, currentGrisEpoch);
        } else {
          r.M = r.M + 1u;
        }
        continue;
      }
      let aX = nrcAccumulateSpread(&runningSum, bounceHit.dist, pSrcBounce, cosArrive);
      // The cache key omits IOR/transmission; glass stays on its dedicated path.
      if (nrcShouldTerminateIntoCache(aX, a0, nrcCfg.spreadC) && xsTransmission <= 0.3) {
        let xsAlbedo = xsPayload.albedo;
        // A6 — real mapped payload roughness (was hardcoded 1.0). The
        // diffuse-default invariant still yields rough≈0.85 for materials
        // without authored roughness, while mapped glossy/metal suffixes now
        // feed their actual roughness to the NRC encoding.
        let xsRough = xsPayload.rough;
        // Query the cache for outgoing radiance toward the visible point
        // (view dir at xs is −wi, the incident bounce direction reversed).
        Lo = select(
          ddgiLo,
          nrcQueryRadiance(xs, ns, -wi, xsRough, xsAlbedo),
          nrcCanSubstitute && !grisOn,
        );
        // Save the first qualifying candidate's input and its pre-cache suffix
        // target together. Reservoir selection below does not alter this pair.
        if (!nrcFired && nrcTeacherEligible) {
          nrcFired = true;
          nrcTrackXs = xs;
          nrcTrackNs = ns;
          nrcTrackWi = wi;
          nrcTrackRough = xsRough;
          nrcTrackAlbedo = xsAlbedo;
          // The teacher uses a private stream so tracing does not perturb RIS.
          // Its target is independent of both DDGI and the cache prediction.
          var teacherRng = pcgInit(
            pixelIdxGi ^ (i * 0x9e3779b9u),
            ubo.frameSeed ^ 0x6e726374u,
            bounceHit.indices.w ^ 0xa511e9b3u,
          );
          nrcTrackTarget = nrcTraceIndependentSuffix(
            bounceHit,
            xs,
            -wi,
            &teacherRng,
          );
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

    var candidateVisibility: f32 = 1.0;
    var pHat: f32;
    if (grisOn) {
      var tMax = 1e20;
      if (sampleKind == GI_SAMPLE_SURFACE) {
        tMax = max(0.0, length(xs - pos) - 2e-3);
      }
      let shadowTint = traceSceneAlphaTintTransmittanceTextured(
        ubo.bvhMode, ubo.tlasNodeCount,

        pos + geoNormal * NORMAL_BIAS_GI, wi, tMax, ubo.triIntersectEpsilon,
        bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
      );
      candidateVisibility = clamp(luminance(shadowTint), 0.0, 1.0);
      pHat = luminance(Lo) * cosTheta * INV_PI * candidateVisibility;
    } else {
      pHat = restir_gi_receiver_phat_from_payload(
        pos,
        normal,
        receiverClearcoatNormal,
        receiverWo,
        receiverPayload,
        xs,
        Lo,
      );
    }
      let invalidPHat = !nrcFinite(pHat) || !(pHat > 0.0) || !nrcFinite(candidateVisibility) || !(candidateVisibility > 0.0);
      if (invalidPHat) { nrcRecordInvalidPdf(); }
      if (invalidPHat) {
        if (grisOn) {
          recordInvalidReservoirGICandidate(&r, sampleKind, currentGrisEpoch);
        } else {
          r.M = r.M + 1u;
        }
        continue;
      }

      var pSrc: f32;
      if (alpha > 0.0) {
        let pCos = cosTheta * INV_PI;
        let pGuide = ppgEvalPdf(pos, wi);
        pSrc = alpha * pGuide + (1.0 - alpha) * pCos;
      } else {
        pSrc = cosTheta * INV_PI;
      }
      if (!nrcFinite(pSrc) || !(pSrc > 0.0)) {
        nrcRecordInvalidPdf();
        if (grisOn) {
          recordInvalidReservoirGICandidate(&r, sampleKind, currentGrisEpoch);
        } else {
          r.M = r.M + 1u;
        }
        continue;
      }
      let w = pHat / pSrc;
      if (!nrcFinite(w) || !(w > 0.0) || !nrcFinite(r.w_sum) || r.w_sum < 0.0 || w > 3.402823466e38 - r.w_sum) {
        nrcRecordInvalidPdf();
        if (grisOn) {
          recordInvalidReservoirGICandidate(&r, sampleKind, currentGrisEpoch);
        } else {
          r.M = r.M + 1u;
        }
        continue;
      }
    if (grisOn) {
      updateReservoirGIWithMetadata(
        &r, xs, ns, Lo, sampleKind, wi,
        pHat, candidateVisibility, currentGrisEpoch, w, &rng,
      );
    } else {
      updateReservoirGI(&r, xs, ns, Lo, w, &rng);
    }
  }

    if (grisOn) {
      if (nrcFinite(r.nativePHat) && r.nativePHat > 0.0 && nrcFinite(r.w_sum) && r.w_sum >= 0.0) {
        finaliseGIReservoirWFromPHat(&r, ubo.restirGiWCap, false, r.nativePHat);
      } else {
        nrcRecordInvalidPdf();
        r.W = 0.0;
        r.w_sum = 0.0;
      }
    }

  if (!grisOn) {
    if (r.M > 0u && r.w_sum > 0.0) {
      let toS = r.xs - r.xv;
      let distS = length(toS);
      if (distS > 1e-4) {
        let wiZ = toS / distS;
        let shadowOrig = r.xv + r.nv * NORMAL_BIAS_GI;
        let shadowTint = traceSceneAlphaTintTransmittanceTextured(
          ubo.bvhMode, ubo.tlasNodeCount,

          shadowOrig, wiZ, distS - 2e-3, ubo.triIntersectEpsilon,
          bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
        );
        let shadowT = clamp(luminance(shadowTint), 0.0, 1.0);
        if (!nrcFinite(shadowT) || !(shadowT > 0.0)) {
          r.w_sum = 0.0;
          r.W = 0.0;
        } else {
          let pHatZ = restir_gi_receiver_phat_from_payload(
            pos,
            normal,
            receiverClearcoatNormal,
            receiverWo,
            receiverPayload,
            r.xs,
            r.Lo,
          );
          if (!nrcFinite(pHatZ) || !(pHatZ > 0.0)) {
            nrcRecordInvalidPdf();
            r.W = 0.0;
            r.w_sum = 0.0;
          } else {
            r.w_sum = r.w_sum * shadowT;
            finaliseGIReservoirWFromPHat(&r, ubo.restirGiWCap, false, pHatZ);
          }
        }
      } else {
        r.W = 0.0;
        r.w_sum = 0.0;
      }
    }
  }

  // Write one matched input/teacher record if any candidate crossed the spread
  // threshold. A zero teacher target remains valid dark-surface data; the host
  // identifies unwritten slots from the encoded input, not from target RGB.
  if (nrcFired) {
    // The bounded block sampler assigns at most one eligible pixel to each slot;
    // nrcWriteRecord retains its atomic claim as a defensive invariant.
    nrcWriteRecord(
      nrcTeacherSlot,
      nrcTrackXs, nrcTrackNs, -nrcTrackWi,
      nrcTrackRough, nrcTrackAlbedo,
      nrcTrackTarget,
    );
  }

  // GRIS Phase-0 reconnection-shift cache (Lin et al. 2022 §5) — written, read
  // by no pass in Phase 0.  Shared with risGi.wgsl via refreshGrisMetadata.
  refreshGrisMetadata(&r);

  storeReservoirGI_rw(pixelIdxGi, r);
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
   * reservoirGi / sharedPrimitives / materialDecode / materialAtlas /
   * restirGiMaterial / cameraRays / ddgiSample / ppgPdf) so the @group(0..3) closure is identical; the only structural delta
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
  // a storage-buffer helper here.
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
    requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'materialDecode', 'materialAtlas', 'surfaceTextures', 'restirGiMaterial', 'cameraRays', 'ddgiGridUbo', 'ppgPdf', 'environmentSample', 'emitterLeAtXi'],
  };
}
