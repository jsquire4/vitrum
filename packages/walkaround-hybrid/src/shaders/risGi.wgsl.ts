/**
 * Sprint 16 — ReSTIR-GI initial-candidate RIS pass.
 *
 * Reference: Majercik et al. 2021, "Dynamic Diffuse Global Illumination
 * Resampling," SIGGRAPH 2021, §4.2 (initial-sample RIS).
 *
 * Per-pixel:
 *   1. Re-cast primary ray. Misses produce an empty reservoir; glass walks up
 *      to four dielectric interfaces to the first diffuse receiver, and
 *      rich receivers keep their material-aware lobe target below.
 *   2. RIS over M_GI = 8 candidates. Each candidate samples a
 *      cosine-weighted hemisphere direction; the reconnection vertex
 *      is the first BVH hit along that direction (or sky).
 *      Outgoing radiance Lo at the reconnection vertex is computed by
 *      sampling the DDGI irradiance atlas, then applying the hit surface's
 *      mapped material response: diffuse albedo / π for ordinary suffixes,
 *      or the extension-aware GGX/clearcoat/sheen proxy for rich suffixes.
 *   3. p̂ = luminance(receiver contribution). Diffuse defaults are
 *      luminance(Lo) × cos(N_visible, wi) × INV_PI; rich receivers add the
 *      glossy/clearcoat/sheen lobes that shade will consume.
 *      pdf_source = the candidate's source pdf. Without path guiding this is
 *        the pure cosine-hemisphere pdf cos/π, so diffuse-default
 *        w_i = p̂/pdf = luminance(Lo)
 *        (the cosθ cancels). With PPG guided sampling (ubo.ppgEnabled == 1) a
 *        Bernoulli(α) chooses guided-dTree vs cosine sampling, and the source
 *        pdf becomes the DEFENSIVE MIXTURE
 *          p_src = α·p_guide(wi) + (1−α)·cos/π        (Müller 2017 §3.4)
 *        evaluated for whichever wi was drawn; the explicit weight is then
 *        w_i = p̂ / p_src. α = 0 (PPG off) still reduces to luminance(Lo) for
 *        default diffuse receivers, while rich receivers keep their lobe target.
 *   4. Visibility test on the chosen sample (one extra BVH ray).
 *   5. W = w_sum / (M · p̂(z)) per the standard RIS estimator
 *      (Talbot 2005 + ReSTIR DI 2020).
 *
 * Half-resolution: dispatches W/2 × H/2 invocations. The visible point
 * is the center of each 2×2 quad in full-res coords. The shade pass
 * reconstructs full-res indirect via reservoir read at gid.xy / 2.
 *
 * Bindings:
 *   group(0) — compact GI frame group (reservoir only; the pass re-casts its
 *              primary ray and therefore does not bind the raster G-buffer)
 *   group(1) — scene (BVH + emitters; reuse existing layout)
 *   group(2) — ubo (camera matrices, frameSeed, aoFullTexture)
 *   group(3) — hybrid (DDGI atlas + sampler + grid params at bindings 0-3;
 *              RC cascade-0 + params at 4-5; W9 PPG sTree/dTree/dTreeOffsets
 *              storage buffers at 6-8, declared by the `ppgPdf` module and
 *              read only when ubo.ppgEnabled == 1)
 *   The GI reservoir buffer is bound as @group(0) @binding(11), added
 *   to the frame BGL by the Sprint 16 pipeline machinery.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import {
  RIS_GI_GLASS_TRANSPORT_PREFIX_WGSL,
  RIS_GI_GLASS_RESERVOIR_LOOP_WGSL,
  RIS_GI_GLASS_VISIBILITY_TAIL_WGSL,
} from './risGiGlassWalk.wgsl.js';
import { reservoirGiAccessorsWgsl } from './reservoirGi.wgsl.js';

export const RIS_GI_WGSL = /* wgsl */ `

@group(0) @binding(11) var<storage, read_write> reservoirGiCurrent: array<u32>;

${reservoirGiAccessorsWgsl({ storeReadWriteBinding: 'reservoirGiCurrent' })}

@group(1) @binding(5) var bvh_beer: texture_2d<u32>;
// WS1 (2026-05-29) — bvh_normal is declared by materialAtlas.wgsl so alpha
// cutout traversal and GI shading share the same UV1/normal source.
// B1-ior-per-tri (2026-06-10) — per-triangle roughness+metalness+IOR texture.
// Declared here so the glass-walk Snell solve can decode per-tri IOR via decodeIor().
// Layout: bits[31:24]=rough×255, bits[23:16]=metal×255, bits[15:8]=ior_quantized.
// IOR decode: 1.0 + ((byte - 1) / 254) * 2.0; byte 0 is the infinite-IOR sentinel.
@group(1) @binding(14) var bvh_material: texture_2d<u32>;

@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;
// Sprint 9 — adaptive sampling tier (r32uint, full-res). 1 = low variance,
// 2 = medium, 4 = high. Read at the centre of each half-res 2×2 quad to
// scale the RIS candidate count: high-variance pixels get more candidates
// where they're needed, low-variance pixels save the compute.
@group(2) @binding(2) var gi_tier: texture_2d<u32>;

// D5.1+D5.2: DDGIGridUBO struct, @group(3) @binding(3) ddgiGrid UBO, and
// sampleDDGIAtPoint are now provided by the shared ddgiGridUbo module.
@group(3) @binding(0) var ddgiIrradiance: texture_2d<f32>;
@group(3) @binding(1) var ddgiVisibility: texture_2d<f32>;
@group(3) @binding(2) var ddgiSampler:    sampler;

// Base RIS-GI candidate count. Scaled per pixel by adaptive-sampling tier:
// tier=1 → M_GI_eff = 4; tier=2 → 8 (default); tier=4 → 16.
const M_GI_BASE: u32 = 8u;

// sampleCosineHemisphere is the canonical helper from @vitrum/shared-samplers'
// bsdfPrimitives.wgsl, injected into the composed shade module via composeWgsl
// (SHARED_PRIMITIVES_MODULE → BSDF_PRIMITIVES_WGSL).

@compute @workgroup_size(8, 8, 1)
fn risGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = restirGiDimensions();
  if (any(gid.xy >= halfDims)) { return; }

  let pixelIdxGi = gid.y * halfDims.x + gid.x;

  // Sample point in full-res: centre of the 2×2 quad.
  let fullPx = restirGiFullPixel(gid.xy);

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0xA5A5u),
    gid.y ^ (ubo.frameSeed * 0x5A5Au),
    ubo.frameSeed ^ 0xC1A2u,
  );

  // Re-cast primary ray to find the visible surface.
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
  // WS1 — smooth shading normal (visible-point normal + hemisphere frame);
  // geometric normal kept for the bounce-ray offset. V21 — applies in TLAS too
  // (transform the LOCAL blend to world by the hit instance inverse-transpose).
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
  // B1 (road-to-100) — metals/glossy now get a GI reservoir. The reservoir uses
  // cosine-hemisphere candidates and the visible-point receiver-lobe target.
  // Diffuse-default p̂ remains luminance(Lo)·cosθ·INV_PI; rich receivers add
  // their specular/clearcoat/sheen lobes. The suffix Lo is material-aware: ordinary
  // suffixes are DDGI irradiance * mapped albedo / π, while rich suffixes route
  // through the extension-aware GGX/clearcoat/sheen proxy. shade then reflects
  // this stored radiance off the receiver via the visible material's indirect
  // lobes (for metals/glossy, shade.lo_indirectSpecular), so the old empty punt
  // is gone without widening the GI reservoir payload.
  //
  // Glass primaries build their reservoir at the first opaque surface reached
  // by the bounded dielectric-prefix walk below. Every crossed interface gets
  // its own Snell/Fresnel solve and medium-stack update; TIR or budget overflow
  // fails closed rather than relabelling a glass exit face as the receiver.
  let scalarMatColor = decodeMaterialColor(hit.matColorPacked);
  let matColor = vec4f(
    scalarMatColor.rgb,
    sampleTransmissionMapForHit(hit, scalarMatColor.a),
  );
  let isGlass = materialHasTransmission(matColor.a);
  let currentGrisEpoch = bitcast<u32>(ubo.sunAngular.y);

${RIS_GI_GLASS_TRANSPORT_PREFIX_WGSL}

    // Build the reservoir AT the post-glass diffuse surface.
    // xv = post-glass surface position, nv = post-glass surface normal.
    // Temporal/spatial reuse compares these consistently (the glass pixel's
    // reservoir stores the diffuse wall's geometry, not the glass pane itself —
    // reuse rejection operates in the same coordinate frame for all pixels).
${RIS_GI_GLASS_RESERVOIR_LOOP_WGSL}

    // Visibility test on the chosen sample (same pattern as opaque path).
${RIS_GI_GLASS_VISIBILITY_TAIL_WGSL}
  }

  let receiverMaterialCoord = vec2u(
    hit.indices.w % BVH_MATERIAL_TEX_WIDTH,
    hit.indices.w / BVH_MATERIAL_TEX_WIDTH,
  );
  let receiverMaterialWord = textureLoad(
    bvh_material,
    vec2i(receiverMaterialCoord),
    0,
  ).r;
  let receiverPayload = sampleRestirDIMaterialPayloadForHit(
    hit,
    smoothNormal,
    normal,
    matColor.rgb,
    receiverMaterialWord,
    safe_normalize(-primaryRay.direction),
  );

  var r: ReservoirGI = emptyReservoirGI();
  r.xv = pos;
  r.nv = normal;
  r.receiverMaterialKey = restir_gi_receiver_domain_key(
    hit.matColorPacked,
    receiverMaterialWord,
    hit.indices.w,
    select(0u, hit.instanceIndex, ubo.bvhMode == 1u),
    receiverPayload,
  );
  r.historyEpoch = currentGrisEpoch;

  // Adaptive-sampling tier read at the full-res quad centre. Clamped to
  // [1,4] in case the sample-budget pass emits a bad/uninitialised value
  // (first frame writes vec4u(2,0,0,0) by default). M_GI scales linearly.
  let tier_raw = textureLoad(gi_tier, vec2i(fullPx), 0).r;
  let tier = clamp(tier_raw, 1u, 4u);
  let M_GI = M_GI_BASE * tier / 2u;

  // ── PPG guided-sampling mixing weight (Müller 2017 §3.4) ──────────────────
  // α is the fraction of RIS candidates drawn from the learned dTree. It is
  // ubo.ppgMixAlpha when PPG guided sampling is live (ubo.ppgEnabled == 1) and
  // EXACTLY 0 otherwise. The host writes ppgEnabled=0 / ppgMixAlpha=0 whenever
  // PPG is off (see uboUpdater.ts), so on the PPG-off path:
  //   - the Bernoulli branch below is gated on alpha > 0.0, so NO extra RNG
  //     draw is consumed → the rng stream stays stable, and
  //   - p_src = (1−0)·p_cos = cosθ/π, so the explicit RIS weight uses the
  //     receiver-lobe target divided by the cosine source pdf. For default
  //     diffuse receivers this algebraically reduces to luminance(Lo); rich
  //     material receivers now guide the reservoir by their actual lobes.
  let ppgGuidedOn = ubo.ppgEnabled == 1u;
  let alpha = select(0.0, ubo.ppgMixAlpha, ppgGuidedOn);

  for (var i: u32 = 0u; i < M_GI; i = i + 1u) {
    // Draw a candidate direction wi. When alpha > 0, flip a Bernoulli(alpha):
    // heads → sample the learned dTree (guided), tails → cosine hemisphere.
    // When alpha == 0 we take the cosine branch WITHOUT consuming a Bernoulli
    // draw, preserving the exact pre-PPG rng sequence (bit-identity).
    var wi: vec3f;
    if (alpha > 0.0) {
      let bern = rand_f32(&rng);
      if (bern < alpha) {
        // Guided: sample ∝ leaf flux from the dTree for this shading cell.
        wi = ppgSampleGuidedDir(pos, &rng);
      } else {
        wi = sampleCosineHemisphere(normal, &rng);
      }
    } else {
      // Cosine-weighted hemisphere candidate (pre-PPG path, bit-identical).
      wi = sampleCosineHemisphere(normal, &rng);
    }
    let cosTheta = max(0.0, dot(normal, wi));
    if (!reservoirGiFinite(cosTheta) || !(cosTheta > 0.0)) {
      recordInvalidReservoirGICandidate(&r, GI_SAMPLE_SURFACE, currentGrisEpoch);
      continue;
    }

    // Trace from the visible point along wi. Reconnection vertex is the
    // first BVH hit (or a scene-scale-relative sky-miss proxy).
    // WS1 — offset the bounce-ray origin along the GEOMETRIC normal.
    let bounceRay = Ray(pos + geoNormal * walkaroundRayOriginBias(), wi);
    let bounceHit = traceSceneFirstHitAlphaMaskTextured(
      ubo.bvhMode, ubo.tlasNodeCount,

      bounceRay, ubo.triIntersectEpsilon,
      bvh_material, BVH_MATERIAL_TEX_WIDTH,
      ubo.frameSeed ^ (i * 0x85ebca6bu) ^ 0x4749424eu,
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
      // Sample DDGI atlas at the reconnection vertex along its normal —
      // gives the incoming irradiance there. Modulate by the hit surface's
      // material response for outgoing radiance toward the visible pt.
      //
      // Defensive cap on the atlas read.  DDGI probes within ~1 spacing of
      // an area light catch Le directly during the probe trace, so atlas
      // reads of 5..8 are possible — but for a Cornell-scale scene with
      // Le=12, legitimate near-light wall irradiance is also in this band,
      // so we can't reject those samples without truncating real indirect.
      // The cap (Cornell-tuned default 5.0, exposed via ubo.restirGiIrrClamp)
      // admits the realistic indirect range while bounding pathological DDGI
      // atlas readings (which would otherwise produce ~10× per-channel spikes
      // in Lo). The previous tighter reject+cap (>2.0 reject, min 1.0) was
      // over-truncating: the magnitude audit showed it was a 5-10× *under*-
      // energizer of the indirect channel.
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
      Lo = xsPayload.Lo;
    } else {
      // Sky miss — the GI ray escaped the scene. B3: sample the directional IBL
      // map along wi (rotationY-aware) as the reconnection radiance; envRadiance
      // falls back to the scalar skyTint × skyIrradiance with no HDRI bound
      // (no-HDRI byte-identity: the cosine RIS shortcut below is unchanged).
      xs = pos + wi * walkaroundReconnectMaxDistance();
      ns = -wi;
      Lo = envRadiance(wi);
    }

    // Evaluate the same mapped receiver lobes that shade will consume. The
    // diffuse-default path still reduces to luminance(Lo) * cos(theta) / pi,
    // while glossy/metal/clearcoat/sheen receivers retain their actual target.
    var candidateVisibility: f32 = 1.0;
    var pHat: f32;
    var tMax = INFINITY;
    if (sampleKind == GI_SAMPLE_SURFACE) {
      tMax = max(0.0, safe_length(xs - pos) - walkaroundRayEndMargin());
    }
    let shadowTint = traceSceneAlphaTintTransmittanceTextured(
      ubo.bvhMode, ubo.tlasNodeCount,

      pos + geoNormal * walkaroundRayOriginBias(), wi, tMax, ubo.triIntersectEpsilon,
      bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
    );
    candidateVisibility = clamp(luminance(shadowTint), 0.0, 1.0);
    var receiverLo = Lo;
    if (sampleKind == GI_SAMPLE_ENVIRONMENT) {
      receiverLo = walkaroundScaleEnvironmentRadiance(
        receiverLo,
        receiverPayload.envMapIntensity,
      );
    }
    pHat = restir_gi_receiver_phat_from_payload(
      pos,
      normal,
      receiverPayload.clearcoatNormal,
      safe_normalize(-primaryRay.direction),
      receiverPayload,
      xs,
      receiverLo,
    ) * candidateVisibility;
    if (!reservoirGiFinite(pHat) || !(pHat > 0.0) || !reservoirGiFinite(candidateVisibility) || !(candidateVisibility > 0.0)) {
      recordInvalidReservoirGICandidate(&r, sampleKind, currentGrisEpoch);
      continue;
    }

    // RIS candidate weight w = p̂ / p_src.
    //
    // ppg-OFF (alpha == 0): the RIS source pdf is the pure cosine pdf
    //   p_src = cosθ/π. Diffuse defaults still reduce to luminance(Lo);
    //   rich-material receivers keep the full p̂ / p_src ratio.
    //
    // ppg-ON (alpha > 0): the RIS source pdf is the DEFENSIVE MIXTURE
    //   (Müller §3.4)   p_src = α·p_guide(wi) + (1−α)·p_cos(wi)
    //   evaluated for WHICHEVER wi was chosen (guided OR cosine). Evaluating
    //   BOTH pdfs for the chosen direction is what keeps the mixture estimator
    //   unbiased — we cannot reuse the cosine shortcut because the cosθ no
    //   longer cancels against a pure-cosine denominator.
    //     p_cos   = cosθ/π                    (cosine-hemisphere solid-angle pdf)
    //     p_guide = ppgEvalPdf(pos, wi)       (dTree solid-angle pdf; mirrors
    //               the CPU dTreePdf in dTree.ts exactly)
    var pSrc: f32;
    if (alpha > 0.0) {
      let pCos = cosTheta * INV_PI;
      let pGuide = ppgEvalPdf(pos, wi);
      pSrc = alpha * pGuide + (1.0 - alpha) * pCos;
    } else {
      pSrc = cosTheta * INV_PI;
    }
    if (!reservoirGiFinite(pSrc) || !(pSrc > 0.0)) {
      recordInvalidReservoirGICandidate(&r, sampleKind, currentGrisEpoch);
      continue;
    }
    let w = pHat / pSrc;
    if (
      !reservoirGiFinite(w) || !(w > 0.0) ||
      !reservoirGiFinite(r.w_sum) || r.w_sum < 0.0 ||
      w > 3.402823466e38 - r.w_sum
    ) {
      recordInvalidReservoirGICandidate(&r, sampleKind, currentGrisEpoch);
      continue;
    }
    updateReservoirGIWithMetadata(
      &r, xs, ns, Lo, sampleKind, wi,
      pHat, candidateVisibility, currentGrisEpoch, w, &rng,
    );
  }

  finaliseGIReservoirWFromPHat(&r, ubo.restirGiWCap, r.nativePHat);

  // ── Generalized-reuse producer metadata (Lin et al. 2022, §5) ────────────
  // The metadata is consumed by the canonical temporal/spatial passes and
  // by the final shading visibility gate. refreshGrisMetadata derives the
  // reconnection direction, distance, outgoing cosine, and prefix count from
  // the selected edge; the winning candidate's sample kind, native target,
  // visibility, and history epoch remain intact for transformed-density reuse.
  refreshGrisMetadata(&r);

  storeReservoirGI_rw(pixelIdxGi, r);
}
`;

/** W1-R6 — declarative include-graph entry.
 *  T9-stepC — narrowed from `['common', 'ddgiSample']` to the modules this
 *  half-res GI-RIS pass actually references:
 *    - `WalkaroundUBO` / `INV_PI`            → walkaroundUbo
 *    - primary cast (`traceScene*` / `BVHNode` / `Ray`) → sceneTraversal
 *    - `ReservoirGI` / `emptyReservoirGI` / `updateReservoirGIWithMetadata` /
 *      `storeReservoirGI_rw`                 → reservoirGi
 *    - `pcgInit` / `luminance` / `sampleCosineHemisphere` → sharedPrimitives
 *    - `decodeMaterialColor` / `decodeIsMetal` / `decodeRoughMetal` / `decodeIor`
 *      / `BVH_MATERIAL_TEX_WIDTH`              → materialDecode
 *    - `invertMat4_common` / `generatePrimaryRay_common` → cameraRays
 *    - `ddgiSample`                          → ddgiSample
 *  Drops emitterSampling / welfordTail (unused).
 *  ReSTIR-GI material parity adds `restirGiMaterial` (normal/bump maps, mapped
 *  base color, rough/metal, and extension-aware suffix radiance).
 *  W9 guided sampling — adds `ppgPdf` (declares the group(3) PPG tree buffers
 *  + provides ppgEvalPdf / ppgSampleGuidedDir). Listed AFTER sharedPrimitives
 *  so `rand_f32` is defined before ppgPdf's source, and after ddgiSample so
 *  the group(3) DDGI bindings (0-3) precede the PPG ones (6-8).
 *  Verified complete by the static ident-resolution gate. */
export const RIS_GI_MODULE: WgslModule = {
  name: 'risGi',
  source: RIS_GI_WGSL,
  // D5.1+D5.2: ddgiSample replaced by ddgiGridUbo (which requires ddgiSample
  // transitively, and adds the DDGIGridUBO struct + binding + sampleDDGIAtPoint).
  requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'materialDecode', 'materialAtlas', 'surfaceTextures', 'restirGiMaterial', 'cameraRays', 'ddgiGridUbo', 'ppgPdf', 'environmentSample'],
};
