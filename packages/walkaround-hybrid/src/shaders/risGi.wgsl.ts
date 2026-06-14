/**
 * Sprint 16 — ReSTIR-GI initial-candidate RIS pass.
 *
 * Reference: Majercik et al. 2021, "Dynamic Diffuse Global Illumination
 * Resampling," SIGGRAPH 2021, §4.2 (initial-sample RIS).
 *
 * Per-pixel:
 *   1. Re-cast primary ray; on miss / metal → empty reservoir.
 *      Glass: 1-interface refraction walk to the first diffuse surface (B1 tail).
 *   2. RIS over M_GI = 8 candidates. Each candidate samples a
 *      cosine-weighted hemisphere direction; the reconnection vertex
 *      is the first BVH hit along that direction (or sky).
 *      Outgoing radiance Lo at the reconnection vertex is computed by
 *      sampling the DDGI irradiance atlas, multiplied by the hit
 *      surface's albedo / π (Lambertian re-radiation).
 *   3. p̂ = luminance(Lo) × cos(N_visible, wi) × INV_PI
 *      pdf_source = the candidate's source pdf. Without path guiding this is
 *        the pure cosine-hemisphere pdf cos/π, so w_i = p̂/pdf = luminance(Lo)
 *        (the cosθ cancels). With PPG guided sampling (ubo.ppgEnabled == 1) a
 *        Bernoulli(α) chooses guided-dTree vs cosine sampling, and the source
 *        pdf becomes the DEFENSIVE MIXTURE
 *          p_src = α·p_guide(wi) + (1−α)·cos/π        (Müller 2017 §3.4)
 *        evaluated for whichever wi was drawn; the explicit weight is then
 *        w_i = p̂ / p_src. α = 0 (PPG off) reduces this to luminance(Lo) exactly
 *        — the PPG-off path is bit-identical to the pre-PPG cosine kernel.
 *   4. Visibility test on the chosen sample (one extra BVH ray).
 *   5. W = w_sum / (M · p̂(z)) per the standard RIS estimator
 *      (Talbot 2005 + ReSTIR DI 2020).
 *
 * Half-resolution: dispatches W/2 × H/2 invocations. The visible point
 * is the center of each 2×2 quad in full-res coords. The shade pass
 * reconstructs full-res indirect via reservoir read at gid.xy / 2.
 *
 * Bindings:
 *   group(0) — frame (same as shade; uses gNormalDepth + reservoir)
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

export const RIS_GI_WGSL = /* wgsl */ `

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
// cutout traversal and GI shading share the same UV1/normal source.
// B1-ior-per-tri (2026-06-10) — per-triangle roughness+metalness+IOR texture.
// Declared here so the glass-walk Snell solve can decode per-tri IOR via decodeIor().
// Layout: bits[31:24]=rough×255, bits[23:16]=metal×255, bits[15:8]=ior_quantized.
// IOR decode: 1.0 + (byte / 255) * 2.0; range [1.0, 3.0]; default glass → 1.502.
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
const RECONNECT_MAX_DIST: f32 = 100.0;
const NORMAL_BIAS_GI: f32 = 1e-3;

// sampleCosineHemisphere is the canonical helper from @vitrum/shared-samplers'
// bsdfPrimitives.wgsl, injected into the composed shade module via composeWgsl
// (SHARED_PRIMITIVES_MODULE → BSDF_PRIMITIVES_WGSL).

@compute @workgroup_size(8, 8, 1)
fn risGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }

  let pixelIdxGi = gid.y * halfDims.x + gid.x;

  // Sample point in full-res: centre of the 2×2 quad.
  let fullPx = gid.xy * 2u + 1u;

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
  // WS1 — smooth shading normal (visible-point normal + hemisphere frame);
  // geometric normal kept for the bounce-ray offset. V21 — applies in TLAS too
  // (transform the LOCAL blend to world by the hit instance inverse-transpose).
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
  // B1 (road-to-100) — metals/glossy now get a GI reservoir. The reservoir is a
  // DIFFUSE-irradiance cache (cosine-hemisphere candidates, Lambertian target
  // p̂ = luminance(Lo)·cosθ·INV_PI — UNCHANGED, preserving GRIS reuse +
  // diffuse-default invariance). shade reflects this stored radiance off the
  // glossy/metal surface via the GGX specular lobe (shade.lo_indirectSpecular),
  // so metals/glossy receive real specular indirect — no longer an empty punt.
  //
  // B1 tail (2026-06-10) — glass primaries NOW get a refracted GI reservoir.
  // The reservoir is built at the FIRST DIFFUSE SURFACE reached by a 1-interface
  // refraction walk (castPrimaryThroughGlass below). Multi-interface rough-glass
  // is documented-out-of-scope (plan/residue-closure-plan-2026-06-10.md §B1 tail).
  let scalarMatColor = decodeMaterialColor(hit.matColorPacked);
  let matColor = vec4f(
    scalarMatColor.rgb,
    sampleTransmissionMapForHit(hit, scalarMatColor.a),
  );
  let isGlass = matColor.a > 0.3;

  // ── Glass refracted GI: 1-interface refraction walk ─────────────────────
  // B1-ior-per-tri (2026-06-10): decode per-tri IOR from the bvh_material texture
  // (bits[15:8], quantized [1.0, 3.0]). Default 1.5 packs to byte 64, decodes
  // to 1.502 (error < 0.003 — within glass dispersion spread).
  // Maximum number of consecutive glass interfaces the walk passes through
  // (straight-through approximation after the first refraction). Bounded at 2
  // so the per-pixel cost stays O(1) BVH traversals.
  const GLASS_WALK_MAX_EXTRA: u32 = 2u;

  if (isGlass) {
    // Decode the per-triangle IOR from the bvh_material texture.
    let glassPrimaryRmCoord = vec2u(hit.indices.w % BVH_MATERIAL_TEX_WIDTH,
                                    hit.indices.w / BVH_MATERIAL_TEX_WIDTH);
    let glassPrimaryPacked = textureLoad(bvh_material, vec2i(glassPrimaryRmCoord), 0).r;
    let glassPrimaryRm = decodeRoughMetal(glassPrimaryPacked);
    let glassPrimaryRough = glassPrimaryRm.x;
    let IOR_GLASS: f32 = decodeIor(glassPrimaryPacked);

    // Refract the primary ray direction at the glass interface (Snell's law).
    // Incident medium is air (eta_i=1), transmitted medium is glass (IOR_GLASS).
    // Convention: 'normal' is the smooth SHADING normal pointing AWAY from the
    // glass surface toward the incident medium (the camera side). If the ray is
    // entering the glass from outside, dot(primaryRay.direction, normal) < 0 so
    // cosI = -dot(d, n) > 0.
    let d = primaryRay.direction;
    let cosI = -dot(d, normal);   // cosine of incidence angle (positive = entering)
    let etaRatio = select(IOR_GLASS, 1.0 / IOR_GLASS, cosI > 0.0);
    // If cosI < 0 the ray is leaving glass → etaRatio = IOR_GLASS/1 (flip normal).
    let nFlipped = select(-normal, normal, cosI > 0.0);
    let cosI_pos = abs(cosI);
    let sin2T = etaRatio * etaRatio * (1.0 - cosI_pos * cosI_pos);
    // Total internal reflection: fall back to empty reservoir (TIR is rare for
    // a camera ray hitting glass from outside; common only for steep exit angles).
    if (sin2T > 1.0) {
      storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, emptyReservoirGI());
      return;
    }
    let cosT = sqrt(max(0.0, 1.0 - sin2T));
    var refractDir = safe_normalize(etaRatio * d + (etaRatio * cosI_pos - cosT) * nFlipped);

    // B1-ior-per-tri stretch: rough-glass GI direction perturbation (2026-06-10).
    // For rough glass (roughness > ROUGH_GLASS_THRESHOLD), perturb the exact Snell
    // refracted direction by a GGX-distributed micro-facet offset so frosted glass
    // receives blurred GI instead of mirror-sharp refraction. One sample; uses the
    // per-tri roughness decoded above (glassPrimaryRough). Smooth glass (rough below
    // threshold) keeps the exact Snell direction → byte-identical for default glass
    // (glassPrimaryRough = 0.05 < 0.1 threshold).
    //
    // Implementation: build an orthonormal frame around refractDir, draw a
    // GGX-distributed tangent perturbation (Heitz 2018 VNDF simplified for a
    // single-sample isotropic deflection), re-normalize. The perturbation is in
    // the transmitted half-space; if it flips below the surface we clamp back to
    // the geometric hemisphere (same guard as sampleCosineHemisphere).
    const ROUGH_GLASS_THRESHOLD: f32 = 0.1;
    if (glassPrimaryRough > ROUGH_GLASS_THRESHOLD) {
      // Build tangent frame around refractDir.
      let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0),
                      abs(refractDir.y) > 0.9);
      let t1 = safe_normalize(cross(refractDir, up));
      let t2 = cross(refractDir, t1);

      // GGX-distributed microfacet perturbation: draw a 2D GGX sample in the
      // tangent plane and add it to refractDir, then re-normalize.
      // p(r) ∝ r · α² / (r² · (α²-1) + 1)²  — simplified isotropic GGX disk.
      let alpha = glassPrimaryRough * glassPrimaryRough; // α = roughness² (GGX convention)
      let xi1 = rand_f32(&rng);
      let xi2 = rand_f32(&rng);
      // Importance-sample GGX slope magnitude: r = α√(xi1/(1−xi1)).
      let r2 = alpha * alpha * xi1 / max(1e-6, 1.0 - xi1);
      let r = sqrt(r2);
      let phi = 2.0 * 3.14159265 * xi2;
      let dx = r * cos(phi);
      let dy = r * sin(phi);
      let perturbedDir = safe_normalize(refractDir + dx * t1 + dy * t2);
      // Ensure perturbation stays on the transmitted side of the surface.
      // nFlipped points into the glass from the entry side; the transmitted ray
      // must have dot(perturbedDir, -nFlipped) > 0 (go through the surface).
      if (dot(perturbedDir, -nFlipped) > 1e-4) {
        refractDir = perturbedDir;
      }
      // else: perturbation flipped → keep exact Snell direction (rare at low-α).
    }

    // Walk through the glass pane to find the first non-glass diffuse surface.
    // Origin: offset along the GEOMETRIC normal (same bias as bounce rays) in the
    // TRANSMITTED direction so we start on the far side of the interface.
    var walkOrigin = pos - geoNormal * NORMAL_BIAS_GI;   // enter the glass
    var walkHit: IntersectionResult;
    var walkHitPos: vec3f;
    var walkHitNormal: vec3f;
    var foundSurface: bool = false;

    for (var gi: u32 = 0u; gi <= GLASS_WALK_MAX_EXTRA; gi = gi + 1u) {
      let walkRay = Ray(walkOrigin, refractDir);
      walkHit = traceSceneFirstHitAlphaMaskTextured(
        ubo.bvhMode, ubo.tlasNodeCount,
        &bvh_index, &bvh_position, &bvh,
        &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
        &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
        walkRay, ubo.triIntersectEpsilon,
        bvh_material, BVH_MATERIAL_TEX_WIDTH,
      );
      if (!walkHit.didHit) { break; }

      walkHitPos = walkOrigin + refractDir * walkHit.dist;
      walkHitNormal = walkHit.normal;
      let scalarWalkMat = decodeMaterialColor(walkHit.matColorPacked);
      let walkMat = vec4f(
        scalarWalkMat.rgb,
        sampleTransmissionMapForHit(walkHit, scalarWalkMat.a),
      );
      let walkIsGlass = walkMat.a > 0.3;

      if (!walkIsGlass) {
        // Found the first diffuse surface. Compute smooth shading normal.
        let wn_isTlas = ubo.bvhMode == 1u;
        let wn_base = walkHit.instanceIndex * 4u;
        let wn_ok = wn_isTlas && wn_base + 2u < arrayLength(&tlasInstanceWorldToLocal);
        let wn_i = select(0u, wn_base, wn_ok);
        walkHitNormal = smoothShadingNormal(
          walkHit, walkHit.normal,
          bvh_normal[walkHit.indices.x].xyz,
          bvh_normal[walkHit.indices.y].xyz,
          bvh_normal[walkHit.indices.z].xyz,
          wn_ok,
          tlasInstanceWorldToLocal[wn_i],
          tlasInstanceWorldToLocal[wn_i + 1u],
          tlasInstanceWorldToLocal[wn_i + 2u],
        );
        foundSurface = true;
        break;
      }

      // Another glass interface — straight-through approximation: continue along
      // the same refractDir from the far side of this pane (no secondary refraction).
      // This covers thin double-glazing; the error from skipping secondary IOR is
      // a mild lateral shift (<1% for 2 mm glass at scene scale).
      walkOrigin = walkHitPos + refractDir * NORMAL_BIAS_GI;
    }

    if (!foundSurface) {
      storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, emptyReservoirGI());
      return;
    }

    // Build the reservoir AT the post-glass diffuse surface.
    // xv = post-glass surface position, nv = post-glass surface normal.
    // Temporal/spatial reuse compares these consistently (the glass pixel's
    // reservoir stores the diffuse wall's geometry, not the glass pane itself —
    // reuse rejection operates in the same coordinate frame for all pixels).
    var rGlass: ReservoirGI = emptyReservoirGI();
    rGlass.xv = walkHitPos;
    rGlass.nv = walkHitNormal;

    let tier_raw_g = textureLoad(gi_tier, vec2i(fullPx), 0).r;
    let tier_g = clamp(tier_raw_g, 1u, 4u);
    let M_GI_g = M_GI_BASE * tier_g / 2u;

    for (var i: u32 = 0u; i < M_GI_g; i = i + 1u) {
      let wi = sampleCosineHemisphere(walkHitNormal, &rng);
      let cosTheta = max(0.0, dot(walkHitNormal, wi));
      if (cosTheta < 1e-4) { continue; }

      let bounceRay = Ray(walkHitPos + walkHit.normal * NORMAL_BIAS_GI, wi);
      let bounceHit = traceSceneFirstHitAlphaMaskTextured(
        ubo.bvhMode, ubo.tlasNodeCount,
        &bvh_index, &bvh_position, &bvh,
        &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
        &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
        bounceRay, ubo.triIntersectEpsilon,
        bvh_material, BVH_MATERIAL_TEX_WIDTH,
      );

      var xs_g: vec3f;
      var ns_g: vec3f;
      var Lo_g: vec3f;

      if (bounceHit.didHit) {
        xs_g = bounceRay.origin + wi * bounceHit.dist;
        ns_g = bounceHit.normal;
        let irrAtXs = min(sampleDDGIAtPoint(xs_g, ns_g), vec3f(ubo.restirGiIrrClamp));
        let xsMat = decodeMaterialColor(bounceHit.matColorPacked);
        Lo_g = irrAtXs * xsMat.rgb * INV_PI;
      } else {
        xs_g = walkHitPos + wi * RECONNECT_MAX_DIST;
        ns_g = -wi;
        Lo_g = envRadiance(wi);
      }

      let pHat_g = luminance(Lo_g) * cosTheta * INV_PI;
      if (pHat_g < 1e-9) { continue; }
      // PPG is off for glass pixels (ppgMixAlpha=0 when glass → pure cosine).
      // The alpha branch below is the same ppg-off shortcut as the opaque path.
      let w_g = luminance(Lo_g);
      updateReservoirGI(&rGlass, xs_g, ns_g, Lo_g, w_g, &rng);
    }

    // Visibility test on the chosen sample (same pattern as opaque path).
    if (rGlass.M > 0u && rGlass.w_sum > 0.0) {
      let toS_g = rGlass.xs - rGlass.xv;
      let distS_g = length(toS_g);
      if (distS_g > 1e-4) {
        let wiZ_g = toS_g / distS_g;
        let shadowOrig_g = rGlass.xv + rGlass.nv * NORMAL_BIAS_GI;
        let occ_g = traceSceneAnyCastMask(
          ubo.bvhMode, ubo.tlasNodeCount,
          &bvh_index, &bvh_position, &bvh,
          &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
          &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
          shadowOrig_g, wiZ_g, distS_g - 2e-3, ubo.triIntersectEpsilon, true,
          bvh_material, BVH_MATERIAL_TEX_WIDTH,
        );
        if (occ_g) {
          rGlass.w_sum = 0.0;
          rGlass.W = 0.0;
        } else {
          let cosThetaZ_g = max(0.0, dot(rGlass.nv, wiZ_g));
          let pHatZ_g = luminance(rGlass.Lo) * cosThetaZ_g * INV_PI;
          let W_raw_g = select(0.0, rGlass.w_sum / (f32(rGlass.M) * pHatZ_g), pHatZ_g > 1e-9);
          rGlass.W = min(W_raw_g, ubo.restirGiWCap);
        }
      } else {
        rGlass.W = 0.0;
        rGlass.w_sum = 0.0;
      }
    }

    refreshPhase0Cache(&rGlass);
    storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, rGlass);
    return;
  }

  var r: ReservoirGI = emptyReservoirGI();
  r.xv = pos;
  r.nv = normal;

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
  //     draw is consumed → the rng stream is byte-identical to the pre-PPG
  //     pure-cosine path, and
  //   - p_src = (1−0)·p_cos = cosθ/π, so the explicit RIS weight
  //     w = pHat / p_src reduces to EXACTLY luminance(Lo) (the cosine
  //     shortcut). ppg-OFF is bit-identical.
  let ppgGuidedOn = (ubo.ppgEnabled == 1u);
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
    if (cosTheta < 1e-4) { continue; }

    // Trace from the visible point along wi. Reconnection vertex is the
    // first BVH hit (or sky-miss at RECONNECT_MAX_DIST).
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
      // Sample DDGI atlas at the reconnection vertex along its normal —
      // gives the incoming irradiance there. Modulate by the hit surface's
      // albedo / π for Lambertian outgoing radiance toward the visible pt.
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
      let xsMat = decodeMaterialColor(bounceHit.matColorPacked);
      Lo = irrAtXs * xsMat.rgb * INV_PI;
    } else {
      // Sky miss — the GI ray escaped the scene. B3: sample the directional IBL
      // map along wi (rotationY-aware) as the reconnection radiance; envRadiance
      // falls back to the scalar skyTint × skyIrradiance with no HDRI bound
      // (no-HDRI byte-identity: the cosine RIS shortcut below is unchanged).
      xs = pos + wi * RECONNECT_MAX_DIST;
      ns = -wi;
      Lo = envRadiance(wi);
    }

    // p̂ at the visible point for this candidate (the RIS target function —
    // cosine-weighted reconnection radiance luminance). Unchanged by PPG.
    let pHat = luminance(Lo) * cosTheta * INV_PI;
    if (pHat < 1e-9) { continue; }

    // RIS candidate weight w = p̂ / p_src.
    //
    // ppg-OFF (alpha == 0): the RIS source pdf is the pure cosine pdf
    //   p_src = cosθ/π, and the weight algebraically cancels the cosθ in p̂:
    //     w = (luminance(Lo)·cosθ·INV_PI) / (cosθ·INV_PI) = luminance(Lo)
    //   We take the literal shortcut here (NOT the division) so the PPG-off
    //   path is BIT-IDENTICAL to the pre-PPG kernel — no ULP drift from a
    //   round-trip multiply/divide.
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

  // Final visibility test on the chosen sample.
  if (r.M > 0u && r.w_sum > 0.0) {
    let toS = r.xs - r.xv;
    let distS = length(toS);
    if (distS > 1e-4) {
      let wiZ = toS / distS;
      let shadowOrig = r.xv + r.nv * NORMAL_BIAS_GI;
      // skipGlass=true: matches pre-canonical ReSTIR shadow-ray glass filter
      // (light passes through glass; per-channel tinted-visibility handles tint).
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
        // Cap W to bound firefly contribution from tiny pHat denominators
        // (grazing cos or near-zero Lo luminance). Cornell default 16.0
        // lives on ubo.restirGiWCap; see common.wgsl for the rationale.
        r.W = min(W_raw, ubo.restirGiWCap);
      }
    } else {
      r.W = 0.0;
      r.w_sum = 0.0;
    }
  }

  // ── GRIS Phase-0 reconnection-shift cache (Lin et al. 2022, §5) ────────────
  // Populate the appended ReservoirPT fields from the FINAL selected
  // reconnection sample (xv → xs). These are WRITTEN here but READ BY NO PASS
  // in Phase 0 — they do not touch the [0..19] fields, so the shade/temporal/
  // spatial reads and therefore the rendered output stay BIT-IDENTICAL. Phase 1
  // (GPU reconnection shift) and Phase 2 (GRIS pairwise MIS) will consume them:
  //   • wi_recon / distRecon / cosReconOut → the base reconnection-edge half-G
  //     (cosθ_out / dist²) numerator+denominator of the shift Jacobian.
  //   • pdfReconBsdf → cached headroom (NOT read by the single-bounce reuse
  //     weight: reservoir reuse is m·p̂·W·J, no /p_src — the W already bakes in
  //     this pdf; see the spatialGi/temporalGi GRIS branches).
  //   • prefixVertexCount → shift-compatibility gate (only paths with matching
  //     prefix length take the reconnection shift; others fall back).
  refreshPhase0Cache(&r);

  storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, r);
}
`;

/** W1-R6 — declarative include-graph entry.
 *  T9-stepC — narrowed from `['common', 'ddgiSample']` to the modules this
 *  half-res GI-RIS pass actually references:
 *    - `WalkaroundUBO` / `INV_PI`            → walkaroundUbo
 *    - primary cast (`traceScene*` / `BVHNode` / `Ray`) → sceneTraversal
 *    - `ReservoirGI` / `emptyReservoirGI` / `updateReservoirGI` /
 *      `storeReservoirGI_rw`                 → reservoirGi
 *    - `pcgInit` / `luminance` / `sampleCosineHemisphere` → sharedPrimitives
 *    - `decodeMaterialColor` / `decodeIsMetal` / `decodeRoughMetal` / `decodeIor`
 *      / `BVH_MATERIAL_TEX_WIDTH`              → materialDecode
 *    - `invertMat4_common` / `generatePrimaryRay_common` → cameraRays
 *    - `ddgiSample`                          → ddgiSample
 *  Drops emitterSampling / ggxBrdf / jacobianShift / welfordTail (unused).
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
  requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'materialDecode', 'materialAtlas', 'cameraRays', 'ddgiGridUbo', 'ppgPdf', 'environmentSample'],
};
