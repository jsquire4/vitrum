/**
 * risGiGlassWalk.wgsl.ts — the byte-identical spans of the glass refracted-GI
 * walk shared by `risGi.wgsl.ts` (RIS_GI_WGSL) and `risGiNrc.wgsl.ts`
 * (RIS_GI_NRC_BODY).
 *
 * D8-1 (complexity-sweep 2026-07-20, T4-1): risGiNrc is a near-verbatim copy of
 * risGi. Its CODE differs in only ~4 surgical NRC deltas (an `alpha`→`alphaGlass`
 * rename in the glass GGX perturbation, an NRC spread-footprint preamble, an NRC
 * cache-termination branch replacing `Lo = xsPayload.Lo;`, and an NRC training
 * record). HOWEVER the two bodies' COMMENTS were independently re-authored and
 * diverge in ~37 places across the full body (16 within the glass branch alone),
 * so the whole body CANNOT be shared as a single template while keeping BOTH
 * composed WGSL strings byte-identical (that is the ABSOLUTE T4 constraint).
 *
 * What IS byte-identical (verified 2026-07-20 by source diff) are two large
 * CONTIGUOUS spans inside the glass branch — the post-glass reservoir-build +
 * candidate loop, and the glass visibility-test tail. Both are free of the 4 NRC
 * deltas and free of internal comment divergence, so they are shared here as
 * raw-string fragments interpolated at the identical position in each body. The
 * `alpha`→`alphaGlass` rename lives BEFORE these spans, so the spans are clean.
 *
 * These reference CONSUMER bindings (`ubo`, `bvh*`, `bvh_material`, `bvh_beer`)
 * and consumer-declared helpers, so per the composeWgsl ordering rule they are
 * RAW-STRING fragments (NOT a WgslModule) interpolated into each consumer body.
 * Emitted byte-for-byte so the composed `gi-ris` (risGi) and `gi-ris-nrc`
 * (risGiNrc) WGSL stay byte-identical.
 *
 * Boundaries (as of 2026-07-20, verified byte-identical):
 *   GLASS_RESERVOIR_LOOP = risGi.wgsl.ts:338–436 ≡ risGiNrc.wgsl.ts:369–467
 *   GLASS_VISIBILITY_TAIL = risGi.wgsl.ts:439–478 ≡ risGiNrc.wgsl.ts:469–508
 * The single divergent comment between the two spans (`// Visibility test ...`,
 * present only in risGi) stays at each call site.
 */

/**
 * Post-glass reservoir build + M_GI candidate RIS loop, at the first diffuse
 * surface reached by the 1-interface refraction walk. Emitted with NO leading or
 * trailing newline; the call site controls the surrounding whitespace.
 */
export const RIS_GI_GLASS_RESERVOIR_LOOP_WGSL = /* wgsl */ `    var rGlass: ReservoirGI = emptyReservoirGI();
    rGlass.xv = walkHitPos;
    rGlass.nv = walkHitNormal;
    let walkMaterialWordCoord = vec2u(
      walkHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
      walkHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
    );
    let walkMaterialWord = textureLoad(bvh_material, vec2i(walkMaterialWordCoord), 0).r;
    let walkPayload = sampleRestirDIMaterialPayloadForHit(
      walkHit,
      walkSmoothNormal,
      walkHitNormal,
      decodeMaterialColor(walkHit.matColorPacked).rgb,
      walkMaterialWord,
    );
    let walkClearcoatNormal = walkPayload.clearcoatNormal;
    let walkWo = -refractDir;

    let tier_raw_g = textureLoad(gi_tier, vec2i(fullPx), 0).r;
    let tier_g = clamp(tier_raw_g, 1u, 4u);
    let M_GI_g = M_GI_BASE * tier_g / 2u;
    let ppgGuidedOn_g = (ubo.ppgEnabled == 1u);
    let alpha_g = select(0.0, ubo.ppgMixAlpha, ppgGuidedOn_g);

    for (var i: u32 = 0u; i < M_GI_g; i = i + 1u) {
      var wi: vec3f;
      if (alpha_g > 0.0) {
        let bern_g = rand_f32(&rng);
        if (bern_g < alpha_g) {
          wi = ppgSampleGuidedDir(walkHitPos, &rng);
        } else {
          wi = sampleCosineHemisphere(walkHitNormal, &rng);
        }
      } else {
        wi = sampleCosineHemisphere(walkHitNormal, &rng);
      }
      let cosTheta = max(0.0, dot(walkHitNormal, wi));
      if (cosTheta < 1e-4) { continue; }

      let bounceRay = Ray(walkHitPos + walkHit.normal * NORMAL_BIAS_GI, wi);
      let bounceHit = traceSceneFirstHitAlphaMaskTexturedOpaqueOnly(
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
        let smoothNs_g = restir_gi_smooth_normal_for_hit(bounceHit, bounceHit.normal);
        ns_g = applyBumpMapForHit(bounceHit, applyNormalMapForHit(bounceHit, smoothNs_g));
        let irrAtXs = min(sampleDDGIAtPoint(xs_g, ns_g), vec3f(ubo.restirGiIrrClamp));
        let xsRmCoord_g = vec2u(
          bounceHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
          bounceHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
        );
        let xsMaterialWord_g = textureLoad(bvh_material, vec2i(xsRmCoord_g), 0).r;
        let xsPayload_g = sampleRestirGIHitMaterialForHit(
          bounceHit,
          smoothNs_g,
          ns_g,
          irrAtXs,
          wi,
          xsMaterialWord_g,
        );
        Lo_g = xsPayload_g.Lo;
      } else {
        xs_g = walkHitPos + wi * RECONNECT_MAX_DIST;
        ns_g = -wi;
        Lo_g = envRadiance(wi);
      }

      let pHat_g = restir_gi_receiver_phat_from_payload(
        walkHitPos,
        walkHitNormal,
        walkClearcoatNormal,
        walkWo,
        walkPayload,
        xs_g,
        Lo_g,
      );
      if (pHat_g < 1e-9) { continue; }
      let pCos_g = cosTheta * INV_PI;
      var w_g: f32;
      if (alpha_g > 0.0) {
        let pGuide_g = ppgEvalPdf(walkHitPos, wi);
        let pSrc_g = alpha_g * pGuide_g + (1.0 - alpha_g) * pCos_g;
        w_g = select(0.0, pHat_g / pSrc_g, pSrc_g > 1e-12);
      } else {
        w_g = select(0.0, pHat_g / pCos_g, pCos_g > 1e-12);
      }
      updateReservoirGI(&rGlass, xs_g, ns_g, Lo_g, w_g, &rng);
    }`;

/**
 * Glass visibility-test tail: shadow-ray the chosen post-glass sample, finalise
 * W, cache Phase-0, store, and return. Emitted with NO leading or trailing
 * newline; the call site controls the surrounding whitespace.
 */
export const RIS_GI_GLASS_VISIBILITY_TAIL_WGSL = /* wgsl */ `    if (rGlass.M > 0u && rGlass.w_sum > 0.0) {
      let toS_g = rGlass.xs - rGlass.xv;
      let distS_g = length(toS_g);
      if (distS_g > 1e-4) {
        let wiZ_g = toS_g / distS_g;
        let shadowOrig_g = rGlass.xv + rGlass.nv * NORMAL_BIAS_GI;
        let shadowTint_g = traceSceneAlphaTintTransmittanceTextured(
          ubo.bvhMode, ubo.tlasNodeCount,
          &bvh_index, &bvh_position, &bvh,
          &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
          &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
          shadowOrig_g, wiZ_g, distS_g - 2e-3, ubo.triIntersectEpsilon,
          bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
        );
        let shadowT_g = clamp(luminance(shadowTint_g), 0.0, 1.0);
        if (shadowT_g <= 0.001) {
          rGlass.w_sum = 0.0;
          rGlass.W = 0.0;
        } else {
          let pHatZ_g = restir_gi_receiver_phat_from_payload(
            rGlass.xv,
            rGlass.nv,
            walkClearcoatNormal,
            walkWo,
            walkPayload,
            rGlass.xs,
            rGlass.Lo,
          );
          rGlass.w_sum = rGlass.w_sum * shadowT_g;
          finaliseGIReservoirWFromPHat(&rGlass, ubo.restirGiWCap, false, pHatZ_g);
        }
      } else {
        rGlass.W = 0.0;
        rGlass.w_sum = 0.0;
      }
    }

    refreshPhase0Cache(&rGlass);
    storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, rGlass);
    return;`;
