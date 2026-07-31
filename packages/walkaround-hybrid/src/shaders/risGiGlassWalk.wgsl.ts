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
 * Bounded camera-prefix transport shared by the default and NRC GI-RIS passes.
 * Tracks up to four dielectric interfaces, a nested IOR stack, exact
 * unpolarised Fresnel transmission, actual-distance Beer/spectral attenuation,
 * and thin-walled zero-thickness sheets. Interface-budget overflow fails
 * closed instead of turning the last dielectric into a diffuse receiver.
 */
export const RIS_GI_GLASS_TRANSPORT_PREFIX_WGSL = /* wgsl */ `  const GLASS_WALK_MAX_INTERFACES: u32 = 4u;

  if (isGlass) {
    let glassPrimaryRmCoord = vec2u(
      hit.indices.w % BVH_MATERIAL_TEX_WIDTH,
      hit.indices.w / BVH_MATERIAL_TEX_WIDTH,
    );
    let glassPrimaryPacked = textureLoad(bvh_material, vec2i(glassPrimaryRmCoord), 0).r;
    let glassPrimaryRm = decodeRoughMetal(glassPrimaryPacked);
    let primaryLayer = sampleFaceLayerControls(hit.indices.w, hit.side >= 0.0);
    let glassPrimaryRough = faceLayerRoughness(glassPrimaryRm.x, primaryLayer);
    let primaryIorRgb = materialDispersionIorRgb(
      hit.indices.w, decodeIor(glassPrimaryPacked),
    );
    var primaryThickness = materialOpticalThickness(hit.indices.w);
    let primaryThicknessTexel = sampleMaterialAtlasRawAtOffset(
      hit.indices.w,
      MATERIAL_MAP_THICKNESS_TEXEL_OFFSET,
      hit.uv,
      materialAtlasUv1ForHit(hit),
    );
    if (primaryThicknessTexel.x >= 0.0) {
      primaryThickness = primaryThickness *
        clamp(primaryThicknessTexel.g, 0.0, 1.0);
    }
    let d = primaryRay.direction;
    // The geometric surface owns enter/exit classification. The mapped
    // shading normal owns Fresnel/refraction, oriented into the corresponding
    // geometric hemisphere and rejected if it cannot oppose the incident ray.
    let primaryEntering = hit.side >= 0.0;
    let primaryAlignedNormal = select(-normal, normal, dot(normal, geoNormal) >= 0.0);
    let primaryFaceNormal = select(
      -primaryAlignedNormal,
      primaryAlignedNormal,
      dot(d, primaryAlignedNormal) < 0.0,
    );
    if (dot(d, primaryFaceNormal) >= 0.0) {
      storeReservoirGI_rw(pixelIdxGi, emptyReservoirGI());
      return;
    }
    // The primary ray starts in air. A thin sheet is reciprocal regardless of
    // winding; a bulk back face is rejected below because no entry was tracked.
    let primaryIncidentIor = vec3f(1.0);
    let primaryTargetIor = primaryIorRgb;
    let primaryBtdf = ggxSampleDielectricTransmission(
      primaryFaceNormal,
      -d,
      glassPrimaryRough,
      primaryIncidentIor.g,
      primaryTargetIor.g,
      &rng,
    );
    if (primaryBtdf.valid == 0u) {
      storeReservoirGI_rw(pixelIdxGi, emptyReservoirGI());
      return;
    }
    let primaryCosI = primaryBtdf.microfacetCos;
    var primaryInterfaceT = dielectricInterfaceTransmissionRgb(
      primaryCosI, primaryIncidentIor, primaryTargetIor,
    );
    let primaryFilm = materialThinFilmResponse(
      hit.indices.w, hit.side >= 0.0, primaryCosI,
    );
    if (primaryFilm.present != 0u) {
      primaryInterfaceT = primaryFilm.transmittance;
    }
    var glassPathThroughput = primaryInterfaceT *
      (primaryBtdf.weight / primaryBtdf.transmission) *
      faceLayerTransmission(primaryLayer);

    var mediumDepth: u32 = 0u;
    var mediumIor: array<vec3f, 4>;
    var mediumTri: array<u32, 4>;
    var mediumMaterialWord: array<u32, 4>;
    var mediumInstance: array<u32, 4>;
    var mediumBeer: array<vec3f, 4>;
    var mediumThickness: array<f32, 4>;
    var refractDir = primaryBtdf.direction;
    // A thin sheet pays reciprocal entry+exit Fresnel at one geometric hit and
    // therefore consumes two physical interface slots.
    var interfaceCount = select(1u, 2u, primaryThickness <= 0.0);

    if (primaryThickness > 0.0) {
      if (!primaryEntering) {
        // The ray starts in an untracked medium; exact Beer distance/state is
        // unavailable, so do not invent an air-to-glass continuation.
        storeReservoirGI_rw(pixelIdxGi, emptyReservoirGI());
        return;
      }
      if (primaryEntering) {
        glassPathThroughput = glassPathThroughput * matColor.a;
      }
      if (primaryEntering) {
        let beerPacked = textureLoad(bvh_beer, vec2i(glassPrimaryRmCoord), 0).r;
        var beer = vec3f(
          f32((beerPacked >> 24u) & 0xffu) / 255.0,
          f32((beerPacked >> 16u) & 0xffu) / 255.0,
          f32((beerPacked >> 8u) & 0xffu) / 255.0,
        );
        beer = applyThicknessMapToBeerTint(
          hit.indices.w, hit.uv, materialAtlasUv1ForHit(hit), beer,
        );
        mediumIor[0] = primaryIorRgb;
        mediumTri[0] = hit.indices.w;
        mediumMaterialWord[0] = glassPrimaryPacked;
        mediumInstance[0] = hit.instanceIndex;
        mediumBeer[0] = beer;
        mediumThickness[0] = primaryThickness;
        mediumDepth = 1u;
      }
    } else {
      // A zero-thickness sheet still owns two physical interfaces. Sample the
      // reciprocal rough BTDF at the coincident exit instead of cancelling the
      // authored roughness into an undeviated smooth ray.
      let primaryExitLayer = sampleFaceLayerControls(hit.indices.w, hit.side < 0.0);
      let primaryExitRough = faceLayerRoughness(glassPrimaryRm.x, primaryExitLayer);
      let primaryExitBtdf = ggxSampleDielectricTransmission(
        primaryFaceNormal,
        -refractDir,
        primaryExitRough,
        primaryTargetIor.g,
        primaryIncidentIor.g,
        &rng,
      );
      if (primaryExitBtdf.valid == 0u) {
        storeReservoirGI_rw(pixelIdxGi, emptyReservoirGI());
        return;
      }
      var primaryExitT = dielectricInterfaceTransmissionRgb(
        primaryExitBtdf.microfacetCos, primaryTargetIor, primaryIncidentIor,
      );
      let primaryExitFilm = materialThinFilmResponse(
        hit.indices.w, hit.side < 0.0, primaryExitBtdf.microfacetCos,
      );
      if (primaryExitFilm.present != 0u) {
        primaryExitT = primaryExitFilm.transmittance;
      }
      glassPathThroughput = glassPathThroughput * primaryExitT *
        (primaryExitBtdf.weight / primaryExitBtdf.transmission) *
        faceLayerTransmission(primaryExitLayer) * matColor.a;
      refractDir = primaryExitBtdf.direction;
    }

    var walkOrigin = pos + refractDir * walkaroundRayOriginBias();
    var walkHit: IntersectionResult;
    var walkHitPos: vec3f;
    var walkHitNormal: vec3f;
    var walkSmoothNormal: vec3f;
    var foundSurface = false;

    // Inspect one terminal hit after the fourth interface. That terminal slot
    // may be the opaque receiver, but it may not consume a fifth interface.
    for (var gi: u32 = 1u; gi <= GLASS_WALK_MAX_INTERFACES; gi = gi + 1u) {
      let walkRay = Ray(walkOrigin, refractDir);
      walkHit = traceSceneFirstHitAlphaMaskTextured(
        ubo.bvhMode, ubo.tlasNodeCount,
        walkRay, ubo.triIntersectEpsilon,
        bvh_material, BVH_MATERIAL_TEX_WIDTH,
        ubo.frameSeed ^ (gi * 0x9e3779b9u) ^ 0x474c4153u,
      );
      if (!walkHit.didHit) { break; }

      // Attenuate the geometric segment in the currently active medium.
      if (mediumDepth > 0u) {
        let top = mediumDepth - 1u;
        if (!(mediumThickness[top] > 0.0)) { break; }
        let segmentDistance = walkHit.dist + walkaroundRayOriginBias();
        let distanceScale = segmentDistance / mediumThickness[top];
        let rgbBeer = pow(clamp(mediumBeer[top], vec3f(0.0), vec3f(1.0)), vec3f(distanceScale));
        glassPathThroughput = glassPathThroughput * materialSpectralAttenuation(
          mediumTri[top], segmentDistance, rgbBeer,
        );
      }
      if (max(glassPathThroughput.r, max(glassPathThroughput.g, glassPathThroughput.b)) <= 0.0) {
        break;
      }

      walkHitPos = walkOrigin + refractDir * walkHit.dist;
      walkHitNormal = walkHit.normal;
      let scalarWalkMat = decodeMaterialColor(walkHit.matColorPacked);
      let walkTransmission = sampleTransmissionMapForHit(walkHit, scalarWalkMat.a);
      let walkIsGlass = materialHasTransmission(walkTransmission);

      let wnIsTlas = ubo.bvhMode == 1u;
      let wnBase = walkHit.instanceIndex * 4u;
      let wnOk = wnIsTlas && wnBase + 2u < tlasWorldToLocalColumnCount();
      let wnI = select(0u, wnBase, wnOk);
      walkSmoothNormal = smoothShadingNormal(
        walkHit, walkHit.normal,
        sceneLoadBvhNormal(walkHit.indices.x).xyz,
        sceneLoadBvhNormal(walkHit.indices.y).xyz,
        sceneLoadBvhNormal(walkHit.indices.z).xyz,
        wnOk,
        tlasLoadWorldToLocalColumn(wnI),
        tlasLoadWorldToLocalColumn(wnI + 1u),
        tlasLoadWorldToLocalColumn(wnI + 2u),
      );
      let walkMappedNormal = applyBumpMapForHit(
        walkHit, applyNormalMapForHit(walkHit, walkSmoothNormal),
      );

      if (!walkIsGlass) {
        walkHitNormal = walkMappedNormal;
        foundSurface = true;
        break;
      }

      let walkCoord = vec2u(
        walkHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
        walkHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
      );
      let walkWord = textureLoad(bvh_material, vec2i(walkCoord), 0).r;
      let walkRm = decodeRoughMetal(walkWord);
      let walkLayer = sampleFaceLayerControls(
        walkHit.indices.w, walkHit.side >= 0.0,
      );
      let walkRough = faceLayerRoughness(walkRm.x, walkLayer);
      let walkIor = materialDispersionIorRgb(walkHit.indices.w, decodeIor(walkWord));
      var walkThickness = materialOpticalThickness(walkHit.indices.w);
      let walkThicknessTexel = sampleMaterialAtlasRawAtOffset(
        walkHit.indices.w,
        MATERIAL_MAP_THICKNESS_TEXEL_OFFSET,
        walkHit.uv,
        materialAtlasUv1ForHit(walkHit),
      );
      if (walkThicknessTexel.x >= 0.0) {
        walkThickness = walkThickness *
          clamp(walkThicknessTexel.g, 0.0, 1.0);
      }
      let entering = walkHit.side >= 0.0;
      if (walkThickness > 0.0 && !entering && mediumDepth == 0u) { break; }
      let interfaceCost = select(1u, 2u, walkThickness <= 0.0);
      if (interfaceCount + interfaceCost > GLASS_WALK_MAX_INTERFACES) { break; }
      interfaceCount = interfaceCount + interfaceCost;
      let alignedNormal = select(
        -walkMappedNormal,
        walkMappedNormal,
        dot(walkMappedNormal, walkHit.normal) >= 0.0,
      );
      let faceNormal = select(
        -alignedNormal,
        alignedNormal,
        dot(refractDir, alignedNormal) < 0.0,
      );
      if (dot(refractDir, faceNormal) >= 0.0) { break; }
      var incidentIor = vec3f(1.0);
      if (mediumDepth > 0u) {
        incidentIor = mediumIor[mediumDepth - 1u];
      } else if (walkThickness > 0.0 && !entering) {
        incidentIor = walkIor;
      }
      var targetIor = walkIor;
      if (walkThickness <= 0.0) {
        // Reciprocal sheet: enclosing medium -> sheet -> enclosing medium.
        targetIor = walkIor;
      } else if (entering) {
        targetIor = walkIor;
      } else {
        let top = mediumDepth - 1u;
        if (
          mediumMaterialWord[top] != walkWord ||
          mediumInstance[top] != walkHit.instanceIndex
        ) {
          break;
        }
        if (mediumDepth > 1u) {
          targetIor = mediumIor[mediumDepth - 2u];
        }
      }
      let interfaceBtdf = ggxSampleDielectricTransmission(
        faceNormal,
        -refractDir,
        walkRough,
        incidentIor.g,
        targetIor.g,
        &rng,
      );
      if (interfaceBtdf.valid == 0u) { break; }
      let interfaceCos = interfaceBtdf.microfacetCos;
      var interfaceT = dielectricInterfaceTransmissionRgb(
        interfaceCos, incidentIor, targetIor,
      );
      let film = materialThinFilmResponse(
        walkHit.indices.w, walkHit.side >= 0.0, interfaceCos,
      );
      if (film.present != 0u) { interfaceT = film.transmittance; }
      glassPathThroughput = glassPathThroughput * interfaceT *
        (interfaceBtdf.weight / interfaceBtdf.transmission) *
        faceLayerTransmission(walkLayer);

      if (walkThickness <= 0.0) {
        let exitLayer = sampleFaceLayerControls(
          walkHit.indices.w, walkHit.side < 0.0,
        );
        let exitRough = faceLayerRoughness(walkRm.x, exitLayer);
        let exitBtdf = ggxSampleDielectricTransmission(
          faceNormal,
          -interfaceBtdf.direction,
          exitRough,
          targetIor.g,
          incidentIor.g,
          &rng,
        );
        if (exitBtdf.valid == 0u) { break; }
        var exitT = dielectricInterfaceTransmissionRgb(
          exitBtdf.microfacetCos, targetIor, incidentIor,
        );
        let exitFilm = materialThinFilmResponse(
          walkHit.indices.w, walkHit.side < 0.0, exitBtdf.microfacetCos,
        );
        if (exitFilm.present != 0u) { exitT = exitFilm.transmittance; }
        glassPathThroughput = glassPathThroughput * exitT *
          (exitBtdf.weight / exitBtdf.transmission) *
          faceLayerTransmission(exitLayer) * walkTransmission;
        refractDir = exitBtdf.direction;
        walkOrigin = walkHitPos + refractDir * walkaroundRayOriginBias();
        continue;
      }

      if (entering) { glassPathThroughput = glassPathThroughput * walkTransmission; }
      let nextDir = interfaceBtdf.direction;

      if (entering) {
        if (mediumDepth >= GLASS_WALK_MAX_INTERFACES) { break; }
        let walkBeerPacked = textureLoad(bvh_beer, vec2i(walkCoord), 0).r;
        var walkBeer = vec3f(
          f32((walkBeerPacked >> 24u) & 0xffu) / 255.0,
          f32((walkBeerPacked >> 16u) & 0xffu) / 255.0,
          f32((walkBeerPacked >> 8u) & 0xffu) / 255.0,
        );
        walkBeer = applyThicknessMapToBeerTint(
          walkHit.indices.w, walkHit.uv, materialAtlasUv1ForHit(walkHit), walkBeer,
        );
        mediumIor[mediumDepth] = walkIor;
        mediumTri[mediumDepth] = walkHit.indices.w;
        mediumMaterialWord[mediumDepth] = walkWord;
        mediumInstance[mediumDepth] = walkHit.instanceIndex;
        mediumBeer[mediumDepth] = walkBeer;
        mediumThickness[mediumDepth] = walkThickness;
        mediumDepth = mediumDepth + 1u;
      } else if (mediumDepth > 0u) {
        mediumDepth = mediumDepth - 1u;
      }
      refractDir = nextDir;
      walkOrigin = walkHitPos + refractDir * walkaroundRayOriginBias();
    }

    if (!foundSurface) {
      storeReservoirGI_rw(pixelIdxGi, emptyReservoirGI());
      return;
    }
`;

/**
 * Post-glass reservoir build + M_GI candidate RIS loop, at the first opaque
 * surface reached by the bounded dielectric-prefix walk. Emitted with NO leading or
 * trailing newline; the call site controls the surrounding whitespace.
 */
export const RIS_GI_GLASS_RESERVOIR_LOOP_WGSL = /* wgsl */ `    var rGlass: ReservoirGI = emptyReservoirGI();
    rGlass.xv = walkHitPos;
    rGlass.nv = walkHitNormal;
    rGlass.historyEpoch = currentGrisEpoch;

    let walkReceiverCoord = vec2u(
      walkHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
      walkHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
    );
    let walkReceiverWord = textureLoad(
      bvh_material,
      vec2i(walkReceiverCoord),
      0,
    ).r;
    let walkReceiverPayload = sampleRestirDIMaterialPayloadForHit(
      walkHit,
      walkSmoothNormal,
      walkHitNormal,
      decodeMaterialColor(walkHit.matColorPacked).rgb,
      walkReceiverWord,
      safe_normalize(-refractDir),
    );
    rGlass.receiverMaterialKey = restir_gi_receiver_domain_key(
      walkHit.matColorPacked,
      walkReceiverWord,
      walkHit.indices.w,
      select(0u, walkHit.instanceIndex, ubo.bvhMode == 1u),
      walkReceiverPayload,
    );

    let tier_raw_g = textureLoad(gi_tier, vec2i(fullPx), 0).r;
    let tier_g = clamp(tier_raw_g, 1u, 4u);
    let M_GI_g = M_GI_BASE * tier_g / 2u;
    let ppgGuidedOn_g = ubo.ppgEnabled == 1u;
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
      if (cosTheta <= 0.0) {
        recordInvalidReservoirGICandidate(&rGlass, GI_SAMPLE_SURFACE, currentGrisEpoch);
        continue;
      }

      let bounceRay = Ray(
        walkHitPos + walkHit.normal * walkaroundRayOriginBias(),
        wi,
      );
      let bounceHit = traceSceneFirstHitAlphaMaskTextured(
        ubo.bvhMode, ubo.tlasNodeCount,

        bounceRay, ubo.triIntersectEpsilon,
        bvh_material, BVH_MATERIAL_TEX_WIDTH,
        ubo.frameSeed ^ (i * 0xc2b2ae35u) ^ 0x474c5358u,
      );

      var xs_g: vec3f;
      var ns_g: vec3f;
      var Lo_g: vec3f;
      var sampleKind_g: u32 = GI_SAMPLE_ENVIRONMENT;

      if (bounceHit.didHit) {
        sampleKind_g = GI_SAMPLE_SURFACE;
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
        xs_g = walkHitPos + wi * walkaroundReconnectMaxDistance();
        ns_g = -wi;
        Lo_g = envRadiance(wi);
      }

      // The camera prefix from the visible glass pane to this receiver is part
      // of the same estimator. Fold its RGB Fresnel, mapped transmission, and
      // distance-dependent Beer/spectral throughput into both the selected
      // payload and p-hat; shade must not apply those factors a second time.
      if (sampleKind_g == GI_SAMPLE_ENVIRONMENT) {
        Lo_g = walkaroundScaleEnvironmentRadiance(
          Lo_g,
          walkReceiverPayload.envMapIntensity,
        );
      }
      Lo_g = Lo_g * glassPathThroughput;

      var candidateVisibility_g: f32 = 1.0;
      var pHat_g: f32;
      var tMax_g = INFINITY;
      if (sampleKind_g == GI_SAMPLE_SURFACE) {
        tMax_g = max(
          0.0,
          safe_length(xs_g - walkHitPos) - walkaroundRayEndMargin(),
        );
      }
      let shadowTintCandidate_g = traceSceneAlphaTintTransmittanceTextured(
        ubo.bvhMode, ubo.tlasNodeCount,

        walkHitPos + walkHit.normal * walkaroundRayOriginBias(),
        wi, tMax_g, ubo.triIntersectEpsilon,
        bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
      );
      candidateVisibility_g = clamp(luminance(shadowTintCandidate_g), 0.0, 1.0);
      pHat_g = restir_gi_receiver_phat_from_payload(
        walkHitPos,
        walkHitNormal,
        walkReceiverPayload.clearcoatNormal,
        safe_normalize(-refractDir),
        walkReceiverPayload,
        xs_g,
        Lo_g,
      ) * candidateVisibility_g;
      if (!reservoirGiFinite(pHat_g) || !(pHat_g > 0.0)
       || !reservoirGiFinite(candidateVisibility_g) || candidateVisibility_g <= 0.0) {
        recordInvalidReservoirGICandidate(&rGlass, sampleKind_g, currentGrisEpoch);
        continue;
      }
      let pCos_g = cosTheta * INV_PI;
      var w_g: f32 = 0.0;
      if (alpha_g > 0.0) {
        let pGuide_g = ppgEvalPdf(walkHitPos, wi);
        let pSrc_g = alpha_g * pGuide_g + (1.0 - alpha_g) * pCos_g;
        if (pSrc_g > 0.0) { w_g = pHat_g / pSrc_g; }
      } else {
        if (pCos_g > 0.0) { w_g = pHat_g / pCos_g; }
      }
      if (!reservoirGiFinite(w_g) || !(w_g > 0.0)) {
        recordInvalidReservoirGICandidate(&rGlass, sampleKind_g, currentGrisEpoch);
        continue;
      }
      updateReservoirGIWithMetadata(
        &rGlass,
        xs_g, ns_g, Lo_g,
        sampleKind_g, wi,
        pHat_g, candidateVisibility_g, currentGrisEpoch,
        w_g,
        &rng,
      );
    }`;

/**
 * Glass visibility-test tail: shadow-ray the chosen post-glass sample, finalise
 * W, cache Phase-0, store, and return. Emitted with NO leading or trailing
 * newline; the call site controls the surrounding whitespace.
 */
export const RIS_GI_GLASS_VISIBILITY_TAIL_WGSL = /* wgsl */ `    if (rGlass.M > 0u && rGlass.w_sum > 0.0) {
      if (
        rGlass.historyEpoch != currentGrisEpoch ||
        rGlass.sampleVisibility <= 0.0 ||
        !(rGlass.nativePHat > 0.0)
      ) {
        rGlass.w_sum = 0.0;
        rGlass.W = 0.0;
      } else {
        finaliseGIReservoirWFromPHat(
          &rGlass,
          ubo.restirGiWCap,
          rGlass.nativePHat,
        );
      }
    }

    refreshGrisMetadata(&rGlass);
    // Lo contains camera-side Fresnel/Beer throughput from a bounded
    // dielectric prefix that is not represented in the reservoir. Keep the
    // sample for this pixel's transmitted-GI shading, but mark it as having no
    // valid temporal/spatial inverse shift.
    if (rGlass.prefixVertexCount == GI_PREFIX_RECONNECTABLE) {
      rGlass.prefixVertexCount = GI_PREFIX_CAMERA_TRANSMISSION;
    }
    storeReservoirGI_rw(pixelIdxGi, rGlass);
    return;`;
