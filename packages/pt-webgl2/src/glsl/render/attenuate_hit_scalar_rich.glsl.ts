/** Transparent-shadow traversal for scalar-rich materials without map fetches. */
export const ATTENUATE_HIT_SCALAR_RICH_GLSL = /* glsl */ `
bool attenuateHit(
  RenderState state,
  Ray ray,
  float rayDist,
  bool hasTargetFace,
  uint targetFaceIndex,
  out vec3 color
) {
  uint originalBounceIndex = sobolBounceIndex;
  int remainingTraversals = max( state.traversals, 0 );
  int transmissiveTraversals = state.transmissiveTraversals;
  FogMaterial fogMaterial = state.fogMaterial;
  MediumStack mediumStack = state.mediumStack;
  if ( state.isShadowRay ) {
    filterShadowMediumStack(
      state.mediumStack, materials, mediumStack, fogMaterial
    );
  }
  bool finiteRayDistance = ! vitrumIsInfiniteDistance( rayDist );
  float traveledDistance = 0.0;
  SurfaceHit surfaceHit;
  color = vec3( 1.0 );
  bool result = true;

  for ( int attenuationStep = 0; attenuationStep < 64; attenuationStep ++ ) {
    if ( remainingTraversals <= 0 ) break;
    remainingTraversals --;
    sobolBounceIndex ++;
    bool invalidRange = false;
    bool surfaceFound = ray.minimumDistanceExclusive >= 0.0
      ? bvhIntersectExactRangeFirstHit( ray, surfaceHit, invalidRange )
      : bvhIntersectCanonicalInitialFirstHit(
          ray, surfaceHit, invalidRange
        );
    if ( invalidRange ) {
      result = true;
      break;
    }
    float remainingDistance = finiteRayDistance
      ? max( rayDist - traveledDistance, 0.0 )
      : INFINITY;
    float segmentDistance = surfaceFound
      ? min( max( surfaceHit.dist, 0.0 ), remainingDistance )
      : remainingDistance;
    float effectiveSegmentDistance = mediumEffectiveSegmentDistance(
      mediumStack, segmentDistance
    );
    if ( effectiveSegmentDistance < 0.0 ) {
      result = true;
      break;
    }
    color *= opticalVisibilitySegmentTransmittance(
      materials, mediumStack, fogMaterial,
      effectiveSegmentDistance, state.wavelength
    );
    if ( ! consumeMediumSegmentDistance(
      mediumStack, segmentDistance, materials, fogMaterial
    ) ) {
      result = true;
      break;
    }
    if ( ! surfaceFound ) {
      result = false;
      break;
    }
    if (
      hasTargetFace &&
      surfaceHit.faceIndices.w == targetFaceIndex
    ) {
      result = false;
      break;
    }
    if (
      finiteRayDistance &&
      surfaceHit.dist >= remainingDistance
    ) {
      result = false;
      break;
    }

    uint materialIndex = uTexelFetch1D(
      materialIndexAttribute, surfaceHit.faceIndices.w
    ).r;
    Material material;
    readMaterialInfo( materials, materialIndex, material );
    vec4 albedo = vec4( material.color, material.opacity );
    if ( material.vertexColors ) {
      albedo *= textureSampleBarycoord(
        attributesArray, ATTR_COLOR, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
      );
    }
    bool useAlphaTest = material.alphaTest != 0.0;
    bool skipSurface =
      material.side != 0.0 && surfaceHit.side != material.side ||
      useAlphaTest && albedo.a < material.alphaTest ||
      material.transparent && ! useAlphaTest &&
        rand( 10 ) >= representedBernoulliProbabilityF32( albedo.a );

    if ( finiteRayDistance ) {
      traveledDistance += max( surfaceHit.dist, 0.0 );
    }
    // Sidedness and alpha describe whether a surface exists for this visibility
    // trial. A rejected surface is a null event and must not mutate medium state.
    if ( skipSurface ) {
      if ( transmissiveTraversals > 0 ) {
        remainingTraversals ++;
        transmissiveTraversals --;
      }
      if ( ray.minimumDistanceExclusive >= 0.0 ) {
        if ( ! setExactRayRangeFromSurfaceHit( ray, surfaceHit ) ) {
          result = true;
          break;
        }
      } else {
        ray.origin = stepRayOrigin(
          ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist
        );
        setOrdinaryRayRange( ray );
      }
      continue;
    }

    if ( ! material.castShadow && state.isShadowRay ) {
      // The filtered shadow stack omits this material's volume extinction too.
      // Its boundary is therefore a null event for the visibility query.
      if ( ! setExactRayRangeFromSurfaceHit( ray, surfaceHit ) ) {
        result = true;
        break;
      }
      if ( transmissiveTraversals > 0 ) {
        remainingTraversals ++;
        transmissiveTraversals --;
      }
      continue;
    }

    bool exactBaseRoughness = material.roughness == 0.0;
    bool exactFrontRoughness =
      ! material.hasFrontLayer ||
      material.frontLayerRoughness < 0.0 ||
      material.frontLayerRoughness == 0.0;
    bool exactBackRoughness =
      ! material.hasBackLayer ||
      material.backLayerRoughness < 0.0 ||
      material.backLayerRoughness == 0.0;
    if (
      material.thinFilm && material.transmission > 0.0 &&
      exactBaseRoughness && exactFrontRoughness && exactBackRoughness
    ) {
      SurfaceRecord sheetSurface;
      int sheetStatus = getSurfaceRecord(
        materialIndex, surfaceHit, attributesArray,
        0.0, 0, state.wavelength, true, sheetSurface
      );
      vec3 sheetAttenuation;
      bool sheetConnectable =
        sheetStatus == HIT_SURFACE &&
        thinSheetExactVisibilityTransmission(
          - ray.direction, ray.direction,
          sheetSurface, state.wavelength, sheetAttenuation
        );
      if ( ! sheetConnectable ) {
        result = true;
        break;
      }
      color *= sheetAttenuation;
      if ( ! setExactRayRangeFromSurfaceHit( ray, surfaceHit ) ) {
        result = true;
        break;
      }
      if ( transmissiveTraversals > 0 ) {
        remainingTraversals ++;
        transmissiveTraversals --;
      }
      continue;
    }

    // Material.transmission is a BSDF event, not an alpha/null event. A straight
    // NEE/BDPT connection cannot silently collapse the interface because that
    // omits refraction, Fresnel, roughness, Jacobians, and the corresponding MIS
    // vertex. Every accepted physical surface therefore occludes this segment.
    result = true;
    break;
  }

  sobolBounceIndex = originalBounceIndex;
  return result;
}
`;
