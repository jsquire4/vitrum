/** Alpha-aware visibility traversal for texture-capable opaque base PBR. */
export const ATTENUATE_HIT_MAPPED_PBR_GLSL = /* glsl */ `
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
  bool finiteRayDistance = ! vitrumIsInfiniteDistance( rayDist );
  float traveledDistance = 0.0;
  SurfaceHit surfaceHit;
  color = vec3( 1.0 );
  bool result = true;

  for ( int attenuationStep = 0; attenuationStep < 32; attenuationStep ++ ) {
    if ( remainingTraversals <= 0 ) break;
    remainingTraversals --;
    sobolBounceIndex ++;
    int hitType = traceScene(
      ray, state.mediumStack,
      state.fogMaterial, state.wavelength, surfaceHit
    );
    if ( hitType == INVALID_HIT ) {
      result = true;
      break;
    }
    if ( hitType != SURFACE_HIT ) {
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
    float remainingDistance = finiteRayDistance
      ? max( rayDist - traveledDistance, 0.0 )
      : INFINITY;
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
    if ( finiteRayDistance ) {
      traveledDistance += max( surfaceHit.dist, 0.0 );
    }
    if ( ! material.castShadow && state.isShadowRay ) {
      if ( ! setExactRayRangeFromSurfaceHit( ray, surfaceHit ) ) {
        result = true;
        break;
      }
      continue;
    }

    #define MAPPED_SHADOW_UV(mapIndex) textureSampleBarycoord( attributesArray, readMaterialMapUvLayer( materials, materialIndex, mapIndex ), surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy
    vec4 albedo = vec4( material.color, material.opacity );
    if ( material.map != -1 ) {
      vec3 uvPrime =
        material.mapTransform * vec3( MAPPED_SHADOW_UV( 0u ), 1.0 );
      vec4 mapSample;
      if (
        sampleMaterialTexture(
          textures, uvPrime.xy, material.map,
          material.mapWrap, true, mapSample
        )
      ) albedo *= mapSample;
    }
    if ( material.vertexColors ) {
      albedo *= textureSampleBarycoord(
        attributesArray, ATTR_COLOR,
        surfaceHit.barycoord, surfaceHit.faceIndices.xyz
      );
    }
    if ( material.alphaMap != -1 ) {
      vec3 uvPrime =
        material.alphaMapTransform * vec3( MAPPED_SHADOW_UV( 6u ), 1.0 );
      vec4 alphaSample;
      if (
        sampleMaterialTexture(
          textures, uvPrime.xy, material.alphaMap,
          material.alphaMapWrap, alphaSample
        )
      ) albedo.a *= alphaSample.r;
    }

    bool useAlphaTest = material.alphaTest != 0.0;
    bool passThrough =
      material.side != 0.0 && surfaceHit.side != material.side ||
      useAlphaTest && albedo.a < material.alphaTest ||
      material.transparent && ! useAlphaTest &&
        rand( 10 ) >= representedBernoulliProbabilityF32( albedo.a );
    #undef MAPPED_SHADOW_UV
    if ( ! passThrough ) {
      result = true;
      break;
    }
    if ( ! setExactRayRangeFromSurfaceHit( ray, surfaceHit ) ) {
      result = true;
      break;
    }
  }

  sobolBounceIndex = originalBounceIndex;
  return result;
}
`;
