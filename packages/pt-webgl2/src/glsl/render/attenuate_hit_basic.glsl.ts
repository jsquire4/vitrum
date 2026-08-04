/** Opaque visibility traversal for the conservative base-PBR material tier. */
export const ATTENUATE_HIT_BASIC_GLSL = /* glsl */ `
  bool attenuateHit(
    RenderState state,
    Ray ray,
    float rayDist,
    bool hasTargetFace,
    uint targetFaceIndex,
    out vec3 color
  ) {
    uint originalBounceIndex = sobolBounceIndex;
    bool finiteRayDistance = ! vitrumIsInfiniteDistance( rayDist );
    float traveledDistance = 0.0;
    SurfaceHit surfaceHit;
    color = vec3( 1.0 );
    bool result = true;

    for ( int step = 0; step < 32; step ++ ) {
      if ( step >= state.traversals ) break;
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
      if ( material.side != 0.0 && surfaceHit.side != material.side ) {
        if ( ! setExactRayRangeFromSurfaceHit( ray, surfaceHit ) ) {
          result = true;
          break;
        }
        continue;
      }
      result = true;
      break;
    }

    sobolBounceIndex = originalBounceIndex;
    return result;
  }
`;
