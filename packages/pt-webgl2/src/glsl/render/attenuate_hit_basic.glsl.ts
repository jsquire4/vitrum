/** Opaque visibility traversal for the conservative base-PBR material tier. */
export const ATTENUATE_HIT_BASIC_GLSL = /* glsl */ `
  bool attenuateHit( RenderState state, Ray ray, float rayDist, out vec3 color ) {
    uint originalBounceIndex = sobolBounceIndex;
    vec3 startPoint = ray.origin;
    SurfaceHit surfaceHit;
    color = vec3( 1.0 );
    bool result = true;

    for ( int step = 0; step < 32; step ++ ) {
      if ( step >= state.traversals ) break;
      sobolBounceIndex ++;
      int hitType = traceScene( ray, state.fogMaterial, surfaceHit );
      if ( hitType != SURFACE_HIT ) {
        result = false;
        break;
      }
      float totalDist = distance(
        startPoint, ray.origin + ray.direction * surfaceHit.dist
      );
      if ( totalDist > rayDist ) {
        result = false;
        break;
      }
      uint materialIndex = uTexelFetch1D(
        materialIndexAttribute, surfaceHit.faceIndices.w
      ).r;
      Material material;
      readMaterialInfo( materials, materialIndex, material );
      if ( ! material.castShadow && state.isShadowRay ) {
        ray.origin = stepRayOrigin(
          ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist
        );
        continue;
      }
      result = true;
      break;
    }

    sobolBounceIndex = originalBounceIndex;
    return result;
  }
`;
