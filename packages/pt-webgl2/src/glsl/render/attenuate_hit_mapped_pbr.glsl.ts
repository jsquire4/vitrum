/** Alpha-aware visibility traversal for texture-capable opaque base PBR. */
export const ATTENUATE_HIT_MAPPED_PBR_GLSL = /* glsl */ `
bool attenuateHit( RenderState state, Ray ray, float rayDist, out vec3 color ) {
  uint originalBounceIndex = sobolBounceIndex;
  int remainingTraversals = max( state.traversals, 0 );
  vec3 startPoint = ray.origin;
  SurfaceHit surfaceHit;
  color = vec3( 1.0 );
  bool result = true;

  for ( int attenuationStep = 0; attenuationStep < 32; attenuationStep ++ ) {
    if ( remainingTraversals <= 0 ) break;
    remainingTraversals --;
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
    ray.origin = stepRayOrigin(
      ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist
    );
    if ( ! material.castShadow && state.isShadowRay ) continue;

    #define MAPPED_SHADOW_UV(mapIndex) textureSampleBarycoord( attributesArray, readMaterialMapUvLayer( materials, materialIndex, mapIndex ), surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy
    vec4 albedo = vec4( material.color, material.opacity );
    if ( material.map != -1 ) {
      vec3 uvPrime =
        material.mapTransform * vec3( MAPPED_SHADOW_UV( 0u ), 1.0 );
      albedo *= sampleMaterialTexture(
        textures, uvPrime.xy, material.map, material.mapWrap, true
      );
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
      albedo.a *= sampleMaterialTexture(
        textures, uvPrime.xy, material.alphaMap, material.alphaMapWrap
      ).r;
    }

    bool useAlphaTest = material.alphaTest != 0.0;
    bool passThrough =
      material.side != 0.0 && surfaceHit.side == material.side ||
      useAlphaTest && albedo.a < material.alphaTest ||
      material.transparent && ! useAlphaTest && albedo.a < rand( 10 );
    #undef MAPPED_SHADOW_UV
    if ( ! passThrough ) {
      result = true;
      break;
    }
  }

  sobolBounceIndex = originalBounceIndex;
  return result;
}
`;
