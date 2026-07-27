/** Transparent-shadow traversal for scalar-rich materials without map fetches. */
export const ATTENUATE_HIT_SCALAR_RICH_GLSL = /* glsl */ `
bool attenuateHit( RenderState state, Ray ray, float rayDist, out vec3 color ) {
  uint originalBounceIndex = sobolBounceIndex;
  int remainingTraversals = max( state.traversals, 0 );
  int transmissiveTraversals = state.transmissiveTraversals;
  FogMaterial fogMaterial = state.fogMaterial;
  MediumStack mediumStack = state.mediumStack;
  vec3 startPoint = ray.origin;
  SurfaceHit surfaceHit;
  color = vec3( 1.0 );
  bool result = true;

  for ( int attenuationStep = 0; attenuationStep < 64; attenuationStep ++ ) {
    if ( remainingTraversals <= 0 ) break;
    remainingTraversals --;
    sobolBounceIndex ++;
    bool surfaceFound = bvhIntersectFirstHit(
      bvh, ray.origin, ray.direction,
      surfaceHit.faceIndices, surfaceHit.faceNormal,
      surfaceHit.barycoord, surfaceHit.side, surfaceHit.dist
    );
    float traveled = distance( startPoint, ray.origin );
    float remainingDistance = max( rayDist - traveled, 0.0 );
    float segmentDistance = surfaceFound
      ? min( max( surfaceHit.dist, 0.0 ), remainingDistance )
      : remainingDistance;
    #if FEATURE_FOG
    if ( fogMaterial.fogVolume ) {
      color *= fogSegmentTransmittance(
        materials, fogMaterial, segmentDistance, state.wavelength
      );
    }
    #endif
    if ( ! surfaceFound || surfaceHit.dist > remainingDistance ) {
      result = false;
      break;
    }

    uint materialIndex = uTexelFetch1D(
      materialIndexAttribute, surfaceHit.faceIndices.w
    ).r;
    Material material;
    readMaterialInfo( materials, materialIndex, material );
    bool isEntering = surfaceHit.side == 1.0;
    ray.origin = stepRayOrigin(
      ray.origin, ray.direction, - surfaceHit.faceNormal, surfaceHit.dist
    );
    #if FEATURE_FOG
    if ( material.fogVolume ) {
      bool stackValid = surfaceHit.side == 1.0
        ? enterMedium( mediumStack, materialIndex, materials, fogMaterial )
        : leaveMedium( mediumStack, materialIndex, materials, fogMaterial );
      if ( ! stackValid ) {
        result = true;
        break;
      }
      if ( transmissiveTraversals > 0 ) {
        remainingTraversals ++;
        transmissiveTraversals --;
      }
      continue;
    }
    #endif
    if ( ! material.castShadow && state.isShadowRay ) continue;

    vec4 albedo = vec4( material.color, material.opacity );
    if ( material.vertexColors ) {
      albedo *= textureSampleBarycoord(
        attributesArray, ATTR_COLOR, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
      );
    }
    bool useAlphaTest = material.alphaTest != 0.0;
    float transmissionFactor = ( 1.0 - material.metalness ) * material.transmission;
    if (
      transmissionFactor < rand( 9 ) && ! (
        material.side != 0.0 && surfaceHit.side == material.side ||
        useAlphaTest && albedo.a < material.alphaTest ||
        material.transparent && ! useAlphaTest && albedo.a < rand( 10 )
      )
    ) {
      result = true;
      break;
    }

    if ( surfaceHit.side == 1.0 && isEntering ) {
      vec3 surfaceTransmission = mix(
        vec3( 1.0 ), albedo.rgb, transmissionFactor
      );
      color *= pathThroughputFromRgb( surfaceTransmission, state.wavelength );
    } else if ( surfaceHit.side == -1.0 ) {
      float attenuationDist = surfaceHit.dist;
      if ( material.thickness > 0.0 ) {
        attenuationDist = min( attenuationDist, material.thickness );
      }
      color *= transmissionAttenuationThroughput(
        materials, attenuationDist, material.attenuationColor,
        material.attenuationDistance, material.hasSpectralAttenuation,
        materialIndex, state.wavelength
      );
    }

    bool isTransmissiveRay =
      dot( ray.direction, surfaceHit.faceNormal * surfaceHit.side ) < 0.0;
    if ( ( isTransmissiveRay || isEntering ) && transmissiveTraversals > 0 ) {
      remainingTraversals ++;
      transmissiveTraversals --;
    }
  }

  sobolBounceIndex = originalBounceIndex;
  return result;
}
`;
