/** Surface decode for the conservative opaque base-PBR material tier. */
export const GET_SURFACE_RECORD_BASIC_GLSL = /* glsl */ `
  #define SKIP_SURFACE 0
  #define HIT_SURFACE 1
  uniform int materialLodDepth;

  int getSurfaceRecord(
    uint materialIndex, SurfaceHit surfaceHit,
    sampler2DArray attributesArray, float accumulatedRoughness,
    int pathDepth, float heroWavelength,
    bool stochasticOpacityAlreadyAccepted, inout SurfaceRecord surf
  ) {
    Material material;
    readMaterialInfo( materials, materialIndex, material );
    if ( material.side != 0.0 && surfaceHit.side != material.side ) {
      return SKIP_SURFACE;
    }
    vec3 normal = normalize( textureSampleBarycoord(
      attributesArray, ATTR_NORMAL, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
    ).xyz );
    if ( length( normal ) <= 1e-6 ) {
      normal = normalize( surfaceHit.faceNormal * surfaceHit.side );
    }
    normal *= surfaceHit.side;

    vec3 albedo = material.color;
    if ( material.vertexColors ) {
      albedo *= textureSampleBarycoord(
        attributesArray, ATTR_COLOR, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
      ).rgb;
    }

    surf.volumeParticle = false;
    surf.faceNormal = surfaceHit.faceNormal;
    surf.frontFace = true;
    surf.normal = normal;
    surf.normalBasis = getBasisFromNormal( normal );
    surf.ior = material.ior;
    surf.eta = 1.0 / material.ior;
    surf.f0 = iorRatioToF0( surf.eta );
    surf.roughness = clamp( material.roughness, 0.0, 1.0 );
    surf.roughness *= surf.roughness;
    surf.filteredRoughness = applyFilteredGlossy( surf.roughness, accumulatedRoughness );
    surf.metalness = clamp( material.metalness, 0.0, 1.0 );
    surf.color = albedo;
    surf.rgbColor = albedo;
    surf.emission = material.emissiveIntensity * material.emissive;

    surf.transmission = 0.0;
    surf.thinFilm = false;
    surf.thinFilmEnabled = 0.0;
    surf.thinFilmLayerCount = 0.0;
    surf.thinFilmIncidentIor = 1.0;
    surf.thinFilmAngleDependent = false;
    surf.dispersionStrength = 0.0;
    surf.sssSigmaT = 0.0;
    surf.sssAnisotropyG = 0.0;
    surf.sssSigmaS = vec3( 0.0 );
    surf.hasSpectralAttenuation = false;
    surf.activeLayerTransmission = vec3( 1.0 );
    surf.hasActiveLayer = false;
    surf.materialIndex = materialIndex;
    surf.attenuationColor = vec3( 1.0 );
    surf.attenuationDistance = INFINITY;
    surf.attenuationThickness = 0.0;
    surf.hasAttenuationThickness = false;

    surf.clearcoatNormal = normal;
    surf.clearcoatBasis = surf.normalBasis;
    surf.clearcoat = 0.0;
    surf.clearcoatRoughness = 1.0;
    surf.filteredClearcoatRoughness = 1.0;
    surf.sheen = 0.0;
    surf.sheenColor = vec3( 0.0 );
    surf.sheenRoughness = 1.0;
    surf.iridescence = 0.0;
    surf.iridescenceIor = 1.0;
    surf.iridescenceThickness = 0.0;
    surf.specularColor = vec3( 1.0 );
    surf.specularIntensity = 1.0;
    surf.anisotropy = 0.0;
    surf.anisotropyRotation = 0.0;
    surf.spectralReflectanceCoeffs = vec3( 0.0 );
    surf.hasSpectralReflectance = false;
    surf.envMapIntensity = 1.0;
    surf.lobeMask = 2u;
    if ( surf.roughness > 0.0 || surf.metalness < 1.0 ) surf.lobeMask |= 1u;
    return HIT_SURFACE;
  }

  int getSurfaceRecord(
    uint materialIndex, SurfaceHit surfaceHit,
    sampler2DArray attributesArray, float accumulatedRoughness,
    int pathDepth, float heroWavelength, inout SurfaceRecord surf
  ) {
    return getSurfaceRecord(
      materialIndex, surfaceHit, attributesArray,
      accumulatedRoughness, pathDepth, heroWavelength, false, surf
    );
  }
`;
