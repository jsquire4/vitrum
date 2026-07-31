/**
 * Texture-capable base-PBR surface reconstruction. The scene classifier only
 * selects this graph when every authored material field is represented here.
 */
export const GET_SURFACE_RECORD_MAPPED_PBR_GLSL = /* glsl */ `
#define SKIP_SURFACE 0
#define HIT_SURFACE 1
uniform int materialLodDepth;

mat3 getBasisFromNormalAndTangent( vec3 normal, vec4 tangentSample ) {
  if ( length( normal ) <= 1e-6 ) return getBasisFromNormal( vec3( 0.0, 0.0, 1.0 ) );
  vec3 n = normalize( normal );
  if ( length( tangentSample.xyz ) <= 1e-6 ) return getBasisFromNormal( n );
  vec3 tangent = tangentSample.xyz - n * dot( tangentSample.xyz, n );
  if ( length( tangent ) <= 1e-6 ) return getBasisFromNormal( n );
  tangent = normalize( tangent );
  float handedness = tangentSample.w < 0.0 ? -1.0 : 1.0;
  vec3 bitangent = cross( n, tangent ) * handedness;
  if ( length( bitangent ) <= 1e-6 ) return getBasisFromNormal( n );
  return mat3( tangent, normalize( bitangent ), n );
}

float evalMappedPbrSpectrum( vec3 coeffs, float lambda ) {
  float x = coeffs.x + coeffs.y * lambda + coeffs.z * lambda * lambda;
  return 0.5 + x * inversesqrt( 1.0 + x * x ) * 0.5;
}

int getSurfaceRecord(
  uint materialIndex, SurfaceHit surfaceHit,
  sampler2DArray attributesArray, float accumulatedRoughness,
  int pathDepth, float heroWavelength,
  bool stochasticOpacityAlreadyAccepted, inout SurfaceRecord surf
) {
  Material material;
  readMaterialInfo( materials, materialIndex, material );
  vec4 vertexColor = textureSampleBarycoord(
    attributesArray, ATTR_COLOR, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
  );
  #define MAPPED_PBR_UV(mapIndex) textureSampleBarycoord( attributesArray, readMaterialMapUvLayer( materials, materialIndex, mapIndex ), surfaceHit.barycoord, surfaceHit.faceIndices.xyz ).xy
  bool useTextures = materialLodDepth == 0 || pathDepth <= materialLodDepth;

  vec4 albedo = vec4( material.color, material.opacity );
  vec3 albedoModulation = vec3( 1.0 );
  if ( useTextures && material.map != -1 ) {
    vec3 uvPrime = material.mapTransform * vec3( MAPPED_PBR_UV( 0u ), 1.0 );
    vec4 sampleValue = sampleMaterialTexture(
      textures, uvPrime.xy, material.map, material.mapWrap, true
    );
    albedo *= sampleValue;
    albedoModulation *= sampleValue.rgb;
  }
  if ( material.vertexColors ) {
    albedo *= vertexColor;
    albedoModulation *= vertexColor.rgb;
  }
  if ( useTextures && material.alphaMap != -1 ) {
    vec3 uvPrime = material.alphaMapTransform * vec3( MAPPED_PBR_UV( 6u ), 1.0 );
    albedo.a *= sampleMaterialTexture(
      textures, uvPrime.xy, material.alphaMap, material.alphaMapWrap
    ).r;
  }
  if ( useTextures && material.aoMap != -1 ) {
    vec3 uvPrime = material.aoMapTransform * vec3( MAPPED_PBR_UV( 16u ), 1.0 );
    float ao = sampleMaterialTexture(
      textures, uvPrime.xy, material.aoMap, material.aoMapWrap
    ).r;
    float aoFactor = clamp(
      mix( 1.0, ao, material.aoMapIntensity ), 0.0, 1.0
    );
    albedo.rgb *= aoFactor;
    albedoModulation *= aoFactor;
  }

  bool useAlphaTest = material.alphaTest != 0.0;
  if (
    material.side != 0.0 && surfaceHit.side != material.side ||
    useAlphaTest && albedo.a < material.alphaTest ||
    ! stochasticOpacityAlreadyAccepted && material.transparent &&
      ! useAlphaTest && albedo.a < rand( 3 )
  ) {
    return SKIP_SURFACE;
  }

  float roughness = material.roughness;
  if ( useTextures && material.roughnessMap != -1 ) {
    vec3 uvPrime =
      material.roughnessMapTransform * vec3( MAPPED_PBR_UV( 2u ), 1.0 );
    roughness *= sampleMaterialTexture(
      textures, uvPrime.xy, material.roughnessMap, material.roughnessMapWrap
    ).g;
  }
  float metalness = material.metalness;
  if ( useTextures && material.metalnessMap != -1 ) {
    vec3 uvPrime =
      material.metalnessMapTransform * vec3( MAPPED_PBR_UV( 1u ), 1.0 );
    metalness *= sampleMaterialTexture(
      textures, uvPrime.xy, material.metalnessMap, material.metalnessMapWrap
    ).b;
  }
  vec3 emission = vitrumFiniteNonNegativeRadianceProduct(
    material.emissive, vec3( material.emissiveIntensity )
  );
  if ( useTextures && material.emissiveMap != -1 ) {
    vec3 uvPrime =
      material.emissiveMapTransform * vec3( MAPPED_PBR_UV( 4u ), 1.0 );
    emission = vitrumFiniteNonNegativeRadianceProduct(
      emission,
      sampleMaterialTexture(
        materialRadianceTextures, uvPrime.xy,
        material.emissiveMap, material.emissiveMapWrap
      ).rgb
    );
  }
  if ( useTextures && material.lightMap != -1 && pathDepth == 0 ) {
    vec3 uvPrime =
      material.lightMapTransform * vec3( MAPPED_PBR_UV( 17u ), 1.0 );
    emission = vitrumFiniteNonNegativeRadianceSum(
      emission,
      vitrumFiniteNonNegativeRadianceProduct(
        vec3( material.lightMapIntensity ),
        sampleMaterialTexture(
          materialRadianceTextures, uvPrime.xy,
          material.lightMap, material.lightMapWrap
        ).rgb
      )
    );
  }

  vec3 normal = normalize( textureSampleBarycoord(
    attributesArray, ATTR_NORMAL, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
  ).xyz );
  if ( length( normal ) <= 1e-6 ) {
    normal = surfaceHit.faceNormal * surfaceHit.side;
  }
  int surfaceBasisUvLayer = ATTR_UV;
  if ( useTextures && material.normalMap != -1 ) {
    vec4 tangentSample = textureSampleBarycoord(
      attributesArray, ATTR_TANGENT, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
    );
    int normalUvLayer = readMaterialMapUvLayer( materials, materialIndex, 5u );
    surfaceBasisUvLayer = normalUvLayer;
    mat3 normalBasis = getBasisFromSelectedUv(
      bvh.position, attributesArray, normalUvLayer,
      surfaceHit.faceIndices.xyz, normal, tangentSample
    );
    vec2 normalUv = textureSampleBarycoord(
      attributesArray, normalUvLayer,
      surfaceHit.barycoord, surfaceHit.faceIndices.xyz
    ).xy;
    vec3 uvPrime = material.normalMapTransform * vec3( normalUv, 1.0 );
    vec3 texNormal = sampleMaterialTexture(
      textures, uvPrime.xy, material.normalMap, material.normalMapWrap
    ).xyz * 2.0 - 1.0;
    texNormal.xy *= material.normalScale;
    vec3 mappedNormal = normalBasis * texNormal;
    if ( length( mappedNormal ) > 1e-6 ) normal = normalize( mappedNormal );
  }
  if ( useTextures && material.bumpMap != -1 ) {
    vec4 tangentSample = textureSampleBarycoord(
      attributesArray, ATTR_TANGENT, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
    );
    int bumpUvLayer = readMaterialMapUvLayer( materials, materialIndex, 18u );
    if ( material.normalMap == -1 ) surfaceBasisUvLayer = bumpUvLayer;
    mat3 bumpBasis = getBasisFromSelectedUv(
      bvh.position, attributesArray, bumpUvLayer,
      surfaceHit.faceIndices.xyz, normal, tangentSample
    );
    vec2 bumpUv = textureSampleBarycoord(
      attributesArray, bumpUvLayer,
      surfaceHit.barycoord, surfaceHit.faceIndices.xyz
    ).xy;
    vec3 uvPrime = material.bumpMapTransform * vec3( bumpUv, 1.0 );
    vec2 bumpTexel = 1.0 / vec2(
      materialTextureSourceSize( textures, material.bumpMapWrap, 0 )
    );
    float hC = sampleMaterialTexture(
      textures, uvPrime.xy, material.bumpMap, material.bumpMapWrap
    ).r;
    float hU = sampleMaterialTexture(
      textures, uvPrime.xy + vec2( bumpTexel.x, 0.0 ),
      material.bumpMap, material.bumpMapWrap
    ).r;
    float hV = sampleMaterialTexture(
      textures, uvPrime.xy + vec2( 0.0, bumpTexel.y ),
      material.bumpMap, material.bumpMapWrap
    ).r;
    vec3 tangent = bumpBasis[ 0 ];
    vec3 bitangent = bumpBasis[ 1 ];
    vec3 perturbed = normal - material.bumpScale * (
      ( hU - hC ) / bumpTexel.x * tangent
      + ( hV - hC ) / bumpTexel.y * bitangent
    );
    if ( length( perturbed ) > 1e-6 ) normal = normalize( perturbed );
  }
  normal *= surfaceHit.side;

  vec3 surfaceColor = albedo.rgb;
  if ( uSpectralRendering == 1 && material.hasSpectralReflectance ) {
    surfaceColor = vec3(
      evalMappedPbrSpectrum(
        material.spectralReflectanceCoeffs, heroWavelength
      ) * heroScalarFromRgb( albedoModulation, heroWavelength )
    );
  }

  surf.volumeParticle = false;
  surf.faceNormal = surfaceHit.faceNormal;
  surf.frontFace = true;
  surf.normal = normal;
  vec4 bsdfTangentSample = textureSampleBarycoord(
    attributesArray, ATTR_TANGENT,
    surfaceHit.barycoord, surfaceHit.faceIndices.xyz
  );
  surf.normalBasis = getBasisFromSelectedUv(
    bvh.position, attributesArray, surfaceBasisUvLayer,
    surfaceHit.faceIndices.xyz, normal, bsdfTangentSample
  );
  surf.ior = material.ior;
  surf.eta = 1.0 / material.ior;
  surf.f0 = iorRatioToF0( surf.eta );
  surf.roughness = clamp( roughness, 0.0, 1.0 );
  surf.roughness *= surf.roughness;
  surf.filteredRoughness = applyFilteredGlossy(
    surf.roughness, accumulatedRoughness
  );
  surf.metalness = clamp( metalness, 0.0, 1.0 );
  surf.color = surfaceColor;
  surf.rgbColor = albedo.rgb;
  surf.emission = emission;
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
  surf.spectralReflectanceCoeffs = material.spectralReflectanceCoeffs;
  surf.hasSpectralReflectance = material.hasSpectralReflectance;
  surf.envMapIntensity = max( material.envMapIntensity, 0.0 );
  surf.lobeMask = 2u;
  if ( surf.roughness > 0.0 || surf.metalness < 1.0 ) surf.lobeMask |= 1u;
  #undef MAPPED_PBR_UV
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
