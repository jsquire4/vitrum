/** Scalar-rich surface construction with no material texture branches. */
export const GET_SURFACE_RECORD_SCALAR_RICH_GLSL = /* glsl */ `
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

int getSurfaceRecord(
  uint materialIndex, SurfaceHit surfaceHit,
  sampler2DArray attributesArray, float accumulatedRoughness,
  int pathDepth, float heroWavelength,
  bool stochasticOpacityAlreadyAccepted, inout SurfaceRecord surf
) {
  Material material;
  readMaterialInfo( materials, materialIndex, material );
  vec4 albedo = vec4( material.color, material.opacity );
  vec3 albedoModulation = vec3( 1.0 );
  if ( material.vertexColors ) {
    vec4 vertexColor = textureSampleBarycoord(
      attributesArray, ATTR_COLOR, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
    );
    albedo *= vertexColor;
    albedoModulation *= vertexColor.rgb;
  }

  bool useAlphaTest = material.alphaTest != 0.0;
  if (
    material.side != 0.0 && surfaceHit.side != material.side ||
    useAlphaTest && albedo.a < material.alphaTest ||
    ! stochasticOpacityAlreadyAccepted && material.transparent &&
      ! useAlphaTest && albedo.a < rand( 3 )
  ) return SKIP_SURFACE;

  vec3 normal = normalize( textureSampleBarycoord(
    attributesArray, ATTR_NORMAL, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
  ).xyz );
  if ( length( normal ) <= 1e-6 ) normal = surfaceHit.faceNormal * surfaceHit.side;
  normal *= surfaceHit.side;

  float transmission = material.transmission;
  bool frontFaceHit = surfaceHit.side == 1.0 || transmission == 0.0;
  bool hasFaceLayer = frontFaceHit ? material.hasFrontLayer : material.hasBackLayer;
  vec3 layerTransmission = frontFaceHit
    ? material.frontLayerTransmission
    : material.backLayerTransmission;
  float layerRoughness = frontFaceHit
    ? material.frontLayerRoughness
    : material.backLayerRoughness;
  float roughness = material.roughness;
  if ( hasFaceLayer && layerRoughness >= 0.0 ) {
    roughness = clamp( layerRoughness, 0.0, 1.0 );
  }

  vec3 surfaceColor = albedo.rgb;
  if ( uSpectralRendering == 1 && material.hasSpectralReflectance ) {
    float baseReflectance = evalSpectrum(
      material.spectralReflectanceCoeffs, heroWavelength
    );
    float modulation = heroScalarFromRgb( albedoModulation, heroWavelength );
    surfaceColor = vec3( baseReflectance * modulation );
  }

  surf.volumeParticle = false;
  surf.faceNormal = surfaceHit.faceNormal;
  surf.frontFace = frontFaceHit;
  surf.normal = normal;
  surf.ior = material.ior;
  surf.eta = material.thinFilm || frontFaceHit ? 1.0 / material.ior : material.ior;
  surf.f0 = iorRatioToF0( surf.eta );
  surf.roughness = roughness * roughness;
  surf.filteredRoughness = applyFilteredGlossy(
    surf.roughness, accumulatedRoughness
  );
  surf.metalness = material.metalness;
  surf.color = surfaceColor;
  surf.rgbColor = albedo.rgb;
  surf.emission = material.emissiveIntensity * material.emissive;

  surf.transmission = transmission;
  surf.thinFilm = material.thinFilm;
  surf.thinFilmEnabled = material.thinFilmEnabled;
  surf.thinFilmLayerCount = material.thinFilmLayerCount;
  surf.thinFilmIncidentIor = material.thinFilmIncidentIor;
  surf.thinFilmAngleDependent = material.thinFilmAngleDependent;
  surf.dispersionStrength = material.dispersionStrength;
  surf.sssSigmaT = material.sssSigmaT;
  surf.sssAnisotropyG = material.sssAnisotropyG;
  surf.sssSigmaS = material.sssSigmaS;
  surf.hasSpectralAttenuation = material.hasSpectralAttenuation;
  surf.activeLayerTransmission = hasFaceLayer
    ? clamp( layerTransmission, vec3( 0.0 ), vec3( 1.0 ) )
    : vec3( 1.0 );
  surf.hasActiveLayer = hasFaceLayer;
  surf.materialIndex = materialIndex;
  surf.attenuationColor = material.attenuationColor;
  surf.attenuationDistance = material.attenuationDistance;
  surf.attenuationThickness = max( material.thickness, 0.0 );
  surf.hasAttenuationThickness = material.thickness > 0.0;

  surf.clearcoatNormal = normal;
  surf.clearcoat = material.clearcoat;
  surf.clearcoatRoughness = material.clearcoatRoughness * material.clearcoatRoughness;
  surf.filteredClearcoatRoughness = applyFilteredGlossy(
    surf.clearcoatRoughness, accumulatedRoughness
  );
  surf.sheen = material.sheen;
  surf.sheenColor = material.sheenColor;
  surf.sheenRoughness = material.sheenRoughness;
  float iridescenceThickness = material.iridescenceThicknessMaximum;
  surf.iridescence = iridescenceThickness == 0.0 ? 0.0 : material.iridescence;
  surf.iridescenceIor = material.iridescenceIor;
  surf.iridescenceThickness = iridescenceThickness;
  surf.specularColor = material.specularColor;
  surf.specularIntensity = material.specularIntensity;
  surf.anisotropy = clamp( material.anisotropy, 0.0, 1.0 );
  surf.anisotropyRotation = material.anisotropyRotation;
  surf.spectralReflectanceCoeffs = material.spectralReflectanceCoeffs;
  surf.hasSpectralReflectance = material.hasSpectralReflectance;
  surf.envMapIntensity = max( material.envMapIntensity, 0.0 );

  vec4 tangentSample = textureSampleBarycoord(
    attributesArray, ATTR_TANGENT, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
  );
  surf.normalBasis = getBasisFromNormalAndTangent( surf.normal, tangentSample );
  surf.clearcoatBasis = getBasisFromNormal( surf.clearcoatNormal );

  surf.lobeMask = 0u;
  if ( surf.roughness > 0.0 || surf.metalness < 1.0 ) surf.lobeMask |= 1u;
  surf.lobeMask |= 2u;
  if ( surf.sheen > 0.0 ) surf.lobeMask |= 4u;
  if ( surf.clearcoat > 0.0 ) surf.lobeMask |= 8u;
  if ( surf.iridescence > 0.0 ) surf.lobeMask |= 16u;
  if ( surf.transmission > 0.0 ) surf.lobeMask |= 32u;
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
