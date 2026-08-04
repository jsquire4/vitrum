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

float evaluateSurfaceCoverage(
  uint materialIndex, const in Material material, SurfaceHit surfaceHit,
  sampler2DArray attributesArray, int pathDepth,
  out vec4 albedo, out vec3 albedoModulation
) {
  albedo = vec4( material.color, material.opacity );
  albedoModulation = vec3( 1.0 );
  if ( material.vertexColors ) {
    vec4 vertexColor = textureSampleBarycoord(
      attributesArray, ATTR_COLOR, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
    );
    albedo *= vertexColor;
    albedoModulation *= vertexColor.rgb;
  }
  return albedo.a;
}

float evaluateAttenuationThickness(
  uint materialIndex, const in Material material, SurfaceHit surfaceHit,
  sampler2DArray attributesArray, int pathDepth,
  out bool hasAttenuationThickness
) {
  hasAttenuationThickness = material.thickness > 0.0;
  return max( material.thickness, 0.0 );
}

int getSurfaceRecord(
  uint materialIndex, SurfaceHit surfaceHit,
  sampler2DArray attributesArray, float accumulatedRoughness,
  int pathDepth, float heroWavelength,
  bool stochasticOpacityAlreadyAccepted, inout SurfaceRecord surf
) {
  Material material;
  readMaterialInfo( materials, materialIndex, material );
  vec4 albedo;
  vec3 albedoModulation;
  float coverage = evaluateSurfaceCoverage(
    materialIndex, material, surfaceHit, attributesArray, pathDepth,
    albedo, albedoModulation
  );
  int coverageStatus = classifySurfaceCoverage(
    material.side, surfaceHit.side, material.alphaTest,
    material.transparent, coverage
  );
  if (
    coverageStatus == SURFACE_COVERAGE_HOLE ||
    ! stochasticOpacityAlreadyAccepted &&
      coverageStatus == SURFACE_COVERAGE_FRACTIONAL &&
      ( ! ( coverage > 0.0 && coverage < 1.0 ) ||
        rand( 3 ) >= representedBernoulliProbabilityF32( coverage ) )
  ) return SKIP_SURFACE;

  vec3 sampledNormal = textureSampleBarycoord(
    attributesArray, ATTR_NORMAL, surfaceHit.barycoord, surfaceHit.faceIndices.xyz
  ).xyz;
  vec3 normal = length( sampledNormal ) > 1e-6
    ? normalize( sampledNormal )
    : surfaceHit.faceNormal;
  normal *= surfaceHit.side;

  float transmission = material.transmission;
  bool frontFaceHit = surfaceHit.side == 1.0;
  bool hasFaceLayer = frontFaceHit ? material.hasFrontLayer : material.hasBackLayer;
	bool hasOppositeLayer = frontFaceHit ? material.hasBackLayer : material.hasFrontLayer;
  vec3 layerTransmission = frontFaceHit
    ? material.frontLayerTransmission
    : material.backLayerTransmission;
  float layerRoughness = frontFaceHit
    ? material.frontLayerRoughness
    : material.backLayerRoughness;
	vec3 oppositeLayerTransmission = frontFaceHit
		? material.backLayerTransmission
		: material.frontLayerTransmission;
	float oppositeLayerRoughness = frontFaceHit
		? material.backLayerRoughness
		: material.frontLayerRoughness;
  float roughness = material.roughness;
	float oppositeRoughness = material.roughness;
  if ( hasFaceLayer && layerRoughness >= 0.0 ) {
    roughness = clamp( layerRoughness, 0.0, 1.0 );
  }
	if ( hasOppositeLayer && oppositeLayerRoughness >= 0.0 ) {
		oppositeRoughness = clamp( oppositeLayerRoughness, 0.0, 1.0 );
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
	surf.oppositeNormal = - normal;
  surf.ior = material.ior;
  surf.eta = material.ior == 0.0
    ? 0.0
    : material.thinFilm || frontFaceHit
      ? 1.0 / material.ior
      : material.ior;
  surf.f0 = iorRatioToF0( surf.eta );
  surf.roughness = roughness * roughness;
  surf.filteredRoughness = applyFilteredGlossy(
    surf.roughness, accumulatedRoughness
  );
	surf.oppositeRoughness = oppositeRoughness * oppositeRoughness;
	surf.oppositeFilteredRoughness = applyFilteredGlossy(
		surf.oppositeRoughness, accumulatedRoughness
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
	surf.oppositeLayerTransmission = hasOppositeLayer
		? clamp( oppositeLayerTransmission, vec3( 0.0 ), vec3( 1.0 ) )
		: vec3( 1.0 );
	surf.hasOppositeLayer = hasOppositeLayer;
  surf.materialIndex = materialIndex;
  surf.attenuationColor = material.attenuationColor;
  surf.attenuationDistance = material.attenuationDistance;
  surf.attenuationThickness = evaluateAttenuationThickness(
    materialIndex, material, surfaceHit, attributesArray, pathDepth,
    surf.hasAttenuationThickness
  );

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
	surf.oppositeNormalBasis = getBasisFromNormalAndTangent(
		surf.oppositeNormal, tangentSample
	);
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
