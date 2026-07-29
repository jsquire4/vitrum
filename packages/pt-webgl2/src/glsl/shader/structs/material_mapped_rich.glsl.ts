import {
  MATERIAL_LAYER_NORMAL_TEXEL_OFFSET,
  MATERIAL_PIXELS,
  MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET,
} from './materialStride.js';

/**
 * Exact compact decoder for every public MaterialSpec field, including mixed
 * texture maps, physical transmission/volume, Disney lobes, per-face layers,
 * spectral data, anisotropy, and multi-layer thin film.
 */
export const MATERIAL_MAPPED_RICH_GLSL = /* glsl */ `
struct Material {
  vec3 color; int map;
  float metalness; int metalnessMap;
  float roughness; int roughnessMap;
  float ior; float transmission; int transmissionMap;
  float emissiveIntensity; vec3 emissive; int emissiveMap;
  int normalMap; vec2 normalScale;
  float clearcoat; int clearcoatMap; int clearcoatNormalMap;
  vec2 clearcoatNormalScale; float clearcoatRoughness;
  int clearcoatRoughnessMap;
  int iridescenceMap; int iridescenceThicknessMap;
  float iridescence; float iridescenceIor;
  float iridescenceThicknessMinimum; float iridescenceThicknessMaximum;
  vec3 specularColor; int specularColorMap;
  float specularIntensity; int specularIntensityMap;
  bool thinFilm; float anisotropy; float anisotropyRotation; int anisotropyMap;
  vec3 attenuationColor; float attenuationDistance;
  float thickness; int thicknessMap;
  int alphaMap;
  bool castShadow; float opacity; float alphaTest; float side; bool matte;
  float sheen; vec3 sheenColor; int sheenColorMap;
  float sheenRoughness; int sheenRoughnessMap;
  bool vertexColors; bool transparent; bool unlit;
  bool meshEmitterCastShadowDisabled; bool fogVolume; uint flags;
  float sssSigmaT; float sssAnisotropyG; vec3 sssSigmaS;
  float dispersionStrength;
  float thinFilmEnabled; float thinFilmLayerCount;
  float thinFilmIncidentIor; bool thinFilmAngleDependent;
  bool hasSpectralAttenuation;
  vec3 frontLayerTransmission; float frontLayerRoughness; bool hasFrontLayer;
  int frontLayerNormalMap; vec2 frontLayerNormalScale;
  float frontLayerNormalTexCoord;
  vec3 backLayerTransmission; float backLayerRoughness; bool hasBackLayer;
  int backLayerNormalMap; vec2 backLayerNormalScale;
  float backLayerNormalTexCoord;
  vec3 spectralReflectanceCoeffs; bool hasSpectralReflectance;
  int aoMap; int lightMap; int bumpMap;
  float aoMapIntensity; float lightMapIntensity; float bumpScale;
  float envMapIntensity;
};

int wrapMaterialTextureIndex( int coord, int size, float mode ) {
  int packedMode = int( round( mode ) );
  int m = packedMode - ( packedMode / 4 ) * 4;
  if ( m == 1 ) return clamp( coord, 0, size - 1 );
  if ( m == 2 ) {
    int period = max( size * 2, 1 );
    int c = coord - period * int( floor( float( coord ) / float( period ) ) );
    return c < size ? c : period - 1 - c;
  }
  return coord - size * int( floor( float( coord ) / float( size ) ) );
}

ivec2 materialTextureSourceSize( sampler2DArray tex, vec4 policy, int level ) {
  ivec2 storageSize = textureSize( tex, 0 ).xy;
  int packedS = int( round( policy.x ) );
  int packedT = int( round( policy.y ) );
  ivec2 baseSize = ivec2(
    packedS / 4 > 0 ? packedS / 4 : storageSize.x,
    packedT / 4 > 0 ? packedT / 4 : storageSize.y
  );
  return max( ivec2( 1 ), baseSize / ( 1 << level ) );
}

ivec2 materialTextureSourceOffset( vec4 policy, int level ) {
  ivec2 baseOffset = ivec2(
    int( round( policy.z ) ) / 4,
    int( round( policy.w ) ) / 4
  );
  return baseOffset / ( 1 << level );
}

bool materialTextureUsesLinearFilter( vec4 policy, bool minifying ) {
  int packed = int( round( policy.w ) );
  int filterPair = packed - ( packed / 4 ) * 4;
  int magFilter = filterPair - ( filterPair / 2 ) * 2;
  int minFilter = filterPair / 2;
  return ( minifying ? minFilter : magFilter ) == 1;
}

float materialTextureRawLod( vec2 uv, vec2 baseSize ) {
  vec2 dx = dFdx( uv * baseSize );
  vec2 dy = dFdy( uv * baseSize );
  return max( log2( max( max( length( dx ), length( dy ) ), 1e-8 ) ), 0.0 );
}

vec4 decodeMaterialTextureTexel( vec4 value, bool srgbRgb ) {
  if ( ! srgbRgb ) return value;
  bvec3 cutoff = lessThanEqual( value.rgb, vec3( 0.04045 ) );
  vec3 low = value.rgb / 12.92;
  vec3 high = pow( ( value.rgb + 0.055 ) / 1.055, vec3( 2.4 ) );
  return vec4( mix( high, low, cutoff ), value.a );
}

vec4 sampleMaterialTextureNearestLevel(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy, int level,
  bool srgbRgb
) {
  ivec2 size = materialTextureSourceSize( tex, policy, level );
  ivec2 p = ivec2( floor( uv * vec2( size ) ) );
  int x = wrapMaterialTextureIndex( p.x, size.x, policy.x );
  int y = wrapMaterialTextureIndex( p.y, size.y, policy.y );
  ivec2 sourceOffset = materialTextureSourceOffset( policy, level );
  return decodeMaterialTextureTexel(
    texelFetch( tex, ivec3( sourceOffset + ivec2( x, y ), layer ), level ),
    srgbRgb
  );
}

vec4 sampleMaterialTextureLinearLevel(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy, int level,
  bool srgbRgb
) {
  ivec2 size = materialTextureSourceSize( tex, policy, level );
  vec2 p = uv * vec2( size ) - vec2( 0.5 );
  ivec2 p0 = ivec2( floor( p ) );
  vec2 f = fract( p );
  int x0 = wrapMaterialTextureIndex( p0.x, size.x, policy.x );
  int y0 = wrapMaterialTextureIndex( p0.y, size.y, policy.y );
  int x1 = wrapMaterialTextureIndex( p0.x + 1, size.x, policy.x );
  int y1 = wrapMaterialTextureIndex( p0.y + 1, size.y, policy.y );
  ivec2 sourceOffset = materialTextureSourceOffset( policy, level );
  vec4 c00 = decodeMaterialTextureTexel(
    texelFetch( tex, ivec3( sourceOffset + ivec2( x0, y0 ), layer ), level ),
    srgbRgb
  );
  vec4 c10 = decodeMaterialTextureTexel(
    texelFetch( tex, ivec3( sourceOffset + ivec2( x1, y0 ), layer ), level ),
    srgbRgb
  );
  vec4 c01 = decodeMaterialTextureTexel(
    texelFetch( tex, ivec3( sourceOffset + ivec2( x0, y1 ), layer ), level ),
    srgbRgb
  );
  vec4 c11 = decodeMaterialTextureTexel(
    texelFetch( tex, ivec3( sourceOffset + ivec2( x1, y1 ), layer ), level ),
    srgbRgb
  );
  return mix( mix( c00, c10, f.x ), mix( c01, c11, f.x ), f.y );
}

vec4 sampleMaterialTextureLevel(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy,
  int level, bool linearFilter, bool srgbRgb
) {
  return linearFilter
    ? sampleMaterialTextureLinearLevel( tex, uv, layer, policy, level, srgbRgb )
    : sampleMaterialTextureNearestLevel( tex, uv, layer, policy, level, srgbRgb );
}

vec4 sampleMaterialTexture(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy, bool srgbRgb
) {
  if ( layer < 0 ) return vec4( 1.0 );
  ivec2 baseSizeI = materialTextureSourceSize( tex, policy, 0 );
  vec2 baseSize = vec2( baseSizeI );
  float rawLod = materialTextureRawLod( uv, baseSize );
  bool linearFilter = materialTextureUsesLinearFilter( policy, rawLod > 0.0 );
  int packedMipFilter = int( round( policy.z ) );
  int mipFilter = packedMipFilter - ( packedMipFilter / 4 ) * 4;
  int maxLevel = max(
    0, int( floor( log2( float( max( baseSizeI.x, baseSizeI.y ) ) ) ) )
  );
  if ( mipFilter == 0 || maxLevel == 0 ) {
    return sampleMaterialTextureLevel(
      tex, uv, layer, policy, 0, linearFilter, srgbRgb
    );
  }
  if ( mipFilter == 1 ) {
    int level = clamp( int( floor( rawLod + 0.5 ) ), 0, maxLevel );
    return sampleMaterialTextureLevel(
      tex, uv, layer, policy, level, linearFilter, srgbRgb
    );
  }
  float clampedLod = clamp( rawLod, 0.0, float( maxLevel ) );
  int level0 = int( floor( clampedLod ) );
  int level1 = min( level0 + 1, maxLevel );
  float t = clampedLod - float( level0 );
  vec4 a = sampleMaterialTextureLevel(
    tex, uv, layer, policy, level0, linearFilter, srgbRgb
  );
  vec4 b = sampleMaterialTextureLevel(
    tex, uv, layer, policy, level1, linearFilter, srgbRgb
  );
  return mix( a, b, t );
}

vec4 sampleMaterialTexture(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy
) {
  return sampleMaterialTexture( tex, uv, layer, policy, false );
}

mat3 readTextureTransform( sampler2D tex, uint index ) {
  vec4 row1 = texelFetch1D( tex, index );
  vec4 row2 = texelFetch1D( tex, index + 1u );
  mat3 transform;
  transform[0] = vec3( row1.r, row2.r, 0.0 );
  transform[1] = vec3( row1.g, row2.g, 0.0 );
  transform[2] = vec3( row1.b, row2.b, 1.0 );
  return transform;
}

mat3 readMaterialMapTransform(
  sampler2D tex, uint materialIndex, uint texelOffset
) {
  return readTextureTransform(
    tex, materialIndex * ${MATERIAL_PIXELS}u + texelOffset
  );
}

vec4 readMaterialMapPolicy(
  sampler2D tex, uint materialIndex, uint texelOffset
) {
  return texelFetch1D(
    tex, materialIndex * ${MATERIAL_PIXELS}u + texelOffset
  );
}

vec4 sampleMappedMaterialTextureDecoded(
  sampler2D materialTex,
  sampler2DArray textureTex,
  uint materialIndex,
  int layer,
  uint transformTexel,
  uint policyTexel,
  vec2 uv,
  bool srgbRgb
) {
  vec3 transformedUv = readMaterialMapTransform(
    materialTex, materialIndex, transformTexel
  ) * vec3( uv, 1.0 );
  vec4 policy = readMaterialMapPolicy(
    materialTex, materialIndex, policyTexel
  );
  return sampleMaterialTexture(
    textureTex, transformedUv.xy, layer, policy, srgbRgb
  );
}

vec4 sampleMappedMaterialTexture(
  sampler2D materialTex, sampler2DArray textureTex, uint materialIndex,
  int layer, uint transformTexel, uint policyTexel, vec2 uv
) {
  return sampleMappedMaterialTextureDecoded(
    materialTex, textureTex, materialIndex, layer,
    transformTexel, policyTexel, uv, false
  );
}

vec4 sampleMappedSrgbMaterialTexture(
  sampler2D materialTex, sampler2DArray textureTex, uint materialIndex,
  int layer, uint transformTexel, uint policyTexel, vec2 uv
) {
  return sampleMappedMaterialTextureDecoded(
    materialTex, textureTex, materialIndex, layer,
    transformTexel, policyTexel, uv, true
  );
}

void readMaterialInfo( sampler2D tex, uint index, out Material m ) {
  uint i = index * ${MATERIAL_PIXELS}u;
  // Reuse one staging texel so ANGLE does not keep 23 vec4 decoder
  // temporaries live while material fields are copied to the out record.
  vec4 s = texelFetch1D( tex, i + 0u );
  m.color = s.rgb; m.map = int( round( s.a ) );
  s = texelFetch1D( tex, i + 1u );
  m.metalness = s.r; m.metalnessMap = int( round( s.g ) );
  m.roughness = s.b; m.roughnessMap = int( round( s.a ) );
  s = texelFetch1D( tex, i + 2u );
  m.ior = max( s.r, 1.0 ); m.transmission = s.g;
  m.transmissionMap = int( round( s.b ) ); m.emissiveIntensity = s.a;
  s = texelFetch1D( tex, i + 3u );
  m.emissive = s.rgb; m.emissiveMap = int( round( s.a ) );
  s = texelFetch1D( tex, i + 4u );
  m.normalMap = int( round( s.r ) ); m.normalScale = s.gb;
  m.clearcoat = s.a;
  s = texelFetch1D( tex, i + 5u );
  m.clearcoatMap = int( round( s.r ) ); m.clearcoatRoughness = s.g;
  m.clearcoatRoughnessMap = int( round( s.b ) );
  m.clearcoatNormalMap = int( round( s.a ) );
  s = texelFetch1D( tex, i + 6u );
  m.clearcoatNormalScale = s.rg; m.anisotropyMap = int( round( s.b ) );
  m.sheen = s.a;
  s = texelFetch1D( tex, i + 7u );
  m.sheenColor = s.rgb; m.sheenColorMap = int( round( s.a ) );
  s = texelFetch1D( tex, i + 8u );
  m.sheenRoughness = s.r; m.sheenRoughnessMap = int( round( s.g ) );
  m.iridescenceMap = int( round( s.b ) );
  m.iridescenceThicknessMap = int( round( s.a ) );
  s = texelFetch1D( tex, i + 9u );
  m.iridescence = s.r; m.iridescenceIor = s.g;
  m.iridescenceThicknessMinimum = s.b; m.iridescenceThicknessMaximum = s.a;
  s = texelFetch1D( tex, i + 10u );
  m.specularColor = s.rgb; m.specularColorMap = int( round( s.a ) );
  s = texelFetch1D( tex, i + 11u );
  m.specularIntensity = s.r; m.specularIntensityMap = int( round( s.g ) );
  m.thinFilm = bool( s.b ); m.anisotropy = clamp( s.a, 0.0, 1.0 );
  s = texelFetch1D( tex, i + 12u );
  m.attenuationColor = s.rgb; m.attenuationDistance = s.a;
  s = texelFetch1D( tex, i + 13u );
  m.alphaMap = int( round( s.r ) ); m.opacity = s.g;
  m.alphaTest = s.b; m.side = s.a;
  s = texelFetch1D( tex, i + 14u );
  m.matte = bool( s.r ); m.castShadow = bool( s.g );
  m.vertexColors = bool( int( s.b ) & 1 );
  m.fogVolume = bool( int( s.b ) & 4 );
  uint packedFlags = uint( round( s.a ) );
  m.transparent = bool( packedFlags & 1u );
  m.unlit = bool( packedFlags & 0x20u );
  m.meshEmitterCastShadowDisabled = bool( packedFlags & 0x40u );
  m.flags = packedFlags;
  s = texelFetch1D( tex, i + 15u );
  m.sssSigmaT = s.r; m.sssAnisotropyG = s.g;
  m.dispersionStrength = s.b; m.thinFilmEnabled = s.a;
  s = texelFetch1D( tex, i + 16u );
  m.sssSigmaS = s.rgb; m.thinFilmLayerCount = s.a;
  s = texelFetch1D( tex, i + 17u );
  m.thinFilmIncidentIor = max( s.r, 1.0 );
  m.thinFilmAngleDependent = s.g > 0.5; m.anisotropyRotation = s.b;
  uint featureFlags = uint( round( s.a ) );
  m.hasSpectralAttenuation = bool( featureFlags & 1u );
  m.hasFrontLayer = bool( featureFlags & 2u );
  m.hasBackLayer = bool( featureFlags & 4u );
  s = texelFetch1D( tex, i + 18u );
  m.frontLayerTransmission = s.rgb; m.frontLayerRoughness = s.a;
  s = texelFetch1D( tex, i + 19u );
  m.backLayerTransmission = s.rgb; m.backLayerRoughness = s.a;
  s = texelFetch1D( tex, i + 85u );
  m.aoMap = int( round( s.r ) ); m.lightMap = int( round( s.g ) );
  m.bumpMap = int( round( s.b ) ); m.envMapIntensity = s.a;
  s = texelFetch1D( tex, i + 86u );
  m.aoMapIntensity = s.r; m.lightMapIntensity = s.g;
  m.bumpScale = s.b;
  s = texelFetch1D( tex, i + 97u );
  m.thickness = max( s.r, 0.0 ); m.thicknessMap = int( round( s.g ) );
  s = texelFetch1D(
    tex, i + ${MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET}u
  );
  m.spectralReflectanceCoeffs = s.xyz;
  m.hasSpectralReflectance = s.w > 0.5;

  vec4 layerNormal = texelFetch1D(
    tex, i + ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET}u
  );
  m.frontLayerNormalMap = int( round( layerNormal.r ) );
  m.frontLayerNormalScale = vec2( layerNormal.g );
  m.backLayerNormalMap = int( round( layerNormal.b ) );
  m.backLayerNormalScale = vec2( layerNormal.a );
  vec4 layerNormalUv = texelFetch1D(
    tex, i + ${MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 7}u
  );
  m.frontLayerNormalTexCoord = layerNormalUv.r;
  m.backLayerNormalTexCoord = layerNormalUv.g;
}
`;
