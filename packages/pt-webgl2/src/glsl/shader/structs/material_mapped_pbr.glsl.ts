import {
  MATERIAL_PIXELS,
  MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET,
  MATERIAL_WRAP_TEXEL_OFFSET,
} from './materialStride.js';

/** Material decoder for texture-capable opaque base PBR scenes. */
export const MATERIAL_MAPPED_PBR_GLSL = /* glsl */ `
struct Material {
  vec3 color; int map;
  float metalness; int metalnessMap;
  float roughness; int roughnessMap;
  float ior; float transmission;
  float emissiveIntensity; vec3 emissive; int emissiveMap;
  int normalMap; vec2 normalScale;
  int alphaMap;
  bool castShadow; float opacity; float alphaTest; float side; bool matte;
  bool vertexColors; bool transparent; bool unlit;
  bool meshEmitterCastShadowDisabled; bool fogVolume; uint flags;
  int aoMap; int lightMap; int bumpMap;
  float aoMapIntensity; float lightMapIntensity; float bumpScale;
  float envMapIntensity; uint uvTexCoordMask;
  mat3 mapTransform; mat3 metalnessMapTransform; mat3 roughnessMapTransform;
  mat3 emissiveMapTransform; mat3 normalMapTransform; mat3 alphaMapTransform;
  mat3 aoMapTransform; mat3 lightMapTransform; mat3 bumpMapTransform;
  vec4 mapWrap; vec4 metalnessMapWrap; vec4 roughnessMapWrap;
  vec4 emissiveMapWrap; vec4 normalMapWrap; vec4 alphaMapWrap;
  vec4 aoMapWrap; vec4 lightMapWrap; vec4 bumpMapWrap;
  vec3 spectralReflectanceCoeffs; bool hasSpectralReflectance;
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

bool materialTextureUsesLinearFilter( vec4 policy, bool minifying ) {
  int packed = int( round( policy.w ) );
  int magFilter = packed - ( packed / 2 ) * 2;
  int minFilter = packed / 2;
  return ( minifying ? minFilter : magFilter ) == 1;
}

float materialTextureRawLod( vec2 uv, vec2 baseSize ) {
  vec2 dx = dFdx( uv * baseSize );
  vec2 dy = dFdy( uv * baseSize );
  return max( log2( max( max( length( dx ), length( dy ) ), 1e-8 ) ), 0.0 );
}

vec4 sampleMaterialTextureNearestLevel(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy, int level
) {
  ivec2 size = materialTextureSourceSize( tex, policy, level );
  ivec2 p = ivec2( floor( uv * vec2( size ) ) );
  int x = wrapMaterialTextureIndex( p.x, size.x, policy.x );
  int y = wrapMaterialTextureIndex( p.y, size.y, policy.y );
  return texelFetch( tex, ivec3( x, y, layer ), level );
}

vec4 sampleMaterialTextureLinearLevel(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy, int level
) {
  ivec2 size = materialTextureSourceSize( tex, policy, level );
  vec2 p = uv * vec2( size ) - vec2( 0.5 );
  ivec2 p0 = ivec2( floor( p ) );
  vec2 f = fract( p );
  int x0 = wrapMaterialTextureIndex( p0.x, size.x, policy.x );
  int y0 = wrapMaterialTextureIndex( p0.y, size.y, policy.y );
  int x1 = wrapMaterialTextureIndex( p0.x + 1, size.x, policy.x );
  int y1 = wrapMaterialTextureIndex( p0.y + 1, size.y, policy.y );
  vec4 c00 = texelFetch( tex, ivec3( x0, y0, layer ), level );
  vec4 c10 = texelFetch( tex, ivec3( x1, y0, layer ), level );
  vec4 c01 = texelFetch( tex, ivec3( x0, y1, layer ), level );
  vec4 c11 = texelFetch( tex, ivec3( x1, y1, layer ), level );
  return mix( mix( c00, c10, f.x ), mix( c01, c11, f.x ), f.y );
}

vec4 sampleMaterialTextureLevel(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy,
  int level, bool linearFilter
) {
  return linearFilter
    ? sampleMaterialTextureLinearLevel( tex, uv, layer, policy, level )
    : sampleMaterialTextureNearestLevel( tex, uv, layer, policy, level );
}

vec4 sampleMaterialTexture(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy
) {
  if ( layer < 0 ) return vec4( 1.0 );
  ivec2 baseSizeI = materialTextureSourceSize( tex, policy, 0 );
  vec2 baseSize = vec2( baseSizeI );
  float rawLod = materialTextureRawLod( uv, baseSize );
  bool linearFilter = materialTextureUsesLinearFilter( policy, rawLod > 0.0 );
  int mipFilter = int( round( policy.z ) );
  int maxLevel = max(
    0, int( floor( log2( float( max( baseSizeI.x, baseSizeI.y ) ) ) ) )
  );
  if ( mipFilter == 0 || maxLevel == 0 ) {
    return sampleMaterialTextureLevel(
      tex, uv, layer, policy, 0, linearFilter
    );
  }
  if ( mipFilter == 1 ) {
    int level = clamp( int( floor( rawLod + 0.5 ) ), 0, maxLevel );
    return sampleMaterialTextureLevel(
      tex, uv, layer, policy, level, linearFilter
    );
  }
  float clampedLod = clamp( rawLod, 0.0, float( maxLevel ) );
  int level0 = int( floor( clampedLod ) );
  int level1 = min( level0 + 1, maxLevel );
  float t = clampedLod - float( level0 );
  vec4 a = sampleMaterialTextureLevel(
    tex, uv, layer, policy, level0, linearFilter
  );
  vec4 b = sampleMaterialTextureLevel(
    tex, uv, layer, policy, level1, linearFilter
  );
  return mix( a, b, t );
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

void readMaterialInfo( sampler2D tex, uint index, out Material m ) {
  uint i = index * ${MATERIAL_PIXELS}u;
  vec4 s0 = texelFetch1D( tex, i + 0u );
  vec4 s1 = texelFetch1D( tex, i + 1u );
  vec4 s2 = texelFetch1D( tex, i + 2u );
  vec4 s3 = texelFetch1D( tex, i + 3u );
  vec4 s4 = texelFetch1D( tex, i + 4u );
  vec4 s13 = texelFetch1D( tex, i + 13u );
  vec4 s14 = texelFetch1D( tex, i + 14u );
  vec4 s20 = texelFetch1D( tex, i + 85u );
  vec4 s21 = texelFetch1D( tex, i + 86u );
  vec4 spectral = texelFetch1D(
    tex, i + ${MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET}u
  );
  uint packedFlags = uint( round( s14.a ) );

  m.color = s0.rgb; m.map = int( round( s0.a ) );
  m.metalness = s1.r; m.metalnessMap = int( round( s1.g ) );
  m.roughness = s1.b; m.roughnessMap = int( round( s1.a ) );
  m.ior = max( s2.r, 1.0 ); m.transmission = s2.g;
  m.emissiveIntensity = s2.a; m.emissive = s3.rgb;
  m.emissiveMap = int( round( s3.a ) );
  m.normalMap = int( round( s4.r ) ); m.normalScale = s4.gb;
  m.alphaMap = int( round( s13.r ) ); m.opacity = s13.g;
  m.alphaTest = s13.b; m.side = s13.a;
  m.matte = bool( s14.r ); m.castShadow = bool( s14.g );
  m.vertexColors = bool( int( s14.b ) & 1 );
  m.fogVolume = false;
  m.transparent = bool( packedFlags & 1u );
  m.unlit = bool( packedFlags & 0x20u );
  m.meshEmitterCastShadowDisabled = bool( packedFlags & 0x40u );
  m.flags = packedFlags;
  m.aoMap = int( round( s20.r ) );
  m.lightMap = int( round( s20.g ) );
  m.bumpMap = int( round( s20.b ) );
  m.envMapIntensity = s20.a;
  m.aoMapIntensity = s21.r;
  m.lightMapIntensity = s21.g;
  m.bumpScale = s21.b;
  m.uvTexCoordMask = uint( round( s21.a ) );
  m.spectralReflectanceCoeffs = spectral.xyz;
  m.hasSpectralReflectance = spectral.w > 0.5;

  m.mapTransform = m.map == -1
    ? mat3( 1.0 ) : readTextureTransform( tex, i + 55u );
  m.metalnessMapTransform = m.metalnessMap == -1
    ? mat3( 1.0 ) : readTextureTransform( tex, i + 57u );
  m.roughnessMapTransform = m.roughnessMap == -1
    ? mat3( 1.0 ) : readTextureTransform( tex, i + 59u );
  m.emissiveMapTransform = m.emissiveMap == -1
    ? mat3( 1.0 ) : readTextureTransform( tex, i + 63u );
  m.normalMapTransform = m.normalMap == -1
    ? mat3( 1.0 ) : readTextureTransform( tex, i + 65u );
  m.aoMapTransform = m.aoMap == -1
    ? mat3( 1.0 ) : readTextureTransform( tex, i + 87u );
  m.lightMapTransform = m.lightMap == -1
    ? mat3( 1.0 ) : readTextureTransform( tex, i + 89u );
  m.bumpMapTransform = m.bumpMap == -1
    ? mat3( 1.0 ) : readTextureTransform( tex, i + 91u );
  m.alphaMapTransform = m.alphaMap == -1
    ? mat3( 1.0 ) : readTextureTransform( tex, i + 93u );

  m.mapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 0}u );
  m.metalnessMapWrap =
    texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 1}u );
  m.roughnessMapWrap =
    texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 2}u );
  m.emissiveMapWrap =
    texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 4}u );
  m.normalMapWrap =
    texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 5}u );
  m.alphaMapWrap =
    texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 6}u );
  m.aoMapWrap =
    texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 16}u );
  m.lightMapWrap =
    texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 17}u );
  m.bumpMapWrap =
    texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 18}u );
}
`;
