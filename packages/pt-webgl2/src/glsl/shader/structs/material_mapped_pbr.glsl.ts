import {
  MATERIAL_PIXELS,
  MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET,
  MATERIAL_WRAP_TEXEL_OFFSET,
} from './materialStride.js';
import { MATERIAL_TEXTURE_SAMPLING_GLSL } from './material_texture_sampling.glsl.js';

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
  float envMapIntensity;
  mat3 mapTransform; mat3 metalnessMapTransform; mat3 roughnessMapTransform;
  mat3 emissiveMapTransform; mat3 normalMapTransform; mat3 alphaMapTransform;
  mat3 aoMapTransform; mat3 lightMapTransform; mat3 bumpMapTransform;
  vec4 mapWrap; vec4 metalnessMapWrap; vec4 roughnessMapWrap;
  vec4 emissiveMapWrap; vec4 normalMapWrap; vec4 alphaMapWrap;
  vec4 aoMapWrap; vec4 lightMapWrap; vec4 bumpMapWrap;
  vec3 spectralReflectanceCoeffs; bool hasSpectralReflectance;
};

${MATERIAL_TEXTURE_SAMPLING_GLSL}

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
