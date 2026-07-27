import {
  MATERIAL_PIXELS,
  MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET,
} from './materialStride.js';

/**
 * Material decoder for scenes with no material texture fetches. It retains
 * every scalar/vector transport field, including volume, spectral, layers,
 * anisotropy, Disney lobes, and multi-layer thin film.
 */
export const MATERIAL_SCALAR_RICH_GLSL = /* glsl */ `
struct Material {
  vec3 color;
  float metalness;
  float roughness;
  float ior;
  float transmission;
  vec3 emissive;
  float emissiveIntensity;
  float clearcoat;
  float clearcoatRoughness;
  float sheen;
  vec3 sheenColor;
  float sheenRoughness;
  float iridescence;
  float iridescenceIor;
  float iridescenceThicknessMinimum;
  float iridescenceThicknessMaximum;
  vec3 specularColor;
  float specularIntensity;
  bool thinFilm;
  float anisotropy;
  float anisotropyRotation;
  vec3 attenuationColor;
  float attenuationDistance;
  float thickness;
  float opacity;
  float alphaTest;
  float side;
  bool matte;
  bool castShadow;
  bool vertexColors;
  bool transparent;
  bool unlit;
  bool meshEmitterCastShadowDisabled;
  bool fogVolume;
  uint flags;
  float sssSigmaT;
  float sssAnisotropyG;
  vec3 sssSigmaS;
  float dispersionStrength;
  float thinFilmEnabled;
  float thinFilmLayerCount;
  float thinFilmIncidentIor;
  bool thinFilmAngleDependent;
  bool hasSpectralAttenuation;
  vec3 frontLayerTransmission;
  float frontLayerRoughness;
  bool hasFrontLayer;
  vec3 backLayerTransmission;
  float backLayerRoughness;
  bool hasBackLayer;
  vec3 spectralReflectanceCoeffs;
  bool hasSpectralReflectance;
  float envMapIntensity;
};

void readMaterialInfo( sampler2D tex, uint index, out Material m ) {
  uint i = index * ${MATERIAL_PIXELS}u;
  vec4 s0 = texelFetch1D( tex, i + 0u );
  vec4 s1 = texelFetch1D( tex, i + 1u );
  vec4 s2 = texelFetch1D( tex, i + 2u );
  vec4 s3 = texelFetch1D( tex, i + 3u );
  vec4 s4 = texelFetch1D( tex, i + 4u );
  vec4 s5 = texelFetch1D( tex, i + 5u );
  vec4 s6 = texelFetch1D( tex, i + 6u );
  vec4 s7 = texelFetch1D( tex, i + 7u );
  vec4 s8 = texelFetch1D( tex, i + 8u );
  vec4 s9 = texelFetch1D( tex, i + 9u );
  vec4 s10 = texelFetch1D( tex, i + 10u );
  vec4 s11 = texelFetch1D( tex, i + 11u );
  vec4 s12 = texelFetch1D( tex, i + 12u );
  vec4 s13 = texelFetch1D( tex, i + 13u );
  vec4 s14 = texelFetch1D( tex, i + 14u );
  vec4 s15 = texelFetch1D( tex, i + 15u );
  vec4 s16 = texelFetch1D( tex, i + 16u );
  vec4 s17 = texelFetch1D( tex, i + 17u );
  vec4 s18 = texelFetch1D( tex, i + 18u );
  vec4 s19 = texelFetch1D( tex, i + 19u );
  vec4 s20 = texelFetch1D( tex, i + 85u );
  vec4 s22 = texelFetch1D( tex, i + 97u );
  vec4 spectral = texelFetch1D(
    tex, i + ${MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET}u
  );
  uint packedFlags = uint( round( s14.a ) );
  uint featureFlags = uint( round( s17.a ) );

  m.color = s0.rgb;
  m.metalness = s1.r;
  m.roughness = s1.b;
  m.ior = max( s2.r, 1.0 );
  m.transmission = s2.g;
  m.emissiveIntensity = s2.a;
  m.emissive = s3.rgb;
  m.clearcoat = s4.a;
  m.clearcoatRoughness = s5.g;
  m.anisotropy = clamp( s11.a, 0.0, 1.0 );
  m.sheen = s6.a;
  m.sheenColor = s7.rgb;
  m.sheenRoughness = s8.r;
  m.iridescence = s9.r;
  m.iridescenceIor = s9.g;
  m.iridescenceThicknessMinimum = s9.b;
  m.iridescenceThicknessMaximum = s9.a;
  m.specularColor = s10.rgb;
  m.specularIntensity = s11.r;
  m.thinFilm = bool( s11.b );
  m.attenuationColor = s12.rgb;
  m.attenuationDistance = s12.a;
  m.thickness = max( s22.r, 0.0 );
  m.opacity = s13.g;
  m.alphaTest = s13.b;
  m.side = s13.a;
  m.matte = bool( s14.r );
  m.castShadow = bool( s14.g );
  m.vertexColors = bool( int( s14.b ) & 1 );
  m.fogVolume = bool( int( s14.b ) & 4 );
  m.transparent = bool( packedFlags & 1u );
  m.unlit = bool( packedFlags & 0x20u );
  m.meshEmitterCastShadowDisabled = bool( packedFlags & 0x40u );
  m.flags = packedFlags;
  m.sssSigmaT = s15.r;
  m.sssAnisotropyG = s15.g;
  m.dispersionStrength = s15.b;
  m.thinFilmEnabled = s15.a;
  m.sssSigmaS = s16.rgb;
  m.thinFilmLayerCount = s16.a;
  m.thinFilmIncidentIor = max( s17.r, 1.0 );
  m.thinFilmAngleDependent = s17.g > 0.5;
  m.anisotropyRotation = s17.b;
  m.hasSpectralAttenuation = bool( featureFlags & 1u );
  m.hasFrontLayer = bool( featureFlags & 2u );
  m.hasBackLayer = bool( featureFlags & 4u );
  m.frontLayerTransmission = s18.rgb;
  m.frontLayerRoughness = s18.a;
  m.backLayerTransmission = s19.rgb;
  m.backLayerRoughness = s19.a;
  m.spectralReflectanceCoeffs = spectral.xyz;
  m.hasSpectralReflectance = spectral.w > 0.5;
  m.envMapIntensity = s20.a;
}
`;
