import { MATERIAL_PIXELS } from './materialStride.js';

/** Compact decoder for scenes proven to use only opaque base PBR fields. */
export const MATERIAL_BASIC_GLSL = /* glsl */ `
  struct Material {
    vec3 color;
    float metalness;
    float roughness;
    float ior;
    float transmission;
    vec3 emissive;
    float emissiveIntensity;
    bool matte;
    bool castShadow;
    bool vertexColors;
    float side;
    bool fogVolume;
    bool unlit;
    bool meshEmitterCastShadowDisabled;
    uint flags;
  };

  void readMaterialInfo( sampler2D tex, uint index, out Material material ) {
    uint i = index * ${MATERIAL_PIXELS}u;
    vec4 s0 = texelFetch1D( tex, i + 0u );
    vec4 s1 = texelFetch1D( tex, i + 1u );
    vec4 s2 = texelFetch1D( tex, i + 2u );
    vec4 s3 = texelFetch1D( tex, i + 3u );
    vec4 s13 = texelFetch1D( tex, i + 13u );
    vec4 s14 = texelFetch1D( tex, i + 14u );
    uint packedFlags = uint( round( s14.a ) );

    material.color = s0.rgb;
    material.metalness = s1.r;
    material.roughness = s1.b;
    material.ior = max( s2.r, 1.0 );
    material.transmission = 0.0;
    material.emissiveIntensity = s2.a;
    material.emissive = s3.rgb;
    material.matte = bool( s14.r );
    material.castShadow = bool( s14.g );
    material.vertexColors = bool( int( s14.b ) & 1 );
    material.side = s13.a;
    material.fogVolume = false;
    material.unlit = bool( packedFlags & 0x20u );
    material.meshEmitterCastShadowDisabled = bool( packedFlags & 0x40u );
    material.flags = packedFlags;
  }
`;
