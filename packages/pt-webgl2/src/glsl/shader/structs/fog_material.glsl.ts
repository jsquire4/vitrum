import { MATERIAL_PIXELS } from './materialStride.js';

/** Minimal persistent medium state; never carry the full rich Material in RenderState. */
export const FOG_MATERIAL_GLSL = /* glsl */ `
struct MaterialControl {
  bool matte;
  bool castShadow;
  bool fogVolume;
  bool unlit;
  bool meshEmitterCastShadowDisabled;
  uint flags;
};

void readMaterialControl(
  sampler2D tex, uint materialIndex, out MaterialControl control
) {
  vec4 flagsTexel = texelFetch1D(
    tex, materialIndex * ${MATERIAL_PIXELS}u + 14u
  );
  uint packedFlags = uint( round( flagsTexel.a ) );
  control.matte = bool( flagsTexel.r );
  control.castShadow = bool( flagsTexel.g );
  control.fogVolume = bool( int( flagsTexel.b ) & 4 );
  control.unlit = bool( packedFlags & 0x20u );
  control.meshEmitterCastShadowDisabled = bool( packedFlags & 0x40u );
  control.flags = packedFlags;
}

struct FogMaterial {
  bool fogVolume;
  vec3 color;
  vec3 emission;
  float opacity;
  vec3 attenuationColor;
  float attenuationDistance;
  vec3 sigmaS;
  float anisotropy;
  bool hasSpectralAttenuation;
  bool matte;
  bool castShadow;
  bool unlit;
  bool meshEmitterCastShadowDisabled;
  uint flags;
  uint materialIndex;
};

FogMaterial readFogMaterialInfo( sampler2D tex, uint materialIndex ) {
  uint i = materialIndex * ${MATERIAL_PIXELS}u;
  vec4 s2 = texelFetch1D( tex, i + 2u );
  vec4 s3 = texelFetch1D( tex, i + 3u );
  vec4 s14 = texelFetch1D( tex, i + 14u );
  vec4 s15 = texelFetch1D( tex, i + 15u );
  vec4 s16 = texelFetch1D( tex, i + 16u );
  vec4 s17 = texelFetch1D( tex, i + 17u );
  vec4 s12 = texelFetch1D( tex, i + 12u );
  uint packedFlags = uint( round( s14.a ) );
  FogMaterial fog;
  fog.fogVolume = bool( int( s14.b ) & 4 );
  float sigmaT = max( s15.r, 0.0 );
  fog.color = sigmaT > 0.0 ? clamp( s16.rgb / sigmaT, vec3( 0.0 ), vec3( 1.0 ) ) : vec3( 0.0 );
  fog.emission = s2.a * s3.rgb;
  // The absorbed fork named this lane opacity; for a participating medium it
  // is the majorant extinction coefficient used by free-flight sampling.
  fog.opacity = sigmaT;
  fog.attenuationColor = s12.rgb;
  fog.attenuationDistance = s12.a;
  fog.sigmaS = max( s16.rgb, vec3( 0.0 ) );
  fog.anisotropy = clamp( s15.g, -1.0, 1.0 );
  fog.hasSpectralAttenuation = bool( uint( round( s17.a ) ) & 1u );
  fog.matte = bool( s14.r );
  fog.castShadow = bool( s14.g );
  fog.unlit = bool( packedFlags & 0x20u );
  fog.meshEmitterCastShadowDisabled = bool( packedFlags & 0x40u );
  fog.flags = packedFlags;
  fog.materialIndex = materialIndex;
  return fog;
}

const int MEDIUM_STACK_CAPACITY = 8;

struct MediumStack {
  int count;
  uint materialIds[ MEDIUM_STACK_CAPACITY ];
};

void initMediumStack( out MediumStack stack ) {
  stack.count = 0;
  for ( int i = 0; i < MEDIUM_STACK_CAPACITY; i ++ ) {
    stack.materialIds[ i ] = 0u;
  }
}

void refreshMediumFromStack(
  const in MediumStack stack,
  sampler2D tex,
  inout FogMaterial fog
) {
  if ( stack.count <= 0 ) {
    fog.fogVolume = false;
    return;
  }
  fog = readFogMaterialInfo( tex, stack.materialIds[ stack.count - 1 ] );
  fog.fogVolume = true;
}

bool enterMedium(
  inout MediumStack stack,
  uint materialId,
  sampler2D tex,
  inout FogMaterial fog
) {
  if ( stack.count >= MEDIUM_STACK_CAPACITY ) return false;
  stack.materialIds[ stack.count ] = materialId;
  stack.count ++;
  refreshMediumFromStack( stack, tex, fog );
  return true;
}

bool leaveMedium(
  inout MediumStack stack,
  uint materialId,
  sampler2D tex,
  inout FogMaterial fog
) {
  // A closed nested-medium walk is LIFO. Removing a matching material from
  // below the top would accept an out-of-order back face and silently corrupt
  // the enclosing Beer/free-flight state used by eye, shadow, and BDPT paths.
  if ( stack.count <= 0 ) return false;
  int top = stack.count - 1;
  if ( stack.materialIds[ top ] != materialId ) return false;
  stack.materialIds[ top ] = 0u;
  stack.count = top;
  refreshMediumFromStack( stack, tex, fog );
  return true;
}
`;
