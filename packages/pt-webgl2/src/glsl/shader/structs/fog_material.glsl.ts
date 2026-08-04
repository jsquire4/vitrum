import {
  MATERIAL_PIXELS,
  MATERIAL_VOLUME_THICKNESS_TEXEL,
} from './materialStride.js';

/** Minimal persistent medium state; never carry the full rich Material in RenderState. */
export const FOG_MATERIAL_GLSL = /* glsl */ `
struct MaterialControl {
  bool matte;
  bool castShadow;
  bool fogVolume;
  bool opticalVolume;
  bool thinFilm;
  bool unlit;
  bool meshEmitterCastShadowDisabled;
  uint flags;
};

const int SURFACE_COVERAGE_HOLE = 0;
const int SURFACE_COVERAGE_SOLID = 1;
const int SURFACE_COVERAGE_FRACTIONAL = 2;

int classifySurfaceCoverage(
  float materialSide,
  float hitSide,
  float alphaTest,
  bool transparent,
  float coverage
) {
  if ( materialSide != 0.0 && hitSide != materialSide ) {
    return SURFACE_COVERAGE_HOLE;
  }
  if ( isnan( coverage ) || isinf( coverage ) ) {
    return SURFACE_COVERAGE_FRACTIONAL;
  }
  float clampedCoverage = clamp( coverage, 0.0, 1.0 );
  if ( alphaTest != 0.0 ) {
    return clampedCoverage < alphaTest
      ? SURFACE_COVERAGE_HOLE
      : SURFACE_COVERAGE_SOLID;
  }
  if ( ! transparent ) return SURFACE_COVERAGE_SOLID;
  if ( clampedCoverage <= 0.0 ) return SURFACE_COVERAGE_HOLE;
  if ( clampedCoverage >= 1.0 ) return SURFACE_COVERAGE_SOLID;
  return SURFACE_COVERAGE_FRACTIONAL;
}

void readMaterialControl(
  sampler2D tex, uint materialIndex, out MaterialControl control
) {
  vec4 opticalTexel = texelFetch1D(
    tex, materialIndex * ${MATERIAL_PIXELS}u + 2u
  );
  vec4 surfaceTexel = texelFetch1D(
    tex, materialIndex * ${MATERIAL_PIXELS}u + 11u
  );
  vec4 flagsTexel = texelFetch1D(
    tex, materialIndex * ${MATERIAL_PIXELS}u + 14u
  );
  uint packedFlags = uint( round( flagsTexel.a ) );
  control.matte = bool( flagsTexel.r );
  control.castShadow = bool( flagsTexel.g );
  control.fogVolume = bool( int( flagsTexel.b ) & 4 );
  control.thinFilm = bool( surfaceTexel.b );
  control.opticalVolume = opticalTexel.g > 0.0 && ! control.thinFilm;
  control.unlit = bool( packedFlags & 0x20u );
  control.meshEmitterCastShadowDisabled = bool( packedFlags & 0x40u );
  control.flags = packedFlags;
}

struct FogMaterial {
  bool fogVolume;
  bool opticalVolume;
  bool thinFilm;
  vec3 color;
  vec3 emission;
  float opacity;
  vec3 attenuationColor;
  float attenuationDistance;
  vec3 sigmaS;
  float anisotropy;
  bool hasSpectralAttenuation;
  float ior;
  float dispersionStrength;
  float attenuationThickness;
  bool hasAttenuationThickness;
  bool matte;
  bool castShadow;
  bool unlit;
  bool meshEmitterCastShadowDisabled;
  uint flags;
  uint materialIndex;
};

void initFogMaterial( out FogMaterial fog ) {
  fog.fogVolume = false;
  fog.opticalVolume = false;
  fog.thinFilm = false;
  fog.color = vec3( 0.0 );
  fog.emission = vec3( 0.0 );
  fog.opacity = 0.0;
  fog.attenuationColor = vec3( 1.0 );
  fog.attenuationDistance = INFINITY;
  fog.sigmaS = vec3( 0.0 );
  fog.anisotropy = 0.0;
  fog.hasSpectralAttenuation = false;
  fog.ior = 1.0;
  fog.dispersionStrength = 0.0;
  fog.attenuationThickness = 0.0;
  fog.hasAttenuationThickness = false;
  fog.matte = false;
  fog.castShadow = true;
  fog.unlit = false;
  fog.meshEmitterCastShadowDisabled = false;
  fog.flags = 0u;
  fog.materialIndex = 0u;
}

FogMaterial readFogMaterialInfo( sampler2D tex, uint materialIndex ) {
  uint i = materialIndex * ${MATERIAL_PIXELS}u;
  vec4 s2 = texelFetch1D( tex, i + 2u );
  vec4 s3 = texelFetch1D( tex, i + 3u );
  vec4 s14 = texelFetch1D( tex, i + 14u );
  vec4 s15 = texelFetch1D( tex, i + 15u );
  vec4 s16 = texelFetch1D( tex, i + 16u );
  vec4 s17 = texelFetch1D( tex, i + 17u );
  vec4 s11 = texelFetch1D( tex, i + 11u );
  vec4 s97 = texelFetch1D(
    tex, i + ${MATERIAL_VOLUME_THICKNESS_TEXEL}u
  );
  vec4 s12 = texelFetch1D( tex, i + 12u );
  uint packedFlags = uint( round( s14.a ) );
  FogMaterial fog;
  fog.fogVolume = bool( int( s14.b ) & 4 );
  fog.thinFilm = bool( s11.b );
  fog.opticalVolume = s2.g > 0.0 && ! fog.thinFilm;
  // The host packs a finite proposal coefficient even when an authored RGB
  // extinction lane is +Infinity (attenuationColor == 0). Treat a malformed
  // non-finite payload as an inactive proposal so sigmaS / sigmaT never emits
  // NaN and contaminates otherwise independent color channels.
  float sigmaT = ! isnan( s15.r ) && ! isinf( s15.r ) && s15.r > 0.0
    ? s15.r
    : 0.0;
  // Free flight is sampled from the exact RGB-mixture (or scalar hero)
  // proposal in the shader. The collision weight owns its marginal density,
  // so the phase vertex carries the authored sigmaS directly.
  fog.color = max( s16.rgb, vec3( 0.0 ) );
  fog.emission = s2.a * s3.rgb;
  // The absorbed fork named this lane opacity; for a participating medium it
  // is the majorant extinction coefficient used by free-flight sampling.
  fog.opacity = sigmaT;
  fog.attenuationColor = s12.rgb;
  fog.attenuationDistance = s12.a;
  fog.sigmaS = max( s16.rgb, vec3( 0.0 ) );
  fog.anisotropy = clamp( s15.g, -1.0, 1.0 );
  fog.hasSpectralAttenuation = bool( uint( round( s17.a ) ) & 1u );
  // Core reserves exactly zero as the infinite-IOR compatibility limit. Keep
  // that sentinel intact; clamping it to air would change both Fresnel and the
  // nested interface on every reconstructed stack path.
  fog.ior = max( s2.r, 0.0 );
  fog.dispersionStrength = max( s15.b, 0.0 );
  fog.attenuationThickness = max( s97.r, 0.0 );
  fog.hasAttenuationThickness = s97.r > 0.0;
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
  uint boundaryIds[ MEDIUM_STACK_CAPACITY ];
  uint materialIds[ MEDIUM_STACK_CAPACITY ];
  // The entry-sampled authored cap is immutable while a layer is live. Keep it
  // beside the mutable remainder so two independently-built BDPT histories can
  // reconcile cumulative consumption instead of treating either remainder as
  // a fresh cap at their connection edge.
  float initialAttenuationThicknesses[ MEDIUM_STACK_CAPACITY ];
  float attenuationThicknesses[ MEDIUM_STACK_CAPACITY ];
  bool hasAttenuationThicknesses[ MEDIUM_STACK_CAPACITY ];
};

void initMediumStack( out MediumStack stack ) {
  stack.count = 0;
  for ( int i = 0; i < MEDIUM_STACK_CAPACITY; i ++ ) {
    stack.boundaryIds[ i ] = 0u;
    stack.materialIds[ i ] = 0u;
    stack.initialAttenuationThicknesses[ i ] = 0.0;
    stack.attenuationThicknesses[ i ] = 0.0;
    stack.hasAttenuationThicknesses[ i ] = false;
  }
}

void refreshMediumFromStack(
  const in MediumStack stack,
  sampler2D tex,
  inout FogMaterial fog
) {
  if ( stack.count <= 0 ) {
    initFogMaterial( fog );
    return;
  }
  int top = stack.count - 1;
  fog = readFogMaterialInfo( tex, stack.materialIds[ top ] );
  // A mapped cap is sampled at entry/start-inside classification and owns the
  // complete interior segment. Restore that exact entry whenever a pop/filter
  // reveals it instead of falling back to the material-wide scalar payload.
  fog.attenuationThickness = stack.attenuationThicknesses[ top ];
  fog.hasAttenuationThickness = stack.hasAttenuationThicknesses[ top ];
}

// Build the medium state seen by a visibility ray. castShadow:false removes
// that material's volume extinction as well as its interface occlusion, while
// retaining any shadow-casting outer media in their original nesting order.
void filterShadowMediumStack(
  const in MediumStack source,
  sampler2D tex,
  out MediumStack filtered,
  inout FogMaterial fog
) {
  initMediumStack( filtered );
  for ( int i = 0; i < MEDIUM_STACK_CAPACITY; i ++ ) {
    if ( i >= source.count ) break;
    uint materialId = source.materialIds[ i ];
    FogMaterial candidate = readFogMaterialInfo( tex, materialId );
    if ( candidate.castShadow ) {
      filtered.boundaryIds[ filtered.count ] = source.boundaryIds[ i ];
      filtered.materialIds[ filtered.count ] = materialId;
      filtered.initialAttenuationThicknesses[ filtered.count ] =
        source.initialAttenuationThicknesses[ i ];
      filtered.attenuationThicknesses[ filtered.count ] =
        source.attenuationThicknesses[ i ];
      filtered.hasAttenuationThicknesses[ filtered.count ] =
        source.hasAttenuationThicknesses[ i ];
      filtered.count ++;
    }
  }
  refreshMediumFromStack( filtered, tex, fog );
}

// KHR_materials_volume thickness is a path-distance budget, not a fresh clamp
// for every segment between null/medium vertices. Only the current innermost
// medium owns a segment. An unauthored cap remains unbounded; a finite cap is
// consumed monotonically while the boundary stays live on the stack.
float mediumEffectiveSegmentDistance(
  const in MediumStack stack,
  float geometricDistance
) {
  if ( isnan( geometricDistance ) || geometricDistance < 0.0 ) return -1.0;
  if ( stack.count <= 0 ) return geometricDistance;
  int top = stack.count - 1;
  if ( ! stack.hasAttenuationThicknesses[ top ] ) {
    return geometricDistance;
  }
  float remainingDistance = stack.attenuationThicknesses[ top ];
  if (
    isnan( remainingDistance ) || isinf( remainingDistance ) ||
    remainingDistance < 0.0
  ) return -1.0;
  return min( geometricDistance, remainingDistance );
}

bool consumeMediumSegmentDistance(
  inout MediumStack stack,
  float geometricDistance,
  sampler2D tex,
  inout FogMaterial fog
) {
  float effectiveDistance = mediumEffectiveSegmentDistance(
    stack, geometricDistance
  );
  if ( effectiveDistance < 0.0 ) return false;
  if ( stack.count <= 0 ) return true;
  int top = stack.count - 1;
  if ( stack.hasAttenuationThicknesses[ top ] ) {
    // effectiveDistance is finite whenever the authored remainder is finite,
    // so this branch never forms Infinity - Infinity.
    stack.attenuationThicknesses[ top ] = max(
      stack.attenuationThicknesses[ top ] - effectiveDistance,
      0.0
    );
    refreshMediumFromStack( stack, tex, fog );
  }
  return true;
}

bool enterMedium(
  inout MediumStack stack,
  uint boundaryId,
  uint materialId,
  bool hasAttenuationThickness,
  float attenuationThickness,
  sampler2D tex,
  inout FogMaterial fog
) {
  if ( stack.count >= MEDIUM_STACK_CAPACITY ) return false;
  if ( boundaryId == 0u ) return false;
  if (
    hasAttenuationThickness &&
    ( attenuationThickness < 0.0 ||
      isnan( attenuationThickness ) || isinf( attenuationThickness ) )
  ) return false;
  stack.boundaryIds[ stack.count ] = boundaryId;
  stack.materialIds[ stack.count ] = materialId;
  stack.initialAttenuationThicknesses[ stack.count ] =
    hasAttenuationThickness ? attenuationThickness : 0.0;
  stack.attenuationThicknesses[ stack.count ] =
    hasAttenuationThickness ? attenuationThickness : 0.0;
  stack.hasAttenuationThicknesses[ stack.count ] =
    hasAttenuationThickness;
  stack.count ++;
  refreshMediumFromStack( stack, tex, fog );
  return true;
}

bool leaveMedium(
  inout MediumStack stack,
  uint boundaryId,
  uint materialId,
  sampler2D tex,
  inout FogMaterial fog
) {
  // A closed nested-medium walk is LIFO. Removing a matching material from
  // below the top would accept an out-of-order back face and silently corrupt
  // the enclosing Beer/free-flight state used by eye, shadow, and BDPT paths.
  if ( stack.count <= 0 ) return false;
  int top = stack.count - 1;
  if (
    stack.boundaryIds[ top ] != boundaryId ||
    stack.materialIds[ top ] != materialId
  ) return false;
  stack.boundaryIds[ top ] = 0u;
  stack.materialIds[ top ] = 0u;
  stack.initialAttenuationThicknesses[ top ] = 0.0;
  stack.attenuationThicknesses[ top ] = 0.0;
  stack.hasAttenuationThicknesses[ top ] = false;
  stack.count = top;
  refreshMediumFromStack( stack, tex, fog );
  return true;
}
`;
