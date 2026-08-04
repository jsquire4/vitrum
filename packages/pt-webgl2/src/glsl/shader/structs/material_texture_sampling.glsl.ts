/**
 * Shared explicit-validity sampler for both mapped WebGL2 material tiers.
 * Invalid descriptors, selectors, UVs, derivatives, taps, or decoded values
 * return false; the consuming material role chooses its own physical fallback.
 */
export const MATERIAL_TEXTURE_SAMPLING_GLSL = /* glsl */ `
bool materialTextureFiniteFloat( float value ) {
  return ! isnan( value ) && ! isinf( value );
}

bool materialTextureFiniteVec2( vec2 value ) {
  return ! any( isnan( value ) ) && ! any( isinf( value ) );
}

bool materialTextureFiniteVec4( vec4 value ) {
  return ! any( isnan( value ) ) && ! any( isinf( value ) );
}

bool materialTextureExactPackedInt( float value ) {
  return materialTextureFiniteFloat( value ) &&
    value >= 0.0 && value <= 2147483520.0 && value == floor( value );
}

bool decodeMaterialTexturePolicy(
  sampler2DArray tex, vec4 policy,
  out ivec2 baseSize, out ivec2 baseOffset,
  out int wrapS, out int wrapT,
  out int mipFilter, out int filterPair
) {
  baseSize = ivec2( 1 );
  baseOffset = ivec2( 0 );
  wrapS = 0;
  wrapT = 0;
  mipFilter = 0;
  filterPair = 0;
  if (
    ! materialTextureExactPackedInt( policy.x ) ||
    ! materialTextureExactPackedInt( policy.y ) ||
    ! materialTextureExactPackedInt( policy.z ) ||
    ! materialTextureExactPackedInt( policy.w )
  ) return false;

  int packedS = int( policy.x );
  int packedT = int( policy.y );
  int packedMip = int( policy.z );
  int packedFilter = int( policy.w );
  wrapS = packedS - ( packedS / 4 ) * 4;
  wrapT = packedT - ( packedT / 4 ) * 4;
  mipFilter = packedMip - ( packedMip / 4 ) * 4;
  filterPair = packedFilter - ( packedFilter / 4 ) * 4;
  if ( wrapS > 2 || wrapT > 2 || mipFilter > 2 || filterPair > 3 ) return false;

  ivec3 storage = textureSize( tex, 0 );
  if ( any( lessThanEqual( storage, ivec3( 0 ) ) ) ) return false;
  ivec2 encodedSize = ivec2( packedS / 4, packedT / 4 );
  if ( any( lessThanEqual( encodedSize, ivec2( 0 ) ) ) ) return false;
  baseSize = encodedSize;
  baseOffset = ivec2( packedMip / 4, packedFilter / 4 );
  if (
    any( lessThanEqual( baseSize, ivec2( 0 ) ) ) ||
    any( lessThan( baseOffset, ivec2( 0 ) ) ) ||
    any( greaterThan( baseSize, storage.xy ) ) ||
    any( greaterThan( baseOffset, storage.xy - baseSize ) )
  ) return false;
  return true;
}

bool materialTextureSourceInfo(
  sampler2DArray tex, vec4 policy, int level,
  out ivec2 size, out ivec2 sourceOffset,
  out int wrapS, out int wrapT
) {
  size = ivec2( 1 );
  sourceOffset = ivec2( 0 );
  wrapS = 0;
  wrapT = 0;
  int mipFilter;
  int filterPair;
  ivec2 baseSize;
  ivec2 baseOffset;
  if (
    level < 0 || level > 30 ||
    ! decodeMaterialTexturePolicy(
      tex, policy, baseSize, baseOffset,
      wrapS, wrapT, mipFilter, filterPair
    )
  ) return false;
  int divisor = 1 << level;
  size = max( ivec2( 1 ), baseSize / divisor );
  sourceOffset = baseOffset / divisor;
  ivec2 storageSize = textureSize( tex, level ).xy;
  if (
    any( lessThanEqual( storageSize, ivec2( 0 ) ) ) ||
    any( greaterThan( size, storageSize ) ) ||
    any( greaterThan( sourceOffset, storageSize - size ) )
  ) return false;
  return true;
}

bool materialTextureSourceSize(
  sampler2DArray tex, vec4 policy, int level, out ivec2 size
) {
  ivec2 sourceOffset;
  int wrapS;
  int wrapT;
  return materialTextureSourceInfo(
    tex, policy, level, size, sourceOffset, wrapS, wrapT
  );
}

int wrapMaterialTextureIndex( int coord, int size, int mode ) {
  if ( mode == 1 ) return clamp( coord, 0, size - 1 );
  if ( mode == 2 ) {
    int period = max( size * 2, 1 );
    int c = coord - period * int( floor( float( coord ) / float( period ) ) );
    return c < size ? c : period - 1 - c;
  }
  return coord - size * int( floor( float( coord ) / float( size ) ) );
}

bool decodeMaterialTextureTexel(
  vec4 encoded, bool srgbRgb, out vec4 value
) {
  value = vec4( 0.0 );
  if ( ! materialTextureFiniteVec4( encoded ) ) return false;
  if ( ! srgbRgb ) {
    value = encoded;
    return true;
  }
  if ( any( lessThan( encoded.rgb, vec3( 0.0 ) ) ) ) return false;
  bvec3 cutoff = lessThanEqual( encoded.rgb, vec3( 0.04045 ) );
  vec3 low = encoded.rgb / 12.92;
  vec3 high = pow( ( encoded.rgb + 0.055 ) / 1.055, vec3( 2.4 ) );
  value = vec4( mix( high, low, cutoff ), encoded.a );
  return materialTextureFiniteVec4( value );
}

bool decodeMaterialTextureNormal(
  vec4 sampleValue, vec2 xyScale, out vec3 tangentNormal
) {
  tangentNormal = vec3( 0.0, 0.0, 1.0 );
  if (
    ! materialTextureFiniteVec4( sampleValue ) ||
    ! materialTextureFiniteVec2( xyScale ) ||
    any( lessThan( sampleValue.rgb, vec3( 0.0 ) ) ) ||
    any( greaterThan( sampleValue.rgb, vec3( 1.0 ) ) )
  ) return false;
  vec3 decoded = sampleValue.rgb * 2.0 - 1.0;
  float scale = max( 1.0, max( abs( xyScale.x ), abs( xyScale.y ) ) );
  vec3 scaled = vec3(
    decoded.x * ( xyScale.x / scale ),
    decoded.y * ( xyScale.y / scale ),
    decoded.z / scale
  );
  float componentScale = max(
    abs( scaled.x ), max( abs( scaled.y ), abs( scaled.z ) )
  );
  if (
    ! materialTextureFiniteVec4( vec4( scaled, componentScale ) ) ||
    ! ( componentScale > 0.0 )
  ) return false;
  vec3 normalizedInput = scaled / componentScale;
  float magnitude = length( normalizedInput );
  if ( ! materialTextureFiniteFloat( magnitude ) || ! ( magnitude > 0.0 ) ) {
    return false;
  }
  tangentNormal = normalizedInput / magnitude;
  return materialTextureFiniteVec4( vec4( tangentNormal, 0.0 ) );
}

bool fetchMaterialTextureTexel(
  sampler2DArray tex, ivec2 coord, int layer, int level,
  bool srgbRgb, out vec4 value
) {
  value = vec4( 0.0 );
  ivec3 storage = textureSize( tex, level );
  if (
    layer < 0 || layer >= storage.z ||
    any( lessThan( coord, ivec2( 0 ) ) ) ||
    any( greaterThanEqual( coord, storage.xy ) )
  ) return false;
  return decodeMaterialTextureTexel(
    texelFetch( tex, ivec3( coord, layer ), level ), srgbRgb, value
  );
}

bool sampleMaterialTextureNearestLevel(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy, int level,
  bool srgbRgb, out vec4 value
) {
  value = vec4( 0.0 );
  ivec2 size;
  ivec2 sourceOffset;
  int wrapS;
  int wrapT;
  if (
    ! materialTextureSourceInfo(
      tex, policy, level, size, sourceOffset, wrapS, wrapT
    )
  ) return false;
  vec2 pFloat = floor( uv * vec2( size ) );
  if (
    ! materialTextureFiniteVec2( pFloat ) ||
    any( greaterThan( abs( pFloat ), vec2( 1073741824.0 ) ) )
  ) return false;
  ivec2 p = ivec2( pFloat );
  ivec2 wrapped = ivec2(
    wrapMaterialTextureIndex( p.x, size.x, wrapS ),
    wrapMaterialTextureIndex( p.y, size.y, wrapT )
  );
  return fetchMaterialTextureTexel(
    tex, sourceOffset + wrapped, layer, level, srgbRgb, value
  );
}

bool sampleMaterialTextureLinearLevel(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy, int level,
  bool srgbRgb, out vec4 value
) {
  value = vec4( 0.0 );
  ivec2 size;
  ivec2 sourceOffset;
  int wrapS;
  int wrapT;
  if (
    ! materialTextureSourceInfo(
      tex, policy, level, size, sourceOffset, wrapS, wrapT
    )
  ) return false;
  vec2 p = uv * vec2( size ) - vec2( 0.5 );
  vec2 p0Float = floor( p );
  if (
    ! materialTextureFiniteVec2( p ) ||
    ! materialTextureFiniteVec2( p0Float ) ||
    any( greaterThan( abs( p0Float ), vec2( 1073741824.0 ) ) )
  ) return false;
  ivec2 p0 = ivec2( p0Float );
  vec2 f = fract( p );
  int x0 = wrapMaterialTextureIndex( p0.x, size.x, wrapS );
  int y0 = wrapMaterialTextureIndex( p0.y, size.y, wrapT );
  int x1 = wrapMaterialTextureIndex( p0.x + 1, size.x, wrapS );
  int y1 = wrapMaterialTextureIndex( p0.y + 1, size.y, wrapT );
  vec4 c00;
  vec4 c10;
  vec4 c01;
  vec4 c11;
  bool valid00 = fetchMaterialTextureTexel(
    tex, sourceOffset + ivec2( x0, y0 ), layer, level, srgbRgb, c00
  );
  bool valid10 = fetchMaterialTextureTexel(
    tex, sourceOffset + ivec2( x1, y0 ), layer, level, srgbRgb, c10
  );
  bool valid01 = fetchMaterialTextureTexel(
    tex, sourceOffset + ivec2( x0, y1 ), layer, level, srgbRgb, c01
  );
  bool valid11 = fetchMaterialTextureTexel(
    tex, sourceOffset + ivec2( x1, y1 ), layer, level, srgbRgb, c11
  );
  if ( ! valid00 || ! valid10 || ! valid01 || ! valid11 ) return false;
  value = mix( mix( c00, c10, f.x ), mix( c01, c11, f.x ), f.y );
  return materialTextureFiniteVec4( value );
}

bool sampleMaterialTextureLevel(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy,
  int level, bool linearFilter, bool srgbRgb, out vec4 value
) {
  return linearFilter
    ? sampleMaterialTextureLinearLevel(
        tex, uv, layer, policy, level, srgbRgb, value
      )
    : sampleMaterialTextureNearestLevel(
        tex, uv, layer, policy, level, srgbRgb, value
      );
}

float materialTextureRawLod( vec2 uv, vec2 baseSize ) {
  vec2 dx = dFdx( uv * baseSize );
  vec2 dy = dFdy( uv * baseSize );
  return max( log2( max( max( length( dx ), length( dy ) ), 1e-8 ) ), 0.0 );
}

bool sampleMaterialTexture(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy,
  bool srgbRgb, out vec4 value
) {
  value = vec4( 0.0 );
  ivec3 storage = textureSize( tex, 0 );
  if (
    layer < 0 || layer >= storage.z ||
    ! materialTextureFiniteVec2( uv )
  ) return false;
  ivec2 baseSizeI;
  ivec2 baseOffset;
  int wrapS;
  int wrapT;
  int mipFilter;
  int filterPair;
  if (
    ! decodeMaterialTexturePolicy(
      tex, policy, baseSizeI, baseOffset,
      wrapS, wrapT, mipFilter, filterPair
    )
  ) return false;
  vec2 baseSize = vec2( baseSizeI );
  vec2 baseCoord = uv * baseSize;
  if (
    ! materialTextureFiniteVec2( baseCoord ) ||
    any( greaterThan( abs( baseCoord ), vec2( 1073741824.0 ) ) )
  ) return false;
  float rawLod = materialTextureRawLod( uv, baseSize );
  if ( ! materialTextureFiniteFloat( rawLod ) ) return false;
  bool minifying = rawLod > 0.0;
  int magFilter = filterPair - ( filterPair / 2 ) * 2;
  int minFilter = filterPair / 2;
  bool linearFilter = ( minifying ? minFilter : magFilter ) == 1;
  int maxLevel = max(
    0, int( floor( log2( float( max( baseSizeI.x, baseSizeI.y ) ) ) ) )
  );
  if ( mipFilter == 0 || maxLevel == 0 ) {
    return sampleMaterialTextureLevel(
      tex, uv, layer, policy, 0, linearFilter, srgbRgb, value
    );
  }
  if ( mipFilter == 1 ) {
    int level = clamp( int( floor( rawLod + 0.5 ) ), 0, maxLevel );
    return sampleMaterialTextureLevel(
      tex, uv, layer, policy, level, linearFilter, srgbRgb, value
    );
  }
  float clampedLod = clamp( rawLod, 0.0, float( maxLevel ) );
  int level0 = int( floor( clampedLod ) );
  int level1 = min( level0 + 1, maxLevel );
  float t = clampedLod - float( level0 );
  vec4 a;
  vec4 b;
  bool validA = sampleMaterialTextureLevel(
    tex, uv, layer, policy, level0, linearFilter, srgbRgb, a
  );
  bool validB = sampleMaterialTextureLevel(
    tex, uv, layer, policy, level1, linearFilter, srgbRgb, b
  );
  if ( ! validA || ! validB ) return false;
  value = mix( a, b, t );
  return materialTextureFiniteVec4( value );
}

bool sampleMaterialTexture(
  sampler2DArray tex, vec2 uv, int layer, vec4 policy, out vec4 value
) {
  return sampleMaterialTexture( tex, uv, layer, policy, false, value );
}
`;
