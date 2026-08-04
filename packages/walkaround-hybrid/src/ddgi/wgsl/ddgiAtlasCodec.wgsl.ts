/**
 * Range-preserving block-float codec for DDGI's required rgba16float atlases.
 *
 * Irradiance stores one shared exponent for each RGB SH coefficient:
 *   rgb = f16 mantissa, a = encoded power-of-two exponent.
 * Visibility stores independent exponents for its two moment lanes:
 *   r/g = f16 mantissas, b/a = encoded exponents.
 *
 * Legacy compatibility is self-describing. Exponent lanes 0 and 1 both mean
 * exponent zero, so cleared rgba=0 texels, historical irradiance alpha=1, and
 * historical visibility (b=0,a=1) all decode with an exact scale of one.
 * Negative exponents are stored directly; positive exponent e is stored as
 * e+1, leaving both legacy sentinels unambiguous. Metadata must be an exact
 * integer in the codec range or the decode fails closed.
 */
export const DDGI_ATLAS_CODEC_WGSL = /* wgsl */`
const DDGI_ATLAS_F32_MAX: f32 = 3.4028234663852886e38;
const DDGI_ATLAS_F16_MAX: f32 = 65504.0;
const DDGI_ATLAS_F16_MIN_SUBNORMAL: f32 = 5.960464477539063e-8;
const DDGI_ATLAS_SAFE_MANTISSA: f32 = 16384.0;
// Exact decimal value of f32 bits 0x5f7fefff. Interpolating the JS number emits
// 18442239374570553000, an abstract-integer literal that portable WGSL parsers
// reject; bitcast is likewise not portable in a constant expression. This
// abstract-float scientific literal converts to the authoritative f32 exactly.
const DDGI_ATLAS_VISIBILITY_DISTANCE_MAX: f32 = 1.8442239374570553344e19;
const DDGI_ATLAS_INVALID_EXPONENT: i32 = 1000;

fn ddgiAtlasFiniteScalar(value: f32) -> bool {
  return value == value && abs(value) <= DDGI_ATLAS_F32_MAX;
}

fn ddgiAtlasFiniteVec3(value: vec3f) -> bool {
  return all(value == value) && all(abs(value) <= vec3f(DDGI_ATLAS_F32_MAX));
}

fn ddgiAtlasDecodeExponent(lane: f32) -> i32 {
  if (
    !ddgiAtlasFiniteScalar(lane) ||
    lane != round(lane) ||
    lane < -149.0 ||
    lane > 115.0
  ) {
    return DDGI_ATLAS_INVALID_EXPONENT;
  }
  let stored = i32(lane);
  if (stored == 0 || stored == 1) { return 0; }
  if (stored >= -149 && stored <= -1) { return stored; }
  if (stored >= 2 && stored <= 115) { return stored - 1; }
  return DDGI_ATLAS_INVALID_EXPONENT;
}

fn ddgiAtlasEncodeExponent(exponent: i32, zeroLane: f32) -> f32 {
  if (exponent == 0) { return zeroLane; }
  if (exponent < 0) { return f32(exponent); }
  return f32(exponent + 1);
}

fn ddgiAtlasExponentForMagnitude(magnitude: f32) -> i32 {
  if (
    magnitude >= DDGI_ATLAS_F16_MIN_SUBNORMAL &&
    magnitude <= DDGI_ATLAS_F16_MAX
  ) {
    return 0;
  }
  if (magnitude < DDGI_ATLAS_F16_MIN_SUBNORMAL) {
    return clamp(i32(floor(log2(magnitude))), -149, -1);
  }
  return clamp(
    i32(ceil(log2(magnitude / DDGI_ATLAS_SAFE_MANTISSA))),
    1,
    114,
  );
}

fn ddgiAtlasDecodeScalar(mantissa: f32, exponentLane: f32) -> f32 {
  let exponent = ddgiAtlasDecodeExponent(exponentLane);
  if (
    exponent == DDGI_ATLAS_INVALID_EXPONENT ||
    !ddgiAtlasFiniteScalar(mantissa) ||
    (exponent != 0 && abs(mantissa) > DDGI_ATLAS_SAFE_MANTISSA)
  ) {
    return 0.0;
  }
  let decoded = ldexp(mantissa, exponent);
  if (ddgiAtlasFiniteScalar(decoded)) { return decoded; }
  // Valid producer encodings are rounded inward before f16 publication. An
  // overflowing reconstruction therefore identifies malformed/imported data.
  return 0.0;
}

fn ddgiAtlasScalarEncodingValid(mantissa: f32, exponentLane: f32) -> bool {
  let exponent = ddgiAtlasDecodeExponent(exponentLane);
  if (
    exponent == DDGI_ATLAS_INVALID_EXPONENT ||
    !ddgiAtlasFiniteScalar(mantissa) ||
    (exponent != 0 && abs(mantissa) > DDGI_ATLAS_SAFE_MANTISSA) ||
    (exponent == 114 && abs(mantissa) >= 16384.0)
  ) { return false; }
  return ddgiAtlasFiniteScalar(ldexp(mantissa, exponent));
}

fn ddgiAtlasEncodeScalar(value: f32, zeroLane: f32) -> vec2f {
  if (!ddgiAtlasFiniteScalar(value) || value == 0.0) {
    return vec2f(0.0, zeroLane);
  }
  let exponent = ddgiAtlasExponentForMagnitude(abs(value));
  let mantissa = ldexp(value, -exponent);
  if (
    !ddgiAtlasFiniteScalar(mantissa) ||
    abs(mantissa) > DDGI_ATLAS_F16_MAX
  ) {
    return vec2f(0.0, zeroLane);
  }
  var publishMantissa = mantissa;
  if (exponent == 114 && abs(publishMantissa) > 16376.0) {
    publishMantissa = select(-16376.0, 16376.0, publishMantissa > 0.0);
  }
  return vec2f(publishMantissa, ddgiAtlasEncodeExponent(exponent, zeroLane));
}

fn ddgiAtlasDecodeIrradiance(encoded: vec4f) -> vec3f {
  if (!ddgiAtlasIrradianceEncodingValid(encoded)) { return vec3f(0.0); }
  return vec3f(
    ddgiAtlasDecodeScalar(encoded.r, encoded.a),
    ddgiAtlasDecodeScalar(encoded.g, encoded.a),
    ddgiAtlasDecodeScalar(encoded.b, encoded.a),
  );
}

fn ddgiAtlasIrradianceEncodingValid(encoded: vec4f) -> bool {
  return
    ddgiAtlasScalarEncodingValid(encoded.r, encoded.a) &&
    ddgiAtlasScalarEncodingValid(encoded.g, encoded.a) &&
    ddgiAtlasScalarEncodingValid(encoded.b, encoded.a);
}

fn ddgiAtlasEncodeIrradiance(value: vec3f) -> vec4f {
  if (!ddgiAtlasFiniteVec3(value)) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let magnitude = max(abs(value.x), max(abs(value.y), abs(value.z)));
  if (magnitude == 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let exponent = ddgiAtlasExponentForMagnitude(magnitude);
  var mantissa = vec3f(
    ldexp(value.x, -exponent),
    ldexp(value.y, -exponent),
    ldexp(value.z, -exponent),
  );
  if (
    !ddgiAtlasFiniteVec3(mantissa) ||
    any(abs(mantissa) > vec3f(DDGI_ATLAS_F16_MAX))
  ) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  if (exponent == 114) {
    mantissa = clamp(mantissa, vec3f(-16376.0), vec3f(16376.0));
  }
  return vec4f(mantissa, ddgiAtlasEncodeExponent(exponent, 1.0));
}

fn ddgiAtlasDecodeVisibility(encoded: vec4f) -> vec2f {
  if (!ddgiAtlasVisibilityEncodingValid(encoded)) { return vec2f(0.0); }
  let mean = ddgiAtlasDecodeScalar(encoded.r, encoded.b);
  let secondMoment = ddgiAtlasDecodeScalar(encoded.g, encoded.a);
  return vec2f(
    mean,
    max(secondMoment, ddgiAtlasSaturatingMul(mean, mean)),
  );
}

fn ddgiAtlasVisibilityEncodingValid(encoded: vec4f) -> bool {
  if (
    !ddgiAtlasScalarEncodingValid(encoded.r, encoded.b) ||
    !ddgiAtlasScalarEncodingValid(encoded.g, encoded.a)
  ) { return false; }
  return
    ddgiAtlasDecodeScalar(encoded.r, encoded.b) >= 0.0 &&
    ddgiAtlasDecodeScalar(encoded.g, encoded.a) >= 0.0;
}

fn ddgiAtlasEncodeVisibility(value: vec2f) -> vec4f {
  if (
    !ddgiAtlasFiniteScalar(value.x) ||
    !ddgiAtlasFiniteScalar(value.y) ||
    any(value < vec2f(0.0))
  ) {
    return vec4f(0.0);
  }
  let orderedSecondMoment = max(
    value.y,
    ddgiAtlasSaturatingMul(value.x, value.x),
  );
  let mean = ddgiAtlasEncodeScalar(value.x, 0.0);
  let secondMoment = ddgiAtlasEncodeScalar(orderedSecondMoment, 0.0);
  return vec4f(mean.x, secondMoment.x, mean.y, secondMoment.y);
}

fn ddgiAtlasSaturatingAdd(a: f32, b: f32) -> f32 {
  if (!ddgiAtlasFiniteScalar(a) || !ddgiAtlasFiniteScalar(b)) { return 0.0; }
  if (b > 0.0 && a > DDGI_ATLAS_F32_MAX - b) { return DDGI_ATLAS_F32_MAX; }
  if (b < 0.0 && a < -DDGI_ATLAS_F32_MAX - b) { return -DDGI_ATLAS_F32_MAX; }
  let result = a + b;
  if (ddgiAtlasFiniteScalar(result)) { return result; }
  return select(-DDGI_ATLAS_F32_MAX, DDGI_ATLAS_F32_MAX, result > 0.0);
}

fn ddgiAtlasSaturatingMul(a: f32, b: f32) -> f32 {
  if (!ddgiAtlasFiniteScalar(a) || !ddgiAtlasFiniteScalar(b)) { return 0.0; }
  if (a == 0.0 || b == 0.0) { return 0.0; }
  if (abs(a) > DDGI_ATLAS_F32_MAX / abs(b)) {
    let negative = (a < 0.0) != (b < 0.0);
    return select(DDGI_ATLAS_F32_MAX, -DDGI_ATLAS_F32_MAX, negative);
  }
  let result = a * b;
  if (ddgiAtlasFiniteScalar(result)) { return result; }
  let negative = (a < 0.0) != (b < 0.0);
  return select(DDGI_ATLAS_F32_MAX, -DDGI_ATLAS_F32_MAX, negative);
}

fn ddgiAtlasSaturatingAdd3(a: vec3f, b: vec3f) -> vec3f {
  return vec3f(
    ddgiAtlasSaturatingAdd(a.x, b.x),
    ddgiAtlasSaturatingAdd(a.y, b.y),
    ddgiAtlasSaturatingAdd(a.z, b.z),
  );
}

fn ddgiAtlasSaturatingMul3(value: vec3f, scale: f32) -> vec3f {
  return vec3f(
    ddgiAtlasSaturatingMul(value.x, scale),
    ddgiAtlasSaturatingMul(value.y, scale),
    ddgiAtlasSaturatingMul(value.z, scale),
  );
}

fn ddgiAtlasSaturatingMulComponents(a: vec3f, b: vec3f) -> vec3f {
  return vec3f(
    ddgiAtlasSaturatingMul(a.x, b.x),
    ddgiAtlasSaturatingMul(a.y, b.y),
    ddgiAtlasSaturatingMul(a.z, b.z),
  );
}

fn ddgiAtlasNormalizeOrZero(value: vec3f) -> vec3f {
  if (!ddgiAtlasFiniteVec3(value)) { return vec3f(0.0); }
  let scale = max(abs(value.x), max(abs(value.y), abs(value.z)));
  if (!(scale > 0.0)) { return vec3f(0.0); }
  let scaled = value / scale;
  let lengthSquared = dot(scaled, scaled);
  if (!(lengthSquared > 0.0) || !ddgiAtlasFiniteScalar(lengthSquared)) {
    return vec3f(0.0);
  }
  return scaled * inverseSqrt(lengthSquared);
}

fn ddgiAtlasSafeLength(value: vec3f) -> f32 {
  if (!ddgiAtlasFiniteVec3(value)) { return DDGI_ATLAS_F32_MAX; }
  let scale = max(abs(value.x), max(abs(value.y), abs(value.z)));
  if (!(scale > 0.0)) { return 0.0; }
  let scaled = value / scale;
  return ddgiAtlasSaturatingMul(scale, sqrt(dot(scaled, scaled)));
}
`;
