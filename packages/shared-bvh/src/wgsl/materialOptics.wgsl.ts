import { buildMaterialAtlasOffsetConstsWGSL } from './materialAtlasOffsets.wgsl.js';

const MATERIAL_OPTICAL_OFFSETS_WGSL = buildMaterialAtlasOffsetConstsWGSL({
  prefix: 'VITRUM_OPTICAL_',
  include: [
    'OPTICAL_HEADER_TEXEL_OFFSET',
    'DISPERSION_IOR_RGB_TEXEL_OFFSET',
    'SPECTRAL_SAMPLES_TEXEL_OFFSET',
    'THIN_FILM_FRONT_REFLECTANCE_TEXEL_OFFSET',
    'THIN_FILM_FRONT_TRANSMITTANCE_TEXEL_OFFSET',
    'THIN_FILM_BACK_REFLECTANCE_TEXEL_OFFSET',
    'THIN_FILM_BACK_TRANSMITTANCE_TEXEL_OFFSET',
    'UV_AFFINE_BASE_TEXEL_OFFSET',
  ],
});

/**
 * Binding-independent decoder for the preintegrated realtime optical ABI.
 *
 * The consumer supplies `materialOpticalLoad(triIndex, metaOffset)`. Keeping
 * that one binding adapter local lets the main shade, DDGI, and RC paths share
 * identical spectral/thin-film math despite their intentionally different
 * atlas bindings and coordinate policies.
 */
export const MATERIAL_OPTICS_WGSL = /* wgsl */ `
${MATERIAL_OPTICAL_OFFSETS_WGSL}

const VITRUM_OPTICAL_SPECTRAL_SAMPLE_COUNT: u32 = 32u;
const VITRUM_OPTICAL_ANGLE_SAMPLE_COUNT: u32 = 8u;
const VITRUM_OPTICAL_MAX_FINITE_F32: f32 = 3.402823466e38;

struct MaterialOpticalResponse {
  reflectance: vec3f,
  transmittance: vec3f,
  present: u32,
};

fn materialOpticalHeader(triIndex: u32) -> vec4f {
  return materialOpticalLoad(
    triIndex,
    VITRUM_OPTICAL_MATERIAL_MAP_OPTICAL_HEADER_TEXEL_OFFSET,
  );
}

fn materialOpticalThickness(triIndex: u32) -> f32 {
  return abs(materialOpticalHeader(triIndex).w);
}

// Header.w sign is topology metadata: positive is an authored thickness cap;
// negative is a synthetic reference for a zero-thickness bulk medium whose
// distance comes from the closed boundary. A thickness texture may scale only
// the former.
fn materialOpticalHasAuthoredThickness(triIndex: u32) -> bool {
  return materialOpticalHeader(triIndex).w > 0.0;
}

fn materialOpticalThicknessMapScale(
  triIndex: u32,
  sampledScale: f32,
) -> f32 {
  return select(
    1.0,
    clamp(sampledScale, 0.0, 1.0),
    materialOpticalHasAuthoredThickness(triIndex),
  );
}

fn materialResolveUv(
  triIndex: u32,
  texCoordCode: u32,
  uv0: vec2f,
  uv1: vec2f,
) -> vec2f {
  if (texCoordCode == 0u) { return uv0; }
  if (texCoordCode == 1u) { return uv1; }
  if (texCoordCode > 15u) { return uv0; }
  let lane = texCoordCode - 2u;
  let base = VITRUM_OPTICAL_MATERIAL_MAP_UV_AFFINE_BASE_TEXEL_OFFSET + lane * 2u;
  let row0 = materialOpticalLoad(triIndex, base);
  let row1 = materialOpticalLoad(triIndex, base + 1u);
  let source = select(uv0, uv1, row0.w >= 0.5);
  if (row1.w < 0.5) { return source; }
  let resolved = vec2f(
    row0.x * source.x + row0.y * source.y + row0.z,
    row1.x * source.x + row1.y * source.y + row1.z,
  );
  let finite = all(resolved == resolved) &&
    all(abs(resolved) <= vec2f(VITRUM_OPTICAL_MAX_FINITE_F32));
  return select(source, resolved, finite);
}

fn materialDispersionIorRgb(triIndex: u32, fallbackIor: f32) -> vec3f {
  let value = materialOpticalLoad(
    triIndex,
    VITRUM_OPTICAL_MATERIAL_MAP_DISPERSION_IOR_RGB_TEXEL_OFFSET,
  ).rgb;
  // The core contract admits every finite IOR >= 1 and maps authored IOR 0 to
  // a finite 1e8 transport surrogate. The optical payload is absent when its
  // lanes are zero; NaN and infinity must likewise fall back to the material
  // word instead of poisoning Fresnel/Snell arithmetic.
  if (
    all(value >= vec3f(1.0)) &&
    all(value <= vec3f(VITRUM_OPTICAL_MAX_FINITE_F32))
  ) {
    return value;
  }
  return vec3f(max(fallbackIor, 1.0));
}

// Exact scalar Beer-Lambert transfer with defined extended-real endpoints.
// Keep this in the shared optical module so RGB volume scattering and every
// spectral consumer use the same zero-extinction × infinite-distance rule.
fn materialBeerTransmittanceExact(mu: f32, distance: f32) -> f32 {
  if (
    mu != mu || distance != distance ||
    mu < 0.0 || distance < 0.0
  ) {
    return 0.0;
  }
  // Zero path length is the identity even at +infinite extinction. Check the
  // exact endpoints before any infinity branch so the shader never forms the
  // indeterminate product Inf*0.
  if (distance == 0.0 || mu == 0.0) { return 1.0; }
  if (mu > VITRUM_OPTICAL_MAX_FINITE_F32) { return 0.0; }
  if (distance > VITRUM_OPTICAL_MAX_FINITE_F32) {
    return 0.0;
  }
  return exp(-mu * distance);
}

fn materialSpectralAttenuation(
  triIndex: u32,
  distanceInMaterial: f32,
  fallbackBeer: vec3f,
) -> vec3f {
  if (distanceInMaterial != distanceInMaterial || distanceInMaterial < 0.0) {
    return vec3f(0.0);
  }
  // Thickness-map scale G=0 is exact identity absorption. Do not reconstruct
  // the spectral basis here: finite-precision weights need not sum to exactly
  // one, and that approximation must not perturb a zero-distance endpoint.
  if (distanceInMaterial == 0.0) { return vec3f(1.0); }
  let header = materialOpticalHeader(triIndex);
  if (u32(round(max(header.x, 0.0))) != VITRUM_OPTICAL_SPECTRAL_SAMPLE_COUNT) {
    return clamp(fallbackBeer, vec3f(0.0), vec3f(1.0));
  }
  var attenuation = vec3f(0.0);
  for (var i = 0u; i < VITRUM_OPTICAL_SPECTRAL_SAMPLE_COUNT; i = i + 1u) {
    let sample = materialOpticalLoad(
      triIndex,
      VITRUM_OPTICAL_MATERIAL_MAP_SPECTRAL_SAMPLES_TEXEL_OFFSET + i,
    );
    attenuation = attenuation + sample.yzw *
      materialBeerTransmittanceExact(sample.x, distanceInMaterial);
  }
  // Linear-sRGB reconstruction weights include the expected negative lobes;
  // clamp only after the complete spectral integral has been accumulated.
  return clamp(attenuation, vec3f(0.0), vec3f(1.0));
}

fn materialThinFilmResponse(
  triIndex: u32,
  frontFacing: bool,
  cosTheta: f32,
) -> MaterialOpticalResponse {
  var out: MaterialOpticalResponse;
  out.reflectance = vec3f(0.0);
  out.transmittance = vec3f(1.0);
  out.present = 0u;
  let header = materialOpticalHeader(triIndex);
  if (header.y < 0.5) {
    return out;
  }

  let scaled = clamp(abs(cosTheta), 0.0, 1.0) * f32(VITRUM_OPTICAL_ANGLE_SAMPLE_COUNT - 1u);
  let lo = min(u32(floor(scaled)), VITRUM_OPTICAL_ANGLE_SAMPLE_COUNT - 1u);
  let hi = min(lo + 1u, VITRUM_OPTICAL_ANGLE_SAMPLE_COUNT - 1u);
  let t = scaled - f32(lo);
  let reflectanceBase = select(
    VITRUM_OPTICAL_MATERIAL_MAP_THIN_FILM_BACK_REFLECTANCE_TEXEL_OFFSET,
    VITRUM_OPTICAL_MATERIAL_MAP_THIN_FILM_FRONT_REFLECTANCE_TEXEL_OFFSET,
    frontFacing,
  );
  let transmittanceBase = select(
    VITRUM_OPTICAL_MATERIAL_MAP_THIN_FILM_BACK_TRANSMITTANCE_TEXEL_OFFSET,
    VITRUM_OPTICAL_MATERIAL_MAP_THIN_FILM_FRONT_TRANSMITTANCE_TEXEL_OFFSET,
    frontFacing,
  );
  out.reflectance = clamp(mix(
    materialOpticalLoad(triIndex, reflectanceBase + lo).rgb,
    materialOpticalLoad(triIndex, reflectanceBase + hi).rgb,
    t,
  ), vec3f(0.0), vec3f(1.0));
  out.transmittance = clamp(mix(
    materialOpticalLoad(triIndex, transmittanceBase + lo).rgb,
    materialOpticalLoad(triIndex, transmittanceBase + hi).rgb,
    t,
  ), vec3f(0.0), vec3f(1.0));
  out.present = 1u;
  return out;
}
`;
