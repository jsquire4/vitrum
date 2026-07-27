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
  return max(materialOpticalHeader(triIndex).w, 0.0);
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
  if (row1.w < 0.5) { return uv0; }
  let source = select(uv0, uv1, row0.w >= 0.5);
  return vec2f(
    row0.x * source.x + row0.y * source.y + row0.z,
    row1.x * source.x + row1.y * source.y + row1.z,
  );
}

fn materialDispersionIorRgb(triIndex: u32, fallbackIor: f32) -> vec3f {
  let value = materialOpticalLoad(
    triIndex,
    VITRUM_OPTICAL_MATERIAL_MAP_DISPERSION_IOR_RGB_TEXEL_OFFSET,
  ).rgb;
  if (all(value >= vec3f(1.0)) && all(value < vec3f(8.0))) {
    return value;
  }
  return vec3f(max(fallbackIor, 1.0));
}

fn materialSpectralAttenuation(
  triIndex: u32,
  distanceInMaterial: f32,
  fallbackBeer: vec3f,
) -> vec3f {
  let header = materialOpticalHeader(triIndex);
  if (u32(round(max(header.x, 0.0))) != VITRUM_OPTICAL_SPECTRAL_SAMPLE_COUNT) {
    return clamp(fallbackBeer, vec3f(0.0), vec3f(1.0));
  }
  var attenuation = vec3f(0.0);
  let distance = max(distanceInMaterial, 0.0);
  for (var i = 0u; i < VITRUM_OPTICAL_SPECTRAL_SAMPLE_COUNT; i = i + 1u) {
    let sample = materialOpticalLoad(
      triIndex,
      VITRUM_OPTICAL_MATERIAL_MAP_SPECTRAL_SAMPLES_TEXEL_OFFSET + i,
    );
    attenuation = attenuation + sample.yzw * exp(-max(sample.x, 0.0) * distance);
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
