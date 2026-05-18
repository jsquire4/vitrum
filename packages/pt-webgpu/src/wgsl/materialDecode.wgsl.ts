/**
 * Material decode + thin-film TMM helpers.
 *
 * Encapsulates everything that reads the packed `materials: array<vec4f>` UBO
 * slice: scalar accessor, spectral table lookup, complex-number primitives for
 * the TMM solver, the TMM solver itself (Belcour & Barla 2017), and the
 * DecodedMaterial struct that downstream WGSL consumes.
 *
 * Refs:
 *  - Belcour & Barla. "A Practical Extension to Microfacet Theory for the
 *    Modeling of Varying Iridescence." ACM TOG 36(4) (SIGGRAPH 2017).
 *    https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
 */
export const PT_WEBGPU_MATERIAL_DECODE_WGSL = /* wgsl */ `
fn materialScalar(matId: u32, scalarOffset: u32) -> f32 {
  let scalarIndex = matId * MATERIAL_SCALAR_STRIDE + scalarOffset;
  let vecIndex = scalarIndex / 4u;
  if (vecIndex >= arrayLength(&materials)) { return 0.0; }
  let c = scalarIndex % 4u;
  let v = materials[vecIndex];
  if (c == 0u) { return v.x; }
  if (c == 1u) { return v.y; }
  if (c == 2u) { return v.z; }
  return v.w;
}

fn sampleMaterialSpectralMu(matId: u32, wavelength01: f32) -> f32 {
  let clamped = clamp(wavelength01, 0.0, 1.0);
  let f = clamped * f32(SPECTRAL_SAMPLE_COUNT - 1u);
  let i0 = u32(floor(f));
  let i1 = min(i0 + 1u, SPECTRAL_SAMPLE_COUNT - 1u);
  let a = materialScalar(matId, SPECTRAL_SCALAR_BASE + i0);
  let b = materialScalar(matId, SPECTRAL_SCALAR_BASE + i1);
  let t = f - f32(i0);
  return mix(a, b, t);
}

fn cMul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cDiv(a: vec2f, b: vec2f) -> vec2f {
  let d = max(dot(b, b), 1e-8);
  return vec2f(
    (a.x * b.x + a.y * b.y) / d,
    (a.y * b.x - a.x * b.y) / d,
  );
}

fn thinFilmTmmRt(
  matId: u32,
  layerCount: u32,
  wavelengthNm: f32,
  substrateIor: f32,
  incidentIor: f32,
  angleDependent: bool,
  viewCos: f32,
) -> vec2f {
  if (layerCount == 0u) {
    return vec2f(0.0, 1.0);
  }
  let lambdaUm = max(wavelengthNm * 0.001, 1e-5);
  let eta0 = max(incidentIor, 1.0);
  let etaS = max(substrateIor, 1.0);
  let angleScale = select(1.0, clamp(viewCos, 0.05, 1.0), angleDependent);
  var absorbAccum = 1.0;
  var m11 = vec2f(1.0, 0.0);
  var m12 = vec2f(0.0, 0.0);
  var m21 = vec2f(0.0, 0.0);
  var m22 = vec2f(1.0, 0.0);
  for (var i = 0u; i < THIN_FILM_LAYER_LIMIT; i = i + 1u) {
    if (i >= layerCount) {
      break;
    }
    let layerBase = THIN_FILM_SCALAR_BASE + i * 3u;
    let layerIor = max(materialScalar(matId, layerBase), 1.0);
    let layerThicknessUm = max(materialScalar(matId, layerBase + 1u) * 0.001, 0.0);
    let layerK = max(materialScalar(matId, layerBase + 2u), 0.0);
    absorbAccum = absorbAccum * exp(-4.0 * PI * layerK * layerThicknessUm * angleScale / lambdaUm);
    let delta = 2.0 * PI * layerIor * layerThicknessUm * angleScale / lambdaUm;
    let c = cos(delta);
    let s = sin(delta);
    let a11 = vec2f(c, 0.0);
    let a12 = vec2f(0.0, -s / layerIor);
    let a21 = vec2f(0.0, -layerIor * s);
    let a22 = vec2f(c, 0.0);
    let nm11 = cMul(m11, a11) + cMul(m12, a21);
    let nm12 = cMul(m11, a12) + cMul(m12, a22);
    let nm21 = cMul(m21, a11) + cMul(m22, a21);
    let nm22 = cMul(m21, a12) + cMul(m22, a22);
    m11 = nm11;
    m12 = nm12;
    m21 = nm21;
    m22 = nm22;
  }
  let eta0m11 = m11 * eta0;
  let eta0etaSm12 = m12 * (eta0 * etaS);
  let etaSm22 = m22 * etaS;
  let den = eta0m11 + eta0etaSm12 + m21 + etaSm22;
  let numR = eta0m11 + eta0etaSm12 - m21 - etaSm22;
  let r = cDiv(numR, den);
  let t = cDiv(vec2f(2.0 * eta0, 0.0), den);
  let R = clamp(dot(r, r), 0.0, 1.0);
  let T = clamp((etaS / eta0) * dot(t, t), 0.0, 1.0);
  return vec2f(R, T * absorbAccum);
}

struct DecodedMaterial {
  baseColor: vec3f,
  roughness: f32,
  emissive: vec3f,
  metallic: f32,
  transmission: f32,
  ior: f32,
  scatteringCoeff: f32,
  scatteringAnisotropy: f32,
  scatteringRgb: vec3f,
  hasSpectralAttenuation: bool,
  frontLayerTx: vec3f,
  frontLayerRoughness: f32,
  backLayerTx: vec3f,
  backLayerRoughness: f32,
  thinFilmEnabled: bool,
  thinFilmLayerCountU: u32,
  thinFilmIncidentIor: f32,
  thinFilmAngleDependent: bool,
  spectralAvgMu: f32,
  spectralSampleCount: u32,
}

fn decodeMaterial(matId: u32) -> DecodedMaterial {
  let m0Index = matId * MATERIAL_VEC4_STRIDE;
  let m1Index = m0Index + 1u;
  let m2Index = m0Index + 2u;
  let m3Index = m0Index + 3u;
  let m4Index = m0Index + 4u;
  let m5Index = m0Index + 5u;
  let m6Index = m0Index + 6u;
  let m19Index = m0Index + 21u;
  let m0 = select(vec4f(0.8, 0.8, 0.8, 0.6), materials[m0Index], m0Index < arrayLength(&materials));
  let m1 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m1Index], m1Index < arrayLength(&materials));
  let m2 = select(vec4f(0.0, 1.5, 0.0, 0.0), materials[m2Index], m2Index < arrayLength(&materials));
  let m3 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m3Index], m3Index < arrayLength(&materials));
  let m4 = select(vec4f(1.0, 1.0, 1.0, -1.0), materials[m4Index], m4Index < arrayLength(&materials));
  let m5 = select(vec4f(1.0, 1.0, 1.0, -1.0), materials[m5Index], m5Index < arrayLength(&materials));
  let m6 = select(vec4f(0.0, 0.0, 1.0, 0.0), materials[m6Index], m6Index < arrayLength(&materials));
  let m19 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m19Index], m19Index < arrayLength(&materials));
  var mat: DecodedMaterial;
  mat.baseColor = m0.rgb;
  mat.roughness = clamp(m0.w, 0.02, 1.0);
  mat.emissive = m1.rgb;
  mat.metallic = clamp(m1.w, 0.0, 1.0);
  mat.transmission = clamp(m2.x, 0.0, 1.0);
  mat.ior = clamp(m2.y, 1.0, 2.5);
  mat.scatteringCoeff = max(m2.z, 0.0);
  mat.scatteringAnisotropy = clamp(m2.w, -0.95, 0.95);
  mat.scatteringRgb = vec3f(max(m3.x, 0.0), max(m3.y, 0.0), max(m3.z, 0.0));
  mat.hasSpectralAttenuation = m3.w > 0.5;
  mat.frontLayerTx = m4.rgb;
  mat.frontLayerRoughness = m4.w;
  mat.backLayerTx = m5.rgb;
  mat.backLayerRoughness = m5.w;
  mat.thinFilmEnabled = m6.x > 0.5;
  mat.thinFilmLayerCountU = u32(max(m6.y, 0.0));
  mat.thinFilmIncidentIor = max(m6.z, 1.0);
  mat.thinFilmAngleDependent = m6.w > 0.5;
  mat.spectralAvgMu = max(m19.x, 0.0);
  mat.spectralSampleCount = u32(max(m19.w, 0.0));
  return mat;
}
`;
