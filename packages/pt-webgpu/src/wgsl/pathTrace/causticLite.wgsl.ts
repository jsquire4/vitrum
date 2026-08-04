/**
 * Lite-tier caustic compatibility functions. MNEE and photon mapping are
 * capability-gated off before this shader composition is selected.
 */
export const PT_WEBGPU_PATH_TRACE_CAUSTIC_LITE_WGSL = /* wgsl */ `
fn manifoldNeeContribution(
  rng: ptr<function, PtRngState>,
  hitPos: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  thinFilm: ThinFilmInterface,
  heroLambda: f32,
  throughput: vec3f,
) -> vec3f {
  _ = rng;
  _ = hitPos;
  _ = normal;
  _ = clearcoatNormal;
  _ = wo;
  _ = baseColor;
  _ = roughness;
  _ = metallic;
  _ = transmission;
  _ = etaTOverI;
  _ = clearcoat;
  _ = clearcoatRoughness;
  _ = sheen;
  _ = sheenRoughness;
  _ = sheenColor;
  _ = iridescence;
  _ = iridescenceIor;
  _ = iridescenceThicknessMin;
  _ = iridescenceThicknessMax;
  _ = specularColor;
  _ = specularIntensity;
  _ = anisotropy;
  _ = anisotropyRotation;
  _ = thinFilm;
  _ = heroLambda;
  _ = throughput;
  return vec3f(0.0);
}

fn photonMapUpdateProgressive(
  pixelIndex: u32,
  hitPos: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  thinFilm: ThinFilmInterface,
  throughput: vec3f,
  heroLambda: f32,
  heroPdf: f32,
  absorbedFluxInvPdf: f32,
) {
  _ = pixelIndex;
  _ = hitPos;
  _ = normal;
  _ = clearcoatNormal;
  _ = wo;
  _ = baseColor;
  _ = roughness;
  _ = metallic;
  _ = transmission;
  _ = etaTOverI;
  _ = clearcoat;
  _ = clearcoatRoughness;
  _ = sheen;
  _ = sheenRoughness;
  _ = sheenColor;
  _ = iridescence;
  _ = iridescenceIor;
  _ = iridescenceThicknessMin;
  _ = iridescenceThicknessMax;
  _ = specularColor;
  _ = specularIntensity;
  _ = anisotropy;
  _ = anisotropyRotation;
  _ = thinFilm;
  _ = throughput;
  _ = heroLambda;
  _ = heroPdf;
  _ = absorbedFluxInvPdf;
}
`;
