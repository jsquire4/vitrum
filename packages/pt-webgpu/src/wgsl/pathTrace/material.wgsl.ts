/**
 * Material module — `FrameParams` UBO + group(0) bindings + material payload
 * accessors, Fresnel / microfacet / MIS primitives, thin-film TMM solver,
 * and the `decodeMaterial` packed-buffer reader.
 *
 * This module is the first concatenated chunk in `pathTraceBruteforce.wgsl.ts`
 * because every later module references the bindings (materials, lights,
 * BVH) and material constants (MATERIAL_VEC4_STRIDE, etc.).
 *
 * Bundled here:
 *  - `FrameParams` struct + 24 `@group(0)` bindings
 *  - Material constants (MATERIAL_VEC4_STRIDE, THIN_FILM_*, SPECTRAL_*)
 *  - `BsdfSample` triple — shared sampler return type
 *  - `materialScalar`, `sampleMaterialSpectralMu` — packed-buffer accessors
 *  - `cMul` / `cDiv` complex-number helpers (used by TMM)
 *  - `thinFilmTmmRt` — Belcour & Barla 2017 transfer-matrix solver
 *  - `luminance`, `fresnelSchlick`, `frDielectric` — Fresnel primitives
 *  - `ggxD`, `smithG1`, `powerHeuristic` — microfacet + MIS helpers
 *  - `DecodedMaterial` struct + `decodeMaterial` reader
 */
/** Bindings 0–11: core mesh path trace + G-buffer aux (≤8 storage buffers, ≤4 storage textures). */
export const PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_BINDINGS_WGSL = /* wgsl */ `
struct FrameParams {
  width: u32,
  height: u32,
  frameIndex: u32,
  frameSeed: u32,
  triangleCount: u32,
  maxBounces: u32,
  bvhNodeCount: u32,
  analyticCount: u32,
  pointLightCount: u32,
  spotLightCount: u32,
  rectAreaLightCount: u32,
  meshAreaLightCount: u32,
  mneeMaxIterations: u32,
  mneeMaxChainLength: u32,
  hasEnvironmentMap: u32,
  causticStrategy: u32,
  environmentMapWidth: u32,
  environmentMapHeight: u32,
  triIntersectEpsilon: f32,
  tlasNodeCount: u32,
  spectralEnabled: u32,
  heroStrategy: u32,
  heroLambdaNm: f32,
  heroPdf: f32,
  cmfIntegralX: f32,
  cmfIntegralY: f32,
  cmfIntegralZ: f32,
  _padBeforeCamera: u32,
  cameraPos: vec4f,
  lightDir: vec4f,
  environmentTint: vec4f,
  environmentSun: vec4f,
  invViewProj: mat4x4f,
  viewProj: mat4x4f,
  prevViewProj: mat4x4f,
};

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: FrameParams;
@group(0) @binding(2) var<storage, read_write> accumBuffer: array<vec4f>;
@group(0) @binding(3) var<storage, read> positions: array<vec4f>;
@group(0) @binding(4) var<storage, read> indices: array<vec4u>;
@group(0) @binding(5) var<storage, read> triMaterialIds: array<u32>;
@group(0) @binding(6) var<storage, read> materials: array<vec4f>;
@group(0) @binding(7) var<storage, read> bvhNodes: array<BVHNode>;
@group(0) @binding(8) var<storage, read> normals: array<vec4f>;
@group(0) @binding(9) var normalDepthTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var albedoTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var varianceTexture: texture_storage_2d<rgba16float, write>;
`;

export const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP0_WGSL = /* wgsl */ `
struct FrameParams {
  width: u32,
  height: u32,
  frameIndex: u32,
  frameSeed: u32,
  triangleCount: u32,
  maxBounces: u32,
  bvhNodeCount: u32,
  analyticCount: u32,
  pointLightCount: u32,
  spotLightCount: u32,
  rectAreaLightCount: u32,
  meshAreaLightCount: u32,
  mneeMaxIterations: u32,
  mneeMaxChainLength: u32,
  hasEnvironmentMap: u32,
  causticStrategy: u32,
  environmentMapWidth: u32,
  environmentMapHeight: u32,
  triIntersectEpsilon: f32, // UBO-plumbed (D12); default metre-scale
  tlasNodeCount: u32,
  spectralEnabled: u32,
  heroStrategy: u32,
  heroLambdaNm: f32,
  heroPdf: f32,
  cmfIntegralX: f32,
  cmfIntegralY: f32,
  cmfIntegralZ: f32,
  _padBeforeCamera: u32,
  cameraPos: vec4f,
  lightDir: vec4f,
  environmentTint: vec4f,
  environmentSun: vec4f,
  invViewProj: mat4x4f,
  viewProj: mat4x4f,
  prevViewProj: mat4x4f,
};

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: FrameParams;
@group(0) @binding(2) var<storage, read_write> accumBuffer: array<vec4f>;
@group(0) @binding(3) var<storage, read> positions: array<vec4f>;
@group(0) @binding(4) var<storage, read> indices: array<vec4u>;
@group(0) @binding(5) var<storage, read> triMaterialIds: array<u32>;
@group(0) @binding(6) var<storage, read> materials: array<vec4f>;
@group(0) @binding(7) var<storage, read> bvhNodes: array<BVHNode>;
@group(0) @binding(8) var<storage, read> normals: array<vec4f>;
@group(0) @binding(9) var normalDepthTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var albedoTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var varianceTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var motionVectorsTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var<storage, read_write> varianceMomentsBuffer: array<vec4f>;
`;

/** Group 1 — analytics + env + area lights (10 storage buffers; adapters ≥10/stage). */
export const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP1_WGSL = /* wgsl */ `
@group(1) @binding(0) var<storage, read> analyticHeaders: array<vec4f>;
@group(1) @binding(1) var<storage, read> analyticParams: array<vec4f>;
@group(1) @binding(2) var<storage, read> analyticLocalToWorld: array<vec4f>;
@group(1) @binding(3) var<storage, read> analyticWorldToLocal: array<vec4f>;
@group(1) @binding(4) var<storage, read> environmentMapTexels: array<vec4f>;
@group(1) @binding(5) var<storage, read> environmentMapCdf: array<f32>;
@group(1) @binding(6) var<storage, read> pointLights: array<vec4f>;
@group(1) @binding(7) var<storage, read> spotLights: array<vec4f>;
@group(1) @binding(8) var<storage, read> rectAreaLights: array<vec4f>;
@group(1) @binding(9) var<storage, read> meshAreaLights: array<vec4f>;
`;

/** Group 2 — TLAS instance table (5 storage buffers). */
export const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP2_WGSL = /* wgsl */ `
@group(2) @binding(0) var<storage, read> tlasNodes: array<BVHNode>;
@group(2) @binding(1) var<storage, read> tlasInstanceIndices: array<u32>;
@group(2) @binding(2) var<storage, read> tlasBlasRoots: array<u32>;
@group(2) @binding(3) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(2) @binding(4) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
`;

export const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_WGSL =
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP0_WGSL +
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP1_WGSL +
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP2_WGSL;

export const PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL = /* wgsl */ `
const LEAFNODE_FLAG = 0xffff0000u;
const MATERIAL_VEC4_STRIDE = 22u;
const MATERIAL_SCALAR_STRIDE = MATERIAL_VEC4_STRIDE * 4u;
const THIN_FILM_LAYER_LIMIT = 8u;
const THIN_FILM_SCALAR_BASE = 28u;
const SPECTRAL_SCALAR_BASE = 52u;
const SPECTRAL_SAMPLE_COUNT = 32u;

// Shared BSDF / light-sample triple. Bundles the {direction, pdf, value}
// outputs every sampler in this kernel produces, so callers can hand a single
// struct between the sample / pdf / eval functions and future MIS code paths.
//
// Semantics:
//   wi     — sampled scattered (or environment) direction in world space.
//   pdf    — probability density at wi. A value <= 0 signals failure for
//            samplers that can fail (currently only sampleEnvironmentImportance).
//   value  — for BSDF samplers, the unitless BRDF "kernel" at wi (Fresnel and
//            albedo are integrated by callers at the throughput level, matching
//            the existing sampleNextBounceDirection pattern). For the
//            environment-importance sampler, the emitted radiance along wi.
struct BsdfSample {
  wi: vec3f,
  pdf: f32,
  value: vec3f,
}

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

// fn luminance(c: vec3f) — canonical from LUMINANCE_WGSL in the orchestrator
// (pathTraceBruteforce.wgsl.ts:50; @vitrum/shared-samplers).

fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  let m5 = m2 * m2 * m;
  return f0 + (vec3f(1.0) - f0) * m5;
}

/**
 * Unpolarised Fresnel reflectance for a smooth dielectric interface.
 * Handles TIR (returns 1.0) and entering-from-inside (cosTheta_i < 0).
 * Ref: Pharr, Jakob, Humphreys. Physically Based Rendering 4th ed. §9.3
 *      "Specular Reflection and Transmission" — FrDielectric().
 *      https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
 */
fn frDielectric(cosTheta_i_in: f32, eta_in: f32) -> f32 {
  var cosTheta_i = clamp(cosTheta_i_in, -1.0, 1.0);
  var eta = eta_in;
  // Entering from the inside — flip so cosTheta_i is positive and invert eta.
  if (cosTheta_i < 0.0) {
    eta = 1.0 / eta;
    cosTheta_i = -cosTheta_i;
  }
  let sin2Theta_i = max(0.0, 1.0 - cosTheta_i * cosTheta_i);
  let sin2Theta_t = sin2Theta_i / (eta * eta);
  if (sin2Theta_t >= 1.0) { return 1.0; } // Total Internal Reflection.
  let cosTheta_t = sqrt(max(0.0, 1.0 - sin2Theta_t));
  let r_par  = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
  let r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
  return 0.5 * (r_par * r_par + r_perp * r_perp);
}

fn ggxD(nDotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-6);
}

fn smithG1(nDotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) * 0.125;
  return nDotV / max(nDotV * (1.0 - k) + k, 1e-6);
}

fn powerHeuristic(pdfA: f32, pdfB: f32) -> f32 {
  let a2 = pdfA * pdfA;
  let b2 = pdfB * pdfB;
  return a2 / max(a2 + b2, 1e-6);
}

// Cauchy dispersion (mirrors @vitrum/shared-samplers/cauchyIor.ts).
fn cauchyIorAtLambda(lambdaNm: f32, baseIor: f32, abbeV: f32) -> f32 {
  if (abbeV < 1.0) {
    return baseIor;
  }
  let lambdaUm = lambdaNm * 0.001;
  let lam2 = lambdaUm * lambdaUm;
  let lamF = 0.4861;
  let lamC = 0.6563;
  let denom = 1.0 / (lamF * lamF) - 1.0 / (lamC * lamC);
  let B = (baseIor - 1.0) / max(abbeV, 1.0) / max(denom, 1e-6);
  return baseIor + B / lam2;
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
  dispersionAbbe: f32,
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
  mat.dispersionAbbe = max(m19.y, 0.0);
  return mat;
}
`;

/** Full trace pass — 3 bind groups (≤10 storage buffers per group). */
export const PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL =
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_WGSL + PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL;

/** Compatibility tier for adapters capped at 10 storage buffers / 4 storage textures. */
export const PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_WGSL =
  PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_BINDINGS_WGSL + PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL;
