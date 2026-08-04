import { lightTreeWgsl } from '@vitrum/shared-samplers';
import {
  MATERIAL_TEX_CLEARCOAT_NORMAL_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET,
  MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP_VEC4_OFFSET,
  MATERIAL_TEX_EXTENSION_INDEX_VEC4_OFFSET,
  MATERIAL_TEX_EXTENSION_UV_FIT_VEC4_OFFSET,
  MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_EXTENSION_WRAP_VEC4_OFFSET,
  MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET,
  MATERIAL_TEX_LAYER_NORMAL_UV_FIT_VEC4_OFFSET,
  MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_LAYER_NORMAL_VEC4_OFFSET,
  MATERIAL_TEX_LAYER_NORMAL_WRAP_VEC4_OFFSET,
  MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET,
  MATERIAL_TEX_THICKNESS_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_THICKNESS_VEC4_OFFSET,
  MATERIAL_TEX_THICKNESS_WRAP_VEC4_OFFSET,
  MATERIAL_TEX_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_UV_META_VEC4S_PER_MAP,
  MATERIAL_TEX_VEC4_STRIDE,
} from '../../scene/materialTextures.js';
import { roughDielectricSmithG1Wgsl } from '../../math/roughDielectric.js';
import { KHR_MATERIALS_IOR_INFINITY_APPROX } from '../../scene/materialPacking.js';

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
/**
 * Shared `FrameParams` UBO struct + `@group(0)` bindings 0–11 — written ONCE
 * and reused by both the lite-tier bindings and the full-tier group-0 bindings.
 *
 * The two tiers differ only in (a) a trailing comment on `triIntersectEpsilon`
 * (full-tier annotates the UBO-plumbing) and (b) the full tier's extra
 * bindings 12–13 (motion vectors + variance-moments aux). Both differences are
 * supplied as parameters so the composed strings stay byte-identical to the
 * pre-dedup monolithic consts.
 *
 * @param epsilonSuffix  appended after `triIntersectEpsilon: f32,` (empty for
 *   lite; the UBO-plumbing comment for full).
 * @param extraBindings  appended after binding 11 (empty for lite; bindings
 *   12–13 for full). Includes its own leading newline when non-empty.
 */
function frameParamsGroup0Bindings(epsilonSuffix: string, extraBindings: string): string {
  return /* wgsl */ `
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
  triIntersectEpsilon: f32,${epsilonSuffix}
  tlasNodeCount: u32,
  spectralEnabled: u32,
  heroLambdaNm: f32,
  heroPdf: f32,
  bdptEnabled: u32,
  bdptMaxLightBounces: u32,
  bdptMaxEyeDepth: u32,
  lightTreeEnabled: u32,
  lightTreeNodeCount: u32,
  // Only the camera position's xyz components are part of the contract.
  cameraPos: vec3f,
  // H14-E: map-backed environment-radiance intensity. This scalar occupies
  // cameraPos's aligned fourth lane, keeping its established slot 31.
  environmentHdriIntensity: f32,
  environmentTint: vec4f,
  invViewProj: mat4x4f,
  viewProj: mat4x4f,
  prevViewProj: mat4x4f,
  // N-directional expansion: total packed directional count read from the
  // directionalLights storage buffer (group 1 binding 10). The kernel loops
  // over params.directionalLightCount records. A single-directional scene keeps
  // directionalLightCount=1 and directionalLights[0] preserves the established
  // single-directional path.
  directionalLightCount: u32,
  // BDPT pseudo-distant emitters use current scene bounds instead of a fixed
  // Cornell-scale 50-unit offset.
  sceneCenterX: f32,
  sceneCenterY: f32,
  sceneCenterZ: f32,
  sceneRadius: f32,
  // 0 = sampled one-of-N direct-light selection; 1 = sum direct candidates.
  // Inverse path replay uses mode 1 so the forward baseline matches the adjoint
  // pass's deterministic direct-light domain.
  directLightingMode: u32,
  // Scene-relative secondary-ray origin offset. Uses the struct's existing tail
  // padding, so adding this field does not enlarge FrameParams.
  rayOriginBias: f32,
  // Exact 24-bit represented PMF of the optional environment candidate in the
  // distant-direct categorical proposal.
  environmentDistantProposalPmf: f32,
};

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: FrameParams;
@group(0) @binding(2) var<storage, read_write> accumBuffer: array<vec4f>;
@group(0) @binding(3) var<storage, read> positions: array<vec4f>;
@group(0) @binding(4) var<storage, read> indices: array<vec4u>;
// Per-triangle vec2u payload: .x material id, .y stable represented primitive-
// instance ordinal+1 used by exact post-transmission source-feature replay.
@group(0) @binding(5) var<storage, read> triMaterialIds: array<vec2u>;
@group(0) @binding(6) var<storage, read> materials: array<vec4f>;
@group(0) @binding(7) var<storage, read> bvhNodes: array<BVHNode>;
@group(0) @binding(8) var<storage, read> normals: array<vec4f>;
@group(0) @binding(9) var normalDepthTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var albedoTexture: texture_storage_2d<rgba16float, write>;
// Scalar luminance variance. r32float is a baseline storage format and avoids
// allocating/writing three semantically dead half-float lanes per pixel.
@group(0) @binding(11) var varianceTexture: texture_storage_2d<r32float, write>;${extraBindings}

const INVALID_TLAS_INSTANCE_INDEX = 0xffffffffu;

fn ptRayOriginBias() -> f32 {
  return max(params.rayOriginBias, 1.175494351e-38);
}

fn ptRayTMin() -> f32 {
  return max(ptRayOriginBias() * 0.1, 1.175494351e-38);
}

fn ptRayEndMargin() -> f32 {
  return ptRayOriginBias() * 2.0;
}

fn ptFiniteSegmentTMax(distance: f32) -> f32 {
  return max(distance - ptRayEndMargin(), ptRayTMin());
}

const PT_F32_MAX: f32 = 3.402823466e38;

fn ptFiniteNonNegativeRadianceLaneProduct(left: f32, right: f32) -> f32 {
  if (
    !(left >= 0.0 && left <= PT_F32_MAX) ||
    !(right >= 0.0 && right <= PT_F32_MAX) ||
    left == 0.0 ||
    right == 0.0
  ) {
    return 0.0;
  }
  // Cap the second operand before multiplication. This is deterministic for
  // opaque host-owned textures whose exact texels cannot be inspected during
  // setScene, and prevents NaN/+Inf radiance from reaching accumulation.
  return left * min(right, PT_F32_MAX / left);
}

fn ptFiniteNonNegativeRadianceProduct(left: vec3f, right: vec3f) -> vec3f {
  return vec3f(
    ptFiniteNonNegativeRadianceLaneProduct(left.r, right.r),
    ptFiniteNonNegativeRadianceLaneProduct(left.g, right.g),
    ptFiniteNonNegativeRadianceLaneProduct(left.b, right.b),
  );
}

fn ptFiniteNonNegativeRadianceLaneSum(left: f32, right: f32) -> f32 {
  if (
    !(left >= 0.0 && left <= PT_F32_MAX) ||
    !(right >= 0.0 && right <= PT_F32_MAX)
  ) {
    return 0.0;
  }
  return left + min(right, PT_F32_MAX - left);
}

fn ptFiniteNonNegativeRadianceSum(left: vec3f, right: vec3f) -> vec3f {
  return vec3f(
    ptFiniteNonNegativeRadianceLaneSum(left.r, right.r),
    ptFiniteNonNegativeRadianceLaneSum(left.g, right.g),
    ptFiniteNonNegativeRadianceLaneSum(left.b, right.b),
  );
}

// A cone narrower than this has a uniform solid-angle density beyond the
// useful f32 range and its stable 1-cos half-angle falls below the normal f32
// domain. Treat that limiting case as the corresponding delta directional.
const PT_DIRECTIONAL_CONE_MIN_SIN_QUARTER = 7.666467e-20;

fn ptDirectionalConeIsDelta(angularDiameter: f32) -> bool {
  if (!(angularDiameter > 0.0)) {
    return true;
  }
  return
    sin(angularDiameter * 0.25) <
    PT_DIRECTIONAL_CONE_MIN_SIN_QUARTER;
}

fn ptDirectionalConePdf(angularDiameter: f32) -> f32 {
  if (ptDirectionalConeIsDelta(angularDiameter)) {
    return 1.0;
  }
  let sinQuarter = sin(angularDiameter * 0.25);
  // log(1 / (4π sin²(d/4))) avoids both the sin² underflow and reciprocal
  // overflow paths of the direct expression.
  return exp(-log(4.0 * PI) - 2.0 * log(sinQuarter));
}

// Returns (sin(theta), cos(theta)) for a uniform-solid-angle cone draw. Forming
// sin(theta) from q=1-cos(theta), rather than subtracting two near-one cosines,
// preserves authored narrow cones that are still representable as non-delta.
fn ptDirectionalConeSinCos(
  angularDiameter: f32,
  xiCenter: f32,
) -> vec2f {
  let sinQuarter = sin(angularDiameter * 0.25);
  let oneMinusCosHalf = (2.0 * sinQuarter) * sinQuarter;
  let q = (1.0 - clamp(xiCenter, 0.0, 1.0)) * oneMinusCosHalf;
  let sinTheta = sqrt(max(q * (2.0 - q), 0.0));
  let cosTheta = sqrt(max(1.0 - sinTheta * sinTheta, 0.0));
  return vec2f(sinTheta, cosTheta);
}

fn ptAreaToSolidAnglePdf(
  distance: f32,
  cosine: f32,
  areaMeasure: AreaVectorMeasure,
) -> f32 {
  if (
    !(distance > 0.0) || !(cosine > 0.0) ||
    areaMeasure.valid == 0u || !(areaMeasure.edgeScale > 0.0)
  ) {
    return 0.0;
  }
  let normalizedArea =
    (areaMeasure.area / areaMeasure.edgeScale) /
    areaMeasure.edgeScale;
  let distanceOverEdge = distance / areaMeasure.edgeScale;
  if (
    !(normalizedArea > 0.0) ||
    normalizedArea > 3.402823466e38 ||
    !(distanceOverEdge > 0.0) ||
    distanceOverEdge > 3.402823466e38
  ) {
    return 0.0;
  }
  let pdf =
    (distanceOverEdge * distanceOverEdge) /
    (cosine * normalizedArea);
  return select(
    0.0,
    pdf,
    pdf > 0.0 && pdf <= 3.402823466e38,
  );
}
`;
}

/** Bindings 0–11: core mesh path trace + G-buffer aux (≤8 storage buffers, ≤4 storage textures). */
const PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_BINDINGS_BASE_WGSL = frameParamsGroup0Bindings('', '');

/**
 * B12 — lite-tier texture bindings (12–14): sampled texture_2d<f32> slots for
 * packed HDRI env radiance+CDF and analytic light data.  These use
 * `maxSampledTexturesPerShaderStage` (≥ 16, WebGPU baseline) — a SEPARATE budget
 * from `maxStorageBuffersPerShaderStage` (= 8 on capped adapters).  No sampler
 * is needed: all access is via `textureLoad` (integer-coordinate fetch).
 *
 * Binding 12 — liteEnvTex     : RGBA32F envWidth×envHeight, .rgb = HDR radiance,
 *                                .a = pdf per steradian (mirrors texel.w in the
 *                                full-tier environmentMapTexels storage buffer).
 * Binding 13 — liteEnvCdfTex  : RGBA32F envWidth×envHeight, .r = normalised CDF
 *                                value for pixel (y*W+x) — used for importance
 *                                sampling.  cdf[0]=0 is implicit.
* Binding 14 — liteLightTex   : RGBA32F liteLightTexWidth×1, packed directional/
*                                point/spot/rect-area light records (same float
*                                layout as the full-tier directionalLights /
*                                pointLights / spotLights / rectAreaLights storage
*                                buffers; loaded via integer texel index).
*/
const PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_EXTRA_BINDINGS_WGSL = /* wgsl */ `
@group(0) @binding(12) var liteEnvTex:    texture_2d<f32>;
@group(0) @binding(13) var liteEnvCdfTex: texture_2d<f32>;
@group(0) @binding(14) var liteLightTex:  texture_2d<f32>;

// The lite tier has one packed mesh BLAS and no TLAS or analytic-shape
// buffers. A straight forward-only ray can therefore meet each packed
// triangle at most once; this is the complete pass-through traversal bound.
fn sceneSurfaceHitLimit() -> u32 {
  return min(params.triangleCount, arrayLength(&indices));
}

// The compatibility tier publishes scalar-only material behavior explicitly:
// it owns no material-texture bindings, so every texture query is the neutral
// element and every normal perturbation is identity. Keeping these named stubs
// lets shared optical evaluators (including compound-sheet visibility) execute
// the same control flow without pretending a texture was sampled.
fn sampleBaseColorTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> vec4f {
  return vec4f(1.0);
}
fn sampleVertexColor(triIndex: u32, baryVW: vec2f) -> vec4f {
  return vec4f(1.0);
}
fn sampleOrmTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> vec4f {
  return vec4f(1.0);
}
fn sampleAoFactor(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  return 1.0;
}
fn sampleTransmissionTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  return 1.0;
}
fn sampleIridescenceTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  return 1.0;
}
fn sampleIridescenceThicknessTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  return -1.0;
}
fn sampleSpecularColorTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> vec3f {
  return vec3f(1.0);
}
fn sampleSpecularIntensityTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  return 1.0;
}
fn sampleClearcoatTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  return 1.0;
}
fn sampleSheenColorTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> vec3f {
  return vec3f(1.0);
}
fn sampleSheenRoughnessTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  return 1.0;
}
fn applyNormalMap(matId: u32, triIndex: u32, baryVW: vec2f, geomNormal: vec3f, instanceIndex: u32, isFrontFace: bool) -> vec3f {
  return geomNormal;
}
fn applyBumpMap(matId: u32, triIndex: u32, baryVW: vec2f, shadingNormal: vec3f, instanceIndex: u32) -> vec3f {
  return shadingNormal;
}
fn applyClearcoatNormalMap(matId: u32, triIndex: u32, baryVW: vec2f, clearcoatNormal: vec3f, instanceIndex: u32) -> vec3f {
  return clearcoatNormal;
}
`;

/**
 * Lite-tier group-0 bindings: base (0–11) + B12 texture slots (12–14).
 * Used in the composed lite trace shader.
 */
const PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_BINDINGS_WGSL =
  PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_BINDINGS_BASE_WGSL +
  PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_EXTRA_BINDINGS_WGSL;

const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP0_WGSL = frameParamsGroup0Bindings(
  ' // UBO-plumbed (D12); default metre-scale',
  `
@group(0) @binding(12) var motionVectorsTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var<storage, read_write> varianceMomentsBuffer: array<vec4f>;`,
);

/** Group 1 — analytics + env + area lights + directional lights (11 storage buffers; adapters ≥11/stage). */
const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP1_WGSL = /* wgsl */ `
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
// N-directional: packed directional light records.
// Stride = 2 vec4f (8 floats) per directional:
//   [di*2+0]: towardLight.xyz, angularDiameter
//   [di*2+1]: irradiance.rgb,  distant-direct represented PMF
// directionalLightCount records total; an empty scene binds a 16-byte placeholder.
@group(1) @binding(10) var<storage, read> directionalLights: array<vec4f>;
`;

/** Group 2 — TLAS instance table (5 storage buffers).
 *
 *  BDPT light and eye prefixes are fixed invocation-private stacks. The light
 *  stack is eight columns × eight vec4 rows; it is generated inside each trace
 *  invocation and therefore has no viewport-sized or frame-shared resource.
 *  Layout: `maxLightBounces` columns × 8 rows of vec4f, flattened row-minor as
 *  `idx = col * BDPT_LIGHT_PATH_ROWS + row` (see `bdptLightPathIndex`). Per
 *  light-vertex: row 0 = pos (+ kind sentinel in .w), row 1 = normal + pdfFwd,
 *  row 2 = throughput + pdfRev, row 3 = (A9) matId + wo-toward-prev for the REAL
 *  light-vertex BSDF in the §10.3 connection (matId < 0 ⇒ emitter, Lambertian),
 *  row 4 = hit-local material coordinates (triIndex plus front-face bit, baryVW,
 *  instanceIndex) for texture-map/material-lobe sampling at surface light vertices,
 *  row 5 = eta_t/eta_i plus the incident/transmitted IOR pair used by nested
 *  dielectric reflection densities, row 6 = incident/transmitted medium IDs
 *  plus their remaining finite-distance budgets, and row 7 = both original
 *  finite-distance caps used to join two partially consumed BDPT halves.
 *  The eye prefix is likewise a fixed `var<private> array<BdptEyeVtx,8>`. */
const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP2_WGSL = /* wgsl */ `
@group(2) @binding(0) var<storage, read> tlasNodes: array<BVHNode>;
@group(2) @binding(1) var<storage, read> tlasInstanceIndices: array<u32>;
@group(2) @binding(2) var<storage, read> tlasBlasRoots: array<u32>;
@group(2) @binding(3) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(2) @binding(4) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
@group(2) @binding(5) var<storage, read> opticalInstanceBoundaryIdBasePlusOne: array<u32>;

fn sceneTraversalSaturatingAdd(a: u32, b: u32) -> u32 {
  if (b > 0xffffffffu - a) { return 0xffffffffu; }
  return a + b;
}

fn sceneTraversalSaturatingMul(a: u32, b: u32) -> u32 {
  if (a == 0u || b == 0u) { return 0u; }
  if (a > 0xffffffffu / b) { return 0xffffffffu; }
  return a * b;
}

// Proven upper bound on distinct surface hits along one forward-only straight
// ray. Each world-space triangle plane can be crossed once per TLAS instance.
// The four convex analytic families are closed solids and contribute at most
// entry+exit; the closed H-channel is the union of three boxes and therefore
// contributes at most six. A tangent double root cannot exceed those counts
// (and the forward origin step normally collapses it to one reported hit).
// Unknown analytic discriminants are not intersected by traceAnalyticShapes.
fn sceneSurfaceHitLimit() -> u32 {
  let triangleSupport = min(params.triangleCount, arrayLength(&indices));
  // The direct single-BLAS path can expose each packed triangle once. TLAS
  // traversal expands that support by the validated instance membership.
  var meshSupport = triangleSupport;
  if (params.tlasNodeCount > 0u) {
    let instanceSupport = min(
      arrayLength(&tlasInstanceIndices),
      arrayLength(&tlasBlasRoots),
    );
    meshSupport = sceneTraversalSaturatingMul(
      triangleSupport,
      instanceSupport,
    );
  }

  var analyticSupport = 0u;
  let analyticTotal = min(params.analyticCount, arrayLength(&analyticHeaders));
  for (var ai = 0u; ai < analyticTotal; ai = ai + 1u) {
    let shapeId = u32(max(analyticHeaders[ai].x, 0.0));
    var multiplicity = 0u;
    if (shapeId >= 1u && shapeId <= 4u) {
      multiplicity = 2u;
    } else if (shapeId == 5u) {
      multiplicity = 6u;
    }
    analyticSupport = sceneTraversalSaturatingAdd(
      analyticSupport,
      multiplicity,
    );
  }
  return sceneTraversalSaturatingAdd(meshSupport, analyticSupport);
}

// Light-path flat index: 8 vec4f rows per light-vertex column. Row 3 carries
// the reached vertex's matId + wo-toward-prev so the §10.3 connection can evaluate
// the REAL light-vertex BSDF for a glossy/metallic light-path vertex; row 4 carries
// hit-local tri/bary/instance payload for texture-map material sampling. The high
// bit of row-4.x stores the front-face flag; real triangle indices are required to
// stay below 2^31. matId < 0 marks the emitter vertex, which keeps its
// Lambertian/emission profile. Row 5 stores interface eta_t/eta_i and both IORs;
// row 6 stores both medium-side IDs and remaining-distance budgets; row 7 stores
// their original caps so a connection subtracts consumption from both halves.
const BDPT_LIGHT_PATH_ROWS = 8u;
const BDPT_MAX_LIGHT_DEPTH = 8u;
var<private> bdptLightPath: array<vec4f, 64>;
var<private> bdptInvocationHeroLambdaNm: f32;
fn bdptSetInvocationHeroLambda(heroLambdaNm: f32) {
  bdptInvocationHeroLambdaNm = heroLambdaNm;
}
fn bdptLightPathIndex(col: i32, row: u32) -> u32 {
  return u32(col) * BDPT_LIGHT_PATH_ROWS + row;
}
`;

/**
 * Group 3 — WS2 many-light importance sampling: the power-weighted light-tree
 * node buffer (FULL TIER ONLY). A DEDICATED bind group so the lite tier — which
 * keeps the uniform light pick and never composes this WGSL — is unaffected, and
 * so adding it does not perturb the existing group 0/1/2 layouts.
 *
 * The binding declaration + the `sampleLightTree` traversal come from the
 * canonical `@vitrum/shared-samplers` source (same descent as the CPU
 * `sampleLightTreeCPU` and walkaround-hybrid's ReSTIR-DI selection). `rand_f32`
 * (PCG) is already in scope from `PT_WEBGPU_COMMON_WGSL`.
 *
 * References: Conty Estévez & Kulla 2018 (power × proximity descent);
 * Shirley et al. 1996 (power-weighted light-list partition).
 */

// D9.2 — TS template that emits both material-layer sampler variants from one
// source of truth. The two WGSL functions differ only in:
//   (a) their name  (b) the texture array they sample
//   (c) the sRGB variant carries extra comment lines before `let xform` plus a
//       trailing comment on the `let xform` line itself
// WGSL cannot parameterise texture bindings, so a single WGSL fn is not possible;
// a TS helper keeps the shared body in one place while emitting byte-identical output.
// `preXformLines` is inserted verbatim before `let xform` (including its trailing newline+indent).
// `xformSuffix` is appended to the `let xform` line (empty string = no suffix).
function materialLayerSamplerWgsl(
  name: string,
  texArray: string,
  sourceRectSampleFunction: string,
  preXformLines: string,
  xformSuffix: string,
): string {
  return `fn ${name}(layerIdx: i32, base: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32, uvMetaOffset: u32, uvFitScale: vec2f, wrapMode: vec2f, mipPolicySlot: u32) -> MaterialTextureSample {
  if (
    layerIdx < 0 ||
    triIndex >= arrayLength(&indices) ||
    !materialTextureDescriptorSpanValid(base, uvMetaOffset, 2u) ||
    !materialTextureFiniteVec2(baryVW) ||
    !materialTextureFiniteVec2(uvFitScale) ||
    any(uvFitScale <= vec2f(0.0)) ||
    any(uvFitScale > vec2f(1.0)) ||
    !materialTextureWrapModesValid(wrapMode)
  ) { return materialTextureInvalidSample(); }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return materialTextureInvalidSample();
  }
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let uvMeta = materialTexDescriptors[base + uvMetaOffset];
  let uvScale = materialTexDescriptors[base + uvMetaOffset + 1u];
  if (!materialTextureFiniteVec4(uvMeta) || !materialTextureFiniteVec2(uvScale.xy)) {
    return materialTextureInvalidSample();
  }
  let vertexCount = arrayLength(&positions);
  if (vertexCount == 0u) { return materialTextureInvalidSample(); }
  let uvSlotCount = arrayLength(&meshUvs) / vertexCount;
  let gpuUvSlot = materialTextureExactU32(uvMeta.x, uvSlotCount);
  if (gpuUvSlot == 0xffffffffu) { return materialTextureInvalidSample(); }
  let rawA = materialUvForVertex(tri.x, gpuUvSlot);
  let rawB = materialUvForVertex(tri.y, gpuUvSlot);
  let rawC = materialUvForVertex(tri.z, gpuUvSlot);
  let rawUv = rawA * u + rawB * v + rawC * w;
  ${preXformLines}let xform = vec4f(uvMeta.y, uvMeta.z, uvScale.x, uvScale.y);${xformSuffix}
  let rot = uvMeta.w;
  if (
    !materialTextureFiniteVec2(rawA) ||
    !materialTextureFiniteVec2(rawB) ||
    !materialTextureFiniteVec2(rawC) ||
    !materialTextureFiniteVec2(rawUv)
  ) { return materialTextureInvalidSample(); }
  let c = cos(rot);
  let s = sin(rot);
  let sx = xform.z;
  let sy = xform.w;
  let uvACandidate = vec2f(
    sx * c * rawA.x - sy * s * rawA.y + xform.x,
    sx * s * rawA.x + sy * c * rawA.y + xform.y,
  );
  if (!materialTextureFiniteVec2(uvACandidate)) { return materialTextureInvalidSample(); }
  let uvA = uvACandidate;
  let uvBCandidate = vec2f(
    sx * c * rawB.x - sy * s * rawB.y + xform.x,
    sx * s * rawB.x + sy * c * rawB.y + xform.y,
  );
  if (!materialTextureFiniteVec2(uvBCandidate)) { return materialTextureInvalidSample(); }
  let uvB = uvBCandidate;
  let uvCCandidate = vec2f(
    sx * c * rawC.x - sy * s * rawC.y + xform.x,
    sx * s * rawC.x + sy * c * rawC.y + xform.y,
  );
  if (!materialTextureFiniteVec2(uvCCandidate)) { return materialTextureInvalidSample(); }
  let uvC = uvCCandidate;
  let uvCandidate = vec2f(
    sx * c * rawUv.x - sy * s * rawUv.y + xform.x,
    sx * s * rawUv.x + sy * c * rawUv.y + xform.y,
  );
  if (!materialTextureFiniteVec2(uvCandidate)) { return materialTextureInvalidSample(); }
  let uv = uvCandidate;
  let sourceBaseSize = materialTextureSourceBaseSize(
    vec2u(textureDimensions(${texArray}, i32(0))), uvFitScale,
  );
  let sourceMipCount = f32(materialTextureSourceMipCount(sourceBaseSize));
  let texDim = vec2f(sourceBaseSize);
  let texelAreaCandidate = max(abs((uvB.x - uvA.x) * (uvC.y - uvA.y) - (uvB.y - uvA.y) * (uvC.x - uvA.x)) * texDim.x * texDim.y, 1.0);
  let texelArea = select(
    1.0,
    texelAreaCandidate,
    materialTextureFiniteF32(texelAreaCandidate) && texelAreaCandidate >= 1.0,
  );
  let footprint = materialTextureWorldFootprint(tri, baryVW, instanceIndex);
  let worldArea = footprint.x;
  let cameraDistance = footprint.y;
  let pixelsPerMeter = 0.5 * f32(max(params.width, params.height)) / cameraDistance;
  let projectedPixels = max(sqrt(worldArea) * pixelsPerMeter, 1.0);
  let lodCandidate = clamp(log2(sqrt(texelArea) / projectedPixels), 0.0, max(sourceMipCount - 1.0, 0.0));
  let lod = select(0.0, lodCandidate, materialTextureFiniteF32(lodCandidate));
  let mipPolicy = materialTextureMipPolicy(base, mipPolicySlot);
  let policyLod = materialTexturePolicyLod(lod, sourceMipCount, mipPolicy);
  let filterPolicy = materialTextureFilterPolicy(base, mipPolicySlot);
  let filterMode = select(filterPolicy.x, filterPolicy.y, lod > 0.0);
  return ${sourceRectSampleFunction}(
    layerIdx, uv, sourceBaseSize, wrapMode,
    policyLod, filterMode, mipPolicy,
  );
}`;
}

const _MATERIAL_SOURCE_RECT_COORDINATES_WGSL = /* wgsl */ `
struct MaterialTextureSample {
  value: vec4f,
  valid: bool,
}

fn materialTextureInvalidSample() -> MaterialTextureSample {
  return MaterialTextureSample(vec4f(0.0), false);
}

fn materialTextureValueOr(sample: MaterialTextureSample, fallback: vec4f) -> vec4f {
  return select(fallback, sample.value, sample.valid);
}

fn materialTextureFiniteF32(value: f32) -> bool {
  return value == value && abs(value) <= PT_F32_MAX;
}

fn materialTextureFiniteVec2(value: vec2f) -> bool {
  return all(value == value) && all(abs(value) <= vec2f(PT_F32_MAX));
}

fn materialTextureFiniteVec3(value: vec3f) -> bool {
  return all(value == value) && all(abs(value) <= vec3f(PT_F32_MAX));
}

fn materialTextureFiniteVec4(value: vec4f) -> bool {
  return all(value == value) && all(abs(value) <= vec4f(PT_F32_MAX));
}

fn materialTextureNormalizeOr(value: vec3f, fallback: vec3f) -> vec3f {
  if (!materialTextureFiniteVec3(value)) { return fallback; }
  let magnitudeScale = max(max(abs(value.x), abs(value.y)), abs(value.z));
  if (!(magnitudeScale > 1e-20)) { return fallback; }
  let scaled = value / magnitudeScale;
  let scaledLength = length(scaled);
  if (!(scaledLength > 1e-6) || !materialTextureFiniteF32(scaledLength)) {
    return fallback;
  }
  return scaled / scaledLength;
}

fn materialTextureDescriptorBase(matId: u32) -> u32 {
  if (matId > 0xffffffffu / MATERIAL_TEX_VEC4_STRIDE) { return 0xffffffffu; }
  return matId * MATERIAL_TEX_VEC4_STRIDE;
}

fn materialTextureDescriptorSpanValid(base: u32, offset: u32, count: u32) -> bool {
  let descriptorCount = arrayLength(&materialTexDescriptors);
  if (count == 0u || base > descriptorCount) { return false; }
  let remaining = descriptorCount - base;
  if (offset > remaining) { return false; }
  return count <= remaining - offset;
}

fn materialTextureLayerIndex(value: f32, layerCount: u32) -> i32 {
  if (
    !materialTextureFiniteF32(value) ||
    value < 0.0 ||
    value != floor(value) ||
    value > 16777216.0 ||
    value >= f32(layerCount)
  ) {
    return -1;
  }
  return i32(value);
}

fn materialTextureExactU32(value: f32, upperExclusive: u32) -> u32 {
  if (
    !materialTextureFiniteF32(value) ||
    value < 0.0 ||
    value != floor(value) ||
    value > 16777216.0 ||
    value >= f32(upperExclusive)
  ) {
    return 0xffffffffu;
  }
  return u32(value);
}

fn materialTextureWrapModesValid(wrapMode: vec2f) -> bool {
  return materialTextureFiniteVec2(wrapMode) &&
    all(wrapMode >= vec2f(0.0)) &&
    all(wrapMode <= vec2f(2.0)) &&
    all(wrapMode == floor(wrapMode));
}

fn materialTextureTexelCastSafe(uv: vec2f, sourceSize: vec2u) -> bool {
  let texelPosition = uv * vec2f(sourceSize);
  return materialTextureFiniteVec2(texelPosition) &&
    all(abs(texelPosition) <= vec2f(1073741824.0));
}

fn materialTextureNearestCoord(uv: vec2f, sourceSize: vec2u, wrapMode: vec2f) -> vec2i {
  let unwrapped = vec2i(floor(uv * vec2f(sourceSize)));
  return vec2i(
    materialTextureWrapTexel(unwrapped.x, i32(sourceSize.x), wrapMode.x),
    materialTextureWrapTexel(unwrapped.y, i32(sourceSize.y), wrapMode.y),
  );
}

fn materialTextureBilinearAxis(baseCoord: i32, size: u32, wrapMode: f32) -> vec2i {
  return vec2i(
    materialTextureWrapTexel(baseCoord, i32(size), wrapMode),
    materialTextureWrapTexel(baseCoord + 1, i32(size), wrapMode),
  );
}
`;

function materialSourceRectSamplerWgsl(name: string, texArray: string): string {
  return /* wgsl */ `
fn ${name}Fetch(layerIdx: i32, coord: vec2i, mip: u32) -> MaterialTextureSample {
  // Keep the mip operand signed at the texture builtin boundary. WGSL accepts
  // an i32 mip level, and Naga's GLSL robustness lowering then emits the GLSL
  // textureSize(sampler2DArray, int) overload instead of an invalid uint call.
  if (layerIdx < 0 || u32(layerIdx) >= textureNumLayers(${texArray})) {
    return materialTextureInvalidSample();
  }
  if (mip >= textureNumLevels(${texArray})) {
    return materialTextureInvalidSample();
  }
  let size = textureDimensions(${texArray}, i32(mip));
  if (coord.x < 0 || coord.y < 0 || u32(coord.x) >= size.x || u32(coord.y) >= size.y) {
    return materialTextureInvalidSample();
  }
  let value = textureLoad(${texArray}, coord, layerIdx, i32(mip));
  if (!materialTextureFiniteVec4(value)) { return materialTextureInvalidSample(); }
  return MaterialTextureSample(value, true);
}

fn ${name}Bilinear(
  layerIdx: i32,
  uv: vec2f,
  sourceSize: vec2u,
  wrapMode: vec2f,
  mip: u32,
) -> MaterialTextureSample {
  let samplePosition = uv * vec2f(sourceSize) - vec2f(0.5);
  let baseCoord = vec2i(floor(samplePosition));
  let blend = fract(samplePosition);
  let xs = materialTextureBilinearAxis(baseCoord.x, sourceSize.x, wrapMode.x);
  let ys = materialTextureBilinearAxis(baseCoord.y, sourceSize.y, wrapMode.y);
  let c00 = ${name}Fetch(layerIdx, vec2i(xs.x, ys.x), mip);
  let c10 = ${name}Fetch(layerIdx, vec2i(xs.y, ys.x), mip);
  let c01 = ${name}Fetch(layerIdx, vec2i(xs.x, ys.y), mip);
  let c11 = ${name}Fetch(layerIdx, vec2i(xs.y, ys.y), mip);
  if (!(c00.valid && c10.valid && c01.valid && c11.valid)) {
    return materialTextureInvalidSample();
  }
  let value = mix(
    mix(c00.value, c10.value, blend.x),
    mix(c01.value, c11.value, blend.x),
    blend.y,
  );
  if (!materialTextureFiniteVec4(value)) { return materialTextureInvalidSample(); }
  return MaterialTextureSample(value, true);
}

fn ${name}AtMip(
  layerIdx: i32,
  uv: vec2f,
  sourceBaseSize: vec2u,
  wrapMode: vec2f,
  mip: u32,
  linearFilter: bool,
) -> MaterialTextureSample {
  if (!linearFilter) {
    let sourceSize = materialTextureSourceMipSize(sourceBaseSize, mip);
    let nearestCoord = materialTextureNearestCoord(uv, sourceSize, wrapMode);
    return ${name}Fetch(
      layerIdx, nearestCoord, mip,
    );
  }
  let sourceSize = materialTextureSourceMipSize(sourceBaseSize, mip);
  return ${name}Bilinear(layerIdx, uv, sourceSize, wrapMode, mip);
}

fn ${name}(
  layerIdx: i32,
  uv: vec2f,
  sourceBaseSize: vec2u,
  wrapMode: vec2f,
  policyLod: f32,
  filterMode: f32,
  mipPolicy: f32,
) -> MaterialTextureSample {
  if (
    !materialTextureFiniteF32(policyLod) ||
    !materialTextureFiniteF32(filterMode) ||
    !materialTextureFiniteF32(mipPolicy) ||
    !materialTextureWrapModesValid(wrapMode) ||
    !materialTextureTexelCastSafe(uv, sourceBaseSize) ||
    filterMode < 0.0 || filterMode > 1.0 ||
    mipPolicy < 0.0 || mipPolicy > 2.0
  ) {
    return materialTextureInvalidSample();
  }
  let maxMip = materialTextureSourceMipCount(sourceBaseSize) - 1u;
  let lod0 = min(u32(floor(max(policyLod, 0.0))), maxMip);
  let lod1 = select(lod0 + 1u, lod0, lod0 == maxMip);
  let linearFilter = filterMode >= 0.5;
  let c0 = ${name}AtMip(layerIdx, uv, sourceBaseSize, wrapMode, lod0, linearFilter);
  let c1 = ${name}AtMip(layerIdx, uv, sourceBaseSize, wrapMode, lod1, linearFilter);
  if (!(c0.valid && c1.valid)) { return materialTextureInvalidSample(); }
  let mipBlend = select(0.0, fract(policyLod), mipPolicy >= 1.5);
  let value = mix(c0.value, c1.value, mipBlend);
  if (!materialTextureFiniteVec4(value)) { return materialTextureInvalidSample(); }
  return MaterialTextureSample(value, true);
}
`;
}

const _MATERIAL_SOURCE_RECT_SAMPLERS_WGSL =
  _MATERIAL_SOURCE_RECT_COORDINATES_WGSL +
  materialSourceRectSamplerWgsl('sampleMaterialSrgbSourceRect', 'materialTextures') +
  materialSourceRectSamplerWgsl('sampleMaterialLinearSourceRect', 'materialTexturesLinear') +
  materialSourceRectSamplerWgsl('sampleMaterialEmissiveSourceRect', 'materialTexturesEmissive');

// Pre-stamp the three variants outside the template literal so the `${...}`
// interpolations inside the WGSL template stay simple (no nested escaping).
// sRGB variant (materialTextures — baseColor and extension color maps):
const _SAMPLE_MAT_LAYER_WGSL = materialLayerSamplerWgsl(
  'sampleMaterialLayer',
  'materialTextures',
  'sampleMaterialSrgbSourceRect',
  // KHR comment block that precedes `let xform` in the sRGB variant:
  '// KHR_texture_transform — core contract: rotate(uv * scale, rotation) + offset.\n' +
  "  // u' = sx·c·u - sy·s·v + tx ;  v' = sx·s·u + sy·c·v + ty.\n" +
  '  ',
  ' // offset.xy, scale.xy',
);
// Linear variant (materialTexturesLinear — normal + scalar/data maps):
const _SAMPLE_MAT_LAYER_LINEAR_WGSL = materialLayerSamplerWgsl(
  'sampleMaterialLayerLinear',
  'materialTexturesLinear',
  'sampleMaterialLinearSourceRect',
  '',
  '',
);
// T1-6 — outgoing-radiance variant (materialTexturesEmissive — dedicated
// rgba16float array; historical binding name retained). Samples exactly like the
// sRGB variant but from the HDR radiance array. Per-layer upload provenance
// determines whether LDR bytes are sRGB-decoded (emissiveMap) or remain linear
// (lightMap); Float32 payloads are linear HDR in both roles. Standalone fn — WGSL
// cannot pass a texture as an argument, hence this parallel sampler.
const _SAMPLE_MAT_LAYER_EMISSIVE_WGSL = materialLayerSamplerWgsl(
  'sampleMaterialLayerEmissive',
  'materialTexturesEmissive',
  'sampleMaterialEmissiveSourceRect',
  // Same KHR UV-transform preamble as the sRGB variant (emissive maps carry the
  // same KHR_texture_transform metadata as baseColor).
  '// KHR_texture_transform — core contract: rotate(uv * scale, rotation) + offset.\n' +
  "  // u' = sx·c·u - sy·s·v + tx ;  v' = sx·s·u + sy·c·v + ty.\n" +
  '  ',
  ' // offset.xy, scale.xy',
);

export const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL =
  /* wgsl */ `
// ============================================================
// WS2 — light-tree storage buffer (full-tier @group(3)) + importance traversal
// ============================================================
${lightTreeWgsl({ group: 3, binding: 0, rngStateType: 'PtRngState' })}
// Selection-only proximity floor for the light-tree descent (caps distance
// importance near a light; NOT the NEE geometry-term clamp). Metre-scale.
const LT_DIST2_FLOOR: f32 = 1e-3;

// ============================================================
// P2 — material textures (full-tier @group(3)): per-vertex UVs + per-material
// descriptors + the sampled baseColor texture_2d_array + a filtering sampler.
// A DEDICATED set of bindings in the existing full-tier group so neither the
// lite tier (no group 3) nor the existing group 0/1/2 layouts are perturbed.
// ============================================================
@group(3) @binding(1) var<storage, read> meshUvs: array<vec4f>;       // primary uv0/uv1 plane + compact UV planes
@group(3) @binding(2) var<storage, read> materialTexDescriptors: array<vec4f>;
@group(3) @binding(3) var materialTextures: texture_2d_array<f32>;        // sRGB (baseColor + color maps)
@group(3) @binding(4) var materialTexSampler: sampler;                    // shared by both arrays
@group(3) @binding(5) var materialTexturesLinear: texture_2d_array<f32>;  // LINEAR (normal + scalar maps)
@group(3) @binding(10) var<storage, read> meshTangents: array<vec4f>;      // xyz = tangent, w = bitangent sign
// T1-6 — dedicated outgoing-RADIANCE rgba16float array. HDR emissive and light-
// map values > 1.0 survive here (the 8-bit arrays clamp to [0,1]). Sampled as
// texture_2d_array<f32>; binding 17 is the next free group-3 slot after the
// tangents/colors/SPPM/CWBVH bindings (6..16).
@group(3) @binding(17) var materialTexturesEmissive: texture_2d_array<f32>;
@group(3) @binding(11) var<storage, read> meshVertexColors: array<vec4f>;  // rgba = glTF COLOR_0, defaults to 1

// GPU slots 0/1 live in the primary per-vertex vec4. Every compact slot >= 2
// owns one appended vec4 plane (xy used). Host descriptors map sparse authored
// TextureRef.texCoord values to these slots, avoiding an allocation proportional
// to the largest sparse TEXCOORD_n index.
fn materialUvForVertex(vertexIndex: u32, gpuUvSlot: u32) -> vec2f {
  let vertexCount = arrayLength(&positions);
  let uvCount = arrayLength(&meshUvs);
  if (vertexCount == 0u || vertexIndex >= vertexCount || vertexIndex >= uvCount) {
    return vec2f(0.0);
  }
  let primary = meshUvs[vertexIndex];
  if (gpuUvSlot == 0u) { return primary.xy; }
  if (gpuUvSlot == 1u) { return primary.zw; }
  let tailPlane = gpuUvSlot - 1u;
  if (tailPlane > (uvCount - 1u - vertexIndex) / vertexCount) {
    return vec2f(0.0);
  }
  let tailIndex = tailPlane * vertexCount + vertexIndex;
  if (tailIndex >= uvCount) { return vec2f(0.0); }
  return meshUvs[tailIndex].xy;
}

// vec4s per material in the descriptor buffer — MUST match the TS
// MATERIAL_TEX_VEC4_STRIDE in scene/materialTextures.ts.
//   0: {baseColorIdx, normalIdx, roughnessMapIdx, emissiveIdx}   (-1 = no map)
//   1: {alphaMode, alphaCutoff, opacity, texCoord}
//   2: {offset.xy, scale.xy}
//   3: {rotation, aoMapIdx, lightMapIdx, bumpMapIdx}      ← D3 (-1 = no map)
//   4: {aoMapIntensity, lightMapIntensity, bumpScale, envMapIntensity}  ← D3
//   5: {anisotropy, anisotropyRotation, anisotropyMapIdx, normalScale}  ← D3/PTWG-MAT
//   6: {alphaMapIdx, transmissionMapIdx, metallicMapIdx, _}      (-1 = no map)
//   7: {baseColorUvScale.xy, emissiveUvScale.xy}
//   8: {normalUvScale.xy, roughnessUvScale.xy}
//   9: {metallicUvScale.xy, aoUvScale.xy}
//  10: {lightMapUvScale.xy, bumpUvScale.xy}
//  11: {anisotropyUvScale.xy, alphaUvScale.xy}
//  12: {transmissionUvScale.xy, _, _}
//  13: {baseColorWrap.xy, emissiveWrap.xy}      (0 repeat / 1 clamp / 2 mirror)
//  14: {normalWrap.xy, roughnessWrap.xy}
//  15: {metallicWrap.xy, aoWrap.xy}
//  16: {lightMapWrap.xy, bumpWrap.xy}
//  17: {anisotropyWrap.xy, alphaWrap.xy}
//  18: {transmissionWrap.xy, _, _}
//  19-40: per-map UV metadata, two vec4s per consumed base map:
//     A = {texCoord, offsetX, offsetY, rotation}
//     B = {scaleX, scaleY, 0, 0}
//  41-42: extension-lobe texture indices
//  43-46: extension-lobe UV-fit scale pairs
//  47-50: extension-lobe wrap pairs
//  51-66: extension-lobe UV metadata, two vec4s per extension map
//  67: {clearcoatNormalMapIdx, clearcoatNormalScale, clearcoatNormalUvFit.xy}
//  68: {clearcoatNormalWrap.xy, 0, 0}
//  69-70: clearcoat-normal UV metadata
//  71: {thicknessMapIdx, thicknessUvFit.xy, _}
//  72: {thicknessWrap.xy, 0, 0}
//  73-74: thicknessMap UV metadata
//  75: {frontLayerNormalMapIdx, frontLayerNormalScale,
//       backLayerNormalMapIdx, backLayerNormalScale}
//  76: {frontLayerNormalUvFit.xy, backLayerNormalUvFit.xy}
//  77: {frontLayerNormalWrap.xy, backLayerNormalWrap.xy}
//  78-81: front/back layer-normal UV metadata
//  82-87: per-map mip policy, one scalar per map:
//     0 none / 1 nearest / 2 linear
//  88-99: per-map filter policy, two scalars per map:
//     {magFilter, minFilter}, 0 nearest / 1 linear
const MATERIAL_TEX_VEC4_STRIDE = ${MATERIAL_TEX_VEC4_STRIDE}u;
const MATERIAL_TEX_UV_BASE_COLOR = ${MATERIAL_TEX_UV_META_VEC4_OFFSET}u;
const MATERIAL_TEX_UV_EMISSIVE = ${MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP}u;
const MATERIAL_TEX_UV_NORMAL = ${MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 2}u;
const MATERIAL_TEX_UV_ROUGHNESS = ${MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 3}u;
const MATERIAL_TEX_UV_METALLIC = ${MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 4}u;
const MATERIAL_TEX_UV_AO = ${MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 5}u;
const MATERIAL_TEX_UV_LIGHT = ${MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 6}u;
const MATERIAL_TEX_UV_BUMP = ${MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 7}u;
const MATERIAL_TEX_UV_ANISOTROPY = ${MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 8}u;
const MATERIAL_TEX_UV_ALPHA = ${MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 9}u;
const MATERIAL_TEX_UV_TRANSMISSION = ${MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 10}u;
const MATERIAL_TEX_EXTENSION_INDEX = ${MATERIAL_TEX_EXTENSION_INDEX_VEC4_OFFSET}u;
const MATERIAL_TEX_EXTENSION_UV_FIT = ${MATERIAL_TEX_EXTENSION_UV_FIT_VEC4_OFFSET}u;
const MATERIAL_TEX_EXTENSION_WRAP = ${MATERIAL_TEX_EXTENSION_WRAP_VEC4_OFFSET}u;
const MATERIAL_TEX_UV_CLEARCOAT = ${MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET}u;
const MATERIAL_TEX_UV_CLEARCOAT_ROUGHNESS = ${MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP}u;
const MATERIAL_TEX_UV_SHEEN_COLOR = ${MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 2}u;
const MATERIAL_TEX_UV_SHEEN_ROUGHNESS = ${MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 3}u;
const MATERIAL_TEX_UV_IRIDESCENCE = ${MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 4}u;
const MATERIAL_TEX_UV_IRIDESCENCE_THICKNESS = ${MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 5}u;
const MATERIAL_TEX_UV_SPECULAR_COLOR = ${MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 6}u;
const MATERIAL_TEX_UV_SPECULAR_INTENSITY = ${MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 7}u;
const MATERIAL_TEX_CLEARCOAT_NORMAL = ${MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET}u;
const MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP = ${MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP_VEC4_OFFSET}u;
const MATERIAL_TEX_UV_CLEARCOAT_NORMAL = ${MATERIAL_TEX_CLEARCOAT_NORMAL_UV_META_VEC4_OFFSET}u;
const MATERIAL_TEX_THICKNESS = ${MATERIAL_TEX_THICKNESS_VEC4_OFFSET}u;
const MATERIAL_TEX_THICKNESS_WRAP = ${MATERIAL_TEX_THICKNESS_WRAP_VEC4_OFFSET}u;
const MATERIAL_TEX_UV_THICKNESS = ${MATERIAL_TEX_THICKNESS_UV_META_VEC4_OFFSET}u;
const MATERIAL_TEX_LAYER_NORMAL = ${MATERIAL_TEX_LAYER_NORMAL_VEC4_OFFSET}u;
const MATERIAL_TEX_LAYER_NORMAL_UV_FIT = ${MATERIAL_TEX_LAYER_NORMAL_UV_FIT_VEC4_OFFSET}u;
const MATERIAL_TEX_LAYER_NORMAL_WRAP = ${MATERIAL_TEX_LAYER_NORMAL_WRAP_VEC4_OFFSET}u;
const MATERIAL_TEX_UV_FRONT_LAYER_NORMAL = ${MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET}u;
const MATERIAL_TEX_UV_BACK_LAYER_NORMAL = ${MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP}u;
const MATERIAL_TEX_MIP_POLICY = ${MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET}u;
const MATERIAL_TEX_MIP_BASE_COLOR = 0u;
const MATERIAL_TEX_MIP_EMISSIVE = 1u;
const MATERIAL_TEX_MIP_NORMAL = 2u;
const MATERIAL_TEX_MIP_ROUGHNESS = 3u;
const MATERIAL_TEX_MIP_METALLIC = 4u;
const MATERIAL_TEX_MIP_AO = 5u;
const MATERIAL_TEX_MIP_LIGHT = 6u;
const MATERIAL_TEX_MIP_BUMP = 7u;
const MATERIAL_TEX_MIP_ANISOTROPY = 8u;
const MATERIAL_TEX_MIP_ALPHA = 9u;
const MATERIAL_TEX_MIP_TRANSMISSION = 10u;
const MATERIAL_TEX_MIP_CLEARCOAT = 11u;
const MATERIAL_TEX_MIP_CLEARCOAT_ROUGHNESS = 12u;
const MATERIAL_TEX_MIP_SHEEN_COLOR = 13u;
const MATERIAL_TEX_MIP_SHEEN_ROUGHNESS = 14u;
const MATERIAL_TEX_MIP_IRIDESCENCE = 15u;
const MATERIAL_TEX_MIP_IRIDESCENCE_THICKNESS = 16u;
const MATERIAL_TEX_MIP_SPECULAR_COLOR = 17u;
const MATERIAL_TEX_MIP_SPECULAR_INTENSITY = 18u;
const MATERIAL_TEX_MIP_CLEARCOAT_NORMAL = 19u;
const MATERIAL_TEX_MIP_THICKNESS = 20u;
const MATERIAL_TEX_MIP_FRONT_LAYER_NORMAL = 21u;
const MATERIAL_TEX_MIP_BACK_LAYER_NORMAL = 22u;
const MATERIAL_TEX_FILTER_POLICY = ${MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET}u;

fn wrapTextureCoord(coord: f32, mode: f32) -> f32 {
  if (
    !materialTextureFiniteF32(coord) ||
    !materialTextureFiniteF32(mode) ||
    mode < 0.0 || mode > 2.0 || mode != floor(mode)
  ) { return 0.0; }
  let m = u32(mode);
  if (m == 1u) {
    return min(clamp(coord, 0.0, 1.0), 0.999999);
  }
  if (m == 2u) {
    let period = coord - 2.0 * floor(coord * 0.5);
    let mirrored = select(2.0 - period, period, period <= 1.0);
    return min(max(mirrored, 0.0), 0.999999);
  }
  return fract(coord);
}

fn materialTextureSourceExtent(arrayExtent: u32, uvFitScale: f32) -> u32 {
  if (
    arrayExtent == 0u ||
    !materialTextureFiniteF32(uvFitScale) ||
    !(uvFitScale > 0.0) ||
    uvFitScale > 1.0
  ) { return 1u; }
  let candidate = round(f32(arrayExtent) * uvFitScale);
  if (
    !materialTextureFiniteF32(candidate) ||
    candidate < 1.0 ||
    candidate > f32(arrayExtent) ||
    candidate > 16777216.0
  ) { return 1u; }
  return u32(candidate);
}

fn materialTextureSourceBaseSize(arraySize: vec2u, uvFitScale: vec2f) -> vec2u {
  // uvFitScale is packed as sourceSize / arraySize. WebGPU's dimension limit is
  // far below f32's exact-integer ceiling, so round(product) recovers the exact
  // authored integer extent instead of inheriting ratio roundoff.
  return vec2u(
    materialTextureSourceExtent(arraySize.x, uvFitScale.x),
    materialTextureSourceExtent(arraySize.y, uvFitScale.y),
  );
}

fn materialTextureSourceMipCount(sourceBaseSize: vec2u) -> u32 {
  return 1u + u32(floor(log2(f32(max(sourceBaseSize.x, sourceBaseSize.y)))));
}

fn materialTextureSourceMipSize(sourceBaseSize: vec2u, mip: u32) -> vec2u {
  return vec2u(
    max(1u, sourceBaseSize.x >> mip),
    max(1u, sourceBaseSize.y >> mip),
  );
}

fn materialTexturePositiveModulo(value: i32, modulus: i32) -> i32 {
  let remainder = value % modulus;
  return select(remainder + modulus, remainder, remainder >= 0);
}

fn materialTextureWrapTexel(index: i32, size: i32, modeValue: f32) -> i32 {
  if (size <= 1) { return 0; }
  if (
    !materialTextureFiniteF32(modeValue) ||
    modeValue < 0.0 || modeValue > 2.0 || modeValue != floor(modeValue)
  ) { return 0; }
  let mode = u32(modeValue);
  if (mode == 1u) { return clamp(index, 0, size - 1); }
  if (mode == 2u) {
    let period = 2 * size;
    let folded = materialTexturePositiveModulo(index, period);
    return select(period - 1 - folded, folded, folded < size);
  }
  return materialTexturePositiveModulo(index, size);
}

fn materialTextureMipPolicy(base: u32, slot: u32) -> f32 {
  let laneGroup = slot / 4u;
  if (laneGroup > 0xffffffffu - MATERIAL_TEX_MIP_POLICY) { return -1.0; }
  let offset = MATERIAL_TEX_MIP_POLICY + laneGroup;
  if (!materialTextureDescriptorSpanValid(base, offset, 1u)) { return -1.0; }
  let vecIdx = base + offset;
  let packed = materialTexDescriptors[vecIdx];
  if (!materialTextureFiniteVec4(packed)) { return -1.0; }
  let lane = slot - laneGroup * 4u;
  if (lane == 0u) { return packed.x; }
  if (lane == 1u) { return packed.y; }
  if (lane == 2u) { return packed.z; }
  return packed.w;
}

fn materialTexturePolicyLod(lod: f32, mipCount: f32, mipPolicy: f32) -> f32 {
  let maxLod = max(mipCount - 1.0, 0.0);
  if (mipPolicy < 0.5) {
    return 0.0;
  }
  if (mipPolicy < 1.5) {
    return clamp(floor(lod + 0.5), 0.0, maxLod);
  }
  return clamp(lod, 0.0, maxLod);
}

fn materialTextureFilterPolicy(base: u32, slot: u32) -> vec2f {
  if (slot > 0xffffffffu / 2u) { return vec2f(-1.0); }
  let scalarOffset = slot * 2u;
  let laneGroup = scalarOffset / 4u;
  if (laneGroup > 0xffffffffu - MATERIAL_TEX_FILTER_POLICY) { return vec2f(-1.0); }
  let offset = MATERIAL_TEX_FILTER_POLICY + laneGroup;
  if (!materialTextureDescriptorSpanValid(base, offset, 1u)) { return vec2f(-1.0); }
  let vecIdx = base + offset;
  let packed = materialTexDescriptors[vecIdx];
  if (!materialTextureFiniteVec4(packed)) { return vec2f(-1.0); }
  let lane = scalarOffset - laneGroup * 4u;
  if (lane == 0u) { return packed.xy; }
  return packed.zw;
}

// Texture LOD geometry is evaluated entirely in world space. TLAS mesh
// positions are BLAS-local, so transform all three vertices through the hit
// instance before measuring area or camera distance. Merged geometry carries
// INVALID_TLAS_INSTANCE_INDEX and is already world-space.
fn materialTexturePointToWorld(point: vec3f, instanceIndex: u32) -> vec3f {
  if (instanceIndex == INVALID_TLAS_INSTANCE_INDEX) { return point; }
  let m = instanceIndex * 4u;
  if (m + 3u >= arrayLength(&tlasInstanceLocalToWorld)) { return point; }
  return transformPointCols(
    tlasInstanceLocalToWorld[m],
    tlasInstanceLocalToWorld[m + 1u],
    tlasInstanceLocalToWorld[m + 2u],
    tlasInstanceLocalToWorld[m + 3u],
    point,
  );
}

fn materialTextureWorldFootprint(tri: vec4u, baryVW: vec2f, instanceIndex: u32) -> vec2f {
  let pa = materialTexturePointToWorld(positions[tri.x].xyz, instanceIndex);
  let pb = materialTexturePointToWorld(positions[tri.y].xyz, instanceIndex);
  let pc = materialTexturePointToWorld(positions[tri.z].xyz, instanceIndex);
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let worldArea = measureAreaVector(pb - pa, pc - pa, 0.5).area;
  let worldHitPos = pa * u + pb * v + pc * w;
  let cameraDistance = max(
    safe_length(worldHitPos - params.cameraPos.xyz),
    ptRayTMin(),
  );
  return vec2f(worldArea, cameraDistance);
}

// Sample array layer \`layerIdx\` for material \`base\` (= matId·stride) at the hit:
// interpolate the per-vertex UV by the hit barycentrics, apply the material's
// KHR_texture_transform, sample the indexed layer. Returns vec4(1) — a no-op
// multiply — when layerIdx < 0 or the hit is not a mesh triangle (analytic shapes
// carry no UVs in v1), so a material lacking that map stays byte-identical.
// textureSampleLevel (explicit LOD) keeps the call valid in non-uniform flow.
// LOD is estimated from triangle UV density, projected world area, and camera
// distance because compute shaders do not have implicit screen derivatives.
// Each map reads its own TextureRef.texCoord + KHR_texture_transform metadata.
${_MATERIAL_SOURCE_RECT_SAMPLERS_WGSL}
${_SAMPLE_MAT_LAYER_WGSL}

// baseColor map (sRGB array) — descriptor vec4[0].x.
fn sampleBaseColorTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> vec4f {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, 14u)) { return vec4f(1.0); }
  let layerIdx = materialTextureLayerIndex(
    materialTexDescriptors[base].x,
    textureNumLayers(materialTextures),
  );
  let sample = sampleMaterialLayer(layerIdx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_BASE_COLOR, materialTexDescriptors[base + 7u].xy, materialTexDescriptors[base + 13u].xy, MATERIAL_TEX_MIP_BASE_COLOR);
  return materialTextureValueOr(sample, vec4f(1.0));
}

fn sampleVertexColor(triIndex: u32, baryVW: vec2f) -> vec4f {
  if (triIndex >= arrayLength(&indices)) { return vec4f(1.0); }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshVertexColors) || tri.y >= arrayLength(&meshVertexColors) || tri.z >= arrayLength(&meshVertexColors)) {
    return vec4f(1.0);
  }
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let value = meshVertexColors[tri.x] * u + meshVertexColors[tri.y] * v + meshVertexColors[tri.z] * w;
  return select(vec4f(1.0), value, materialTextureFiniteVec4(value));
}

// T1-6 — emissive sampler variant (dedicated rgba16float emissive array).
${_SAMPLE_MAT_LAYER_EMISSIVE_WGSL}

// emissive map (dedicated rgba16float emissive array) — descriptor vec4[0].w.
// HDR emissive texture values > 1.0 are preserved (the sRGB 8-bit array clamped
// them). emissiveIdx indexes the emissive array's layer space (materialTextures.ts
// indexOfEmissive), and the UV-fit lane vec4[7].zw is filled from the emissive
// array's per-layer scales.
fn sampleEmissiveTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> vec4f {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, 14u)) { return vec4f(1.0); }
  let layerIdx = materialTextureLayerIndex(
    materialTexDescriptors[base].w,
    textureNumLayers(materialTexturesEmissive),
  );
  let sample = sampleMaterialLayerEmissive(layerIdx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_EMISSIVE, materialTexDescriptors[base + 7u].zw, materialTexDescriptors[base + 13u].zw, MATERIAL_TEX_MIP_EMISSIVE);
  return materialTextureValueOr(sample, vec4f(1.0));
}

// As sampleMaterialLayer, but samples the LINEAR array (materialTexturesLinear)
// — for normal + scalar/data maps, which must NOT be sRGB-decoded. Standalone (not a
// refactor of sampleMaterialLayer) so the validated sRGB path is untouched;
// WGSL can't pass a texture as an argument, hence the parallel function.
${_SAMPLE_MAT_LAYER_LINEAR_WGSL}

fn sampleMaterialLayerLinearRawUvPolicy(layerIdx: i32, base: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32, rawUv: vec2f, uvMetaOffset: u32, uvFitScale: vec2f, wrapMode: vec2f, mipPolicySlot: u32) -> MaterialTextureSample {
  if (
    layerIdx < 0 ||
    triIndex >= arrayLength(&indices) ||
    !materialTextureDescriptorSpanValid(base, uvMetaOffset, 2u) ||
    !materialTextureFiniteVec2(baryVW) ||
    !materialTextureFiniteVec2(rawUv) ||
    !materialTextureFiniteVec2(uvFitScale) ||
    any(uvFitScale <= vec2f(0.0)) ||
    any(uvFitScale > vec2f(1.0)) ||
    !materialTextureWrapModesValid(wrapMode)
  ) { return materialTextureInvalidSample(); }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs) ||
      tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return materialTextureInvalidSample();
  }
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let uvMeta = materialTexDescriptors[base + uvMetaOffset];
  let uvScale = materialTexDescriptors[base + uvMetaOffset + 1u];
  if (!materialTextureFiniteVec4(uvMeta) || !materialTextureFiniteVec2(uvScale.xy)) {
    return materialTextureInvalidSample();
  }
  let vertexCount = arrayLength(&positions);
  if (vertexCount == 0u) { return materialTextureInvalidSample(); }
  let gpuUvSlot = materialTextureExactU32(
    uvMeta.x,
    arrayLength(&meshUvs) / vertexCount,
  );
  if (gpuUvSlot == 0xffffffffu) { return materialTextureInvalidSample(); }
  let xform = vec4f(uvMeta.y, uvMeta.z, uvScale.x, uvScale.y);
  let rot = uvMeta.w;
  let c = cos(rot);
  let s = sin(rot);
  let sx = xform.z;
  let sy = xform.w;
  let rawA = materialUvForVertex(tri.x, gpuUvSlot);
  let rawB = materialUvForVertex(tri.y, gpuUvSlot);
  let rawC = materialUvForVertex(tri.z, gpuUvSlot);
  let uvACandidate = vec2f(
    sx * c * rawA.x - sy * s * rawA.y + xform.x,
    sx * s * rawA.x + sy * c * rawA.y + xform.y,
  );
  if (!materialTextureFiniteVec2(uvACandidate)) { return materialTextureInvalidSample(); }
  let uvA = uvACandidate;
  let uvBCandidate = vec2f(
    sx * c * rawB.x - sy * s * rawB.y + xform.x,
    sx * s * rawB.x + sy * c * rawB.y + xform.y,
  );
  if (!materialTextureFiniteVec2(uvBCandidate)) { return materialTextureInvalidSample(); }
  let uvB = uvBCandidate;
  let uvCCandidate = vec2f(
    sx * c * rawC.x - sy * s * rawC.y + xform.x,
    sx * s * rawC.x + sy * c * rawC.y + xform.y,
  );
  if (!materialTextureFiniteVec2(uvCCandidate)) { return materialTextureInvalidSample(); }
  let uvC = uvCCandidate;
  let uvCandidate = vec2f(
    sx * c * rawUv.x - sy * s * rawUv.y + xform.x,
    sx * s * rawUv.x + sy * c * rawUv.y + xform.y,
  );
  if (!materialTextureFiniteVec2(uvCandidate)) { return materialTextureInvalidSample(); }
  let uv = uvCandidate;
  let sourceBaseSize = materialTextureSourceBaseSize(
    vec2u(textureDimensions(materialTexturesLinear, i32(0))), uvFitScale,
  );
  let sourceMipCount = f32(materialTextureSourceMipCount(sourceBaseSize));
  let texDim = vec2f(sourceBaseSize);
  let texelAreaCandidate = max(abs((uvB.x - uvA.x) * (uvC.y - uvA.y) - (uvB.y - uvA.y) * (uvC.x - uvA.x)) * texDim.x * texDim.y, 1.0);
  let texelArea = select(
    1.0,
    texelAreaCandidate,
    materialTextureFiniteF32(texelAreaCandidate) && texelAreaCandidate >= 1.0,
  );
  let footprint = materialTextureWorldFootprint(tri, baryVW, instanceIndex);
  let worldArea = footprint.x;
  let cameraDistance = footprint.y;
  let pixelsPerMeter = 0.5 * f32(max(params.width, params.height)) / cameraDistance;
  let projectedPixels = max(sqrt(worldArea) * pixelsPerMeter, 1.0);
  let lodCandidate = clamp(log2(sqrt(texelArea) / projectedPixels), 0.0, max(sourceMipCount - 1.0, 0.0));
  let lod = select(0.0, lodCandidate, materialTextureFiniteF32(lodCandidate));
  let mipPolicy = materialTextureMipPolicy(base, mipPolicySlot);
  let policyLod = materialTexturePolicyLod(lod, sourceMipCount, mipPolicy);
  let filterPolicy = materialTextureFilterPolicy(base, mipPolicySlot);
  let filterMode = select(filterPolicy.x, filterPolicy.y, lod > 0.0);
  return sampleMaterialLinearSourceRect(
    layerIdx, uv, sourceBaseSize, wrapMode,
    policyLod, filterMode, mipPolicy,
  );
}

// Roughness/metallic maps (linear array). glTF's canonical metallicRoughness
// texture packs roughness in G and metallic in B, and the host packer preserves
// that by pointing both descriptors at the same layer when a combined map is
// supplied. Distinct authored maps keep independent UV/wrap metadata.
fn sampleOrmTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> vec4f {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, 16u)) { return vec4f(1.0); }
  let roughnessSample = sampleMaterialLayerLinear(
    materialTextureLayerIndex(
      materialTexDescriptors[base].z,
      textureNumLayers(materialTexturesLinear),
    ),
    base,
    triIndex,
    baryVW,
    instanceIndex,
    MATERIAL_TEX_UV_ROUGHNESS,
    materialTexDescriptors[base + 8u].zw,
    materialTexDescriptors[base + 14u].zw,
    MATERIAL_TEX_MIP_ROUGHNESS,
  );
  let metallicSample = sampleMaterialLayerLinear(
    materialTextureLayerIndex(
      materialTexDescriptors[base + 6u].z,
      textureNumLayers(materialTexturesLinear),
    ),
    base,
    triIndex,
    baryVW,
    instanceIndex,
    MATERIAL_TEX_UV_METALLIC,
    materialTexDescriptors[base + 9u].xy,
    materialTexDescriptors[base + 15u].xy,
    MATERIAL_TEX_MIP_METALLIC,
  );
  let roughness = materialTextureValueOr(roughnessSample, vec4f(1.0)).g;
  let metallic = materialTextureValueOr(metallicSample, vec4f(1.0)).b;
  return vec4f(1.0, roughness, metallic, 1.0);
}

// D9.3 / H52 — shared tangent-frame builder for normal/bump/clearcoat-normal maps.
// Prefer authored/generated glTF tangent.xyzw (including handedness) when present;
// fall back to Lengyel UV-gradient derivation for old scenes with no tangent data.
// Ref: Lengyel, "Computing Tangent Space Basis Vectors for an Arbitrary Mesh".
struct ShadingTangentFrame {
  tangent: vec3f,
  bitangent: vec3f,
  valid: bool,
}
fn buildShadingTangentFrame(triIndex: u32, baryVW: vec2f, normal: vec3f, gpuUvSlot: u32, instanceIndex: u32) -> ShadingTangentFrame {
  var frame: ShadingTangentFrame;
  frame.valid = false;
  let tri = indices[triIndex];
  // glTF tangent.xyzw is defined against TEXCOORD_0. It is not a valid basis
  // for a normal/height map selecting another UV set; derive that set's basis
  // from its own compact GPU UV plane instead.
  if (gpuUvSlot == 0u && tri.x < arrayLength(&meshTangents) && tri.y < arrayLength(&meshTangents) && tri.z < arrayLength(&meshTangents)) {
    let v = baryVW.x;
    let w = baryVW.y;
    let u = 1.0 - v - w;
    let ta = meshTangents[tri.x];
    let tb = meshTangents[tri.y];
    let tc = meshTangents[tri.z];
    var tangent = ta.xyz * u + tb.xyz * v + tc.xyz * w;
    let handednessRaw = ta.w * u + tb.w * v + tc.w * w;
    if (length(tangent) > 1e-8 && abs(handednessRaw) > 0.5) {
      var instanceHandedness = 1.0;
      if (instanceIndex != INVALID_TLAS_INSTANCE_INDEX && params.tlasNodeCount != 0u) {
        let m = instanceIndex * 4u;
        if (m + 3u < arrayLength(&tlasInstanceLocalToWorld)) {
          let l2w0 = tlasInstanceLocalToWorld[m];
          let l2w1 = tlasInstanceLocalToWorld[m + 1u];
          let l2w2 = tlasInstanceLocalToWorld[m + 2u];
          tangent = transformDirectionCols(l2w0, l2w1, l2w2, tangent);
          instanceHandedness =
            transformLinearOrientationSign(l2w0, l2w1, l2w2);
        }
      }
      tangent = tangent - normal * dot(normal, tangent);
      let tlen = length(tangent);
      if (tlen > 1e-8) {
        tangent = tangent / tlen;
        let handedness =
          select(-1.0, 1.0, handednessRaw >= 0.0) * instanceHandedness;
        frame.tangent = tangent;
        frame.bitangent = cross(normal, tangent) * handedness;
        frame.valid = true;
        return frame;
      }
    }
  }
  let p0 = positions[tri.x].xyz;
  let e1 = positions[tri.y].xyz - p0;
  let e2 = positions[tri.z].xyz - p0;
  let uv0 = materialUvForVertex(tri.x, gpuUvSlot);
  let duv1 = materialUvForVertex(tri.y, gpuUvSlot) - uv0;
  let duv2 = materialUvForVertex(tri.z, gpuUvSlot) - uv0;
  let uvScale = max(
    max(abs(duv1.x), abs(duv1.y)),
    max(abs(duv2.x), abs(duv2.y)),
  );
  if (!(uvScale > 0.0) || uvScale > 3.402823e38) { return frame; }
  let normalizedDuv1 = duv1 / uvScale;
  let normalizedDuv2 = duv2 / uvScale;
  let det =
    normalizedDuv1.x * normalizedDuv2.y -
    normalizedDuv2.x * normalizedDuv1.y;
  if (!(abs(det) > 1e-7)) { return frame; }
  let f = 1.0 / det;
  var tangent =
    f * (normalizedDuv2.y * e1 - normalizedDuv1.y * e2);
  var bitangent =
    f * (-normalizedDuv2.x * e1 + normalizedDuv1.x * e2);
  if (instanceIndex != INVALID_TLAS_INSTANCE_INDEX && params.tlasNodeCount != 0u) {
    let m = instanceIndex * 4u;
    if (m + 3u < arrayLength(&tlasInstanceLocalToWorld)) {
      let l2w0 = tlasInstanceLocalToWorld[m];
      let l2w1 = tlasInstanceLocalToWorld[m + 1u];
      let l2w2 = tlasInstanceLocalToWorld[m + 2u];
      tangent = transformDirectionCols(l2w0, l2w1, l2w2, tangent);
      bitangent = transformDirectionCols(l2w0, l2w1, l2w2, bitangent);
    }
  }
  tangent = tangent - normal * dot(normal, tangent);
  let tlen = length(tangent);
  if (tlen < 1e-8) { return frame; }
  tangent = tangent / tlen;
  bitangent = bitangent - normal * dot(normal, bitangent);
  let handedness = select(
    -1.0, 1.0, dot(cross(normal, tangent), bitangent) >= 0.0,
  );
  frame.tangent = tangent;
  frame.bitangent = cross(normal, tangent) * handedness;
  frame.valid = true;
  return frame;
}

fn materialTextureApplyTangentNormal(
  sample: MaterialTextureSample,
  normalScale: f32,
  frame: ShadingTangentFrame,
  fallbackNormal: vec3f,
) -> vec3f {
  if (
    !sample.valid ||
    !materialTextureFiniteF32(normalScale) ||
    !materialTextureFiniteVec3(frame.tangent) ||
    !materialTextureFiniteVec3(frame.bitangent) ||
    !materialTextureFiniteVec3(fallbackNormal)
  ) {
    return fallbackNormal;
  }
  let tangentNormal = sample.value.xyz * 2.0 - vec3f(1.0);
  let coefficients = vec3f(
    tangentNormal.x * normalScale,
    tangentNormal.y * normalScale,
    tangentNormal.z,
  );
  if (!materialTextureFiniteVec3(coefficients)) { return fallbackNormal; }
  let coefficientScale = max(
    max(abs(coefficients.x), abs(coefficients.y)),
    abs(coefficients.z),
  );
  if (!(coefficientScale > 1e-20)) { return fallbackNormal; }
  let scaled = coefficients / coefficientScale;
  let perturbed =
    frame.tangent * scaled.x +
    frame.bitangent * scaled.y +
    fallbackNormal * scaled.z;
  if (!materialTextureFiniteVec3(perturbed)) { return fallbackNormal; }
  let perturbedScale = max(
    max(abs(perturbed.x), abs(perturbed.y)),
    abs(perturbed.z),
  );
  if (!(perturbedScale > 1e-20)) { return fallbackNormal; }
  let safePerturbed = perturbed / perturbedScale;
  let safeLength = length(safePerturbed);
  if (!(safeLength > 1e-6) || !materialTextureFiniteF32(safeLength)) {
    return fallbackNormal;
  }
  return safePerturbed / safeLength;
}

// Reconstruct the unperturbed, face-oriented smooth interface normal from the
// same triangle payload used by intersection. BDPT stores the mapped entry
// normal, so its virtual exit interface uses this helper to resolve a distinct
// opposite-side layer normal from the original frame instead of from -entry.
fn materialBaseInterfaceNormal(
  triIndex: u32,
  baryVW: vec2f,
  instanceIndex: u32,
  isFrontFace: bool,
  fallback: vec3f,
) -> vec3f {
  if (
    triIndex >= arrayLength(&indices) ||
    !materialTextureFiniteVec2(baryVW)
  ) {
    return fallback;
  }
  let tri = indices[triIndex];
  if (
    tri.x >= arrayLength(&normals) ||
    tri.y >= arrayLength(&normals) ||
    tri.z >= arrayLength(&normals)
  ) {
    return fallback;
  }
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  var worldNormal = materialTextureNormalizeOr(
    normals[tri.x].xyz * u + normals[tri.y].xyz * v + normals[tri.z].xyz * w,
    fallback,
  );
  if (
    instanceIndex != INVALID_TLAS_INSTANCE_INDEX &&
    params.tlasNodeCount != 0u
  ) {
    if (instanceIndex > 0xffffffffu / 4u) { return fallback; }
    let matrixBase = instanceIndex * 4u;
    if (
      matrixBase > arrayLength(&tlasInstanceWorldToLocal) ||
      3u > arrayLength(&tlasInstanceWorldToLocal) - matrixBase
    ) {
      return fallback;
    }
    worldNormal = transformNormalFromWorldToLocalCols(
      tlasInstanceWorldToLocal[matrixBase],
      tlasInstanceWorldToLocal[matrixBase + 1u],
      tlasInstanceWorldToLocal[matrixBase + 2u],
      worldNormal,
    );
  }
  let oriented = select(-worldNormal, worldNormal, isFrontFace);
  return materialTextureNormalizeOr(oriented, fallback);
}

// Normal map (linear array) — descriptor vec4[0].y. Perturbs the geometric shading
// normal by the tangent-space normal map. The tangent frame is DERIVED per-hit
// from the triangle's positions + UVs (Lengyel) — no precomputed tangents needed
// — then transformed through the hit TLAS instance and Gram-Schmidt-
// orthonormalized against geomNormal. Returns geomNormal unchanged when there's
// no normal map (→ byte-identical). normalScale follows glTF normalTexture.scale:
// scale tangent-space xy before combining with the derived frame, leaving z as
// authored. Merged-BLAS / lite / analytic paths pass the invalid instance
// sentinel and keep the historical local-space tangent.
// Ref: Lengyel, "Computing Tangent Space Basis Vectors for an Arbitrary Mesh".
fn applyNormalMap(matId: u32, triIndex: u32, baryVW: vec2f, geomNormal: vec3f, instanceIndex: u32, isFrontFace: bool) -> vec3f {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) {
    return geomNormal;
  }
  let linearLayerCount = textureNumLayers(materialTexturesLinear);
  var normalIdx = materialTextureLayerIndex(
    materialTexDescriptors[base].y,
    linearLayerCount,
  );
  if (triIndex >= arrayLength(&indices)) { return geomNormal; }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs) ||
      tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return geomNormal;
  }
  var normalUvMetaOffset: u32 = MATERIAL_TEX_UV_NORMAL;
  var normalUvFitScale = materialTexDescriptors[base + 8u].xy;
  var normalWrapMode = materialTexDescriptors[base + 14u].xy;
  var normalMipPolicySlot = MATERIAL_TEX_MIP_NORMAL;
  var normalScale = materialTexDescriptors[base + 5u].w;
  let layerNormal = materialTexDescriptors[base + MATERIAL_TEX_LAYER_NORMAL];
  let layerIdx = select(
    materialTextureLayerIndex(layerNormal.z, linearLayerCount),
    materialTextureLayerIndex(layerNormal.x, linearLayerCount),
    isFrontFace,
  );
  if (layerIdx >= 0) {
    normalIdx = layerIdx;
    normalScale = select(layerNormal.w, layerNormal.y, isFrontFace);
    normalUvMetaOffset = select(MATERIAL_TEX_UV_BACK_LAYER_NORMAL, MATERIAL_TEX_UV_FRONT_LAYER_NORMAL, isFrontFace);
    normalMipPolicySlot = select(MATERIAL_TEX_MIP_BACK_LAYER_NORMAL, MATERIAL_TEX_MIP_FRONT_LAYER_NORMAL, isFrontFace);
    let layerUvFit = materialTexDescriptors[base + MATERIAL_TEX_LAYER_NORMAL_UV_FIT];
    normalUvFitScale = select(layerUvFit.zw, layerUvFit.xy, isFrontFace);
    let layerWrap = materialTexDescriptors[base + MATERIAL_TEX_LAYER_NORMAL_WRAP];
    normalWrapMode = select(layerWrap.zw, layerWrap.xy, isFrontFace);
  }
  if (normalIdx < 0) { return geomNormal; }
  let vertexCount = arrayLength(&positions);
  if (vertexCount == 0u) { return geomNormal; }
  let normalGpuUvSlot = materialTextureExactU32(
    materialTexDescriptors[base + normalUvMetaOffset].x,
    arrayLength(&meshUvs) / vertexCount,
  );
  if (normalGpuUvSlot == 0xffffffffu) { return geomNormal; }
  let frame = buildShadingTangentFrame(
    triIndex, baryVW, geomNormal, normalGpuUvSlot, instanceIndex,
  );
  if (!frame.valid) { return geomNormal; }
  let sample = sampleMaterialLayerLinear(normalIdx, base, triIndex, baryVW, instanceIndex, normalUvMetaOffset, normalUvFitScale, normalWrapMode, normalMipPolicySlot);
  return materialTextureApplyTangentNormal(sample, normalScale, frame, geomNormal);
}

// KHR_materials_clearcoat clearcoatNormalTexture. This is a LINEAR tangent-space
// normal map with its own scale/UV/wrap metadata. It returns the caller-supplied
// normal unchanged when absent, so zero-map scenes keep the historical clearcoat
// lobe normal exactly.
fn applyClearcoatNormalMap(matId: u32, triIndex: u32, baryVW: vec2f, clearcoatNormal: vec3f, instanceIndex: u32) -> vec3f {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) {
    return clearcoatNormal;
  }
  let clearcoatNormalIdx = materialTextureLayerIndex(
    materialTexDescriptors[base + MATERIAL_TEX_CLEARCOAT_NORMAL].x,
    textureNumLayers(materialTexturesLinear),
  );
  if (clearcoatNormalIdx < 0 || triIndex >= arrayLength(&indices)) { return clearcoatNormal; }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs) ||
      tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return clearcoatNormal;
  }
  let vertexCount = arrayLength(&positions);
  if (vertexCount == 0u) { return clearcoatNormal; }
  let clearcoatNormalGpuUvSlot = materialTextureExactU32(
    materialTexDescriptors[base + MATERIAL_TEX_UV_CLEARCOAT_NORMAL].x,
    arrayLength(&meshUvs) / vertexCount,
  );
  if (clearcoatNormalGpuUvSlot == 0xffffffffu) { return clearcoatNormal; }
  let frame = buildShadingTangentFrame(
    triIndex, baryVW, clearcoatNormal, clearcoatNormalGpuUvSlot, instanceIndex,
  );
  if (!frame.valid) { return clearcoatNormal; }
  let sample = sampleMaterialLayerLinear(
    clearcoatNormalIdx,
    base,
    triIndex,
    baryVW,
    instanceIndex,
    MATERIAL_TEX_UV_CLEARCOAT_NORMAL,
    materialTexDescriptors[base + MATERIAL_TEX_CLEARCOAT_NORMAL].zw,
    materialTexDescriptors[base + MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP].xy,
    MATERIAL_TEX_MIP_CLEARCOAT_NORMAL,
  );
  let clearcoatNormalScale = materialTexDescriptors[base + MATERIAL_TEX_CLEARCOAT_NORMAL].y;
  return materialTextureApplyTangentNormal(
    sample,
    clearcoatNormalScale,
    frame,
    clearcoatNormal,
  );
}

// ── D3 — reserved-field consumption (aoMap / lightMap / bumpMap / envMapIntensity
//        / anisotropy). All gated so a material lacking the field is a no-op. ──

// AO map (LINEAR array) — descriptor vec4[3].y; intensity vec4[4].x.
// Returns the baked occlusion factor ∈ [0,1] lerped by aoMapIntensity:
//   ao = mix(1, sampledR, intensity).  Returns 1 (no occlusion) when absent →
// byte-identical. SEMANTICS (documented, biased): a baked AO map encodes the
// fraction of the hemisphere occluded by *nearby* geometry that the path tracer
// does NOT cheaply re-derive at the primary hit. The honest PT interpretation is
// that AO double-counts occlusion the global solve already integrates, so we apply
// it ONLY as a multiplier on baseColor (the standard glTF occlusionTexture
// semantics, R channel). This darkens cavities consistently with the artist
// intent at the cost of slight energy loss vs ground-truth GI. Hosts wanting
// unbiased transport should omit aoMap. Ref: glTF 2.0 occlusionTexture.
fn sampleAoFactor(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, 16u)) { return 1.0; }
  let aoIdx = materialTextureLayerIndex(
    materialTexDescriptors[base + 3u].y,
    textureNumLayers(materialTexturesLinear),
  );
  if (aoIdx < 0) { return 1.0; }
  let intensityRaw = materialTexDescriptors[base + 4u].x;
  if (!materialTextureFiniteF32(intensityRaw)) { return 1.0; }
  let intensity = clamp(intensityRaw, 0.0, 1.0);
  let sample = sampleMaterialLayerLinear(aoIdx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_AO, materialTexDescriptors[base + 9u].zw, materialTexDescriptors[base + 15u].zw, MATERIAL_TEX_MIP_AO);
  let r = materialTextureValueOr(sample, vec4f(1.0)).r;
  return clamp(mix(1.0, r, intensity), 0.0, 1.0);
}

// Light map (dedicated linear-float radiance array) — descriptor vec4[3].z;
// intensity vec4[4].y.
// Baked OUTGOING radiance added to the surface emission. SEMANTICS: a light map
// is precomputed *outgoing* radiance, so it is added to \`emissive\` at
// camera-visible (emissive-on-hit) shade points ONLY. Adding it inside NEE would
// double-count the real lights it bakes; the path-tracer's own NEE/indirect
// terms already integrate live light. Returns 0 (no addition) when absent →
// byte-identical. The map is treated as linear data (it is radiance, not albedo);
// hosts that authored an sRGB-encoded light map should decode before upload.
// Ref: glTF lightmap convention; THREE.MeshStandardMaterial.lightMap (additive).
fn sampleLightMapRadiance(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> vec3f {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, 17u)) { return vec3f(0.0); }
  let lmIdx = materialTextureLayerIndex(
    materialTexDescriptors[base + 3u].z,
    textureNumLayers(materialTexturesEmissive),
  );
  if (lmIdx < 0) { return vec3f(0.0); }
  let intensity = materialTexDescriptors[base + 4u].y;
  if (!materialTextureFiniteF32(intensity) || intensity < 0.0) { return vec3f(0.0); }
  let sample = sampleMaterialLayerEmissive(lmIdx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_LIGHT, materialTexDescriptors[base + 10u].xy, materialTexDescriptors[base + 16u].xy, MATERIAL_TEX_MIP_LIGHT);
  if (!sample.valid) { return vec3f(0.0); }
  let texel = sample.value.rgb;
  return ptFiniteNonNegativeRadianceProduct(texel, vec3f(intensity));
}

// Per-material environment-map intensity scale — descriptor vec4[4].w (default 1).
fn materialEnvMapIntensity(matId: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 4u, 1u)) { return 1.0; }
  let intensity = materialTexDescriptors[base + 4u].w;
  if (!materialTextureFiniteF32(intensity)) { return 1.0; }
  return max(materialTexDescriptors[base + 4u].w, 0.0);
}

// Anisotropy strength ∈ [0,1] (descriptor vec4[5].x), optionally modulated by the
// KHR_materials_anisotropy map's B channel. 0 ⇒ isotropic (default) ⇒ the caller
// keeps the existing isotropic GGX path → byte-identical.
fn materialAnisotropy(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, 18u)) { return 0.0; }
  let authored = materialTexDescriptors[base + 5u].x;
  if (!materialTextureFiniteF32(authored)) { return 0.0; }
  var a = clamp(authored, 0.0, 1.0);
  let anisoIdx = materialTextureLayerIndex(
    materialTexDescriptors[base + 5u].z,
    textureNumLayers(materialTexturesLinear),
  );
  if (anisoIdx >= 0) {
    let sample = sampleMaterialLayerLinear(anisoIdx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_ANISOTROPY, materialTexDescriptors[base + 11u].xy, materialTexDescriptors[base + 17u].xy, MATERIAL_TEX_MIP_ANISOTROPY);
    a = a * materialTextureValueOr(sample, vec4f(1.0)).b;
  }
  return clamp(a, 0.0, 1.0);
}

// Anisotropy rotation in radians (descriptor vec4[5].y), optionally offset by the
// anisotropy map's RG direction (KHR_materials_anisotropy: RG encodes a 2D tangent
// rotation as cos/sin in [0,1]→[-1,1]). The returned angle is expressed in the
// BSDF's deterministic ONB, but its direction is authored in the selected UV
// tangent frame (including mirrored tangent handedness).
fn materialAnisotropyRotation(matId: u32, triIndex: u32, baryVW: vec2f, shadingNormal: vec3f, instanceIndex: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, 18u)) { return 0.0; }
  var rot = materialTexDescriptors[base + 5u].y;
  if (!materialTextureFiniteF32(rot)) { rot = 0.0; }
  let anisoIdx = materialTextureLayerIndex(
    materialTexDescriptors[base + 5u].z,
    textureNumLayers(materialTexturesLinear),
  );
  var gpuUvSlot = 0u;
  if (anisoIdx >= 0) {
    let vertexCount = arrayLength(&positions);
    if (vertexCount == 0u) { return rot; }
    gpuUvSlot = materialTextureExactU32(
      materialTexDescriptors[base + MATERIAL_TEX_UV_ANISOTROPY].x,
      arrayLength(&meshUvs) / vertexCount,
    );
    if (gpuUvSlot == 0xffffffffu) { return rot; }
    let sample = sampleMaterialLayerLinear(anisoIdx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_ANISOTROPY, materialTexDescriptors[base + 11u].xy, materialTexDescriptors[base + 17u].xy, MATERIAL_TEX_MIP_ANISOTROPY);
    if (sample.valid) {
      let rg = sample.value.rg * 2.0 - vec2f(1.0);
      let rgScale = max(abs(rg.x), abs(rg.y));
      if (rgScale > 1e-20 && materialTextureFiniteF32(rgScale)) {
        let safeRg = rg / rgScale;
        rot = rot + atan2(safeRg.y, safeRg.x);
      }
    }
  }
  if (!materialTextureFiniteF32(rot)) { return 0.0; }
  if (triIndex >= arrayLength(&indices)) { return rot; }
  let frame = buildShadingTangentFrame(
    triIndex, baryVW, shadingNormal, gpuUvSlot, instanceIndex,
  );
  if (
    !frame.valid ||
    !materialTextureFiniteVec3(shadingNormal) ||
    !materialTextureFiniteVec3(frame.tangent) ||
    !materialTextureFiniteVec3(frame.bitangent)
  ) { return rot; }
  var canonicalT: vec3f;
  var canonicalB: vec3f;
  buildOnb(shadingNormal, &canonicalT, &canonicalB);
  let authoredDirection =
    cos(rot) * frame.tangent + sin(rot) * frame.bitangent;
  if (
    !materialTextureFiniteVec3(authoredDirection) ||
    !materialTextureFiniteVec3(canonicalT) ||
    !materialTextureFiniteVec3(canonicalB)
  ) { return rot; }
  let resolved = atan2(
    dot(authoredDirection, canonicalB),
    dot(authoredDirection, canonicalT),
  );
  return select(rot, resolved, materialTextureFiniteF32(resolved));
}

// Bump map (LINEAR height field) — descriptor vec4[3].w; scale vec4[4].z.
// Perturbs the shading normal by the gradient of the height field in UV space,
// finite-differenced from the texture. Mirrors applyNormalMap's tangent-frame
// derivation + TLAS-instance transform + Gram-Schmidt; combines additively with a
// normal map when both are present (apply normal map first, bump second). Returns
// the input normal unchanged when there is no bump map → byte-identical.
// Ref: Blinn 1978, "Simulation of Wrinkled Surfaces"; height-gradient perturbation.
fn applyBumpMap(matId: u32, triIndex: u32, baryVW: vec2f, shadingNormal: vec3f, instanceIndex: u32) -> vec3f {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) {
    return shadingNormal;
  }
  let bumpIdx = materialTextureLayerIndex(
    materialTexDescriptors[base + 3u].w,
    textureNumLayers(materialTexturesLinear),
  );
  if (bumpIdx < 0 || triIndex >= arrayLength(&indices)) { return shadingNormal; }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs) ||
      tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return shadingNormal;
  }
  let bumpScale = materialTexDescriptors[base + 4u].z;
  if (!materialTextureFiniteF32(bumpScale)) { return shadingNormal; }
  let uvMeta = materialTexDescriptors[base + MATERIAL_TEX_UV_BUMP];
  let vertexCount = arrayLength(&positions);
  if (vertexCount == 0u) { return shadingNormal; }
  let bumpGpuUvSlot = materialTextureExactU32(
    uvMeta.x,
    arrayLength(&meshUvs) / vertexCount,
  );
  if (bumpGpuUvSlot == 0xffffffffu) { return shadingNormal; }
  // Build the same world-space tangent frame applyNormalMap uses (D9.3 shared helper).
  let frame = buildShadingTangentFrame(
    triIndex, baryVW, shadingNormal, bumpGpuUvSlot, instanceIndex,
  );
  if (!frame.valid) { return shadingNormal; }
  let tangent = frame.tangent;
  let bitangent = frame.bitangent;
  // Finite-difference the height (R channel) in raw UV space by one uploaded
  // source texel; the height-gradient slopes the normal by -scale·(dh/du, dh/dv).
  let bumpUvFitScale = materialTexDescriptors[base + 10u].zw;
  let bumpWrapMode = materialTexDescriptors[base + 16u].zw;
  if (
    !materialTextureFiniteVec2(bumpUvFitScale) ||
    any(bumpUvFitScale <= vec2f(0.0)) ||
    any(bumpUvFitScale > vec2f(1.0)) ||
    !materialTextureWrapModesValid(bumpWrapMode)
  ) { return shadingNormal; }
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let rawUv =
    materialUvForVertex(tri.x, bumpGpuUvSlot) * u +
    materialUvForVertex(tri.y, bumpGpuUvSlot) * v +
    materialUvForVertex(tri.z, bumpGpuUvSlot) * w;
  let linearDims = vec2f(textureDimensions(materialTexturesLinear, i32(0)));
  let sourceDims = max(linearDims * bumpUvFitScale, vec2f(1.0));
  let texelStep = vec2f(1.0 / sourceDims.x, 1.0 / sourceDims.y);
  if (!materialTextureFiniteVec2(texelStep)) { return shadingNormal; }
  let sampleC = sampleMaterialLayerLinearRawUvPolicy(bumpIdx, base, triIndex, baryVW, instanceIndex, rawUv, MATERIAL_TEX_UV_BUMP, bumpUvFitScale, bumpWrapMode, MATERIAL_TEX_MIP_BUMP);
  let sampleU = sampleMaterialLayerLinearRawUvPolicy(bumpIdx, base, triIndex, baryVW, instanceIndex, rawUv + vec2f(texelStep.x, 0.0), MATERIAL_TEX_UV_BUMP, bumpUvFitScale, bumpWrapMode, MATERIAL_TEX_MIP_BUMP);
  let sampleV = sampleMaterialLayerLinearRawUvPolicy(bumpIdx, base, triIndex, baryVW, instanceIndex, rawUv + vec2f(0.0, texelStep.y), MATERIAL_TEX_UV_BUMP, bumpUvFitScale, bumpWrapMode, MATERIAL_TEX_MIP_BUMP);
  if (!(sampleC.valid && sampleU.valid && sampleV.valid)) { return shadingNormal; }
  let hC = sampleC.value.r;
  let hU = sampleU.value.r;
  let hV = sampleV.value.r;
  let dhdu = (hU - hC) / texelStep.x;
  let dhdv = (hV - hC) / texelStep.y;
  let scaledGradient = vec2f(dhdu, dhdv) * bumpScale;
  if (!materialTextureFiniteVec2(scaledGradient)) { return shadingNormal; }
  let perturbed =
    shadingNormal - scaledGradient.x * tangent - scaledGradient.y * bitangent;
  return materialTextureNormalizeOr(perturbed, shadingNormal);
}

// Standalone alpha map (LINEAR coverage data) — descriptor vec4[6].x.
// Multiplies the baseColor texture alpha and material opacity in alphaMode
// mask/blend. Returns 1 when absent, so legacy alpha behavior is unchanged.
fn sampleAlphaTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, 18u)) { return 1.0; }
  let alphaIdx = materialTextureLayerIndex(
    materialTexDescriptors[base + 6u].x,
    textureNumLayers(materialTexturesLinear),
  );
  if (alphaIdx < 0) { return 1.0; }
  let sample = sampleMaterialLayerLinear(alphaIdx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_ALPHA, materialTexDescriptors[base + 11u].zw, materialTexDescriptors[base + 17u].zw, MATERIAL_TEX_MIP_ALPHA);
  return clamp(materialTextureValueOr(sample, vec4f(1.0)).r, 0.0, 1.0);
}

// Transmission map (LINEAR scalar data) — descriptor vec4[6].y.
// Multiplies MaterialSpec.transmission using glTF KHR_materials_transmission's
// R channel. Returns 1 when absent, so scalar-only transmission is unchanged.
fn sampleTransmissionTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, 19u)) { return 1.0; }
  let transmissionIdx = materialTextureLayerIndex(
    materialTexDescriptors[base + 6u].y,
    textureNumLayers(materialTexturesLinear),
  );
  if (transmissionIdx < 0) { return 1.0; }
  let sample = sampleMaterialLayerLinear(transmissionIdx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_TRANSMISSION, materialTexDescriptors[base + 12u].xy, materialTexDescriptors[base + 18u].xy, MATERIAL_TEX_MIP_TRANSMISSION);
  return clamp(materialTextureValueOr(sample, vec4f(1.0)).r, 0.0, 1.0);
}

// KHR_materials_volume thicknessTexture (LINEAR scalar data) — descriptor
// vec4[71].x, sampled from G per glTF. Returns -1 when absent so callers can
// distinguish "no map" from a legitimate zero-thickness texel.
fn sampleVolumeThicknessTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) {
    return -1.0;
  }
  let thicknessIdx = materialTextureLayerIndex(
    materialTexDescriptors[base + MATERIAL_TEX_THICKNESS].x,
    textureNumLayers(materialTexturesLinear),
  );
  if (thicknessIdx < 0) { return -1.0; }
  let sample = sampleMaterialLayerLinear(thicknessIdx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_THICKNESS, materialTexDescriptors[base + MATERIAL_TEX_THICKNESS].yz, materialTexDescriptors[base + MATERIAL_TEX_THICKNESS_WRAP].xy, MATERIAL_TEX_MIP_THICKNESS);
  if (!sample.valid) { return -1.0; }
  return clamp(sample.value.g, 0.0, 1.0);
}

// Decode the material at a concrete represented boundary and apply the
// KHR_materials_volume thickness map exactly as an ordinary surface hit does.
// Containment reconstruction uses the unmatched exit hit, so retaining its
// triangle, barycentrics, and instance is required for start-inside parity.
fn materialAtOpticalBoundary(
  matId: u32,
  triIndex: u32,
  baryVW: vec2f,
  instanceIndex: u32,
) -> DecodedMaterial {
  var mat = decodeMaterial(matId);
  let thicknessSample = sampleVolumeThicknessTexture(
    matId, triIndex, baryVW, instanceIndex,
  );
  if (thicknessSample >= 0.0 && mat.hasVolumeThickness) {
    mat.volumeThickness = max(mat.volumeThickness * thicknessSample, 0.0);
  }
  return mat;
}

// Extension-lobe map samplers. Mirrors pt-webgl2/glTF channel conventions:
// clearcoat R, clearcoatRoughness G, sheenColor RGB, sheenRoughness A,
// iridescence R, iridescenceThickness G, specularColor RGB, specularIntensity A.
fn sampleClearcoatTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) { return 1.0; }
  let idx = materialTextureLayerIndex(materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_INDEX].x, textureNumLayers(materialTexturesLinear));
  if (idx < 0) { return 1.0; }
  let sample = sampleMaterialLayerLinear(idx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_CLEARCOAT, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_UV_FIT].xy, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_WRAP].xy, MATERIAL_TEX_MIP_CLEARCOAT);
  return clamp(materialTextureValueOr(sample, vec4f(1.0)).r, 0.0, 1.0);
}

fn sampleClearcoatRoughnessTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) { return 1.0; }
  let idx = materialTextureLayerIndex(materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_INDEX].y, textureNumLayers(materialTexturesLinear));
  if (idx < 0) { return 1.0; }
  let sample = sampleMaterialLayerLinear(idx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_CLEARCOAT_ROUGHNESS, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_UV_FIT].zw, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_WRAP].zw, MATERIAL_TEX_MIP_CLEARCOAT_ROUGHNESS);
  return clamp(materialTextureValueOr(sample, vec4f(1.0)).g, 0.0, 1.0);
}

fn sampleSheenColorTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> vec3f {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) { return vec3f(1.0); }
  let idx = materialTextureLayerIndex(materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_INDEX].z, textureNumLayers(materialTextures));
  if (idx < 0) { return vec3f(1.0); }
  let sample = sampleMaterialLayer(idx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_SHEEN_COLOR, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_UV_FIT + 1u].xy, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_WRAP + 1u].xy, MATERIAL_TEX_MIP_SHEEN_COLOR);
  return clamp(materialTextureValueOr(sample, vec4f(1.0)).rgb, vec3f(0.0), vec3f(1.0));
}

fn sampleSheenRoughnessTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) { return 1.0; }
  let idx = materialTextureLayerIndex(materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_INDEX].w, textureNumLayers(materialTexturesLinear));
  if (idx < 0) { return 1.0; }
  let sample = sampleMaterialLayerLinear(idx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_SHEEN_ROUGHNESS, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_UV_FIT + 1u].zw, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_WRAP + 1u].zw, MATERIAL_TEX_MIP_SHEEN_ROUGHNESS);
  return clamp(materialTextureValueOr(sample, vec4f(1.0)).a, 0.0, 1.0);
}

fn sampleIridescenceTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) { return 1.0; }
  let idx = materialTextureLayerIndex(materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_INDEX + 1u].x, textureNumLayers(materialTexturesLinear));
  if (idx < 0) { return 1.0; }
  let sample = sampleMaterialLayerLinear(idx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_IRIDESCENCE, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_UV_FIT + 2u].xy, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_WRAP + 2u].xy, MATERIAL_TEX_MIP_IRIDESCENCE);
  return clamp(materialTextureValueOr(sample, vec4f(1.0)).r, 0.0, 1.0);
}

fn sampleIridescenceThicknessTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) { return -1.0; }
  let idx = materialTextureLayerIndex(materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_INDEX + 1u].y, textureNumLayers(materialTexturesLinear));
  if (idx < 0) { return -1.0; }
  let sample = sampleMaterialLayerLinear(idx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_IRIDESCENCE_THICKNESS, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_UV_FIT + 2u].zw, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_WRAP + 2u].zw, MATERIAL_TEX_MIP_IRIDESCENCE_THICKNESS);
  if (!sample.valid) { return -1.0; }
  return clamp(sample.value.g, 0.0, 1.0);
}

fn sampleSpecularColorTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> vec3f {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) { return vec3f(1.0); }
  let idx = materialTextureLayerIndex(materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_INDEX + 1u].z, textureNumLayers(materialTextures));
  if (idx < 0) { return vec3f(1.0); }
  let sample = sampleMaterialLayer(idx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_SPECULAR_COLOR, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_UV_FIT + 3u].xy, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_WRAP + 3u].xy, MATERIAL_TEX_MIP_SPECULAR_COLOR);
  return clamp(materialTextureValueOr(sample, vec4f(1.0)).rgb, vec3f(0.0), vec3f(1.0));
}

fn sampleSpecularIntensityTexture(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) { return 1.0; }
  let idx = materialTextureLayerIndex(materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_INDEX + 1u].w, textureNumLayers(materialTexturesLinear));
  if (idx < 0) { return 1.0; }
  let sample = sampleMaterialLayerLinear(idx, base, triIndex, baryVW, instanceIndex, MATERIAL_TEX_UV_SPECULAR_INTENSITY, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_UV_FIT + 3u].zw, materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_WRAP + 3u].zw, MATERIAL_TEX_MIP_SPECULAR_INTENSITY);
  return clamp(materialTextureValueOr(sample, vec4f(1.0)).a, 0.0, 1.0);
}

// P2 alpha test — should this hit be treated as TRANSPARENT (the ray passes
// straight through, as if there were no surface here)? Drives glTF alphaMode:
//   opaque (0) → never (returns false immediately → opaque is byte-identical).
//   mask   (1) → pass through where baseColorTexAlpha·alphaMap·opacity < alphaCutoff
//                (hard cutout — foliage, fences, decals).
//   blend  (2) → STOCHASTIC pass-through with probability 1 − alpha·opacity
//                (unbiased screen-door transparency in a path tracer; the
//                converged mean equals true alpha compositing).
// The base-color texture's .a and standalone alphaMap supply per-texel alpha
// (both 1 when absent, so an untextured material with material.opacity<1 still
// blends/cuts by opacity).
// Ref: glTF 2.0 §3.9.4 (alphaMode); PBR screen-door / stochastic transparency.
fn alphaTestPassThrough(matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32, rng: ptr<function, PtRngState>) -> bool {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, 4u)) { return false; }
  let alphaMode = materialTextureExactU32(materialTexDescriptors[base + 1u].x, 3u);
  if (alphaMode == 0xffffffffu) { return false; }
  if (alphaMode == 0u) { return false; } // opaque — byte-identical
  let alphaCutoff = materialTexDescriptors[base + 1u].y;
  let opacity = materialTexDescriptors[base + 1u].z;
  if (!materialTextureFiniteF32(alphaCutoff) || !materialTextureFiniteF32(opacity)) {
    return false;
  }
  let alpha = sampleBaseColorTexture(matId, triIndex, baryVW, instanceIndex).a *
    sampleVertexColor(triIndex, baryVW).a *
    sampleAlphaTexture(matId, triIndex, baryVW, instanceIndex) *
    opacity;
  if (!materialTextureFiniteF32(alpha)) { return false; }
  if (alphaMode == 1u) { return alpha < alphaCutoff; }   // mask
  let representedOpacity = represented_bernoulli_probability_f32(
    clamp(alpha, 0.0, 1.0),
  );
  return !(rand_f32(rng) < representedOpacity);           // blend (stochastic)
}
`;

const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_WGSL =
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP0_WGSL +
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP1_WGSL +
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP2_WGSL +
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;

export const PT_WEBGPU_POINT_SPOT_ATTENUATION_WGSL = /* wgsl */ `
// Canonical point/spot attenuation shared by every pt-webgpu transport path.
// decay == 0 is constant intensity; decay == 2 is physical inverse square.
// A finite cutoff uses KHR_lights_punctual's unsquared quartic range window,
// matching the converged WebGL2 backend's emitter contract.
fn pointSpotDistanceAttenuation(
  distance: f32,
  cutoffDistance: f32,
  decay: f32,
) -> f32 {
  if (!(distance > 0.0) || distance > 3.402823466e38) {
    return 0.0;
  }
  var attenuation = 1.0;
  if (decay > 0.01) {
    let denominator = pow(distance, decay);
    if (!(denominator > 0.0)) {
      attenuation = 3.402823466e38;
    } else if (denominator > 3.402823466e38) {
      return 0.0;
    } else {
      attenuation = min(1.0 / denominator, 3.402823466e38);
    }
  }
  if (cutoffDistance > 0.0) {
    let window = clamp(
      1.0 - pow(distance / cutoffDistance, 4.0),
      0.0,
      1.0,
    );
    attenuation = attenuation * window;
  }
  return attenuation;
}

// Direction-sampled point/spot estimators already acquire inverse-square
// spreading through their solid-angle-to-area measure. Multiply by this ratio
// to replace that physical spreading with the authored decay/cutoff law.
fn pointSpotPathMeasureScale(distance: f32, cutoffDistance: f32, decay: f32) -> f32 {
  if (!(distance > 0.0) || distance > 3.402823466e38) {
    return 0.0;
  }
  var window = 1.0;
  if (cutoffDistance > 0.0) {
    window = clamp(
      1.0 - pow(distance / cutoffDistance, 4.0),
      0.0,
      1.0,
    );
  }
  let distancePower = pow(
    distance,
    select(2.0, 2.0 - decay, decay > 0.01),
  );
  return select(
    0.0,
    distancePower * window,
    distancePower >= 0.0 && distancePower <= 3.402823466e38,
  );
}
`;

export const PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL = /* wgsl */ `
const LEAFNODE_FLAG = 0xffff0000u;
// MUST stay in lockstep with TS \`MATERIAL_VEC4_STRIDE\` in scene/materialPacking.ts.
// WS4 bumped 22 → 23: vec4 #22 carries volumetric σ_a.rgb + hasSigmaA flag.
// H52 bumped 23 → 26: vec4s #23–#25 carry clearcoat / sheen / iridescence lobes.
// A3 bumped 26 → 27: vec4 #26 carries the baseColor Jakob-Hanika sigmoid coeffs.
// SPEC-01 bumped 27 → 28: vec4 #27 carries KHR_materials_specular scalar factors.
// VOL-THICKNESS bumped 28 → 29: vec4 #28 carries KHR volume thickness clamp.
const MATERIAL_VEC4_STRIDE = 29u;
const THIN_FILM_RGB_LUT_BINS = 64u;
const THIN_FILM_RGB_LUT_SCALARS_PER_BIN = 16u;
const MATERIAL_SCALAR_STRIDE = MATERIAL_VEC4_STRIDE * 4u;
const THIN_FILM_LAYER_LIMIT = 8u;
const THIN_FILM_SCALAR_BASE = 28u;
const SPECTRAL_SCALAR_BASE = 52u;
const SPECTRAL_SAMPLE_COUNT = 32u;
// MUST stay in lockstep with TS POINT_LIGHT_FLOAT_STRIDE / SPOT_LIGHT_FLOAT_STRIDE
// in scene/emitterPacking.ts (H51-D: 3 vec4 / 4 vec4).  Caustic + photon-map loops
// use these so the stride lives in one place rather than being repeated at five sites.
const POINT_LIGHT_VEC4_STRIDE = 3u;
const SPOT_LIGHT_VEC4_STRIDE = 4u;
const MATERIAL_INVALID_INDEX = 0xffffffffu;
${PT_WEBGPU_POINT_SPOT_ATTENUATION_WGSL}

fn materialCheckedAddU32(a: u32, b: u32) -> u32 {
  if (a > MATERIAL_INVALID_INDEX - b) { return MATERIAL_INVALID_INDEX; }
  return a + b;
}

fn materialCheckedMulU32(a: u32, b: u32) -> u32 {
  if (b != 0u && a > MATERIAL_INVALID_INDEX / b) {
    return MATERIAL_INVALID_INDEX;
  }
  return a * b;
}

fn materialSpanValid(base: u32, count: u32, length: u32) -> bool {
  if (base == MATERIAL_INVALID_INDEX || base > length) { return false; }
  return count <= length - base;
}

fn materialRecordFiniteF32(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

fn materialRecordExactU32(value: f32, upperExclusive: u32) -> u32 {
  if (
    !materialRecordFiniteF32(value) ||
    value < 0.0 ||
    value != floor(value) ||
    value > 16777216.0 ||
    value >= f32(upperExclusive)
  ) {
    return MATERIAL_INVALID_INDEX;
  }
  return u32(value);
}

fn materialRecordBase(matId: u32) -> u32 {
  let base = materialCheckedMulU32(matId, MATERIAL_VEC4_STRIDE);
  if (!materialSpanValid(base, MATERIAL_VEC4_STRIDE, arrayLength(&materials))) {
    return MATERIAL_INVALID_INDEX;
  }
  return base;
}

fn materialRecordIndex(base: u32, offset: u32) -> u32 {
  if (base == MATERIAL_INVALID_INDEX || offset >= MATERIAL_VEC4_STRIDE) {
    return MATERIAL_INVALID_INDEX;
  }
  return materialCheckedAddU32(base, offset);
}

fn materialRecordVec4Or(index: u32, fallback: vec4f) -> vec4f {
  if (index >= arrayLength(&materials)) { return fallback; }
  return materials[index];
}

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
  if (scalarOffset >= MATERIAL_SCALAR_STRIDE) { return 0.0; }
  let recordBase = materialRecordBase(matId);
  if (recordBase == MATERIAL_INVALID_INDEX) { return 0.0; }
  let vecOffset = scalarOffset / 4u;
  let vecIndex = materialRecordIndex(recordBase, vecOffset);
  if (vecIndex == MATERIAL_INVALID_INDEX) { return 0.0; }
  let c = scalarOffset % 4u;
  let v = materials[vecIndex];
  if (c == 0u) { return v.x; }
  if (c == 1u) { return v.y; }
  if (c == 2u) { return v.z; }
  return v.w;
}

fn materialStorageScalar(scalarIndex: u32) -> f32 {
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
  if (!materialRecordFiniteF32(wavelength01)) { return 0.0; }
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
  let d = dot(b, b);
  if (!(d > 1e-20) || d != d || d > 1e30) { return vec2f(0.0); }
  return vec2f(
    (a.x * b.x + a.y * b.y) / d,
    (a.y * b.x - a.x * b.y) / d,
  );
}
fn cSqrtPhysical(z: vec2f) -> vec2f {
  let radius = sqrt(max(dot(z, z), 0.0));
  let re = sqrt(max(0.0, 0.5 * (radius + z.x)));
  let imMagnitude = sqrt(max(0.0, 0.5 * (radius - z.x)));
  return vec2f(re, select(imMagnitude, -imMagnitude, z.y < 0.0));
}
fn cExpI(z: vec2f) -> vec2f {
  let amplitude = exp(clamp(-z.y, -80.0, 0.0));
  return amplitude * vec2f(cos(z.x), sin(z.x));
}
fn tfFinite(v: f32) -> bool {
  return v == v && abs(v) < 1e30;
}

struct ThinFilmScatter {
  rL: vec2f,
  tLR: vec2f,
  rR: vec2f,
  tRL: vec2f,
}
fn thinFilmCascade(a: ThinFilmScatter, b: ThinFilmScatter) -> ThinFilmScatter {
  let inv = cDiv(vec2f(1.0, 0.0), vec2f(1.0, 0.0) - cMul(a.rR, b.rL));
  var out: ThinFilmScatter;
  out.rL = a.rL + cMul(cMul(cMul(a.tRL, b.rL), inv), a.tLR);
  out.tLR = cMul(cMul(b.tLR, inv), a.tLR);
  out.rR = b.rR + cMul(cMul(cMul(b.tLR, a.rR), inv), b.tRL);
  out.tRL = cMul(cMul(a.tRL, inv), b.tRL);
  return out;
}

fn thinFilmLayerN(matId: u32, layerIndex: u32) -> vec2f {
  if (layerIndex >= THIN_FILM_LAYER_LIMIT) { return vec2f(1.0, 0.0); }
  let layerOffset = materialCheckedMulU32(layerIndex, 3u);
  let base = materialCheckedAddU32(THIN_FILM_SCALAR_BASE, layerOffset);
  let extinctionOffset = materialCheckedAddU32(base, 2u);
  if (extinctionOffset == MATERIAL_INVALID_INDEX || extinctionOffset >= MATERIAL_SCALAR_STRIDE) {
    return vec2f(1.0, 0.0);
  }
  return vec2f(
    max(materialScalar(matId, base), 1.0),
    max(materialScalar(matId, extinctionOffset), 0.0),
  );
}
fn thinFilmLayerThicknessNm(matId: u32, layerIndex: u32) -> f32 {
  if (layerIndex >= THIN_FILM_LAYER_LIMIT) { return 0.0; }
  let layerOffset = materialCheckedMulU32(layerIndex, 3u);
  let base = materialCheckedAddU32(THIN_FILM_SCALAR_BASE, layerOffset);
  let thicknessOffset = materialCheckedAddU32(base, 1u);
  if (thicknessOffset == MATERIAL_INVALID_INDEX || thicknessOffset >= MATERIAL_SCALAR_STRIDE) {
    return 0.0;
  }
  return max(materialScalar(
    matId, thicknessOffset,
  ), 0.0);
}
fn thinFilmMediumN(
  matId: u32,
  mediumIndex: u32,
  layerCount: u32,
  incidentIor: f32,
  substrateIor: f32,
  frontFace: bool,
) -> vec2f {
  if (mediumIndex == 0u) {
    return vec2f(select(substrateIor, incidentIor, frontFace), 0.0);
  }
  if (mediumIndex == layerCount + 1u) {
    return vec2f(select(incidentIor, substrateIor, frontFace), 0.0);
  }
  let authoredIndex = select(
    layerCount - mediumIndex,
    mediumIndex - 1u,
    frontFace,
  );
  return thinFilmLayerN(matId, authoredIndex);
}
fn thinFilmMediumThicknessNm(
  matId: u32,
  mediumIndex: u32,
  layerCount: u32,
  frontFace: bool,
) -> f32 {
  let authoredIndex = select(
    layerCount - mediumIndex,
    mediumIndex - 1u,
    frontFace,
  );
  return thinFilmLayerThicknessNm(matId, authoredIndex);
}
fn thinFilmPhysicalCosine(n: vec2f, transverseIndex: f32) -> vec2f {
  let ratio = cDiv(vec2f(transverseIndex, 0.0), n);
  var cosine = cSqrtPhysical(vec2f(1.0, 0.0) - cMul(ratio, ratio));
  let kz = cMul(n, cosine);
  if (kz.y < -1e-7 || (abs(kz.y) <= 1e-7 && kz.x < 0.0)) {
    cosine = -cosine;
  }
  return cosine;
}
fn thinFilmAdmittance(n: vec2f, cosine: vec2f, pPolarized: bool) -> vec2f {
  return select(cMul(n, cosine), cDiv(cosine, n), pPolarized);
}

fn thinFilmPolarizedRt(
  matId: u32,
  layerCount: u32,
  wavelengthNm: f32,
  substrateIor: f32,
  incidentIor: f32,
  angleDependent: bool,
  microfacetCos: f32,
  frontFace: bool,
  pPolarized: bool,
) -> vec2f {
  let count = min(layerCount, THIN_FILM_LAYER_LIMIT);
  let etaIncident = select(substrateIor, incidentIor, frontFace);
  let cos0 = select(
    1.0, clamp(microfacetCos, 0.0, 1.0), angleDependent,
  );
  let transverseIndex =
    etaIncident * sqrt(max(0.0, 1.0 - cos0 * cos0));
  var network: ThinFilmScatter;
  network.rL = vec2f(0.0);
  network.tLR = vec2f(1.0, 0.0);
  network.rR = vec2f(0.0);
  network.tRL = vec2f(1.0, 0.0);

  for (var interfaceIndex = 0u;
       interfaceIndex < THIN_FILM_LAYER_LIMIT + 1u;
       interfaceIndex = interfaceIndex + 1u) {
    if (interfaceIndex > count) { break; }
    let n0 = thinFilmMediumN(
      matId, interfaceIndex, count, incidentIor, substrateIor, frontFace,
    );
    let n1 = thinFilmMediumN(
      matId, interfaceIndex + 1u, count,
      incidentIor, substrateIor, frontFace,
    );
    let cos0Layer = thinFilmPhysicalCosine(n0, transverseIndex);
    let cos1Layer = thinFilmPhysicalCosine(n1, transverseIndex);
    let q0 = thinFilmAdmittance(n0, cos0Layer, pPolarized);
    let q1 = thinFilmAdmittance(n1, cos1Layer, pPolarized);
    let qSum = q0 + q1;
    let rL = cDiv(q0 - q1, qSum);
    var boundary: ThinFilmScatter;
    boundary.rL = rL;
    boundary.tLR = cDiv(2.0 * q0, qSum);
    boundary.rR = -rL;
    boundary.tRL = cDiv(2.0 * q1, qSum);
    network = thinFilmCascade(network, boundary);

    if (interfaceIndex < count) {
      let thicknessNm = thinFilmMediumThicknessNm(
        matId, interfaceIndex + 1u, count, frontFace,
      );
      let phase = cMul(n1, cos1Layer) *
        (2.0 * PI * thicknessNm / max(wavelengthNm, 1e-4));
      let propagationFactor = cExpI(phase);
      var propagation: ThinFilmScatter;
      propagation.rL = vec2f(0.0);
      propagation.tLR = propagationFactor;
      propagation.rR = vec2f(0.0);
      propagation.tRL = propagationFactor;
      network = thinFilmCascade(network, propagation);
    }
  }
  let nIncident = thinFilmMediumN(
    matId, 0u, count, incidentIor, substrateIor, frontFace,
  );
  let nTransmitted = thinFilmMediumN(
    matId, count + 1u, count, incidentIor, substrateIor, frontFace,
  );
  let qIncident = thinFilmAdmittance(
    nIncident, thinFilmPhysicalCosine(nIncident, transverseIndex), pPolarized,
  );
  let qTransmitted = thinFilmAdmittance(
    nTransmitted,
    thinFilmPhysicalCosine(nTransmitted, transverseIndex),
    pPolarized,
  );
  return vec2f(
    dot(network.rL, network.rL),
    max(qTransmitted.x, 0.0) / max(qIncident.x, 1e-8) *
      dot(network.tLR, network.tLR),
  );
}

fn thinFilmTmmRt(
  matId: u32,
  layerCount: u32,
  wavelengthNm: f32,
  substrateIor: f32,
  incidentIor: f32,
  angleDependent: bool,
  microfacetCos: f32,
  frontFace: bool,
) -> vec3f {
  let rtS = thinFilmPolarizedRt(
    matId, layerCount, wavelengthNm, substrateIor, incidentIor,
    angleDependent, microfacetCos, frontFace, false,
  );
  let rtP = thinFilmPolarizedRt(
    matId, layerCount, wavelengthNm, substrateIor, incidentIor,
    angleDependent, microfacetCos, frontFace, true,
  );
  var r = max(0.0, 0.5 * (rtS.x + rtP.x));
  var t = max(0.0, 0.5 * (rtS.y + rtP.y));
  if (!tfFinite(r) || !tfFinite(t) || r + t > 1.0001) {
    // A shader invocation cannot raise the CPU's ThinFilmNumericError. Lose
    // the invalid sample as absorption instead of silently synthesizing a
    // perfect mirror that was never authored.
    return vec3f(0.0, 0.0, 1.0);
  }
  if (r + t > 1.0) {
    let invSum = 1.0 / (r + t);
    r = r * invSum;
    t = t * invSum;
  }
  return vec3f(r, t, max(0.0, 1.0 - r - t));
}

struct ThinFilmTransportRt {
  reflectance: vec3f,
  transmittance: vec3f,
  reflectanceEnergy: f32,
  transmittanceEnergy: f32,
  absorptionEnergy: f32,
}
struct ThinFilmInterface {
  enabled: bool,
  matId: u32,
  layerCount: u32,
  incidentIor: f32,
  substrateIor: f32,
  angleDependent: bool,
  frontFace: bool,
  spectralEnabled: bool,
  heroLambdaNm: f32,
  transmissionScale: f32,
}

// KHR_materials_transmission gates only the coherent transmitted energy. The
// interface reflection remains the TMM result; energy rejected by the scalar
// map joins absorption. Scaling both the colored value and its scalar proposal
// probability keeps T/p exactly unchanged on transmitted Monte Carlo events.
fn thinFilmApplyTransmissionScale(
  raw: ThinFilmTransportRt,
  transmissionScaleRaw: f32,
) -> ThinFilmTransportRt {
  let transmissionScale = clamp(transmissionScaleRaw, 0.0, 1.0);
  var out = raw;
  out.transmittance = raw.transmittance * transmissionScale;
  out.transmittanceEnergy = raw.transmittanceEnergy * transmissionScale;
  out.absorptionEnergy = max(
    0.0, 1.0 - out.reflectanceEnergy - out.transmittanceEnergy,
  );
  return out;
}

// Structural corruption must fail dark. CPU scene publication validates every
// descriptor/LUT range before upload, but a stale or externally corrupted GPU
// buffer must not turn into an unbounded bright perfect-mirror contribution.
fn thinFilmAbsorbedTransportRt() -> ThinFilmTransportRt {
  var out: ThinFilmTransportRt;
  out.reflectance = vec3f(0.0);
  out.transmittance = vec3f(0.0);
  out.reflectanceEnergy = 0.0;
  out.transmittanceEnergy = 0.0;
  out.absorptionEnergy = 1.0;
  return out;
}

fn thinFilmTransportRt(
  film: ThinFilmInterface,
  microfacetCos: f32,
) -> ThinFilmTransportRt {
  var out: ThinFilmTransportRt;
  if (film.spectralEnabled) {
    let rt = thinFilmTmmRt(
      film.matId, film.layerCount, film.heroLambdaNm,
      film.substrateIor, film.incidentIor, film.angleDependent,
      microfacetCos, film.frontFace,
    );
    out.reflectance = vec3f(rt.x);
    out.transmittance = vec3f(rt.y);
    out.reflectanceEnergy = rt.x;
    out.transmittanceEnergy = rt.y;
    out.absorptionEnergy = rt.z;
    return thinFilmApplyTransmissionScale(out, film.transmissionScale);
  }

  let cosTheta = select(1.0, clamp(microfacetCos, 0.0, 1.0), film.angleDependent);
  let etaIncident = select(film.substrateIor, film.incidentIor, film.frontFace);
  let etaTransmitted = select(film.incidentIor, film.substrateIor, film.frontFace);
  let criticalCos = sqrt(max(0.0,
    1.0 - (etaTransmitted * etaTransmitted) /
      max(etaIncident * etaIncident, 1e-8)));
  let hasCritical = etaIncident > etaTransmitted && criticalCos > 1e-6;
  let halfBins = THIN_FILM_RGB_LUT_BINS / 2u;
  var lutPosition: f32;
  if (hasCritical && cosTheta <= criticalCos) {
    lutPosition = sqrt(clamp(cosTheta / criticalCos, 0.0, 1.0)) *
      f32(halfBins - 1u);
  } else if (hasCritical) {
    lutPosition = f32(halfBins) +
      sqrt(clamp((cosTheta - criticalCos) / max(1.0 - criticalCos, 1e-8), 0.0, 1.0)) *
      f32(halfBins - 1u);
  } else {
    lutPosition = sqrt(cosTheta) * f32(THIN_FILM_RGB_LUT_BINS - 1u);
  }
  if (!materialRecordFiniteF32(lutPosition)) {
    return thinFilmAbsorbedTransportRt();
  }
  let bin0 = materialRecordExactU32(floor(lutPosition), THIN_FILM_RGB_LUT_BINS);
  if (bin0 == MATERIAL_INVALID_INDEX) {
    return thinFilmAbsorbedTransportRt();
  }
  let bin1 = min(
    materialCheckedAddU32(bin0, 1u),
    THIN_FILM_RGB_LUT_BINS - 1u,
  );
  let alpha = lutPosition - f32(bin0);
  let directionOffset = select(8u, 0u, film.frontFace);
  let materialBase = materialRecordBase(film.matId);
  let descriptorIndex = materialRecordIndex(materialBase, 28u);
  if (descriptorIndex == MATERIAL_INVALID_INDEX) {
    return thinFilmAbsorbedTransportRt();
  }
  let lutBaseVec4 = materialRecordExactU32(
    materials[descriptorIndex].z,
    arrayLength(&materials),
  );
  let scalarCount = materialCheckedMulU32(arrayLength(&materials), 4u);
  let lutBaseScalar = materialCheckedMulU32(lutBaseVec4, 4u);
  let lutScalarCount = materialCheckedMulU32(
    THIN_FILM_RGB_LUT_BINS,
    THIN_FILM_RGB_LUT_SCALARS_PER_BIN,
  );
  if (
    lutBaseVec4 == MATERIAL_INVALID_INDEX ||
    lutBaseVec4 == 0u ||
    scalarCount == MATERIAL_INVALID_INDEX ||
    lutScalarCount == MATERIAL_INVALID_INDEX ||
    !materialSpanValid(lutBaseScalar, lutScalarCount, scalarCount)
  ) {
    return thinFilmAbsorbedTransportRt();
  }
  let bin0Offset = materialCheckedMulU32(bin0, THIN_FILM_RGB_LUT_SCALARS_PER_BIN);
  let bin1Offset = materialCheckedMulU32(bin1, THIN_FILM_RGB_LUT_SCALARS_PER_BIN);
  let base0 = materialCheckedAddU32(
    lutBaseScalar,
    materialCheckedAddU32(bin0Offset, directionOffset),
  );
  let base1 = materialCheckedAddU32(
    lutBaseScalar,
    materialCheckedAddU32(bin1Offset, directionOffset),
  );
  if (!materialSpanValid(base0, 8u, scalarCount) || !materialSpanValid(base1, 8u, scalarCount)) {
    return thinFilmAbsorbedTransportRt();
  }
  out.reflectance = mix(vec3f(
    materialStorageScalar(base0),
    materialStorageScalar(base0 + 1u),
    materialStorageScalar(base0 + 2u),
  ), vec3f(
    materialStorageScalar(base1),
    materialStorageScalar(base1 + 1u),
    materialStorageScalar(base1 + 2u),
  ), alpha);
  out.transmittance = mix(vec3f(
    materialStorageScalar(base0 + 3u),
    materialStorageScalar(base0 + 4u),
    materialStorageScalar(base0 + 5u),
  ), vec3f(
    materialStorageScalar(base1 + 3u),
    materialStorageScalar(base1 + 4u),
    materialStorageScalar(base1 + 5u),
  ), alpha);
  out.reflectanceEnergy = mix(
    materialStorageScalar(base0 + 6u),
    materialStorageScalar(base1 + 6u), alpha,
  );
  out.transmittanceEnergy = mix(
    materialStorageScalar(base0 + 7u),
    materialStorageScalar(base1 + 7u), alpha,
  );
  out.absorptionEnergy =
    max(0.0, 1.0 - out.reflectanceEnergy - out.transmittanceEnergy);
  return thinFilmApplyTransmissionScale(out, film.transmissionScale);
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
  let n2 = clamp(nDotH * nDotH, 0.0, 1.0);
  // Cancellation-free form of n²(α²-1)+1. The shared finite-alpha floor
  // guarantees d>0, so a denominator floor must not flatten narrow lobes.
  let d = (1.0 - n2) + n2 * a2;
  return a2 / (PI * d * d);
}

${roughDielectricSmithG1Wgsl('smithG1')}

fn powerHeuristic(pdfA: f32, pdfB: f32) -> f32 {
  if (
    !(pdfA >= 0.0) || !(pdfB >= 0.0) ||
    pdfA > 3.402823466e38 || pdfB > 3.402823466e38
  ) {
    return 0.0;
  }
  let scale = max(pdfA, pdfB);
  if (!(scale > 0.0)) { return 0.0; }
  let a = pdfA / scale;
  let b = pdfB / scale;
  let a2 = a * a;
  let b2 = b * b;
  return a2 / (a2 + b2);
}

// ── B9 — GGX multiple-scattering energy compensation (Kulla-Conty 2017) ───────
// The single-scatter Cook-Torrance/Smith microfacet BRDF loses energy at high
// roughness because it models only ONE bounce off the microsurface; light that
// would bounce multiple times between microfacets is dropped, darkening rough
// metals/dielectrics. Kulla & Conty (2017, "Revisiting Physically Based Shading
// at Imageworks", SIGGRAPH course) restore it by adding a diffuse-like
// multiscatter lobe weighted so the total BRDF integrates to ~1 (white furnace).
//
//   f_ms = (1 − E(μ_o)) · (1 − E(μ_i)) / (π · (1 − E_avg))      [Kulla-Conty]
//
// scaled by an averaged Fresnel F_avg so coloured metals tint the extra bounces.
// We avoid a precomputed 2-D E LUT by using Turquin's analytic fit (Turquin,
// "Practical multiple scattering compensation for microfacet models," 2019) of
// the GGX single-scatter directional albedo E_ss(μ,α). The fit is a cheap
// polynomial that matches the tabulated albedo to <1 % and is what most
// production engines embed in lieu of the table.
// Refs: Kulla & Conty 2017; Turquin 2019 (https://blog.selfshadow.com/
//       publications/turquin/ms_comp_final.pdf); Fdez-Agüera 2019.

// Precomputed 8×8 single-scatter GGX directional-albedo LUT, E_ss(roughness, μ).
// Rows = roughness 0..1 (8 steps), cols = μ = N·V 0..1 (8 steps), row-major. The
// table is the hemispherical-integrated single-scatter throughput E[G1(wi)] under
// VNDF sampling (F=1), computed offline by the same VNDF sampler the kernel uses
// (the values match the ggxMultiscatterFurnace CPU harness to <1%). This is the
// standard Kulla-Conty E LUT; embedding the 64-entry table is cheaper and far more
// accurate than an analytic fit (which mis-sized the missing energy). Smooth
// surfaces (r→0, high μ) read ≈1 (no loss); very rough (r→1) reads down to ~0.31.
const GGX_E_LUT_DIM = 8u;
const GGX_E_LUT = array<f32, 64>(
  0.1375, 0.5617, 0.7546, 0.8522, 0.9111, 0.9505, 0.9788, 1.0,
  0.2955, 0.515,  0.7091, 0.8192, 0.889,  0.937,  0.9721, 0.9988,
  0.5794, 0.5541, 0.6677, 0.7691, 0.8451, 0.9021, 0.9457, 0.98,
  0.7011, 0.6486, 0.6669, 0.7199, 0.7776, 0.8305, 0.8764, 0.9155,
  0.7335, 0.6901, 0.6696, 0.6756, 0.6972, 0.7262, 0.7578, 0.7893,
  0.7153, 0.6712, 0.6355, 0.6145, 0.6052, 0.6045, 0.6101, 0.6199,
  0.6669, 0.6137, 0.5657, 0.5286, 0.5,    0.478,  0.4611, 0.4483,
  0.6017, 0.537,  0.4773, 0.4296, 0.3905, 0.358,  0.3305, 0.3069,
);
// Hemispherical-average E_avg(roughness) = 2∫₀¹ E_ss(μ,r)·μ dμ, per roughness row
// (the Kulla-Conty denominator). Same offline integration as the LUT.
const GGX_EAVG_LUT = array<f32, 8>(
  0.9106, 0.8931, 0.8629, 0.8094, 0.725, 0.6147, 0.4931, 0.3766,
);

// Bilinear lookup of the single-scatter GGX directional albedo E_ss(μ, roughness).
fn ggxDirectionalAlbedo(cosTheta: f32, roughness: f32) -> f32 {
  let mu = clamp(cosTheta, 0.0, 1.0);
  let r = clamp(roughness, 0.0, 1.0);
  let fr = r * f32(GGX_E_LUT_DIM - 1u);
  let fm = mu * f32(GGX_E_LUT_DIM - 1u);
  let r0 = u32(floor(fr));
  let m0 = u32(floor(fm));
  let r1 = min(r0 + 1u, GGX_E_LUT_DIM - 1u);
  let m1 = min(m0 + 1u, GGX_E_LUT_DIM - 1u);
  let tr = fr - f32(r0);
  let tm = fm - f32(m0);
  let e00 = GGX_E_LUT[r0 * GGX_E_LUT_DIM + m0];
  let e01 = GGX_E_LUT[r0 * GGX_E_LUT_DIM + m1];
  let e10 = GGX_E_LUT[r1 * GGX_E_LUT_DIM + m0];
  let e11 = GGX_E_LUT[r1 * GGX_E_LUT_DIM + m1];
  let e0 = mix(e00, e01, tm);
  let e1 = mix(e10, e11, tm);
  return clamp(mix(e0, e1, tr), 0.02, 1.0);
}

// Linear lookup of E_avg(roughness).
fn ggxAverageAlbedo(roughness: f32) -> f32 {
  let r = clamp(roughness, 0.0, 1.0);
  let fr = r * f32(GGX_E_LUT_DIM - 1u);
  let r0 = u32(floor(fr));
  let r1 = min(r0 + 1u, GGX_E_LUT_DIM - 1u);
  let tr = fr - f32(r0);
  return clamp(mix(GGX_EAVG_LUT[r0], GGX_EAVG_LUT[r1], tr), 0.3, 1.0);
}

// Kulla-Conty multiple-scattering compensation BRDF kernel (WITHOUT nDotL; the
// caller multiplies by nDotL once, matching evaluateBrdf's convention). f0 tints
// the extra bounces by an averaged Fresnel so coloured metals stay coloured.
// Directional roughness inputs let the anisotropic GGX path query the same
// isotropic E LUT with a projected per-axis roughness instead of ignoring the
// stretched lobe. The isotropic wrapper below passes one roughness to all three
// inputs, preserving the historical zero-anisotropy path exactly.
fn ggxMultiscatterLobeRoughness(
  f0: vec3f,
  roughnessV: f32,
  roughnessL: f32,
  roughnessAvg: f32,
  nDotV: f32,
  nDotL: f32,
) -> vec3f {
  let eAvg = ggxAverageAlbedo(roughnessAvg);
  let oneMinusEavg = 1.0 - eAvg;
  if (oneMinusEavg < 1e-4) { return vec3f(0.0); } // smooth → no missing energy.
  let eo = ggxDirectionalAlbedo(nDotV, roughnessV);
  let ei = ggxDirectionalAlbedo(nDotL, roughnessL);
  // Averaged Fresnel for the multiscatter tint (Kulla-Conty): F_avg ≈ F0 + (1−F0)/21.
  let fAvg = f0 + (vec3f(1.0) - f0) * (1.0 / 21.0);
  // Multiscatter energy: the geometric series of bounces sums to F_avg·E_avg /
  // (1 − F_avg·(1−E_avg)) for the colour, times the Kulla-Conty directional shape.
  let fMs = fAvg * fAvg * eAvg / max(vec3f(1.0) - fAvg * oneMinusEavg, vec3f(1e-4));
  let shape = (1.0 - eo) * (1.0 - ei) / max(PI * oneMinusEavg, 1e-6);
  return fMs * shape;
}

// Returns 0 for smooth surfaces (E_avg≈1) → zero loss → byte-identical low-r.
fn ggxMultiscatterLobe(f0: vec3f, roughness: f32, nDotV: f32, nDotL: f32) -> vec3f {
  return ggxMultiscatterLobeRoughness(f0, roughness, roughness, roughness, nDotV, nDotL);
}

// B9 — multiscatter energy boost for the SAMPLED specular lobe (Kulla-Conty). The
// VNDF sampler realises single-scatter only; multiply the sampled throughput by
//   1 + F_avg · (1 − E_ss(μ_o)) / E_ss(μ_o)
// to recover the missing multi-bounce energy. F_avg here is the Fresnel at the
// view (passed as the already-evaluated fresnel vec). Returns 1 at low roughness
// (E_ss→1 → factor→1) so smooth surfaces are unchanged.
fn ggxMultiscatterBoost(fresnel: vec3f, roughness: f32, nDotV: f32) -> vec3f {
  let eo = ggxDirectionalAlbedo(nDotV, roughness);
  let missing = clamp(1.0 - eo, 0.0, 1.0);
  if (missing < 1e-4) { return vec3f(1.0); }
  let fAvg = fresnel + (vec3f(1.0) - fresnel) * (1.0 / 21.0);
  return vec3f(1.0) + fAvg * (missing / max(eo, 1e-3));
}

fn ggxMultiscatterBoostRoughness(fresnel: vec3f, roughnessV: f32, nDotV: f32) -> vec3f {
  let eo = ggxDirectionalAlbedo(nDotV, roughnessV);
  let missing = clamp(1.0 - eo, 0.0, 1.0);
  if (missing < 1e-4) { return vec3f(1.0); }
  let fAvg = fresnel + (vec3f(1.0) - fresnel) * (1.0 / 21.0);
  return vec3f(1.0) + fAvg * (missing / max(eo, 1e-3));
}

// Two-term Cauchy dispersion reconstructed from MaterialSpec's authored
// Fraunhofer d-line IOR and Abbe number:
//   V_d = (n_d - 1) / (n_F - n_C)
//   n(lambda) = n_d + B * (1/lambda^2 - 1/lambda_d^2)
// Wavelengths stay in nanometres, matching shared-samplers' Abbe reduction and
// making lambda == lambda_d an exact zero-delta operation in shader arithmetic.
fn cauchyIorAtLambda(lambdaNm: f32, baseIor: f32, abbeV: f32) -> f32 {
  if (abbeV <= 0.0) {
    return baseIor;
  }
  let lam2 = lambdaNm * lambdaNm;
  let lamD = 589.3;
  let lamF = 486.1;
  let lamC = 656.3;
  let denom = 1.0 / (lamF * lamF) - 1.0 / (lamC * lamC);
  let B = (baseIor - 1.0) / abbeV / max(denom, 1e-6);
  return max(1.0, baseIor + B * (1.0 / lam2 - 1.0 / (lamD * lamD)));
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
  // Authored optical role.  Texture samples never change boundary topology:
  // a material is either a closed bulk-medium boundary or a virtual zero-
  // thickness two-interface sheet for the lifetime of a published scene.
  isBulkMedium: bool,
  isThinSheet: bool,
  // Compatibility alias used by older call sites while the transport kernels
  // are migrated.  This is exactly isBulkMedium, never mapped transmission.
  isTranslucent: bool,
  // WS4 — Beer-Lambert absorption coefficient σ_a (per channel), derived host-side
  // from attenuationColor/attenuationDistance. hasSigmaA distinguishes a clear
  // medium (σ_a = 0) from "no absorption authored".
  sigmaA: vec3f,
  hasSigmaA: bool,
  // H52 — Disney extension lobes (clearcoat / sheen / iridescence).
  // All three default to 0; zero-default scenes are numerically identical to
  // the pre-H52 path because each lobe is gated on its scalar being > 0.
  // Refs: glTF KHR_materials_clearcoat, KHR_materials_sheen, KHR_materials_iridescence.
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  // A3 — Jakob & Hanika 2019 sigmoid-polynomial coefficients (raw-nm) for the
  // baseColor's spectral reflectance S(λ) = sigmoid(c0 + c1·λ + c2·λ²). Consumed
  // ONLY in spectral mode (params.spectralEnabled != 0) to evaluate per-λ
  // reflectance; the RGB path never reads these.
  spectralReflCoeffs: vec3f,
  hasSpectralReflectance: bool,
  isUnlit: bool,
  doubleSided: bool,
  // SPEC-01 — KHR_materials_specular scalar factors. specularColor is
  // non-negative and intentionally unbounded until IOR-derived F0 composition.
  specularColor: vec3f,
  specularIntensity: f32,
  volumeThickness: f32,
  hasVolumeThickness: bool,
}

// A3 — evaluate the Jakob & Hanika 2019 sigmoid-polynomial spectral reflectance
// at wavelength λ (nm). S(λ) = ½ + x/(2·√(1+x²)),  x = c0 + c1·λ + c2·λ².
// Bounded in (0,1) and numerically safe for large |x| (no exp overflow). Mirrors
// the TS evaluateSpectrum in shared-samplers/jakobHanika.ts.
fn evalJakobHanikaSpectrum(coeffs: vec3f, lambdaNm: f32) -> f32 {
  let x = coeffs.x + coeffs.y * lambdaNm + coeffs.z * lambdaNm * lambdaNm;
  return 0.5 + x / (2.0 * sqrt(1.0 + x * x));
}

// Evaluate an RGB multiplicative factor at the hero wavelength without CMF
// reconstruction. This stays inside scalar spectral transport; reconstruction
// belongs exclusively at the final eye-path boundary.
fn spectralRgbFactorAtHero(rgb: vec3f, lambdaNm: f32) -> f32 {
  let value = max(rgb, vec3f(0.0));
  let t = heroLambdaTo01(lambdaNm);
  let wB = max(1.0 - abs(t - 0.15) / 0.35, 0.0);
  let wG = max(1.0 - abs(t - 0.50) / 0.35, 0.0);
  let wR = max(1.0 - abs(t - 0.85) / 0.35, 0.0);
  let wSum = max(wR + wG + wB, 1e-6);
  return max((value.r * wR + value.g * wG + value.b * wB) / wSum, 0.0);
}

// Preserve the packed Jakob-Hanika spectrum for the authored base colour while
// applying vertex colour, textures, AO, face layers, and thin-film modulation
// at the same hero wavelength.
fn spectralCombinedReflectanceAtHero(
  combinedRgb: vec3f,
  authoredRgb: vec3f,
  coeffs: vec3f,
  hasCoeffs: bool,
  lambdaNm: f32,
) -> f32 {
  let combinedApprox = spectralRgbFactorAtHero(combinedRgb, lambdaNm);
  let authoredApprox = spectralRgbFactorAtHero(authoredRgb, lambdaNm);
  if (authoredApprox <= 1e-7) {
    return clamp(combinedApprox, 0.0, 1.0);
  }
  let authoredExact = select(
    authoredApprox,
    evalJakobHanikaSpectrum(coeffs, lambdaNm),
    hasCoeffs,
  );
  return clamp(authoredExact * combinedApprox / authoredApprox, 0.0, 1.0);
}

// A3 — hero-λ spectral EMISSION from an authored RGB emitter colour. Jakob-Hanika
// upsampling targets reflectances in [0,1]; HDR emission is not bounded there, so
// we factor the emission into luminance × unit-chroma, upsample only the bounded
// chroma to a reflectance-shaped SPD, evaluate S(λ) at the hero wavelength, and
// rescale by the emission luminance. The result is a scalar single-wavelength
// radiance whose CMF reconstruction (heroWavelengthToRgb) integrates back to the
// authored RGB in the flat (neutral) limit and otherwise carries the emitter's
// chromaticity spectrally. A near-black emitter returns 0 (no chroma to shape).
// Note: this is a coefficient solve at runtime would be too costly per-hit, so we
// approximate the chroma SPD by normalising the RGB and reading the per-channel
// reflectance via the standard sigmoid basis seeded from the chroma directly —
// here we use the simpler, robust route of weighting the hero-λ position within
// the chroma triple, documented as the flat-spectrum × chroma approximation.
fn spectralEmissionAtHero(emissionRgb: vec3f, lambdaNm: f32) -> vec3f {
  let lum = max(luminance(emissionRgb), 0.0);
  if (lum < 1e-8) { return vec3f(0.0); }
  // Map hero λ to a chroma weight across the RGB primaries (long→R, mid→G,
  // short→B), so the emitter's chromaticity reaches the hero path. A NEUTRAL
  // emitter (r==g==b) yields a flat chroma == its scalar value.
  let t = heroLambdaTo01(lambdaNm); // 0 (380nm) .. 1 (780nm)
  let wB = max(1.0 - abs(t - 0.15) / 0.35, 0.0);
  let wG = max(1.0 - abs(t - 0.50) / 0.35, 0.0);
  let wR = max(1.0 - abs(t - 0.85) / 0.35, 0.0);
  let wSum = max(wR + wG + wB, 1e-6);
  let chroma = (emissionRgb.r * wR + emissionRgb.g * wG + emissionRgb.b * wB) / wSum;
  // Multiply by the D65-normalised SPD: the reflectance upsampling (Jakob-Hanika)
  // is defined relative to D65, so the transport's "white" illuminant is D65, not
  // equal-energy. With this factor a NEUTRAL grey reflectance under a NEUTRAL
  // emitter reconstructs (through heroWavelengthToRgb) to EXACTLY the RGB the RGB
  // path produces — the flat-spectrum invariant the A3 harness pins. The /Y norm
  // inside heroSampleD65Normalised keeps the overall luminance scale at 1.
  return vec3f(chroma * heroSampleD65Normalised(lambdaNm));
}

// RFE-03 / fork activeLayerWeight: scalar throughput through face layer at hero λ.
fn activeLayerWeightRgb(layerRgb: vec3f, heroLambda: f32, spectralEnabled: bool) -> vec3f {
  if (!spectralEnabled) {
    return layerRgb;
  }
  return vec3f(spectralRgbFactorAtHero(layerRgb, heroLambda));
}

fn decodeMaterial(matId: u32) -> DecodedMaterial {
  let m0Index = materialRecordBase(matId);
  let m1Index = materialRecordIndex(m0Index, 1u);
  let m2Index = materialRecordIndex(m0Index, 2u);
  let m3Index = materialRecordIndex(m0Index, 3u);
  let m4Index = materialRecordIndex(m0Index, 4u);
  let m5Index = materialRecordIndex(m0Index, 5u);
  let m6Index = materialRecordIndex(m0Index, 6u);
  let m19Index = materialRecordIndex(m0Index, 21u);
  let m22Index = materialRecordIndex(m0Index, 22u); // WS4 σ_a vec4
  let m23Index = materialRecordIndex(m0Index, 23u); // H52 clearcoat/sheen vec4
  let m24Index = materialRecordIndex(m0Index, 24u); // H52 sheenColor + iridescence vec4
  let m25Index = materialRecordIndex(m0Index, 25u); // H52 iridescence params vec4
  let m26Index = materialRecordIndex(m0Index, 26u); // A3 spectral coeffs + flags
  let m27Index = materialRecordIndex(m0Index, 27u); // SPEC-01 specular factors
  let m28Index = materialRecordIndex(m0Index, 28u); // VOL-THICKNESS clamp
  let m0 = materialRecordVec4Or(m0Index, vec4f(0.8, 0.8, 0.8, 0.6));
  let m1 = materialRecordVec4Or(m1Index, vec4f(0.0));
  let m2 = materialRecordVec4Or(m2Index, vec4f(0.0, 1.5, 0.0, 0.0));
  let m3 = materialRecordVec4Or(m3Index, vec4f(0.0));
  let m4 = materialRecordVec4Or(m4Index, vec4f(1.0, 1.0, 1.0, -1.0));
  let m5 = materialRecordVec4Or(m5Index, vec4f(1.0, 1.0, 1.0, -1.0));
  let m6 = materialRecordVec4Or(m6Index, vec4f(0.0, 0.0, 1.0, 0.0));
  let m19 = materialRecordVec4Or(m19Index, vec4f(0.0));
  let m22 = materialRecordVec4Or(m22Index, vec4f(0.0));
  // H52: defaults for all three lobes are 0 (zero-default = numerically identical to pre-H52 path).
  let m23 = materialRecordVec4Or(m23Index, vec4f(0.0));
  let m24 = materialRecordVec4Or(m24Index, vec4f(0.0));
  let m25 = materialRecordVec4Or(m25Index, vec4f(1.3, 100.0, 400.0, 0.0));
  // A3 default: a flat grey (c0=0,c1=0,c2=0 ⇒ x=0 ⇒ S≡½) with flag 0 so an
  // unpacked material is treated as having no spectral curve (RGB fallback).
  let m26 = materialRecordVec4Or(m26Index, vec4f(0.0));
  // SPEC-01 default: KHR_materials_specular no-op (dielectric F0 = 0.04).
  let m27 = materialRecordVec4Or(m27Index, vec4f(1.0));
  // VOL-THICKNESS default: no slab clamp, use geometric path length.
  let m28 = materialRecordVec4Or(m28Index, vec4f(0.0));
  let thinFilmLayerCountDecoded = materialRecordExactU32(
    m6.y,
    THIN_FILM_LAYER_LIMIT + 1u,
  );
  let thinFilmLayerCountValid = thinFilmLayerCountDecoded != MATERIAL_INVALID_INDEX;
  let spectralSampleCountDecoded = materialRecordExactU32(
    m19.w,
    SPECTRAL_SAMPLE_COUNT + 1u,
  );
  let materialFlagsDecoded = materialRecordExactU32(m26.w, 8u);
  let materialFlags = select(
    materialFlagsDecoded,
    0u,
    materialFlagsDecoded == MATERIAL_INVALID_INDEX,
  );
  var mat: DecodedMaterial;
  mat.baseColor = m0.rgb;
  mat.roughness = clamp(m0.w, 0.0, 1.0);
  mat.emissive = m1.rgb;
  mat.metallic = clamp(m1.w, 0.0, 1.0);
  mat.transmission = clamp(m2.x, 0.0, 1.0);
  // KHR_materials_ior defines authored zero as effective positive infinity.
  // Keep the transport finite while preserving an exact f32 Fresnel limit of 1.
  mat.ior = select(
    max(m2.y, 1.0),
    ${KHR_MATERIALS_IOR_INFINITY_APPROX},
    m2.y == 0.0,
  );
  mat.scatteringCoeff = max(m2.z, 0.0);
  mat.scatteringAnisotropy = clamp(m2.w, -0.999999, 0.999999);
  mat.scatteringRgb = vec3f(max(m3.x, 0.0), max(m3.y, 0.0), max(m3.z, 0.0));
  mat.hasSpectralAttenuation = m3.w > 0.5;
  mat.frontLayerTx = m4.rgb;
  mat.frontLayerRoughness = m4.w;
  mat.backLayerTx = m5.rgb;
  mat.backLayerRoughness = m5.w;
  mat.thinFilmEnabled = materialRecordFiniteF32(m6.x) &&
    m6.x > 0.5 && thinFilmLayerCountValid;
  mat.thinFilmLayerCountU = select(
    thinFilmLayerCountDecoded,
    0u,
    !thinFilmLayerCountValid,
  );
  mat.thinFilmIncidentIor = max(m6.z, 1.0);
  mat.thinFilmAngleDependent = m6.w > 0.5;
  mat.spectralAvgMu = max(m19.x, 0.0);
  mat.spectralSampleCount = select(
    spectralSampleCountDecoded,
    0u,
    spectralSampleCountDecoded == MATERIAL_INVALID_INDEX,
  );
  mat.dispersionAbbe = max(m19.y, 0.0);
  mat.sigmaA = vec3f(max(m22.x, 0.0), max(m22.y, 0.0), max(m22.z, 0.0));
  mat.hasSigmaA = m22.w > 0.5;
  // H52 — clearcoat / sheen / iridescence lobe decode.
  mat.clearcoat = clamp(m23.x, 0.0, 1.0);
  mat.clearcoatRoughness = clamp(m23.y, 0.0, 1.0);
  mat.sheen = clamp(m23.z, 0.0, 1.0);
  mat.sheenRoughness = clamp(m23.w, 0.0, 1.0);
  mat.sheenColor = clamp(m24.rgb, vec3f(0.0), vec3f(1.0));
  mat.iridescence = clamp(m24.w, 0.0, 1.0);
  mat.iridescenceIor = max(m25.x, 1.0);
  mat.iridescenceThicknessMin = max(m25.y, 0.0);
  mat.iridescenceThicknessMax = max(m25.z, 0.0);
  // A3 — baseColor spectral reflectance (Jakob-Hanika sigmoid coeffs).
  mat.spectralReflCoeffs = m26.xyz;
  mat.hasSpectralReflectance = (materialFlags & 1u) != 0u;
  mat.isUnlit = (materialFlags & 2u) != 0u;
  mat.doubleSided = (materialFlags & 4u) != 0u;
	  mat.specularColor = max(m27.rgb, vec3f(0.0));
	  mat.specularIntensity = clamp(m27.w, 0.0, 1.0);
	  mat.volumeThickness = max(m28.x, 0.0);
	  mat.hasVolumeThickness = m28.y > 0.5;
  // Canonical authored bulk predicate, mirrored by shared-bvh topology
  // validation.  Positive base thickness alone is a real clear dielectric
  // volume (it still changes IOR).  A clear attenuation payload [1,1,1] does
  // not create bulk, while spectral-attenuation presence intentionally does.
  // scatteringRgb is already the canonical effective value: the host copies
  // the scalar fallback into RGB only when an RGB override is absent.
  let hasScattering =
    max(mat.scatteringRgb.x, max(mat.scatteringRgb.y, mat.scatteringRgb.z)) > 0.0;
  let hasAbsorption =
    max(mat.sigmaA.x, max(mat.sigmaA.y, mat.sigmaA.z)) > 0.0;
  mat.isBulkMedium = mat.transmission > 0.0 && (
    mat.volumeThickness > 0.0 ||
    hasScattering ||
    hasAbsorption ||
    mat.hasSpectralAttenuation
  );
  mat.isThinSheet = mat.transmission > 0.0 && !mat.isBulkMedium;
  mat.isTranslucent = mat.isBulkMedium;
  return mat;
}

// glTF defaults materials to single-sided. Opaque back faces are not
// surfaces and traversal continues behind them. Transmissive back faces stay
// admissible even when doubleSided is false so a path can exit an authored
// closed dielectric/volume boundary.
fn materialAcceptsSidedHit(matId: u32, isFrontFace: bool) -> bool {
  if (isFrontFace) { return true; }
  let mat = decodeMaterial(matId);
  return mat.doubleSided || mat.transmission > 0.0;
}

// Endpoint-exact Beer--Lambert transport.  Multiplying IEEE infinity by zero
// produces NaN, so the identity endpoints must be selected before the product.
// Negative/NaN inputs are malformed and fail dark.  Positive infinity is an
// authored, useful endpoint: zero extinction remains identity, any positive
// extinction becomes zero transmittance.
fn materialBeerLane(sigma: f32, distance: f32) -> f32 {
  if (!(sigma >= 0.0) || !(distance >= 0.0)) { return 0.0; }
  if (distance == 0.0 || sigma == 0.0) { return 1.0; }
  if (sigma > PT_F32_MAX || distance > PT_F32_MAX) { return 0.0; }
  return exp(-sigma * distance);
}

fn materialBeer(sigma: vec3f, distance: f32) -> vec3f {
  return vec3f(
    materialBeerLane(sigma.x, distance),
    materialBeerLane(sigma.y, distance),
    materialBeerLane(sigma.z, distance),
  );
}

fn materialFiniteMajorant(sigma: vec3f) -> f32 {
  var majorant = 0.0;
  if (sigma.x > 0.0 && sigma.x <= PT_F32_MAX) { majorant = max(majorant, sigma.x); }
  if (sigma.y > 0.0 && sigma.y <= PT_F32_MAX) { majorant = max(majorant, sigma.y); }
  if (sigma.z > 0.0 && sigma.z <= PT_F32_MAX) { majorant = max(majorant, sigma.z); }
  return majorant;
}

// Marginal proposal terms for an equal-weight RGB channel-mixture free flight.
// The no-collision weight T_c / mean(T) and collision weight
// sigmaS_c*T_c / mean(sigmaT*T) are each bounded by three. This avoids the
// unbounded max-majorant survival ratio while remaining exactly unbiased.
fn materialRgbSurvivalPdf(sigmaT: vec3f, distance: f32) -> f32 {
  let t = materialBeer(sigmaT, distance);
  return (t.x + t.y + t.z) * (1.0 / 3.0);
}

fn materialExtinctionDensityLane(sigma: f32, distance: f32) -> f32 {
  if (!(sigma >= 0.0) || !(distance >= 0.0)) { return 0.0; }
  if (sigma == 0.0) { return 0.0; }
  // An infinite-rate channel is handled as an immediate absorbing/null event
  // before this finite-distance density is evaluated.
  if (sigma > PT_F32_MAX || distance > PT_F32_MAX) { return 0.0; }
  return sigma * materialBeerLane(sigma, distance);
}

fn materialRgbCollisionPdf(sigmaT: vec3f, distance: f32) -> f32 {
  return (
    materialExtinctionDensityLane(sigmaT.x, distance) * (1.0 / 3.0) +
    materialExtinctionDensityLane(sigmaT.y, distance) * (1.0 / 3.0) +
    materialExtinctionDensityLane(sigmaT.z, distance) * (1.0 / 3.0)
  );
}

fn materialAttenuationDistance(segmentDistance: f32, mat: DecodedMaterial) -> f32 {
  if (!(segmentDistance >= 0.0)) { return -1.0; }
  return select(
    segmentDistance,
    min(segmentDistance, mat.volumeThickness),
    mat.hasVolumeThickness,
  );
}
	`;

/** Full trace pass — 3 bind groups (≤10 storage buffers per group). */
export const PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL =
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_WGSL + PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL;

/** Compatibility tier for adapters capped at 10 storage buffers / 4 storage textures. */
export const PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_WGSL =
  PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_BINDINGS_WGSL + PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL;
