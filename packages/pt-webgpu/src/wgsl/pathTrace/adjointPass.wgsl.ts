/**
 * adjointPass.wgsl.ts — the engine-side WS5 Phase-1 path-replay adjoint COMPUTE
 * PASS (the last V24 piece). For each pixel it re-traces the frozen-seed primary
 * ray (brute-force closest-hit), re-derives the single-bounce direct lighting,
 * and accumulates `∂loss/∂θ`. Local material fields use frozen-sample central
 * differences of the complete direct-light estimator: BRDF, cosine, light
 * measure, BRDF PDF, and power-MIS weight. The shared BRDF evaluator includes
 * the production GGX multiscatter model. Fixed-point `adjointScatter` remains
 * the accumulation primitive validated by the focused GPU harness.
 *
 * It deliberately does NOT call the forward `evaluateBrdf` — the per-pixel
 * `dLoss/dRendered` is handed in by `inverseSession` (computed from the baseline
 * render vs target), so the pass only needs the DERIVATIVES of the shading.
 *
 * Scope (Phase 1, matching the differentiable set): single bounce, brute-force
 * intersection (Phase-1 inverse scenes are small — Cornell-scale), directional
 * delta/soft-sun directional + point + spot + stochastic area-measure rect/disc/mesh-area direct lights,
 * plus stochastic environment-map NEE in the same direct-light domain.
 * Indirect and BRDF/transmissive/
 * layered/volume/spectral mapped material terms remain deliberate
 * finite-difference fallbacks until their source terms are mirrored here and
 * GPU-validated. Mapped terms replayed here are scoped to the camera-direct
 * emissive texel multiplier for `emissive` / `emissiveIntensity`, primary-hit
 * light-map radiance for `lightMapIntensity`, plus
 * baseColorMap / COLOR_0 / aoMap, roughnessMap / metallicMap, clearcoat/sheen/
 * iridescence/anisotropy maps, and specular color/intensity local factors used
 * by lit direct BRDF derivatives.
 * Direct lights are summed over all eligible lights (no MC light selection:
 * the adjoint estimates the direct-light expectation for each source; finite
 * area/environment lights use the same area/env measure samples and per-light
 * BRDF/light MIS weights as the forward NEE branch, but not the one-of-N light
 * selection lottery). baseColor/roughness/metallic/specular/clearcoat/
 * sheen scalar params share this direct-light BRDF path. Map-free unlit baseColor is a primary-hit
 * contribution-level identity (`radiance += throughput * baseColor`) and is
 * scattered without requiring any light. The shading normal is faced toward the viewer
 * (the same flip the forward shade prologue applies). The primary-ray jitter
 * sequence matches the inverse baseline render:
 * sample `s` uses `frameSeed = 0x5eed5eed + s`, `frameIndex = 0`, then the pass
 * averages the per-sample derivatives. The sampled directions are FROZEN (path
 * replay differentiates only the continuous shading, never the light/BSDF
 * sampling — sidesteps visibility discontinuities).
 *
 * A FOCUSED single-group pipeline (not a forward-kernel variant): the forward
 * spends all 4 bind groups, so the adjoint binds only the read subset it needs +
 * its own I/O, in group 0.
 *
 * Ref: Vicini 2021 (Path Replay Backprop); Möller-Trumbore 1997 (intersection).
 */
import {
  MATERIAL_TEX_CLEARCOAT_NORMAL_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET,
  MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP_VEC4_OFFSET,
  MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET,
  MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET,
  MATERIAL_TEX_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_UV_META_VEC4S_PER_MAP,
  MATERIAL_TEX_VEC4_STRIDE,
} from '../../scene/materialTextures.js';
import {
  PT_WEBGPU_MICROFACET_ALPHA_FLOOR,
  roughDielectricSmithG1Wgsl,
} from '../../math/roughDielectric.js';
import {
  composePtWebgpuRngWgsl,
  type PtWebgpuSamplingMode,
} from '../common.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL } from './pathTraceAdjoint.wgsl.js';
import { PT_WEBGPU_POINT_SPOT_ATTENUATION_WGSL } from './material.wgsl.js';

/** Field codes in the adjointParams descriptor (matches inverseSession fields). */
export const ADJOINT_FIELD_BASECOLOR = 0;
export const ADJOINT_FIELD_ROUGHNESS = 1;
/** Emissive (rgb). UNLIKE baseColor/roughness this is NOT a lit-surface NEE term:
 *  the forward adds `throughput · emissive` for the emission a CAMERA ray sees the
 *  surface emit DIRECTLY at the PRIMARY hit (shadePrologue.wgsl.ts:63, with
 *  throughput = 1 and prevSampleAllowsAreaMis false on a camera ray). Its partial
 *  ∂rendered_c/∂emissive_c = throughput_c · emissiveIntensity is a self-source — it
 *  needs no light. The descriptor carries the (fixed) emissiveIntensity in `.w`
 *  (bitcast f32), because the packed material folds intensity INTO emissive.rgb. */
export const ADJOINT_FIELD_EMISSIVE = 2;
export const ADJOINT_FIELD_SPECULAR_COLOR = 3;
export const ADJOINT_FIELD_SPECULAR_INTENSITY = 4;
export const ADJOINT_FIELD_METALLIC = 5;
export const ADJOINT_FIELD_EMISSIVE_INTENSITY = 6;
export const ADJOINT_FIELD_CLEARCOAT = 7;
export const ADJOINT_FIELD_CLEARCOAT_ROUGHNESS = 8;
export const ADJOINT_FIELD_SHEEN = 9;
export const ADJOINT_FIELD_SHEEN_ROUGHNESS = 10;
export const ADJOINT_FIELD_SHEEN_COLOR = 11;
export const ADJOINT_FIELD_IRIDESCENCE = 12;
export const ADJOINT_FIELD_IRIDESCENCE_IOR = 13;
export const ADJOINT_FIELD_ANISOTROPY = 14;
export const ADJOINT_FIELD_ANISOTROPY_ROTATION = 15;
export const ADJOINT_FIELD_EMITTER_COLOR = 16;
export const ADJOINT_FIELD_EMITTER_INTENSITY = 17;
export const ADJOINT_FIELD_IRIDESCENCE_THICKNESS_RANGE = 18;
export const ADJOINT_FIELD_AO_MAP_INTENSITY = 19;
export const ADJOINT_FIELD_LIGHT_MAP_INTENSITY = 20;
export const ADJOINT_FIELD_ENV_MAP_INTENSITY = 21;
export const ADJOINT_FIELD_NORMAL_SCALE = 22;
export const ADJOINT_FIELD_BUMP_SCALE = 23;
export const ADJOINT_FIELD_CLEARCOAT_NORMAL_SCALE = 24;

export const ADJOINT_EMITTER_TARGET_DIRECTIONAL = 1;
export const ADJOINT_EMITTER_TARGET_POINT = 2;
export const ADJOINT_EMITTER_TARGET_SPOT = 3;
export const ADJOINT_EMITTER_TARGET_RECT = 4;
export const ADJOINT_EMITTER_TARGET_MESH = 5;

/** AdjointParams UBO size in bytes (mat4 + vec4 + 3×uvec4 + env uvec4 + env vec4). */
export const ADJOINT_PARAMS_UBO_BYTES = 64 + 16 + 16 + 16 + 16 + 16 + 16;

const ADJOINT_MATERIAL_TEX_UV_EMISSIVE =
  MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP;
const ADJOINT_MATERIAL_TEX_UV_BASE_COLOR = MATERIAL_TEX_UV_META_VEC4_OFFSET;
const ADJOINT_MATERIAL_TEX_UV_NORMAL =
  MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 2;
const ADJOINT_MATERIAL_TEX_UV_ROUGHNESS =
  MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 3;
const ADJOINT_MATERIAL_TEX_UV_METALLIC =
  MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 4;
const ADJOINT_MATERIAL_TEX_UV_AO =
  MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 5;
const ADJOINT_MATERIAL_TEX_UV_LIGHT =
  MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 6;
const ADJOINT_MATERIAL_TEX_UV_BUMP =
  MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 7;
const ADJOINT_MATERIAL_TEX_UV_CLEARCOAT = MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET;
const ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_ROUGHNESS =
  MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP;
const ADJOINT_MATERIAL_TEX_UV_SHEEN_COLOR =
  MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 2;
const ADJOINT_MATERIAL_TEX_UV_SHEEN_ROUGHNESS =
  MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 3;
const ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE =
  MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 4;
const ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE_THICKNESS =
  MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 5;
const ADJOINT_MATERIAL_TEX_UV_SPECULAR_COLOR =
  MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 6;
const ADJOINT_MATERIAL_TEX_UV_SPECULAR_INTENSITY =
  MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 7;
const ADJOINT_MATERIAL_TEX_UV_ANISOTROPY =
  MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 8;
const ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL = MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET;
const ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP =
  MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP_VEC4_OFFSET;
const ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_NORMAL =
  MATERIAL_TEX_CLEARCOAT_NORMAL_UV_META_VEC4_OFFSET;

function adjointSourceRectSamplerWgsl(name: string, texArray: string): string {
  return /* wgsl */ `
fn ${name}AtMip(
  layerIdx: i32,
  uv: vec2f,
  sourceBaseSize: vec2u,
  wrapMode: vec2f,
  mip: u32,
  linearFilter: bool,
) -> vec4f {
  let sourceSize = adjointMaterialTextureSourceMipSize(sourceBaseSize, mip);
  if (!linearFilter) {
    let raw = vec2i(floor(uv * vec2f(sourceSize)));
    let coord = vec2i(
      adjointMaterialTextureWrapTexel(raw.x, i32(sourceSize.x), wrapMode.x),
      adjointMaterialTextureWrapTexel(raw.y, i32(sourceSize.y), wrapMode.y),
    );
    return textureLoad(${texArray}, coord, layerIdx, mip);
  }
  let samplePosition = uv * vec2f(sourceSize) - vec2f(0.5);
  let baseCoord = vec2i(floor(samplePosition));
  let blend = fract(samplePosition);
  let x0 = adjointMaterialTextureWrapTexel(baseCoord.x, i32(sourceSize.x), wrapMode.x);
  let x1 = adjointMaterialTextureWrapTexel(baseCoord.x + 1, i32(sourceSize.x), wrapMode.x);
  let y0 = adjointMaterialTextureWrapTexel(baseCoord.y, i32(sourceSize.y), wrapMode.y);
  let y1 = adjointMaterialTextureWrapTexel(baseCoord.y + 1, i32(sourceSize.y), wrapMode.y);
  let c00 = textureLoad(${texArray}, vec2i(x0, y0), layerIdx, mip);
  let c10 = textureLoad(${texArray}, vec2i(x1, y0), layerIdx, mip);
  let c01 = textureLoad(${texArray}, vec2i(x0, y1), layerIdx, mip);
  let c11 = textureLoad(${texArray}, vec2i(x1, y1), layerIdx, mip);
  return mix(mix(c00, c10, blend.x), mix(c01, c11, blend.x), blend.y);
}

fn ${name}(
  layerIdx: i32,
  uv: vec2f,
  sourceBaseSize: vec2u,
  wrapMode: vec2f,
  policyLod: f32,
  filterMode: f32,
  mipPolicy: f32,
) -> vec4f {
  let maxMip = adjointMaterialTextureSourceMipCount(sourceBaseSize) - 1u;
  let lod0 = min(u32(floor(max(policyLod, 0.0))), maxMip);
  let lod1 = min(lod0 + 1u, maxMip);
  let linearFilter = filterMode >= 0.5;
  let c0 = ${name}AtMip(layerIdx, uv, sourceBaseSize, wrapMode, lod0, linearFilter);
  let c1 = ${name}AtMip(layerIdx, uv, sourceBaseSize, wrapMode, lod1, linearFilter);
  return mix(c0, c1, select(0.0, fract(policyLod), mipPolicy >= 1.5));
}
`;
}

const ADJOINT_SOURCE_RECT_SAMPLERS_WGSL =
  adjointSourceRectSamplerWgsl('sampleAdjointSrgbSourceRect', 'materialTextures') +
  adjointSourceRectSamplerWgsl('sampleAdjointLinearSourceRect', 'materialTexturesLinear') +
  adjointSourceRectSamplerWgsl('sampleAdjointEmissiveSourceRect', 'materialTexturesEmissive');

export function composePtWebgpuAdjointPassWgsl(
  sampling: PtWebgpuSamplingMode = 'pcg',
): string {
  return /* wgsl */ `
const PI = 3.14159265358979;
const INV_PI = 0.31830988618;
${PT_WEBGPU_POINT_SPOT_ATTENUATION_WGSL}
// MUST match the canonical MATERIAL_VEC4_STRIDE (material.wgsl.ts / materialPacking.ts).
// This adjoint pass reads the SAME materials storage buffer the forward kernel
// uploads, so its per-material stride must equal the forward stride or every
// matId>0 material read is misaligned. Was a stale 23u (the stride at the time
// this pass was written); the forward stride has since grown through WS4/H52/A3,
// SPEC-01, and VOL-THICKNESS. matId=0 is unaffected by a stale stride because
// 0*stride=0, so single-material adjoint tests can miss this latent multi-material
// inverse-fit misalignment.
const MATERIAL_VEC4_STRIDE = 29u;

struct AdjointParams {
  invViewProj: mat4x4f,
  cameraPos:   vec4f,
  width:       u32,
  height:      u32,
  triangleCount: u32,
  pointLightCount: u32,
  paramCount:  u32,
  channels:    u32,
  rectAreaLightCount: u32,
  sampleCount: u32,
  directionalLightCount: u32,
  spotLightCount: u32,
  meshAreaLightCount: u32,
  _pad1: u32,
  environmentMapWidth: u32,
  environmentMapHeight: u32,
  hasEnvironmentMap: u32,
  _pad2: u32,
  environmentParams: vec4f,
}

@group(0) @binding(0) var<uniform>             params:        AdjointParams;
@group(0) @binding(1) var<storage, read>       positions:     array<vec4f>;
@group(0) @binding(2) var<storage, read>       indices:       array<vec4u>;
@group(0) @binding(3) var<storage, read>       triMaterialIds: array<u32>;
@group(0) @binding(4) var<storage, read>       materials:     array<vec4f>;
@group(0) @binding(5) var<storage, read>       normals:       array<vec4f>;
@group(0) @binding(6) var<storage, read>       pointLights:   array<vec4f>;
@group(0) @binding(7) var<storage, read>       dLossDRendered: array<f32>;
@group(0) @binding(8) var<storage, read_write> gradAccum:     array<atomic<i32>>;
// adjointParams:
//   material fields: {matId, fieldCode, gradOffset, fieldPayloadBits}
//   emitter fields: {kind-local light slot/range start, fieldCode, gradOffset, emitterTargetMeta}
// adjointParamDescs: two vec4u records per optimized param:
//   record 0: {targetIdOrSlot, fieldCode, gradOffset, fieldPayloadBitsOrEmitterKind}
//   record 1: {payloadXBits, payloadYBits, payloadZBits, payloadWBits}
// emissive uses record0.w for fixed emissiveIntensity. emissiveIntensity
// uses record1.xyz for UNFACTORED emissive RGB so intensity=0 is differentiable.
// emitter color/intensity use record1.xyz = unfactored emitter color and
// record1.w = fixed intensity; emitterTargetMeta packs target kind in the low
// 8 bits and a contiguous range count in the upper bits. Mapped mesh-area
// emitters read meshAreaLightSourceFactors, so zero authored color channels do
// not require a packedRadiance/color quotient. Mesh-area source-factor w
// stores an explicit owner slot (1-based, 0 = implicit/unowned), letting capped
// or power-sorted mesh-light replay scatter to the owning emitter without
// relying on packed triangle contiguity.
@group(0) @binding(9) var<storage, read>       adjointParamDescs: array<vec4u>;
// rect-area lights: per light {position, uAxis, vAxis, radiance} (4 vec4 stride).
@group(0) @binding(10) var<storage, read>      rectAreaLights: array<vec4f>;
// directional lights: per light {towardLight+angularDiameter, irradiance+mean} (2 vec4 stride).
@group(0) @binding(11) var<storage, read>      directionalLights: array<vec4f>;
// spot lights: per light {position, axis+cosOuter, radiance+cosInner, distance+decay+shadowFlag} (4 vec4 stride).
@group(0) @binding(12) var<storage, read>      spotLights: array<vec4f>;
// mesh-area lights: per triangle {a, b, c, proposalLe+shadowFlag,
// rawUvA+rawUvB, rawUvC+materialToken+sourceArea, baseLe} (7 vec4 stride).
@group(0) @binding(13) var<storage, read>      meshAreaLights: array<vec4f>;
// Material-map replay subset: mirrors the forward texture samplers for local
// base/ORM/AO/specular/clearcoat/sheen/iridescence/anisotropy chain factors,
// camera-direct emissive partials, top-level normal/bump-map shading, and
// clearcoat-normal maps for the additive clearcoat lobe. Path-changing
// maps (alpha, transmission, displacement) still route
// through finite difference until their visibility/transport/geometry terms are
// replayed here.
@group(0) @binding(14) var<storage, read>      meshUvs: array<vec4f>;
@group(0) @binding(15) var<storage, read>      materialTexDescriptors: array<vec4f>;
@group(0) @binding(16) var                      materialTextures: texture_2d_array<f32>;
@group(0) @binding(17) var                      materialTexSampler: sampler;
@group(0) @binding(18) var<storage, read>       meshVertexColors: array<vec4f>;
@group(0) @binding(19) var                      materialTexturesLinear: texture_2d_array<f32>;
// Environment-map replay subset: rgba = radiance.rgb + solid-angle pdf, plus
// normalized CDF. Mirrors scene/uploadSceneBuffers.ts environment packing.
@group(0) @binding(20) var<storage, read>       environmentMapTexels: array<vec4f>;
@group(0) @binding(21) var<storage, read>       environmentMapCdf: array<f32>;
// mesh-area source factors: per triangle {emissiveMapFactor.rgb, ownerSlot+1}.
// Unmapped mesh-area lights store 1,1,1. ownerSlot=0 means implicit/unowned.
// This keeps emitter-color gradients defined when an authored color channel is
// currently zero and keeps capped/reordered explicit mesh emitters targetable.
@group(0) @binding(22) var<storage, read>       meshAreaLightSourceFactors: array<vec4f>;
// xyz = tangent, w = bitangent sign; mirrors the forward normal-map path.
@group(0) @binding(23) var<storage, read>       meshTangents: array<vec4f>;
// T1-6 — dedicated rgba16float emissive array (HDR emissive). emissiveIdx now
// indexes this array's layer space (NOT the sRGB baseColor array), so the
// emissive replay below samples here to stay consistent with the forward pass.
@group(0) @binding(24) var                      materialTexturesEmissive: texture_2d_array<f32>;

fn adjointUvForVertex(vertexIndex: u32, gpuUvSlot: u32) -> vec2f {
  let vertexCount = arrayLength(&positions);
  if (vertexIndex >= vertexCount || vertexIndex >= arrayLength(&meshUvs)) {
    return vec2f(0.0);
  }
  let primary = meshUvs[vertexIndex];
  if (gpuUvSlot == 0u) { return primary.xy; }
  if (gpuUvSlot == 1u) { return primary.zw; }
  let tailIndex = vertexCount + (gpuUvSlot - 2u) * vertexCount + vertexIndex;
  if (tailIndex >= arrayLength(&meshUvs)) { return vec2f(0.0); }
  return meshUvs[tailIndex].xy;
}

// ── BRDF primitives ──────────────────────────────────────────────────────────
const ADJOINT_FROZEN_SEED_BASE = 0x5eed5eedu;
${composePtWebgpuRngWgsl(sampling)}
//
// MIRROR SITE — these four functions are intentionally duplicated here from
// their canonical definitions so this adjoint-pass shader is a self-contained
// compute module (it is NOT composed with the megakernel prefix stack).
//
//   safe_normalize   → common.wgsl.ts:42
//   ggxD             → material.wgsl.ts:741  (GGX NDF, Trowbridge-Reitz)
//   smithG1          → material.wgsl.ts:747  (Smith masking/shadowing term)
//   fresnelSchlick   → material.wgsl.ts:710  (Schlick Fresnel approximation)
//
// If you change the body of any of these functions in their canonical location
// you MUST apply the same change here (and vice-versa).
fn safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  if (l < 1e-8) { return vec3f(0.0); }
  return v / l;
}
fn rotateYPos(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}
fn ggxD(nDotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let n2 = clamp(nDotH * nDotH, 0.0, 1.0);
  let d = (1.0 - n2) + n2 * a2;
  return a2 / (PI * d * d);
}
${roughDielectricSmithG1Wgsl('smithG1')}
fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  let m5 = m2 * m2 * m;
  return f0 + (vec3f(1.0) - f0) * m5;
}
struct AdjointEnvironmentSample {
  wi: vec3f,
  value: vec3f,
  pdf: f32,
}

fn sampleAdjointEnvironmentImportance(rng: ptr<function, PtRngState>) -> AdjointEnvironmentSample {
  var result: AdjointEnvironmentSample;
  result.wi = vec3f(0.0, 1.0, 0.0);
  result.value = vec3f(0.0);
  result.pdf = 0.0;
  if (params.hasEnvironmentMap == 0u || params.environmentMapWidth == 0u || params.environmentMapHeight == 0u) {
    return result;
  }
  let count = params.environmentMapWidth * params.environmentMapHeight;
  if (count == 0u || arrayLength(&environmentMapCdf) < count + 1u || arrayLength(&environmentMapTexels) < count) {
    return result;
  }
  let xi = rand_f32(rng);
  var lo = 0u;
  var hi = count;
  loop {
    if (lo + 1u >= hi) { break; }
    let mid = (lo + hi) >> 1u;
    if (environmentMapCdf[mid] <= xi) { lo = mid; } else { hi = mid; }
  }
  let idx = min(lo, count - 1u);
  let x = idx % params.environmentMapWidth;
  let y = idx / params.environmentMapWidth;
  let u = (f32(x) + 0.5) / f32(params.environmentMapWidth);
  let v = (f32(y) + 0.5) / f32(params.environmentMapHeight);
  let phi = (u - 0.5) * (2.0 * PI);
  let theta = v * PI;
  let sinTheta = sin(theta);
  let mapDir = vec3f(cos(phi) * sinTheta, cos(theta), sin(phi) * sinTheta);
  let texel = environmentMapTexels[idx];
  if (!(texel.w > 0.0) || !(texel.w == texel.w) || abs(texel.w) > 3.402823e38) {
    return result;
  }
  result.wi = safe_normalize(rotateYPos(mapDir, params.environmentParams.y));
  result.value = texel.rgb * max(params.environmentParams.x, 0.0);
  result.pdf = texel.w;
  return result;
}

fn adjointMaterialEnvMapIntensity(matId: u32) -> f32 {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 4u >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  return max(materialTexDescriptors[base + 4u].w, 0.0);
}

// ── emissive texture replay subset (mirror of material.wgsl sampleEmissiveTexture) ──
const ADJOINT_MATERIAL_TEX_VEC4_STRIDE = ${MATERIAL_TEX_VEC4_STRIDE}u;
const ADJOINT_MATERIAL_TEX_UV_BASE_COLOR = ${ADJOINT_MATERIAL_TEX_UV_BASE_COLOR}u;
const ADJOINT_MATERIAL_TEX_UV_EMISSIVE = ${ADJOINT_MATERIAL_TEX_UV_EMISSIVE}u;
const ADJOINT_MATERIAL_TEX_UV_NORMAL = ${ADJOINT_MATERIAL_TEX_UV_NORMAL}u;
const ADJOINT_MATERIAL_TEX_UV_ROUGHNESS = ${ADJOINT_MATERIAL_TEX_UV_ROUGHNESS}u;
const ADJOINT_MATERIAL_TEX_UV_METALLIC = ${ADJOINT_MATERIAL_TEX_UV_METALLIC}u;
const ADJOINT_MATERIAL_TEX_UV_AO = ${ADJOINT_MATERIAL_TEX_UV_AO}u;
const ADJOINT_MATERIAL_TEX_UV_LIGHT = ${ADJOINT_MATERIAL_TEX_UV_LIGHT}u;
const ADJOINT_MATERIAL_TEX_UV_BUMP = ${ADJOINT_MATERIAL_TEX_UV_BUMP}u;
const ADJOINT_MATERIAL_TEX_UV_CLEARCOAT = ${ADJOINT_MATERIAL_TEX_UV_CLEARCOAT}u;
const ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_ROUGHNESS = ${ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_ROUGHNESS}u;
const ADJOINT_MATERIAL_TEX_UV_SHEEN_COLOR = ${ADJOINT_MATERIAL_TEX_UV_SHEEN_COLOR}u;
const ADJOINT_MATERIAL_TEX_UV_SHEEN_ROUGHNESS = ${ADJOINT_MATERIAL_TEX_UV_SHEEN_ROUGHNESS}u;
const ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE = ${ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE}u;
const ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE_THICKNESS = ${ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE_THICKNESS}u;
const ADJOINT_MATERIAL_TEX_UV_SPECULAR_COLOR = ${ADJOINT_MATERIAL_TEX_UV_SPECULAR_COLOR}u;
const ADJOINT_MATERIAL_TEX_UV_SPECULAR_INTENSITY = ${ADJOINT_MATERIAL_TEX_UV_SPECULAR_INTENSITY}u;
const ADJOINT_MATERIAL_TEX_UV_ANISOTROPY = ${ADJOINT_MATERIAL_TEX_UV_ANISOTROPY}u;
const ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL = ${ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL}u;
const ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP = ${ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP}u;
const ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_NORMAL = ${ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_NORMAL}u;
const ADJOINT_MATERIAL_TEX_MIP_POLICY = ${MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET}u;
const ADJOINT_MATERIAL_TEX_MIP_BASE_COLOR = 0u;
const ADJOINT_MATERIAL_TEX_MIP_EMISSIVE = 1u;
const ADJOINT_MATERIAL_TEX_MIP_NORMAL = 2u;
const ADJOINT_MATERIAL_TEX_MIP_ROUGHNESS = 3u;
const ADJOINT_MATERIAL_TEX_MIP_METALLIC = 4u;
const ADJOINT_MATERIAL_TEX_MIP_AO = 5u;
const ADJOINT_MATERIAL_TEX_MIP_LIGHT = 6u;
const ADJOINT_MATERIAL_TEX_MIP_BUMP = 7u;
const ADJOINT_MATERIAL_TEX_MIP_ANISOTROPY = 8u;
const ADJOINT_MATERIAL_TEX_MIP_CLEARCOAT = 11u;
const ADJOINT_MATERIAL_TEX_MIP_CLEARCOAT_ROUGHNESS = 12u;
const ADJOINT_MATERIAL_TEX_MIP_SHEEN_COLOR = 13u;
const ADJOINT_MATERIAL_TEX_MIP_SHEEN_ROUGHNESS = 14u;
const ADJOINT_MATERIAL_TEX_MIP_IRIDESCENCE = 15u;
const ADJOINT_MATERIAL_TEX_MIP_IRIDESCENCE_THICKNESS = 16u;
const ADJOINT_MATERIAL_TEX_MIP_SPECULAR_COLOR = 17u;
const ADJOINT_MATERIAL_TEX_MIP_SPECULAR_INTENSITY = 18u;
const ADJOINT_MATERIAL_TEX_MIP_CLEARCOAT_NORMAL = 19u;
const ADJOINT_MATERIAL_TEX_FILTER_POLICY = ${MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET}u;

struct AdjointAnisotropyMapSample {
  strength: f32,
  rotationOffset: f32,
}

fn adjointWrapTextureCoord(coord: f32, mode: f32) -> f32 {
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

fn adjointMaterialTextureSourceBaseSize(arraySize: vec2u, uvFitScale: vec2f) -> vec2u {
  return max(vec2u(1u), vec2u(round(vec2f(arraySize) * uvFitScale)));
}

fn adjointMaterialTextureSourceMipCount(sourceBaseSize: vec2u) -> u32 {
  return 1u + u32(floor(log2(f32(max(sourceBaseSize.x, sourceBaseSize.y)))));
}

fn adjointMaterialTextureSourceMipSize(sourceBaseSize: vec2u, mip: u32) -> vec2u {
  return vec2u(
    max(1u, sourceBaseSize.x >> mip),
    max(1u, sourceBaseSize.y >> mip),
  );
}

fn adjointMaterialTexturePositiveModulo(value: i32, modulus: i32) -> i32 {
  let remainder = value % modulus;
  return select(remainder + modulus, remainder, remainder >= 0);
}

fn adjointMaterialTextureWrapTexel(index: i32, size: i32, modeValue: f32) -> i32 {
  if (size <= 1) { return 0; }
  let mode = u32(modeValue);
  if (mode == 1u) { return clamp(index, 0, size - 1); }
  if (mode == 2u) {
    let period = 2 * size;
    let folded = adjointMaterialTexturePositiveModulo(index, period);
    return select(period - 1 - folded, folded, folded < size);
  }
  return adjointMaterialTexturePositiveModulo(index, size);
}

${ADJOINT_SOURCE_RECT_SAMPLERS_WGSL}

fn adjointMaterialTextureMipPolicy(base: u32, slot: u32) -> f32 {
  let vecIdx = base + ADJOINT_MATERIAL_TEX_MIP_POLICY + slot / 4u;
  if (vecIdx >= arrayLength(&materialTexDescriptors)) { return 2.0; }
  let packed = materialTexDescriptors[vecIdx];
  let lane = slot - (slot / 4u) * 4u;
  if (lane == 0u) { return packed.x; }
  if (lane == 1u) { return packed.y; }
  if (lane == 2u) { return packed.z; }
  return packed.w;
}

fn adjointMaterialTexturePolicyLod(lod: f32, mipCount: f32, mipPolicy: f32) -> f32 {
  let maxLod = max(mipCount - 1.0, 0.0);
  if (mipPolicy < 0.5) {
    return 0.0;
  }
  if (mipPolicy < 1.5) {
    return clamp(floor(lod + 0.5), 0.0, maxLod);
  }
  return clamp(lod, 0.0, maxLod);
}

fn adjointMaterialTextureFilterPolicy(base: u32, slot: u32) -> vec2f {
  let scalarOffset = slot * 2u;
  let vecIdx = base + ADJOINT_MATERIAL_TEX_FILTER_POLICY + scalarOffset / 4u;
  if (vecIdx >= arrayLength(&materialTexDescriptors)) { return vec2f(1.0); }
  let packed = materialTexDescriptors[vecIdx];
  let lane = scalarOffset - (scalarOffset / 4u) * 4u;
  if (lane == 0u) { return packed.xy; }
  return packed.zw;
}

fn sampleAdjointMaterialLayer(layerIdx: i32, base: u32, triIndex: u32, baryVW: vec2f, uvMetaOffset: u32, uvFitScale: vec2f, wrapMode: vec2f, mipPolicySlot: u32) -> vec4f {
  if (layerIdx < 0 || triIndex >= arrayLength(&indices) || base + uvMetaOffset + 1u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return vec4f(1.0);
  }
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let uvMeta = materialTexDescriptors[base + uvMetaOffset];
  let uvScale = materialTexDescriptors[base + uvMetaOffset + 1u];
  let gpuUvSlot = u32(uvMeta.x);
  let rawA = adjointUvForVertex(tri.x, gpuUvSlot);
  let rawB = adjointUvForVertex(tri.y, gpuUvSlot);
  let rawC = adjointUvForVertex(tri.z, gpuUvSlot);
  let rawUv = rawA * u + rawB * v + rawC * w;
  let xform = vec4f(uvMeta.y, uvMeta.z, uvScale.x, uvScale.y);
  let rot = uvMeta.w;
  let c = cos(rot);
  let s = sin(rot);
  let sx = xform.z;
  let sy = xform.w;
  let uvA = vec2f(
    sx * c * rawA.x + sx * s * rawA.y + xform.x,
    -sy * s * rawA.x + sy * c * rawA.y + xform.y,
  );
  let uvB = vec2f(
    sx * c * rawB.x + sx * s * rawB.y + xform.x,
    -sy * s * rawB.x + sy * c * rawB.y + xform.y,
  );
  let uvC = vec2f(
    sx * c * rawC.x + sx * s * rawC.y + xform.x,
    -sy * s * rawC.x + sy * c * rawC.y + xform.y,
  );
  let uv = vec2f(
    sx * c * rawUv.x + sx * s * rawUv.y + xform.x,
    -sy * s * rawUv.x + sy * c * rawUv.y + xform.y,
  );
  let sourceBaseSize = adjointMaterialTextureSourceBaseSize(
    vec2u(textureDimensions(materialTextures, 0)), uvFitScale,
  );
  let sourceMipCount = f32(adjointMaterialTextureSourceMipCount(sourceBaseSize));
  let texDim = vec2f(sourceBaseSize);
  let texelArea = max(abs((uvB.x - uvA.x) * (uvC.y - uvA.y) - (uvB.y - uvA.y) * (uvC.x - uvA.x)) * texDim.x * texDim.y, 1.0);
  let pa = positions[tri.x].xyz;
  let pb = positions[tri.y].xyz;
  let pc = positions[tri.z].xyz;
  let worldArea = 0.5 * length(cross(pb - pa, pc - pa));
  let hitPos = pa * u + pb * v + pc * w;
  let cameraDistance = max(length(hitPos - params.cameraPos.xyz), 1e-3);
  let pixelsPerMeter = 0.5 * f32(max(params.width, params.height)) / cameraDistance;
  let projectedPixels = max(sqrt(worldArea) * pixelsPerMeter, 1.0);
  let lod = clamp(log2(sqrt(texelArea) / projectedPixels), 0.0, max(sourceMipCount - 1.0, 0.0));
  let mipPolicy = adjointMaterialTextureMipPolicy(base, mipPolicySlot);
  let policyLod = adjointMaterialTexturePolicyLod(lod, sourceMipCount, mipPolicy);
  let filterPolicy = adjointMaterialTextureFilterPolicy(base, mipPolicySlot);
  let filterMode = select(filterPolicy.x, filterPolicy.y, lod > 0.0);
  return sampleAdjointSrgbSourceRect(
    layerIdx, uv, sourceBaseSize, wrapMode,
    policyLod, filterMode, mipPolicy,
  );
}

// T1-6 — emissive variant: same sampling as sampleAdjointMaterialLayer but from
// the dedicated rgba16float emissive array (already linear; sRGB decode applied
// on upload). Mirrors the sRGB fn exactly, hence the parallel definition (WGSL
// can't pass a texture as an argument) — same pattern as the *Linear variant.
fn sampleAdjointMaterialLayerEmissive(layerIdx: i32, base: u32, triIndex: u32, baryVW: vec2f, uvMetaOffset: u32, uvFitScale: vec2f, wrapMode: vec2f, mipPolicySlot: u32) -> vec4f {
  if (layerIdx < 0 || triIndex >= arrayLength(&indices) || base + uvMetaOffset + 1u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return vec4f(1.0);
  }
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let uvMeta = materialTexDescriptors[base + uvMetaOffset];
  let uvScale = materialTexDescriptors[base + uvMetaOffset + 1u];
  let gpuUvSlot = u32(uvMeta.x);
  let rawA = adjointUvForVertex(tri.x, gpuUvSlot);
  let rawB = adjointUvForVertex(tri.y, gpuUvSlot);
  let rawC = adjointUvForVertex(tri.z, gpuUvSlot);
  let rawUv = rawA * u + rawB * v + rawC * w;
  let xform = vec4f(uvMeta.y, uvMeta.z, uvScale.x, uvScale.y);
  let rot = uvMeta.w;
  let c = cos(rot);
  let s = sin(rot);
  let sx = xform.z;
  let sy = xform.w;
  let uvA = vec2f(
    sx * c * rawA.x + sx * s * rawA.y + xform.x,
    -sy * s * rawA.x + sy * c * rawA.y + xform.y,
  );
  let uvB = vec2f(
    sx * c * rawB.x + sx * s * rawB.y + xform.x,
    -sy * s * rawB.x + sy * c * rawB.y + xform.y,
  );
  let uvC = vec2f(
    sx * c * rawC.x + sx * s * rawC.y + xform.x,
    -sy * s * rawC.x + sy * c * rawC.y + xform.y,
  );
  let uv = vec2f(
    sx * c * rawUv.x + sx * s * rawUv.y + xform.x,
    -sy * s * rawUv.x + sy * c * rawUv.y + xform.y,
  );
  let sourceBaseSize = adjointMaterialTextureSourceBaseSize(
    vec2u(textureDimensions(materialTexturesEmissive, 0)), uvFitScale,
  );
  let sourceMipCount = f32(adjointMaterialTextureSourceMipCount(sourceBaseSize));
  let texDim = vec2f(sourceBaseSize);
  let texelArea = max(abs((uvB.x - uvA.x) * (uvC.y - uvA.y) - (uvB.y - uvA.y) * (uvC.x - uvA.x)) * texDim.x * texDim.y, 1.0);
  let pa = positions[tri.x].xyz;
  let pb = positions[tri.y].xyz;
  let pc = positions[tri.z].xyz;
  let worldArea = 0.5 * length(cross(pb - pa, pc - pa));
  let hitPos = pa * u + pb * v + pc * w;
  let cameraDistance = max(length(hitPos - params.cameraPos.xyz), 1e-3);
  let pixelsPerMeter = 0.5 * f32(max(params.width, params.height)) / cameraDistance;
  let projectedPixels = max(sqrt(worldArea) * pixelsPerMeter, 1.0);
  let lod = clamp(log2(sqrt(texelArea) / projectedPixels), 0.0, max(sourceMipCount - 1.0, 0.0));
  let mipPolicy = adjointMaterialTextureMipPolicy(base, mipPolicySlot);
  let policyLod = adjointMaterialTexturePolicyLod(lod, sourceMipCount, mipPolicy);
  let filterPolicy = adjointMaterialTextureFilterPolicy(base, mipPolicySlot);
  let filterMode = select(filterPolicy.x, filterPolicy.y, lod > 0.0);
  return sampleAdjointEmissiveSourceRect(
    layerIdx, uv, sourceBaseSize, wrapMode,
    policyLod, filterMode, mipPolicy,
  );
}

// Exact emissive-map multiplier for a sampled packed mesh-area point. The
// packed record retains the source triangle's selected raw UVs and original
// world-space footprint, so this uses the same transform, wrap, filter and mip
// policy as the forward NEE path instead of a triangle-average approximation.
fn sampleAdjointMeshEmitterSourceFactor(
  index: u32,
  weights: vec3f,
  worldPosition: vec3f,
) -> vec3f {
  let base = index * 7u;
  let uvAB = meshAreaLights[base + 4u];
  let uvCAndMaterial = meshAreaLights[base + 5u];
  let materialIdPlusOne = uvCAndMaterial.z;
  if (materialIdPlusOne < 0.5) { return vec3f(1.0); }
  let matId = u32(materialIdPlusOne - 1.0);
  let descriptorBase = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (descriptorBase + 13u >= arrayLength(&materialTexDescriptors)) {
    return vec3f(0.0);
  }
  let layerIdx = i32(materialTexDescriptors[descriptorBase].w);
  if (layerIdx < 0) { return vec3f(0.0); }

  let rawA = uvAB.xy;
  let rawB = uvAB.zw;
  let rawC = uvCAndMaterial.xy;
  let rawUv = rawA * weights.x + rawB * weights.y + rawC * weights.z;
  let uvMeta = materialTexDescriptors[
    descriptorBase + ADJOINT_MATERIAL_TEX_UV_EMISSIVE
  ];
  let uvScale = materialTexDescriptors[
    descriptorBase + ADJOINT_MATERIAL_TEX_UV_EMISSIVE + 1u
  ];
  let c = cos(uvMeta.w);
  let s = sin(uvMeta.w);
  let transformUv = mat2x2f(
    uvScale.x * c, -uvScale.y * s,
    uvScale.x * s, uvScale.y * c,
  );
  let offset = uvMeta.yz;
  let uvA = transformUv * rawA + offset;
  let uvB = transformUv * rawB + offset;
  let uvC = transformUv * rawC + offset;
  let uv = transformUv * rawUv + offset;
  let wrapMode = materialTexDescriptors[descriptorBase + 13u].zw;
  let uvFitScale = materialTexDescriptors[descriptorBase + 7u].zw;
  let sourceBaseSize = adjointMaterialTextureSourceBaseSize(
    vec2u(textureDimensions(materialTexturesEmissive, 0)), uvFitScale,
  );
  let sourceMipCount = f32(adjointMaterialTextureSourceMipCount(sourceBaseSize));
  let texDim = vec2f(sourceBaseSize);
  let texelArea = max(
    abs((uvB.x - uvA.x) * (uvC.y - uvA.y) -
        (uvB.y - uvA.y) * (uvC.x - uvA.x)) * texDim.x * texDim.y,
    1.0,
  );
  let worldArea = uvCAndMaterial.w;
  if (worldArea <= 0.0) { return vec3f(0.0); }
  let cameraDistance = max(
    length(worldPosition - params.cameraPos.xyz), 1e-3,
  );
  let pixelsPerMeter =
    0.5 * f32(max(params.width, params.height)) / cameraDistance;
  let projectedPixels = max(sqrt(worldArea) * pixelsPerMeter, 1.0);
  let lod = clamp(
    log2(sqrt(texelArea) / projectedPixels),
    0.0,
    max(sourceMipCount - 1.0, 0.0),
  );
  let mipPolicy = adjointMaterialTextureMipPolicy(
    descriptorBase, ADJOINT_MATERIAL_TEX_MIP_EMISSIVE,
  );
  let policyLod = adjointMaterialTexturePolicyLod(
    lod, sourceMipCount, mipPolicy,
  );
  let filterPolicy = adjointMaterialTextureFilterPolicy(
    descriptorBase, ADJOINT_MATERIAL_TEX_MIP_EMISSIVE,
  );
  let filterMode = select(filterPolicy.x, filterPolicy.y, lod > 0.0);
  return sampleAdjointEmissiveSourceRect(
    layerIdx, uv, sourceBaseSize, wrapMode,
    policyLod, filterMode, mipPolicy,
  ).rgb;
}

fn sampleAdjointMaterialLayerLinear(layerIdx: i32, base: u32, triIndex: u32, baryVW: vec2f, uvMetaOffset: u32, uvFitScale: vec2f, wrapMode: vec2f, mipPolicySlot: u32) -> vec4f {
  if (layerIdx < 0 || triIndex >= arrayLength(&indices) || base + uvMetaOffset + 1u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return vec4f(1.0);
  }
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let uvMeta = materialTexDescriptors[base + uvMetaOffset];
  let uvScale = materialTexDescriptors[base + uvMetaOffset + 1u];
  let gpuUvSlot = u32(uvMeta.x);
  let rawA = adjointUvForVertex(tri.x, gpuUvSlot);
  let rawB = adjointUvForVertex(tri.y, gpuUvSlot);
  let rawC = adjointUvForVertex(tri.z, gpuUvSlot);
  let rawUv = rawA * u + rawB * v + rawC * w;
  let xform = vec4f(uvMeta.y, uvMeta.z, uvScale.x, uvScale.y);
  let rot = uvMeta.w;
  let c = cos(rot);
  let s = sin(rot);
  let sx = xform.z;
  let sy = xform.w;
  let uvA = vec2f(
    sx * c * rawA.x + sx * s * rawA.y + xform.x,
    -sy * s * rawA.x + sy * c * rawA.y + xform.y,
  );
  let uvB = vec2f(
    sx * c * rawB.x + sx * s * rawB.y + xform.x,
    -sy * s * rawB.x + sy * c * rawB.y + xform.y,
  );
  let uvC = vec2f(
    sx * c * rawC.x + sx * s * rawC.y + xform.x,
    -sy * s * rawC.x + sy * c * rawC.y + xform.y,
  );
  let uv = vec2f(
    sx * c * rawUv.x + sx * s * rawUv.y + xform.x,
    -sy * s * rawUv.x + sy * c * rawUv.y + xform.y,
  );
  let sourceBaseSize = adjointMaterialTextureSourceBaseSize(
    vec2u(textureDimensions(materialTexturesLinear, 0)), uvFitScale,
  );
  let sourceMipCount = f32(adjointMaterialTextureSourceMipCount(sourceBaseSize));
  let texDim = vec2f(sourceBaseSize);
  let texelArea = max(abs((uvB.x - uvA.x) * (uvC.y - uvA.y) - (uvB.y - uvA.y) * (uvC.x - uvA.x)) * texDim.x * texDim.y, 1.0);
  let pa = positions[tri.x].xyz;
  let pb = positions[tri.y].xyz;
  let pc = positions[tri.z].xyz;
  let worldArea = 0.5 * length(cross(pb - pa, pc - pa));
  let hitPos = pa * u + pb * v + pc * w;
  let cameraDistance = max(length(hitPos - params.cameraPos.xyz), 1e-3);
  let pixelsPerMeter = 0.5 * f32(max(params.width, params.height)) / cameraDistance;
  let projectedPixels = max(sqrt(worldArea) * pixelsPerMeter, 1.0);
  let lod = clamp(log2(sqrt(texelArea) / projectedPixels), 0.0, max(sourceMipCount - 1.0, 0.0));
  let mipPolicy = adjointMaterialTextureMipPolicy(base, mipPolicySlot);
  let policyLod = adjointMaterialTexturePolicyLod(lod, sourceMipCount, mipPolicy);
  let filterPolicy = adjointMaterialTextureFilterPolicy(base, mipPolicySlot);
  let filterMode = select(filterPolicy.x, filterPolicy.y, lod > 0.0);
  return sampleAdjointLinearSourceRect(
    layerIdx, uv, sourceBaseSize, wrapMode,
    policyLod, filterMode, mipPolicy,
  );
}

fn sampleAdjointMaterialLayerLinearRawUvPolicy(layerIdx: i32, base: u32, triIndex: u32, baryVW: vec2f, rawUv: vec2f, uvMetaOffset: u32, uvFitScale: vec2f, wrapMode: vec2f, mipPolicySlot: u32) -> vec4f {
  if (layerIdx < 0 || triIndex >= arrayLength(&indices) || base + uvMetaOffset + 1u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs) ||
      tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return vec4f(1.0);
  }
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let uvMeta = materialTexDescriptors[base + uvMetaOffset];
  let uvScale = materialTexDescriptors[base + uvMetaOffset + 1u];
  let gpuUvSlot = u32(uvMeta.x);
  let xform = vec4f(uvMeta.y, uvMeta.z, uvScale.x, uvScale.y);
  let rot = uvMeta.w;
  let c = cos(rot);
  let s = sin(rot);
  let sx = xform.z;
  let sy = xform.w;
  let rawA = adjointUvForVertex(tri.x, gpuUvSlot);
  let rawB = adjointUvForVertex(tri.y, gpuUvSlot);
  let rawC = adjointUvForVertex(tri.z, gpuUvSlot);
  let uvA = vec2f(
    sx * c * rawA.x + sx * s * rawA.y + xform.x,
    -sy * s * rawA.x + sy * c * rawA.y + xform.y,
  );
  let uvB = vec2f(
    sx * c * rawB.x + sx * s * rawB.y + xform.x,
    -sy * s * rawB.x + sy * c * rawB.y + xform.y,
  );
  let uvC = vec2f(
    sx * c * rawC.x + sx * s * rawC.y + xform.x,
    -sy * s * rawC.x + sy * c * rawC.y + xform.y,
  );
  let uv = vec2f(
    sx * c * rawUv.x + sx * s * rawUv.y + xform.x,
    -sy * s * rawUv.x + sy * c * rawUv.y + xform.y,
  );
  let sourceBaseSize = adjointMaterialTextureSourceBaseSize(
    vec2u(textureDimensions(materialTexturesLinear, 0)), uvFitScale,
  );
  let sourceMipCount = f32(adjointMaterialTextureSourceMipCount(sourceBaseSize));
  let texDim = vec2f(sourceBaseSize);
  let texelArea = max(abs((uvB.x - uvA.x) * (uvC.y - uvA.y) - (uvB.y - uvA.y) * (uvC.x - uvA.x)) * texDim.x * texDim.y, 1.0);
  let pa = positions[tri.x].xyz;
  let pb = positions[tri.y].xyz;
  let pc = positions[tri.z].xyz;
  let worldArea = 0.5 * length(cross(pb - pa, pc - pa));
  let hitPos = pa * u + pb * v + pc * w;
  let cameraDistance = max(length(hitPos - params.cameraPos.xyz), 1e-3);
  let pixelsPerMeter = 0.5 * f32(max(params.width, params.height)) / cameraDistance;
  let projectedPixels = max(sqrt(worldArea) * pixelsPerMeter, 1.0);
  let lod = clamp(log2(sqrt(texelArea) / projectedPixels), 0.0, max(sourceMipCount - 1.0, 0.0));
  let mipPolicy = adjointMaterialTextureMipPolicy(base, mipPolicySlot);
  let policyLod = adjointMaterialTexturePolicyLod(lod, sourceMipCount, mipPolicy);
  let filterPolicy = adjointMaterialTextureFilterPolicy(base, mipPolicySlot);
  let filterMode = select(filterPolicy.x, filterPolicy.y, lod > 0.0);
  return sampleAdjointLinearSourceRect(
    layerIdx, uv, sourceBaseSize, wrapMode,
    policyLod, filterMode, mipPolicy,
  );
}

// T1-6 — samples the dedicated rgba16float emissive array (emissiveIdx indexes
// that array's layer space, matching the forward sampleEmissiveTexture path).
fn sampleAdjointEmissiveTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> vec4f {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 13u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  return sampleAdjointMaterialLayerEmissive(
    i32(materialTexDescriptors[base].w),
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_EMISSIVE,
    materialTexDescriptors[base + 7u].zw,
    materialTexDescriptors[base + 13u].zw,
    ADJOINT_MATERIAL_TEX_MIP_EMISSIVE,
  );
}

fn sampleAdjointBaseColorTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> vec4f {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 13u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  return sampleAdjointMaterialLayer(
    i32(materialTexDescriptors[base].x),
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_BASE_COLOR,
    materialTexDescriptors[base + 7u].xy,
    materialTexDescriptors[base + 13u].xy,
    ADJOINT_MATERIAL_TEX_MIP_BASE_COLOR,
  );
}

fn sampleAdjointVertexColor(triIndex: u32, baryVW: vec2f) -> vec4f {
  if (triIndex >= arrayLength(&indices)) { return vec4f(1.0); }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshVertexColors) || tri.y >= arrayLength(&meshVertexColors) || tri.z >= arrayLength(&meshVertexColors)) {
    return vec4f(1.0);
  }
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  return meshVertexColors[tri.x] * u + meshVertexColors[tri.y] * v + meshVertexColors[tri.z] * w;
}

struct AdjointTangentFrame {
  tangent: vec3f,
  bitangent: vec3f,
  valid: bool,
}

fn buildAdjointTangentFrame(triIndex: u32, baryVW: vec2f, normal: vec3f, gpuUvSlot: u32) -> AdjointTangentFrame {
  var frame: AdjointTangentFrame;
  frame.valid = false;
  if (triIndex >= arrayLength(&indices)) { return frame; }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs) ||
      tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return frame;
  }
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
      tangent = tangent - normal * dot(normal, tangent);
      let tlen = length(tangent);
      if (tlen > 1e-8) {
        tangent = tangent / tlen;
        let handedness = select(-1.0, 1.0, handednessRaw >= 0.0);
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
  let uv0 = adjointUvForVertex(tri.x, gpuUvSlot);
  let duv1 = adjointUvForVertex(tri.y, gpuUvSlot) - uv0;
  let duv2 = adjointUvForVertex(tri.z, gpuUvSlot) - uv0;
  let det = duv1.x * duv2.y - duv2.x * duv1.y;
  if (abs(det) < 1e-10) { return frame; }
  var tangent = (duv2.y * e1 - duv1.y * e2) / det;
  var bitangent = (-duv2.x * e1 + duv1.x * e2) / det;
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

struct AdjointAnisotropyOrientation {
  angle: f32,
  localRotationDerivative: f32,
}

fn adjointAnisotropyOrientation(
  matId: u32,
  triIndex: u32,
  baryVW: vec2f,
  normal: vec3f,
  localRotation: f32,
) -> AdjointAnisotropyOrientation {
  var out: AdjointAnisotropyOrientation;
  out.angle = localRotation;
  out.localRotationDerivative = 1.0;
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 17u >= arrayLength(&materialTexDescriptors)) { return out; }
  var gpuUvSlot = 0u;
  if (i32(materialTexDescriptors[base + 5u].z) >= 0) {
    gpuUvSlot = u32(
      materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_UV_ANISOTROPY].x,
    );
  }
  let frame = buildAdjointTangentFrame(triIndex, baryVW, normal, gpuUvSlot);
  if (!frame.valid) { return out; }
  let canonicalT = adjointBuildTangent(normal);
  let canonicalB = cross(normal, canonicalT);
  let authoredDirection =
    cos(localRotation) * frame.tangent + sin(localRotation) * frame.bitangent;
  out.angle = atan2(
    dot(authoredDirection, canonicalB),
    dot(authoredDirection, canonicalT),
  );
  out.localRotationDerivative = select(
    -1.0, 1.0, dot(cross(frame.tangent, frame.bitangent), normal) >= 0.0,
  );
  return out;
}

struct AdjointNormalMapSample {
  normal: vec3f,
  dNormal_dScale: vec3f,
}

fn sampleAdjointNormalMap(matId: u32, triIndex: u32, baryVW: vec2f, geomNormal: vec3f) -> AdjointNormalMapSample {
  var out: AdjointNormalMapSample;
  out.normal = geomNormal;
  out.dNormal_dScale = vec3f(0.0);
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 14u >= arrayLength(&materialTexDescriptors)) { return out; }
  let normalIdx = i32(materialTexDescriptors[base].y);
  if (normalIdx < 0) { return out; }
  let normalGpuUvSlot = u32(
    materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_UV_NORMAL].x,
  );
  let frame = buildAdjointTangentFrame(triIndex, baryVW, geomNormal, normalGpuUvSlot);
  if (!frame.valid) { return out; }
  let texel = sampleAdjointMaterialLayerLinear(
    normalIdx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_NORMAL,
    materialTexDescriptors[base + 8u].xy,
    materialTexDescriptors[base + 14u].xy,
    ADJOINT_MATERIAL_TEX_MIP_NORMAL,
  ).xyz;
  let xy = texel.xy * 2.0 - vec2f(1.0);
  let z = texel.z * 2.0 - 1.0;
  let normalScale = materialTexDescriptors[base + 5u].w;
  let perturbed = frame.tangent * (xy.x * normalScale) +
    frame.bitangent * (xy.y * normalScale) +
    geomNormal * z;
  let plen = length(perturbed);
  if (plen <= 1e-6) { return out; }
  let n = perturbed / plen;
  let dPerturbed_dScale = frame.tangent * xy.x + frame.bitangent * xy.y;
  out.normal = n;
  out.dNormal_dScale = (dPerturbed_dScale - n * dot(n, dPerturbed_dScale)) / plen;
  return out;
}

struct AdjointClearcoatNormalMapSample {
  normal: vec3f,
  dNormal_dScale: vec3f,
}

fn sampleAdjointClearcoatNormalMap(matId: u32, triIndex: u32, baryVW: vec2f, clearcoatNormal: vec3f) -> AdjointClearcoatNormalMapSample {
  var out: AdjointClearcoatNormalMapSample;
  out.normal = clearcoatNormal;
  out.dNormal_dScale = vec3f(0.0);
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_NORMAL + 1u >= arrayLength(&materialTexDescriptors)) { return out; }
  let clearcoatNormalIdx = i32(materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL].x);
  if (clearcoatNormalIdx < 0) { return out; }
  let clearcoatNormalGpuUvSlot = u32(
    materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_NORMAL].x,
  );
  let frame = buildAdjointTangentFrame(
    triIndex, baryVW, clearcoatNormal, clearcoatNormalGpuUvSlot,
  );
  if (!frame.valid) { return out; }
  let texel = sampleAdjointMaterialLayerLinear(
    clearcoatNormalIdx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_NORMAL,
    materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL].zw,
    materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP].xy,
    ADJOINT_MATERIAL_TEX_MIP_CLEARCOAT_NORMAL,
  ).xyz;
  let xy = texel.xy * 2.0 - vec2f(1.0);
  let z = texel.z * 2.0 - 1.0;
  let clearcoatNormalScale = materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL].y;
  let perturbed = frame.tangent * (xy.x * clearcoatNormalScale) +
    frame.bitangent * (xy.y * clearcoatNormalScale) +
    clearcoatNormal * z;
  let plen = length(perturbed);
  if (plen <= 1e-6) { return out; }
  let n = perturbed / plen;
  let dPerturbed_dScale = frame.tangent * xy.x + frame.bitangent * xy.y;
  out.normal = n;
  out.dNormal_dScale = (dPerturbed_dScale - n * dot(n, dPerturbed_dScale)) / plen;
  return out;
}

struct AdjointBumpMapSample {
  normal: vec3f,
  dNormal_dScale: vec3f,
}

fn sampleAdjointBumpMap(matId: u32, triIndex: u32, baryVW: vec2f, shadingNormal: vec3f) -> AdjointBumpMapSample {
  var out: AdjointBumpMapSample;
  out.normal = shadingNormal;
  out.dNormal_dScale = vec3f(0.0);
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 16u >= arrayLength(&materialTexDescriptors) || triIndex >= arrayLength(&indices)) { return out; }
  let bumpIdx = i32(materialTexDescriptors[base + 3u].w);
  if (bumpIdx < 0) { return out; }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs) ||
      tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return out;
  }
  let bumpUvFitScale = materialTexDescriptors[base + 10u].zw;
  let bumpWrapMode = materialTexDescriptors[base + 16u].zw;
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let uvMeta = materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_UV_BUMP];
  let gpuUvSlot = u32(uvMeta.x);
  let frame = buildAdjointTangentFrame(triIndex, baryVW, shadingNormal, gpuUvSlot);
  if (!frame.valid) { return out; }
  let rawUv =
    adjointUvForVertex(tri.x, gpuUvSlot) * u +
    adjointUvForVertex(tri.y, gpuUvSlot) * v +
    adjointUvForVertex(tri.z, gpuUvSlot) * w;
  let linearDims = vec2f(textureDimensions(materialTexturesLinear, 0));
  let sourceDims = max(linearDims * bumpUvFitScale, vec2f(1.0));
  let texelStep = vec2f(1.0 / sourceDims.x, 1.0 / sourceDims.y);
  let hC = sampleAdjointMaterialLayerLinearRawUvPolicy(bumpIdx, base, triIndex, baryVW, rawUv, ADJOINT_MATERIAL_TEX_UV_BUMP, bumpUvFitScale, bumpWrapMode, ADJOINT_MATERIAL_TEX_MIP_BUMP).r;
  let hU = sampleAdjointMaterialLayerLinearRawUvPolicy(bumpIdx, base, triIndex, baryVW, rawUv + vec2f(texelStep.x, 0.0), ADJOINT_MATERIAL_TEX_UV_BUMP, bumpUvFitScale, bumpWrapMode, ADJOINT_MATERIAL_TEX_MIP_BUMP).r;
  let hV = sampleAdjointMaterialLayerLinearRawUvPolicy(bumpIdx, base, triIndex, baryVW, rawUv + vec2f(0.0, texelStep.y), ADJOINT_MATERIAL_TEX_UV_BUMP, bumpUvFitScale, bumpWrapMode, ADJOINT_MATERIAL_TEX_MIP_BUMP).r;
  let dhdu = (hU - hC) / texelStep.x;
  let dhdv = (hV - hC) / texelStep.y;
  let gradient = dhdu * frame.tangent + dhdv * frame.bitangent;
  let bumpScale = materialTexDescriptors[base + 4u].z;
  let perturbed = shadingNormal - bumpScale * gradient;
  let plen = length(perturbed);
  if (plen <= 1e-6) { return out; }
  let n = perturbed / plen;
  let dPerturbed_dScale = -gradient;
  out.normal = n;
  out.dNormal_dScale = (dPerturbed_dScale - n * dot(n, dPerturbed_dScale)) / plen;
  return out;
}

struct AdjointNormalStackSample {
  normal: vec3f,
  clearcoatNormal: vec3f,
}

fn adjointMaterialNormalScale(matId: u32) -> f32 {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 5u >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  return materialTexDescriptors[base + 5u].w;
}

fn adjointMaterialBumpScale(matId: u32) -> f32 {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 4u >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  return materialTexDescriptors[base + 4u].z;
}

fn adjointMaterialClearcoatNormalScale(matId: u32) -> f32 {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  return materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL].y;
}

fn sampleAdjointNormalMapWithScale(
  matId: u32,
  triIndex: u32,
  baryVW: vec2f,
  geomNormal: vec3f,
  normalScale: f32,
) -> vec3f {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 14u >= arrayLength(&materialTexDescriptors)) { return geomNormal; }
  let normalIdx = i32(materialTexDescriptors[base].y);
  if (normalIdx < 0) { return geomNormal; }
  let normalGpuUvSlot = u32(
    materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_UV_NORMAL].x,
  );
  let frame = buildAdjointTangentFrame(triIndex, baryVW, geomNormal, normalGpuUvSlot);
  if (!frame.valid) { return geomNormal; }
  let texel = sampleAdjointMaterialLayerLinear(
    normalIdx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_NORMAL,
    materialTexDescriptors[base + 8u].xy,
    materialTexDescriptors[base + 14u].xy,
    ADJOINT_MATERIAL_TEX_MIP_NORMAL,
  ).xyz;
  let xy = texel.xy * 2.0 - vec2f(1.0);
  let z = texel.z * 2.0 - 1.0;
  let perturbed = frame.tangent * (xy.x * normalScale) +
    frame.bitangent * (xy.y * normalScale) +
    geomNormal * z;
  let plen = length(perturbed);
  if (plen <= 1e-6) { return geomNormal; }
  return perturbed / plen;
}

fn sampleAdjointBumpMapWithScale(
  matId: u32,
  triIndex: u32,
  baryVW: vec2f,
  shadingNormal: vec3f,
  bumpScale: f32,
) -> vec3f {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 16u >= arrayLength(&materialTexDescriptors) || triIndex >= arrayLength(&indices)) { return shadingNormal; }
  let bumpIdx = i32(materialTexDescriptors[base + 3u].w);
  if (bumpIdx < 0) { return shadingNormal; }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs) ||
      tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return shadingNormal;
  }
  let bumpUvFitScale = materialTexDescriptors[base + 10u].zw;
  let bumpWrapMode = materialTexDescriptors[base + 16u].zw;
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let uvMeta = materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_UV_BUMP];
  let gpuUvSlot = u32(uvMeta.x);
  let frame = buildAdjointTangentFrame(triIndex, baryVW, shadingNormal, gpuUvSlot);
  if (!frame.valid) { return shadingNormal; }
  let rawUv =
    adjointUvForVertex(tri.x, gpuUvSlot) * u +
    adjointUvForVertex(tri.y, gpuUvSlot) * v +
    adjointUvForVertex(tri.z, gpuUvSlot) * w;
  let linearDims = vec2f(textureDimensions(materialTexturesLinear, 0));
  let sourceDims = max(linearDims * bumpUvFitScale, vec2f(1.0));
  let texelStep = vec2f(1.0 / sourceDims.x, 1.0 / sourceDims.y);
  let hC = sampleAdjointMaterialLayerLinearRawUvPolicy(bumpIdx, base, triIndex, baryVW, rawUv, ADJOINT_MATERIAL_TEX_UV_BUMP, bumpUvFitScale, bumpWrapMode, ADJOINT_MATERIAL_TEX_MIP_BUMP).r;
  let hU = sampleAdjointMaterialLayerLinearRawUvPolicy(bumpIdx, base, triIndex, baryVW, rawUv + vec2f(texelStep.x, 0.0), ADJOINT_MATERIAL_TEX_UV_BUMP, bumpUvFitScale, bumpWrapMode, ADJOINT_MATERIAL_TEX_MIP_BUMP).r;
  let hV = sampleAdjointMaterialLayerLinearRawUvPolicy(bumpIdx, base, triIndex, baryVW, rawUv + vec2f(0.0, texelStep.y), ADJOINT_MATERIAL_TEX_UV_BUMP, bumpUvFitScale, bumpWrapMode, ADJOINT_MATERIAL_TEX_MIP_BUMP).r;
  let dhdu = (hU - hC) / texelStep.x;
  let dhdv = (hV - hC) / texelStep.y;
  let gradient = dhdu * frame.tangent + dhdv * frame.bitangent;
  let perturbed = shadingNormal - bumpScale * gradient;
  let plen = length(perturbed);
  if (plen <= 1e-6) { return shadingNormal; }
  return perturbed / plen;
}

fn sampleAdjointClearcoatNormalMapWithScale(
  matId: u32,
  triIndex: u32,
  baryVW: vec2f,
  clearcoatNormal: vec3f,
  clearcoatNormalScale: f32,
) -> vec3f {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_NORMAL + 1u >= arrayLength(&materialTexDescriptors)) { return clearcoatNormal; }
  let clearcoatNormalIdx = i32(materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL].x);
  if (clearcoatNormalIdx < 0) { return clearcoatNormal; }
  let clearcoatNormalGpuUvSlot = u32(
    materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_NORMAL].x,
  );
  let frame = buildAdjointTangentFrame(
    triIndex, baryVW, clearcoatNormal, clearcoatNormalGpuUvSlot,
  );
  if (!frame.valid) { return clearcoatNormal; }
  let texel = sampleAdjointMaterialLayerLinear(
    clearcoatNormalIdx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_NORMAL,
    materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL].zw,
    materialTexDescriptors[base + ADJOINT_MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP].xy,
    ADJOINT_MATERIAL_TEX_MIP_CLEARCOAT_NORMAL,
  ).xyz;
  let xy = texel.xy * 2.0 - vec2f(1.0);
  let z = texel.z * 2.0 - 1.0;
  let perturbed = frame.tangent * (xy.x * clearcoatNormalScale) +
    frame.bitangent * (xy.y * clearcoatNormalScale) +
    clearcoatNormal * z;
  let plen = length(perturbed);
  if (plen <= 1e-6) { return clearcoatNormal; }
  return perturbed / plen;
}

fn sampleAdjointNormalStackWithScales(
  matId: u32,
  triIndex: u32,
  baryVW: vec2f,
  geomNormal: vec3f,
  normalScale: f32,
  bumpScale: f32,
  clearcoatNormalScale: f32,
) -> AdjointNormalStackSample {
  var out: AdjointNormalStackSample;
  let normalMapped = sampleAdjointNormalMapWithScale(matId, triIndex, baryVW, geomNormal, normalScale);
  out.normal = sampleAdjointBumpMapWithScale(matId, triIndex, baryVW, normalMapped, bumpScale);
  out.clearcoatNormal = sampleAdjointClearcoatNormalMapWithScale(
    matId,
    triIndex,
    baryVW,
    out.normal,
    clearcoatNormalScale,
  );
  return out;
}

fn sampleAdjointOrmTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> vec4f {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 15u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  let roughness = sampleAdjointMaterialLayerLinear(
    i32(materialTexDescriptors[base].z),
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_ROUGHNESS,
    materialTexDescriptors[base + 8u].zw,
    materialTexDescriptors[base + 14u].zw,
    ADJOINT_MATERIAL_TEX_MIP_ROUGHNESS,
  ).g;
  let metallic = sampleAdjointMaterialLayerLinear(
    i32(materialTexDescriptors[base + 6u].z),
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_METALLIC,
    materialTexDescriptors[base + 9u].xy,
    materialTexDescriptors[base + 15u].xy,
    ADJOINT_MATERIAL_TEX_MIP_METALLIC,
  ).b;
  return vec4f(1.0, roughness, metallic, 1.0);
}

struct AdjointAoSample {
  factor: f32,
  dFactor_dIntensity: f32,
}

fn sampleAdjointAo(matId: u32, triIndex: u32, baryVW: vec2f) -> AdjointAoSample {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 15u >= arrayLength(&materialTexDescriptors)) {
    return AdjointAoSample(1.0, 0.0);
  }
  let idx = i32(materialTexDescriptors[base + 3u].y);
  if (idx < 0) {
    return AdjointAoSample(1.0, 0.0);
  }
  let intensity = clamp(materialTexDescriptors[base + 4u].x, 0.0, 1.0);
  let r = clamp(sampleAdjointMaterialLayerLinear(
    idx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_AO,
    materialTexDescriptors[base + 9u].zw,
    materialTexDescriptors[base + 15u].zw,
    ADJOINT_MATERIAL_TEX_MIP_AO,
  ).r, 0.0, 1.0);
  return AdjointAoSample(mix(1.0, r, intensity), r - 1.0);
}

fn sampleAdjointLightMapRadiancePerUnitIntensity(matId: u32, triIndex: u32, baryVW: vec2f) -> vec3f {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 16u >= arrayLength(&materialTexDescriptors)) { return vec3f(0.0); }
  let idx = i32(materialTexDescriptors[base + 3u].z);
  if (idx < 0) { return vec3f(0.0); }
  return sampleAdjointMaterialLayerLinear(
    idx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_LIGHT,
    materialTexDescriptors[base + 10u].xy,
    materialTexDescriptors[base + 16u].xy,
    ADJOINT_MATERIAL_TEX_MIP_LIGHT,
  ).rgb;
}

fn sampleAdjointClearcoatTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> f32 {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 66u >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  let idx = i32(materialTexDescriptors[base + 41u].x);
  if (idx < 0) { return 1.0; }
  return clamp(sampleAdjointMaterialLayerLinear(
    idx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_CLEARCOAT,
    materialTexDescriptors[base + 43u].xy,
    materialTexDescriptors[base + 47u].xy,
    ADJOINT_MATERIAL_TEX_MIP_CLEARCOAT,
  ).r, 0.0, 1.0);
}

fn sampleAdjointClearcoatRoughnessTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> f32 {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 66u >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  let idx = i32(materialTexDescriptors[base + 41u].y);
  if (idx < 0) { return 1.0; }
  return clamp(sampleAdjointMaterialLayerLinear(
    idx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_ROUGHNESS,
    materialTexDescriptors[base + 43u].zw,
    materialTexDescriptors[base + 47u].zw,
    ADJOINT_MATERIAL_TEX_MIP_CLEARCOAT_ROUGHNESS,
  ).g, 0.0, 1.0);
}

fn sampleAdjointSheenColorTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> vec3f {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 66u >= arrayLength(&materialTexDescriptors)) { return vec3f(1.0); }
  let idx = i32(materialTexDescriptors[base + 41u].z);
  if (idx < 0) { return vec3f(1.0); }
  return clamp(sampleAdjointMaterialLayer(
    idx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_SHEEN_COLOR,
    materialTexDescriptors[base + 44u].xy,
    materialTexDescriptors[base + 48u].xy,
    ADJOINT_MATERIAL_TEX_MIP_SHEEN_COLOR,
  ).rgb, vec3f(0.0), vec3f(1.0));
}

fn sampleAdjointSheenRoughnessTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> f32 {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 66u >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  let idx = i32(materialTexDescriptors[base + 41u].w);
  if (idx < 0) { return 1.0; }
  return clamp(sampleAdjointMaterialLayerLinear(
    idx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_SHEEN_ROUGHNESS,
    materialTexDescriptors[base + 44u].zw,
    materialTexDescriptors[base + 48u].zw,
    ADJOINT_MATERIAL_TEX_MIP_SHEEN_ROUGHNESS,
  ).a, 0.0, 1.0);
}

fn sampleAdjointIridescenceTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> f32 {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 66u >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  let idx = i32(materialTexDescriptors[base + 42u].x);
  if (idx < 0) { return 1.0; }
  return clamp(sampleAdjointMaterialLayerLinear(
    idx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE,
    materialTexDescriptors[base + 45u].xy,
    materialTexDescriptors[base + 49u].xy,
    ADJOINT_MATERIAL_TEX_MIP_IRIDESCENCE,
  ).r, 0.0, 1.0);
}

fn sampleAdjointIridescenceThicknessTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> f32 {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 66u >= arrayLength(&materialTexDescriptors)) { return -1.0; }
  let idx = i32(materialTexDescriptors[base + 42u].y);
  if (idx < 0) { return -1.0; }
  return clamp(sampleAdjointMaterialLayerLinear(
    idx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE_THICKNESS,
    materialTexDescriptors[base + 45u].zw,
    materialTexDescriptors[base + 49u].zw,
    ADJOINT_MATERIAL_TEX_MIP_IRIDESCENCE_THICKNESS,
  ).g, 0.0, 1.0);
}

fn sampleAdjointSpecularColorTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> vec3f {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 66u >= arrayLength(&materialTexDescriptors)) { return vec3f(1.0); }
  let idx = i32(materialTexDescriptors[base + 42u].z);
  if (idx < 0) { return vec3f(1.0); }
  return clamp(sampleAdjointMaterialLayer(
    idx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_SPECULAR_COLOR,
    materialTexDescriptors[base + 46u].xy,
    materialTexDescriptors[base + 50u].xy,
    ADJOINT_MATERIAL_TEX_MIP_SPECULAR_COLOR,
  ).rgb, vec3f(0.0), vec3f(1.0));
}

fn sampleAdjointSpecularIntensityTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> f32 {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 66u >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  let idx = i32(materialTexDescriptors[base + 42u].w);
  if (idx < 0) { return 1.0; }
  return clamp(sampleAdjointMaterialLayerLinear(
    idx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_SPECULAR_INTENSITY,
    materialTexDescriptors[base + 46u].zw,
    materialTexDescriptors[base + 50u].zw,
    ADJOINT_MATERIAL_TEX_MIP_SPECULAR_INTENSITY,
  ).a, 0.0, 1.0);
}

fn sampleAdjointAnisotropyTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> AdjointAnisotropyMapSample {
  var out: AdjointAnisotropyMapSample;
  out.strength = 1.0;
  out.rotationOffset = 0.0;
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 17u >= arrayLength(&materialTexDescriptors)) { return out; }
  let idx = i32(materialTexDescriptors[base + 5u].z);
  if (idx < 0) { return out; }
  let texel = sampleAdjointMaterialLayerLinear(
    idx,
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_ANISOTROPY,
    materialTexDescriptors[base + 11u].xy,
    materialTexDescriptors[base + 17u].xy,
    ADJOINT_MATERIAL_TEX_MIP_ANISOTROPY,
  );
  let rg = texel.rg * 2.0 - vec2f(1.0);
  out.strength = clamp(texel.b, 0.0, 1.0);
  out.rotationOffset = atan2(rg.y, rg.x);
  return out;
}

// Mirror of kernelCore.wgsl.ts concentricDiscSample for adjoint-pass standalone
// composition. The caller remaps xi from [0,1]² to [-1,1]².
fn adjointConcentricDiscSample(xi: vec2f) -> vec2f {
  let a = xi.x;
  let b = xi.y;
  if (a == 0.0 && b == 0.0) {
    return vec2f(0.0);
  }
  var cr: f32;
  var cphi: f32;
  if (abs(a) >= abs(b)) {
    cr = a;
    cphi = (PI / 4.0) * (b / a);
  } else {
    cr = b;
    cphi = (PI / 2.0) - (PI / 4.0) * (a / b);
  }
  return vec2f(cr * cos(cphi), cr * sin(cphi));
}

// ── the GPU-validated BRDF partials + adjointScatter (gradAccum at binding 8) ──
${PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL}

fn adjointLuminance(v: vec3f) -> f32 {
  return dot(v, vec3f(0.2126, 0.7152, 0.0722));
}

fn adjointPowerHeuristic(pdfA: f32, pdfB: f32) -> f32 {
  let a2 = pdfA * pdfA;
  let b2 = pdfB * pdfB;
  let denominator = a2 + b2;
  if (!(denominator > 0.0) || !(denominator == denominator) ||
      abs(denominator) > 3.402823e38) { return 0.0; }
  return a2 / denominator;
}

fn adjointClearcoatPdf(clearcoat: f32, clearcoatRoughness: f32, normal: vec3f, wo: vec3f, wi: vec3f) -> f32 {
  if (!(clearcoat > 0.0)) { return 0.0; }
  let nDotV = max(dot(normal, wo), 0.0);
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotV <= 1e-5 || nDotL <= 1e-5) { return 0.0; }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let alpha = max(clearcoatRoughness * clearcoatRoughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let d = ggxD(nDotH, alpha);
  let g1Wo = smithG1(nDotV, clearcoatRoughness);
  return (d * g1Wo) / max(4.0 * nDotV, 1e-6);
}

fn adjointCharliePdfD(nDotH: f32, alpha: f32) -> f32 {
  let invAlpha = 1.0 / max(alpha, 1e-4);
  let sinThetaH = sqrt(max(0.0, 1.0 - nDotH * nDotH));
  return (2.0 + invAlpha) * pow(sinThetaH, invAlpha) / (2.0 * PI);
}

fn adjointCharlieSheenPdf(sheen: f32, sheenRoughness: f32, normal: vec3f, wo: vec3f, wi: vec3f) -> f32 {
  if (!(sheen > 0.0)) { return 0.0; }
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return 0.0; }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 1e-6);
  let alpha = max(sheenRoughness * sheenRoughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  return (adjointCharliePdfD(nDotH, alpha) * nDotH) / max(4.0 * vDotH, 1e-6);
}

fn adjointBrdfAnisotropicSpecPdf(
  roughness: f32,
  anisotropy: f32,
  normal: vec3f,
  tangent: vec3f,
  bitangent: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> f32 {
  let nDotV = max(dot(normal, wo), 1e-6);
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) { return 0.0; }
  let h = safe_normalize(wo + wi);
  let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let aspect = sqrt(max(1.0 - 0.9 * anisotropy, 1e-4));
  let ax = max(alpha / aspect, 1e-4);
  let ay = max(alpha * aspect, 1e-4);
  let hT = dot(h, tangent);
  let hB = dot(h, bitangent);
  let hN = max(dot(h, normal), 0.0);
  let woT = dot(wo, tangent);
  let woB = dot(wo, bitangent);
  let woN = max(dot(wo, normal), 1e-6);
  let d = adjointGgxDAnis(hT, hB, hN, ax, ay);
  let g1 = adjointSmithG1Anis(woT, woB, woN, ax, ay);
  return (d * g1) / max(4.0 * nDotV, 1e-6);
}

fn adjointBrdfDirectionalPdfFullSampledWithClearcoatNormal(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> f32 {
  let wiDotN = dot(normal, wi);
  let woDotN = dot(normal, wo);
  let nDotV = max(woDotN, 0.0);
  if (nDotV <= 1e-5 || wiDotN * woDotN <= 0.0) { return 0.0; }
  let h = safe_normalize(wi + wo);
  let vDotH = max(dot(wo, h), 1e-6);
  let f0 = adjointMaterialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  let fresnel = fresnelSchlick(vDotH, f0);
  let baseSpecProb = clamp(mix(0.04, 0.96, max(adjointLuminance(fresnel), metallic)), 0.04, 0.96);
  let baseDiffProb = max(0.0, 1.0 - metallic);
  let sumProb = baseSpecProb + baseDiffProb;
  let specProb = baseSpecProb / sumProb;
  let diffProb = baseDiffProb / sumProb;
  let nDotL = max(wiDotN, 0.0);
  if (nDotL <= 1e-5) { return 0.0; }
  var pdfSpec: f32;
  if (anisotropy > 0.0) {
    let tanT = adjointBuildTangent(normal);
    let tanB = cross(normal, tanT);
    let c = cos(anisotropyRotation);
    let s = sin(anisotropyRotation);
    let anisoT = c * tanT + s * tanB;
    let anisoB = -s * tanT + c * tanB;
    pdfSpec = adjointBrdfAnisotropicSpecPdf(roughness, anisotropy, normal, anisoT, anisoB, wo, wi);
  } else {
    let nDotH = max(dot(normal, h), 0.0);
    let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
    let d = ggxD(nDotH, alpha);
    let g1Wo = smithG1(nDotV, roughness);
    pdfSpec = (d * g1Wo) / max(4.0 * nDotV, 1e-6);
  }
  let pdfDiff = nDotL * INV_PI;
  let basePdf = diffProb * pdfDiff + specProb * pdfSpec;
  let ccPdf = clearcoat * adjointClearcoatPdf(clearcoat, clearcoatRoughness, clearcoatNormal, wo, wi);
  let sheenPdf = sheen * adjointCharlieSheenPdf(sheen, sheenRoughness, normal, wo, wi);
  let lobeWeight = max(1.0 + max(clearcoat, 0.0) + max(sheen, 0.0), 1.0);
  return (basePdf + ccPdf + sheenPdf) / lobeWeight;
}

fn adjointDirectLightMisWeight(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  n: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
  specularColor: vec3f,
  specularIntensity: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  lightPdf: f32,
) -> f32 {
  if (lightPdf <= 0.0) { return 1.0; }
  let brdfPdf = adjointBrdfDirectionalPdfFullSampledWithClearcoatNormal(
    baseColor, roughness, metallic, n, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    specularColor, specularIntensity, anisotropy, anisotropyRotation,
  );
  return adjointPowerHeuristic(lightPdf, brdfPdf);
}

// ── camera (mirror kernelCore.generatePrimaryRay) ───────────────────────────
struct Ray { origin: vec3f, direction: vec3f }
fn generatePrimaryRay(px: u32, py: u32, jitter: vec2f) -> Ray {
  let uv = (vec2f(f32(px), f32(py)) + jitter) / vec2f(f32(params.width), f32(params.height));
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let far4 = params.invViewProj * vec4f(ndc, 1.0, 1.0);
  let near4 = params.invViewProj * vec4f(ndc, -1.0, 1.0);
  var ray: Ray;
  ray.origin = params.cameraPos.xyz;
  ray.direction = safe_normalize((far4.xyz / far4.w) - (near4.xyz / near4.w));
  return ray;
}

// ── brute-force intersection (Möller-Trumbore) ──────────────────────────────
struct Hit { valid: bool, t: f32, tri: u32, bary: vec3f }
fn closestHit(ro: vec3f, rd: vec3f) -> Hit {
  var best: Hit;
  best.valid = false;
  best.t = 1e30;
  for (var i = 0u; i < params.triangleCount; i = i + 1u) {
    let idx = indices[i];
    let v0 = positions[idx.x].xyz;
    let e1 = positions[idx.y].xyz - v0;
    let e2 = positions[idx.z].xyz - v0;
    let p = cross(rd, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-9) { continue; }
    let invDet = 1.0 / det;
    let tvec = ro - v0;
    let u = dot(tvec, p) * invDet;
    if (u < 0.0 || u > 1.0) { continue; }
    let q = cross(tvec, e1);
    let v = dot(rd, q) * invDet;
    if (v < 0.0 || u + v > 1.0) { continue; }
    let t = dot(e2, q) * invDet;
    if (t > 1e-4 && t < best.t) {
      best.valid = true; best.t = t; best.tri = i; best.bary = vec3f(1.0 - u - v, u, v);
    }
  }
  return best;
}
fn anyHit(ro: vec3f, rd: vec3f, tMax: f32) -> bool {
  for (var i = 0u; i < params.triangleCount; i = i + 1u) {
    let idx = indices[i];
    let v0 = positions[idx.x].xyz;
    let e1 = positions[idx.y].xyz - v0;
    let e2 = positions[idx.z].xyz - v0;
    let p = cross(rd, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-9) { continue; }
    let invDet = 1.0 / det;
    let tvec = ro - v0;
    let u = dot(tvec, p) * invDet;
    if (u < 0.0 || u > 1.0) { continue; }
    let q = cross(tvec, e1);
    let v = dot(rd, q) * invDet;
    if (v < 0.0 || u + v > 1.0) { continue; }
    let t = dot(e2, q) * invDet;
    if (t > 1e-4 && t < tMax) { return true; }
  }
  return false;
}

struct DirectLightAdjoint {
  baseColor: vec3f,
  roughness: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  metallicGrad: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescenceGrad: f32,
  iridescenceIorGrad: f32,
  iridescenceThicknessRangeGrad: vec2f,
  anisotropyGrad: f32,
  anisotropyRotationGrad: f32,
}

struct AdjointDirectMaterial {
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
}

const ADJOINT_DIRECT_PARAM_STEP = 1e-3;

fn adjointDirectContributionForMaterial(
  material: AdjointDirectMaterial,
  n: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
  Li: vec3f,
  lightPdf: f32,
) -> vec3f {
  return directLightContributionValue(
    material.baseColor,
    material.roughness,
    material.metallic,
    n,
    clearcoatNormal,
    wo,
    wi,
    material.specularColor,
    material.specularIntensity,
    material.clearcoat,
    material.clearcoatRoughness,
    material.sheen,
    material.sheenRoughness,
    material.sheenColor,
    material.iridescence,
    material.iridescenceIor,
    material.iridescenceThicknessMin,
    material.iridescenceThicknessMax,
    material.anisotropy,
    material.anisotropyRotation,
    Li,
    lightPdf,
  );
}

fn adjointDirectScalarGradient(
  dLoss_dR: vec3f,
  plus: vec3f,
  minus: vec3f,
  denominator: f32,
) -> f32 {
  if (!(denominator > 0.0) || !(denominator == denominator) ||
      abs(denominator) > 3.402823e38) { return 0.0; }
  let derivative = (plus - minus) / denominator;
  if (!all(derivative == derivative) ||
      !all(abs(derivative) <= vec3f(3.402823e38))) { return 0.0; }
  return dot(dLoss_dR, derivative);
}

fn adjointEffectiveIridescenceThickness(
  authoredMin: f32,
  authoredMax: f32,
  texel: f32,
) -> vec2f {
  if (texel >= 0.0) {
    let thickness = mix(authoredMin, authoredMax, texel);
    return vec2f(thickness);
  }
  return vec2f(authoredMin, authoredMax);
}

// Differentiate the complete frozen-sample direct-light estimator. In
// particular, finite-area and environment samples re-evaluate the BRDF PDF and
// power-heuristic weight for every parameter perturbation; MIS is not frozen.
fn directLightAdjointFull(
  dLoss_dR: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  n: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
  specularColor: vec3f,
  specularIntensity: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  authoredIridescenceThicknessMin: f32,
  authoredIridescenceThicknessMax: f32,
  iridescenceThicknessTexel: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  Li: vec3f,
  lightPdf: f32,
) -> DirectLightAdjoint {
  let original = AdjointDirectMaterial(
    baseColor, roughness, metallic, specularColor, specularIntensity,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    anisotropy, anisotropyRotation,
  );

  var gBaseColor = vec3f(0.0);
  for (var channel = 0u; channel < 3u; channel = channel + 1u) {
    var plus = original;
    var minus = original;
    plus.baseColor[channel] = min(baseColor[channel] + ADJOINT_DIRECT_PARAM_STEP, 1.0);
    minus.baseColor[channel] = max(baseColor[channel] - ADJOINT_DIRECT_PARAM_STEP, 0.0);
    gBaseColor[channel] = adjointDirectScalarGradient(
      dLoss_dR,
      adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
      adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
      plus.baseColor[channel] - minus.baseColor[channel],
    );
  }

  var plus = original;
  var minus = original;
  plus.roughness = min(roughness + ADJOINT_DIRECT_PARAM_STEP, 1.0);
  minus.roughness = max(roughness - ADJOINT_DIRECT_PARAM_STEP, 0.0);
  let gRoughness = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    plus.roughness - minus.roughness,
  );

  var gSpecularColor = vec3f(0.0);
  for (var channel = 0u; channel < 3u; channel = channel + 1u) {
    plus = original;
    minus = original;
    plus.specularColor[channel] = min(specularColor[channel] + ADJOINT_DIRECT_PARAM_STEP, 1.0);
    minus.specularColor[channel] = max(specularColor[channel] - ADJOINT_DIRECT_PARAM_STEP, 0.0);
    gSpecularColor[channel] = adjointDirectScalarGradient(
      dLoss_dR,
      adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
      adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
      plus.specularColor[channel] - minus.specularColor[channel],
    );
  }

  plus = original;
  minus = original;
  plus.specularIntensity = min(specularIntensity + ADJOINT_DIRECT_PARAM_STEP, 1.0);
  minus.specularIntensity = max(specularIntensity - ADJOINT_DIRECT_PARAM_STEP, 0.0);
  let gSpecularIntensity = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    plus.specularIntensity - minus.specularIntensity,
  );

  plus = original;
  minus = original;
  plus.metallic = min(metallic + ADJOINT_DIRECT_PARAM_STEP, 1.0);
  minus.metallic = max(metallic - ADJOINT_DIRECT_PARAM_STEP, 0.0);
  let gMetallic = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    plus.metallic - minus.metallic,
  );

  plus = original;
  minus = original;
  plus.clearcoat = min(clearcoat + ADJOINT_DIRECT_PARAM_STEP, 1.0);
  minus.clearcoat = max(clearcoat - ADJOINT_DIRECT_PARAM_STEP, 0.0);
  let gClearcoat = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    plus.clearcoat - minus.clearcoat,
  );

  plus = original;
  minus = original;
  plus.clearcoatRoughness = min(clearcoatRoughness + ADJOINT_DIRECT_PARAM_STEP, 1.0);
  minus.clearcoatRoughness = max(clearcoatRoughness - ADJOINT_DIRECT_PARAM_STEP, 0.0);
  let gClearcoatRoughness = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    plus.clearcoatRoughness - minus.clearcoatRoughness,
  );

  plus = original;
  minus = original;
  plus.sheen = min(sheen + ADJOINT_DIRECT_PARAM_STEP, 1.0);
  minus.sheen = max(sheen - ADJOINT_DIRECT_PARAM_STEP, 0.0);
  let gSheen = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    plus.sheen - minus.sheen,
  );

  plus = original;
  minus = original;
  plus.sheenRoughness = min(sheenRoughness + ADJOINT_DIRECT_PARAM_STEP, 1.0);
  minus.sheenRoughness = max(sheenRoughness - ADJOINT_DIRECT_PARAM_STEP, 0.0);
  let gSheenRoughness = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    plus.sheenRoughness - minus.sheenRoughness,
  );

  var gSheenColor = vec3f(0.0);
  for (var channel = 0u; channel < 3u; channel = channel + 1u) {
    plus = original;
    minus = original;
    plus.sheenColor[channel] = min(sheenColor[channel] + ADJOINT_DIRECT_PARAM_STEP, 1.0);
    minus.sheenColor[channel] = max(sheenColor[channel] - ADJOINT_DIRECT_PARAM_STEP, 0.0);
    gSheenColor[channel] = adjointDirectScalarGradient(
      dLoss_dR,
      adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
      adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
      plus.sheenColor[channel] - minus.sheenColor[channel],
    );
  }

  plus = original;
  minus = original;
  plus.iridescence = min(iridescence + ADJOINT_DIRECT_PARAM_STEP, 1.0);
  minus.iridescence = max(iridescence - ADJOINT_DIRECT_PARAM_STEP, 0.0);
  let gIridescence = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    plus.iridescence - minus.iridescence,
  );

  plus = original;
  minus = original;
  plus.iridescenceIor = iridescenceIor + ADJOINT_DIRECT_PARAM_STEP;
  minus.iridescenceIor = max(iridescenceIor - ADJOINT_DIRECT_PARAM_STEP, 1.0);
  let gIridescenceIor = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    plus.iridescenceIor - minus.iridescenceIor,
  );

  var authoredMinPlus = authoredIridescenceThicknessMin + ADJOINT_DIRECT_PARAM_STEP;
  var authoredMinMinus = max(authoredIridescenceThicknessMin - ADJOINT_DIRECT_PARAM_STEP, 0.0);
  plus = original;
  minus = original;
  var effectiveThickness = adjointEffectiveIridescenceThickness(
    authoredMinPlus, authoredIridescenceThicknessMax, iridescenceThicknessTexel,
  );
  plus.iridescenceThicknessMin = effectiveThickness.x;
  plus.iridescenceThicknessMax = effectiveThickness.y;
  effectiveThickness = adjointEffectiveIridescenceThickness(
    authoredMinMinus, authoredIridescenceThicknessMax, iridescenceThicknessTexel,
  );
  minus.iridescenceThicknessMin = effectiveThickness.x;
  minus.iridescenceThicknessMax = effectiveThickness.y;
  let gThicknessMin = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    authoredMinPlus - authoredMinMinus,
  );

  let authoredMaxPlus = authoredIridescenceThicknessMax + ADJOINT_DIRECT_PARAM_STEP;
  let authoredMaxMinus = max(authoredIridescenceThicknessMax - ADJOINT_DIRECT_PARAM_STEP, 0.0);
  plus = original;
  minus = original;
  effectiveThickness = adjointEffectiveIridescenceThickness(
    authoredIridescenceThicknessMin, authoredMaxPlus, iridescenceThicknessTexel,
  );
  plus.iridescenceThicknessMin = effectiveThickness.x;
  plus.iridescenceThicknessMax = effectiveThickness.y;
  effectiveThickness = adjointEffectiveIridescenceThickness(
    authoredIridescenceThicknessMin, authoredMaxMinus, iridescenceThicknessTexel,
  );
  minus.iridescenceThicknessMin = effectiveThickness.x;
  minus.iridescenceThicknessMax = effectiveThickness.y;
  let gThicknessMax = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    authoredMaxPlus - authoredMaxMinus,
  );

  plus = original;
  minus = original;
  plus.anisotropy = min(anisotropy + ADJOINT_DIRECT_PARAM_STEP, 1.0);
  minus.anisotropy = max(anisotropy - ADJOINT_DIRECT_PARAM_STEP, 0.0);
  let gAnisotropy = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    plus.anisotropy - minus.anisotropy,
  );

  plus = original;
  minus = original;
  plus.anisotropyRotation = anisotropyRotation + ADJOINT_DIRECT_PARAM_STEP;
  minus.anisotropyRotation = anisotropyRotation - ADJOINT_DIRECT_PARAM_STEP;
  let gAnisotropyRotation = adjointDirectScalarGradient(
    dLoss_dR,
    adjointDirectContributionForMaterial(plus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    adjointDirectContributionForMaterial(minus, n, clearcoatNormal, wo, wi, Li, lightPdf),
    plus.anisotropyRotation - minus.anisotropyRotation,
  );

  return DirectLightAdjoint(
    gBaseColor, gRoughness, gSpecularColor, gSpecularIntensity, gMetallic,
    gClearcoat, gClearcoatRoughness, gSheen, gSheenRoughness, gSheenColor,
    gIridescence, gIridescenceIor, vec2f(gThicknessMin, gThicknessMax),
    gAnisotropy, gAnisotropyRotation,
  );
}

fn directLightBrdfValue(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  n: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
  specularColor: vec3f,
  specularIntensity: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> vec3f {
  return adjointEvaluateBrdfWithAnisotropyAndIridescence(
    baseColor, roughness, metallic, n, wo, wi,
    anisotropy, anisotropyRotation, specularColor, specularIntensity,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
  ) +
    adjointClearcoatLobe(clearcoat, clearcoatRoughness, clearcoatNormal, wo, wi) +
    adjointSheenLobe(sheen, sheenRoughness, sheenColor, n, wo, wi);
}

const ADJOINT_NORMAL_SCALE_DERIV_STEP = 1e-3;

fn directLightContributionValue(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  n: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
  specularColor: vec3f,
  specularIntensity: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  Li: vec3f,
  lightPdf: f32,
) -> vec3f {
  let nDotL = max(0.0, dot(n, wi));
  if (nDotL <= 0.0) { return vec3f(0.0); }
  let misWeight = adjointDirectLightMisWeight(
    baseColor, roughness, metallic, n, clearcoatNormal, wo, wi, specularColor, specularIntensity,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, anisotropy, anisotropyRotation, lightPdf,
  );
  return directLightBrdfValue(
    baseColor, roughness, metallic, n, clearcoatNormal, wo, wi, specularColor, specularIntensity,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    anisotropy, anisotropyRotation,
  ) * nDotL * Li * misWeight;
}

fn directLightNormalStackScaleGradient(
  dLoss_dR: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  n: vec3f,
  clearcoatNormal: vec3f,
  dNormal_dScale: vec3f,
  dClearcoatNormal_dScale: vec3f,
  wo: vec3f,
  wi: vec3f,
  specularColor: vec3f,
  specularIntensity: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  Li: vec3f,
  lightPdf: f32,
) -> f32 {
  if (dot(dNormal_dScale, dNormal_dScale) <= 1e-12 &&
      dot(dClearcoatNormal_dScale, dClearcoatNormal_dScale) <= 1e-12) {
    return 0.0;
  }
  let h = ADJOINT_NORMAL_SCALE_DERIV_STEP;
  let nPlus = safe_normalize(n + dNormal_dScale * h);
  let nMinus = safe_normalize(n - dNormal_dScale * h);
  let ccPlus = safe_normalize(clearcoatNormal + dClearcoatNormal_dScale * h);
  let ccMinus = safe_normalize(clearcoatNormal - dClearcoatNormal_dScale * h);
  let cPlus = directLightContributionValue(
    baseColor, roughness, metallic, nPlus, ccPlus, wo, wi, specularColor, specularIntensity,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    anisotropy, anisotropyRotation, Li, lightPdf,
  );
  let cMinus = directLightContributionValue(
    baseColor, roughness, metallic, nMinus, ccMinus, wo, wi, specularColor, specularIntensity,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    anisotropy, anisotropyRotation, Li, lightPdf,
  );
  return dot(dLoss_dR, (cPlus - cMinus) / (2.0 * h));
}

fn directLightClearcoatNormalScaleGradient(
  dLoss_dR: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  n: vec3f,
  clearcoatNormal: vec3f,
  dClearcoatNormal_dScale: vec3f,
  wo: vec3f,
  wi: vec3f,
  specularColor: vec3f,
  specularIntensity: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  Li: vec3f,
  lightPdf: f32,
) -> f32 {
  if (dot(dClearcoatNormal_dScale, dClearcoatNormal_dScale) <= 1e-12 || clearcoat <= 0.0) { return 0.0; }
  let h = ADJOINT_NORMAL_SCALE_DERIV_STEP;
  let ccPlus = safe_normalize(clearcoatNormal + dClearcoatNormal_dScale * h);
  let ccMinus = safe_normalize(clearcoatNormal - dClearcoatNormal_dScale * h);
  let cPlus = directLightContributionValue(
    baseColor, roughness, metallic, n, ccPlus, wo, wi, specularColor, specularIntensity,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    anisotropy, anisotropyRotation, Li, lightPdf,
  );
  let cMinus = directLightContributionValue(
    baseColor, roughness, metallic, n, ccMinus, wo, wi, specularColor, specularIntensity,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    anisotropy, anisotropyRotation, Li, lightPdf,
  );
  return dot(dLoss_dR, (cPlus - cMinus) / (2.0 * h));
}

struct EmitterRadianceAdjoint {
  color: vec3f,
  intensity: f32,
}

fn emitterRadianceAdjoint(
  dLoss_dPackedRadiance: vec3f,
  sourceFactor: vec3f,
  emitterColor: vec3f,
  emitterIntensity: f32,
  invReplaySamples: f32,
) -> EmitterRadianceAdjoint {
  return EmitterRadianceAdjoint(
    dLoss_dPackedRadiance * sourceFactor * emitterIntensity * invReplaySamples,
    dot(dLoss_dPackedRadiance, sourceFactor * emitterColor) * invReplaySamples,
  );
}

fn scatterEmitterRadianceGradient(
  targetKind: u32,
  targetSlot: u32,
  dLoss_dPackedRadiance: vec3f,
  sourceFactor: vec3f,
  invReplaySamples: f32,
) {
  for (var k = 0u; k < params.paramCount; k = k + 1u) {
    let descBase = k * 2u;
    let d = adjointParamDescs[descBase];
    if (d.y != ${ADJOINT_FIELD_EMITTER_COLOR}u && d.y != ${ADJOINT_FIELD_EMITTER_INTENSITY}u) {
      continue;
    }
    let descKind = d.w & 255u;
    let descCount = max(1u, d.w >> 8u);
    if (descKind != targetKind || targetSlot < d.x || targetSlot >= d.x + descCount) { continue; }
    let payload = adjointParamDescs[descBase + 1u];
    let emitterColor = vec3f(
      bitcast<f32>(payload.x),
      bitcast<f32>(payload.y),
      bitcast<f32>(payload.z),
    );
    let emitterIntensity = bitcast<f32>(payload.w);
    let gradOffset = d.z;
    let emitterGradient = emitterRadianceAdjoint(
      dLoss_dPackedRadiance,
      sourceFactor,
      emitterColor,
      emitterIntensity,
      invReplaySamples,
    );
    if (d.y == ${ADJOINT_FIELD_EMITTER_COLOR}u) {
      adjointScatter(gradOffset, emitterGradient.color.x);
      adjointScatter(gradOffset + 1u, emitterGradient.color.y);
      adjointScatter(gradOffset + 2u, emitterGradient.color.z);
    } else {
      adjointScatter(gradOffset, emitterGradient.intensity);
    }
  }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }

  // Per-pixel ∂loss/∂rendered (the session computed it from baseline vs target).
  let pixel = gid.y * params.width + gid.x;
  let base = pixel * params.channels;
  let dLoss_dR = vec3f(dLossDRendered[base], dLossDRendered[base + 1u], dLossDRendered[base + 2u]);

  let replaySamples = max(params.sampleCount, 1u);
  let invReplaySamples = 1.0 / f32(replaySamples);
  for (var sampleIdx = 0u; sampleIdx < replaySamples; sampleIdx = sampleIdx + 1u) {
    let frameSeed = ADJOINT_FROZEN_SEED_BASE + sampleIdx;
    var rng = pcgInit(gid.x, gid.y, ptRngFrameKey(frameSeed, 0u));
    let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));
    let ray = generatePrimaryRay(gid.x, gid.y, jitter);
    let hit = closestHit(ray.origin, ray.direction);
    if (!hit.valid) { continue; }

    let matId = triMaterialIds[hit.tri];
    let m0 = materials[matId * MATERIAL_VEC4_STRIDE];
    let m1 = materials[matId * MATERIAL_VEC4_STRIDE + 1u];
    let m23 = materials[matId * MATERIAL_VEC4_STRIDE + 23u];
    let m24 = materials[matId * MATERIAL_VEC4_STRIDE + 24u];
    let m25 = materials[matId * MATERIAL_VEC4_STRIDE + 25u];
    let m26 = materials[matId * MATERIAL_VEC4_STRIDE + 26u];
    let m27 = materials[matId * MATERIAL_VEC4_STRIDE + 27u];
    let baseColor = m0.rgb;
    let hitBaryVW = vec2f(hit.bary.y, hit.bary.z);
    let baseColorNoAoFactor = sampleAdjointVertexColor(hit.tri, hitBaryVW).rgb *
      sampleAdjointBaseColorTexture(matId, hit.tri, hitBaryVW).rgb;
    let aoSample = sampleAdjointAo(matId, hit.tri, hitBaryVW);
    let baseColorFactor = baseColorNoAoFactor * aoSample.factor;
    let effectiveBaseColor = baseColor * baseColorFactor;
    let roughness = clamp(m0.w, 0.0, 1.0);
    let metallic = clamp(m1.w, 0.0, 1.0);
    let ormFactor = sampleAdjointOrmTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z));
    let effectiveRoughness = clamp(roughness * ormFactor.g, 0.0, 1.0);
    let effectiveMetallic = clamp(metallic * ormFactor.b, 0.0, 1.0);
    let isUnlit = (u32(max(m26.w, 0.0)) & 2u) != 0u;
    let specularColor = m27.rgb;
    let specularIntensity = clamp(m27.w, 0.0, 1.0);
    let specularColorFactor = sampleAdjointSpecularColorTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z));
    let specularIntensityFactor = sampleAdjointSpecularIntensityTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z));
    let effectiveSpecularColor = clamp(specularColor * specularColorFactor, vec3f(0.0), vec3f(1.0));
    let effectiveSpecularIntensity = clamp(specularIntensity * specularIntensityFactor, 0.0, 1.0);
    let clearcoat = clamp(m23.x, 0.0, 1.0);
    let clearcoatRoughness = clamp(m23.y, 0.0, 1.0);
    let sheen = clamp(m23.z, 0.0, 1.0);
    let sheenRoughness = clamp(m23.w, 0.0, 1.0);
    let sheenColor = clamp(m24.rgb, vec3f(0.0), vec3f(1.0));
    let clearcoatFactor = sampleAdjointClearcoatTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z));
    let clearcoatRoughnessFactor = sampleAdjointClearcoatRoughnessTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z));
    let sheenColorFactor = sampleAdjointSheenColorTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z));
    let sheenRoughnessFactor = sampleAdjointSheenRoughnessTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z));
    let effectiveClearcoat = clamp(clearcoat * clearcoatFactor, 0.0, 1.0);
    let effectiveClearcoatRoughness = clamp(clearcoatRoughness * clearcoatRoughnessFactor, 0.0, 1.0);
    let effectiveSheenColor = clamp(sheenColor * sheenColorFactor, vec3f(0.0), vec3f(1.0));
    let effectiveSheenRoughness = clamp(sheenRoughness * sheenRoughnessFactor, 0.0, 1.0);
    let iridescence = clamp(m24.w, 0.0, 1.0);
    let iridescenceIor = max(m25.x, 1.0);
    let iridescenceThicknessMin = max(m25.y, 0.0);
    let iridescenceThicknessMax = max(m25.z, 0.0);
    let iridescenceFactor = sampleAdjointIridescenceTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z));
    let iridescenceThicknessSample = sampleAdjointIridescenceThicknessTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z));
    var effectiveIridescence = clamp(iridescence * iridescenceFactor, 0.0, 1.0);
    var effectiveIridescenceThicknessMin = iridescenceThicknessMin;
    var effectiveIridescenceThicknessMax = iridescenceThicknessMax;
    var iridescenceGradientFactor = iridescenceFactor;
    if (iridescenceThicknessSample >= 0.0) {
      let iridescenceThickness = mix(iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample);
      effectiveIridescenceThicknessMin = iridescenceThickness;
      effectiveIridescenceThicknessMax = iridescenceThickness;
      if (iridescenceThickness <= 0.0) {
        effectiveIridescence = 0.0;
        iridescenceGradientFactor = 0.0;
      }
    }
    let materialTexBase = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
    var anisotropy = 0.0;
    var anisotropyRotation = 0.0;
    let anisotropyMapSample = sampleAdjointAnisotropyTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z));
    if (materialTexBase + 5u < arrayLength(&materialTexDescriptors)) {
      let anisoDesc = materialTexDescriptors[materialTexBase + 5u];
      anisotropy = clamp(anisoDesc.x, 0.0, 1.0);
      anisotropyRotation = anisoDesc.y;
    }
    let effectiveAnisotropy = clamp(anisotropy * anisotropyMapSample.strength, 0.0, 1.0);
    let localAnisotropyRotation = anisotropyRotation + anisotropyMapSample.rotationOffset;

    let idx = indices[hit.tri];
    let nGeo = safe_normalize(hit.bary.x * normals[idx.x].xyz + hit.bary.y * normals[idx.y].xyz + hit.bary.z * normals[idx.z].xyz);
    // Face the shading normal toward the viewer — the SAME flip the forward shade
    // prologue applies (shadePrologue.wgsl.ts). Without it, back-facing geometry
    // gets nDotL<=0 against an interior light and contributes no gradient.
    let nFace = select(-nGeo, nGeo, dot(nGeo, ray.direction) < 0.0);
    let normalMapSample = sampleAdjointNormalMap(matId, hit.tri, hitBaryVW, nFace);
    let bumpMapSample = sampleAdjointBumpMap(matId, hit.tri, hitBaryVW, normalMapSample.normal);
    let n = bumpMapSample.normal;
    let anisotropyOrientation = adjointAnisotropyOrientation(
      matId, hit.tri, hitBaryVW, n, localAnisotropyRotation,
    );
    let effectiveAnisotropyRotation = anisotropyOrientation.angle;
    let clearcoatNormalMapSample = sampleAdjointClearcoatNormalMap(matId, hit.tri, hitBaryVW, n);
    let clearcoatNormal = clearcoatNormalMapSample.normal;
    var dNormal_dNormalScale = normalMapSample.dNormal_dScale;
    var dClearcoatNormal_dNormalScale = dNormal_dNormalScale;
    var dNormal_dBumpScale = bumpMapSample.dNormal_dScale;
    var dClearcoatNormal_dBumpScale = dNormal_dBumpScale;
    let dClearcoatNormal_dClearcoatNormalScale = clearcoatNormalMapSample.dNormal_dScale;
    let normalScaleBase = adjointMaterialNormalScale(matId);
    let bumpScaleBase = adjointMaterialBumpScale(matId);
    let clearcoatNormalScaleBase = adjointMaterialClearcoatNormalScale(matId);
    let stackH = ADJOINT_NORMAL_SCALE_DERIV_STEP;
    let stackNormalPlus = sampleAdjointNormalStackWithScales(
      matId, hit.tri, hitBaryVW, nFace,
      normalScaleBase + stackH, bumpScaleBase, clearcoatNormalScaleBase,
    );
    let stackNormalMinus = sampleAdjointNormalStackWithScales(
      matId, hit.tri, hitBaryVW, nFace,
      normalScaleBase - stackH, bumpScaleBase, clearcoatNormalScaleBase,
    );
    dNormal_dNormalScale = (stackNormalPlus.normal - stackNormalMinus.normal) / (2.0 * stackH);
    dClearcoatNormal_dNormalScale = (stackNormalPlus.clearcoatNormal - stackNormalMinus.clearcoatNormal) / (2.0 * stackH);
    let stackBumpPlus = sampleAdjointNormalStackWithScales(
      matId, hit.tri, hitBaryVW, nFace,
      normalScaleBase, bumpScaleBase + stackH, clearcoatNormalScaleBase,
    );
    let stackBumpMinus = sampleAdjointNormalStackWithScales(
      matId, hit.tri, hitBaryVW, nFace,
      normalScaleBase, bumpScaleBase - stackH, clearcoatNormalScaleBase,
    );
    dNormal_dBumpScale = (stackBumpPlus.normal - stackBumpMinus.normal) / (2.0 * stackH);
    dClearcoatNormal_dBumpScale = (stackBumpPlus.clearcoatNormal - stackBumpMinus.clearcoatNormal) / (2.0 * stackH);
    let pos = ray.origin + ray.direction * hit.t;
    let wo = -ray.direction;

    // Emissive partial — NOT a NEE term. The forward adds throughput * emissive for
    // the emission this surface is seen to emit DIRECTLY by the camera ray at THIS
    // (primary) hit (shadePrologue.wgsl.ts:63). Path-replay's primary hit has
    // throughput = 1, so d(rendered_c)/d(emissive_c) = emissiveIntensity (dContribution_
    // dEmissive with throughput = 1), and d(loss)/d(emissive_c) = dLoss_dR_c * intensity.
    // Independent of light visibility — computed here, scattered per-descriptor below
    // with that descriptor's fixed emissiveIntensity (carried in .w). It is gated by
    // the matId match in the scatter loop, so a pixel only contributes to the emissive
    // gradient when ITS primary-hit material is the optimized emissive primitive.
    let dRendered_dEmissivePerUnitIntensity = dContribution_dEmissive(vec3f(1.0), 1.0); // = (1,1,1)
    let emissiveTexel = sampleAdjointEmissiveTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z)).rgb;
    let lightMapRadiancePerUnitIntensity = sampleAdjointLightMapRadiancePerUnitIntensity(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z));

    // Single-bounce direct lighting, summed deterministically over every direct
    // light source the scoped adjoint pass mirrors.
    var gBaseColor = vec3f(0.0);
    var gRough = 0.0;
    var gSpecularColor = vec3f(0.0);
    var gSpecularIntensity = 0.0;
    var gMetallic = 0.0;
    var gClearcoat = 0.0;
    var gClearcoatRoughness = 0.0;
    var gSheen = 0.0;
    var gSheenRoughness = 0.0;
    var gSheenColor = vec3f(0.0);
    var gIridescence = 0.0;
    var gIridescenceIor = 0.0;
    var gIridescenceThicknessRange = vec2f(0.0);
    var gAnisotropy = 0.0;
    var gAnisotropyRotation = 0.0;
    var gEnvMapIntensity = 0.0;
    var gNormalScale = 0.0;
    var gBumpScale = 0.0;
    var gClearcoatNormalScale = 0.0;
    for (var di = 0u; di < params.directionalLightCount; di = di + 1u) {
      let dBase = di * 2u;
      let dDirAD = directionalLights[dBase];
      let dIrrMean = directionalLights[dBase + 1u];
      var wi = safe_normalize(dDirAD.xyz);
      let angularDiameterRaw = dDirAD.w;
      let directionalShadowDisabled = angularDiameterRaw < 0.0;
      let angularDiameter = select(angularDiameterRaw, -1.0 - angularDiameterRaw, directionalShadowDisabled);
      if (angularDiameter > 0.0) {
        let cosHalfAngle = cos(angularDiameter * 0.5);
        let xi1 = rand_f32(&rng);
        let xi2 = rand_f32(&rng);
        let cosTheta = mix(cosHalfAngle, 1.0, xi1);
        let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        let phi = 6.28318530718 * xi2;
        let tangentX = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(wi.x) > 0.9);
        let basisY = normalize(cross(wi, tangentX));
        let basisX = cross(basisY, wi);
        wi = normalize(sinTheta * cos(phi) * basisX + sinTheta * sin(phi) * basisY + cosTheta * wi);
      }
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      if (!directionalShadowDisabled && anyHit(pos + n * 1e-3, wi, 1e30)) { continue; }
      let lg = directLightAdjointFull(
        dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
        effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
        effectiveIridescence,
        iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
        iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample,
        effectiveAnisotropy, effectiveAnisotropyRotation,
        dIrrMean.rgb, 0.0,
      );
      gBaseColor = gBaseColor + lg.baseColor;
      gRough = gRough + lg.roughness;
      gSpecularColor = gSpecularColor + lg.specularColor;
      gSpecularIntensity = gSpecularIntensity + lg.specularIntensity;
      gMetallic = gMetallic + lg.metallicGrad;
      gClearcoat = gClearcoat + lg.clearcoat;
      gClearcoatRoughness = gClearcoatRoughness + lg.clearcoatRoughness;
      gSheen = gSheen + lg.sheen;
      gSheenRoughness = gSheenRoughness + lg.sheenRoughness;
      gSheenColor = gSheenColor + lg.sheenColor;
      gIridescence = gIridescence + lg.iridescenceGrad;
      gIridescenceIor = gIridescenceIor + lg.iridescenceIorGrad;
      gIridescenceThicknessRange = gIridescenceThicknessRange + lg.iridescenceThicknessRangeGrad;
      gAnisotropy = gAnisotropy + lg.anisotropyGrad;
      gAnisotropyRotation = gAnisotropyRotation + lg.anisotropyRotationGrad;
      if (!isUnlit) {
        gNormalScale = gNormalScale + directLightNormalStackScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dNormal_dNormalScale, dClearcoatNormal_dNormalScale,
          wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, dIrrMean.rgb, 0.0,
        );
        gBumpScale = gBumpScale + directLightNormalStackScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dNormal_dBumpScale, dClearcoatNormal_dBumpScale,
          wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, dIrrMean.rgb, 0.0,
        );
        gClearcoatNormalScale = gClearcoatNormalScale + directLightClearcoatNormalScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dClearcoatNormal_dClearcoatNormalScale, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, dIrrMean.rgb, 0.0,
        );
        let brdfValue = directLightBrdfValue(
          effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation,
        );
        scatterEmitterRadianceGradient(
          ${ADJOINT_EMITTER_TARGET_DIRECTIONAL}u,
          di,
          dLoss_dR * brdfValue * nDotL,
          vec3f(1.0),
          invReplaySamples,
        );
      }
    }

    for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
      let lp = pointLights[pi * 3u].xyz;
      let rad = pointLights[pi * 3u + 1u].rgb;
      let ptExtra = pointLights[pi * 3u + 2u];
      let ptMaxDist = ptExtra.x;
      let ptDecay   = ptExtra.y;
      let toPoint = lp - pos;
      let dist2 = dot(toPoint, toPoint);
      if (!(dist2 > 0.0)) { continue; }
      let dist = sqrt(dist2);
      if (ptMaxDist > 0.0 && dist > ptMaxDist) { continue; }
      let wi = toPoint / dist;
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      if (ptExtra.z <= 0.5 && anyHit(pos + n * 1e-3, wi, dist - 2e-3)) { continue; } // shadowed
      let attenuation = pointSpotDistanceAttenuation(dist, ptMaxDist, ptDecay);
      let Li = rad * attenuation;
      let lg = directLightAdjointFull(
        dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
        effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
        effectiveIridescence,
        iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
        iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample,
        effectiveAnisotropy, effectiveAnisotropyRotation,
        Li, 0.0,
      );
      gBaseColor = gBaseColor + lg.baseColor;
      gRough = gRough + lg.roughness;
      gSpecularColor = gSpecularColor + lg.specularColor;
      gSpecularIntensity = gSpecularIntensity + lg.specularIntensity;
      gMetallic = gMetallic + lg.metallicGrad;
      gClearcoat = gClearcoat + lg.clearcoat;
      gClearcoatRoughness = gClearcoatRoughness + lg.clearcoatRoughness;
      gSheen = gSheen + lg.sheen;
      gSheenRoughness = gSheenRoughness + lg.sheenRoughness;
      gSheenColor = gSheenColor + lg.sheenColor;
      gIridescence = gIridescence + lg.iridescenceGrad;
      gIridescenceIor = gIridescenceIor + lg.iridescenceIorGrad;
      gIridescenceThicknessRange = gIridescenceThicknessRange + lg.iridescenceThicknessRangeGrad;
      gAnisotropy = gAnisotropy + lg.anisotropyGrad;
      gAnisotropyRotation = gAnisotropyRotation + lg.anisotropyRotationGrad;
      if (!isUnlit) {
        gNormalScale = gNormalScale + directLightNormalStackScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dNormal_dNormalScale, dClearcoatNormal_dNormalScale,
          wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, Li, 0.0,
        );
        gBumpScale = gBumpScale + directLightNormalStackScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dNormal_dBumpScale, dClearcoatNormal_dBumpScale,
          wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, Li, 0.0,
        );
        gClearcoatNormalScale = gClearcoatNormalScale + directLightClearcoatNormalScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dClearcoatNormal_dClearcoatNormalScale, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, Li, 0.0,
        );
        let brdfValue = directLightBrdfValue(
          effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation,
        );
        scatterEmitterRadianceGradient(
          ${ADJOINT_EMITTER_TARGET_POINT}u,
          pi,
          dLoss_dR * brdfValue * (nDotL * attenuation),
          vec3f(1.0),
          invReplaySamples,
        );
      }
    }

    for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
      let sb = si * 4u;
      let spos = spotLights[sb].xyz;
      let saxis = spotLights[sb + 1u];
      let sradW = spotLights[sb + 2u];
      let spExtra = spotLights[sb + 3u];
      let spotDir = safe_normalize(saxis.xyz);
      let cosOuter = saxis.w;
      let cosInner = sradW.w;
      let toSpot = spos - pos;
      let dist2 = dot(toSpot, toSpot);
      if (!(dist2 > 0.0)) { continue; }
      let dist = sqrt(dist2);
      if (spExtra.x > 0.0 && dist > spExtra.x) { continue; }
      let wi = toSpot / dist;
      let coneCos = dot(-wi, spotDir);
      if (coneCos < cosOuter) { continue; }
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      if (spExtra.z <= 0.5 && anyHit(pos + n * 1e-3, wi, dist - 2e-3)) { continue; }
      let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), coneCos);
      let attenuation = pointSpotDistanceAttenuation(dist, spExtra.x, spExtra.y);
      let Li = sradW.rgb * softness * attenuation;
      let lg = directLightAdjointFull(
        dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
        effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
        effectiveIridescence,
        iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
        iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample,
        effectiveAnisotropy, effectiveAnisotropyRotation,
        Li, 0.0,
      );
      gBaseColor = gBaseColor + lg.baseColor;
      gRough = gRough + lg.roughness;
      gSpecularColor = gSpecularColor + lg.specularColor;
      gSpecularIntensity = gSpecularIntensity + lg.specularIntensity;
      gMetallic = gMetallic + lg.metallicGrad;
      gClearcoat = gClearcoat + lg.clearcoat;
      gClearcoatRoughness = gClearcoatRoughness + lg.clearcoatRoughness;
      gSheen = gSheen + lg.sheen;
      gSheenRoughness = gSheenRoughness + lg.sheenRoughness;
      gSheenColor = gSheenColor + lg.sheenColor;
      gIridescence = gIridescence + lg.iridescenceGrad;
      gIridescenceIor = gIridescenceIor + lg.iridescenceIorGrad;
      gIridescenceThicknessRange = gIridescenceThicknessRange + lg.iridescenceThicknessRangeGrad;
      gAnisotropy = gAnisotropy + lg.anisotropyGrad;
      gAnisotropyRotation = gAnisotropyRotation + lg.anisotropyRotationGrad;
      if (!isUnlit) {
        gNormalScale = gNormalScale + directLightNormalStackScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dNormal_dNormalScale, dClearcoatNormal_dNormalScale,
          wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, Li, 0.0,
        );
        gBumpScale = gBumpScale + directLightNormalStackScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dNormal_dBumpScale, dClearcoatNormal_dBumpScale,
          wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, Li, 0.0,
        );
        gClearcoatNormalScale = gClearcoatNormalScale + directLightClearcoatNormalScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dClearcoatNormal_dClearcoatNormalScale, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, Li, 0.0,
        );
        let brdfValue = directLightBrdfValue(
          effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation,
        );
        scatterEmitterRadianceGradient(
          ${ADJOINT_EMITTER_TARGET_SPOT}u,
          si,
          dLoss_dR * brdfValue * (nDotL * softness * attenuation),
          vec3f(1.0),
          invReplaySamples,
        );
      }
    }

    // Rect/disc-area lights: stochastic area-measure replay of the same geometric
    // term the forward NEE integrates (brdf·nDotL·radiance·MIS / pdf_area). The pass
    // still sums every light rather than replaying the forward one-of-N light
    // selection lottery, but it no longer approximates finite emitters by centers
    // or drops the forward area-light MIS weight.
    for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
      let rb = ri * 4u;
      let rpos = rectAreaLights[rb].xyz;
      let ru = rectAreaLights[rb + 1u].xyz;
      let rv = rectAreaLights[rb + 2u].xyz;
      let rshape = rectAreaLights[rb + 3u];
      let rad = rshape.rgb;
      let isDisc = abs(rshape.w - 1.0) < 0.5;
      let xi1 = rand_f32(&rng);
      let xi2 = rand_f32(&rng);
      var lpos: vec3f;
      var area: f32;
      if (isDisc) {
        let r = length(ru);
        let disc = adjointConcentricDiscSample(vec2f(xi1 * 2.0 - 1.0, xi2 * 2.0 - 1.0));
        lpos = rpos + ru * disc.x + rv * disc.y;
        area = PI * r * r;
      } else {
        lpos = rpos + ru * (xi1 * 2.0 - 1.0) + rv * (xi2 * 2.0 - 1.0);
        area = 4.0 * length(cross(ru, rv));
      }
      let toLight = lpos - pos;
      let dist2 = dot(toLight, toLight);
      if (dist2 <= 0.0 || area <= 0.0) { continue; }
      let dist = sqrt(dist2);
      let wi = toLight / dist;
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      let lightNormal = safe_normalize(cross(ru, rv));
      let cosLight = max(dot(lightNormal, -wi), 0.0);
      if (cosLight <= 0.0) { continue; }
      if (rectAreaLights[rb].w <= 0.5 && anyHit(pos + n * 1e-3, wi, max(dist - 2e-3, 1e-3))) { continue; } // shadowed
      let lightPdf = dist2 / (cosLight * area);
      let LiPerMisUnit = rad / lightPdf;
      let misWeight = adjointDirectLightMisWeight(
        effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi,
        effectiveSpecularColor, effectiveSpecularIntensity,
        effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness,
        effectiveAnisotropy, effectiveAnisotropyRotation, lightPdf,
      );
      let lg = directLightAdjointFull(
        dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
        effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
        effectiveIridescence,
        iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
        iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample,
        effectiveAnisotropy, effectiveAnisotropyRotation,
        LiPerMisUnit, lightPdf,
      );
      gBaseColor = gBaseColor + lg.baseColor;
      gRough = gRough + lg.roughness;
      gSpecularColor = gSpecularColor + lg.specularColor;
      gSpecularIntensity = gSpecularIntensity + lg.specularIntensity;
      gMetallic = gMetallic + lg.metallicGrad;
      gClearcoat = gClearcoat + lg.clearcoat;
      gClearcoatRoughness = gClearcoatRoughness + lg.clearcoatRoughness;
      gSheen = gSheen + lg.sheen;
      gSheenRoughness = gSheenRoughness + lg.sheenRoughness;
      gSheenColor = gSheenColor + lg.sheenColor;
      gIridescence = gIridescence + lg.iridescenceGrad;
      gIridescenceIor = gIridescenceIor + lg.iridescenceIorGrad;
      gIridescenceThicknessRange = gIridescenceThicknessRange + lg.iridescenceThicknessRangeGrad;
      gAnisotropy = gAnisotropy + lg.anisotropyGrad;
      gAnisotropyRotation = gAnisotropyRotation + lg.anisotropyRotationGrad;
      if (!isUnlit) {
        gNormalScale = gNormalScale + directLightNormalStackScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dNormal_dNormalScale, dClearcoatNormal_dNormalScale,
          wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, LiPerMisUnit, lightPdf,
        );
        gBumpScale = gBumpScale + directLightNormalStackScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dNormal_dBumpScale, dClearcoatNormal_dBumpScale,
          wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, LiPerMisUnit, lightPdf,
        );
        gClearcoatNormalScale = gClearcoatNormalScale + directLightClearcoatNormalScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dClearcoatNormal_dClearcoatNormalScale, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, LiPerMisUnit, lightPdf,
        );
        let areaFactor = misWeight / lightPdf;
        let brdfValue = directLightBrdfValue(
          effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation,
        );
        scatterEmitterRadianceGradient(
          ${ADJOINT_EMITTER_TARGET_RECT}u,
          ri,
          dLoss_dR * brdfValue * (nDotL * areaFactor),
          vec3f(1.0),
          invReplaySamples,
        );
      }
    }

    // Mesh-area lights: stochastic triangle-area replay of each packed emissive
    // source triangle. This mirrors the forward NEE
    // sampler and its per-light MIS weight without pretending to replay the
    // forward one-of-N light-selection lottery.
    for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
      let mb = mi * 7u;
      let a = meshAreaLights[mb].xyz;
      let b = meshAreaLights[mb + 1u].xyz;
      let c = meshAreaLights[mb + 2u].xyz;
      let mr = meshAreaLights[mb + 3u];
      var sourceOwnerSlot = 0xffffffffu;
      if (mi < arrayLength(&meshAreaLightSourceFactors)) {
        let sourceRecord = meshAreaLightSourceFactors[mi];
        let ownerToken = u32(max(sourceRecord.w, 0.0) + 0.5);
        if (ownerToken > 0u) {
          sourceOwnerSlot = ownerToken - 1u;
        }
      }
      let r1 = rand_f32(&rng);
      let r2 = rand_f32(&rng);
      let su = sqrt(r1);
      let uu = 1.0 - su;
      let vv = r2 * su;
      let ww = 1.0 - uu - vv;
      let lpos = a * uu + b * vv + c * ww;
      let sourceFactor = sampleAdjointMeshEmitterSourceFactor(
        mi, vec3f(uu, vv, ww), lpos,
      );
      let sampledRadiance = meshAreaLights[mb + 6u].rgb * sourceFactor;
      let edgeCross = cross(b - a, c - a);
      let area = 0.5 * length(edgeCross);
      let lightNormal = safe_normalize(edgeCross);
      let toLight = lpos - pos;
      let dist2 = dot(toLight, toLight);
      if (dist2 <= 0.0 || area <= 0.0) { continue; }
      let dist = sqrt(dist2);
      let wi = toLight / dist;
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      let cosLight = max(dot(lightNormal, -wi), 0.0);
      if (cosLight <= 0.0) { continue; }
      if (mr.w <= 0.5 && anyHit(pos + n * 1e-3, wi, max(dist - 2e-3, 1e-3))) { continue; }
      let lightPdf = dist2 / (cosLight * area);
      let LiPerMisUnit = sampledRadiance / lightPdf;
      let misWeight = adjointDirectLightMisWeight(
        effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi,
        effectiveSpecularColor, effectiveSpecularIntensity,
        effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness,
        effectiveAnisotropy, effectiveAnisotropyRotation, lightPdf,
      );
      let lg = directLightAdjointFull(
        dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
        effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
        effectiveIridescence,
        iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
        iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample,
        effectiveAnisotropy, effectiveAnisotropyRotation,
        LiPerMisUnit, lightPdf,
      );
      gBaseColor = gBaseColor + lg.baseColor;
      gRough = gRough + lg.roughness;
      gSpecularColor = gSpecularColor + lg.specularColor;
      gSpecularIntensity = gSpecularIntensity + lg.specularIntensity;
      gMetallic = gMetallic + lg.metallicGrad;
      gClearcoat = gClearcoat + lg.clearcoat;
      gClearcoatRoughness = gClearcoatRoughness + lg.clearcoatRoughness;
      gSheen = gSheen + lg.sheen;
      gSheenRoughness = gSheenRoughness + lg.sheenRoughness;
      gSheenColor = gSheenColor + lg.sheenColor;
      gIridescence = gIridescence + lg.iridescenceGrad;
      gIridescenceIor = gIridescenceIor + lg.iridescenceIorGrad;
      gIridescenceThicknessRange = gIridescenceThicknessRange + lg.iridescenceThicknessRangeGrad;
      gAnisotropy = gAnisotropy + lg.anisotropyGrad;
      gAnisotropyRotation = gAnisotropyRotation + lg.anisotropyRotationGrad;
      if (!isUnlit) {
        gNormalScale = gNormalScale + directLightNormalStackScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dNormal_dNormalScale, dClearcoatNormal_dNormalScale,
          wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, LiPerMisUnit, lightPdf,
        );
        gBumpScale = gBumpScale + directLightNormalStackScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dNormal_dBumpScale, dClearcoatNormal_dBumpScale,
          wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, LiPerMisUnit, lightPdf,
        );
        gClearcoatNormalScale = gClearcoatNormalScale + directLightClearcoatNormalScaleGradient(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
          dClearcoatNormal_dClearcoatNormalScale, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation, LiPerMisUnit, lightPdf,
        );
        let areaFactor = misWeight / lightPdf;
        let brdfValue = directLightBrdfValue(
          effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          effectiveAnisotropy, effectiveAnisotropyRotation,
        );
        scatterEmitterRadianceGradient(
          ${ADJOINT_EMITTER_TARGET_MESH}u,
          sourceOwnerSlot,
          dLoss_dR * brdfValue * (nDotL * areaFactor),
          sourceFactor,
          invReplaySamples,
        );
      }
    }

    // Environment NEE: stochastic CDF replay of the same equirect/procedural-sky
    // environment map the forward direct-light branch samples. Like the finite
    // area-light adjoint above, this estimates the source expectation directly
    // (radiance·MIS / pdf) instead of replaying the forward one-of-N light-selection
    // lottery. Environment BSDF-escape / indirect paths remain outside the scoped
    // single-bounce adjoint regime and are guarded by the render-regime gate.
    let envSample = sampleAdjointEnvironmentImportance(&rng);
    if (envSample.pdf > 0.0) {
      let wi = envSample.wi;
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL > 0.0 && !anyHit(pos + n * 1e-3, wi, 1e30)) {
        let envLiPerUnitIntensity = envSample.value / envSample.pdf;
        let envMapIntensity = adjointMaterialEnvMapIntensity(matId);
        let LiPerMisUnit = envLiPerUnitIntensity * envMapIntensity;
        let misWeight = adjointDirectLightMisWeight(
          effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi,
          effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness,
          effectiveAnisotropy, effectiveAnisotropyRotation, envSample.pdf,
        );
        let lg = directLightAdjointFull(
          dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveIridescence,
          iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
          iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample,
          effectiveAnisotropy, effectiveAnisotropyRotation,
          LiPerMisUnit, envSample.pdf,
        );
        gBaseColor = gBaseColor + lg.baseColor;
        gRough = gRough + lg.roughness;
        gSpecularColor = gSpecularColor + lg.specularColor;
        gSpecularIntensity = gSpecularIntensity + lg.specularIntensity;
        gMetallic = gMetallic + lg.metallicGrad;
        gClearcoat = gClearcoat + lg.clearcoat;
        gClearcoatRoughness = gClearcoatRoughness + lg.clearcoatRoughness;
        gSheen = gSheen + lg.sheen;
        gSheenRoughness = gSheenRoughness + lg.sheenRoughness;
        gSheenColor = gSheenColor + lg.sheenColor;
        gIridescence = gIridescence + lg.iridescenceGrad;
        gIridescenceIor = gIridescenceIor + lg.iridescenceIorGrad;
        gIridescenceThicknessRange = gIridescenceThicknessRange + lg.iridescenceThicknessRangeGrad;
        gAnisotropy = gAnisotropy + lg.anisotropyGrad;
        gAnisotropyRotation = gAnisotropyRotation + lg.anisotropyRotationGrad;
        if (!isUnlit) {
          gNormalScale = gNormalScale + directLightNormalStackScaleGradient(
            dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
            dNormal_dNormalScale, dClearcoatNormal_dNormalScale,
            wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
            effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
            effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
            effectiveAnisotropy, effectiveAnisotropyRotation, LiPerMisUnit, envSample.pdf,
          );
          gBumpScale = gBumpScale + directLightNormalStackScaleGradient(
            dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
            dNormal_dBumpScale, dClearcoatNormal_dBumpScale,
            wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
            effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
            effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
            effectiveAnisotropy, effectiveAnisotropyRotation, LiPerMisUnit, envSample.pdf,
          );
          gClearcoatNormalScale = gClearcoatNormalScale + directLightClearcoatNormalScaleGradient(
            dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal,
            dClearcoatNormal_dClearcoatNormalScale, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
            effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
            effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
            effectiveAnisotropy, effectiveAnisotropyRotation, LiPerMisUnit, envSample.pdf,
          );
          let brdfValue = directLightBrdfValue(
            effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, clearcoatNormal, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
            effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
            effectiveIridescence, iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
            effectiveAnisotropy, effectiveAnisotropyRotation,
          );
          gEnvMapIntensity = gEnvMapIntensity + dot(dLoss_dR, brdfValue * (nDotL * envLiPerUnitIntensity * misWeight));
        }
      }
    }

    // Scatter into the gradient slot of every param that targets THIS hit's material
    // (the matId gate is what makes the emissive gradient respond to the optimized
    // primitive's own pixels — a pixel whose primary hit is a different material
    // contributes nothing to that primitive's emissive slot). Scale by
    // 1/sampleCount because the baseline render is the mean of the same frozen
    // sample sequence.
    for (var k = 0u; k < params.paramCount; k = k + 1u) {
      let descBase = k * 2u;
      let d = adjointParamDescs[descBase];
      let payload = adjointParamDescs[descBase + 1u];
      if (d.y == ${ADJOINT_FIELD_EMITTER_COLOR}u || d.y == ${ADJOINT_FIELD_EMITTER_INTENSITY}u) { continue; }
      if (d.x != matId) { continue; }
      // An unlit primary hit has no BRDF, environment, or normal response.
      // Only its base/occlusion colour and additive emission/light-map terms
      // remain differentiable. Skip the other slots rather than scattering a
      // numerically-computed value so their gradients are bit-exact zero.
      if (isUnlit &&
          d.y != ${ADJOINT_FIELD_BASECOLOR}u &&
          d.y != ${ADJOINT_FIELD_AO_MAP_INTENSITY}u &&
          d.y != ${ADJOINT_FIELD_LIGHT_MAP_INTENSITY}u &&
          d.y != ${ADJOINT_FIELD_EMISSIVE}u &&
          d.y != ${ADJOINT_FIELD_EMISSIVE_INTENSITY}u) {
        continue;
      }
      let gradOffset = d.z;
      if (d.y == ${ADJOINT_FIELD_BASECOLOR}u) {
        let gUnlitBaseColor = dLoss_dR * baseColorFactor;
        let gBase = select(gBaseColor * baseColorFactor, gUnlitBaseColor, isUnlit);
        adjointScatter(gradOffset, gBase.x * invReplaySamples);
        adjointScatter(gradOffset + 1u, gBase.y * invReplaySamples);
        adjointScatter(gradOffset + 2u, gBase.z * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_ROUGHNESS}u) {
        adjointScatter(gradOffset, gRough * ormFactor.g * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_SPECULAR_COLOR}u) {
        adjointScatter(gradOffset, gSpecularColor.x * specularColorFactor.x * invReplaySamples);
        adjointScatter(gradOffset + 1u, gSpecularColor.y * specularColorFactor.y * invReplaySamples);
        adjointScatter(gradOffset + 2u, gSpecularColor.z * specularColorFactor.z * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_SPECULAR_INTENSITY}u) {
        adjointScatter(gradOffset, gSpecularIntensity * specularIntensityFactor * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_METALLIC}u) {
        adjointScatter(gradOffset, gMetallic * ormFactor.b * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_AO_MAP_INTENSITY}u) {
        let gAoBase = baseColor * baseColorNoAoFactor * aoSample.dFactor_dIntensity;
        let gAoMapIntensity = select(dot(gBaseColor, gAoBase), dot(dLoss_dR, gAoBase), isUnlit);
        adjointScatter(gradOffset, gAoMapIntensity * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_LIGHT_MAP_INTENSITY}u) {
        let gLightMapIntensity = dot(
          dLoss_dR,
          dContribution_dEmissiveIntensity(vec3f(1.0), lightMapRadiancePerUnitIntensity),
        );
        adjointScatter(gradOffset, gLightMapIntensity * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_ENV_MAP_INTENSITY}u) {
        adjointScatter(gradOffset, gEnvMapIntensity * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_NORMAL_SCALE}u) {
        adjointScatter(gradOffset, gNormalScale * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_BUMP_SCALE}u) {
        adjointScatter(gradOffset, gBumpScale * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_CLEARCOAT_NORMAL_SCALE}u) {
        adjointScatter(gradOffset, gClearcoatNormalScale * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_CLEARCOAT}u) {
        adjointScatter(gradOffset, gClearcoat * clearcoatFactor * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_CLEARCOAT_ROUGHNESS}u) {
        adjointScatter(gradOffset, gClearcoatRoughness * clearcoatRoughnessFactor * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_SHEEN}u) {
        adjointScatter(gradOffset, gSheen * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_SHEEN_ROUGHNESS}u) {
        adjointScatter(gradOffset, gSheenRoughness * sheenRoughnessFactor * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_SHEEN_COLOR}u) {
        adjointScatter(gradOffset, gSheenColor.x * sheenColorFactor.x * invReplaySamples);
        adjointScatter(gradOffset + 1u, gSheenColor.y * sheenColorFactor.y * invReplaySamples);
        adjointScatter(gradOffset + 2u, gSheenColor.z * sheenColorFactor.z * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_IRIDESCENCE}u) {
        adjointScatter(gradOffset, gIridescence * iridescenceGradientFactor * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_IRIDESCENCE_IOR}u) {
        adjointScatter(gradOffset, gIridescenceIor * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_IRIDESCENCE_THICKNESS_RANGE}u) {
        adjointScatter(gradOffset, gIridescenceThicknessRange.x * invReplaySamples);
        adjointScatter(gradOffset + 1u, gIridescenceThicknessRange.y * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_ANISOTROPY}u) {
        adjointScatter(gradOffset, gAnisotropy * anisotropyMapSample.strength * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_ANISOTROPY_ROTATION}u) {
        adjointScatter(
          gradOffset,
          gAnisotropyRotation * anisotropyOrientation.localRotationDerivative *
            invReplaySamples,
        );
      } else if (d.y == ${ADJOINT_FIELD_EMISSIVE}u) {
        // ∂loss/∂emissive_c = dLoss_dR_c · emissiveIntensity. The packed material
        // folds intensity into emissive.rgb, so the host hands the fixed
        // emissiveIntensity in the descriptor's .w (bitcast f32); the partial per
        // unit intensity is (1,1,1) at the primary hit (throughput = 1).
        let emissiveIntensity = bitcast<f32>(d.w);
        let gEmissive = dLoss_dR * dRendered_dEmissivePerUnitIntensity * emissiveTexel * emissiveIntensity;
        adjointScatter(gradOffset, gEmissive.x * invReplaySamples);
        adjointScatter(gradOffset + 1u, gEmissive.y * invReplaySamples);
        adjointScatter(gradOffset + 2u, gEmissive.z * invReplaySamples);
      } else if (d.y == ${ADJOINT_FIELD_EMISSIVE_INTENSITY}u) {
        // ∂loss/∂emissiveIntensity = Σ_c dLoss_dR_c · emissive_c. The host
        // descriptor carries unfactored emissive RGB in payload.xyz; using packed
        // material emissive/intensity would be undefined when intensity is zero.
        let emissiveRgb = vec3f(
          bitcast<f32>(payload.x),
          bitcast<f32>(payload.y),
          bitcast<f32>(payload.z),
        );
        let gIntensity = dot(
          dLoss_dR,
          dContribution_dEmissiveIntensity(vec3f(1.0), emissiveRgb * emissiveTexel),
        );
        adjointScatter(gradOffset, gIntensity * invReplaySamples);
      } else {
        // Unknown field codes are rejected before descriptor upload.
      }
    }
  }
}
`;
}

/** Default PCG composition retained for direct imports and shader tests. */
export const PT_WEBGPU_ADJOINT_PASS_WGSL = composePtWebgpuAdjointPassWgsl();
