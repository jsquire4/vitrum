/**
 * adjointPass.wgsl.ts — the engine-side WS5 Phase-1 path-replay adjoint COMPUTE
 * PASS (the last V24 piece). For each pixel it re-traces the frozen-seed primary
 * ray (brute-force closest-hit), re-derives the single-bounce direct lighting
 * (point-light NEE with shadow rays — the SAME `rad/dist²` model + packing the
 * forward `kernel.wgsl.ts` uses), and accumulates `∂loss/∂θ` for the optimized
 * material parameters through the two GPU-VALIDATED adjoint stages:
 *   - the BRDF partials `dBrdf_dBaseColor` / `dBrdf_dRoughness`
 *     (`pathTraceAdjoint.wgsl.ts`, GPU == FD oracle to f32),
 *   - the chain rule + fixed-point `adjointScatter` accumulation
 *     (`adjointHarness.wgsl.ts`, analytic == on-device FD).
 *
 * It deliberately does NOT call the forward `evaluateBrdf` — the per-pixel
 * `dLoss/dRendered` is handed in by `inverseSession` (computed from the baseline
 * render vs target), so the pass only needs the DERIVATIVES of the shading.
 *
 * Scope (Phase 1, matching the differentiable set): single bounce, brute-force
 * intersection (Phase-1 inverse scenes are small — Cornell-scale), directional
 * delta + point + spot + stochastic area-measure rect/disc/mesh-area direct lights.
 * Environment, indirect, soft-sun angular diameter, and BRDF/transmissive/
 * layered/volume/spectral mapped material terms remain deliberate
 * finite-difference fallbacks until their source terms are mirrored here and
 * GPU-validated. Mapped terms replayed here are scoped to the camera-direct
 * emissive texel multiplier for `emissive` / `emissiveIntensity`, plus
 * baseColorMap / COLOR_0 / aoMap, roughnessMap / metallicMap, clearcoat/sheen/
 * iridescence/anisotropy maps, and specular color/intensity local factors used
 * by lit direct BRDF derivatives.
 * Direct lights are summed over all eligible lights (no MC light selection:
 * the adjoint estimates the direct-light expectation for each source; finite
 * area lights use the same area-measure surface samples as the forward NEE
 * branch). baseColor/roughness/metallic/specular/clearcoat/
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
import { PCG_WGSL } from '@vitrum/shared-samplers';
import {
  MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_UV_META_VEC4S_PER_MAP,
  MATERIAL_TEX_VEC4_STRIDE,
} from '../../scene/materialTextures.js';
import { PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL } from './pathTraceAdjoint.wgsl.js';

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

export const ADJOINT_EMITTER_TARGET_DIRECTIONAL = 1;
export const ADJOINT_EMITTER_TARGET_POINT = 2;
export const ADJOINT_EMITTER_TARGET_SPOT = 3;
export const ADJOINT_EMITTER_TARGET_RECT = 4;
export const ADJOINT_EMITTER_TARGET_MESH = 5;

/** AdjointParams UBO size in bytes (mat4 + vec4 + 3×uvec4). */
export const ADJOINT_PARAMS_UBO_BYTES = 64 + 16 + 16 + 16 + 16;

const ADJOINT_MATERIAL_TEX_UV_EMISSIVE =
  MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP;
const ADJOINT_MATERIAL_TEX_UV_BASE_COLOR = MATERIAL_TEX_UV_META_VEC4_OFFSET;
const ADJOINT_MATERIAL_TEX_UV_ROUGHNESS =
  MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 3;
const ADJOINT_MATERIAL_TEX_UV_METALLIC =
  MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 4;
const ADJOINT_MATERIAL_TEX_UV_AO =
  MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 5;
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

export const PT_WEBGPU_ADJOINT_PASS_WGSL = /* wgsl */ `
const PI = 3.14159265358979;
const INV_PI = 0.31830988618;
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
// 8 bits and a contiguous range count in the upper bits. Mesh-area emitter
// targets use this only for uncapped explicit emitters whose packed triangles
// remain contiguous; capped/reordered mesh lights and full stochastic sampling
// stay on finite difference.
@group(0) @binding(9) var<storage, read>       adjointParamDescs: array<vec4u>;
// rect-area lights: per light {position, uAxis, vAxis, radiance} (4 vec4 stride).
@group(0) @binding(10) var<storage, read>      rectAreaLights: array<vec4f>;
// directional lights: per light {towardLight+angularDiameter, irradiance+mean} (2 vec4 stride).
@group(0) @binding(11) var<storage, read>      directionalLights: array<vec4f>;
// spot lights: per light {position, axis+cosOuter, radiance+cosInner, distance+decay+shadowFlag} (4 vec4 stride).
@group(0) @binding(12) var<storage, read>      spotLights: array<vec4f>;
// mesh-area lights: per triangle {a, b, c, radiance+shadowFlag} (4 vec4 stride).
@group(0) @binding(13) var<storage, read>      meshAreaLights: array<vec4f>;
// Material-map replay subset: mirrors the forward texture samplers for local
// base/ORM/AO/specular/clearcoat/sheen/iridescence/anisotropy chain factors and
// camera-direct emissive partials. Path-changing maps (alpha, transmission,
// normal/bump/clearcoat-normal, displacement) still route through finite
// difference until their visibility/transport/normal terms are replayed here.
@group(0) @binding(14) var<storage, read>      meshUvs: array<vec4f>;
@group(0) @binding(15) var<storage, read>      materialTexDescriptors: array<vec4f>;
@group(0) @binding(16) var                      materialTextures: texture_2d_array<f32>;
@group(0) @binding(17) var                      materialTexSampler: sampler;
@group(0) @binding(18) var<storage, read>       meshVertexColors: array<vec4f>;
@group(0) @binding(19) var                      materialTexturesLinear: texture_2d_array<f32>;

// ── BRDF primitives ──────────────────────────────────────────────────────────
const ADJOINT_FROZEN_SEED_BASE = 0x5eed5eedu;
${PCG_WGSL}
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
fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  let m5 = m2 * m2 * m;
  return f0 + (vec3f(1.0) - f0) * m5;
}

// ── emissive texture replay subset (mirror of material.wgsl sampleEmissiveTexture) ──
const ADJOINT_MATERIAL_TEX_VEC4_STRIDE = ${MATERIAL_TEX_VEC4_STRIDE}u;
const ADJOINT_MATERIAL_TEX_UV_BASE_COLOR = ${ADJOINT_MATERIAL_TEX_UV_BASE_COLOR}u;
const ADJOINT_MATERIAL_TEX_UV_EMISSIVE = ${ADJOINT_MATERIAL_TEX_UV_EMISSIVE}u;
const ADJOINT_MATERIAL_TEX_UV_ROUGHNESS = ${ADJOINT_MATERIAL_TEX_UV_ROUGHNESS}u;
const ADJOINT_MATERIAL_TEX_UV_METALLIC = ${ADJOINT_MATERIAL_TEX_UV_METALLIC}u;
const ADJOINT_MATERIAL_TEX_UV_AO = ${ADJOINT_MATERIAL_TEX_UV_AO}u;
const ADJOINT_MATERIAL_TEX_UV_CLEARCOAT = ${ADJOINT_MATERIAL_TEX_UV_CLEARCOAT}u;
const ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_ROUGHNESS = ${ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_ROUGHNESS}u;
const ADJOINT_MATERIAL_TEX_UV_SHEEN_COLOR = ${ADJOINT_MATERIAL_TEX_UV_SHEEN_COLOR}u;
const ADJOINT_MATERIAL_TEX_UV_SHEEN_ROUGHNESS = ${ADJOINT_MATERIAL_TEX_UV_SHEEN_ROUGHNESS}u;
const ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE = ${ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE}u;
const ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE_THICKNESS = ${ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE_THICKNESS}u;
const ADJOINT_MATERIAL_TEX_UV_SPECULAR_COLOR = ${ADJOINT_MATERIAL_TEX_UV_SPECULAR_COLOR}u;
const ADJOINT_MATERIAL_TEX_UV_SPECULAR_INTENSITY = ${ADJOINT_MATERIAL_TEX_UV_SPECULAR_INTENSITY}u;
const ADJOINT_MATERIAL_TEX_UV_ANISOTROPY = ${ADJOINT_MATERIAL_TEX_UV_ANISOTROPY}u;

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

fn sampleAdjointMaterialLayer(layerIdx: i32, base: u32, triIndex: u32, baryVW: vec2f, uvMetaOffset: u32, uvFitScale: vec2f, wrapMode: vec2f) -> vec4f {
  if (layerIdx < 0 || triIndex >= arrayLength(&indices) || base + uvMetaOffset + 1u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs)) {
    return vec4f(1.0);
  }
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let uva = meshUvs[tri.x];
  let uvb = meshUvs[tri.y];
  let uvc = meshUvs[tri.z];
  let ch0 = uva.xy * u + uvb.xy * v + uvc.xy * w;
  let ch1 = uva.zw * u + uvb.zw * v + uvc.zw * w;
  let uvMeta = materialTexDescriptors[base + uvMetaOffset];
  let uvScale = materialTexDescriptors[base + uvMetaOffset + 1u];
  let texCoord = u32(uvMeta.x);
  let rawUv = select(ch0, ch1, texCoord == 1u);
  let xform = vec4f(uvMeta.y, uvMeta.z, uvScale.x, uvScale.y);
  let rot = uvMeta.w;
  let c = cos(rot);
  let s = sin(rot);
  let sx = xform.z;
  let sy = xform.w;
  let rawA = select(uva.xy, uva.zw, texCoord == 1u);
  let rawB = select(uvb.xy, uvb.zw, texCoord == 1u);
  let rawC = select(uvc.xy, uvc.zw, texCoord == 1u);
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
  let wrappedUv = vec2f(adjointWrapTextureCoord(uv.x, wrapMode.x), adjointWrapTextureCoord(uv.y, wrapMode.y));
  let fittedUv = wrappedUv * uvFitScale;
  let texDim = vec2f(textureDimensions(materialTextures, 0));
  let mipCount = f32(textureNumLevels(materialTextures));
  let texelArea = max(abs((uvB.x - uvA.x) * (uvC.y - uvA.y) - (uvB.y - uvA.y) * (uvC.x - uvA.x)) * texDim.x * texDim.y, 1.0);
  let pa = positions[tri.x].xyz;
  let pb = positions[tri.y].xyz;
  let pc = positions[tri.z].xyz;
  let worldArea = max(0.5 * length(cross(pb - pa, pc - pa)), 1e-8);
  let hitPos = pa * u + pb * v + pc * w;
  let cameraDistance = max(length(hitPos - params.cameraPos.xyz), 1e-3);
  let pixelsPerMeter = 0.5 * f32(max(params.width, params.height)) / cameraDistance;
  let projectedPixels = max(sqrt(worldArea) * pixelsPerMeter, 1.0);
  let lod = clamp(log2(sqrt(texelArea) / projectedPixels), 0.0, max(mipCount - 1.0, 0.0));
  return textureSampleLevel(materialTextures, materialTexSampler, fittedUv, layerIdx, lod);
}

fn sampleAdjointMaterialLayerLinear(layerIdx: i32, base: u32, triIndex: u32, baryVW: vec2f, uvMetaOffset: u32, uvFitScale: vec2f, wrapMode: vec2f) -> vec4f {
  if (layerIdx < 0 || triIndex >= arrayLength(&indices) || base + uvMetaOffset + 1u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs)) {
    return vec4f(1.0);
  }
  let v = baryVW.x;
  let w = baryVW.y;
  let u = 1.0 - v - w;
  let uva = meshUvs[tri.x];
  let uvb = meshUvs[tri.y];
  let uvc = meshUvs[tri.z];
  let ch0 = uva.xy * u + uvb.xy * v + uvc.xy * w;
  let ch1 = uva.zw * u + uvb.zw * v + uvc.zw * w;
  let uvMeta = materialTexDescriptors[base + uvMetaOffset];
  let uvScale = materialTexDescriptors[base + uvMetaOffset + 1u];
  let texCoord = u32(uvMeta.x);
  let rawUv = select(ch0, ch1, texCoord == 1u);
  let xform = vec4f(uvMeta.y, uvMeta.z, uvScale.x, uvScale.y);
  let rot = uvMeta.w;
  let c = cos(rot);
  let s = sin(rot);
  let sx = xform.z;
  let sy = xform.w;
  let rawA = select(uva.xy, uva.zw, texCoord == 1u);
  let rawB = select(uvb.xy, uvb.zw, texCoord == 1u);
  let rawC = select(uvc.xy, uvc.zw, texCoord == 1u);
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
  let wrappedUv = vec2f(adjointWrapTextureCoord(uv.x, wrapMode.x), adjointWrapTextureCoord(uv.y, wrapMode.y));
  let fittedUv = wrappedUv * uvFitScale;
  let texDim = vec2f(textureDimensions(materialTexturesLinear, 0));
  let mipCount = f32(textureNumLevels(materialTexturesLinear));
  let texelArea = max(abs((uvB.x - uvA.x) * (uvC.y - uvA.y) - (uvB.y - uvA.y) * (uvC.x - uvA.x)) * texDim.x * texDim.y, 1.0);
  let pa = positions[tri.x].xyz;
  let pb = positions[tri.y].xyz;
  let pc = positions[tri.z].xyz;
  let worldArea = max(0.5 * length(cross(pb - pa, pc - pa)), 1e-8);
  let hitPos = pa * u + pb * v + pc * w;
  let cameraDistance = max(length(hitPos - params.cameraPos.xyz), 1e-3);
  let pixelsPerMeter = 0.5 * f32(max(params.width, params.height)) / cameraDistance;
  let projectedPixels = max(sqrt(worldArea) * pixelsPerMeter, 1.0);
  let lod = clamp(log2(sqrt(texelArea) / projectedPixels), 0.0, max(mipCount - 1.0, 0.0));
  return textureSampleLevel(materialTexturesLinear, materialTexSampler, fittedUv, layerIdx, lod);
}

fn sampleAdjointEmissiveTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> vec4f {
  let base = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE;
  if (base + 13u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  return sampleAdjointMaterialLayer(
    i32(materialTexDescriptors[base].w),
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_EMISSIVE,
    materialTexDescriptors[base + 7u].zw,
    materialTexDescriptors[base + 13u].zw,
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
  ).g;
  let metallic = sampleAdjointMaterialLayerLinear(
    i32(materialTexDescriptors[base + 6u].z),
    base,
    triIndex,
    baryVW,
    ADJOINT_MATERIAL_TEX_UV_METALLIC,
    materialTexDescriptors[base + 9u].xy,
    materialTexDescriptors[base + 15u].xy,
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
  ).r, 0.0, 1.0);
  return AdjointAoSample(mix(1.0, r, intensity), r - 1.0);
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
  var cr: f32;
  var cphi: f32;
  if (abs(a) >= abs(b)) {
    cr = a;
    cphi = (PI / 4.0) * (b / max(abs(a), 1e-9));
  } else {
    cr = b;
    cphi = (PI / 2.0) - (PI / 4.0) * (a / max(abs(b), 1e-9));
  }
  return vec2f(cr * cos(cphi), cr * sin(cphi));
}

// ── the GPU-validated BRDF partials + adjointScatter (gradAccum at binding 8) ──
${PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL}

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

fn directLightAdjoint(
  dLoss_dR: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  n: vec3f,
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
  nDotL: f32,
  Li: vec3f,
) -> DirectLightAdjoint {
  let gBaseColor = dLoss_dR * dBrdf_dBaseColorWithSpecular(
    baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
  ) * nDotL * Li;
  let gRough = dot(dLoss_dR, dBrdf_dRoughnessWithSpecular(
    baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
  ) * nDotL * Li);
  let gSpecularColor = dLoss_dR * dBrdf_dSpecularColor(
    baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
  ) * nDotL * Li;
  let gSpecularIntensity = dot(dLoss_dR, dBrdf_dSpecularIntensity(
    baseColor, roughness, metallic, n, wo, wi, specularColor,
  ) * nDotL * Li);
  let gMetallic = dot(dLoss_dR, dBrdf_dMetallic(
    baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
  ) * nDotL * Li);
  let gClearcoat = dot(dLoss_dR, dBrdf_dClearcoat(
    clearcoatRoughness, n, wo, wi,
  ) * nDotL * Li);
  let gClearcoatRoughness = dot(dLoss_dR, dBrdf_dClearcoatRoughness(
    clearcoat, clearcoatRoughness, n, wo, wi,
  ) * nDotL * Li);
  let gSheen = dot(dLoss_dR, dBrdf_dSheen(
    sheenRoughness, sheenColor, n, wo, wi,
  ) * nDotL * Li);
  let gSheenColor = dLoss_dR * dBrdf_dSheenColor(
    sheen, sheenRoughness, n, wo, wi,
  ) * nDotL * Li;
  let gSheenRoughness = dot(dLoss_dR, dBrdf_dSheenRoughness(
    sheen, sheenRoughness, sheenColor, n, wo, wi,
  ) * nDotL * Li);
  let gIridescence = dot(dLoss_dR, dBrdf_dIridescence(
    baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
    iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
  ) * nDotL * Li);
  let gIridescenceIor = dot(dLoss_dR, dBrdf_dIridescenceIor(
    baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
  ) * nDotL * Li);
  let gIridescenceThicknessRangePartial = dBrdf_dIridescenceThicknessRange(
    baseColor, roughness, metallic, n, wo, wi, specularColor, specularIntensity,
    iridescence, iridescenceIor, authoredIridescenceThicknessMin,
    authoredIridescenceThicknessMax, iridescenceThicknessTexel,
  );
  let gIridescenceThicknessRange = vec2f(
    dot(dLoss_dR, gIridescenceThicknessRangePartial.min * nDotL * Li),
    dot(dLoss_dR, gIridescenceThicknessRangePartial.max * nDotL * Li),
  );
  let gAnisotropy = dot(dLoss_dR, dBrdf_dAnisotropy(
    baseColor, roughness, metallic, n, wo, wi,
    anisotropy, anisotropyRotation, specularColor, specularIntensity,
  ) * nDotL * Li);
  let gAnisotropyRotation = dot(dLoss_dR, dBrdf_dAnisotropyRotation(
    baseColor, roughness, metallic, n, wo, wi,
    anisotropy, anisotropyRotation, specularColor, specularIntensity,
  ) * nDotL * Li);
  return DirectLightAdjoint(
    gBaseColor, gRough, gSpecularColor, gSpecularIntensity, gMetallic,
    gClearcoat, gClearcoatRoughness, gSheen, gSheenRoughness, gSheenColor,
    gIridescence, gIridescenceIor, gIridescenceThicknessRange, gAnisotropy, gAnisotropyRotation,
  );
}

fn directLightBrdfValue(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  n: vec3f,
  wo: vec3f,
  wi: vec3f,
  specularColor: vec3f,
  specularIntensity: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> vec3f {
  return adjointEvaluateBrdfWithAnisotropy(
    baseColor, roughness, metallic, n, wo, wi,
    anisotropy, anisotropyRotation, specularColor, specularIntensity,
  ) +
    adjointClearcoatLobe(clearcoat, clearcoatRoughness, n, wo, wi) +
    adjointSheenLobe(sheen, sheenRoughness, sheenColor, n, wo, wi);
}

fn scatterEmitterRadianceGradient(
  targetKind: u32,
  targetSlot: u32,
  dLoss_dPackedRadiance: vec3f,
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
    if (d.y == ${ADJOINT_FIELD_EMITTER_COLOR}u) {
      let gColor = dLoss_dPackedRadiance * emitterIntensity;
      adjointScatter(gradOffset, gColor.x * invReplaySamples);
      adjointScatter(gradOffset + 1u, gColor.y * invReplaySamples);
      adjointScatter(gradOffset + 2u, gColor.z * invReplaySamples);
    } else {
      adjointScatter(gradOffset, dot(dLoss_dPackedRadiance, emitterColor) * invReplaySamples);
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
    var rng = pcgInit(gid.x, gid.y, ADJOINT_FROZEN_SEED_BASE + sampleIdx);
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
    let roughness = clamp(m0.w, 0.02, 1.0);
    let metallic = clamp(m1.w, 0.0, 1.0);
    let ormFactor = sampleAdjointOrmTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z));
    let effectiveRoughness = clamp(roughness * ormFactor.g, 0.02, 1.0);
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
    let effectiveAnisotropyRotation = anisotropyRotation + anisotropyMapSample.rotationOffset;

    let idx = indices[hit.tri];
    let nGeo = safe_normalize(hit.bary.x * normals[idx.x].xyz + hit.bary.y * normals[idx.y].xyz + hit.bary.z * normals[idx.z].xyz);
    // Face the shading normal toward the viewer — the SAME flip the forward shade
    // prologue applies (shadePrologue.wgsl.ts). Without it, back-facing geometry
    // gets nDotL<=0 against an interior light and contributes no gradient.
    let n = select(-nGeo, nGeo, dot(nGeo, ray.direction) < 0.0);
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
    for (var di = 0u; di < params.directionalLightCount; di = di + 1u) {
      let dBase = di * 2u;
      let dDirAD = directionalLights[dBase];
      let dIrrMean = directionalLights[dBase + 1u];
      if (dIrrMean.w <= 1e-6) { continue; }
      // Delta directional adjoint: exact for angularDiameter≈0. Soft-sun cone
      // scenes are kept on finite-difference by inverseSession routing; if this
      // pass is called directly anyway, use the cone center as a stable estimate.
      let wi = safe_normalize(dDirAD.xyz);
      let directionalShadowDisabled = dDirAD.w < 0.0;
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      if (!directionalShadowDisabled && anyHit(pos + n * 1e-3, wi, 1e30)) { continue; }
      let lg = directLightAdjoint(
        dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
        effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
        effectiveIridescence,
        iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
        iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample,
        effectiveAnisotropy, effectiveAnisotropyRotation,
        nDotL, dIrrMean.rgb,
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
        let brdfValue = directLightBrdfValue(
          effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveAnisotropy, effectiveAnisotropyRotation,
        );
        scatterEmitterRadianceGradient(
          ${ADJOINT_EMITTER_TARGET_DIRECTIONAL}u,
          di,
          dLoss_dR * brdfValue * nDotL,
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
      let dist2 = max(dot(toPoint, toPoint), 1e-5);
      let dist = sqrt(dist2);
      if (ptMaxDist > 0.0 && dist > ptMaxDist) { continue; }
      let wi = toPoint / dist;
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      if (ptExtra.z <= 0.5 && anyHit(pos + n * 1e-3, wi, dist - 2e-3)) { continue; } // shadowed
      let attenuation = select(1.0 / dist2, pow(max(dist, 1.0), -ptDecay), ptDecay > 0.01);
      let Li = rad * attenuation;
      let lg = directLightAdjoint(
        dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
        effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
        effectiveIridescence,
        iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
        iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample,
        effectiveAnisotropy, effectiveAnisotropyRotation,
        nDotL, Li,
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
        let brdfValue = directLightBrdfValue(
          effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveAnisotropy, effectiveAnisotropyRotation,
        );
        scatterEmitterRadianceGradient(
          ${ADJOINT_EMITTER_TARGET_POINT}u,
          pi,
          dLoss_dR * brdfValue * (nDotL * attenuation),
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
      let dist2 = max(dot(toSpot, toSpot), 1e-5);
      let dist = sqrt(dist2);
      if (spExtra.x > 0.0 && dist > spExtra.x) { continue; }
      let wi = toSpot / dist;
      let coneCos = dot(-wi, spotDir);
      if (coneCos < cosOuter) { continue; }
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      if (spExtra.z <= 0.5 && anyHit(pos + n * 1e-3, wi, dist - 2e-3)) { continue; }
      let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), coneCos);
      let attenuation = select(1.0 / dist2, pow(max(dist, 1.0), -spExtra.y), spExtra.y > 0.01);
      let Li = sradW.rgb * softness * attenuation;
      let lg = directLightAdjoint(
        dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
        effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
        effectiveIridescence,
        iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
        iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample,
        effectiveAnisotropy, effectiveAnisotropyRotation,
        nDotL, Li,
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
        let brdfValue = directLightBrdfValue(
          effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveAnisotropy, effectiveAnisotropyRotation,
        );
        scatterEmitterRadianceGradient(
          ${ADJOINT_EMITTER_TARGET_SPOT}u,
          si,
          dLoss_dR * brdfValue * (nDotL * softness * attenuation),
          invReplaySamples,
        );
      }
    }

    // Rect/disc-area lights: stochastic area-measure replay of the same geometric
    // term the forward NEE integrates (brdf·nDotL·radiance / pdf_area). The pass
    // still sums every light rather than replaying forward light-selection MIS,
    // but it no longer approximates finite emitters by their centers.
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
        area = max(PI * r * r, 1e-6);
      } else {
        lpos = rpos + ru * (xi1 * 2.0 - 1.0) + rv * (xi2 * 2.0 - 1.0);
        area = max(4.0 * length(cross(ru, rv)), 1e-6);
      }
      let toLight = lpos - pos;
      let dist2 = max(dot(toLight, toLight), 1e-6);
      let dist = sqrt(dist2);
      let wi = toLight / dist;
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      let lightNormal = safe_normalize(cross(ru, rv));
      let cosLight = max(dot(lightNormal, -wi), 0.0);
      if (cosLight <= 0.0) { continue; }
      if (rectAreaLights[rb].w <= 0.5 && anyHit(pos + n * 1e-3, wi, dist - 2e-3)) { continue; } // shadowed
      let lightPdf = dist2 / max(cosLight * area, 1e-6);
      let Li = rad / max(lightPdf, 1e-6);
      let lg = directLightAdjoint(
        dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
        effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
        effectiveIridescence,
        iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
        iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample,
        effectiveAnisotropy, effectiveAnisotropyRotation,
        nDotL, Li,
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
        let areaFactor = 1.0 / max(lightPdf, 1e-6);
        let brdfValue = directLightBrdfValue(
          effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveAnisotropy, effectiveAnisotropyRotation,
        );
        scatterEmitterRadianceGradient(
          ${ADJOINT_EMITTER_TARGET_RECT}u,
          ri,
          dLoss_dR * brdfValue * (nDotL * areaFactor),
          invReplaySamples,
        );
      }
    }

    // Mesh-area lights: stochastic uniform triangle-area replay of each packed
    // emissive triangle. This mirrors the forward NEE triangle sampler without
    // pretending to cover light-selection MIS or exact emissive texel-PDFs.
    for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
      let mb = mi * 4u;
      let a = meshAreaLights[mb].xyz;
      let b = meshAreaLights[mb + 1u].xyz;
      let c = meshAreaLights[mb + 2u].xyz;
      let mr = meshAreaLights[mb + 3u];
      let r1 = rand_f32(&rng);
      let r2 = rand_f32(&rng);
      let su = sqrt(r1);
      let uu = 1.0 - su;
      let vv = r2 * su;
      let ww = 1.0 - uu - vv;
      let lpos = a * uu + b * vv + c * ww;
      let edgeCross = cross(b - a, c - a);
      let area = max(0.5 * length(edgeCross), 1e-6);
      let lightNormal = safe_normalize(edgeCross);
      let toLight = lpos - pos;
      let dist2 = max(dot(toLight, toLight), 1e-6);
      let dist = sqrt(dist2);
      let wi = toLight / dist;
      let nDotL = max(0.0, dot(n, wi));
      if (nDotL <= 0.0) { continue; }
      let cosLight = max(dot(lightNormal, -wi), 0.0);
      if (cosLight <= 0.0) { continue; }
      if (mr.w <= 0.5 && anyHit(pos + n * 1e-3, wi, dist - 2e-3)) { continue; }
      let lightPdf = dist2 / max(cosLight * area, 1e-6);
      let Li = mr.rgb / max(lightPdf, 1e-6);
      let lg = directLightAdjoint(
        dLoss_dR, effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
        effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
        effectiveIridescence,
        iridescenceIor, effectiveIridescenceThicknessMin, effectiveIridescenceThicknessMax,
        iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample,
        effectiveAnisotropy, effectiveAnisotropyRotation,
        nDotL, Li,
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
        let areaFactor = 1.0 / max(lightPdf, 1e-6);
        let brdfValue = directLightBrdfValue(
          effectiveBaseColor, effectiveRoughness, effectiveMetallic, n, wo, wi, effectiveSpecularColor, effectiveSpecularIntensity,
          effectiveClearcoat, effectiveClearcoatRoughness, sheen, effectiveSheenRoughness, effectiveSheenColor,
          effectiveAnisotropy, effectiveAnisotropyRotation,
        );
        scatterEmitterRadianceGradient(
          ${ADJOINT_EMITTER_TARGET_MESH}u,
          mi,
          dLoss_dR * brdfValue * (nDotL * areaFactor),
          invReplaySamples,
        );
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
        adjointScatter(gradOffset, gAnisotropyRotation * invReplaySamples);
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
        adjointScatter(gradOffset, gRough * invReplaySamples);
      }
    }
  }
}
`;
