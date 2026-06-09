import { lightTreeWgsl } from '@vitrum/shared-samplers';

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
  cmfIntegralX: f32,
  cmfIntegralY: f32,
  cmfIntegralZ: f32,
  bdptEnabled: u32,
  bdptMaxLightBounces: u32,
  bdptMaxEyeDepth: u32,
  lightTreeEnabled: u32,
  lightTreeNodeCount: u32,
  _padAuto0: u32,
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
@group(0) @binding(11) var varianceTexture: texture_storage_2d<rgba16float, write>;${extraBindings}

const INVALID_TLAS_INSTANCE_INDEX = 0xffffffffu;
`;
}

/** Bindings 0–11: core mesh path trace + G-buffer aux (≤8 storage buffers, ≤4 storage textures). */
export const PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_BINDINGS_WGSL = frameParamsGroup0Bindings('', '');

export const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP0_WGSL = frameParamsGroup0Bindings(
  ' // UBO-plumbed (D12); default metre-scale',
  `
@group(0) @binding(12) var motionVectorsTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var<storage, read_write> varianceMomentsBuffer: array<vec4f>;`,
);

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

/** Group 2 — TLAS instance table (5 storage buffers) + BDPT light-path scratch
 *  buffer + BDPT eye-subpath scratch stack.
 *
 *  `bdptLightPath` is a read_write storage BUFFER of vec4f (NOT a storage
 *  texture). Core WebGPU only permits `read_write` storage-texture access for
 *  `r32float/uint/sint` (gpuweb #4651); `rgba32float` read_write storage
 *  textures are rejected at bind-group creation on every conformant impl
 *  (Dawn + wgpu-native), so the light-path cache is a storage buffer instead.
 *  Layout: `maxLightBounces` columns × 3 rows of vec4f, flattened row-minor as
 *  `idx = col * BDPT_LIGHT_PATH_ROWS + row` (see `bdptLightPathIndex`). Per
 *  light-vertex: row 0 = pos (+ kind sentinel in .w), row 1 = normal + pdfFwd,
 *  row 2 = throughput + pdfRev.
 *
 *  `bdptEyeStack` is a per-pixel × bdptMaxEyeDepth read_write storage stack of
 *  eye-vertex pdf/pos/normal data (2× vec4 / vertex; specular packed as a
 *  negative-pdfFwd sentinel) consumed by the full Veach §10.3 connection
 *  sweep. */
export const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP2_WGSL = /* wgsl */ `
@group(2) @binding(0) var<storage, read> tlasNodes: array<BVHNode>;
@group(2) @binding(1) var<storage, read> tlasInstanceIndices: array<u32>;
@group(2) @binding(2) var<storage, read> tlasBlasRoots: array<u32>;
@group(2) @binding(3) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(2) @binding(4) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
@group(2) @binding(5) var<storage, read_write> bdptLightPath: array<vec4f>;
@group(2) @binding(6) var<storage, read_write> bdptEyeStack: array<vec4f>;

// Light-path flat index: 3 vec4f rows per light-vertex column.
const BDPT_LIGHT_PATH_ROWS = 3u;
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
export const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL =
  /* wgsl */ `
// ============================================================
// WS2 — light-tree storage buffer (full-tier @group(3)) + importance traversal
// ============================================================
${lightTreeWgsl({ group: 3, binding: 0 })}
// Selection-only proximity floor for the light-tree descent (caps distance
// importance near a light; NOT the NEE geometry-term clamp). Metre-scale.
const LT_DIST2_FLOOR: f32 = 1e-3;

// ============================================================
// P2 — material textures (full-tier @group(3)): per-vertex UVs + per-material
// descriptors + the sampled baseColor texture_2d_array + a filtering sampler.
// A DEDICATED set of bindings in the existing full-tier group so neither the
// lite tier (no group 3) nor the existing group 0/1/2 layouts are perturbed.
// ============================================================
@group(3) @binding(1) var<storage, read> meshUvs: array<vec4f>;       // .xy = uv0, .zw = uv1
@group(3) @binding(2) var<storage, read> materialTexDescriptors: array<vec4f>;
@group(3) @binding(3) var materialTextures: texture_2d_array<f32>;        // sRGB (baseColor + emissive)
@group(3) @binding(4) var materialTexSampler: sampler;                    // shared by both arrays
@group(3) @binding(5) var materialTexturesLinear: texture_2d_array<f32>;  // LINEAR (normal + ORM)

// vec4s per material in the descriptor buffer — MUST match the TS
// MATERIAL_TEX_VEC4_STRIDE in scene/materialTextures.ts.
//   0: {baseColorIdx, normalIdx, ormIdx, emissiveIdx}   (-1 = no map)
//   1: {alphaMode, alphaCutoff, opacity, texCoord}
//   2: {offset.xy, scale.xy}   3: {rotation, _, _, _}
const MATERIAL_TEX_VEC4_STRIDE = 4u;

// Sample array layer \`layerIdx\` for material \`base\` (= matId·stride) at the hit:
// interpolate the per-vertex UV by the hit barycentrics, apply the material's
// KHR_texture_transform, sample the indexed layer. Returns vec4(1) — a no-op
// multiply — when layerIdx < 0 or the hit is not a mesh triangle (analytic shapes
// carry no UVs in v1), so a material lacking that map stays byte-identical.
// textureSampleLevel (explicit LOD) keeps the call valid in non-uniform flow.
// All maps of a material share its baseColor UV transform (v1 simplification).
fn sampleMaterialLayer(layerIdx: i32, base: u32, triIndex: u32, baryVW: vec2f) -> vec4f {
  if (layerIdx < 0 || triIndex >= arrayLength(&indices)) { return vec4f(1.0); }
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
  let texCoord = u32(materialTexDescriptors[base + 1u].w);
  let rawUv = select(ch0, ch1, texCoord == 1u);
  // KHR_texture_transform — matches THREE.Matrix3.setUvTransform (center 0), the
  // convention the importer (three-bindings toTextureRef) extracts offset/repeat/
  // rotation in:  u' = sx·c·u + sx·s·v + tx ;  v' = -sy·s·u + sy·c·v + ty.
  let xform = materialTexDescriptors[base + 2u]; // offset.xy, scale.xy
  let rot = materialTexDescriptors[base + 3u].x;
  let c = cos(rot);
  let s = sin(rot);
  let sx = xform.z;
  let sy = xform.w;
  let uv = vec2f(
    sx * c * rawUv.x + sx * s * rawUv.y + xform.x,
    -sy * s * rawUv.x + sy * c * rawUv.y + xform.y,
  );
  return textureSampleLevel(materialTextures, materialTexSampler, uv, layerIdx, 0.0);
}

// baseColor map (sRGB array) — descriptor vec4[0].x.
fn sampleBaseColorTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> vec4f {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 3u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  return sampleMaterialLayer(i32(materialTexDescriptors[base].x), base, triIndex, baryVW);
}

// emissive map (sRGB array, same layers as baseColor) — descriptor vec4[0].w.
fn sampleEmissiveTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> vec4f {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 3u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  return sampleMaterialLayer(i32(materialTexDescriptors[base].w), base, triIndex, baryVW);
}

// As sampleMaterialLayer, but samples the LINEAR array (materialTexturesLinear)
// — for normal + ORM maps, which must NOT be sRGB-decoded. Standalone (not a
// refactor of sampleMaterialLayer) so the validated sRGB path is untouched;
// WGSL can't pass a texture as an argument, hence the parallel function.
fn sampleMaterialLayerLinear(layerIdx: i32, base: u32, triIndex: u32, baryVW: vec2f) -> vec4f {
  if (layerIdx < 0 || triIndex >= arrayLength(&indices)) { return vec4f(1.0); }
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
  let texCoord = u32(materialTexDescriptors[base + 1u].w);
  let rawUv = select(ch0, ch1, texCoord == 1u);
  let xform = materialTexDescriptors[base + 2u];
  let rot = materialTexDescriptors[base + 3u].x;
  let c = cos(rot);
  let s = sin(rot);
  let sx = xform.z;
  let sy = xform.w;
  let uv = vec2f(
    sx * c * rawUv.x + sx * s * rawUv.y + xform.x,
    -sy * s * rawUv.x + sy * c * rawUv.y + xform.y,
  );
  return textureSampleLevel(materialTexturesLinear, materialTexSampler, uv, layerIdx, 0.0);
}

// ORM map (linear array) — descriptor vec4[0].z. glTF metallicRoughness packing:
// G = roughness, B = metallic (R = occlusion, applied by the caller if present).
// vec4(1) when absent → roughness·1, metallic·1 (no modulation → byte-identical).
fn sampleOrmTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> vec4f {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 3u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  return sampleMaterialLayerLinear(i32(materialTexDescriptors[base].z), base, triIndex, baryVW);
}

// Normal map (linear array) — descriptor vec4[0].y. Perturbs the geometric shading
// normal by the tangent-space normal map. The tangent frame is DERIVED per-hit
// from the triangle's positions + UVs (Lengyel) — no precomputed tangents needed
// — then transformed through the hit TLAS instance and Gram-Schmidt-
// orthonormalized against geomNormal. Returns geomNormal unchanged when there's
// no normal map (→ byte-identical). Merged-BLAS / lite / analytic paths pass the
// invalid instance sentinel and keep the historical local-space tangent.
// Ref: Lengyel, "Computing Tangent Space Basis Vectors for an Arbitrary Mesh".
fn applyNormalMap(matId: u32, triIndex: u32, baryVW: vec2f, geomNormal: vec3f, instanceIndex: u32) -> vec3f {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 3u >= arrayLength(&materialTexDescriptors)) { return geomNormal; }
  let normalIdx = i32(materialTexDescriptors[base].y);
  if (normalIdx < 0 || triIndex >= arrayLength(&indices)) { return geomNormal; }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs) ||
      tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return geomNormal;
  }
  let p0 = positions[tri.x].xyz;
  let e1 = positions[tri.y].xyz - p0;
  let e2 = positions[tri.z].xyz - p0;
  let uv0 = meshUvs[tri.x].xy;
  let duv1 = meshUvs[tri.y].xy - uv0;
  let duv2 = meshUvs[tri.z].xy - uv0;
  let det = duv1.x * duv2.y - duv2.x * duv1.y;
  if (abs(det) < 1e-10) { return geomNormal; }
  let f = 1.0 / det;
  var tangent = f * (duv2.y * e1 - duv1.y * e2);
  if (instanceIndex != INVALID_TLAS_INSTANCE_INDEX && params.tlasNodeCount != 0u) {
    let m = instanceIndex * 4u;
    if (m + 3u < arrayLength(&tlasInstanceLocalToWorld)) {
      let l2w0 = tlasInstanceLocalToWorld[m];
      let l2w1 = tlasInstanceLocalToWorld[m + 1u];
      let l2w2 = tlasInstanceLocalToWorld[m + 2u];
      tangent = transformDirectionCols(l2w0, l2w1, l2w2, tangent);
    }
  }
  // Gram-Schmidt orthonormalize against the (world) geometric normal.
  tangent = tangent - geomNormal * dot(geomNormal, tangent);
  let tlen = length(tangent);
  if (tlen < 1e-8) { return geomNormal; }
  tangent = tangent / tlen;
  let bitangent = cross(geomNormal, tangent);
  let ts = sampleMaterialLayerLinear(normalIdx, base, triIndex, baryVW).xyz;
  let tn = ts * 2.0 - vec3f(1.0); // [0,1] → [-1,1] tangent-space normal
  let perturbed = tangent * tn.x + bitangent * tn.y + geomNormal * tn.z;
  let plen = length(perturbed);
  return select(geomNormal, perturbed / plen, plen > 1e-6);
}

// P2 alpha test — should this hit be treated as TRANSPARENT (the ray passes
// straight through, as if there were no surface here)? Drives glTF alphaMode:
//   opaque (0) → never (returns false immediately → opaque is byte-identical).
//   mask   (1) → pass through where baseColorTexAlpha·opacity < alphaCutoff
//                (hard cutout — foliage, fences, decals).
//   blend  (2) → STOCHASTIC pass-through with probability 1 − alpha·opacity
//                (unbiased screen-door transparency in a path tracer; the
//                converged mean equals true alpha compositing).
// The base-color texture's .a supplies the per-texel alpha (1 when no map, so an
// untextured material with material.opacity<1 still blends/cuts by opacity).
// Ref: glTF 2.0 §3.9.4 (alphaMode); PBR screen-door / stochastic transparency.
fn alphaTestPassThrough(matId: u32, triIndex: u32, baryVW: vec2f, rng: ptr<function, u32>) -> bool {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 3u >= arrayLength(&materialTexDescriptors)) { return false; }
  let alphaMode = u32(materialTexDescriptors[base + 1u].x);
  if (alphaMode == 0u) { return false; } // opaque — byte-identical
  let alphaCutoff = materialTexDescriptors[base + 1u].y;
  let opacity = materialTexDescriptors[base + 1u].z;
  let alpha = sampleBaseColorTexture(matId, triIndex, baryVW).a * opacity;
  if (alphaMode == 1u) { return alpha < alphaCutoff; }   // mask
  return rand_f32(rng) >= alpha;                          // blend (stochastic)
}
`;

export const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_WGSL =
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP0_WGSL +
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP1_WGSL +
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP2_WGSL +
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;

export const PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL = /* wgsl */ `
const LEAFNODE_FLAG = 0xffff0000u;
// MUST stay in lockstep with TS \`MATERIAL_VEC4_STRIDE\` in scene/materialPacking.ts.
// WS4 bumped 22 → 23: vec4 #22 carries volumetric σ_a.rgb + hasSigmaA flag.
const MATERIAL_VEC4_STRIDE = 23u;
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
  isTranslucent: bool,
  // WS4 — Beer-Lambert absorption coefficient σ_a (per channel), derived host-side
  // from attenuationColor/attenuationDistance. hasSigmaA distinguishes a clear
  // medium (σ_a = 0) from "no absorption authored".
  sigmaA: vec3f,
  hasSigmaA: bool,
}

// RFE-03 / fork activeLayerWeight: scalar throughput through face layer at hero λ.
fn activeLayerWeightRgb(layerRgb: vec3f, heroLambda: f32, spectralEnabled: bool) -> vec3f {
  if (!spectralEnabled) {
    return layerRgb;
  }
  let lum = max(luminance(layerRgb), 0.0);
  return heroWavelengthToRgb(heroLambda, lum, 1.0);
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
  let m22Index = m0Index + 22u; // WS4 σ_a vec4
  let m0 = select(vec4f(0.8, 0.8, 0.8, 0.6), materials[m0Index], m0Index < arrayLength(&materials));
  let m1 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m1Index], m1Index < arrayLength(&materials));
  let m2 = select(vec4f(0.0, 1.5, 0.0, 0.0), materials[m2Index], m2Index < arrayLength(&materials));
  let m3 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m3Index], m3Index < arrayLength(&materials));
  let m4 = select(vec4f(1.0, 1.0, 1.0, -1.0), materials[m4Index], m4Index < arrayLength(&materials));
  let m5 = select(vec4f(1.0, 1.0, 1.0, -1.0), materials[m5Index], m5Index < arrayLength(&materials));
  let m6 = select(vec4f(0.0, 0.0, 1.0, 0.0), materials[m6Index], m6Index < arrayLength(&materials));
  let m19 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m19Index], m19Index < arrayLength(&materials));
  let m22 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m22Index], m22Index < arrayLength(&materials));
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
  mat.sigmaA = vec3f(max(m22.x, 0.0), max(m22.y, 0.0), max(m22.z, 0.0));
  mat.hasSigmaA = m22.w > 0.5;
  // A material has a PARTICIPATING MEDIUM the eye path must traverse when it is
  // transmissive AND has either scattering (σ_s) OR Beer-Lambert absorption
  // (σ_a from attenuationColor / a spectral-attenuation curve). The σ_a-only case
  // is exactly chromatic stained glass — pure absorption, no scattering — whose
  // attenuationColor was packed but UNCONSUMED while this gate required σ_s > 0
  // (the medium walk + enteredMedium/exitedMedium never fired). V23, 2026-05-29.
  let hasScattering =
    mat.scatteringCoeff > 0.0 ||
    max(mat.scatteringRgb.x, max(mat.scatteringRgb.y, mat.scatteringRgb.z)) > 0.0;
  mat.isTranslucent =
    mat.transmission > 0.0 && (hasScattering || mat.hasSigmaA || mat.hasSpectralAttenuation);
  return mat;
}
`;

/** Full trace pass — 3 bind groups (≤10 storage buffers per group). */
export const PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL =
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_WGSL + PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL;

/** Compatibility tier for adapters capped at 10 storage buffers / 4 storage textures. */
export const PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_WGSL =
  PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_BINDINGS_WGSL + PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL;
