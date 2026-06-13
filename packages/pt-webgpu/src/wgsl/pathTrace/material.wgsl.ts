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
  // H14-E: HDRI radiance intensity multiplier — separate from environmentSun.w
  // (which drives the procedural-sky sun-strength gate). f32 occupies the same
  // slot (31) formerly held by _padAuto0. Equirect lookup uses this lane so a
  // pure-HDRI scene with sun.w=0 does not silently produce a black environment.
  environmentHdriIntensity: f32,
  cameraPos: vec4f,
  lightDir: vec4f,
  environmentTint: vec4f,
  environmentSun: vec4f,
  invViewProj: mat4x4f,
  viewProj: mat4x4f,
  prevViewProj: mat4x4f,
  // N-directional expansion: total packed directional count read from the
  // directionalLights storage buffer (group 1 binding 10). The kernel loops
  // over params.directionalLightCount records. A single-directional scene keeps
  // directionalLightCount=1 and directionalLights[0] byte-identical to the old
  // lightDir single path (gate unchanged).
  directionalLightCount: u32,
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
 * Binding 14 — liteLightTex   : RGBA32F liteLightTexWidth×1, packed point/spot/
 *                                rect-area light records (same float layout as the
 *                                full-tier pointLights/spotLights/rectAreaLights
 *                                storage buffers; loaded via integer texel index).
 */
const PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_EXTRA_BINDINGS_WGSL = /* wgsl */ `
@group(0) @binding(12) var liteEnvTex:    texture_2d<f32>;
@group(0) @binding(13) var liteEnvCdfTex: texture_2d<f32>;
@group(0) @binding(14) var liteLightTex:  texture_2d<f32>;
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
//   [di*2+1]: irradiance.rgb,  mean_irradiance
// directionalLightCount records total; an empty scene binds a 16-byte placeholder.
@group(1) @binding(10) var<storage, read> directionalLights: array<vec4f>;
`;

/** Group 2 — TLAS instance table (5 storage buffers) + BDPT light-path scratch
 *  buffer + BDPT eye-subpath scratch stack.
 *
 *  `bdptLightPath` is a read_write storage BUFFER of vec4f (NOT a storage
 *  texture). Core WebGPU only permits `read_write` storage-texture access for
 *  `r32float/uint/sint` (gpuweb #4651); `rgba32float` read_write storage
 *  textures are rejected at bind-group creation on every conformant impl
 *  (Dawn + wgpu-native), so the light-path cache is a storage buffer instead.
 *  Layout: `maxLightBounces` columns × 4 rows of vec4f, flattened row-minor as
 *  `idx = col * BDPT_LIGHT_PATH_ROWS + row` (see `bdptLightPathIndex`). Per
 *  light-vertex: row 0 = pos (+ kind sentinel in .w), row 1 = normal + pdfFwd,
 *  row 2 = throughput + pdfRev, row 3 = (A9) matId + wo-toward-prev for the REAL
 *  light-vertex BSDF in the §10.3 connection (matId < 0 ⇒ emitter, Lambertian).
 *
 *  `bdptEyeStack` is a per-pixel × bdptMaxEyeDepth read_write storage stack of
 *  eye-vertex pdf/pos/normal data (2× vec4 / vertex; specular packed as a
 *  negative-pdfFwd sentinel) consumed by the full Veach §10.3 connection
 *  sweep. */
const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP2_WGSL = /* wgsl */ `
@group(2) @binding(0) var<storage, read> tlasNodes: array<BVHNode>;
@group(2) @binding(1) var<storage, read> tlasInstanceIndices: array<u32>;
@group(2) @binding(2) var<storage, read> tlasBlasRoots: array<u32>;
@group(2) @binding(3) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(2) @binding(4) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
@group(2) @binding(5) var<storage, read_write> bdptLightPath: array<vec4f>;
@group(2) @binding(6) var<storage, read_write> bdptEyeStack: array<vec4f>;

// Light-path flat index: 4 vec4f rows per light-vertex column (A9 — row 3 carries
// the reached vertex's matId + wo-toward-prev so the §10.3 connection can evaluate
// the REAL light-vertex BSDF for a glossy/metallic light-path vertex; matId < 0
// marks the emitter vertex, which keeps its Lambertian/emission profile).
const BDPT_LIGHT_PATH_ROWS = 4u;
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
  preXformLines: string,
  xformSuffix: string,
): string {
  return `fn ${name}(layerIdx: i32, base: u32, triIndex: u32, baryVW: vec2f, uvFitScale: vec2f) -> vec4f {
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
  ${preXformLines}let xform = materialTexDescriptors[base + 2u];${xformSuffix}
  let rot = materialTexDescriptors[base + 3u].x;
  let c = cos(rot);
  let s = sin(rot);
  let sx = xform.z;
  let sy = xform.w;
  let uv = vec2f(
    sx * c * rawUv.x + sx * s * rawUv.y + xform.x,
    -sy * s * rawUv.x + sy * c * rawUv.y + xform.y,
  );
  let fittedUv = fract(uv) * uvFitScale;
  return textureSampleLevel(${texArray}, materialTexSampler, fittedUv, layerIdx, 0.0);
}`;
}

// Pre-stamp the two variants outside the template literal so the `${...}`
// interpolations inside the WGSL template stay simple (no nested escaping).
// sRGB variant (materialTextures — baseColor + emissive):
const _SAMPLE_MAT_LAYER_WGSL = materialLayerSamplerWgsl(
  'sampleMaterialLayer',
  'materialTextures',
  // KHR comment block that precedes `let xform` in the sRGB variant:
  '// KHR_texture_transform — matches THREE.Matrix3.setUvTransform (center 0), the\n' +
  '  // convention the importer (three-bindings toTextureRef) extracts offset/repeat/\n' +
  "  // rotation in:  u' = sx·c·u + sx·s·v + tx ;  v' = -sy·s·u + sy·c·v + ty.\n" +
  '  ',
  ' // offset.xy, scale.xy',
);
// Linear variant (materialTexturesLinear — normal + ORM + bump):
const _SAMPLE_MAT_LAYER_LINEAR_WGSL = materialLayerSamplerWgsl(
  'sampleMaterialLayerLinear',
  'materialTexturesLinear',
  '',
  '',
);

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
//   2: {offset.xy, scale.xy}
//   3: {rotation, aoMapIdx, lightMapIdx, bumpMapIdx}      ← D3 (-1 = no map)
//   4: {aoMapIntensity, lightMapIntensity, bumpScale, envMapIntensity}  ← D3
//   5: {anisotropy, anisotropyRotation, anisotropyMapIdx, normalScale}  ← D3/PTWG-MAT
//   6: {alphaMapIdx, transmissionMapIdx, _, _}             (-1 = no map)
//   7: {baseColorUvScale.xy, emissiveUvScale.xy}
//   8: {normalUvScale.xy, ormUvScale.xy}
//   9: {aoUvScale.xy, lightMapUvScale.xy}
//  10: {bumpUvScale.xy, anisotropyUvScale.xy}
//  11: {alphaUvScale.xy, transmissionUvScale.xy}
const MATERIAL_TEX_VEC4_STRIDE = 12u;

// Sample array layer \`layerIdx\` for material \`base\` (= matId·stride) at the hit:
// interpolate the per-vertex UV by the hit barycentrics, apply the material's
// KHR_texture_transform, sample the indexed layer. Returns vec4(1) — a no-op
// multiply — when layerIdx < 0 or the hit is not a mesh triangle (analytic shapes
// carry no UVs in v1), so a material lacking that map stays byte-identical.
// textureSampleLevel (explicit LOD) keeps the call valid in non-uniform flow.
// All maps of a material share its baseColor UV transform (v1 simplification).
${_SAMPLE_MAT_LAYER_WGSL}

// baseColor map (sRGB array) — descriptor vec4[0].x.
fn sampleBaseColorTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> vec4f {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 7u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  return sampleMaterialLayer(i32(materialTexDescriptors[base].x), base, triIndex, baryVW, materialTexDescriptors[base + 7u].xy);
}

// emissive map (sRGB array, same layers as baseColor) — descriptor vec4[0].w.
fn sampleEmissiveTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> vec4f {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 7u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  return sampleMaterialLayer(i32(materialTexDescriptors[base].w), base, triIndex, baryVW, materialTexDescriptors[base + 7u].zw);
}

// As sampleMaterialLayer, but samples the LINEAR array (materialTexturesLinear)
// — for normal + ORM maps, which must NOT be sRGB-decoded. Standalone (not a
// refactor of sampleMaterialLayer) so the validated sRGB path is untouched;
// WGSL can't pass a texture as an argument, hence the parallel function.
${_SAMPLE_MAT_LAYER_LINEAR_WGSL}

// ORM map (linear array) — descriptor vec4[0].z. glTF metallicRoughness packing:
// G = roughness, B = metallic (R = occlusion, applied by the caller if present).
// vec4(1) when absent → roughness·1, metallic·1 (no modulation → byte-identical).
fn sampleOrmTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> vec4f {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 8u >= arrayLength(&materialTexDescriptors)) { return vec4f(1.0); }
  return sampleMaterialLayerLinear(i32(materialTexDescriptors[base].z), base, triIndex, baryVW, materialTexDescriptors[base + 8u].zw);
}

// D9.3 — shared tangent-frame derivation for applyNormalMap + applyBumpMap.
// Builds the world-space (tangent, bitangent) pair for a triangle hit using
// Lengyel's method: position edges + UV deltas → raw tangent → TLAS instance
// transform → Gram-Schmidt against 'normal'.  Both map functions share this
// exact ~25-line block; extracting it keeps the formulas in one place.
// Ref: Lengyel, "Computing Tangent Space Basis Vectors for an Arbitrary Mesh".
struct ShadingTangentFrame {
  tangent: vec3f,
  bitangent: vec3f,
  valid: bool,
}
fn buildShadingTangentFrame(triIndex: u32, normal: vec3f, instanceIndex: u32) -> ShadingTangentFrame {
  var frame: ShadingTangentFrame;
  frame.valid = false;
  let tri = indices[triIndex];
  let p0 = positions[tri.x].xyz;
  let e1 = positions[tri.y].xyz - p0;
  let e2 = positions[tri.z].xyz - p0;
  let uv0 = meshUvs[tri.x].xy;
  let duv1 = meshUvs[tri.y].xy - uv0;
  let duv2 = meshUvs[tri.z].xy - uv0;
  let det = duv1.x * duv2.y - duv2.x * duv1.y;
  if (abs(det) < 1e-10) { return frame; }
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
  tangent = tangent - normal * dot(normal, tangent);
  let tlen = length(tangent);
  if (tlen < 1e-8) { return frame; }
  tangent = tangent / tlen;
  frame.tangent = tangent;
  frame.bitangent = cross(normal, tangent);
  frame.valid = true;
  return frame;
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
fn applyNormalMap(matId: u32, triIndex: u32, baryVW: vec2f, geomNormal: vec3f, instanceIndex: u32) -> vec3f {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 8u >= arrayLength(&materialTexDescriptors)) { return geomNormal; }
  let normalIdx = i32(materialTexDescriptors[base].y);
  if (normalIdx < 0 || triIndex >= arrayLength(&indices)) { return geomNormal; }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs) ||
      tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return geomNormal;
  }
  let frame = buildShadingTangentFrame(triIndex, geomNormal, instanceIndex);
  if (!frame.valid) { return geomNormal; }
  let ts = sampleMaterialLayerLinear(normalIdx, base, triIndex, baryVW, materialTexDescriptors[base + 8u].xy).xyz;
  var tn = ts * 2.0 - vec3f(1.0); // [0,1] → [-1,1] tangent-space normal
  let normalScale = materialTexDescriptors[base + 5u].w;
  tn.x = tn.x * normalScale;
  tn.y = tn.y * normalScale;
  let perturbed = frame.tangent * tn.x + frame.bitangent * tn.y + geomNormal * tn.z;
  let plen = length(perturbed);
  return select(geomNormal, perturbed / plen, plen > 1e-6);
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
fn sampleAoFactor(matId: u32, triIndex: u32, baryVW: vec2f) -> f32 {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 9u >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  let aoIdx = i32(materialTexDescriptors[base + 3u].y);
  if (aoIdx < 0) { return 1.0; }
  let intensity = clamp(materialTexDescriptors[base + 4u].x, 0.0, 1.0);
  let r = sampleMaterialLayerLinear(aoIdx, base, triIndex, baryVW, materialTexDescriptors[base + 9u].xy).r;
  return clamp(mix(1.0, r, intensity), 0.0, 1.0);
}

// Light map (LINEAR array) — descriptor vec4[3].z; intensity vec4[4].y.
// Baked OUTGOING radiance added to the surface emission. SEMANTICS: a light map
// is precomputed *outgoing* radiance, so it is added to \`emissive\` at
// camera-visible (emissive-on-hit) shade points ONLY. Adding it inside NEE would
// double-count the real lights it bakes; the path-tracer's own NEE/indirect
// terms already integrate live light. Returns 0 (no addition) when absent →
// byte-identical. The map is treated as linear data (it is radiance, not albedo);
// hosts that authored an sRGB-encoded light map should decode before upload.
// Ref: glTF lightmap convention; THREE.MeshStandardMaterial.lightMap (additive).
fn sampleLightMapRadiance(matId: u32, triIndex: u32, baryVW: vec2f) -> vec3f {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 9u >= arrayLength(&materialTexDescriptors)) { return vec3f(0.0); }
  let lmIdx = i32(materialTexDescriptors[base + 3u].z);
  if (lmIdx < 0) { return vec3f(0.0); }
  let intensity = max(materialTexDescriptors[base + 4u].y, 0.0);
  return sampleMaterialLayerLinear(lmIdx, base, triIndex, baryVW, materialTexDescriptors[base + 9u].zw).rgb * intensity;
}

// Per-material environment-map intensity scale — descriptor vec4[4].w (default 1).
fn materialEnvMapIntensity(matId: u32) -> f32 {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 5u >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  return max(materialTexDescriptors[base + 4u].w, 0.0);
}

// Anisotropy strength ∈ [0,1] (descriptor vec4[5].x), optionally modulated by the
// KHR_materials_anisotropy map's B channel. 0 ⇒ isotropic (default) ⇒ the caller
// keeps the existing isotropic GGX path → byte-identical.
fn materialAnisotropy(matId: u32, triIndex: u32, baryVW: vec2f) -> f32 {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 10u >= arrayLength(&materialTexDescriptors)) { return 0.0; }
  var a = clamp(materialTexDescriptors[base + 5u].x, 0.0, 1.0);
  let anisoIdx = i32(materialTexDescriptors[base + 5u].z);
  if (anisoIdx >= 0) {
    a = a * sampleMaterialLayerLinear(anisoIdx, base, triIndex, baryVW, materialTexDescriptors[base + 10u].zw).b;
  }
  return clamp(a, 0.0, 1.0);
}

// Anisotropy rotation in radians (descriptor vec4[5].y), optionally offset by the
// anisotropy map's RG direction (KHR_materials_anisotropy: RG encodes a 2D tangent
// rotation as cos/sin in [0,1]→[-1,1]). Returns the scalar rotation when no map.
fn materialAnisotropyRotation(matId: u32, triIndex: u32, baryVW: vec2f) -> f32 {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 10u >= arrayLength(&materialTexDescriptors)) { return 0.0; }
  var rot = materialTexDescriptors[base + 5u].y;
  let anisoIdx = i32(materialTexDescriptors[base + 5u].z);
  if (anisoIdx >= 0) {
    let rg = sampleMaterialLayerLinear(anisoIdx, base, triIndex, baryVW, materialTexDescriptors[base + 10u].zw).rg * 2.0 - vec2f(1.0);
    rot = rot + atan2(rg.y, rg.x);
  }
  return rot;
}

// Bump map (LINEAR height field) — descriptor vec4[3].w; scale vec4[4].z.
// Perturbs the shading normal by the gradient of the height field in UV space,
// finite-differenced from the texture. Mirrors applyNormalMap's tangent-frame
// derivation + TLAS-instance transform + Gram-Schmidt; combines additively with a
// normal map when both are present (apply normal map first, bump second). Returns
// the input normal unchanged when there is no bump map → byte-identical.
// Ref: Blinn 1978, "Simulation of Wrinkled Surfaces"; height-gradient perturbation.
fn applyBumpMap(matId: u32, triIndex: u32, baryVW: vec2f, shadingNormal: vec3f, instanceIndex: u32) -> vec3f {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 10u >= arrayLength(&materialTexDescriptors)) { return shadingNormal; }
  let bumpIdx = i32(materialTexDescriptors[base + 3u].w);
  if (bumpIdx < 0 || triIndex >= arrayLength(&indices)) { return shadingNormal; }
  let tri = indices[triIndex];
  if (tri.x >= arrayLength(&meshUvs) || tri.y >= arrayLength(&meshUvs) || tri.z >= arrayLength(&meshUvs) ||
      tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
    return shadingNormal;
  }
  let bumpScale = materialTexDescriptors[base + 4u].z;
  // Build the same world-space tangent frame applyNormalMap uses (D9.3 shared helper).
  let frame = buildShadingTangentFrame(triIndex, shadingNormal, instanceIndex);
  if (!frame.valid) { return shadingNormal; }
  let tangent = frame.tangent;
  let bitangent = frame.bitangent;
  // Central finite difference of the height (R channel) in UV space. A small UV
  // step; the height-gradient slopes the normal by -scale·(dh/du, dh/dv).
  let bumpUvFitScale = materialTexDescriptors[base + 10u].xy;
  let hC = sampleMaterialLayerLinear(bumpIdx, base, triIndex, baryVW, bumpUvFitScale).r;
  // Approximate dh/du, dh/dv by sampling a small step along the interpolated UV
  // via barycentric perturbation toward each triangle edge.
  let du = 1.0 / 512.0;
  let baryU = vec2f(baryVW.x + du, baryVW.y);
  let baryV = vec2f(baryVW.x, baryVW.y + du);
  let hU = sampleMaterialLayerLinear(bumpIdx, base, triIndex, baryU, bumpUvFitScale).r;
  let hV = sampleMaterialLayerLinear(bumpIdx, base, triIndex, baryV, bumpUvFitScale).r;
  let dhdu = (hU - hC) / du;
  let dhdv = (hV - hC) / du;
  let perturbed = shadingNormal - bumpScale * (dhdu * tangent + dhdv * bitangent);
  let plen = length(perturbed);
  return select(shadingNormal, perturbed / plen, plen > 1e-6);
}

// Standalone alpha map (LINEAR coverage data) — descriptor vec4[6].x.
// Multiplies the baseColor texture alpha and material opacity in alphaMode
// mask/blend. Returns 1 when absent, so legacy alpha behavior is unchanged.
fn sampleAlphaTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> f32 {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 11u >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  let alphaIdx = i32(materialTexDescriptors[base + 6u].x);
  if (alphaIdx < 0) { return 1.0; }
  return clamp(sampleMaterialLayerLinear(alphaIdx, base, triIndex, baryVW, materialTexDescriptors[base + 11u].xy).r, 0.0, 1.0);
}

// Transmission map (LINEAR scalar data) — descriptor vec4[6].y.
// Multiplies MaterialSpec.transmission using glTF KHR_materials_transmission's
// R channel. Returns 1 when absent, so scalar-only transmission is unchanged.
fn sampleTransmissionTexture(matId: u32, triIndex: u32, baryVW: vec2f) -> f32 {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 11u >= arrayLength(&materialTexDescriptors)) { return 1.0; }
  let transmissionIdx = i32(materialTexDescriptors[base + 6u].y);
  if (transmissionIdx < 0) { return 1.0; }
  return clamp(sampleMaterialLayerLinear(transmissionIdx, base, triIndex, baryVW, materialTexDescriptors[base + 11u].zw).r, 0.0, 1.0);
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
fn alphaTestPassThrough(matId: u32, triIndex: u32, baryVW: vec2f, rng: ptr<function, u32>) -> bool {
  let base = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (base + 3u >= arrayLength(&materialTexDescriptors)) { return false; }
  let alphaMode = u32(materialTexDescriptors[base + 1u].x);
  if (alphaMode == 0u) { return false; } // opaque — byte-identical
  let alphaCutoff = materialTexDescriptors[base + 1u].y;
  let opacity = materialTexDescriptors[base + 1u].z;
  let alpha = sampleBaseColorTexture(matId, triIndex, baryVW).a *
    sampleAlphaTexture(matId, triIndex, baryVW) *
    opacity;
  if (alphaMode == 1u) { return alpha < alphaCutoff; }   // mask
  return rand_f32(rng) >= alpha;                          // blend (stochastic)
}
`;

const PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_WGSL =
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP0_WGSL +
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP1_WGSL +
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP2_WGSL +
  PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;

export const PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL = /* wgsl */ `
const LEAFNODE_FLAG = 0xffff0000u;
// MUST stay in lockstep with TS \`MATERIAL_VEC4_STRIDE\` in scene/materialPacking.ts.
// WS4 bumped 22 → 23: vec4 #22 carries volumetric σ_a.rgb + hasSigmaA flag.
// H52 bumped 23 → 26: vec4s #23–#25 carry clearcoat / sheen / iridescence lobes.
// A3 bumped 26 → 27: vec4 #26 carries the baseColor Jakob-Hanika sigmoid coeffs.
const MATERIAL_VEC4_STRIDE = 27u;
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
// Returns 0 for smooth surfaces (E_avg≈1) → zero loss → byte-identical low-r.
fn ggxMultiscatterLobe(f0: vec3f, roughness: f32, nDotV: f32, nDotL: f32) -> vec3f {
  let eAvg = ggxAverageAlbedo(roughness);
  let oneMinusEavg = 1.0 - eAvg;
  if (oneMinusEavg < 1e-4) { return vec3f(0.0); } // smooth → no missing energy.
  let eo = ggxDirectionalAlbedo(nDotV, roughness);
  let ei = ggxDirectionalAlbedo(nDotL, roughness);
  // Averaged Fresnel for the multiscatter tint (Kulla-Conty): F_avg ≈ F0 + (1−F0)/21.
  let fAvg = f0 + (vec3f(1.0) - f0) * (1.0 / 21.0);
  // Multiscatter energy: the geometric series of bounces sums to F_avg·E_avg /
  // (1 − F_avg·(1−E_avg)) for the colour, times the Kulla-Conty directional shape.
  let fMs = fAvg * fAvg * eAvg / max(vec3f(1.0) - fAvg * oneMinusEavg, vec3f(1e-4));
  let shape = (1.0 - eo) * (1.0 - ei) / max(PI * oneMinusEavg, 1e-6);
  return fMs * shape;
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
}

// A3 — evaluate the Jakob & Hanika 2019 sigmoid-polynomial spectral reflectance
// at wavelength λ (nm). S(λ) = ½ + x/(2·√(1+x²)),  x = c0 + c1·λ + c2·λ².
// Bounded in (0,1) and numerically safe for large |x| (no exp overflow). Mirrors
// the TS evaluateSpectrum in shared-samplers/jakobHanika.ts.
fn evalJakobHanikaSpectrum(coeffs: vec3f, lambdaNm: f32) -> f32 {
  let x = coeffs.x + coeffs.y * lambdaNm + coeffs.z * lambdaNm * lambdaNm;
  return 0.5 + x / (2.0 * sqrt(1.0 + x * x));
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
  let m23Index = m0Index + 23u; // H52 clearcoat/sheen vec4
  let m24Index = m0Index + 24u; // H52 sheenColor + iridescence vec4
  let m25Index = m0Index + 25u; // H52 iridescence params vec4
  let m26Index = m0Index + 26u; // A3 baseColor Jakob-Hanika coeffs + material flags vec4
  let m0 = select(vec4f(0.8, 0.8, 0.8, 0.6), materials[m0Index], m0Index < arrayLength(&materials));
  let m1 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m1Index], m1Index < arrayLength(&materials));
  let m2 = select(vec4f(0.0, 1.5, 0.0, 0.0), materials[m2Index], m2Index < arrayLength(&materials));
  let m3 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m3Index], m3Index < arrayLength(&materials));
  let m4 = select(vec4f(1.0, 1.0, 1.0, -1.0), materials[m4Index], m4Index < arrayLength(&materials));
  let m5 = select(vec4f(1.0, 1.0, 1.0, -1.0), materials[m5Index], m5Index < arrayLength(&materials));
  let m6 = select(vec4f(0.0, 0.0, 1.0, 0.0), materials[m6Index], m6Index < arrayLength(&materials));
  let m19 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m19Index], m19Index < arrayLength(&materials));
  let m22 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m22Index], m22Index < arrayLength(&materials));
  // H52: defaults for all three lobes are 0 (zero-default = numerically identical to pre-H52 path).
  let m23 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m23Index], m23Index < arrayLength(&materials));
  let m24 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m24Index], m24Index < arrayLength(&materials));
  let m25 = select(vec4f(1.3, 100.0, 400.0, 0.0), materials[m25Index], m25Index < arrayLength(&materials));
  // A3 default: a flat grey (c0=0,c1=0,c2=0 ⇒ x=0 ⇒ S≡½) with flag 0 so an
  // unpacked material is treated as having no spectral curve (RGB fallback).
  let m26 = select(vec4f(0.0, 0.0, 0.0, 0.0), materials[m26Index], m26Index < arrayLength(&materials));
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
  mat.hasSpectralReflectance = (u32(max(m26.w, 0.0)) & 1u) != 0u;
  mat.isUnlit = (u32(max(m26.w, 0.0)) & 2u) != 0u;
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
