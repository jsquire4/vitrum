import {
  MATERIAL_OPTICS_WGSL,
  buildMaterialAtlasOffsetConstsWGSL,
} from '@vitrum/shared-bvh';
import type { WgslModule } from '../pipeline/wgslComposer.js';

// 62-texel material-atlas offset ABI — single-sourced in @vitrum/shared-bvh
// (T4-2, 2026-07-20). The subset + order below reproduces the historical
// hand-written const block byte-for-byte (pinned by the shade composed-WGSL
// golden). The decode functions below are NOT single-sourced — they diverge
// semantically from the DDGI/RC copies (bindings, meta-coord scheme, atlas
// filter policy); see materialAtlasOffsets.wgsl.ts for the divergence note.
const MATERIAL_ATLAS_OFFSET_CONSTS = buildMaterialAtlasOffsetConstsWGSL({
  prefix: '',
  include: [
    'META_TEXELS_PER_TRI',
    'SLOT_BASE_COLOR',
    'SLOT_ROUGHNESS',
    'SLOT_METALLIC',
    'SLOT_AO',
    'SLOT_ALPHA',
    'ALPHA_COVERAGE_TEXEL_OFFSET',
    'EMISSIVE_TEXEL_OFFSET',
    'TRANSMISSION_TEXEL_OFFSET',
    'NORMAL_TEXEL_OFFSET',
    'NORMAL_SCALE_TEXEL_OFFSET',
    'LIGHT_TEXEL_OFFSET',
    'LIGHT_INTENSITY_TEXEL_OFFSET',
    'SPECULAR_TEXEL_OFFSET',
    'CLEARCOAT_TEXEL_OFFSET',
    'SHEEN_COLOR_TEXEL_OFFSET',
    'SPECULAR_COLOR_TEXEL_OFFSET',
    'SPECULAR_INTENSITY_TEXEL_OFFSET',
    'CLEARCOAT_FACTOR_TEXEL_OFFSET',
    'CLEARCOAT_ROUGHNESS_TEXEL_OFFSET',
    'SHEEN_COLOR_MAP_TEXEL_OFFSET',
    'SHEEN_ROUGHNESS_TEXEL_OFFSET',
    'CLEARCOAT_NORMAL_TEXEL_OFFSET',
    'CLEARCOAT_NORMAL_SCALE_TEXEL_OFFSET',
    'ANISOTROPY_TEXEL_OFFSET',
    'ANISOTROPY_SCALAR_TEXEL_OFFSET',
    'IRIDESCENCE_TEXEL_OFFSET',
    'IRIDESCENCE_THICKNESS_TEXEL_OFFSET',
    'IRIDESCENCE_SCALAR_TEXEL_OFFSET',
    'THICKNESS_TEXEL_OFFSET',
    'BUMP_TEXEL_OFFSET',
    'BUMP_SCALE_TEXEL_OFFSET',
    'ENV_INTENSITY_TEXEL_OFFSET',
    'FRONT_LAYER_TEXEL_OFFSET',
    'BACK_LAYER_TEXEL_OFFSET',
    'VOLUME_SCATTERING_TEXEL_OFFSET',
    'FRONT_LAYER_NORMAL_TEXEL_OFFSET',
    'FRONT_LAYER_NORMAL_SCALE_TEXEL_OFFSET',
    'BACK_LAYER_NORMAL_TEXEL_OFFSET',
    'BACK_LAYER_NORMAL_SCALE_TEXEL_OFFSET',
    'SIDE_FLAGS_TEXEL_OFFSET',
  ],
});

/**
 * The textured first-hit alpha-coverage walk-wrapper (D8-3, T4-2 2026-07-20).
 * Secondary transport uses stochastic blend coverage with an explicit
 * frame/sample seed, while camera-primary passes use the opaque-only predicate
 * and leave fractional layers to TransparentOitPass. Both paths share the same
 * bounded 32-layer walk. Exhausting the budget returns the next surface as a
 * conservative blocker instead of leaking through it.
 *
 * References consumer bindings (`materialMask`, BVH storage) → raw-string
 * template interpolated into the consumer body, NOT a WgslModule (composeWgsl
 * ordering constraint).
 */
function makeTexturedFirstHitAlphaMaskWalkerWGSL(fnName: string, discardPredicate: string): string {
  return /* wgsl */ `fn ${fnName}(
  bvhMode: u32,
  tlasNodeCount: u32,
  ray: Ray,
  triEps: f32,
  materialMask: texture_2d<u32>,
  materialMaskWidth: u32,
  sampleSeed: u32,
) -> IntersectionResult {
  var walkRay = ray;
  var traveled = 0.0;
  let step = max(1e-4, triEps * 4.0);
  for (var i = 0u; i < 32u; i = i + 1u) {
    var hit = traceSceneFirstHit(
      bvhMode, tlasNodeCount,
      walkRay, triEps,
    );
    if (!hit.didHit) {
      return hit;
    }
    let word = textureLoad(
      materialMask,
      vec2i(i32(hit.indices.w % materialMaskWidth), i32(hit.indices.w / materialMaskWidth)),
      0,
    ).r;
    if (!${discardPredicate}(hit, word, ray, i, sampleSeed)) {
      hit.dist = hit.dist + traveled;
      return hit;
    }
    traveled = traveled + hit.dist + step;
    walkRay.origin = ray.origin + ray.direction * traveled;
  }
  var exhausted = traceSceneFirstHit(
    bvhMode, tlasNodeCount,
    walkRay, triEps,
  );
  // Conservative overflow: after the bounded transparent-layer budget, any
  // further surface is returned as a blocker instead of leaking radiance.
  if (exhausted.didHit) {
    exhausted.dist = exhausted.dist + traveled;
  }
  return exhausted;
}`;
}

export const MATERIAL_ATLAS_WGSL = /* wgsl */ `
// Material maps enter through either CPU pixel payloads or nominal
// WalkaroundWebGpuTextureSource descriptors. Both become RGBA32F array layers
// with full mip chains plus per-triangle metadata. Sampler policy is implemented
// manually with textureLoad so compute and fragment passes agree exactly.
@group(1) @binding(20) var materialTextureAtlas: texture_2d_array<f32>;
@group(1) @binding(21) var baseColorMapMeta: texture_2d<f32>;
@group(1) @binding(22) var bvh_tangent: texture_2d<f32>;
@group(1) @binding(23) var bvh_vertex_color: texture_2d<f32>;

const BASE_COLOR_MAP_META_TEX_WIDTH: u32 = 4096u;
${MATERIAL_ATLAS_OFFSET_CONSTS}

fn baseColorMapMetaCoord(texel: u32) -> vec2i {
  return vec2i(i32(texel % BASE_COLOR_MAP_META_TEX_WIDTH), i32(texel / BASE_COLOR_MAP_META_TEX_WIDTH));
}

fn materialOpticalLoad(triIndex: u32, metaOffset: u32) -> vec4f {
  let texel = triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset;
  return textureLoad(baseColorMapMeta, baseColorMapMetaCoord(texel), 0);
}

${MATERIAL_OPTICS_WGSL}

fn wrapMaterialUv1(v: f32, mode: u32) -> f32 {
  if (mode == 1u) {
    return clamp(v, 0.0, 1.0);
  }
  if (mode == 2u) {
    return 1.0 - abs(fract(v * 0.5) * 2.0 - 1.0);
  }
  return fract(v);
}

fn wrapMaterialUv(uv: vec2f, wrapPacked: u32) -> vec2f {
  let wrapS = wrapPacked & 0x3u;
  let wrapT = (wrapPacked >> 2u) & 0x3u;
  return vec2f(wrapMaterialUv1(uv.x, wrapS), wrapMaterialUv1(uv.y, wrapT));
}

fn wrapMaterialTexelIndex(index: i32, size: i32, mode: u32) -> i32 {
  if (size <= 1) {
    return 0;
  }
  if (mode == 1u) {
    return clamp(index, 0, size - 1);
  }
  if (mode == 2u) {
    let period = size * 2;
    var x = index % period;
    if (x < 0) {
      x = x + period;
    }
    if (x >= size) {
      return period - x - 1;
    }
    return x;
  }
  var x = index % size;
  if (x < 0) {
    x = x + size;
  }
  return x;
}

fn materialAtlasFilterMode(samplerPacked: u32, lod: f32) -> u32 {
  let magFilter = (samplerPacked >> 10u) & 0x1u;
  let minFilter = (samplerPacked >> 11u) & 0x1u;
  return select(magFilter, minFilter, lod > 0.0);
}

fn sampleMaterialAtlasNearestLevel(wrapped: vec2f, layer: i32, level: u32) -> vec4f {
  let dims = textureDimensions(materialTextureAtlas, level);
  let texel = vec2i(
    i32(min(u32(floor(wrapped.x * f32(dims.x))), dims.x - 1u)),
    i32(min(u32(floor(wrapped.y * f32(dims.y))), dims.y - 1u)),
  );
  return textureLoad(materialTextureAtlas, texel, layer, i32(level));
}

fn sampleMaterialAtlasLinearLevel(
  wrapped: vec2f,
  layer: i32,
  samplerPacked: u32,
  level: u32,
) -> vec4f {
  let dims = textureDimensions(materialTextureAtlas, level);
  let size = vec2i(i32(dims.x), i32(dims.y));
  let coord = wrapped * vec2f(f32(dims.x), f32(dims.y)) - vec2f(0.5);
  let base = vec2i(i32(floor(coord.x)), i32(floor(coord.y)));
  let f = coord - vec2f(floor(coord.x), floor(coord.y));
  let wrapS = samplerPacked & 0x3u;
  let wrapT = (samplerPacked >> 2u) & 0x3u;
  let x0 = wrapMaterialTexelIndex(base.x, size.x, wrapS);
  let x1 = wrapMaterialTexelIndex(base.x + 1, size.x, wrapS);
  let y0 = wrapMaterialTexelIndex(base.y, size.y, wrapT);
  let y1 = wrapMaterialTexelIndex(base.y + 1, size.y, wrapT);
  let c00 = textureLoad(materialTextureAtlas, vec2i(x0, y0), layer, i32(level));
  let c10 = textureLoad(materialTextureAtlas, vec2i(x1, y0), layer, i32(level));
  let c01 = textureLoad(materialTextureAtlas, vec2i(x0, y1), layer, i32(level));
  let c11 = textureLoad(materialTextureAtlas, vec2i(x1, y1), layer, i32(level));
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

fn sampleMaterialAtlasLevel(
  wrapped: vec2f,
  layer: i32,
  samplerPacked: u32,
  level: u32,
  lod: f32,
) -> vec4f {
  if (materialAtlasFilterMode(samplerPacked, lod) == 0u) {
    return sampleMaterialAtlasNearestLevel(wrapped, layer, level);
  }
  return sampleMaterialAtlasLinearLevel(wrapped, layer, samplerPacked, level);
}

fn sampleMaterialAtlasAtLod(
  wrapped: vec2f,
  layer: i32,
  samplerPacked: u32,
  lod: f32,
) -> vec4f {
  let mipFilter = (samplerPacked >> 8u) & 0x3u;
  let lastLevel = max(textureNumLevels(materialTextureAtlas), 1u) - 1u;
  if (mipFilter == 0u || lastLevel == 0u) {
    return sampleMaterialAtlasLevel(wrapped, layer, samplerPacked, 0u, lod);
  }
  let clampedLod = clamp(lod, 0.0, f32(lastLevel));
  if (mipFilter == 1u) {
    let level = min(u32(floor(clampedLod + 0.5)), lastLevel);
    return sampleMaterialAtlasLevel(wrapped, layer, samplerPacked, level, lod);
  }
  let level0 = min(u32(floor(clampedLod)), lastLevel);
  let level1 = min(level0 + 1u, lastLevel);
  let c0 = sampleMaterialAtlasLevel(wrapped, layer, samplerPacked, level0, lod);
  let c1 = sampleMaterialAtlasLevel(wrapped, layer, samplerPacked, level1, lod);
  return mix(c0, c1, clampedLod - floor(clampedLod));
}

fn interpolateUv1FromNormalW(hit: IntersectionResult, n0: vec4f, n1: vec4f, n2: vec4f) -> vec2f {
  let uvA = unpack2x16float(bitcast<u32>(n0.w));
  let uvB = unpack2x16float(bitcast<u32>(n1.w));
  let uvC = unpack2x16float(bitcast<u32>(n2.w));
  return hit.barycoord.x * uvA + hit.barycoord.y * uvB + hit.barycoord.z * uvC;
}

fn materialAtlasUv1ForHit(hit: IntersectionResult) -> vec2f {
  let n0 = sceneLoadBvhNormal(hit.indices.x);
  let n1 = sceneLoadBvhNormal(hit.indices.y);
  let n2 = sceneLoadBvhNormal(hit.indices.z);
  return interpolateUv1FromNormalW(hit, n0, n1, n2);
}

fn materialAtlasPackedUvFromVec4(v: vec4f) -> vec2f {
  return unpack2x16float(bitcast<u32>(v.w));
}

fn materialAtlasDefaultLod(meta1: vec4f) -> f32 {
  let atlasSize = vec2f(textureDimensions(materialTextureAtlas));
  let screenSize = vec2f(max(ubo.screenSize, vec2u(1u)));
  let footprint = abs(meta1.xy) * atlasSize / screenSize;
  return log2(max(max(footprint.x, footprint.y), 1e-8));
}

fn materialAtlasTransformPointForHit(hit: IntersectionResult, p: vec3f) -> vec3f {
  let base = hit.instanceIndex * 4u;
  if (ubo.bvhMode != 1u || base + 3u >= tlasLocalToWorldColumnCount()) {
    return p;
  }
  let c0 = tlasLoadLocalToWorldColumn(base);
  let c1 = tlasLoadLocalToWorldColumn(base + 1u);
  let c2 = tlasLoadLocalToWorldColumn(base + 2u);
  let c3 = tlasLoadLocalToWorldColumn(base + 3u);
  return c0.xyz * p.x + c1.xyz * p.y + c2.xyz * p.z + c3.xyz;
}

fn materialAtlasProjectToPixels(p: vec3f) -> vec3f {
  let clip = ubo.projMatrix * ubo.viewMatrix * vec4f(p, 1.0);
  if (clip.w <= 1e-6) {
    return vec3f(0.0);
  }
  let ndc = clip.xy / clip.w;
  return vec3f(
    (ndc * 0.5 + vec2f(0.5)) * vec2f(max(ubo.screenSize, vec2u(1u))),
    1.0,
  );
}

fn materialAtlasTransformUvForLod(uv: vec2f, meta1: vec4f) -> vec2f {
  let scaled = uv * meta1.xy;
  return vec2f(
    scaled.x * meta1.z - scaled.y * meta1.w,
    scaled.x * meta1.w + scaled.y * meta1.z,
  );
}

// Bounded realtime footprint model: project the hit triangle into the active
// camera and derive a geometric UV footprint. This is intentionally used for
// primary and secondary hits; it is stable and host-independent, but is not a
// propagated ray-differential model for indirect/specular paths.
fn materialAtlasLodForHit(hit: IntersectionResult, metaOffset: u32) -> f32 {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset;
  let meta0 = textureLoad(baseColorMapMeta, baseColorMapMetaCoord(metaTexel), 0);
  let meta1 = textureLoad(baseColorMapMeta, baseColorMapMetaCoord(metaTexel + 1u), 0);
  let flags = u32(max(meta0.y, 0.0) + 0.5);
  let texCoord = (flags >> 4u) & 0xFu;

  let p0Packed = bvhLoadPosition(hit.indices.x);
  let p1Packed = bvhLoadPosition(hit.indices.y);
  let p2Packed = bvhLoadPosition(hit.indices.z);
  let n0Packed = sceneLoadBvhNormal(hit.indices.x);
  let n1Packed = sceneLoadBvhNormal(hit.indices.y);
  let n2Packed = sceneLoadBvhNormal(hit.indices.z);
  let uv0 = materialResolveUv(
    triIndex,
    texCoord,
    materialAtlasPackedUvFromVec4(p0Packed),
    materialAtlasPackedUvFromVec4(n0Packed),
  );
  let uv1 = materialResolveUv(
    triIndex,
    texCoord,
    materialAtlasPackedUvFromVec4(p1Packed),
    materialAtlasPackedUvFromVec4(n1Packed),
  );
  let uv2 = materialResolveUv(
    triIndex,
    texCoord,
    materialAtlasPackedUvFromVec4(p2Packed),
    materialAtlasPackedUvFromVec4(n2Packed),
  );
  let screen0 = materialAtlasProjectToPixels(materialAtlasTransformPointForHit(hit, p0Packed.xyz));
  let screen1 = materialAtlasProjectToPixels(materialAtlasTransformPointForHit(hit, p1Packed.xyz));
  let screen2 = materialAtlasProjectToPixels(materialAtlasTransformPointForHit(hit, p2Packed.xyz));
  if (screen0.z == 0.0 || screen1.z == 0.0 || screen2.z == 0.0) {
    return materialAtlasDefaultLod(meta1);
  }

  let screenEdge1 = screen1.xy - screen0.xy;
  let screenEdge2 = screen2.xy - screen0.xy;
  let det = screenEdge1.x * screenEdge2.y - screenEdge1.y * screenEdge2.x;
  if (abs(det) <= 1e-8) {
    return materialAtlasDefaultLod(meta1);
  }
  let uvEdge1 = materialAtlasTransformUvForLod(uv1, meta1) - materialAtlasTransformUvForLod(uv0, meta1);
  let uvEdge2 = materialAtlasTransformUvForLod(uv2, meta1) - materialAtlasTransformUvForLod(uv0, meta1);
  let duvDx = (uvEdge1 * screenEdge2.y - uvEdge2 * screenEdge1.y) / det;
  let duvDy = (-uvEdge1 * screenEdge2.x + uvEdge2 * screenEdge1.x) / det;
  let atlasSize = vec2f(textureDimensions(materialTextureAtlas));
  let rho = max(length(duvDx * atlasSize), length(duvDy * atlasSize));
  return log2(max(rho, 1e-8));
}

fn sampleMaterialAtlasRawAtOffsetDeltaLod(
  triIndex: u32,
  metaOffset: u32,
  uv0: vec2f,
  uv1: vec2f,
  transformedDelta: vec2f,
  lod: f32,
  explicitLod: bool,
) -> vec4f {
  let metaTexel = triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset;
  let meta0 = textureLoad(baseColorMapMeta, baseColorMapMetaCoord(metaTexel), 0);
  let layer = i32(meta0.x);
  if (layer < 0) {
    return vec4f(-1.0, -1.0, -1.0, -1.0);
  }
  let wrapPacked = u32(max(meta0.y, 0.0) + 0.5);
  let texCoord = (wrapPacked >> 4u) & 0xFu;
  let uv = materialResolveUv(triIndex, texCoord, uv0, uv1);
  let meta1 = textureLoad(baseColorMapMeta, baseColorMapMetaCoord(metaTexel + 1u), 0);
  let scaled = uv * meta1.xy;
  let transformed = vec2f(
    scaled.x * meta1.z - scaled.y * meta1.w,
    scaled.x * meta1.w + scaled.y * meta1.z,
  ) + meta0.zw + transformedDelta;
  let wrapped = wrapMaterialUv(transformed, wrapPacked);
  let resolvedLod = select(materialAtlasDefaultLod(meta1), lod, explicitLod);
  return sampleMaterialAtlasAtLod(wrapped, layer, wrapPacked, resolvedLod);
}

fn sampleMaterialAtlasRawAtOffsetDelta(
  triIndex: u32,
  metaOffset: u32,
  uv0: vec2f,
  uv1: vec2f,
  transformedDelta: vec2f,
) -> vec4f {
  return sampleMaterialAtlasRawAtOffsetDeltaLod(
    triIndex,
    metaOffset,
    uv0,
    uv1,
    transformedDelta,
    0.0,
    false,
  );
}

fn sampleMaterialAtlasRawAtOffset(triIndex: u32, metaOffset: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  return sampleMaterialAtlasRawAtOffsetDelta(triIndex, metaOffset, uv0, uv1, vec2f(0.0));
}

fn sampleMaterialAtlasRaw(triIndex: u32, slot: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  return sampleMaterialAtlasRawAtOffset(triIndex, slot * 2u, uv0, uv1);
}

fn sampleMaterialAtlasRawAtOffsetDeltaForHit(
  hit: IntersectionResult,
  metaOffset: u32,
  transformedDelta: vec2f,
) -> vec4f {
  return sampleMaterialAtlasRawAtOffsetDeltaLod(
    hit.indices.w,
    metaOffset,
    hit.uv,
    materialAtlasUv1ForHit(hit),
    transformedDelta,
    materialAtlasLodForHit(hit, metaOffset),
    true,
  );
}

fn sampleMaterialAtlasRawAtOffsetForHit(hit: IntersectionResult, metaOffset: u32) -> vec4f {
  return sampleMaterialAtlasRawAtOffsetDeltaForHit(hit, metaOffset, vec2f(0.0));
}

fn sampleMaterialAtlasRawForHit(hit: IntersectionResult, slot: u32) -> vec4f {
  return sampleMaterialAtlasRawAtOffsetForHit(hit, slot * 2u);
}

fn materialMapChannel(v: vec4f, channel: u32) -> f32 {
  if (channel == 1u) { return v.g; }
  if (channel == 2u) { return v.b; }
  if (channel == 3u) { return v.a; }
  return v.r;
}

fn sampleBaseColorMap(hit: IntersectionResult, scalarBaseColor: vec3f) -> vec3f {
  let texelColor = sampleMaterialAtlasRawForHit(hit, MATERIAL_MAP_SLOT_BASE_COLOR);
  if (texelColor.x < 0.0) {
    return scalarBaseColor;
  }
  return scalarBaseColor * texelColor.rgb;
}

fn sampleMaterialScalarMap(hit: IntersectionResult, slot: u32, channel: u32, fallback: f32) -> f32 {
  let texelColor = sampleMaterialAtlasRawForHit(hit, slot);
  if (texelColor.x < 0.0) {
    return fallback;
  }
  return clamp(fallback * materialMapChannel(texelColor, channel), 0.0, 1.0);
}

fn sampleAoMapFactor(hit: IntersectionResult, materialWord: u32) -> f32 {
  let rawOcclusion = sampleMaterialScalarMap(hit, MATERIAL_MAP_SLOT_AO, 0u, 1.0);
  let strength = decodeAoMapIntensity(materialWord);
  return mix(1.0, rawOcclusion, strength);
}

fn sampleEmissiveMap(triIndex: u32, uv0: vec2f, uv1: vec2f, scalarEmissive: vec3f) -> vec3f {
  let texelColor = sampleMaterialAtlasRawAtOffset(
    triIndex,
    MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET,
    uv0,
    uv1,
  );
  if (texelColor.x < 0.0) {
    return scalarEmissive;
  }
  return scalarEmissive * texelColor.rgb;
}

fn sampleTransmissionMapForHit(hit: IntersectionResult, scalarTransmission: f32) -> f32 {
  let texelColor = sampleMaterialAtlasRawAtOffsetForHit(
    hit,
    MATERIAL_MAP_TRANSMISSION_TEXEL_OFFSET,
  );
  if (texelColor.x < 0.0) {
    return scalarTransmission;
  }
  return clamp(scalarTransmission * texelColor.r, 0.0, 1.0);
}

fn sampleLightMap(hit: IntersectionResult) -> vec3f {
  let triIndex = hit.indices.w;
  let texelColor = sampleMaterialAtlasRawAtOffsetForHit(
    hit,
    MATERIAL_MAP_LIGHT_TEXEL_OFFSET,
  );
  if (texelColor.x < 0.0) {
    return vec3f(0.0);
  }
  let intensityMeta = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_LIGHT_INTENSITY_TEXEL_OFFSET),
    0,
  );
  return texelColor.rgb * max(intensityMeta.x, 0.0);
}

fn sampleEnvMapIntensity(triIndex: u32) -> f32 {
  let intensityMeta = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_ENV_INTENSITY_TEXEL_OFFSET),
    0,
  );
  return max(intensityMeta.x, 0.0);
}

fn sampleFaceLayerControls(triIndex: u32, isFrontFace: bool) -> vec4f {
  let offset = select(
    MATERIAL_MAP_BACK_LAYER_TEXEL_OFFSET,
    MATERIAL_MAP_FRONT_LAYER_TEXEL_OFFSET,
    isFrontFace,
  );
  return textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + offset),
    0,
  );
}

fn faceLayerTransmission(layer: vec4f) -> vec3f {
  return clamp(layer.rgb, vec3f(0.0), vec3f(1.0));
}

fn faceLayerRoughness(roughness: f32, layer: vec4f) -> f32 {
  return select(roughness, clamp(layer.a, 0.0, 1.0), layer.a >= 0.0);
}

// Exact unpolarised dielectric Fresnel transmission for three IOR lanes.
// etaIncident/etaTarget are absolute medium IORs and cosIncident is the
// positive cosine against the oriented interface normal. A TIR lane returns 0.
fn dielectricInterfaceTransmissionRgb(
  cosIncident: f32,
  etaIncident: vec3f,
  etaTarget: vec3f,
) -> vec3f {
  let ci = clamp(abs(cosIncident), 0.0, 1.0);
  let eta = max(etaIncident, vec3f(1e-6)) / max(etaTarget, vec3f(1e-6));
  let sin2Target = eta * eta * (1.0 - ci * ci);
  let ct = sqrt(max(vec3f(0.0), vec3f(1.0) - sin2Target));
  let rsNumerator = etaIncident * ci - etaTarget * ct;
  let rsDenominator = etaIncident * ci + etaTarget * ct;
  let rpNumerator = etaTarget * ci - etaIncident * ct;
  let rpDenominator = etaTarget * ci + etaIncident * ct;
  let rs = rsNumerator / max(abs(rsDenominator), vec3f(1e-6));
  let rp = rpNumerator / max(abs(rpDenominator), vec3f(1e-6));
  let transmission = max(vec3f(0.0), vec3f(1.0) - 0.5 * (rs * rs + rp * rp));
  return transmission * vec3f(
    select(0.0, 1.0, sin2Target.r < 1.0),
    select(0.0, 1.0, sin2Target.g < 1.0),
    select(0.0, 1.0, sin2Target.b < 1.0),
  );
}

fn sampleVolumeScatteringControls(triIndex: u32) -> vec4f {
  let scatter = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(
      triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI +
      MATERIAL_MAP_VOLUME_SCATTERING_TEXEL_OFFSET
    ),
    0,
  );
  return vec4f(max(scatter.rgb, vec3f(0.0)), clamp(scatter.a, -0.99, 0.99));
}

fn homogeneousBeerTransmittanceRgb(sigmaT: vec3f, distance: f32) -> vec3f {
  return exp(-max(sigmaT, vec3f(0.0)) * max(distance, 0.0));
}

// Normalized Henyey-Greenstein phase density in inverse steradians.
fn henyeyGreensteinPhase(cosTheta: f32, g: f32) -> f32 {
  let anisotropy = clamp(g, -0.99, 0.99);
  let denominator = 1.0 + anisotropy * anisotropy -
    2.0 * anisotropy * clamp(cosTheta, -1.0, 1.0);
  return (1.0 - anisotropy * anisotropy) /
    (4.0 * PI * denominator * sqrt(denominator));
}

fn applyHomogeneousVolumeSingleScatter(
  radiance: vec3f,
  albedo: vec3f,
  scatter: vec4f,
  pathLength: f32,
  normal: vec3f,
  wo: vec3f,
) -> vec3f {
  let sigmaS = max(scatter.rgb, vec3f(0.0));
  if (all(sigmaS <= vec3f(0.0)) || pathLength <= 0.0) { return radiance; }
  let n = safe_normalize(normal);
  let v = safe_normalize(wo);
  // This overload is for directionally aggregated lighting (DDGI irradiance,
  // accumulated direct/indirect buffers, and OIT's multi-source resolve).
  // Once incident directions have been integrated there is no scattering
  // angle left to evaluate.  Use the isotropic average instead of inventing
  // one from the surface normal; anisotropic HG is evaluated only by the
  // directional overload below.
  let phase = henyeyGreensteinPhase(0.0, 0.0);
  let source = dot(max(radiance, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722)) *
    max(albedo, vec3f(0.0)) * phase;
  let projectedCosine = abs(dot(n, v));
  if (projectedCosine <= 0.0) { return source; }
  let distance = pathLength / projectedCosine;
  let transmittance = homogeneousBeerTransmittanceRgb(sigmaS, distance);
  // sigmaT == sigmaS for this contract, so the closed-form source integral
  // sigmaS * (1-exp(-sigmaT*d)) / sigmaT reduces channel-wise to (1-T).
  return radiance * transmittance + source * (vec3f(1.0) - transmittance);
}

fn applyHomogeneousVolumeSingleScatterDirectional(
  radiance: vec3f,
  albedo: vec3f,
  scatter: vec4f,
  pathLength: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let sigmaS = max(scatter.rgb, vec3f(0.0));
  if (all(sigmaS <= vec3f(0.0)) || pathLength <= 0.0) { return radiance; }
  let n = safe_normalize(normal);
  let v = safe_normalize(wo);
  // wi points from the receiver toward the source, so the incident
  // propagation direction is -wi. HG depends on that propagation/outgoing
  // angle, not on the surface normal.
  let phase = henyeyGreensteinPhase(
    dot(safe_normalize(-wi), v),
    scatter.a,
  );
  let source = dot(max(radiance, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722)) *
    max(albedo, vec3f(0.0)) * phase;
  let projectedCosine = abs(dot(n, v));
  if (projectedCosine <= 0.0) { return source; }
  let distance = pathLength / projectedCosine;
  let transmittance = homogeneousBeerTransmittanceRgb(sigmaS, distance);
  return radiance * transmittance + source * (vec3f(1.0) - transmittance);
}

fn sampleSpecularControls(hit: IntersectionResult) -> vec4f {
  let triIndex = hit.indices.w;
  let spec = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_SPECULAR_TEXEL_OFFSET),
    0,
  );
  // RGB is absolute dielectric F0 packed from material IOR and the
  // nonnegative KHR specularColor factor. Values above one remain observable.
  var color = max(spec.rgb, vec3f(0.0));
  var intensity = clamp(spec.a, 0.0, 1.0);

  let colorMap = sampleMaterialAtlasRawAtOffsetForHit(hit, MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET);
  if (colorMap.x >= 0.0) {
    color = max(color * colorMap.rgb, vec3f(0.0));
  }

  let intensityMap = sampleMaterialAtlasRawAtOffsetForHit(hit, MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET);
  if (intensityMap.x >= 0.0) {
    intensity = clamp(intensity * intensityMap.a, 0.0, 1.0);
  }

  return vec4f(color, intensity);
}

fn sampleClearcoatControls(hit: IntersectionResult) -> vec2f {
  let triIndex = hit.indices.w;
  let cc = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET),
    0,
  );
  var factor = clamp(cc.x, 0.0, 1.0);
  var roughness = clamp(cc.y, 0.0, 1.0);

  let clearcoatMap = sampleMaterialAtlasRawAtOffsetForHit(hit, MATERIAL_MAP_CLEARCOAT_FACTOR_TEXEL_OFFSET);
  if (clearcoatMap.x >= 0.0) {
    factor = clamp(factor * clearcoatMap.r, 0.0, 1.0);
  }

  let roughnessMap = sampleMaterialAtlasRawAtOffsetForHit(hit, MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET);
  if (roughnessMap.x >= 0.0) {
    roughness = clamp(roughness * roughnessMap.g, 0.0, 1.0);
  }

  return vec2f(factor, roughness);
}

fn sampleSheenControls(hit: IntersectionResult) -> vec4f {
  let triIndex = hit.indices.w;
  let scalars = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET),
    0,
  );
  let color = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_SHEEN_COLOR_TEXEL_OFFSET),
    0,
  );
  var sheenColor = clamp(color.rgb, vec3f(0.0), vec3f(1.0));
  var sheen = clamp(scalars.z, 0.0, 1.0);

  let colorMap = sampleMaterialAtlasRawAtOffsetForHit(hit, MATERIAL_MAP_SHEEN_COLOR_MAP_TEXEL_OFFSET);
  if (colorMap.x >= 0.0) {
    sheenColor = clamp(sheenColor * colorMap.rgb, vec3f(0.0), vec3f(1.0));
  }

  return vec4f(sheenColor, sheen);
}

fn sampleSheenRoughness(hit: IntersectionResult) -> f32 {
  let triIndex = hit.indices.w;
  let scalars = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET),
    0,
  );
  var roughness = clamp(scalars.w, 0.0, 1.0);
  let roughnessMap = sampleMaterialAtlasRawAtOffsetForHit(hit, MATERIAL_MAP_SHEEN_ROUGHNESS_TEXEL_OFFSET);
  if (roughnessMap.x >= 0.0) {
    roughness = clamp(roughness * roughnessMap.a, 0.0, 1.0);
  }
  return roughness;
}

fn sampleAnisotropyControls(hit: IntersectionResult) -> vec2f {
  let triIndex = hit.indices.w;
  let scalars = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_ANISOTROPY_SCALAR_TEXEL_OFFSET),
    0,
  );
  var strength = clamp(scalars.x, 0.0, 1.0);
  var rotation = scalars.y;

  let anisoMap = sampleMaterialAtlasRawAtOffsetForHit(hit, MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET);
  if (anisoMap.x >= 0.0) {
    strength = clamp(strength * anisoMap.b, 0.0, 1.0);
    let direction = anisoMap.rg * 2.0 - vec2f(1.0);
    if (dot(direction, direction) > 0.0) {
      rotation += atan2(direction.y, direction.x);
    }
  }

  return vec2f(strength, rotation);
}

fn sampleIridescenceControls(hit: IntersectionResult) -> vec4f {
  let triIndex = hit.indices.w;
  let scalars = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_IRIDESCENCE_SCALAR_TEXEL_OFFSET),
    0,
  );
  var factor = clamp(scalars.x, 0.0, 1.0);
  let ior = max(1.0, scalars.y);
  var thicknessMin = max(0.0, scalars.z);
  var thicknessMax = max(0.0, scalars.w);

  let iridescenceMap = sampleMaterialAtlasRawAtOffsetForHit(hit, MATERIAL_MAP_IRIDESCENCE_TEXEL_OFFSET);
  if (iridescenceMap.x >= 0.0) {
    factor = clamp(factor * iridescenceMap.r, 0.0, 1.0);
  }

  let thicknessMap = sampleMaterialAtlasRawAtOffsetForHit(hit, MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET);
  if (thicknessMap.x >= 0.0) {
    let thickness = mix(thicknessMin, thicknessMax, clamp(thicknessMap.g, 0.0, 1.0));
    thicknessMin = thickness;
    thicknessMax = thickness;
    if (thickness <= 0.0) {
      factor = 0.0;
    }
  }

  return vec4f(factor, ior, thicknessMin, thicknessMax);
}

fn applyThicknessMapToBeerTint(triIndex: u32, uv0: vec2f, uv1: vec2f, beerAlbedo: vec3f) -> vec3f {
  let thicknessMap = sampleMaterialAtlasRawAtOffset(triIndex, MATERIAL_MAP_THICKNESS_TEXEL_OFFSET, uv0, uv1);
  if (thicknessMap.x < 0.0) {
    return beerAlbedo;
  }
  // KHR_materials_volume.thicknessTexture stores thickness in G. The host's
  // bvh_beer lane already holds attenuationColor^(thicknessFactor / distance),
  // so exponentiating it by the sampled G channel applies the map without
  // adding another per-triangle attenuation-distance buffer.
  let thicknessFactor = clamp(thicknessMap.g, 0.0, 1.0);
  if (thicknessFactor <= 0.0) { return vec3f(1.0); }
  return pow(max(beerAlbedo, vec3f(0.0)), vec3f(thicknessFactor));
}

fn fallbackBitangentForNormal(n: vec3f, t: vec3f) -> vec3f {
  let b = cross(n, t);
  let len2 = dot(b, b);
  if (len2 < 1e-8) {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.95);
    return normalize(cross(n, up));
  }
  return b * inverseSqrt(len2);
}

fn bvhTangentTexel(vertexIndex: u32) -> vec4f {
  let dims = textureDimensions(bvh_tangent);
  let width = u32(dims.x);
  let height = u32(dims.y);
  if (width == 0u || height == 0u) {
    return vec4f(0.0);
  }
  let y = vertexIndex / width;
  if (y >= height) {
    return vec4f(0.0);
  }
  return textureLoad(bvh_tangent, vec2i(i32(vertexIndex % width), i32(y)), 0);
}

fn bvhVertexColorTexel(vertexIndex: u32) -> vec4f {
  let dims = textureDimensions(bvh_vertex_color);
  let width = u32(dims.x);
  let height = u32(dims.y);
  if (width == 0u || height == 0u) {
    return vec4f(1.0);
  }
  let y = vertexIndex / width;
  if (y >= height) {
    return vec4f(1.0);
  }
  return textureLoad(bvh_vertex_color, vec2i(i32(vertexIndex % width), i32(y)), 0);
}

fn sampleVertexColorForHit(hit: IntersectionResult) -> vec4f {
  let ca = bvhVertexColorTexel(hit.indices.x);
  let cb = bvhVertexColorTexel(hit.indices.y);
  let cc = bvhVertexColorTexel(hit.indices.z);
  return clamp(
    hit.barycoord.x * ca +
    hit.barycoord.y * cb +
    hit.barycoord.z * cc,
    vec4f(0.0),
    vec4f(1.0)
  );
}

fn transformDirectionCols(l2w0: vec4f, l2w1: vec4f, l2w2: vec4f, v: vec3f) -> vec3f {
  return l2w0.xyz * v.x + l2w1.xyz * v.y + l2w2.xyz * v.z;
}

fn tangentHandednessForLocalToWorld(l2w0: vec4f, l2w1: vec4f, l2w2: vec4f) -> f32 {
  let det = dot(l2w0.xyz, cross(l2w1.xyz, l2w2.xyz));
  return select(-1.0, 1.0, det >= 0.0);
}

struct MaterialTangentFrame {
  tangent: vec3f,
  bitangent: vec3f,
};

fn preferAuthoredTangentFrameForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  fallbackTangent: vec3f,
  fallbackBitangent: vec3f,
) -> MaterialTangentFrame {
  var tangent = fallbackTangent;
  var bitangent = fallbackBitangent;

  let ta = bvhTangentTexel(hit.indices.x);
  let tb = bvhTangentTexel(hit.indices.y);
  let tc = bvhTangentTexel(hit.indices.z);
  var authoredTangent =
    hit.barycoord.x * ta.xyz +
    hit.barycoord.y * tb.xyz +
    hit.barycoord.z * tc.xyz;
  var authoredHandedness =
    hit.barycoord.x * ta.w +
    hit.barycoord.y * tb.w +
    hit.barycoord.z * tc.w;

  if (length(authoredTangent) > 1e-8 && abs(authoredHandedness) > 0.5) {
    let isTlas = ubo.bvhMode == 1u;
    let tBase = hit.instanceIndex * 4u;
    let tOk = isTlas && tBase + 2u < tlasLocalToWorldColumnCount();
    if (tOk) {
      authoredTangent = transformDirectionCols(
        tlasLoadLocalToWorldColumn(tBase),
        tlasLoadLocalToWorldColumn(tBase + 1u),
        tlasLoadLocalToWorldColumn(tBase + 2u),
        authoredTangent,
      );
      authoredHandedness = authoredHandedness * tangentHandednessForLocalToWorld(
        tlasLoadLocalToWorldColumn(tBase),
        tlasLoadLocalToWorldColumn(tBase + 1u),
        tlasLoadLocalToWorldColumn(tBase + 2u),
      );
    }

    authoredTangent = authoredTangent - frameNormal * dot(frameNormal, authoredTangent);
    let tLen2 = dot(authoredTangent, authoredTangent);
    if (tLen2 > 1e-8) {
      tangent = authoredTangent * inverseSqrt(tLen2);
      bitangent = cross(frameNormal, tangent) * select(-1.0, 1.0, authoredHandedness >= 0.0);
    }
  }

  return MaterialTangentFrame(tangent, bitangent);
}

fn materialTangentFrameForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  mapOffset: u32,
) -> MaterialTangentFrame {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + mapOffset;
  let meta0 = textureLoad(baseColorMapMeta, baseColorMapMetaCoord(metaTexel), 0);
  let flags = u32(max(meta0.y, 0.0) + 0.5);
  let texCoord = (flags >> 4u) & 0xFu;
  let p0 = bvhLoadPosition(hit.indices.x);
  let p1 = bvhLoadPosition(hit.indices.y);
  let p2 = bvhLoadPosition(hit.indices.z);
  let n0 = sceneLoadBvhNormal(hit.indices.x);
  let n1 = sceneLoadBvhNormal(hit.indices.y);
  let n2 = sceneLoadBvhNormal(hit.indices.z);
  let uv0a = materialAtlasPackedUvFromVec4(p0);
  let uv0b = materialAtlasPackedUvFromVec4(p1);
  let uv0c = materialAtlasPackedUvFromVec4(p2);
  let uv1a = materialAtlasPackedUvFromVec4(n0);
  let uv1b = materialAtlasPackedUvFromVec4(n1);
  let uv1c = materialAtlasPackedUvFromVec4(n2);
  let ta = materialResolveUv(triIndex, texCoord, uv0a, uv1a);
  let tb = materialResolveUv(triIndex, texCoord, uv0b, uv1b);
  let tc = materialResolveUv(triIndex, texCoord, uv0c, uv1c);

  let dp1 = p1.xyz - p0.xyz;
  let dp2 = p2.xyz - p0.xyz;
  let duv1 = tb - ta;
  let duv2 = tc - ta;
  let det = duv1.x * duv2.y - duv1.y * duv2.x;
  var tangent = dp1;
  var bitangent = dp2;
  if (abs(det) > 1e-8) {
    let invDet = 1.0 / det;
    tangent = (dp1 * duv2.y - dp2 * duv1.y) * invDet;
    bitangent = (dp2 * duv1.x - dp1 * duv2.x) * invDet;
  }
  let isTlas = ubo.bvhMode == 1u;
  let tBase = hit.instanceIndex * 4u;
  let tOk = isTlas && tBase + 2u < tlasLocalToWorldColumnCount();
  if (tOk) {
    tangent = transformDirectionCols(
      tlasLoadLocalToWorldColumn(tBase),
      tlasLoadLocalToWorldColumn(tBase + 1u),
      tlasLoadLocalToWorldColumn(tBase + 2u),
      tangent,
    );
    bitangent = transformDirectionCols(
      tlasLoadLocalToWorldColumn(tBase),
      tlasLoadLocalToWorldColumn(tBase + 1u),
      tlasLoadLocalToWorldColumn(tBase + 2u),
      bitangent,
    );
  }

  tangent = tangent - frameNormal * dot(frameNormal, tangent);
  let tLen2 = dot(tangent, tangent);
  if (tLen2 < 1e-8) {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(frameNormal.y) > 0.95);
    tangent = normalize(cross(up, frameNormal));
  } else {
    tangent = tangent * inverseSqrt(tLen2);
  }
  bitangent = bitangent - frameNormal * dot(frameNormal, bitangent) - tangent * dot(tangent, bitangent);
  let bLen2 = dot(bitangent, bitangent);
  if (bLen2 < 1e-8) {
    bitangent = fallbackBitangentForNormal(frameNormal, tangent);
  } else {
    bitangent = bitangent * inverseSqrt(bLen2);
  }

  // glTF's authored TANGENT attribute is defined for TEXCOORD_0. A texture
  // selecting another UV lane needs the derivative frame derived above from
  // that exact lane; replacing it with the UV0 tangent would orient the map
  // against the wrong parameterization.
  if (texCoord == 0u) {
    return preferAuthoredTangentFrameForHit(hit, frameNormal, tangent, bitangent);
  }
  return MaterialTangentFrame(tangent, bitangent);
}

fn applyNormalMapAtOffsetForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  fallbackNormal: vec3f,
  normalMapOffset: u32,
  normalScaleOffset: u32,
) -> vec3f {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + normalMapOffset;
  let meta0 = textureLoad(baseColorMapMeta, baseColorMapMetaCoord(metaTexel), 0);
  if (i32(meta0.x) < 0) {
    return fallbackNormal;
  }

  let texelColor = sampleMaterialAtlasRawAtOffsetForHit(
    hit,
    normalMapOffset,
  );
  if (texelColor.x < 0.0) {
    return fallbackNormal;
  }

  let scaleMeta = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + normalScaleOffset),
    0,
  );
  let normalScale = max(scaleMeta.x, 0.0);
  let tangentSample = normalize(vec3f(
    (texelColor.r * 2.0 - 1.0) * normalScale,
    (texelColor.g * 2.0 - 1.0) * normalScale,
    texelColor.b * 2.0 - 1.0,
  ));

  let frame = materialTangentFrameForHit(hit, frameNormal, normalMapOffset);
  let perturbed = normalize(frame.tangent * tangentSample.x + frame.bitangent * tangentSample.y + frameNormal * tangentSample.z);
  return select(-perturbed, perturbed, dot(perturbed, frameNormal) >= 0.0);
}

fn applyFaceLayerNormalMapForHit(hit: IntersectionResult, frameNormal: vec3f, fallbackNormal: vec3f) -> vec3f {
  let isFrontFace = hit.side >= 0.0;
  let normalMapOffset = select(
    MATERIAL_MAP_BACK_LAYER_NORMAL_TEXEL_OFFSET,
    MATERIAL_MAP_FRONT_LAYER_NORMAL_TEXEL_OFFSET,
    isFrontFace,
  );
  let normalScaleOffset = select(
    MATERIAL_MAP_BACK_LAYER_NORMAL_SCALE_TEXEL_OFFSET,
    MATERIAL_MAP_FRONT_LAYER_NORMAL_SCALE_TEXEL_OFFSET,
    isFrontFace,
  );
  return applyNormalMapAtOffsetForHit(hit, frameNormal, fallbackNormal, normalMapOffset, normalScaleOffset);
}

fn applyNormalMapForHit(hit: IntersectionResult, baseNormal: vec3f) -> vec3f {
  let baseMapped = applyNormalMapAtOffsetForHit(
    hit,
    baseNormal,
    baseNormal,
    MATERIAL_MAP_NORMAL_TEXEL_OFFSET,
    MATERIAL_MAP_NORMAL_SCALE_TEXEL_OFFSET,
  );
  return applyFaceLayerNormalMapForHit(hit, baseNormal, baseMapped);
}

fn applyClearcoatNormalMapForHit(hit: IntersectionResult, frameNormal: vec3f, fallbackNormal: vec3f) -> vec3f {
  return applyNormalMapAtOffsetForHit(
    hit,
    frameNormal,
    fallbackNormal,
    MATERIAL_MAP_CLEARCOAT_NORMAL_TEXEL_OFFSET,
    MATERIAL_MAP_CLEARCOAT_NORMAL_SCALE_TEXEL_OFFSET,
  );
}

fn applyBumpMapForHit(hit: IntersectionResult, shadingNormal: vec3f) -> vec3f {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_BUMP_TEXEL_OFFSET;
  let meta0 = textureLoad(baseColorMapMeta, baseColorMapMetaCoord(metaTexel), 0);
  if (i32(meta0.x) < 0) {
    return shadingNormal;
  }

  let scaleMeta = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_BUMP_SCALE_TEXEL_OFFSET),
    0,
  );
  let bumpScale = scaleMeta.x;
  if (abs(bumpScale) < 1e-8) {
    return shadingNormal;
  }

  let hC = sampleMaterialAtlasRawAtOffsetForHit(
    hit,
    MATERIAL_MAP_BUMP_TEXEL_OFFSET,
  );
  if (hC.x < 0.0) {
    return shadingNormal;
  }

  let atlasDims = textureDimensions(materialTextureAtlas);
  let atlasTexelStep = vec2f(
    1.0 / f32(max(atlasDims.x, 1u)),
    1.0 / f32(max(atlasDims.y, 1u)),
  );
  let bumpTexelStep = vec2f(
    1.0 / max(scaleMeta.y, 1.0),
    1.0 / max(scaleMeta.z, 1.0),
  );
  let texelStep = select(atlasTexelStep, bumpTexelStep, scaleMeta.y > 0.0 && scaleMeta.z > 0.0);
  let hU = sampleMaterialAtlasRawAtOffsetDeltaForHit(
    hit,
    MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    vec2f(texelStep.x, 0.0),
  ).r;
  let hV = sampleMaterialAtlasRawAtOffsetDeltaForHit(
    hit,
    MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    vec2f(0.0, texelStep.y),
  ).r;
  let dhdu = (hU - hC.r) / texelStep.x;
  let dhdv = (hV - hC.r) / texelStep.y;
  let frame = materialTangentFrameForHit(hit, shadingNormal, MATERIAL_MAP_BUMP_TEXEL_OFFSET);
  let perturbed = shadingNormal - bumpScale * (dhdu * frame.tangent + dhdv * frame.bitangent);
  let plen = length(perturbed);
  let n = select(shadingNormal, perturbed / plen, plen > 1e-6);
  return select(-n, n, dot(n, shadingNormal) >= 0.0);
}

struct RestirDIMaterialPayload {
  albedo: vec3f,
  rough: f32,
  metal: f32,
  envMapIntensity: f32,
  clearcoatNormal: vec3f,
  specular: vec4f,
  anisotropy: vec2f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  iridescence: vec4f,
  clearcoat: vec2f,
  sheen: vec4f,
  sheenRoughness: f32,
  layerTransmission: vec3f,
  volumeScattering: vec4f,
  bulkThickness: f32,
};

fn sampleRestirDIMaterialPayloadForHit(
  hit: IntersectionResult,
  smoothNormal: vec3f,
  shadingNormal: vec3f,
  scalarBaseColor: vec3f,
  materialWord: u32,
  viewDirection: vec3f,
) -> RestirDIMaterialPayload {
  let vertexColor = sampleVertexColorForHit(hit);
  let layerControls = sampleFaceLayerControls(hit.indices.w, hit.side >= 0.0);
  var payload: RestirDIMaterialPayload;
  payload.albedo = sampleBaseColorMap(hit, scalarBaseColor * vertexColor.rgb);
  payload.rough = faceLayerRoughness(
    sampleMaterialScalarMap(hit, MATERIAL_MAP_SLOT_ROUGHNESS, 1u, decodeRoughMetal(materialWord).x),
    layerControls,
  );
  payload.metal = sampleMaterialScalarMap(hit, MATERIAL_MAP_SLOT_METALLIC, 2u, decodeRoughMetal(materialWord).y);
  payload.envMapIntensity = sampleEnvMapIntensity(hit.indices.w);
  payload.clearcoatNormal = applyClearcoatNormalMapForHit(hit, smoothNormal, shadingNormal);
  payload.specular = sampleSpecularControls(hit);
  payload.anisotropy = sampleAnisotropyControls(hit);
  let anisotropyFrame = materialTangentFrameForHit(hit, shadingNormal, MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET);
  payload.anisotropyTangent = anisotropyFrame.tangent;
  payload.anisotropyBitangent = anisotropyFrame.bitangent;
  payload.iridescence = sampleIridescenceControls(hit);
  payload.clearcoat = sampleClearcoatControls(hit);
  payload.sheen = sampleSheenControls(hit);
  payload.sheenRoughness = sampleSheenRoughness(hit);
  payload.layerTransmission = faceLayerTransmission(layerControls);
  payload.volumeScattering = sampleVolumeScatteringControls(hit.indices.w);
  payload.bulkThickness = materialOpticalThickness(hit.indices.w);
  let film = materialThinFilmResponse(
    hit.indices.w,
    hit.side >= 0.0,
    abs(dot(shadingNormal, safe_normalize(viewDirection))),
  );
  if (film.present != 0u) {
    // RGB always represents absolute dielectric F0, so thin-film reflectance
    // has an explicit representation and needs no value-range sentinel.
    payload.specular = vec4f(film.reflectance, 1.0);
    payload.iridescence = vec4f(0.0);
    payload.layerTransmission = payload.layerTransmission * film.transmittance;
  }
  return payload;
}

fn materialScalarAlphaDiscardedFromWord(materialWord: u32) -> bool {
  return (materialWord & 4u) != 0u;
}

// A dedicated per-triangle atlas metadata texel records MaterialSpec side
// semantics without borrowing any of the independent rough/metal/IOR/AO word.
const MATERIAL_SIDE_FLAG_DOUBLE_SIDED: u32 = 1u;

fn materialSideFlagsForTri(triIndex: u32) -> u32 {
  let sideMeta = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(
      triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_SIDE_FLAGS_TEXEL_OFFSET,
    ),
    0,
  );
  return u32(max(sideMeta.x, 0.0) + 0.5);
}

fn materialSideAdmittedForHit(hit: IntersectionResult) -> bool {
  if (hit.side >= 0.0) { return true; }
  let doubleSided =
    (materialSideFlagsForTri(hit.indices.w) & MATERIAL_SIDE_FLAG_DOUBLE_SIDED) != 0u;
  // A back interface of a closed transmissive volume must remain traversable
  // even when doubleSided is false, otherwise a path can enter but never exit.
  let transmissive = ((hit.matColorPacked >> 4u) & 0xFu) != 0u;
  return doubleSided || transmissive;
}

fn materialAlphaBlendCoverageHash(
  hit: IntersectionResult,
  ray: Ray,
  layer: u32,
  sampleSeed: u32,
) -> f32 {
  let uvSeed = vec2u(
    u32(clamp(hit.uv.x, 0.0, 1.0) * 65535.0),
    u32(clamp(hit.uv.y, 0.0, 1.0) * 65535.0),
  );
  let barySeed = vec3u(
    u32(clamp(hit.barycoord.x, 0.0, 1.0) * 65535.0),
    u32(clamp(hit.barycoord.y, 0.0, 1.0) * 65535.0),
    u32(clamp(hit.barycoord.z, 0.0, 1.0) * 65535.0),
  );
  let originBits = bitcast<vec3u>(ray.origin);
  let directionBits = bitcast<vec3u>(ray.direction);
  var seed =
    (hit.indices.x * 73856093u) ^
    (hit.indices.y * 19349663u) ^
    (hit.indices.z * 83492791u) ^
    (hit.indices.w * 2654435761u) ^
    (hit.instanceIndex * 1597334677u) ^
    (uvSeed.x * 3812015801u) ^
    (uvSeed.y * 2798796415u) ^
    (barySeed.x * 1103515245u) ^
    (barySeed.y * 12345u) ^
    (barySeed.z * 374761393u) ^
    (originBits.x * 2246822519u) ^
    (originBits.y * 3266489917u) ^
    (originBits.z * 668265263u) ^
    (directionBits.x * 374761393u) ^
    (directionBits.y * 1274126177u) ^
    (directionBits.z * 1431374977u) ^
    (layer * 0x9e3779b9u) ^
    (sampleSeed * 0x85ebca6bu);
  seed = seed * 747796405u + 2891336453u;
  let word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
  return f32((word >> 22u) ^ word) / 4294967296.0;
}

struct MaterialAlphaCoverage {
  mode: u32,
  coverage: f32,
  cutoff: f32,
  scalarDiscarded: u32,
};

fn materialAlphaCoverageForHit(
  hit: IntersectionResult,
  materialWord: u32,
) -> MaterialAlphaCoverage {
  var out: MaterialAlphaCoverage;
  out.mode = 0u;
  out.coverage = 1.0;
  out.cutoff = 0.0;
  out.scalarDiscarded = select(0u, 1u, materialScalarAlphaDiscardedFromWord(materialWord));
  if (out.scalarDiscarded != 0u) {
    return out;
  }

  let coverageMeta = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(hit.indices.w * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET),
    0,
  );
  out.mode = u32(max(coverageMeta.x, 0.0) + 0.5);
  if (out.mode == 0u) {
    return out;
  }

  let baseColorTexel = sampleMaterialAtlasRawForHit(hit, MATERIAL_MAP_SLOT_BASE_COLOR);
  let baseColorAlpha = select(clamp(baseColorTexel.a, 0.0, 1.0), 1.0, baseColorTexel.x < 0.0);
  let alphaTexel = sampleMaterialAtlasRawForHit(hit, MATERIAL_MAP_SLOT_ALPHA);
  let alphaMapCoverage = select(clamp(alphaTexel.r, 0.0, 1.0), 1.0, alphaTexel.x < 0.0);
  let vertexColorAlpha = sampleVertexColorForHit(hit).a;
  let opacity = clamp(coverageMeta.y, 0.0, 1.0);
  out.cutoff = clamp(coverageMeta.z, 0.0, 1.0);
  out.coverage = clamp(opacity * vertexColorAlpha * baseColorAlpha * alphaMapCoverage, 0.0, 1.0);
  return out;
}

fn materialAlphaDiscardedForHit(
  hit: IntersectionResult,
  materialWord: u32,
  ray: Ray,
  layer: u32,
  sampleSeed: u32,
) -> bool {
  if (!materialSideAdmittedForHit(hit)) {
    return true;
  }
  let alpha = materialAlphaCoverageForHit(hit, materialWord);
  if (alpha.scalarDiscarded != 0u) {
    return true;
  }

  if (alpha.mode == 0u) {
    return false;
  }
  if (alpha.mode == 1u) {
    return alpha.coverage < alpha.cutoff;
  }
  if (alpha.mode == 2u) {
    return alpha.coverage < 1.0 &&
      materialAlphaBlendCoverageHash(hit, ray, layer, sampleSeed) >= alpha.coverage;
  }
  return alpha.coverage <= 0.0;
}

${makeTexturedFirstHitAlphaMaskWalkerWGSL('traceSceneFirstHitAlphaMaskTextured', 'materialAlphaDiscardedForHit')}

fn materialAlphaDiscardedForOpaquePass(
  hit: IntersectionResult,
  materialWord: u32,
  _ray: Ray,
  _layer: u32,
  _sampleSeed: u32,
) -> bool {
  if (!materialSideAdmittedForHit(hit)) {
    return true;
  }
  let alpha = materialAlphaCoverageForHit(hit, materialWord);
  if (alpha.scalarDiscarded != 0u) {
    return true;
  }
  if (alpha.mode == 0u) {
    return false;
  }
  if (alpha.mode == 1u) {
    return alpha.coverage < alpha.cutoff;
  }
  if (alpha.mode == 2u) {
    // Camera-primary coverage is owned by TransparentOitPass. Keeping partial
    // and fully-covered blend surfaces out of the opaque background prevents
    // double counting and ensures their authored blend shading is not skipped.
    return true;
  }
  return alpha.coverage <= 0.0;
}

${makeTexturedFirstHitAlphaMaskWalkerWGSL(
  'traceSceneFirstHitAlphaMaskTexturedOpaqueOnly',
  'materialAlphaDiscardedForOpaquePass',
)}

fn materialShadowTransmittanceForHit(
  hit: IntersectionResult,
  materialWord: u32,
  skipGlass: bool,
) -> f32 {
  if (!materialSideAdmittedForHit(hit)) {
    return 1.0;
  }
  if ((materialWord & 1u) != 0u) {
    return 1.0;
  }
  if (skipGlass) {
    if (packedMaterialHasTransmission(hit.matColorPacked)) {
      return 1.0;
    }
  }
  let alpha = materialAlphaCoverageForHit(hit, materialWord);
  if (alpha.scalarDiscarded != 0u) {
    return 1.0;
  }
  if (alpha.mode == 0u) {
    return 0.0;
  }
  if (alpha.mode == 1u) {
    return select(0.0, 1.0, alpha.coverage < alpha.cutoff);
  }
  if (alpha.mode == 2u) {
    return clamp(1.0 - alpha.coverage, 0.0, 1.0);
  }
  return select(0.0, 1.0, alpha.coverage <= 0.0);
}

fn traceSceneAlphaTransmittanceTextured(
  bvhMode: u32,
  tlasNodeCount: u32,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  skipGlass: bool,
  materialMask: texture_2d<u32>,
  materialMaskWidth: u32,
) -> f32 {
  var walkRay: Ray;
  walkRay.origin = origin;
  walkRay.direction = dir;
  var traveled = 0.0;
  var tau = 1.0;
  let step = max(1e-4, triEps * 4.0);
  for (var i = 0u; i < 32u; i = i + 1u) {
    let remaining = tMax - traveled;
    if (remaining <= step || tau <= 0.0) {
      return clamp(tau, 0.0, 1.0);
    }
    let hit = traceSceneFirstHit(
      bvhMode, tlasNodeCount,
      walkRay, triEps,
    );
    if (!hit.didHit || hit.dist >= remaining) {
      return clamp(tau, 0.0, 1.0);
    }
    let word = textureLoad(
      materialMask,
      vec2i(i32(hit.indices.w % materialMaskWidth), i32(hit.indices.w / materialMaskWidth)),
      0,
    ).r;
    tau = tau * materialShadowTransmittanceForHit(hit, word, skipGlass);
    if (tau <= 0.0) {
      return 0.0;
    }
    traveled = traveled + hit.dist + step;
    walkRay.origin = origin + dir * traveled;
  }

  if (traceSceneAnyCastMask(
    bvhMode, tlasNodeCount,
    walkRay.origin, dir, max(0.0, tMax - traveled), triEps, skipGlass,
    materialMask, materialMaskWidth,
  )) {
    return 0.0;
  }
  return clamp(tau, 0.0, 1.0);
}

`;

export const MATERIAL_ATLAS_MODULE: WgslModule = {
  name: 'materialAtlas',
  source: MATERIAL_ATLAS_WGSL,
  requires: ['sceneTraversal', 'materialDecode'],
};
