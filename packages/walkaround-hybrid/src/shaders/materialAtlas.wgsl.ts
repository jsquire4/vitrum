import type { WgslModule } from '../pipeline/wgslComposer.js';

export const MATERIAL_ATLAS_WGSL = /* wgsl */ `
// Phase-3D material-map atlas. The host stores readable material TextureRefs as
// RGBA32F array layers plus per-triangle metadata. textureLoad keeps this path
// sampler-free and preserves the scene group's storage-buffer budget.
@group(1) @binding(20) var materialTextureAtlas: texture_2d_array<f32>;
@group(1) @binding(21) var baseColorMapMeta: texture_2d<f32>;
@group(1) @binding(11) var<storage, read> bvh_normal: array<vec4f>;

const BASE_COLOR_MAP_META_TEX_WIDTH: u32 = 4096u;
const MATERIAL_MAP_META_TEXELS_PER_TRI: u32 = 28u;
const MATERIAL_MAP_SLOT_BASE_COLOR: u32 = 0u;
const MATERIAL_MAP_SLOT_ROUGHNESS: u32 = 1u;
const MATERIAL_MAP_SLOT_METALLIC: u32 = 2u;
const MATERIAL_MAP_SLOT_AO: u32 = 3u;
const MATERIAL_MAP_SLOT_ALPHA: u32 = 4u;
const MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET: u32 = 10u;
const MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET: u32 = 11u;
const MATERIAL_MAP_TRANSMISSION_TEXEL_OFFSET: u32 = 13u;
const MATERIAL_MAP_NORMAL_TEXEL_OFFSET: u32 = 15u;
const MATERIAL_MAP_NORMAL_SCALE_TEXEL_OFFSET: u32 = 17u;
const MATERIAL_MAP_LIGHT_TEXEL_OFFSET: u32 = 18u;
const MATERIAL_MAP_LIGHT_INTENSITY_TEXEL_OFFSET: u32 = 20u;
const MATERIAL_MAP_SPECULAR_TEXEL_OFFSET: u32 = 21u;
const MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET: u32 = 22u;
const MATERIAL_MAP_SHEEN_COLOR_TEXEL_OFFSET: u32 = 23u;
const MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET: u32 = 24u;
const MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET: u32 = 26u;

fn baseColorMapMetaCoord(texel: u32) -> vec2i {
  return vec2i(i32(texel % BASE_COLOR_MAP_META_TEX_WIDTH), i32(texel / BASE_COLOR_MAP_META_TEX_WIDTH));
}

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

fn interpolateUv1FromNormalW(hit: IntersectionResult, n0: vec4f, n1: vec4f, n2: vec4f) -> vec2f {
  let uvA = unpack2x16unorm(bitcast<u32>(n0.w));
  let uvB = unpack2x16unorm(bitcast<u32>(n1.w));
  let uvC = unpack2x16unorm(bitcast<u32>(n2.w));
  return hit.barycoord.x * uvA + hit.barycoord.y * uvB + hit.barycoord.z * uvC;
}

fn materialAtlasUv1ForHit(hit: IntersectionResult) -> vec2f {
  let n0 = bvh_normal[hit.indices.x];
  let n1 = bvh_normal[hit.indices.y];
  let n2 = bvh_normal[hit.indices.z];
  return interpolateUv1FromNormalW(hit, n0, n1, n2);
}

fn materialAtlasPackedUvFromVec4(v: vec4f) -> vec2f {
  return unpack2x16unorm(bitcast<u32>(v.w));
}

fn sampleMaterialAtlasRawAtOffset(triIndex: u32, metaOffset: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  let metaTexel = triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset;
  let meta0 = textureLoad(baseColorMapMeta, baseColorMapMetaCoord(metaTexel), 0);
  let layer = i32(meta0.x);
  if (layer < 0) {
    return vec4f(-1.0, -1.0, -1.0, -1.0);
  }
  let wrapPacked = u32(max(meta0.y, 0.0) + 0.5);
  let texCoord = (wrapPacked >> 4u) & 0x3u;
  let uv = select(uv0, uv1, texCoord == 1u);
  let meta1 = textureLoad(baseColorMapMeta, baseColorMapMetaCoord(metaTexel + 1u), 0);
  let scaled = uv * meta1.xy;
  let transformed = vec2f(
    scaled.x * meta1.z - scaled.y * meta1.w,
    scaled.x * meta1.w + scaled.y * meta1.z,
  ) + meta0.zw;
  let wrapped = wrapMaterialUv(transformed, wrapPacked);
  let dims = textureDimensions(materialTextureAtlas);
  let texel = vec2i(
    i32(min(u32(floor(wrapped.x * f32(dims.x))), dims.x - 1u)),
    i32(min(u32(floor(wrapped.y * f32(dims.y))), dims.y - 1u)),
  );
  return textureLoad(materialTextureAtlas, texel, layer, 0);
}

fn sampleMaterialAtlasRaw(triIndex: u32, slot: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  return sampleMaterialAtlasRawAtOffset(triIndex, slot * 2u, uv0, uv1);
}

fn materialMapChannel(v: vec4f, channel: u32) -> f32 {
  if (channel == 1u) { return v.g; }
  if (channel == 2u) { return v.b; }
  if (channel == 3u) { return v.a; }
  return v.r;
}

fn sampleBaseColorMap(triIndex: u32, uv0: vec2f, uv1: vec2f, scalarBaseColor: vec3f) -> vec3f {
  let texelColor = sampleMaterialAtlasRaw(triIndex, MATERIAL_MAP_SLOT_BASE_COLOR, uv0, uv1);
  if (texelColor.x < 0.0) {
    return scalarBaseColor;
  }
  return scalarBaseColor * texelColor.rgb;
}

fn sampleMaterialScalarMap(triIndex: u32, slot: u32, channel: u32, uv0: vec2f, uv1: vec2f, fallback: f32) -> f32 {
  let texelColor = sampleMaterialAtlasRaw(triIndex, slot, uv0, uv1);
  if (texelColor.x < 0.0) {
    return fallback;
  }
  return clamp(materialMapChannel(texelColor, channel), 0.0, 1.0);
}

fn sampleAoMapFactor(triIndex: u32, materialWord: u32, uv0: vec2f, uv1: vec2f) -> f32 {
  let rawOcclusion = sampleMaterialScalarMap(triIndex, MATERIAL_MAP_SLOT_AO, 0u, uv0, uv1, 1.0);
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
  let texelColor = sampleMaterialAtlasRawAtOffset(
    hit.indices.w,
    MATERIAL_MAP_TRANSMISSION_TEXEL_OFFSET,
    hit.uv,
    materialAtlasUv1ForHit(hit),
  );
  if (texelColor.x < 0.0) {
    return scalarTransmission;
  }
  return clamp(scalarTransmission * texelColor.r, 0.0, 1.0);
}

fn sampleLightMap(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec3f {
  let texelColor = sampleMaterialAtlasRawAtOffset(
    triIndex,
    MATERIAL_MAP_LIGHT_TEXEL_OFFSET,
    uv0,
    uv1,
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

fn sampleSpecularControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  let spec = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_SPECULAR_TEXEL_OFFSET),
    0,
  );
  var color = clamp(spec.rgb, vec3f(0.0), vec3f(1.0));
  var intensity = clamp(spec.a, 0.0, 1.0);

  let colorMap = sampleMaterialAtlasRawAtOffset(triIndex, MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET, uv0, uv1);
  if (colorMap.x >= 0.0) {
    color = clamp(color * colorMap.rgb, vec3f(0.0), vec3f(1.0));
  }

  let intensityMap = sampleMaterialAtlasRawAtOffset(triIndex, MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET, uv0, uv1);
  if (intensityMap.x >= 0.0) {
    intensity = clamp(intensity * intensityMap.a, 0.0, 1.0);
  }

  return vec4f(color, intensity);
}

fn sampleClearcoatControls(triIndex: u32) -> vec2f {
  let cc = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET),
    0,
  );
  return clamp(cc.xy, vec2f(0.0), vec2f(1.0));
}

fn sampleSheenControls(triIndex: u32) -> vec4f {
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
  return vec4f(clamp(color.rgb, vec3f(0.0), vec3f(1.0)), clamp(scalars.z, 0.0, 1.0));
}

fn sampleSheenRoughness(triIndex: u32) -> f32 {
  let scalars = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET),
    0,
  );
  return clamp(scalars.w, 0.0, 1.0);
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

fn applyNormalMapForHit(hit: IntersectionResult, baseNormal: vec3f) -> vec3f {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_NORMAL_TEXEL_OFFSET;
  let meta0 = textureLoad(baseColorMapMeta, baseColorMapMetaCoord(metaTexel), 0);
  if (i32(meta0.x) < 0) {
    return baseNormal;
  }

  let uv1 = materialAtlasUv1ForHit(hit);
  let texelColor = sampleMaterialAtlasRawAtOffset(
    triIndex,
    MATERIAL_MAP_NORMAL_TEXEL_OFFSET,
    hit.uv,
    uv1,
  );
  if (texelColor.x < 0.0) {
    return baseNormal;
  }

  let scaleMeta = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_NORMAL_SCALE_TEXEL_OFFSET),
    0,
  );
  let normalScale = max(scaleMeta.x, 0.0);
  let tangentSample = normalize(vec3f(
    (texelColor.r * 2.0 - 1.0) * normalScale,
    (texelColor.g * 2.0 - 1.0) * normalScale,
    texelColor.b * 2.0 - 1.0,
  ));

  let flags = u32(max(meta0.y, 0.0) + 0.5);
  let useUv1 = ((flags >> 4u) & 0x3u) == 1u;
  let p0 = bvh_position[hit.indices.x];
  let p1 = bvh_position[hit.indices.y];
  let p2 = bvh_position[hit.indices.z];
  let n0 = bvh_normal[hit.indices.x];
  let n1 = bvh_normal[hit.indices.y];
  let n2 = bvh_normal[hit.indices.z];
  let uv0a = materialAtlasPackedUvFromVec4(p0);
  let uv0b = materialAtlasPackedUvFromVec4(p1);
  let uv0c = materialAtlasPackedUvFromVec4(p2);
  let uv1a = materialAtlasPackedUvFromVec4(n0);
  let uv1b = materialAtlasPackedUvFromVec4(n1);
  let uv1c = materialAtlasPackedUvFromVec4(n2);
  let ta = select(uv0a, uv1a, useUv1);
  let tb = select(uv0b, uv1b, useUv1);
  let tc = select(uv0c, uv1c, useUv1);

  let dp1 = p1.xyz - p0.xyz;
  let dp2 = p2.xyz - p0.xyz;
  let duv1 = tb - ta;
  let duv2 = tc - ta;
  let det = duv1.x * duv2.y - duv1.y * duv2.x;
  var tangent = dp1;
  var bitangent = fallbackBitangentForNormal(baseNormal, tangent);
  if (abs(det) > 1e-8) {
    let invDet = 1.0 / det;
    tangent = (dp1 * duv2.y - dp2 * duv1.y) * invDet;
    bitangent = (dp2 * duv1.x - dp1 * duv2.x) * invDet;
  }

  tangent = tangent - baseNormal * dot(baseNormal, tangent);
  let tLen2 = dot(tangent, tangent);
  if (tLen2 < 1e-8) {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(baseNormal.y) > 0.95);
    tangent = normalize(cross(up, baseNormal));
  } else {
    tangent = tangent * inverseSqrt(tLen2);
  }
  bitangent = bitangent - baseNormal * dot(baseNormal, bitangent) - tangent * dot(tangent, bitangent);
  let bLen2 = dot(bitangent, bitangent);
  if (bLen2 < 1e-8) {
    bitangent = fallbackBitangentForNormal(baseNormal, tangent);
  } else {
    bitangent = bitangent * inverseSqrt(bLen2);
  }

  let perturbed = normalize(tangent * tangentSample.x + bitangent * tangentSample.y + baseNormal * tangentSample.z);
  return select(-perturbed, perturbed, dot(perturbed, baseNormal) >= 0.0);
}

fn materialScalarAlphaDiscardedFromWord(materialWord: u32) -> bool {
  return (materialWord & 4u) != 0u;
}

fn materialAlphaDiscardedForHit(
  hit: IntersectionResult,
  materialWord: u32,
) -> bool {
  if (materialScalarAlphaDiscardedFromWord(materialWord)) {
    return true;
  }

  let metaTexel = hit.indices.w * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_SLOT_ALPHA * 2u;
  let alphaMeta0 = textureLoad(baseColorMapMeta, baseColorMapMetaCoord(metaTexel), 0);
  if (i32(alphaMeta0.x) < 0) {
    return false;
  }

  let coverageMeta = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(hit.indices.w * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET),
    0,
  );
  let mode = u32(max(coverageMeta.x, 0.0) + 0.5);
  if (mode == 0u) {
    return false;
  }

  let uv1 = materialAtlasUv1ForHit(hit);
  let alphaTexel = sampleMaterialAtlasRaw(hit.indices.w, MATERIAL_MAP_SLOT_ALPHA, hit.uv, uv1);
  if (alphaTexel.x < 0.0) {
    return false;
  }
  let opacity = clamp(coverageMeta.y, 0.0, 1.0);
  let cutoff = clamp(coverageMeta.z, 0.0, 1.0);
  let coverage = opacity * clamp(alphaTexel.r, 0.0, 1.0);
  if (mode == 1u) {
    return coverage < cutoff;
  }
  return coverage <= 0.0;
}

fn traceSceneFirstHitAlphaMaskTextured(
  bvhMode: u32,
  tlasNodeCount: u32,
  bvh_index: ptr<storage, array<vec4u>, read>,
  bvh_position: ptr<storage, array<vec4f>, read>,
  bvh: ptr<storage, array<BVHNode>, read>,
  tlasNodes: ptr<storage, array<BVHNode>, read>,
  tlasInstanceIndices: ptr<storage, array<u32>, read>,
  tlasBlasRoots: ptr<storage, array<u32>, read>,
  tlasInstanceWorldToLocal: ptr<storage, array<vec4f>, read>,
  tlasInstanceLocalToWorld: ptr<storage, array<vec4f>, read>,
  ray: Ray,
  triEps: f32,
  materialMask: texture_2d<u32>,
  materialMaskWidth: u32,
) -> IntersectionResult {
  var walkRay = ray;
  var traveled = 0.0;
  let step = max(1e-4, triEps * 4.0);
  for (var i = 0u; i < 32u; i = i + 1u) {
    var hit = traceSceneFirstHit(
      bvhMode, tlasNodeCount,
      bvh_index, bvh_position, bvh,
      tlasNodes, tlasInstanceIndices, tlasBlasRoots,
      tlasInstanceWorldToLocal, tlasInstanceLocalToWorld,
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
    if (!materialAlphaDiscardedForHit(hit, word)) {
      hit.dist = hit.dist + traveled;
      return hit;
    }
    traveled = traveled + hit.dist + step;
    walkRay.origin = ray.origin + ray.direction * traveled;
  }
  var exhausted = traceSceneFirstHit(
    bvhMode, tlasNodeCount,
    bvh_index, bvh_position, bvh,
    tlasNodes, tlasInstanceIndices, tlasBlasRoots,
    tlasInstanceWorldToLocal, tlasInstanceLocalToWorld,
    walkRay, triEps,
  );
  if (exhausted.didHit) {
    let word = textureLoad(
      materialMask,
      vec2i(i32(exhausted.indices.w % materialMaskWidth), i32(exhausted.indices.w / materialMaskWidth)),
      0,
    ).r;
    if (materialAlphaDiscardedForHit(exhausted, word)) {
      exhausted.didHit = false;
    }
  }
  if (exhausted.didHit) {
    exhausted.dist = exhausted.dist + traveled;
  }
  return exhausted;
}
`;

export const MATERIAL_ATLAS_MODULE: WgslModule = {
  name: 'materialAtlas',
  source: MATERIAL_ATLAS_WGSL,
  requires: ['sceneTraversal', 'materialDecode'],
};
