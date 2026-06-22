import type { WgslModule } from '../pipeline/wgslComposer.js';

export const MATERIAL_ATLAS_WGSL = /* wgsl */ `
// Phase-3D material-map atlas. The host stores readable material TextureRefs as
// RGBA32F array layers plus per-triangle metadata. The helper below implements
// sampler policy manually with textureLoad so compute passes and fragment passes
// consume identical material-map samples.
@group(1) @binding(20) var materialTextureAtlas: texture_2d_array<f32>;
@group(1) @binding(21) var baseColorMapMeta: texture_2d<f32>;
@group(1) @binding(22) var bvh_tangent: texture_2d<f32>;
@group(1) @binding(23) var bvh_vertex_color: texture_2d<f32>;
@group(1) @binding(11) var<storage, read> bvh_normal: array<vec4f>;

const BASE_COLOR_MAP_META_TEX_WIDTH: u32 = 4096u;
const MATERIAL_MAP_META_TEXELS_PER_TRI: u32 = 55u;
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
const MATERIAL_MAP_CLEARCOAT_FACTOR_TEXEL_OFFSET: u32 = 28u;
const MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET: u32 = 30u;
const MATERIAL_MAP_SHEEN_COLOR_MAP_TEXEL_OFFSET: u32 = 32u;
const MATERIAL_MAP_SHEEN_ROUGHNESS_TEXEL_OFFSET: u32 = 34u;
const MATERIAL_MAP_CLEARCOAT_NORMAL_TEXEL_OFFSET: u32 = 36u;
const MATERIAL_MAP_CLEARCOAT_NORMAL_SCALE_TEXEL_OFFSET: u32 = 38u;
const MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET: u32 = 39u;
const MATERIAL_MAP_ANISOTROPY_SCALAR_TEXEL_OFFSET: u32 = 41u;
const MATERIAL_MAP_IRIDESCENCE_TEXEL_OFFSET: u32 = 42u;
const MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET: u32 = 44u;
const MATERIAL_MAP_IRIDESCENCE_SCALAR_TEXEL_OFFSET: u32 = 46u;
const MATERIAL_MAP_THICKNESS_TEXEL_OFFSET: u32 = 47u;
const MATERIAL_MAP_BUMP_TEXEL_OFFSET: u32 = 49u;
const MATERIAL_MAP_BUMP_SCALE_TEXEL_OFFSET: u32 = 51u;
const MATERIAL_MAP_ENV_INTENSITY_TEXEL_OFFSET: u32 = 52u;
const MATERIAL_MAP_FRONT_LAYER_TEXEL_OFFSET: u32 = 53u;
const MATERIAL_MAP_BACK_LAYER_TEXEL_OFFSET: u32 = 54u;

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

fn materialAtlasFilterMode(samplerPacked: u32) -> u32 {
  let magFilter = (samplerPacked >> 8u) & 0x1u;
  let minFilter = (samplerPacked >> 9u) & 0x1u;
  return select(magFilter, minFilter, magFilter != minFilter);
}

fn sampleMaterialAtlasNearestBaseLevel(wrapped: vec2f, layer: i32) -> vec4f {
  let dims = textureDimensions(materialTextureAtlas);
  let texel = vec2i(
    i32(min(u32(floor(wrapped.x * f32(dims.x))), dims.x - 1u)),
    i32(min(u32(floor(wrapped.y * f32(dims.y))), dims.y - 1u)),
  );
  return textureLoad(materialTextureAtlas, texel, layer, 0);
}

fn sampleMaterialAtlasLinearBaseLevel(wrapped: vec2f, layer: i32, samplerPacked: u32) -> vec4f {
  let dims = textureDimensions(materialTextureAtlas);
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
  let c00 = textureLoad(materialTextureAtlas, vec2i(x0, y0), layer, 0);
  let c10 = textureLoad(materialTextureAtlas, vec2i(x1, y0), layer, 0);
  let c01 = textureLoad(materialTextureAtlas, vec2i(x0, y1), layer, 0);
  let c11 = textureLoad(materialTextureAtlas, vec2i(x1, y1), layer, 0);
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

fn sampleMaterialAtlasBaseLevel(wrapped: vec2f, layer: i32, samplerPacked: u32) -> vec4f {
  if (materialAtlasFilterMode(samplerPacked) == 0u) {
    return sampleMaterialAtlasNearestBaseLevel(wrapped, layer);
  }
  return sampleMaterialAtlasLinearBaseLevel(wrapped, layer, samplerPacked);
}

fn interpolateUv1FromNormalW(hit: IntersectionResult, n0: vec4f, n1: vec4f, n2: vec4f) -> vec2f {
  let uvA = unpack2x16float(bitcast<u32>(n0.w));
  let uvB = unpack2x16float(bitcast<u32>(n1.w));
  let uvC = unpack2x16float(bitcast<u32>(n2.w));
  return hit.barycoord.x * uvA + hit.barycoord.y * uvB + hit.barycoord.z * uvC;
}

fn materialAtlasUv1ForHit(hit: IntersectionResult) -> vec2f {
  let n0 = bvh_normal[hit.indices.x];
  let n1 = bvh_normal[hit.indices.y];
  let n2 = bvh_normal[hit.indices.z];
  return interpolateUv1FromNormalW(hit, n0, n1, n2);
}

fn materialAtlasPackedUvFromVec4(v: vec4f) -> vec2f {
  return unpack2x16float(bitcast<u32>(v.w));
}

fn sampleMaterialAtlasRawAtOffsetDelta(
  triIndex: u32,
  metaOffset: u32,
  uv0: vec2f,
  uv1: vec2f,
  transformedDelta: vec2f,
) -> vec4f {
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
  ) + meta0.zw + transformedDelta;
  let wrapped = wrapMaterialUv(transformed, wrapPacked);
  return sampleMaterialAtlasBaseLevel(wrapped, layer, wrapPacked);
}

fn sampleMaterialAtlasRawAtOffset(triIndex: u32, metaOffset: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  return sampleMaterialAtlasRawAtOffsetDelta(triIndex, metaOffset, uv0, uv1, vec2f(0.0));
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
  return clamp(fallback * materialMapChannel(texelColor, channel), 0.0, 1.0);
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

fn sampleEnvMapIntensity(triIndex: u32) -> f32 {
  let intensityMeta = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_ENV_INTENSITY_TEXEL_OFFSET),
    0,
  );
  return max(intensityMeta.x, 0.0);
}

fn sampleFaceLayerControls(triIndex: u32, isFrontFace: bool) -> vec4f {
  let front = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_FRONT_LAYER_TEXEL_OFFSET),
    0,
  );
  let back = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_BACK_LAYER_TEXEL_OFFSET),
    0,
  );
  return select(back, front, isFrontFace);
}

fn faceLayerTransmission(layer: vec4f) -> vec3f {
  return clamp(layer.rgb, vec3f(0.0), vec3f(1.0));
}

fn faceLayerRoughness(roughness: f32, layer: vec4f) -> f32 {
  return select(roughness, clamp(layer.a, 0.0, 1.0), layer.a >= 0.0);
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

fn sampleClearcoatControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
  let cc = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET),
    0,
  );
  var factor = clamp(cc.x, 0.0, 1.0);
  var roughness = clamp(cc.y, 0.0, 1.0);

  let clearcoatMap = sampleMaterialAtlasRawAtOffset(triIndex, MATERIAL_MAP_CLEARCOAT_FACTOR_TEXEL_OFFSET, uv0, uv1);
  if (clearcoatMap.x >= 0.0) {
    factor = clamp(factor * clearcoatMap.r, 0.0, 1.0);
  }

  let roughnessMap = sampleMaterialAtlasRawAtOffset(triIndex, MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET, uv0, uv1);
  if (roughnessMap.x >= 0.0) {
    roughness = clamp(roughness * roughnessMap.g, 0.0, 1.0);
  }

  return vec2f(factor, roughness);
}

fn sampleSheenControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
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

  let colorMap = sampleMaterialAtlasRawAtOffset(triIndex, MATERIAL_MAP_SHEEN_COLOR_MAP_TEXEL_OFFSET, uv0, uv1);
  if (colorMap.x >= 0.0) {
    sheenColor = clamp(sheenColor * colorMap.rgb, vec3f(0.0), vec3f(1.0));
  }

  return vec4f(sheenColor, sheen);
}

fn sampleSheenRoughness(triIndex: u32, uv0: vec2f, uv1: vec2f) -> f32 {
  let scalars = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET),
    0,
  );
  var roughness = clamp(scalars.w, 0.0, 1.0);
  let roughnessMap = sampleMaterialAtlasRawAtOffset(triIndex, MATERIAL_MAP_SHEEN_ROUGHNESS_TEXEL_OFFSET, uv0, uv1);
  if (roughnessMap.x >= 0.0) {
    roughness = clamp(roughness * roughnessMap.a, 0.0, 1.0);
  }
  return roughness;
}

fn sampleAnisotropyControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
  let scalars = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_ANISOTROPY_SCALAR_TEXEL_OFFSET),
    0,
  );
  var strength = clamp(scalars.x, 0.0, 1.0);
  var rotation = scalars.y;

  let anisoMap = sampleMaterialAtlasRawAtOffset(triIndex, MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET, uv0, uv1);
  if (anisoMap.x >= 0.0) {
    strength = clamp(strength * anisoMap.b, 0.0, 1.0);
    let direction = anisoMap.rg * 2.0 - vec2f(1.0);
    if (dot(direction, direction) > 1e-6) {
      rotation += atan2(direction.y, direction.x);
    }
  }

  return vec2f(strength, rotation);
}

fn sampleIridescenceControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  let scalars = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_IRIDESCENCE_SCALAR_TEXEL_OFFSET),
    0,
  );
  var factor = clamp(scalars.x, 0.0, 1.0);
  let ior = max(1.0, scalars.y);
  var thicknessMin = max(0.0, scalars.z);
  var thicknessMax = max(0.0, scalars.w);

  let iridescenceMap = sampleMaterialAtlasRawAtOffset(triIndex, MATERIAL_MAP_IRIDESCENCE_TEXEL_OFFSET, uv0, uv1);
  if (iridescenceMap.x >= 0.0) {
    factor = clamp(factor * iridescenceMap.r, 0.0, 1.0);
  }

  let thicknessMap = sampleMaterialAtlasRawAtOffset(triIndex, MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET, uv0, uv1);
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
  return pow(max(beerAlbedo, vec3f(1e-6)), vec3f(thicknessFactor));
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
    let tOk = isTlas && tBase + 2u < arrayLength(&tlasInstanceLocalToWorld);
    if (tOk) {
      authoredTangent = transformDirectionCols(
        tlasInstanceLocalToWorld[tBase],
        tlasInstanceLocalToWorld[tBase + 1u],
        tlasInstanceLocalToWorld[tBase + 2u],
        authoredTangent,
      );
      authoredHandedness = authoredHandedness * tangentHandednessForLocalToWorld(
        tlasInstanceLocalToWorld[tBase],
        tlasInstanceLocalToWorld[tBase + 1u],
        tlasInstanceLocalToWorld[tBase + 2u],
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
  var bitangent = dp2;
  if (abs(det) > 1e-8) {
    let invDet = 1.0 / det;
    tangent = (dp1 * duv2.y - dp2 * duv1.y) * invDet;
    bitangent = (dp2 * duv1.x - dp1 * duv2.x) * invDet;
  }
  let isTlas = ubo.bvhMode == 1u;
  let tBase = hit.instanceIndex * 4u;
  let tOk = isTlas && tBase + 2u < arrayLength(&tlasInstanceLocalToWorld);
  if (tOk) {
    tangent = transformDirectionCols(
      tlasInstanceLocalToWorld[tBase],
      tlasInstanceLocalToWorld[tBase + 1u],
      tlasInstanceLocalToWorld[tBase + 2u],
      tangent,
    );
    bitangent = transformDirectionCols(
      tlasInstanceLocalToWorld[tBase],
      tlasInstanceLocalToWorld[tBase + 1u],
      tlasInstanceLocalToWorld[tBase + 2u],
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

  return preferAuthoredTangentFrameForHit(hit, frameNormal, tangent, bitangent);
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

  let uv1 = materialAtlasUv1ForHit(hit);
  let texelColor = sampleMaterialAtlasRawAtOffset(
    triIndex,
    normalMapOffset,
    hit.uv,
    uv1,
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

fn applyNormalMapForHit(hit: IntersectionResult, baseNormal: vec3f) -> vec3f {
  return applyNormalMapAtOffsetForHit(
    hit,
    baseNormal,
    baseNormal,
    MATERIAL_MAP_NORMAL_TEXEL_OFFSET,
    MATERIAL_MAP_NORMAL_SCALE_TEXEL_OFFSET,
  );
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

  let uv1 = materialAtlasUv1ForHit(hit);
  let hC = sampleMaterialAtlasRawAtOffset(
    triIndex,
    MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    hit.uv,
    uv1,
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
  let hU = sampleMaterialAtlasRawAtOffsetDelta(
    triIndex,
    MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    hit.uv,
    uv1,
    vec2f(texelStep.x, 0.0),
  ).r;
  let hV = sampleMaterialAtlasRawAtOffsetDelta(
    triIndex,
    MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    hit.uv,
    uv1,
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
};

fn sampleRestirDIMaterialPayloadForHit(
  hit: IntersectionResult,
  smoothNormal: vec3f,
  shadingNormal: vec3f,
  scalarBaseColor: vec3f,
  materialWord: u32,
) -> RestirDIMaterialPayload {
  let uv1 = materialAtlasUv1ForHit(hit);
  let vertexColor = sampleVertexColorForHit(hit);
  let layerControls = sampleFaceLayerControls(hit.indices.w, hit.side >= 0.0);
  var payload: RestirDIMaterialPayload;
  payload.albedo = sampleBaseColorMap(hit.indices.w, hit.uv, uv1, scalarBaseColor * vertexColor.rgb);
  payload.rough = faceLayerRoughness(
    sampleMaterialScalarMap(hit.indices.w, MATERIAL_MAP_SLOT_ROUGHNESS, 1u, hit.uv, uv1, decodeRoughMetal(materialWord).x),
    layerControls,
  );
  payload.metal = sampleMaterialScalarMap(hit.indices.w, MATERIAL_MAP_SLOT_METALLIC, 2u, hit.uv, uv1, decodeRoughMetal(materialWord).y);
  payload.envMapIntensity = sampleEnvMapIntensity(hit.indices.w);
  payload.clearcoatNormal = applyClearcoatNormalMapForHit(hit, smoothNormal, shadingNormal);
  payload.specular = sampleSpecularControls(hit.indices.w, hit.uv, uv1);
  payload.anisotropy = sampleAnisotropyControls(hit.indices.w, hit.uv, uv1);
  let anisotropyFrame = materialTangentFrameForHit(hit, shadingNormal, MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET);
  payload.anisotropyTangent = anisotropyFrame.tangent;
  payload.anisotropyBitangent = anisotropyFrame.bitangent;
  payload.iridescence = sampleIridescenceControls(hit.indices.w, hit.uv, uv1);
  payload.clearcoat = sampleClearcoatControls(hit.indices.w, hit.uv, uv1);
  payload.sheen = sampleSheenControls(hit.indices.w, hit.uv, uv1);
  payload.sheenRoughness = sampleSheenRoughness(hit.indices.w, hit.uv, uv1);
  payload.layerTransmission = faceLayerTransmission(layerControls);
  return payload;
}

fn materialScalarAlphaDiscardedFromWord(materialWord: u32) -> bool {
  return (materialWord & 4u) != 0u;
}

fn materialAlphaBlendCoverageHash(hit: IntersectionResult) -> f32 {
  let uvSeed = vec2u(
    u32(clamp(hit.uv.x, 0.0, 1.0) * 65535.0),
    u32(clamp(hit.uv.y, 0.0, 1.0) * 65535.0),
  );
  let barySeed = vec3u(
    u32(clamp(hit.barycoord.x, 0.0, 1.0) * 65535.0),
    u32(clamp(hit.barycoord.y, 0.0, 1.0) * 65535.0),
    u32(clamp(hit.barycoord.z, 0.0, 1.0) * 65535.0),
  );
  var seed =
    hit.indices.x * 73856093u ^
    hit.indices.y * 19349663u ^
    hit.indices.z * 83492791u ^
    hit.indices.w * 2654435761u ^
    hit.instanceIndex * 1597334677u ^
    uvSeed.x * 3812015801u ^
    uvSeed.y * 2798796415u ^
    barySeed.x * 1103515245u ^
    barySeed.y * 12345u ^
    barySeed.z * 374761393u;
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

  let uv1 = materialAtlasUv1ForHit(hit);
  let baseColorTexel = sampleMaterialAtlasRaw(hit.indices.w, MATERIAL_MAP_SLOT_BASE_COLOR, hit.uv, uv1);
  let baseColorAlpha = select(clamp(baseColorTexel.a, 0.0, 1.0), 1.0, baseColorTexel.x < 0.0);
  let alphaTexel = sampleMaterialAtlasRaw(hit.indices.w, MATERIAL_MAP_SLOT_ALPHA, hit.uv, uv1);
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
) -> bool {
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
    return alpha.coverage < 1.0 && materialAlphaBlendCoverageHash(hit) >= alpha.coverage;
  }
  return alpha.coverage <= 0.0;
}

fn materialAlphaDiscardedForOpaquePass(
  hit: IntersectionResult,
  materialWord: u32,
) -> bool {
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
    return alpha.coverage < 0.999;
  }
  return alpha.coverage <= 0.0;
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

fn materialShadowOccluderForHit(
  hit: IntersectionResult,
  materialWord: u32,
  skipGlass: bool,
) -> bool {
  return materialShadowTransmittanceForHit(hit, materialWord, skipGlass) <= 0.001;
}

fn materialShadowTransmittanceForHit(
  hit: IntersectionResult,
  materialWord: u32,
  skipGlass: bool,
) -> f32 {
  if ((materialWord & 1u) != 0u) {
    return 1.0;
  }
  if (skipGlass) {
    let trans4 = (hit.matColorPacked >> 4u) & 0xFu;
    if (trans4 > 4u) {
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
  bvh_index: ptr<storage, array<vec4u>, read>,
  bvh_position: ptr<storage, array<vec4f>, read>,
  bvh: ptr<storage, array<BVHNode>, read>,
  tlasNodes: ptr<storage, array<BVHNode>, read>,
  tlasInstanceIndices: ptr<storage, array<u32>, read>,
  tlasBlasRoots: ptr<storage, array<u32>, read>,
  tlasInstanceWorldToLocal: ptr<storage, array<vec4f>, read>,
  tlasInstanceLocalToWorld: ptr<storage, array<vec4f>, read>,
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
    if (remaining <= step || tau <= 0.001) {
      return clamp(tau, 0.0, 1.0);
    }
    let hit = traceSceneFirstHit(
      bvhMode, tlasNodeCount,
      bvh_index, bvh_position, bvh,
      tlasNodes, tlasInstanceIndices, tlasBlasRoots,
      tlasInstanceWorldToLocal, tlasInstanceLocalToWorld,
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
    if (tau <= 0.001) {
      return 0.0;
    }
    traveled = traveled + hit.dist + step;
    walkRay.origin = origin + dir * traveled;
  }

  if (traceSceneAnyCastMask(
    bvhMode, tlasNodeCount,
    bvh_index, bvh_position, bvh,
    tlasNodes, tlasInstanceIndices, tlasBlasRoots,
    tlasInstanceWorldToLocal, tlasInstanceLocalToWorld,
    walkRay.origin, dir, max(0.0, tMax - traveled), triEps, skipGlass,
    materialMask, materialMaskWidth,
  )) {
    return 0.0;
  }
  return clamp(tau, 0.0, 1.0);
}

fn traceSceneAnyAlphaMaskTextured(
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
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  skipGlass: bool,
  materialMask: texture_2d<u32>,
  materialMaskWidth: u32,
) -> bool {
  return traceSceneAlphaTransmittanceTextured(
    bvhMode, tlasNodeCount,
    bvh_index, bvh_position, bvh,
    tlasNodes, tlasInstanceIndices, tlasBlasRoots,
    tlasInstanceWorldToLocal, tlasInstanceLocalToWorld,
    origin, dir, tMax, triEps, skipGlass,
    materialMask, materialMaskWidth,
  ) <= 0.001;
}

fn traceSceneFirstHitAlphaMaskTexturedOpaqueOnly(
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
    if (!materialAlphaDiscardedForOpaquePass(hit, word)) {
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
    if (materialAlphaDiscardedForOpaquePass(exhausted, word)) {
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
