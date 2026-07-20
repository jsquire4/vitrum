/**
 * RC material-atlas decode helpers — raw WGSL fragment.
 *
 * Extracted verbatim (byte-identical) from `probeRayCast.wgsl.ts` so the 1500-line
 * assembly root stays legible. This fragment carries the RC material-map texel-slot
 * constants and the atlas/meta sampling helpers (meta coord math, UV wrap/pack,
 * per-slot atlas sampling, normal/bump map perturbation, tangent-frame derivation,
 * and the `rcSampleProbeHitMaterial` gather).
 *
 * IMPORTANT: these helpers reference **consumer bindings** declared in
 * `probeRayCast.wgsl.ts` (`rc_materialMapMeta`, `rc_materialTextureAtlas`,
 * `rc_geom_*`, `rc_tlas_l2w`, `rc_u_arr`). Per the composeWgsl ordering constraint,
 * this fragment is a RAW STRING interpolated into the consumer body AFTER those
 * bindings are declared — it is NOT a standalone `WgslModule`. The composed output
 * is byte-identical to the pre-split single-file literal (pinned by
 * `__tests__/probeRayCastByteIdentity.test.ts`).
 *
 * The `RC_MATERIAL_MAP_META_TEXELS_PER_TRI = 62u` stride is cross-checked against
 * the CPU atlas producer + main material shader by `probeRayCastWgsl.test.ts`.
 *
 * The offset-constant block is single-sourced from `@vitrum/shared-bvh`
 * (`buildMaterialAtlasOffsetConstsWGSL`, T4-2 2026-07-20) — the emitted string
 * reproduces the historical hand-written `RC_MATERIAL_MAP_*` block byte-for-byte
 * (pinned by `probeRayCastByteIdentity.test.ts`). The decode fns below stay RC-
 * specific (see the divergence note in materialAtlasOffsets.wgsl.ts).
 */
import { buildMaterialAtlasOffsetConstsWGSL } from '@vitrum/shared-bvh';

export const RC_MATERIAL_ATLAS_WGSL = /* wgsl */ `${buildMaterialAtlasOffsetConstsWGSL({
  prefix: 'RC_',
  include: [
    'META_TEXELS_PER_TRI',
    'SLOT_BASE_COLOR',
    'SLOT_ROUGHNESS',
    'SLOT_METALLIC',
    'SLOT_ALPHA',
    'ALPHA_COVERAGE_TEXEL_OFFSET',
    'EMISSIVE_TEXEL_OFFSET',
    'NORMAL_TEXEL_OFFSET',
    'NORMAL_SCALE_TEXEL_OFFSET',
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
    'BUMP_TEXEL_OFFSET',
    'BUMP_SCALE_TEXEL_OFFSET',
    'FRONT_LAYER_NORMAL_TEXEL_OFFSET',
    'FRONT_LAYER_NORMAL_SCALE_TEXEL_OFFSET',
    'BACK_LAYER_NORMAL_TEXEL_OFFSET',
    'BACK_LAYER_NORMAL_SCALE_TEXEL_OFFSET',
  ],
})}
const RC_PI: f32 = 3.14159265;
const RC_INV_PI: f32 = 0.31830988618;

fn rcMaterialMetaCoord(texel: u32) -> vec2i {
  let dims = textureDimensions(rc_materialMapMeta);
  let w = max(dims.x, 1u);
  return vec2i(i32(texel % w), i32(texel / w));
}

fn rcMaterialMetaAvailable(triIndex: u32, metaOffset: u32) -> bool {
  let dims = textureDimensions(rc_materialMapMeta);
  let texel = triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset;
  return texel < dims.x * dims.y;
}

fn rcMaterialMetaLoadOrZero(triIndex: u32, metaOffset: u32) -> vec4f {
  let texel = triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset;
  if (!rcMaterialMetaAvailable(triIndex, metaOffset)) {
    return vec4f(0.0);
  }
  return textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(texel), 0);
}

fn rcWrapMaterialUv1(v: f32, mode: u32) -> f32 {
  if (mode == 1u) {
    return clamp(v, 0.0, 1.0);
  }
  if (mode == 2u) {
    return 1.0 - abs(fract(v * 0.5) * 2.0 - 1.0);
  }
  return fract(v);
}

fn rcWrapMaterialUv(uv: vec2f, wrapPacked: u32) -> vec2f {
  let wrapS = wrapPacked & 0x3u;
  let wrapT = (wrapPacked >> 2u) & 0x3u;
  return vec2f(rcWrapMaterialUv1(uv.x, wrapS), rcWrapMaterialUv1(uv.y, wrapT));
}

fn rcPackedUvFromVec4(v: vec4f) -> vec2f {
  return unpack2x16float(bitcast<u32>(v.w));
}

struct RCHitMaterialUvs {
  valid: u32,
  uv0: vec2f,
  uv1: vec2f,
};

fn rcHitMaterialUvs(hit: IntersectionResult) -> RCHitMaterialUvs {
  var out: RCHitMaterialUvs;
  out.valid = 0u;
  out.uv0 = vec2f(0.0);
  out.uv1 = vec2f(0.0);

  let i0 = hit.indices.x;
  let i1 = hit.indices.y;
  let i2 = hit.indices.z;
  if (
    hit.indices.w >= arrayLength(&rc_geom_index) ||
    i0 >= arrayLength(&rc_geom_position) || i1 >= arrayLength(&rc_geom_position) || i2 >= arrayLength(&rc_geom_position) ||
    i0 >= arrayLength(&rc_geom_normal) || i1 >= arrayLength(&rc_geom_normal) || i2 >= arrayLength(&rc_geom_normal)
  ) {
    return out;
  }

  out.valid = 1u;
  out.uv0 =
    hit.barycoord.x * rcPackedUvFromVec4(rc_geom_position[i0]) +
    hit.barycoord.y * rcPackedUvFromVec4(rc_geom_position[i1]) +
    hit.barycoord.z * rcPackedUvFromVec4(rc_geom_position[i2]);
  out.uv1 =
    hit.barycoord.x * rcPackedUvFromVec4(rc_geom_normal[i0]) +
    hit.barycoord.y * rcPackedUvFromVec4(rc_geom_normal[i1]) +
    hit.barycoord.z * rcPackedUvFromVec4(rc_geom_normal[i2]);
  return out;
}

fn rcEmitterSubdivWeightAt(i: u32, j: u32, level: u32) -> vec3f {
  let invLevel = 1.0 / f32(max(level, 1u));
  let u = f32(i) * invLevel;
  let v = f32(j) * invLevel;
  return vec3f(1.0 - u - v, u, v);
}

fn rcEmitterParentBarycentricFromLocal(localBary: vec3f, levelF: f32, ordinalF: f32) -> vec3f {
  let level = min(16u, max(1u, u32(round(max(levelF, 1.0)))));
  if (level <= 1u) {
    return localBary;
  }

  let ordinal = u32(round(max(ordinalF, 0.0)));
  var cursor = 0u;
  for (var i = 0u; i < level; i = i + 1u) {
    for (var j = 0u; j < level - i; j = j + 1u) {
      let a = rcEmitterSubdivWeightAt(i, j, level);
      let b = rcEmitterSubdivWeightAt(i + 1u, j, level);
      let c = rcEmitterSubdivWeightAt(i, j + 1u, level);
      if (cursor == ordinal) {
        return localBary.x * a + localBary.y * b + localBary.z * c;
      }
      cursor = cursor + 1u;

      if (i + j < level - 1u) {
        let d = rcEmitterSubdivWeightAt(i + 1u, j + 1u, level);
        if (cursor == ordinal) {
          return localBary.x * b + localBary.y * d + localBary.z * c;
        }
        cursor = cursor + 1u;
      }
    }
  }

  return localBary;
}

fn rcSampleMaterialAtlasRawAtOffsetDelta(
  triIndex: u32,
  metaOffset: u32,
  uv0: vec2f,
  uv1: vec2f,
  transformedDelta: vec2f,
) -> vec4f {
  let metaDims = textureDimensions(rc_materialMapMeta);
  let metaTexel = triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset;
  if (metaTexel + 1u >= metaDims.x * metaDims.y) {
    return vec4f(-1.0);
  }
  let meta0 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(metaTexel), 0);
  let layer = i32(meta0.x);
  if (layer < 0 || u32(layer) >= textureNumLayers(rc_materialTextureAtlas)) {
    return vec4f(-1.0);
  }
  let wrapPacked = u32(max(meta0.y, 0.0) + 0.5);
  let texCoord = (wrapPacked >> 4u) & 0x3u;
  let uv = select(uv0, uv1, texCoord == 1u);
  let meta1 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(metaTexel + 1u), 0);
  let scaled = uv * meta1.xy;
  let transformed = vec2f(
    scaled.x * meta1.z - scaled.y * meta1.w,
    scaled.x * meta1.w + scaled.y * meta1.z,
  ) + meta0.zw + transformedDelta;
  let wrapped = rcWrapMaterialUv(transformed, wrapPacked);
  let dims = textureDimensions(rc_materialTextureAtlas);
  let texel = vec2i(
    i32(min(u32(floor(wrapped.x * f32(dims.x))), dims.x - 1u)),
    i32(min(u32(floor(wrapped.y * f32(dims.y))), dims.y - 1u)),
  );
  return textureLoad(rc_materialTextureAtlas, texel, layer, 0);
}

fn rcSampleMaterialAtlasRawAtOffset(triIndex: u32, metaOffset: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  return rcSampleMaterialAtlasRawAtOffsetDelta(triIndex, metaOffset, uv0, uv1, vec2f(0.0));
}

fn rcSampleMaterialAtlasRaw(triIndex: u32, slot: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  return rcSampleMaterialAtlasRawAtOffset(triIndex, slot * 2u, uv0, uv1);
}

fn rcSampleSurfaceEmissiveMap(hit: IntersectionResult, scalarEmission: vec3f) -> vec3f {
  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return scalarEmission;
  }
  let texel = rcSampleMaterialAtlasRawAtOffset(
    hit.indices.w,
    RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
  );
  if (texel.x < 0.0) {
    return scalarEmission;
  }
  return scalarEmission * texel.rgb;
}

fn rcMaterialMapChannel(v: vec4f, channel: u32) -> f32 {
  if (channel == 1u) { return v.g; }
  if (channel == 2u) { return v.b; }
  if (channel == 3u) { return v.a; }
  return v.r;
}

fn rcSampleMaterialScalarMap(
  triIndex: u32,
  slot: u32,
  channel: u32,
  uv0: vec2f,
  uv1: vec2f,
  fallback: f32,
) -> f32 {
  let texel = rcSampleMaterialAtlasRaw(triIndex, slot, uv0, uv1);
  if (texel.x < 0.0) {
    return fallback;
  }
  return clamp(fallback * rcMaterialMapChannel(texel, channel), 0.0, 1.0);
}

fn rcSampleSpecularControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  var color = vec3f(1.0);
  var intensity = 1.0;
  if (rcMaterialMetaAvailable(triIndex, RC_MATERIAL_MAP_SPECULAR_TEXEL_OFFSET)) {
    let spec = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_SPECULAR_TEXEL_OFFSET);
    color = clamp(spec.rgb, vec3f(0.0), vec3f(1.0));
    intensity = clamp(spec.a, 0.0, 1.0);
  }

  let colorMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET, uv0, uv1);
  if (colorMap.x >= 0.0) {
    color = clamp(color * colorMap.rgb, vec3f(0.0), vec3f(1.0));
  }
  let intensityMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET, uv0, uv1);
  if (intensityMap.x >= 0.0) {
    intensity = clamp(intensity * intensityMap.a, 0.0, 1.0);
  }
  return vec4f(color, intensity);
}

fn rcSampleClearcoatControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
  let cc = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET);
  var factor = clamp(cc.x, 0.0, 1.0);
  var roughness = clamp(cc.y, 0.0, 1.0);

  let clearcoatMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_CLEARCOAT_FACTOR_TEXEL_OFFSET, uv0, uv1);
  if (clearcoatMap.x >= 0.0) {
    factor = clamp(factor * clearcoatMap.r, 0.0, 1.0);
  }
  let roughnessMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET, uv0, uv1);
  if (roughnessMap.x >= 0.0) {
    roughness = clamp(roughness * roughnessMap.g, 0.0, 1.0);
  }
  return vec2f(factor, roughness);
}

fn rcSampleSheenControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  let scalars = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET);
  let colorMeta = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_SHEEN_COLOR_TEXEL_OFFSET);
  var sheenColor = clamp(colorMeta.rgb, vec3f(0.0), vec3f(1.0));
  var sheen = clamp(scalars.z, 0.0, 1.0);

  let colorMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_SHEEN_COLOR_MAP_TEXEL_OFFSET, uv0, uv1);
  if (colorMap.x >= 0.0) {
    sheenColor = clamp(sheenColor * colorMap.rgb, vec3f(0.0), vec3f(1.0));
  }
  return vec4f(sheenColor, sheen);
}

fn rcSampleSheenRoughness(triIndex: u32, uv0: vec2f, uv1: vec2f) -> f32 {
  let scalars = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET);
  var roughness = clamp(scalars.w, 0.0, 1.0);
  let roughnessMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_SHEEN_ROUGHNESS_TEXEL_OFFSET, uv0, uv1);
  if (roughnessMap.x >= 0.0) {
    roughness = clamp(roughness * roughnessMap.a, 0.0, 1.0);
  }
  return roughness;
}

fn rcSampleAnisotropyControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
  let scalars = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_ANISOTROPY_SCALAR_TEXEL_OFFSET);
  var strength = clamp(scalars.x, 0.0, 1.0);
  var rotation = scalars.y;

  let anisoMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET, uv0, uv1);
  if (anisoMap.x >= 0.0) {
    strength = clamp(strength * anisoMap.b, 0.0, 1.0);
    let direction = anisoMap.rg * 2.0 - vec2f(1.0);
    if (dot(direction, direction) > 1e-6) {
      rotation += atan2(direction.y, direction.x);
    }
  }
  return vec2f(strength, rotation);
}

fn rcSampleIridescenceControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  let scalars = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_IRIDESCENCE_SCALAR_TEXEL_OFFSET);
  var factor = clamp(scalars.x, 0.0, 1.0);
  let ior = max(1.0, scalars.y);
  var thicknessMin = max(0.0, scalars.z);
  var thicknessMax = max(0.0, scalars.w);

  let iridescenceMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_IRIDESCENCE_TEXEL_OFFSET, uv0, uv1);
  if (iridescenceMap.x >= 0.0) {
    factor = clamp(factor * iridescenceMap.r, 0.0, 1.0);
  }
  let thicknessMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET, uv0, uv1);
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

fn rcSmoothNormalForHit(hit: IntersectionResult, fallbackNormal: vec3f) -> vec3f {
  let i0 = hit.indices.x;
  let i1 = hit.indices.y;
  let i2 = hit.indices.z;
  if (i0 >= arrayLength(&rc_geom_normal) || i1 >= arrayLength(&rc_geom_normal) || i2 >= arrayLength(&rc_geom_normal)) {
    return fallbackNormal;
  }
  let n = normalize(
    hit.barycoord.x * rc_geom_normal[i0].xyz +
    hit.barycoord.y * rc_geom_normal[i1].xyz +
    hit.barycoord.z * rc_geom_normal[i2].xyz
  );
  return select(-n, n, dot(n, fallbackNormal) >= 0.0);
}

struct RCMaterialTangentFrame {
  tangent: vec3f,
  bitangent: vec3f,
}

fn rcBvhTangentTexel(vertexIndex: u32) -> vec4f {
  let dims = textureDimensions(rc_geom_tangent);
  let width = u32(dims.x);
  let height = u32(dims.y);
  if (width == 0u || height == 0u) {
    return vec4f(0.0);
  }
  let y = vertexIndex / width;
  if (y >= height) {
    return vec4f(0.0);
  }
  return textureLoad(rc_geom_tangent, vec2i(i32(vertexIndex % width), i32(y)), 0);
}

fn rcBvhVertexColorTexel(vertexIndex: u32) -> vec4f {
  let dims = textureDimensions(rc_geom_vertex_color);
  let width = u32(dims.x);
  let height = u32(dims.y);
  if (width == 0u || height == 0u) {
    return vec4f(1.0);
  }
  let y = vertexIndex / width;
  if (y >= height) {
    return vec4f(1.0);
  }
  return clamp(textureLoad(rc_geom_vertex_color, vec2i(i32(vertexIndex % width), i32(y)), 0), vec4f(0.0), vec4f(1.0));
}

fn rcSampleVertexColorForHit(hit: IntersectionResult) -> vec4f {
  let ca = rcBvhVertexColorTexel(hit.indices.x);
  let cb = rcBvhVertexColorTexel(hit.indices.y);
  let cc = rcBvhVertexColorTexel(hit.indices.z);
  return clamp(
    ca * hit.barycoord.x +
    cb * hit.barycoord.y +
    cc * hit.barycoord.z,
    vec4f(0.0),
    vec4f(1.0)
  );
}

fn rcTransformDirectionCols(l2w0: vec4f, l2w1: vec4f, l2w2: vec4f, v: vec3f) -> vec3f {
  return l2w0.xyz * v.x + l2w1.xyz * v.y + l2w2.xyz * v.z;
}

fn rcTangentHandednessForLocalToWorld(l2w0: vec4f, l2w1: vec4f, l2w2: vec4f) -> f32 {
  let det = dot(l2w0.xyz, cross(l2w1.xyz, l2w2.xyz));
  return select(-1.0, 1.0, det >= 0.0);
}

fn rcPreferAuthoredTangentFrameForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  fallbackTangent: vec3f,
  fallbackBitangent: vec3f,
) -> RCMaterialTangentFrame {
  var tangent = fallbackTangent;
  var bitangent = fallbackBitangent;

  let ta = rcBvhTangentTexel(hit.indices.x);
  let tb = rcBvhTangentTexel(hit.indices.y);
  let tc = rcBvhTangentTexel(hit.indices.z);
  var authoredTangent =
    hit.barycoord.x * ta.xyz +
    hit.barycoord.y * tb.xyz +
    hit.barycoord.z * tc.xyz;
  var authoredHandedness =
    hit.barycoord.x * ta.w +
    hit.barycoord.y * tb.w +
    hit.barycoord.z * tc.w;

  if (length(authoredTangent) > 1e-8 && abs(authoredHandedness) > 0.5) {
    let isTlas = rc_u_arr[0].bvhMode == 1u;
    let tBase = hit.instanceIndex * 4u;
    let tOk = isTlas && tBase + 2u < arrayLength(&rc_tlas_l2w);
    if (tOk) {
      authoredTangent = rcTransformDirectionCols(
        rc_tlas_l2w[tBase],
        rc_tlas_l2w[tBase + 1u],
        rc_tlas_l2w[tBase + 2u],
        authoredTangent,
      );
      authoredHandedness = authoredHandedness * rcTangentHandednessForLocalToWorld(
        rc_tlas_l2w[tBase],
        rc_tlas_l2w[tBase + 1u],
        rc_tlas_l2w[tBase + 2u],
      );
    }

    authoredTangent = authoredTangent - frameNormal * dot(frameNormal, authoredTangent);
    let tLen2 = dot(authoredTangent, authoredTangent);
    if (tLen2 > 1e-8) {
      tangent = authoredTangent * inverseSqrt(tLen2);
      bitangent = normalize(cross(frameNormal, tangent)) * select(-1.0, 1.0, authoredHandedness >= 0.0);
    }
  }

  return RCMaterialTangentFrame(tangent, bitangent);
}

fn rcFallbackBitangentForNormal(n: vec3f, t: vec3f) -> vec3f {
  let b = cross(n, t);
  let len2 = dot(b, b);
  if (len2 < 1e-8) {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.95);
    return normalize(cross(n, up));
  }
  return b * inverseSqrt(len2);
}

fn rcMaterialTangentFrameForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  mapOffset: u32,
) -> RCMaterialTangentFrame {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + mapOffset;
  let meta0 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(metaTexel), 0);
  let flags = u32(max(meta0.y, 0.0) + 0.5);
  let useUv1 = ((flags >> 4u) & 0x3u) == 1u;

  let p0 = rc_geom_position[hit.indices.x];
  let p1 = rc_geom_position[hit.indices.y];
  let p2 = rc_geom_position[hit.indices.z];
  let n0 = rc_geom_normal[hit.indices.x];
  let n1 = rc_geom_normal[hit.indices.y];
  let n2 = rc_geom_normal[hit.indices.z];
  let uv0a = rcPackedUvFromVec4(p0);
  let uv0b = rcPackedUvFromVec4(p1);
  let uv0c = rcPackedUvFromVec4(p2);
  let uv1a = rcPackedUvFromVec4(n0);
  let uv1b = rcPackedUvFromVec4(n1);
  let uv1c = rcPackedUvFromVec4(n2);
  let ta = select(uv0a, uv1a, useUv1);
  let tb = select(uv0b, uv1b, useUv1);
  let tc = select(uv0c, uv1c, useUv1);

  let dp1 = p1.xyz - p0.xyz;
  let dp2 = p2.xyz - p0.xyz;
  let duv1 = tb - ta;
  let duv2 = tc - ta;
  let det = duv1.x * duv2.y - duv1.y * duv2.x;
  var tangent = dp1;
  var bitangent = rcFallbackBitangentForNormal(frameNormal, tangent);
  if (abs(det) > 1e-8) {
    let invDet = 1.0 / det;
    tangent = (dp1 * duv2.y - dp2 * duv1.y) * invDet;
    bitangent = (dp2 * duv1.x - dp1 * duv2.x) * invDet;
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
    bitangent = rcFallbackBitangentForNormal(frameNormal, tangent);
  } else {
    bitangent = bitangent * inverseSqrt(bLen2);
  }

  return rcPreferAuthoredTangentFrameForHit(hit, frameNormal, tangent, bitangent);
}

fn rcApplyNormalMapAtOffsetForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  fallbackNormal: vec3f,
  normalMapOffset: u32,
  normalScaleOffset: u32,
) -> vec3f {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + normalMapOffset;
  let metaDims = textureDimensions(rc_materialMapMeta);
  if (metaTexel + 1u >= metaDims.x * metaDims.y) {
    return fallbackNormal;
  }
  let meta0 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(metaTexel), 0);
  if (i32(meta0.x) < 0) {
    return fallbackNormal;
  }

  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return fallbackNormal;
  }
  let texelColor = rcSampleMaterialAtlasRawAtOffset(triIndex, normalMapOffset, uvs.uv0, uvs.uv1);
  if (texelColor.x < 0.0) {
    return fallbackNormal;
  }

  let scaleMeta = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaCoord(triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + normalScaleOffset),
    0,
  );
  let normalScale = max(scaleMeta.x, 0.0);
  let tangentSample = normalize(vec3f(
    (texelColor.r * 2.0 - 1.0) * normalScale,
    (texelColor.g * 2.0 - 1.0) * normalScale,
    texelColor.b * 2.0 - 1.0,
  ));

  let frame = rcMaterialTangentFrameForHit(hit, frameNormal, normalMapOffset);
  let perturbed = normalize(frame.tangent * tangentSample.x + frame.bitangent * tangentSample.y + frameNormal * tangentSample.z);
  return select(-perturbed, perturbed, dot(perturbed, frameNormal) >= 0.0);
}

fn rcApplyFaceLayerNormalMapForHit(hit: IntersectionResult, frameNormal: vec3f, fallbackNormal: vec3f) -> vec3f {
  let isFrontFace = hit.side >= 0.0;
  let normalMapOffset = select(
    RC_MATERIAL_MAP_BACK_LAYER_NORMAL_TEXEL_OFFSET,
    RC_MATERIAL_MAP_FRONT_LAYER_NORMAL_TEXEL_OFFSET,
    isFrontFace,
  );
  let normalScaleOffset = select(
    RC_MATERIAL_MAP_BACK_LAYER_NORMAL_SCALE_TEXEL_OFFSET,
    RC_MATERIAL_MAP_FRONT_LAYER_NORMAL_SCALE_TEXEL_OFFSET,
    isFrontFace,
  );
  return rcApplyNormalMapAtOffsetForHit(hit, frameNormal, fallbackNormal, normalMapOffset, normalScaleOffset);
}

fn rcApplyNormalMapForHit(hit: IntersectionResult, baseNormal: vec3f) -> vec3f {
  let baseMapped = rcApplyNormalMapAtOffsetForHit(
    hit,
    baseNormal,
    baseNormal,
    RC_MATERIAL_MAP_NORMAL_TEXEL_OFFSET,
    RC_MATERIAL_MAP_NORMAL_SCALE_TEXEL_OFFSET,
  );
  return rcApplyFaceLayerNormalMapForHit(hit, baseNormal, baseMapped);
}

fn rcApplyClearcoatNormalMapForHit(hit: IntersectionResult, frameNormal: vec3f, fallbackNormal: vec3f) -> vec3f {
  return rcApplyNormalMapAtOffsetForHit(
    hit,
    frameNormal,
    fallbackNormal,
    RC_MATERIAL_MAP_CLEARCOAT_NORMAL_TEXEL_OFFSET,
    RC_MATERIAL_MAP_CLEARCOAT_NORMAL_SCALE_TEXEL_OFFSET,
  );
}

fn rcApplyBumpMapForHit(hit: IntersectionResult, shadingNormal: vec3f) -> vec3f {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET;
  let metaDims = textureDimensions(rc_materialMapMeta);
  if (metaTexel + 1u >= metaDims.x * metaDims.y) {
    return shadingNormal;
  }
  let meta0 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(metaTexel), 0);
  if (i32(meta0.x) < 0) {
    return shadingNormal;
  }

  let scaleMeta = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaCoord(triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + RC_MATERIAL_MAP_BUMP_SCALE_TEXEL_OFFSET),
    0,
  );
  let bumpScale = scaleMeta.x;
  if (abs(bumpScale) < 1e-8) {
    return shadingNormal;
  }

  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return shadingNormal;
  }
  let hC = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET, uvs.uv0, uvs.uv1);
  if (hC.x < 0.0) {
    return shadingNormal;
  }

  let atlasDims = textureDimensions(rc_materialTextureAtlas);
  let atlasTexelStep = vec2f(
    1.0 / f32(max(atlasDims.x, 1u)),
    1.0 / f32(max(atlasDims.y, 1u)),
  );
  let bumpTexelStep = vec2f(
    1.0 / max(scaleMeta.y, 1.0),
    1.0 / max(scaleMeta.z, 1.0),
  );
  let texelStep = select(atlasTexelStep, bumpTexelStep, scaleMeta.y > 0.0 && scaleMeta.z > 0.0);
  let hU = rcSampleMaterialAtlasRawAtOffsetDelta(
    triIndex,
    RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
    vec2f(texelStep.x, 0.0),
  ).r;
  let hV = rcSampleMaterialAtlasRawAtOffsetDelta(
    triIndex,
    RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
    vec2f(0.0, texelStep.y),
  ).r;
  let dhdu = (hU - hC.r) / texelStep.x;
  let dhdv = (hV - hC.r) / texelStep.y;
  let frame = rcMaterialTangentFrameForHit(hit, shadingNormal, RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET);
  let perturbed = shadingNormal - bumpScale * (dhdu * frame.tangent + dhdv * frame.bitangent);
  let plen = length(perturbed);
  let n = select(shadingNormal, perturbed / plen, plen > 1e-6);
  return select(-n, n, dot(n, shadingNormal) >= 0.0);
}

struct RCProbeHitMaterial {
  albedo: vec3f,
  roughness: f32,
  metalness: f32,
  specular: vec4f,
  clearcoat: vec2f,
  clearcoatNormal: vec3f,
  sheen: vec4f,
  sheenRoughness: f32,
  anisotropy: vec2f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  iridescence: vec4f,
}

fn rcSampleProbeHitMaterial(
  hit: IntersectionResult,
  scalarBaseColor: vec3f,
  scalarRoughness: f32,
  scalarMetalness: f32,
  frameNormal: vec3f,
  shadingNormal: vec3f,
) -> RCProbeHitMaterial {
  var out: RCProbeHitMaterial;
  out.albedo = scalarBaseColor;
  out.roughness = scalarRoughness;
  out.metalness = scalarMetalness;
  out.specular = vec4f(1.0);
  out.clearcoat = vec2f(0.0);
  out.clearcoatNormal = shadingNormal;
  out.sheen = vec4f(0.0);
  out.sheenRoughness = 0.0;
  out.anisotropy = vec2f(0.0);
  let defaultAnisotropyFrame = rcMaterialTangentFrameForHit(hit, shadingNormal, RC_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET);
  out.anisotropyTangent = defaultAnisotropyFrame.tangent;
  out.anisotropyBitangent = defaultAnisotropyFrame.bitangent;
  out.iridescence = vec4f(0.0, 1.0, 0.0, 0.0);
  out.clearcoatNormal = rcApplyClearcoatNormalMapForHit(hit, frameNormal, shadingNormal);

  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return out;
  }

  let baseColorTexel = rcSampleMaterialAtlasRaw(
    hit.indices.w,
    RC_MATERIAL_MAP_SLOT_BASE_COLOR,
    uvs.uv0,
    uvs.uv1,
  );
  if (baseColorTexel.x >= 0.0) {
    out.albedo = scalarBaseColor * baseColorTexel.rgb;
  }
  out.roughness = rcSampleMaterialScalarMap(
    hit.indices.w,
    RC_MATERIAL_MAP_SLOT_ROUGHNESS,
    1u,
    uvs.uv0,
    uvs.uv1,
    scalarRoughness,
  );
  out.metalness = rcSampleMaterialScalarMap(
    hit.indices.w,
    RC_MATERIAL_MAP_SLOT_METALLIC,
    2u,
    uvs.uv0,
    uvs.uv1,
    scalarMetalness,
  );
  out.specular = rcSampleSpecularControls(hit.indices.w, uvs.uv0, uvs.uv1);
  out.clearcoat = rcSampleClearcoatControls(hit.indices.w, uvs.uv0, uvs.uv1);
  out.sheen = rcSampleSheenControls(hit.indices.w, uvs.uv0, uvs.uv1);
  out.sheenRoughness = rcSampleSheenRoughness(hit.indices.w, uvs.uv0, uvs.uv1);
  out.anisotropy = rcSampleAnisotropyControls(hit.indices.w, uvs.uv0, uvs.uv1);
  let anisotropyFrame = rcMaterialTangentFrameForHit(hit, shadingNormal, RC_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET);
  out.anisotropyTangent = anisotropyFrame.tangent;
  out.anisotropyBitangent = anisotropyFrame.bitangent;
  out.iridescence = rcSampleIridescenceControls(hit.indices.w, uvs.uv0, uvs.uv1);
  return out;
}`;
