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
 * `rc_geom_*`, packed TLAS loaders, `rc_u`). Per the composeWgsl ordering constraint,
 * this fragment is a RAW STRING interpolated into the consumer body AFTER those
 * bindings are declared — it is NOT a standalone `WgslModule`. The composed output
 * is byte-identical to the pre-split single-file literal (pinned by
 * `__tests__/probeRayCastByteIdentity.test.ts`).
 *
 * The `RC_MATERIAL_MAP_META_TEXELS_PER_TRI = 157u` logical stride is cross-checked against
 * the CPU atlas producer + main material shader by `probeRayCastWgsl.test.ts`.
 *
 * The offset-constant block is single-sourced from `@vitrum/shared-bvh`
 * (`buildMaterialAtlasOffsetConstsWGSL`, T4-2 2026-07-20) — the emitted string
 * reproduces the historical hand-written `RC_MATERIAL_MAP_*` block byte-for-byte
 * (pinned by `probeRayCastByteIdentity.test.ts`). The decode fns below stay RC-
 * specific (see the divergence note in materialAtlasOffsets.wgsl.ts).
 */
import {
  MATERIAL_OPTICS_WGSL,
  buildMaterialAtlasOffsetConstsWGSL,
} from '@vitrum/shared-bvh';

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
    'FRONT_LAYER_TEXEL_OFFSET',
    'BACK_LAYER_TEXEL_OFFSET',
    'VOLUME_SCATTERING_TEXEL_OFFSET',
    'FRONT_LAYER_NORMAL_TEXEL_OFFSET',
    'FRONT_LAYER_NORMAL_SCALE_TEXEL_OFFSET',
    'BACK_LAYER_NORMAL_TEXEL_OFFSET',
    'BACK_LAYER_NORMAL_SCALE_TEXEL_OFFSET',
  ],
})}
const RC_PI: f32 = 3.14159265;
const RC_INV_PI: f32 = 0.31830988618;

fn rcMaterialMetaRawCoord(texel: u32) -> vec2i {
  let dims = textureDimensions(rc_materialMapMeta);
  let w = max(dims.x, 1u);
  return vec2i(i32(texel % w), i32(texel / w));
}

fn rcMaterialMetaExactU32(value: f32) -> u32 {
  if (!(value >= 0.0) || value > 16777216.0 || value != floor(value)) {
    return 0xffffffffu;
  }
  return u32(value);
}

fn rcMaterialMetaPhysicalTexel(triIndex: u32, metaOffset: u32) -> u32 {
  let metaDims = textureDimensions(rc_materialMapMeta);
  let totalTexels = metaDims.x * metaDims.y;
  if (totalTexels < 4u) { return totalTexels; }
  let formatHeader = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaRawCoord(totalTexels - 4u),
    0,
  );
  let addressHeader = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaRawCoord(totalTexels - 3u),
    0,
  );
  let materialRecordCount = rcMaterialMetaExactU32(formatHeader.y);
  let triangleCount = rcMaterialMetaExactU32(formatHeader.z);
  if (
    rcMaterialMetaExactU32(formatHeader.x) != 3u ||
    materialRecordCount == 0u ||
    materialRecordCount == 0xffffffffu ||
    triangleCount == 0xffffffffu ||
    triIndex >= triangleCount ||
    rcMaterialMetaExactU32(formatHeader.w) != RC_MATERIAL_MAP_META_TEXELS_PER_TRI
  ) {
    return totalTexels;
  }
  let materialBase = rcMaterialMetaExactU32(addressHeader.x);
  let triangleMaterialBase = rcMaterialMetaExactU32(addressHeader.y);
  let uvAffineBase = rcMaterialMetaExactU32(addressHeader.z);
  let activeUvLaneCount = rcMaterialMetaExactU32(addressHeader.w);
  let payloadEnd = totalTexels - 4u;
  if (
    materialBase == 0xffffffffu ||
    triangleMaterialBase == 0xffffffffu ||
    uvAffineBase == 0xffffffffu ||
    activeUvLaneCount == 0xffffffffu ||
    materialBase > triangleMaterialBase ||
    triangleMaterialBase > uvAffineBase ||
    triangleMaterialBase > payloadEnd ||
    uvAffineBase > payloadEnd ||
    activeUvLaneCount > 14u
  ) {
    return totalTexels;
  }
  let directoryHeader = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaRawCoord(totalTexels - 2u),
    0,
  );
  let atlasAddressBase = rcMaterialMetaExactU32(directoryHeader.x);
  if (
    atlasAddressBase == 0xffffffffu ||
    uvAffineBase > atlasAddressBase ||
    atlasAddressBase > payloadEnd
  ) {
    return totalTexels;
  }
  let materialRegionTexels = triangleMaterialBase - materialBase;
  if (
    materialRecordCount >
      materialRegionTexels / RC_MATERIAL_MAP_META_TEXELS_PER_TRI ||
    materialRecordCount * RC_MATERIAL_MAP_META_TEXELS_PER_TRI !=
      materialRegionTexels
  ) {
    return totalTexels;
  }
  let triangleTableTexels = (triangleCount + 3u) / 4u;
  if (triangleTableTexels != uvAffineBase - triangleMaterialBase) {
    return totalTexels;
  }
  let uvStride = activeUvLaneCount * 2u;
  let uvRegionTexels = atlasAddressBase - uvAffineBase;
  if (
    (uvStride == 0u && uvRegionTexels != 0u) ||
    (uvStride != 0u &&
      (triangleCount > uvRegionTexels / uvStride ||
        triangleCount * uvStride != uvRegionTexels))
  ) {
    return totalTexels;
  }
  if (metaOffset >= 128u && metaOffset < 156u) {
    let laneWord = metaOffset - 128u;
    let lane = laneWord / 2u;
    if (lane >= activeUvLaneCount) { return totalTexels; }
    let availableUvTexels = atlasAddressBase - uvAffineBase;
    if (uvStride == 0u || triIndex > availableUvTexels / uvStride) {
      return totalTexels;
    }
    let triangleUvOffset = triIndex * uvStride;
    if (
      triangleUvOffset > availableUvTexels ||
      laneWord >= availableUvTexels - triangleUvOffset
    ) {
      return totalTexels;
    }
    return uvAffineBase + triangleUvOffset + laneWord;
  }
  let idTableOffset = triIndex / 4u;
  if (idTableOffset >= uvAffineBase - triangleMaterialBase) {
    return totalTexels;
  }
  let idTableTexel = triangleMaterialBase + idTableOffset;
  let ids = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaRawCoord(idTableTexel),
    0,
  );
  let materialId = rcMaterialMetaExactU32(ids[triIndex & 3u]);
  if (materialId >= materialRecordCount) { return totalTexels; }
  if (
    metaOffset >= RC_MATERIAL_MAP_META_TEXELS_PER_TRI ||
    materialId >
      (triangleMaterialBase - materialBase) /
        RC_MATERIAL_MAP_META_TEXELS_PER_TRI
  ) {
    return totalTexels;
  }
  let materialOffset = materialId * RC_MATERIAL_MAP_META_TEXELS_PER_TRI;
  if (
    materialOffset >= triangleMaterialBase - materialBase ||
    metaOffset >= triangleMaterialBase - materialBase - materialOffset
  ) {
    return totalTexels;
  }
  return materialBase + materialOffset + metaOffset;
}

fn rcMaterialMetaCoord(triIndex: u32, metaOffset: u32) -> vec2i {
  return rcMaterialMetaRawCoord(
    rcMaterialMetaPhysicalTexel(triIndex, metaOffset),
  );
}

fn rcMaterialMetaAvailable(triIndex: u32, metaOffset: u32) -> bool {
  let dims = textureDimensions(rc_materialMapMeta);
  let texel = rcMaterialMetaPhysicalTexel(triIndex, metaOffset);
  return texel < dims.x * dims.y;
}

fn rcMaterialMetaLoadOrZero(triIndex: u32, metaOffset: u32) -> vec4f {
  if (!rcMaterialMetaAvailable(triIndex, metaOffset)) {
    return vec4f(0.0);
  }
  return textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(triIndex, metaOffset), 0);
}

fn materialOpticalLoad(triIndex: u32, metaOffset: u32) -> vec4f {
  return rcMaterialMetaLoadOrZero(triIndex, metaOffset);
}

${MATERIAL_OPTICS_WGSL}

fn rcMaterialAtlasFiniteF32(value: f32) -> bool {
  return value == value && abs(value) <= VITRUM_OPTICAL_MAX_FINITE_F32;
}

fn rcMaterialAtlasFiniteVec2(value: vec2f) -> bool {
  return all(value == value) &&
    all(abs(value) <= vec2f(VITRUM_OPTICAL_MAX_FINITE_F32));
}

fn rcMaterialAtlasFiniteVec4(value: vec4f) -> bool {
  return all(value == value) &&
    all(abs(value) <= vec4f(VITRUM_OPTICAL_MAX_FINITE_F32));
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

fn rcWrapMaterialTexelIndex(index: i32, size: i32, mode: u32) -> i32 {
  if (size <= 1) { return 0; }
  if (mode == 1u) { return clamp(index, 0, size - 1); }
  if (mode == 2u) {
    let period = size * 2;
    var x = index % period;
    if (x < 0) { x = x + period; }
    return select(x, period - x - 1, x >= size);
  }
  var x = index % size;
  if (x < 0) { x = x + size; }
  return x;
}

fn rcMaterialAtlasFilterMode(samplerPacked: u32, lod: f32) -> u32 {
  let magFilter = (samplerPacked >> 10u) & 0x1u;
  let minFilter = (samplerPacked >> 11u) & 0x1u;
  return select(magFilter, minFilter, lod > 0.0);
}

const RC_MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER: u32 = 18u;
const RC_MATERIAL_ATLAS_ENCODING_RGBA8_UNORM: u32 = 0u;
const RC_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM: u32 = 1u;
const RC_MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT: u32 = 2u;
const RC_MATERIAL_ATLAS_ENCODING_RGBA16_UNORM: u32 = 3u;
const RC_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM: u32 = 4u;
const RC_MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT: u32 = 5u;

struct RCMaterialAtlasLayerAddress {
  encoding: u32,
  width: u32,
  height: u32,
  mipLevelCount: u32,
  decodeSrgb: u32,
  planeCount: u32,
  recordTexel: u32,
  valid: u32,
};

fn rcMaterialAtlasLayerAddress(layer: i32) -> RCMaterialAtlasLayerAddress {
  var out: RCMaterialAtlasLayerAddress;
  out.valid = 0u;
  if (layer < 0) { return out; }
  let metaDims = textureDimensions(rc_materialMapMeta);
  let totalTexels = metaDims.x * metaDims.y;
  if (totalTexels < 4u) { return out; }
  let directoryHeader = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaRawCoord(totalTexels - 2u),
    0,
  );
  let addressBase = rcMaterialMetaExactU32(directoryHeader.x);
  let layerCount = rcMaterialMetaExactU32(directoryHeader.y);
  let directoryEnd = totalTexels - 4u;
  if (
    addressBase == 0xffffffffu ||
    layerCount == 0xffffffffu ||
    addressBase > directoryEnd
  ) {
    return out;
  }
  let logicalLayer = u32(layer);
  if (logicalLayer >= layerCount) { return out; }
  let availableDirectoryTexels = directoryEnd - addressBase;
  if (
    logicalLayer >
      availableDirectoryTexels / RC_MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER
  ) {
    return out;
  }
  let layerOffset =
    logicalLayer * RC_MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER;
  if (
    layerOffset > availableDirectoryTexels ||
    RC_MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER >
      availableDirectoryTexels - layerOffset
  ) {
    return out;
  }
  let recordTexel = addressBase + layerOffset;
  if (
    recordTexel + RC_MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER >
    totalTexels - 4u
  ) {
    return out;
  }
  let info0 = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaRawCoord(recordTexel),
    0,
  );
  let info1 = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaRawCoord(recordTexel + 1u),
    0,
  );
  out.encoding = rcMaterialMetaExactU32(info0.x);
  out.width = rcMaterialMetaExactU32(info0.y);
  out.height = rcMaterialMetaExactU32(info0.z);
  out.mipLevelCount = rcMaterialMetaExactU32(info0.w);
  out.decodeSrgb = rcMaterialMetaExactU32(info1.x);
  out.planeCount = rcMaterialMetaExactU32(info1.y);
  out.recordTexel = recordTexel;
  let atlasDimensions = textureDimensions(rc_materialTextureAtlas);
  let encodingPlanePairValid =
    ((out.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA8_UNORM ||
      out.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM) &&
      out.planeCount == 1u) ||
    ((out.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT ||
      out.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA16_UNORM ||
      out.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM) &&
      out.planeCount == 2u) ||
    (out.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT &&
      out.planeCount == 4u);
  out.valid = select(
    0u,
    1u,
    out.width > 0u &&
    out.height > 0u &&
    atlasDimensions.x <= 1073741823u &&
    atlasDimensions.y <= 1073741823u &&
    out.width <= atlasDimensions.x &&
    out.height <= atlasDimensions.y &&
    out.mipLevelCount > 0u &&
    out.mipLevelCount <= 16u &&
    out.decodeSrgb <= 1u &&
    encodingPlanePairValid,
  );
  return out;
}

fn rcMaterialAtlasMapAvailableAtOffset(
  triIndex: u32,
  metaOffset: u32,
) -> bool {
  let metaDims = textureDimensions(rc_materialMapMeta);
  let totalTexels = metaDims.x * metaDims.y;
  let physicalTexel = rcMaterialMetaPhysicalTexel(triIndex, metaOffset);
  if (physicalTexel >= totalTexels) { return false; }
  let meta0 = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaRawCoord(physicalTexel),
    0,
  );
  if (
    !rcMaterialAtlasFiniteF32(meta0.x) ||
    meta0.x < 0.0 ||
    meta0.x > 16777215.0 ||
    floor(meta0.x) != meta0.x
  ) {
    return false;
  }
  return rcMaterialAtlasLayerAddress(i32(meta0.x)).valid != 0u;
}

fn rcMaterialAtlasLevelDimensions(
  address: RCMaterialAtlasLayerAddress,
  level: u32,
) -> vec2u {
  let divisor = 1u << min(level, 31u);
  return max(vec2u(1u), vec2u(address.width, address.height) / divisor);
}

fn rcMaterialAtlasSigned16(value: u32) -> i32 {
  let word = value & 0xffffu;
  return select(i32(word), i32(word) - 65536, word >= 32768u);
}

fn rcMaterialAtlasSrgbChannelToLinear(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  return select(c / 12.92, pow((c + 0.055) / 1.055, 2.4), c > 0.04045);
}

struct RCMaterialAtlasSampleResult {
  value: vec4f,
  valid: u32,
  encoding: u32,
};

fn rcMaterialAtlasInvalidSample() -> RCMaterialAtlasSampleResult {
  return RCMaterialAtlasSampleResult(vec4f(0.0), 0u, 0u);
}

fn rcMaterialAtlasValidSample(
  value: vec4f,
  encoding: u32,
) -> RCMaterialAtlasSampleResult {
  let finite = rcMaterialAtlasFiniteVec4(value);
  return RCMaterialAtlasSampleResult(
    select(vec4f(0.0), value, finite),
    select(0u, 1u, finite),
    encoding,
  );
}

fn rcMaterialAtlasDecodeTexel(
  address: RCMaterialAtlasLayerAddress,
  logicalTexel: vec2i,
  level: u32,
) -> RCMaterialAtlasSampleResult {
  if (address.valid == 0u || level >= address.mipLevelCount) {
    return rcMaterialAtlasInvalidSample();
  }
  let mipRecord = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaRawCoord(address.recordTexel + 2u + level),
    0,
  );
  let originX = rcMaterialMetaExactU32(mipRecord.x);
  let originY = rcMaterialMetaExactU32(mipRecord.y);
  let baseLayer = rcMaterialMetaExactU32(mipRecord.z);
  if (
    originX == 0xffffffffu ||
    originY == 0xffffffffu ||
    baseLayer == 0xffffffffu ||
    logicalTexel.x < 0 ||
    logicalTexel.y < 0
  ) {
    return rcMaterialAtlasInvalidSample();
  }
  let origin = vec2u(originX, originY);
  let atlasDims = textureDimensions(rc_materialTextureAtlas);
  let atlasLayers = textureNumLayers(rc_materialTextureAtlas);
  if (
    origin.x >= atlasDims.x ||
    origin.y >= atlasDims.y ||
    u32(logicalTexel.x) >= atlasDims.x - origin.x ||
    u32(logicalTexel.y) >= atlasDims.y - origin.y ||
    baseLayer >= atlasLayers ||
    address.planeCount > atlasLayers - baseLayer
  ) {
    return rcMaterialAtlasInvalidSample();
  }
  let coord = origin + vec2u(logicalTexel);
  let p0 = textureLoad(
    rc_materialTextureAtlas,
    vec2i(coord),
    i32(baseLayer),
    0,
  ).r;
  var value = vec4f(0.0);
  if (address.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA8_UNORM) {
    value = unpack4x8unorm(p0);
  } else if (address.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM) {
    value = unpack4x8snorm(p0);
  } else {
    let p1 = textureLoad(
      rc_materialTextureAtlas,
      vec2i(coord),
      i32(baseLayer + 1u),
      0,
    ).r;
    if (address.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT) {
      value = vec4f(unpack2x16float(p0), unpack2x16float(p1));
    } else if (address.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA16_UNORM) {
      value = vec4f(
        f32(p0 & 0xffffu),
        f32(p0 >> 16u),
        f32(p1 & 0xffffu),
        f32(p1 >> 16u),
      ) / 65535.0;
    } else if (address.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM) {
      value = max(
        vec4f(
          f32(rcMaterialAtlasSigned16(p0)),
          f32(rcMaterialAtlasSigned16(p0 >> 16u)),
          f32(rcMaterialAtlasSigned16(p1)),
          f32(rcMaterialAtlasSigned16(p1 >> 16u)),
        ) / 32767.0,
        vec4f(-1.0),
      );
    } else if (
      address.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT &&
      address.planeCount == 4u
    ) {
      let p2 = textureLoad(
        rc_materialTextureAtlas,
        vec2i(coord),
        i32(baseLayer + 2u),
        0,
      ).r;
      let p3 = textureLoad(
        rc_materialTextureAtlas,
        vec2i(coord),
        i32(baseLayer + 3u),
        0,
      ).r;
      value = vec4f(
        bitcast<f32>(p0),
        bitcast<f32>(p1),
        bitcast<f32>(p2),
        bitcast<f32>(p3),
      );
    } else {
      return rcMaterialAtlasInvalidSample();
    }
  }
  let decoded = rcMaterialAtlasValidSample(value, address.encoding);
  if (decoded.valid == 0u) {
    return rcMaterialAtlasInvalidSample();
  }
  value = decoded.value;
  if (address.decodeSrgb != 0u) {
    value = vec4f(
      rcMaterialAtlasSrgbChannelToLinear(value.r),
      rcMaterialAtlasSrgbChannelToLinear(value.g),
      rcMaterialAtlasSrgbChannelToLinear(value.b),
      value.a,
    );
  }
  return rcMaterialAtlasValidSample(value, address.encoding);
}

fn rcSampleMaterialAtlasNearestLevel(
  wrapped: vec2f,
  layer: i32,
  level: u32,
) -> RCMaterialAtlasSampleResult {
  let address = rcMaterialAtlasLayerAddress(layer);
  if (
    address.valid == 0u ||
    level >= address.mipLevelCount ||
    !rcMaterialAtlasFiniteVec2(wrapped) ||
    !all(wrapped >= vec2f(0.0)) ||
    !all(wrapped <= vec2f(1.0))
  ) {
    return rcMaterialAtlasInvalidSample();
  }
  let dims = rcMaterialAtlasLevelDimensions(address, level);
  let position = wrapped * vec2f(dims);
  if (!rcMaterialAtlasFiniteVec2(position)) {
    return rcMaterialAtlasInvalidSample();
  }
  let texel = vec2i(
    i32(min(u32(floor(position.x)), dims.x - 1u)),
    i32(min(u32(floor(position.y)), dims.y - 1u)),
  );
  return rcMaterialAtlasDecodeTexel(address, texel, level);
}

fn rcSampleMaterialAtlasLinearLevel(
  wrapped: vec2f,
  layer: i32,
  samplerPacked: u32,
  level: u32,
) -> RCMaterialAtlasSampleResult {
  let address = rcMaterialAtlasLayerAddress(layer);
  if (
    address.valid == 0u ||
    level >= address.mipLevelCount ||
    !rcMaterialAtlasFiniteVec2(wrapped) ||
    !all(wrapped >= vec2f(0.0)) ||
    !all(wrapped <= vec2f(1.0))
  ) {
    return rcMaterialAtlasInvalidSample();
  }
  let dims = rcMaterialAtlasLevelDimensions(address, level);
  let size = vec2i(i32(dims.x), i32(dims.y));
  let coord = wrapped * vec2f(f32(dims.x), f32(dims.y)) - vec2f(0.5);
  let base = vec2i(i32(floor(coord.x)), i32(floor(coord.y)));
  let fraction = coord - floor(coord);
  let wrapS = samplerPacked & 0x3u;
  let wrapT = (samplerPacked >> 2u) & 0x3u;
  let x0 = rcWrapMaterialTexelIndex(base.x, size.x, wrapS);
  let x1 = rcWrapMaterialTexelIndex(base.x + 1, size.x, wrapS);
  let y0 = rcWrapMaterialTexelIndex(base.y, size.y, wrapT);
  let y1 = rcWrapMaterialTexelIndex(base.y + 1, size.y, wrapT);
  let c00 = rcMaterialAtlasDecodeTexel(address, vec2i(x0, y0), level);
  let c10 = rcMaterialAtlasDecodeTexel(address, vec2i(x1, y0), level);
  let c01 = rcMaterialAtlasDecodeTexel(address, vec2i(x0, y1), level);
  let c11 = rcMaterialAtlasDecodeTexel(address, vec2i(x1, y1), level);
  if (c00.valid == 0u || c10.valid == 0u || c01.valid == 0u || c11.valid == 0u) {
    return rcMaterialAtlasInvalidSample();
  }
  return rcMaterialAtlasValidSample(
    mix(
      mix(c00.value, c10.value, fraction.x),
      mix(c01.value, c11.value, fraction.x),
      fraction.y,
    ),
    address.encoding,
  );
}

fn rcSampleMaterialAtlasLevel(
  wrapped: vec2f,
  layer: i32,
  samplerPacked: u32,
  level: u32,
  lod: f32,
) -> RCMaterialAtlasSampleResult {
  if (rcMaterialAtlasFilterMode(samplerPacked, lod) == 0u) {
    return rcSampleMaterialAtlasNearestLevel(wrapped, layer, level);
  }
  return rcSampleMaterialAtlasLinearLevel(
    wrapped,
    layer,
    samplerPacked,
    level,
  );
}

fn rcSampleMaterialAtlasAtLod(
  wrapped: vec2f,
  layer: i32,
  samplerPacked: u32,
  lod: f32,
) -> RCMaterialAtlasSampleResult {
  let finiteLod = select(0.0, lod, rcMaterialAtlasFiniteF32(lod));
  let mipFilter = (samplerPacked >> 8u) & 0x3u;
  if (
    samplerPacked > 4095u ||
    (samplerPacked & 0x3u) == 3u ||
    ((samplerPacked >> 2u) & 0x3u) == 3u ||
    mipFilter == 3u
  ) {
    return rcMaterialAtlasInvalidSample();
  }
  let address = rcMaterialAtlasLayerAddress(layer);
  if (address.valid == 0u) { return rcMaterialAtlasInvalidSample(); }
  let lastLevel = address.mipLevelCount - 1u;
  if (mipFilter == 0u || lastLevel == 0u) {
    return rcSampleMaterialAtlasLevel(
      wrapped,
      layer,
      samplerPacked,
      0u,
      finiteLod,
    );
  }
  let clampedLod = clamp(finiteLod, 0.0, f32(lastLevel));
  if (mipFilter == 1u) {
    let level = min(u32(floor(clampedLod + 0.5)), lastLevel);
    return rcSampleMaterialAtlasLevel(
      wrapped,
      layer,
      samplerPacked,
      level,
      finiteLod,
    );
  }
  let level0 = min(u32(floor(clampedLod)), lastLevel);
  let level1 = min(level0 + 1u, lastLevel);
  let c0 = rcSampleMaterialAtlasLevel(
    wrapped,
    layer,
    samplerPacked,
    level0,
    finiteLod,
  );
  let c1 = rcSampleMaterialAtlasLevel(
    wrapped,
    layer,
    samplerPacked,
    level1,
    finiteLod,
  );
  if (c0.valid == 0u || c1.valid == 0u) {
    return rcMaterialAtlasInvalidSample();
  }
  return rcMaterialAtlasValidSample(
    mix(c0.value, c1.value, clampedLod - floor(clampedLod)),
    c0.encoding,
  );
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

fn rcEmitterParentBarycentricFromLocal(
  localBary: vec3f,
  level: u32,
  ordinal: u32,
) -> vec3f {
  if (level <= 1u) {
    return localBary;
  }

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
) -> RCMaterialAtlasSampleResult {
  if (
    metaOffset >= RC_MATERIAL_MAP_META_TEXELS_PER_TRI - 1u ||
    !rcMaterialMetaAvailable(triIndex, metaOffset) ||
    !rcMaterialMetaAvailable(triIndex, metaOffset + 1u)
  ) {
    return rcMaterialAtlasInvalidSample();
  }
  let meta0 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(triIndex, metaOffset), 0);
  if (
    !rcMaterialAtlasFiniteF32(meta0.x) ||
    meta0.x < 0.0 ||
    meta0.x > 16777215.0 ||
    floor(meta0.x) != meta0.x
  ) {
    return rcMaterialAtlasInvalidSample();
  }
  let layer = i32(meta0.x);
  let address = rcMaterialAtlasLayerAddress(layer);
  if (address.valid == 0u) { return rcMaterialAtlasInvalidSample(); }
  if (
    !rcMaterialAtlasFiniteF32(meta0.y) ||
    meta0.y < 0.0 ||
    meta0.y > 4095.0 ||
    floor(meta0.y) != meta0.y
  ) {
    return rcMaterialAtlasInvalidSample();
  }
  let wrapPacked = u32(meta0.y);
  if (
    (wrapPacked & 0x3u) == 3u ||
    ((wrapPacked >> 2u) & 0x3u) == 3u ||
    ((wrapPacked >> 8u) & 0x3u) == 3u
  ) {
    return rcMaterialAtlasInvalidSample();
  }
  let texCoord = (wrapPacked >> 4u) & 0xFu;
  let uv = materialResolveUv(triIndex, texCoord, uv0, uv1);
  if (!rcMaterialAtlasFiniteVec2(uv)) {
    return rcMaterialAtlasInvalidSample();
  }
  let meta1 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(triIndex, metaOffset + 1u), 0);
  if (
    !rcMaterialAtlasFiniteVec4(meta0) ||
    !rcMaterialAtlasFiniteVec4(meta1) ||
    !rcMaterialAtlasFiniteVec2(transformedDelta)
  ) {
    return rcMaterialAtlasInvalidSample();
  }
  let scaled = uv * meta1.xy;
  let transformedCandidate = vec2f(
    scaled.x * meta1.z - scaled.y * meta1.w,
    scaled.x * meta1.w + scaled.y * meta1.z,
  ) + meta0.zw + transformedDelta;
  if (!rcMaterialAtlasFiniteVec2(transformedCandidate)) {
    return rcMaterialAtlasInvalidSample();
  }
  let transformed = transformedCandidate;
  let wrapped = rcWrapMaterialUv(transformed, wrapPacked);
  if (!rcMaterialAtlasFiniteVec2(wrapped)) {
    return rcMaterialAtlasInvalidSample();
  }
  // Probe rays have no screen-space derivatives. Use the logical source
  // footprint per angular probe-ray sample as the bounded minification model;
  // authored mip/nearest/linear policy still controls the actual lookup.
  let logicalSize = vec2f(f32(address.width), f32(address.height));
  let angularSamples = sqrt(f32(max(rc_u.raysPerProbe, 1u)));
  let footprintCandidate = abs(meta1.xy) * logicalSize / angularSamples;
  let footprint = select(
    vec2f(1.0),
    footprintCandidate,
    rcMaterialAtlasFiniteVec2(footprintCandidate),
  );
  let lodCandidate = log2(max(max(footprint.x, footprint.y), 1e-8));
  let lod = select(0.0, lodCandidate, rcMaterialAtlasFiniteF32(lodCandidate));
  return rcSampleMaterialAtlasAtLod(
    wrapped,
    layer,
    wrapPacked,
    lod,
  );
}

fn rcSampleMaterialAtlasRawAtOffset(triIndex: u32, metaOffset: u32, uv0: vec2f, uv1: vec2f) -> RCMaterialAtlasSampleResult {
  return rcSampleMaterialAtlasRawAtOffsetDelta(triIndex, metaOffset, uv0, uv1, vec2f(0.0));
}

fn rcSampleMaterialAtlasRaw(triIndex: u32, slot: u32, uv0: vec2f, uv1: vec2f) -> RCMaterialAtlasSampleResult {
  if (slot > (RC_MATERIAL_MAP_META_TEXELS_PER_TRI - 2u) / 2u) {
    return rcMaterialAtlasInvalidSample();
  }
  return rcSampleMaterialAtlasRawAtOffset(triIndex, slot * 2u, uv0, uv1);
}

fn rcMaterialAtlasFiniteNonNegativeRadianceOrBlack(value: vec3f) -> vec3f {
  let maxFiniteF32 = bitcast<f32>(0x7f7fffffu);
  let valid =
    all(value == value) &&
    all(abs(value) <= vec3f(maxFiniteF32)) &&
    all(value >= vec3f(0.0));
  return select(vec3f(0.0), value, valid);
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
  if (texel.valid == 0u) { return scalarEmission; }
  return rcMaterialAtlasFiniteNonNegativeRadianceOrBlack(
    scalarEmission * texel.value.rgb,
  );
}

fn rcSampleLightMapIrradiance(hit: IntersectionResult) -> vec3f {
  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return vec3f(0.0);
  }
  let texel = rcSampleMaterialAtlasRawAtOffset(
    hit.indices.w,
    RC_MATERIAL_MAP_LIGHT_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
  );
  if (texel.valid == 0u) { return vec3f(0.0); }
  let intensity = rcMaterialMetaLoadOrZero(
    hit.indices.w,
    RC_MATERIAL_MAP_LIGHT_INTENSITY_TEXEL_OFFSET,
  ).x;
  return rcMaterialAtlasFiniteNonNegativeRadianceOrBlack(
    max(texel.value.rgb, vec3f(0.0)) * max(intensity, 0.0),
  );
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
  if (texel.valid == 0u) {
    return fallback;
  }
  return clamp(fallback * rcMaterialMapChannel(texel.value, channel), 0.0, 1.0);
}

fn rcSampleSpecularMeta(triIndex: u32) -> vec4f {
  var color = vec3f(0.04);
  var intensity = 1.0;
  if (rcMaterialMetaAvailable(triIndex, RC_MATERIAL_MAP_SPECULAR_TEXEL_OFFSET)) {
    let spec = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_SPECULAR_TEXEL_OFFSET);
    color = max(spec.rgb, vec3f(0.0));
    intensity = clamp(spec.a, 0.0, 1.0);
  }
  return vec4f(color, intensity);
}

fn rcSampleSpecularControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  let scalar = rcSampleSpecularMeta(triIndex);
  var color = scalar.rgb;
  var intensity = scalar.a;
  let colorMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET, uv0, uv1);
  if (colorMap.valid != 0u) {
    let mappedColor =
      color * clamp(colorMap.value.rgb, vec3f(0.0), vec3f(1.0));
    if (rcFiniteVec3(mappedColor)) {
      color = max(mappedColor, vec3f(0.0));
    }
  }
  let intensityMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET, uv0, uv1);
  if (intensityMap.valid != 0u) {
    intensity = clamp(intensity * intensityMap.value.a, 0.0, 1.0);
  }
  return vec4f(color, intensity);
}

fn rcSampleClearcoatControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
  let cc = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET);
  var factor = clamp(cc.x, 0.0, 1.0);
  var roughness = clamp(cc.y, 0.0, 1.0);

  let clearcoatMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_CLEARCOAT_FACTOR_TEXEL_OFFSET, uv0, uv1);
  if (clearcoatMap.valid != 0u) {
    factor = clamp(factor * clearcoatMap.value.r, 0.0, 1.0);
  }
  let roughnessMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET, uv0, uv1);
  if (roughnessMap.valid != 0u) {
    roughness = clamp(roughness * roughnessMap.value.g, 0.0, 1.0);
  }
  return vec2f(factor, roughness);
}

fn rcSampleSheenControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  let scalars = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET);
  let colorMeta = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_SHEEN_COLOR_TEXEL_OFFSET);
  var sheenColor = clamp(colorMeta.rgb, vec3f(0.0), vec3f(1.0));
  var sheen = clamp(scalars.z, 0.0, 1.0);

  let colorMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_SHEEN_COLOR_MAP_TEXEL_OFFSET, uv0, uv1);
  if (colorMap.valid != 0u) {
    sheenColor = clamp(sheenColor * colorMap.value.rgb, vec3f(0.0), vec3f(1.0));
  }
  return vec4f(sheenColor, sheen);
}

fn rcSampleSheenRoughness(triIndex: u32, uv0: vec2f, uv1: vec2f) -> f32 {
  let scalars = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET);
  var roughness = clamp(scalars.w, 0.0, 1.0);
  let roughnessMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_SHEEN_ROUGHNESS_TEXEL_OFFSET, uv0, uv1);
  if (roughnessMap.valid != 0u) {
    roughness = clamp(roughness * roughnessMap.value.a, 0.0, 1.0);
  }
  return roughness;
}

fn rcSampleAnisotropyControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
  let scalars = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_ANISOTROPY_SCALAR_TEXEL_OFFSET);
  var strength = clamp(scalars.x, 0.0, 1.0);
  var rotation = scalars.y;

  let anisoMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET, uv0, uv1);
  if (anisoMap.valid != 0u) {
    strength = clamp(strength * anisoMap.value.b, 0.0, 1.0);
    let isSnorm =
      anisoMap.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM ||
      anisoMap.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM;
    let direction = select(
      clamp(anisoMap.value.rg, vec2f(0.0), vec2f(1.0)) * 2.0 - vec2f(1.0),
      clamp(anisoMap.value.rg, vec2f(-1.0), vec2f(1.0)),
      isSnorm,
    );
    if (dot(direction, direction) > 0.0) {
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
  if (iridescenceMap.valid != 0u) {
    factor = clamp(factor * iridescenceMap.value.r, 0.0, 1.0);
  }
  let thicknessMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET, uv0, uv1);
  if (thicknessMap.valid != 0u) {
    let thickness = mix(thicknessMin, thicknessMax, clamp(thicknessMap.value.g, 0.0, 1.0));
    thicknessMin = thickness;
    thicknessMax = thickness;
    if (thickness <= 0.0) {
      factor = 0.0;
    }
  }
  return vec4f(factor, ior, thicknessMin, thicknessMax);
}

fn rcFiniteF32(v: f32) -> bool {
  return v == v && abs(v) <= 3.402823e+38;
}

fn rcFiniteVec3(v: vec3f) -> bool {
  return all(v == v) && all(abs(v) <= vec3f(3.402823e+38));
}

fn rcCanNormalize(v: vec3f) -> bool {
  let maxComponent = max(abs(v.x), max(abs(v.y), abs(v.z)));
  if (
    !rcFiniteVec3(v) ||
    !rcFiniteF32(maxComponent) ||
    !(maxComponent > 0.0)
  ) {
    return false;
  }
  let scaled = v / maxComponent;
  let scaledLen2 = dot(scaled, scaled);
  return rcFiniteF32(scaledLen2) && scaledLen2 > 0.0;
}

fn rcSafeNormalizeOr(v: vec3f, fallback: vec3f) -> vec3f {
  if (rcCanNormalize(v)) {
    let maxComponent = max(abs(v.x), max(abs(v.y), abs(v.z)));
    let scaled = v / maxComponent;
    return scaled * inverseSqrt(dot(scaled, scaled));
  }
  if (rcCanNormalize(fallback)) {
    let fallbackMaxComponent = max(
      abs(fallback.x),
      max(abs(fallback.y), abs(fallback.z)),
    );
    let scaledFallback = fallback / fallbackMaxComponent;
    return scaledFallback * inverseSqrt(dot(scaledFallback, scaledFallback));
  }
  return vec3f(0.0, 1.0, 0.0);
}

fn rcSmoothNormalForHit(hit: IntersectionResult, fallbackNormal: vec3f) -> vec3f {
  let i0 = hit.indices.x;
  let i1 = hit.indices.y;
  let i2 = hit.indices.z;
  if (i0 >= arrayLength(&rc_geom_normal) || i1 >= arrayLength(&rc_geom_normal) || i2 >= arrayLength(&rc_geom_normal)) {
    return fallbackNormal;
  }
  let nLocalRaw =
    hit.barycoord.x * rc_geom_normal[i0].xyz +
    hit.barycoord.y * rc_geom_normal[i1].xyz +
    hit.barycoord.z * rc_geom_normal[i2].xyz;
  let nLocal = rcSafeNormalizeOr(nLocalRaw, fallbackNormal);
  var n = nLocal;
  if (rc_u.bvhMode == 1u) {
    let base = hit.instanceIndex * 4u;
    if (base + 2u >= tlasWorldToLocalColumnCount()) {
      return rcSafeNormalizeOr(fallbackNormal, vec3f(0.0, 1.0, 0.0));
    }
    n = tlasTransformNormalFromLocalCols(
      tlasLoadWorldToLocalColumn(base),
      tlasLoadWorldToLocalColumn(base + 1u),
      tlasLoadWorldToLocalColumn(base + 2u),
      nLocal,
    );
  }
  n = rcSafeNormalizeOr(n, fallbackNormal);
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

fn rcMaxAbsVec2(v: vec2f) -> f32 {
  return max(abs(v.x), abs(v.y));
}

fn rcMaxAbsVec3(v: vec3f) -> f32 {
  return max(abs(v.x), max(abs(v.y), abs(v.z)));
}

fn rcTransformDirectionCols(l2w0: vec4f, l2w1: vec4f, l2w2: vec4f, v: vec3f) -> vec3f {
  let matrixScale = max(
    rcMaxAbsVec3(l2w0.xyz),
    max(rcMaxAbsVec3(l2w1.xyz), rcMaxAbsVec3(l2w2.xyz)),
  );
  let vectorScale = rcMaxAbsVec3(v);
  if (!(matrixScale > 0.0) || !(vectorScale > 0.0)) {
    return vec3f(0.0);
  }
  let scaledV = v / vectorScale;
  return
    (l2w0.xyz / matrixScale) * scaledV.x +
    (l2w1.xyz / matrixScale) * scaledV.y +
    (l2w2.xyz / matrixScale) * scaledV.z;
}

fn rcTangentHandednessForLocalToWorld(l2w0: vec4f, l2w1: vec4f, l2w2: vec4f) -> f32 {
  let s0 = rcMaxAbsVec3(l2w0.xyz);
  let s1 = rcMaxAbsVec3(l2w1.xyz);
  let s2 = rcMaxAbsVec3(l2w2.xyz);
  if (!(s0 > 0.0) || !(s1 > 0.0) || !(s2 > 0.0)) {
    return 1.0;
  }
  let det = dot(
    l2w0.xyz / s0,
    cross(l2w1.xyz / s1, l2w2.xyz / s2),
  );
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

  if (rcCanNormalize(authoredTangent) && abs(authoredHandedness) > 0.5) {
    let isTlas = rc_u.bvhMode == 1u;
    let tBase = hit.instanceIndex * 4u;
    let tOk = isTlas && tBase + 2u < tlasLocalToWorldColumnCount();
    if (isTlas && !tOk) {
      return RCMaterialTangentFrame(fallbackTangent, fallbackBitangent);
    }
    if (tOk) {
      let t0 = tlasLoadLocalToWorldColumn(tBase);
      let t1 = tlasLoadLocalToWorldColumn(tBase + 1u);
      let t2 = tlasLoadLocalToWorldColumn(tBase + 2u);
      authoredTangent = rcTransformDirectionCols(
        t0,
        t1,
        t2,
        authoredTangent,
      );
      authoredHandedness = authoredHandedness * rcTangentHandednessForLocalToWorld(
        t0,
        t1,
        t2,
      );
    }

    authoredTangent = authoredTangent - frameNormal * dot(frameNormal, authoredTangent);
    if (rcCanNormalize(authoredTangent)) {
      tangent = rcSafeNormalizeOr(authoredTangent, fallbackTangent);
      bitangent = rcSafeNormalizeOr(
        cross(frameNormal, tangent),
        fallbackBitangent,
      ) * select(-1.0, 1.0, authoredHandedness >= 0.0);
    }
  }

  return RCMaterialTangentFrame(tangent, bitangent);
}

fn rcFallbackBitangentForNormal(n: vec3f, t: vec3f) -> vec3f {
  let b = cross(n, t);
  if (!rcCanNormalize(b)) {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.95);
    return rcSafeNormalizeOr(cross(n, up), vec3f(0.0, 0.0, 1.0));
  }
  return rcSafeNormalizeOr(b, vec3f(0.0, 0.0, 1.0));
}

fn rcMaterialTangentFrameForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  mapOffset: u32,
) -> RCMaterialTangentFrame {
  let triIndex = hit.indices.w;
  let meta0 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(triIndex, mapOffset), 0);
  var texCoord = 0u;
  if (
    rcMaterialAtlasFiniteF32(meta0.y) &&
    meta0.y >= 0.0 &&
    meta0.y <= 4095.0 &&
    floor(meta0.y) == meta0.y
  ) {
    texCoord = (u32(meta0.y) >> 4u) & 0xFu;
  }

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
  let ta = materialResolveUv(triIndex, texCoord, uv0a, uv1a);
  let tb = materialResolveUv(triIndex, texCoord, uv0b, uv1b);
  let tc = materialResolveUv(triIndex, texCoord, uv0c, uv1c);

  var dp1 = p1.xyz - p0.xyz;
  var dp2 = p2.xyz - p0.xyz;
  let positionScale = max(rcMaxAbsVec3(dp1), rcMaxAbsVec3(dp2));
  if (positionScale > 0.0) {
    dp1 = dp1 / positionScale;
    dp2 = dp2 / positionScale;
  }
  var duv1 = tb - ta;
  var duv2 = tc - ta;
  let uvScale = max(rcMaxAbsVec2(duv1), rcMaxAbsVec2(duv2));
  if (uvScale > 0.0) {
    duv1 = duv1 / uvScale;
    duv2 = duv2 / uvScale;
  }
  let det = duv1.x * duv2.y - duv1.y * duv2.x;
  var tangent = dp1;
  var bitangent = dp2;
  var desiredHandedness = 1.0;
  if (uvScale > 0.0 && abs(det) > 1e-12) {
    let invDet = 1.0 / det;
    tangent = (dp1 * duv2.y - dp2 * duv1.y) * invDet;
    bitangent = (dp2 * duv1.x - dp1 * duv2.x) * invDet;
    desiredHandedness = select(-1.0, 1.0, det >= 0.0);
  }
  if (rc_u.bvhMode == 1u) {
    let base = hit.instanceIndex * 4u;
    if (base + 2u < tlasLocalToWorldColumnCount()) {
      let l2w0 = tlasLoadLocalToWorldColumn(base);
      let l2w1 = tlasLoadLocalToWorldColumn(base + 1u);
      let l2w2 = tlasLoadLocalToWorldColumn(base + 2u);
      tangent = rcTransformDirectionCols(l2w0, l2w1, l2w2, tangent);
      bitangent = rcTransformDirectionCols(l2w0, l2w1, l2w2, bitangent);
      desiredHandedness =
        desiredHandedness *
        rcTangentHandednessForLocalToWorld(l2w0, l2w1, l2w2);
    } else {
      tangent = vec3f(0.0);
      bitangent = vec3f(0.0);
    }
  }

  tangent = tangent - frameNormal * dot(frameNormal, tangent);
  if (!rcCanNormalize(tangent)) {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(frameNormal.y) > 0.95);
    tangent = rcSafeNormalizeOr(cross(up, frameNormal), vec3f(1.0, 0.0, 0.0));
  } else {
    tangent = rcSafeNormalizeOr(tangent, vec3f(1.0, 0.0, 0.0));
  }

  bitangent = bitangent - frameNormal * dot(frameNormal, bitangent) - tangent * dot(tangent, bitangent);
  if (!rcCanNormalize(bitangent)) {
    bitangent = rcFallbackBitangentForNormal(frameNormal, tangent) * desiredHandedness;
  } else {
    bitangent = rcSafeNormalizeOr(
      bitangent,
      rcFallbackBitangentForNormal(frameNormal, tangent),
    );
    if (dot(cross(tangent, bitangent), frameNormal) * desiredHandedness < 0.0) {
      bitangent = -bitangent;
    }
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
  if (
    !rcMaterialMetaAvailable(triIndex, normalMapOffset) ||
    !rcMaterialMetaAvailable(triIndex, normalMapOffset + 1u)
  ) {
    return fallbackNormal;
  }
  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return fallbackNormal;
  }
  let texelColor = rcSampleMaterialAtlasRawAtOffset(triIndex, normalMapOffset, uvs.uv0, uvs.uv1);
  if (texelColor.valid == 0u) { return fallbackNormal; }

  let scaleMeta = textureLoad(
    rc_materialMapMeta,
      rcMaterialMetaCoord(triIndex, normalScaleOffset),
    0,
  );
  if (!rcFiniteVec3(texelColor.value.rgb) || !rcFiniteF32(scaleMeta.x)) {
    return fallbackNormal;
  }
  let normalScale = max(scaleMeta.x, 0.0);
  let isSnorm =
    texelColor.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM ||
    texelColor.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM;
  let decodedNormal = select(
    clamp(texelColor.value.rgb, vec3f(0.0), vec3f(1.0)) * 2.0 -
      vec3f(1.0),
    clamp(texelColor.value.rgb, vec3f(-1.0), vec3f(1.0)),
    isSnorm,
  );
  let tangentSampleRaw = vec3f(
    decodedNormal.x * normalScale,
    decodedNormal.y * normalScale,
    decodedNormal.z,
  );
  if (!rcCanNormalize(tangentSampleRaw)) {
    return fallbackNormal;
  }
  let tangentSample = rcSafeNormalizeOr(
    tangentSampleRaw,
    vec3f(0.0, 0.0, 1.0),
  );

  let frame = rcMaterialTangentFrameForHit(hit, frameNormal, normalMapOffset);
  let perturbedRaw = frame.tangent * tangentSample.x + frame.bitangent * tangentSample.y + frameNormal * tangentSample.z;
  let perturbed = rcSafeNormalizeOr(perturbedRaw, fallbackNormal);
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
  if (
    !rcMaterialMetaAvailable(triIndex, RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET) ||
    !rcMaterialMetaAvailable(triIndex, RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET + 1u)
  ) {
    return shadingNormal;
  }
  let scaleMeta = textureLoad(
    rc_materialMapMeta,
      rcMaterialMetaCoord(triIndex, RC_MATERIAL_MAP_BUMP_SCALE_TEXEL_OFFSET),
    0,
  );
  let bumpScale = scaleMeta.x;
  if (
    !rcMaterialAtlasFiniteVec4(scaleMeta) ||
    abs(bumpScale) < 1e-8
  ) {
    return shadingNormal;
  }

  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return shadingNormal;
  }
  let hC = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET, uvs.uv0, uvs.uv1);
  if (hC.valid == 0u) { return shadingNormal; }

  let bumpMeta = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaCoord(triIndex, RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET),
    0,
  );
  if (
    !rcMaterialAtlasFiniteF32(bumpMeta.x) ||
    bumpMeta.x < 0.0 ||
    bumpMeta.x > 16777215.0 ||
    floor(bumpMeta.x) != bumpMeta.x
  ) {
    return shadingNormal;
  }
  let bumpAddress = rcMaterialAtlasLayerAddress(i32(bumpMeta.x));
  if (bumpAddress.valid == 0u) {
    return shadingNormal;
  }
  let logicalTexelStep = vec2f(
    1.0 / f32(max(bumpAddress.width, 1u)),
    1.0 / f32(max(bumpAddress.height, 1u)),
  );
  let bumpTexelStep = vec2f(
    1.0 / max(scaleMeta.y, 1.0),
    1.0 / max(scaleMeta.z, 1.0),
  );
  let authoredDimensionsValid =
    scaleMeta.y >= 1.0 &&
    scaleMeta.z >= 1.0 &&
    scaleMeta.y <= 16777216.0 &&
    scaleMeta.z <= 16777216.0 &&
    floor(scaleMeta.y) == scaleMeta.y &&
    floor(scaleMeta.z) == scaleMeta.z;
  let texelStep = select(
    logicalTexelStep,
    bumpTexelStep,
    authoredDimensionsValid,
  );
  let hUSample = rcSampleMaterialAtlasRawAtOffsetDelta(
    triIndex,
    RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
    vec2f(texelStep.x, 0.0),
  );
  let hVSample = rcSampleMaterialAtlasRawAtOffsetDelta(
    triIndex,
    RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
    vec2f(0.0, texelStep.y),
  );
  if (hUSample.valid == 0u || hVSample.valid == 0u) {
    return shadingNormal;
  }
  if (
    !rcMaterialAtlasFiniteVec2(texelStep) ||
    !all(texelStep > vec2f(0.0))
  ) {
    return shadingNormal;
  }
  let heightMagnitude = max(
    1.0,
    max(
      abs(hC.value.r),
      max(abs(hUSample.value.r), abs(hVSample.value.r)),
    ),
  );
  let dhdu =
    ((hUSample.value.r / heightMagnitude) -
      (hC.value.r / heightMagnitude)) *
    heightMagnitude / texelStep.x;
  let dhdv =
    ((hVSample.value.r / heightMagnitude) -
      (hC.value.r / heightMagnitude)) *
    heightMagnitude / texelStep.y;
  if (!rcFiniteF32(dhdu) || !rcFiniteF32(dhdv)) {
    return shadingNormal;
  }
  let frame = rcMaterialTangentFrameForHit(hit, shadingNormal, RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET);
  let perturbed = shadingNormal - bumpScale * (dhdu * frame.tangent + dhdv * frame.bitangent);
  let n = rcSafeNormalizeOr(perturbed, shadingNormal);
  return select(-n, n, dot(n, shadingNormal) >= 0.0);
}

fn rcSampleFaceLayerControls(triIndex: u32, isFrontFace: bool) -> vec4f {
  let offset = select(
    RC_MATERIAL_MAP_BACK_LAYER_TEXEL_OFFSET,
    RC_MATERIAL_MAP_FRONT_LAYER_TEXEL_OFFSET,
    isFrontFace,
  );
  if (!rcMaterialMetaAvailable(triIndex, offset)) {
    // Optional raw-dispatch metadata must preserve the no-layer identity:
    // unit transmission and the negative roughness sentinel. A zero fallback
    // would turn every valid opaque receiver black.
    return vec4f(1.0, 1.0, 1.0, -1.0);
  }
  return select(
    rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_BACK_LAYER_TEXEL_OFFSET),
    rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_FRONT_LAYER_TEXEL_OFFSET),
    isFrontFace,
  );
}

fn rcSampleVolumeScatteringControls(triIndex: u32) -> vec4f {
  let scatter = rcMaterialMetaLoadOrZero(
    triIndex, RC_MATERIAL_MAP_VOLUME_SCATTERING_TEXEL_OFFSET,
  );
  return vec4f(max(scatter.rgb, vec3f(0.0)), clamp(scatter.a, -0.99, 0.99));
}

fn rcHomogeneousBeerTransmittanceRgb(sigmaT: vec3f, distance: f32) -> vec3f {
  return exp(-max(sigmaT, vec3f(0.0)) * max(distance, 0.0));
}

fn rcHenyeyGreensteinPhase(cosTheta: f32, g: f32) -> f32 {
  let anisotropy = clamp(g, -0.99, 0.99);
  let denominator = 1.0 + anisotropy * anisotropy -
    2.0 * anisotropy * clamp(cosTheta, -1.0, 1.0);
  return (1.0 - anisotropy * anisotropy) /
    (4.0 * RC_PI * denominator * sqrt(denominator));
}

fn rcApplyHomogeneousVolumeSingleScatter(
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
  // The incoming radiance is already directionally aggregated. There is no
  // incident direction left with which to evaluate anisotropic HG, so use its
  // isotropic angular average rather than inventing one from the surface normal.
  let phase = rcHenyeyGreensteinPhase(0.0, 0.0);
  let source = dot(max(radiance, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722)) *
    max(albedo, vec3f(0.0)) * phase;
  let projectedCosine = abs(dot(n, v));
  if (projectedCosine <= 0.0) { return source; }
  let distance = pathLength / projectedCosine;
  let transmittance = rcHomogeneousBeerTransmittanceRgb(sigmaS, distance);
  return radiance * transmittance + source * (vec3f(1.0) - transmittance);
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
  // Exact event lobes apply absolute film R/T themselves.
  dielectricLayerTransmission: vec3f,
  // Direct reflection pays face absorption but never film T.
  reflectionLayerTransmission: vec3f,
  // Base/source closures pay face absorption and film T.
  layerTransmission: vec3f,
  volumeScattering: vec4f,
  transmission: f32,
  opticalIor: vec3f,
  bulkThickness: f32,
  thicknessMapScale: f32,
}

fn rcSampleProbeHitMaterial(
  hit: IntersectionResult,
  scalarBaseColor: vec3f,
  scalarRoughness: f32,
  scalarMetalness: f32,
  frameNormal: vec3f,
  shadingNormal: vec3f,
  scalarTransmission: f32,
  scalarIor: f32,
  viewDirection: vec3f,
) -> RCProbeHitMaterial {
  var out: RCProbeHitMaterial;
  let vertexColor = rcSampleVertexColorForHit(hit);
  out.albedo = scalarBaseColor * vertexColor.rgb;
  out.roughness = scalarRoughness;
  out.metalness = scalarMetalness;
  out.specular = rcSampleSpecularMeta(hit.indices.w);
  out.clearcoat = vec2f(0.0);
  out.clearcoatNormal = shadingNormal;
  out.sheen = vec4f(0.0);
  out.sheenRoughness = 0.0;
  out.anisotropy = vec2f(0.0);
  let defaultAnisotropyFrame = rcMaterialTangentFrameForHit(hit, shadingNormal, RC_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET);
  out.anisotropyTangent = defaultAnisotropyFrame.tangent;
  out.anisotropyBitangent = defaultAnisotropyFrame.bitangent;
  out.iridescence = vec4f(0.0, 1.0, 0.0, 0.0);
  out.dielectricLayerTransmission = vec3f(1.0);
  out.reflectionLayerTransmission = vec3f(1.0);
  out.layerTransmission = vec3f(1.0);
  out.volumeScattering = vec4f(0.0);
  out.transmission = scalarTransmission;
  let transportIor = select(max(scalarIor, 1.0), 1e6, scalarIor == 0.0);
  out.opticalIor = materialDispersionIorRgb(hit.indices.w, transportIor);
  out.bulkThickness = materialOpticalThickness(hit.indices.w);
  out.thicknessMapScale = 1.0;
  out.clearcoatNormal = rcApplyClearcoatNormalMapForHit(hit, frameNormal, shadingNormal);

  let layerControls = rcSampleFaceLayerControls(hit.indices.w, hit.side >= 0.0);
  out.roughness = select(out.roughness, clamp(layerControls.a, 0.0, 1.0), layerControls.a >= 0.0);
  out.dielectricLayerTransmission = clamp(
    layerControls.rgb, vec3f(0.0), vec3f(1.0),
  );
  out.reflectionLayerTransmission = out.dielectricLayerTransmission;
  out.layerTransmission = out.reflectionLayerTransmission;
  out.volumeScattering = rcSampleVolumeScatteringControls(hit.indices.w);
  let film = materialThinFilmResponse(
    hit.indices.w,
    hit.side >= 0.0,
    abs(dot(shadingNormal, rcSafeNormalizeOr(viewDirection, shadingNormal))),
  );
  if (film.present != 0u) {
    out.specular = vec4f(film.reflectance, 1.0);
    out.iridescence = vec4f(0.0);
    out.layerTransmission = out.layerTransmission * film.transmittance;
  }

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
  if (baseColorTexel.valid != 0u) {
    out.albedo = out.albedo * baseColorTexel.value.rgb;
  }
  out.roughness = rcSampleMaterialScalarMap(
    hit.indices.w,
    RC_MATERIAL_MAP_SLOT_ROUGHNESS,
    1u,
    uvs.uv0,
    uvs.uv1,
    scalarRoughness,
  );
  out.roughness = select(
    out.roughness,
    clamp(layerControls.a, 0.0, 1.0),
    layerControls.a >= 0.0,
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
  let transmissionMap = rcSampleMaterialAtlasRawAtOffset(
    hit.indices.w, RC_MATERIAL_MAP_TRANSMISSION_TEXEL_OFFSET, uvs.uv0, uvs.uv1,
  );
  if (transmissionMap.valid != 0u) {
    out.transmission = clamp(out.transmission * transmissionMap.value.r, 0.0, 1.0);
  }
  let thicknessMap = rcSampleMaterialAtlasRawAtOffset(
    hit.indices.w, RC_MATERIAL_MAP_THICKNESS_TEXEL_OFFSET, uvs.uv0, uvs.uv1,
  );
  if (thicknessMap.valid != 0u) {
    // The packed optical-header sign distinguishes a positive authored
    // thickness cap from a synthetic reference distance for zero-thickness
    // bulk. Only the former admits texture scaling; synthetic bulk always uses
    // the closed-geometry segment length.
    out.thicknessMapScale = materialOpticalThicknessMapScale(
      hit.indices.w,
      thicknessMap.value.g,
    );
  }
  // Mapped specular/iridescence values are substrate controls; a full authored
  // thin-film stack is the outermost optical layer and therefore overrides
  // them after texture sampling.
  let mappedFilm = materialThinFilmResponse(
    hit.indices.w,
    hit.side >= 0.0,
    abs(dot(shadingNormal, rcSafeNormalizeOr(viewDirection, shadingNormal))),
  );
  if (mappedFilm.present != 0u) {
    out.specular = vec4f(mappedFilm.reflectance, 1.0);
    out.iridescence = vec4f(0.0);
  }
  return out;
}`;
