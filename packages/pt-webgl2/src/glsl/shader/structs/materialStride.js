// Single source of truth for the materials-texture stride (RGBA32F texels per
// material). Imported by BOTH the TS packer (scene/materialsTexture.ts) and every
// GLSL chunk that fetches material texels. Packer↔shader stride drift is the
// repo's recurring upload-gap bug class (H1 / H41 / the 2026-06-10 D3 85-vs-93
// working-tree break) — never hardcode this number inside a shader string again;
// interpolate `${MATERIAL_PIXELS}u`.
//
// Layout: texels 0..54 = fork data layout with reserved lanes now consumed by
// scalar anisotropy (s6.b = anisotropyMap, s11.a = strength, s17.b = rotation),
// 55..84 = 15 texture transforms (2 texels each), 85/86 = D3 ao/light/bump map
// ids + scalars + envMapIntensity, 87..92 = D3 ao/light/bump transforms
// (2 texels each), 93..94 = alphaMap transform, 95..96 = anisotropyMap transform,
// 97 = volume thickness payload, 98..99 = thicknessMap transform,
// 100..120 = per-map sampler policy (wrapS, wrapT, mipFilter, packed mag/min),
// 121 = Jakob-Hanika spectral reflectance coefficients + validity flag,
// 122..129 = front/back layer normal payload:
//   122 = {frontLayer.normalMap, frontLayer.normalScale,
//          backLayer.normalMap, backLayer.normalScale}
//   123..124 = frontLayer.normalMap transform
//   125..126 = backLayer.normalMap transform
//   127 = frontLayer.normalMap sampler policy
//   128 = backLayer.normalMap sampler policy
//   129 = {front texCoord, back texCoord, _, _}
export const MATERIAL_WRAP_TEXEL_OFFSET = 100;

// Map order shared by the UV-set bitmask and the sampler-policy payload. Bit k in
// UV_SET_BIT and texel k in the sampler texels describe the same MaterialSpec map.
export const MATERIAL_MAP_FIELD_ORDER = /** @type {readonly string[]} */ ([
  'baseColorMap',
  'metallicMap',
  'roughnessMap',
  'transmissionMap',
  'emissiveMap',
  'normalMap',
  'alphaMap',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'specularColorMap',
  'specularIntensityMap',
  'aoMap',
  'lightMap',
  'bumpMap',
  'anisotropyMap',
  'thicknessMap',
]);

export const MATERIAL_WRAP_TEXELS = MATERIAL_MAP_FIELD_ORDER.length;
export const MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET = MATERIAL_WRAP_TEXEL_OFFSET + MATERIAL_WRAP_TEXELS;
export const MATERIAL_LAYER_NORMAL_TEXEL_OFFSET = MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET + 1;
export const MATERIAL_LAYER_NORMAL_TEXELS = 8;
export const MATERIAL_PIXELS = MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + MATERIAL_LAYER_NORMAL_TEXELS;

// UV-set bitmask — packed at texel 86.a (the former pad lane).
// Bit k set = the k-th map samples uv1 (ATTR_UV1) instead of uv0 (ATTR_UV).
// Map→bit assignments (must match the packer in materialsTexture.ts AND the
// GLSL decoder in material_struct.glsl.js — single source here):
//   bit 0  = baseColorMap          bit 10 = sheenColorMap
//   bit 1  = metallicMap           bit 11 = sheenRoughnessMap
//   bit 2  = roughnessMap          bit 12 = iridescenceMap
//   bit 3  = transmissionMap       bit 13 = iridescenceThicknessMap
//   bit 4  = emissiveMap           bit 14 = specularColorMap
//   bit 5  = normalMap             bit 15 = specularIntensityMap
//   bit 6  = alphaMap              bit 16 = aoMap
//   bit 7  = clearcoatMap          bit 17 = lightMap
//   bit 8  = clearcoatRoughnessMap bit 18 = bumpMap
//   bit 9  = clearcoatNormalMap    bit 19 = anisotropyMap
//                                     bit 20 = thicknessMap
export const UV_SET_BIT = /** @type {Record<string, number>} */ (
  Object.fromEntries(MATERIAL_MAP_FIELD_ORDER.map((field, i) => [field, 1 << i]))
);
