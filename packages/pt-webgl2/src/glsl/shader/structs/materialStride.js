// Single source of truth for the materials-texture stride (RGBA32F texels per
// material). Imported by BOTH the TS packer (scene/materialsTexture.ts) and every
// GLSL chunk that fetches material texels. Packer↔shader stride drift is the
// repo's recurring upload-gap bug class (H1 / H41 / the 2026-06-10 D3 85-vs-93
// working-tree break) — never hardcode this number inside a shader string again;
// interpolate `${MATERIAL_PIXELS}u`.
//
// Layout: texels 0..54 = fork data layout, 55..84 = 15 texture transforms
// (2 texels each), 85/86 = D3 ao/light/bump map ids + scalars + envMapIntensity,
// 87..92 = D3 ao/light/bump transforms (2 texels each), 93..94 = alphaMap
// transform.
export const MATERIAL_PIXELS = 95;

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
//   bit 9  = clearcoatNormalMap
export const UV_SET_BIT = /** @type {Record<string, number>} */ ({
  baseColorMap:              1 << 0,
  metallicMap:               1 << 1,
  roughnessMap:              1 << 2,
  transmissionMap:           1 << 3,
  emissiveMap:               1 << 4,
  normalMap:                 1 << 5,
  alphaMap:                  1 << 6,
  clearcoatMap:              1 << 7,
  clearcoatRoughnessMap:     1 << 8,
  clearcoatNormalMap:        1 << 9,
  sheenColorMap:             1 << 10,
  sheenRoughnessMap:         1 << 11,
  iridescenceMap:            1 << 12,
  iridescenceThicknessMap:   1 << 13,
  specularColorMap:          1 << 14,
  specularIntensityMap:      1 << 15,
  aoMap:                     1 << 16,
  lightMap:                  1 << 17,
  bumpMap:                   1 << 18,
});
