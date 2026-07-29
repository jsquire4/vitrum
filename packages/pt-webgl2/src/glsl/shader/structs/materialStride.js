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
//   129 = {front UV attribute layer, back UV attribute layer, _, _}
// 130..135 = dense attribute-layer selectors for the 21 base map slots
//             (four selectors per texel, MATERIAL_MAP_FIELD_ORDER order).
export const MATERIAL_WRAP_TEXEL_OFFSET = 100;

// ── Named absolute texel offsets (single source of truth for the packer AND the
// GLSL fetch sites). The packer (scene/materialsTexture.ts) previously hardcoded
// these as bare literals (55, 57, …, 98) — the exact stride-drift surface this
// module exists to eliminate. Each texture-transform occupies 2 texels (mat3
// rows). Ordering matches the GLSL `readTextureTransform` calls in
// material_mapped_rich.glsl.ts (firstTextureTransformIdx + 2k).

/** First texture-transform texel — the fork's `firstTextureTransformIdx`. */
export const MATERIAL_FIRST_TRANSFORM_TEXEL = 55;
/** Texels per texture-transform (mat3 packed as 2 rgba texels). */
export const MATERIAL_TRANSFORM_TEXELS = 2;

// Core 15 texture-transform slots (55..84), in GLSL `readTextureTransform` order.
// texel = MATERIAL_FIRST_TRANSFORM_TEXEL + MATERIAL_TRANSFORM_TEXELS * k
export const MATERIAL_TRANSFORM_TEXEL = /** @type {Record<string, number>} */ ({
  baseColorMap: 55,
  metallicMap: 57,
  roughnessMap: 59,
  transmissionMap: 61,
  emissiveMap: 63,
  normalMap: 65,
  clearcoatMap: 67,
  clearcoatNormalMap: 69,
  clearcoatRoughnessMap: 71,
  sheenColorMap: 73,
  sheenRoughnessMap: 75,
  iridescenceMap: 77,
  iridescenceThicknessMap: 79,
  specularColorMap: 81,
  specularIntensityMap: 83,
});

// D3 auxiliary block.
/** ao/light/bump map ids + scalars + envMapIntensity (texels 85/86). */
export const MATERIAL_D3_AUX_TEXEL = 85;
/** ao/light/bump transforms (texels 87/89/91, 2 texels each). */
export const MATERIAL_AO_TRANSFORM_TEXEL = 87;
export const MATERIAL_LIGHTMAP_TRANSFORM_TEXEL = 89;
export const MATERIAL_BUMP_TRANSFORM_TEXEL = 91;
/** alphaMap transform (texels 93/94). */
export const MATERIAL_ALPHA_TRANSFORM_TEXEL = 93;
/** anisotropyMap transform (texels 95/96). */
export const MATERIAL_ANISOTROPY_TRANSFORM_TEXEL = 95;
/** Volume thickness payload (texel 97). */
export const MATERIAL_VOLUME_THICKNESS_TEXEL = 97;
/** thicknessMap transform (texels 98/99). */
export const MATERIAL_THICKNESS_TRANSFORM_TEXEL = 98;

// Map order shared by sampler-policy and arbitrary UV-layer selector payloads.
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
export const MATERIAL_UV_SELECTOR_TEXEL_OFFSET = MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + MATERIAL_LAYER_NORMAL_TEXELS;
export const MATERIAL_UV_SELECTOR_TEXELS = Math.ceil(MATERIAL_MAP_FIELD_ORDER.length / 4);
export const MATERIAL_PIXELS = MATERIAL_UV_SELECTOR_TEXEL_OFFSET + MATERIAL_UV_SELECTOR_TEXELS;
