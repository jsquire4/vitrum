// Single source of truth for the materials-texture stride (RGBA32F texels per
// material). Imported by BOTH the TS packer (scene/materialsTexture.ts) and every
// GLSL chunk that fetches material texels. Packer↔shader stride drift is the
// repo's recurring upload-gap bug class (H1 / H41 / the 2026-06-10 D3 85-vs-93
// working-tree break) — never hardcode this number inside a shader string again;
// interpolate `${MATERIAL_PIXELS}u`.
//
// Layout: texels 0..54 = fork data layout, 55..84 = 15 texture transforms
// (2 texels each), 85/86 = D3 ao/light/bump map ids + scalars + envMapIntensity,
// 87..92 = D3 ao/light/bump transforms (2 texels each).
export const MATERIAL_PIXELS = 93;
