/**
 * Canonical WGSL declaration of the per-material entry struct (W2-C5).
 *
 * Matches the byte layout produced by `packMaterials` in
 * `@vitrum/shared-bvh/src/materialEntry.ts`. See that file's docstring for
 * the full byte map. Briefly:
 *
 *   slot 0..2  → baseColor.xyz
 *   slot 3     → roughness
 *   slot 4..6  → emissive.xyz
 *   slot 7     → metalness
 *   slot 8     → ior
 *   slot 9     → transmission
 *   slot 10    → attenuationDistance
 *   slot 11    → thickness
 *   slot 12..14→ attenuationColor.xyz
 *   slot 15    → flags (u32; bit 0 = isGlass, bit 1 = castShadow disabled)
 *
 * Std140 alignment: every `vec3<f32>` field is followed by an `f32` so the
 * subsequent field starts on a 16-byte boundary. The struct is 64 bytes
 * total and is layout-compatible with both `uniform var<...>` and
 * `storage var<storage, ..., read>` bindings.
 *
 * Field-name contract: consumers that previously declared their own struct
 * (DDGI's `DDGIMaterial`, RC's `MaterialEntry` flat-struct) now import this
 * module's `MATERIAL_ENTRY_WGSL` string and address fields by these names:
 *   mat.baseColor, mat.roughness, mat.emissive, mat.metalness, mat.ior,
 *   mat.transmission, mat.attenuationDistance, mat.thickness,
 *   mat.attenuationColor, mat.flags.
 *
 * @since W2-C5 (premium-grade-refactor-20260517.md §W2 sub-task 5).
 */

export const MATERIAL_ENTRY_WGSL = /* wgsl */ `

// Canonical MaterialEntry — 64 bytes / 16 f32 lanes (W2-C5).
// CPU packer: packMaterials() in @vitrum/shared-bvh/materialEntry.ts.
struct MaterialEntry {
  baseColor:           vec3<f32>,
  roughness:           f32,
  emissive:            vec3<f32>,
  metalness:           f32,
  ior:                 f32,
  transmission:        f32,
  attenuationDistance: f32,
  thickness:           f32,
  attenuationColor:    vec3<f32>,
  flags:               u32,  // bit 0 = isGlass, bit 1 = castShadow disabled
};

// Canonical flag bits.
const MATERIAL_FLAG_IS_GLASS: u32 = 1u;
const MATERIAL_FLAG_CAST_SHADOW_DISABLED: u32 = 2u;

`;
