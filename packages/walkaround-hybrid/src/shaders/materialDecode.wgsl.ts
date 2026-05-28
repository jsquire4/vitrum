/**
 * Material-bit decoders for the RGB888 + (trans4|texType4) payload packed
 * into `bvhIndex[triIdx].w`.
 *
 * Split out of common.wgsl.ts (T9-stepA): `decodeMaterialColor`,
 * `decodeSurfaceTextureId`, and `decodeIsMetal`.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const MATERIAL_DECODE_WGSL = /* wgsl */ `// Decode RGB888 + (trans4|texType4) packed material data from bvhIndex[triIdx].w.
// Returns vec4f(r, g, b, transmission) in [0, 1].  The texture-type id is
// retrieved separately via decodeSurfaceTextureId.
fn decodeMaterialColor(packed: u32) -> vec4f {
  let r = f32((packed >> 24u) & 0xFFu) / 255.0;
  let g = f32((packed >> 16u) & 0xFFu) / 255.0;
  let b = f32((packed >>  8u) & 0xFFu) / 255.0;
  // Transmission is a 4-bit unorm in bits [7:4] of the low byte.
  let t = f32((packed >> 4u) & 0xFu) / 15.0;
  return vec4f(r, g, b, t);
}

// Decode the authored surface-texture id from bvhIndex[triIdx].w.
// Uses only 3 bits (bits 0-2) — bit 3 of the low nybble is isMetal.
//   0=smooth 1=hammered 2=ripple 3=granite
//   4=baroque 5=waterglass 6=catspaw 7=flemish
fn decodeSurfaceTextureId(packed: u32) -> u32 {
  return packed & 0x7u;
}

// Decode the isMetal flag — true for came / solder / metallic surfaces
// (metalness > 0.5 in the source material). Used to skip the noisy
// Lo_direct ReSTIR DI sampling on thin metallic geometry where the
// single-sample variance produces visible firefly speckle that atrous
// can't smooth across the thin came strips.
fn decodeIsMetal(packed: u32) -> bool {
  return ((packed >> 3u) & 0x1u) != 0u;
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const MATERIAL_DECODE_MODULE: WgslModule = {
  name: "materialDecode",
  source: MATERIAL_DECODE_WGSL,
  requires: [],
};
