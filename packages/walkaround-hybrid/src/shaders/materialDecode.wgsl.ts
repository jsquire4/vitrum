/**
 * Material-bit decoders for the RGB888 + (trans4|texType4) payload packed
 * into `bvhIndex[triIdx].w`.
 *
 * Split out of common.wgsl.ts (T9-stepA): `decodeMaterialColor`,
 * `decodeSurfaceTextureId`, and `decodeIsMetal`.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const MATERIAL_DECODE_WGSL = /* wgsl */ `// B1 — fixed texel width of the per-triangle bvh_material (roughness+metalness)
// r32uint texture. Matches BVH_BEER_TEX_WIDTH in bvhBeerTexture.ts (the
// roughMetal texture is uploaded via the SAME beer-texture helper / 4096 width),
// so triIndex → vec2u(tri % W, tri / W) addresses identically. Declared here
// (not surfaceTextures) so ris/risGi/cast — which do not require surfaceTextures
// — can address bvh_material without pulling in the surface-texture module.
const BVH_MATERIAL_TEX_WIDTH: u32 = 4096u;

// Decode RGB888 + (trans4|texType4) packed material data from bvhIndex[triIdx].w.
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

// B1 (road-to-100) — decode per-triangle roughness+metalness from the packed
// bvh_material u32 (one u32 per triangle): bits[31:24]=rough×255,
// bits[23:16]=metal×255. Returns vec2f(roughness, metalness) in [0,1]. The
// caller textureLoads bvh_material at the triangle's texel (same addressing as
// bvh_beer) and passes the .r value here. See packingHelpers.packBVHRoughMetal
// for the DIFFUSE-DEFAULT INVARIANT (no authored roughness → 0.85; glass → 0.05).
fn decodeRoughMetal(packed: u32) -> vec2f {
  let rough = f32((packed >> 24u) & 0xFFu) / 255.0;
  let metal = f32((packed >> 16u) & 0xFFu) / 255.0;
  return vec2f(rough, metal);
}

// B1-ior-per-tri (2026-06-10) — decode per-triangle IOR from bits[15:8] of the
// packed bvh_material u32. The quantization maps [1.0, 3.0] → [0, 255]:
//   encode: byte = round(clamp((ior − 1) / 2 * 255, 0, 255))
//   decode: ior  = 1.0 + (byte / 255.0) * 2.0
// Covers water (1.33), glass (1.5→1.502), diamond (2.42), TiO₂ (≈2.9).
// Quantization step ≈ 0.0078 (sub-dispersion-spread for all common glasses).
// Default glass IOR = 1.5 encodes to byte 64, decodes to 1.502 (error < 0.003).
// Opaque surfaces pack 0 (IOR = 1.0); consumers gate on isGlass before calling.
fn decodeIor(packed: u32) -> f32 {
  let byte = (packed >> 8u) & 0xFFu;
  return 1.0 + f32(byte) / 255.0 * 2.0;
}

// SHADOW-01 (2026-06-11) — bv_material bits[7:0] (formerly reserved): bit 0 =
// castShadowDisabled (1 ⟺ the source primitive set castShadow:false). The DI
// shadow predicates consume it via the shared-bvh cast-shadow-masked any-hit
// traversal (traceSceneAnyCastMask), which reads the raw word directly —
// no decode helper needed here. Bits 1-7 remain reserved (zero).

`;


/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const MATERIAL_DECODE_MODULE: WgslModule = {
  name: "materialDecode",
  source: MATERIAL_DECODE_WGSL,
  requires: [],
};
