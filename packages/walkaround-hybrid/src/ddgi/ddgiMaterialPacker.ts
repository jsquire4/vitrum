/**
 * DDGI material packer — pure-function adapter from `THREE.Material[]` to
 * the GPU-bound `DDGIMaterial[]` std140 uniform-buffer layout.
 *
 * Extracted from probeUpdatePass.ts (W4-A8) so the orchestrator stays
 * focused on the 5-stage dispatch chain and the packer can be re-exported
 * from there for the test fixture without dragging the entire pass module
 * into the test's import graph.
 *
 * Re-exported by `probeUpdatePass.ts` for back-compat:
 *   `import { packDDGIMaterials, DDGI_MAX_MATERIALS, DDGI_MATERIAL_STRIDE_BYTES,
 *            DDGI_MATERIAL_ENTRY_FLOATS } from '.../probeUpdatePass.js';`
 * still works.
 */

import * as THREE from 'three';
import { extractThreePbrScalars } from '@vitrum/three-bindings';

// DDGI material buffer sizing constants.
// `materialsBuf` holds one DDGIMaterial struct per material slot.
// DDGIMaterial WGSL layout: 64 bytes = 16 × f32 (std140, see packDDGIMaterials).
/** Maximum number of distinct materials the DDGI probe pass supports. */
export const DDGI_MAX_MATERIALS = 64;
/** Byte stride of one DDGIMaterial struct (must match the WGSL layout). */
export const DDGI_MATERIAL_STRIDE_BYTES = 64;
/** Float stride of one DDGIMaterial entry (64 bytes = 16 × f32). */
export const DDGI_MATERIAL_ENTRY_FLOATS = 16;

/**
 * Pack a list of THREE materials into the GPU-bound DDGIMaterial std140 layout
 * (64 bytes per material, 16 floats each). Pure function — does no GPU calls.
 * Used by both `ProbeUpdatePass._uploadMaterials` (at runtime, via
 * {@link packDDGIMaterialsN}) and the byte-equivalence test fixture.
 *
 * WGSL layout (offsets in bytes per entry):
 *   offset  0: baseColor: vec3f  (12) + _pad0:  f32 (4)
 *   offset 16: emissive:  vec3f  (12) + roughness: f32 (4)
 *   offset 32: metalness, ior, transmission, _pad1 (4 × f32)
 *   offset 48: attenuationColor: vec3f (12) + flags: u32 (4)
 *     flags bit 0: isGlass (transmission > 0)
 *
 * Defaults (when a THREE field is absent): baseColor [1,1,1], emissive
 * [0,0,0], roughness 0.5, metallic 0, transmission 0, ior 1.5,
 * attenuationColor [1,1,1]. Matches the pre-P2-6.1 inline packer.
 */
export function packDDGIMaterials(mats: readonly THREE.Material[]): ArrayBuffer {
  return packDDGIMaterialsN(mats, DDGI_MAX_MATERIALS);
}

/**
 * Like {@link packDDGIMaterials} but accepts an explicit max-material count
 * so instances with `maxMaterials !== 64` get a correctly-sized buffer.
 * Used internally by `ProbeUpdatePass._uploadMaterials`.
 */
export function packDDGIMaterialsN(mats: readonly THREE.Material[], maxMaterials: number): ArrayBuffer {
  const ENTRY = DDGI_MATERIAL_ENTRY_FLOATS;
  const buf = new ArrayBuffer(maxMaterials * DDGI_MATERIAL_STRIDE_BYTES);
  const data = new Float32Array(buf);
  // u32 view onto the same backing buffer so the `flags` slot can be written
  // as a real u32 (the WGSL struct declares it as u32; writing 1.0 as f32
  // would land 0x3F800000 ≠ 1u in the GPU read).
  const u32view = new Uint32Array(buf);
  const matsToUse = mats.slice(0, maxMaterials);
  matsToUse.forEach((mat, i) => {
    const base = i * ENTRY;
    const pbr = extractThreePbrScalars(mat);
    data[base + 0] = pbr.baseColor[0];
    data[base + 1] = pbr.baseColor[1];
    data[base + 2] = pbr.baseColor[2];
    data[base + 3] = 0; // _pad0
    data[base + 4] = pbr.emissive[0];
    data[base + 5] = pbr.emissive[1];
    data[base + 6] = pbr.emissive[2];
    data[base + 7] = pbr.roughness;
    data[base + 8] = pbr.metallic;
    data[base + 9] = pbr.ior;
    data[base + 10] = pbr.transmission;
    data[base + 11] = 0; // _pad1
    data[base + 12] = pbr.attenuationColor[0];
    data[base + 13] = pbr.attenuationColor[1];
    data[base + 14] = pbr.attenuationColor[2];
    u32view[base + 15] = pbr.transmission > 0 ? 1 : 0;
  });
  return buf;
}
