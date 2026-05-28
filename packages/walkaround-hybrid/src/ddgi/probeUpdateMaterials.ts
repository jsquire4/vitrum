/**
 * DDGI probe material packing (W4c — extracted from probeUpdatePass.ts).
 */

import * as THREE from 'three';
import { extractThreePbrScalars } from '@vitrum/three-bindings';
import {
  packMaterials,
  MATERIAL_ENTRY_FLOATS,
  MATERIAL_ENTRY_STRIDE_BYTES,
  type MaterialEntryInput,
} from '@vitrum/shared-bvh';

/** Maximum number of distinct materials the DDGI probe pass supports. */
export const DDGI_MAX_MATERIALS = 64;
/** Byte stride of one MaterialEntry struct (must match the WGSL layout). */
export const DDGI_MATERIAL_STRIDE_BYTES = MATERIAL_ENTRY_STRIDE_BYTES;
/** Float stride of one MaterialEntry (64 bytes = 16 × f32). */
export const DDGI_MATERIAL_ENTRY_FLOATS = MATERIAL_ENTRY_FLOATS;

function threeToMaterialEntryInput(mat: THREE.Material): MaterialEntryInput {
  const pbr = extractThreePbrScalars(mat);
  return {
    baseColor: pbr.baseColor,
    roughness: pbr.roughness,
    metalness: pbr.metallic,
    emissive: pbr.emissive,
    ior: pbr.ior,
    transmission: pbr.transmission,
    attenuationColor: pbr.attenuationColor,
    attenuationDistance: pbr.attenuationDistance,
    thickness: pbr.thickness,
  };
}

/** Pack THREE materials into canonical MaterialEntry bytes (default cap 64). */
export function packDDGIMaterials(mats: readonly THREE.Material[]): ArrayBuffer {
  return packDDGIMaterialsN(mats, DDGI_MAX_MATERIALS);
}

/** Pack with an explicit max slot count (matches WGSL compile-time array size). */
export function packDDGIMaterialsN(mats: readonly THREE.Material[], maxMaterials: number): ArrayBuffer {
  const inputs = mats.map(threeToMaterialEntryInput);
  const out = packMaterials(inputs, maxMaterials);
  return out.buffer as ArrayBuffer;
}

/** Pad stride-3 triangle indices for vec4u WGSL storage (standalone SceneBvh path). */
export function padTriangleIndicesToVec4(indices: Uint32Array): Uint32Array {
  const triCount = Math.floor(indices.length / 3);
  const out = new Uint32Array(triCount * 4);
  for (let t = 0; t < triCount; t += 1) {
    out[t * 4] = indices[t * 3]!;
    out[t * 4 + 1] = indices[t * 3 + 1]!;
    out[t * 4 + 2] = indices[t * 3 + 2]!;
    out[t * 4 + 3] = 0;
  }
  return out;
}
