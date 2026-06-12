/**
 * DDGI probe material packing (W4c — extracted from probeUpdatePass.ts).
 */

import type { MaterialSpec } from '@vitrum/core';
import {
  coreMaterialToMaterialEntry,
  packMaterials,
  MATERIAL_ENTRY_FLOATS,
  MATERIAL_ENTRY_STRIDE_BYTES,
  type MaterialEntryInput,
} from '@vitrum/shared-bvh';
import { extractPbrScalars } from '../pbrScalars.js';
import type { PbrScalarSource } from '../pbrScalars.js';

/** Maximum number of distinct materials the DDGI probe pass supports. */
export const DDGI_MAX_MATERIALS = 64;
/** Byte stride of one MaterialEntry struct (must match the WGSL layout). */
export const DDGI_MATERIAL_STRIDE_BYTES = MATERIAL_ENTRY_STRIDE_BYTES;
/** Float stride of one MaterialEntry (64 bytes = 16 × f32). */
export const DDGI_MATERIAL_ENTRY_FLOATS = MATERIAL_ENTRY_FLOATS;

function pbrToMaterialEntryInput(mat: PbrScalarSource): MaterialEntryInput {
  const pbr = extractPbrScalars(mat);
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

/** Pack structural PBR materials into canonical MaterialEntry bytes (default cap 64). */
export function packDDGIMaterials(mats: readonly PbrScalarSource[]): ArrayBuffer {
  return packDDGIMaterialsN(mats, DDGI_MAX_MATERIALS);
}

/** Pack with an explicit max slot count (matches WGSL compile-time array size). */
export function packDDGIMaterialsN(mats: readonly PbrScalarSource[], maxMaterials: number): ArrayBuffer {
  const inputs = mats.map(pbrToMaterialEntryInput);
  const out = packMaterials(inputs, maxMaterials);
  return out.buffer as ArrayBuffer;
}

/**
 * Apply the production emissive convention to a core material before packing:
 * treat `emissive` as the FINAL radiance-space colour and force
 * `emissiveIntensity = 1`, so `coreMaterialToMaterialEntry` writes
 * `emissive * 1` into the packed MaterialEntry.
 *
 * This is the SAME ei-collapse fix the ReSTIR-DI emitter decouple needed
 * (`restir/bvhCore.ts:toProductionEmissiveRadiance`, commit `46a0078`): a raw
 * `coreMaterialToMaterialEntry` computes `emissive · emissiveIntensity`, so a
 * core emitter with `ei = 4` would pack 4x the intended radiance: the exact
 * divergence that emitter-decouple GPU A/B caught.
 *
 * Force `ei = 1` whenever `emissive` is PRESENT. A material with no `emissive`
 * is returned unchanged.
 *
 * The DDGI probe-ray kernel (`probeUpdateRays.wgsl`) reads `mat.emissive` for
 * direct probe hits on plain material-emissive surfaces. Keep the packed bytes
 * on the same production convention as the ReSTIR/RC material paths: the core
 * material's `emissive` field is already the radiance-space colour, so do not
 * multiply it again by `emissiveIntensity`.
 */
function toProductionEmissiveRadiance(m: MaterialSpec): MaterialSpec {
  if (m.emissive === undefined) return m;
  if (m.emissiveIntensity === 1) return m;
  return { ...m, emissiveIntensity: 1 };
}

/**
 * Core-first counterpart to {@link packDDGIMaterialsN}: pack a deduped
 * `MaterialSpec[]` (from `mergeWorldSpaceFromCore` / `SceneBvh.updateFromCore`)
 * straight into MaterialEntry bytes via `coreMaterialToMaterialEntry`, with NO
 * material round-trip. Applies {@link toProductionEmissiveRadiance} so the
 * emissive slot follows the production `emissive * 1` convention.
 */
export function packDDGIMaterialsFromCoreN(
  mats: readonly MaterialSpec[],
  maxMaterials: number,
): ArrayBuffer {
  const inputs = mats.map((m) => coreMaterialToMaterialEntry(toProductionEmissiveRadiance(m)));
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
