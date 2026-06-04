/**
 * DDGI probe material packing (W4c — extracted from probeUpdatePass.ts).
 */

import * as THREE from 'three';
import type { MaterialSpec } from '@vitrum/core';
import { extractThreePbrScalars } from '@vitrum/three-bindings';
import {
  coreMaterialToMaterialEntry,
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

/**
 * Apply the `vitrumSceneToThree` emissive convention to a core material before
 * packing: treat `emissive` as the FINAL radiance-space colour and force
 * `emissiveIntensity = 1`, so `coreMaterialToMaterialEntry` writes
 * `emissive · 1` into the packed MaterialEntry — exactly what the THREE
 * round-trip produces (`vitrumSceneToThree.ts:201-211` forces THREE
 * `emissiveIntensity = 1` for EVERY converted material, "avoid double-scale on
 * round-trip"), and therefore what the THREE-path DDGI materials carry.
 *
 * This is the SAME ei-collapse fix the ReSTIR-DI emitter decouple needed
 * (`sceneBvhFromCore.ts:toProductionEmissiveRadiance`, commit `46a0078`): a raw
 * `coreMaterialToMaterialEntry` computes `emissive · emissiveIntensity`, so a
 * core emitter with `ei = 4` would pack 4× the radiance the THREE path packs —
 * the exact divergence that emitter-decouple GPU A/B caught.
 *
 * Force `ei = 1` whenever `emissive` is PRESENT (including when the core
 * `emissiveIntensity` is `undefined`, since `vitrumSceneToThree` forces THREE
 * `ei = 1` unconditionally). A material with no `emissive` is returned unchanged.
 *
 * NOTE on relevance: the DDGI PROBE-RAY kernel (`probeUpdateRays.wgsl`) does NOT
 * currently read `mat.emissive` (probe radiance is sun + sky + glass-tint, not
 * surface emission), so this convention is presently invisible to rendered DDGI
 * output. It is applied anyway so (a) the packed MaterialEntry bytes are
 * BYTE-IDENTICAL to the THREE path (the CPU material-set equivalence gate), and
 * (b) the path stays correct if a future probe-shading revision starts reading
 * the emissive slot. The non-emissive fields the kernel DOES read — `flags`
 * (glass bit), `transmission`, `attenuationColor`, `baseColor` — pass through
 * `coreMaterialToMaterialEntry` identically to `extractThreePbrScalars`.
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
 * `THREE.Material` round-trip. Applies {@link toProductionEmissiveRadiance} so
 * the emissive slot matches the THREE path's `emissive · 1`.
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
