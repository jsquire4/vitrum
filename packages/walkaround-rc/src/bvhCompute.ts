/**
 * BVH build + GPU buffer packing — RC cascade-pipeline edition.
 *
 * Thin adapter over `@vitrum/shared-bvh`'s `buildSceneBVH`:
 *   1. Calls `buildSceneBVH({positionStride: 4})` to produce a single-root
 *      MeshBVH + raw typed arrays (positions / indices / normals /
 *      triMaterialId) plus a deduped `materials: THREE.Material[]` list
 *      in mesh-traversal order.
 *   2. Wraps the typed arrays in `StorageBufferAttribute` so the cascade
 *      compute pipeline (`cascadeDispatch.ts`) can access them via the
 *      Three.js WebGPU renderer backend.
 *   3. Packs RC's MaterialEntry flat-struct (16 floats per material) for
 *      the cascade compute SSBO layout in `probeRayCast.wgsl.ts`.
 *
 * The flat-struct packing is RC-specific (DDGI uses different per-material
 * binding; ReSTIR packs per-tri colour into bvhIndex.w) and stays in this
 * file rather than in `@vitrum/shared-bvh`.
 *
 * Re-build policy:
 * - Geometry topology changes (scene mount swap, room swap) → full rebuild.
 * - Material parameter edits currently trigger a full BVH rebuild (no fast SSBO-only patch path implemented).
 *
 * Caller debounces the rebuild call to avoid thrashing on rapid edits.
 *
 * Note: `StorageBufferAttribute` from `three/webgpu` is retained here because
 * the BVH buffers are consumed by the Three.js WebGPU renderer backend in
 * `cascadeDispatch.ts`.  See TSL_TO_RAW_MAPPING.md for rationale.
 */

import { StorageBufferAttribute } from 'three/webgpu';
import * as THREE from 'three';
import type { MeshBVH } from 'three-mesh-bvh';
import {
  buildSceneBVH as buildSharedBVH,
  packMaterials,
  MATERIAL_ENTRY_FLOATS,
  type MaterialEntryInput,
} from '@vitrum/shared-bvh';
import { extractThreePbrScalars } from '@vitrum/three-bindings';

export interface SceneBVH {
  bvh:           MeshBVH;
  bvhNodes:      StorageBufferAttribute;   // packed BVHNode (8 floats per node)
  positions:     StorageBufferAttribute;   // vec3f per vertex (16-byte stride: xyz + 0 pad)
  indices:       StorageBufferAttribute;   // vec3u per triangle (3 x uint32)
  materials:     StorageBufferAttribute;   // MaterialEntry per material (16 floats)
  triMaterialId: StorageBufferAttribute;   // u32 per triangle
  bounds:        THREE.Box3;
}

export interface BvhBuildOpts {
  /** Filter predicate: which Object3D's contribute geometry. */
  filter?: (obj: THREE.Object3D) => boolean;
}

/**
 * Adapt a THREE.Material to the canonical {@link MaterialEntryInput} bag.
 * Pure function; identical signature to the DDGI adapter in
 * `ddgi/probeUpdatePass.ts`, but kept module-local because each engine
 * owns whatever quirks its own material-type encoding has.
 *
 * RC-specific quirks vs the bare {@link extractThreePbrScalars} default:
 *   - `thickness` falls back to 0.1 (small but non-zero) when the source
 *     material doesn't specify one. RC's per-tri Beer-Lambert uses
 *     `thickness / attenuationDistance` and needs a non-zero numerator to
 *     produce ANY attenuation on opaque-cast non-physical materials whose
 *     attenuationColor field was nonetheless populated. Pre-W2-C5 the
 *     legacy RC packer also defaulted to 0.1.
 *   - `emissive` is pre-multiplied by `emissiveIntensity` so the GPU side
 *     sees a single radiance triple. Same as the legacy packer.
 */
function threeToMaterialEntryInput(mat: THREE.Material): MaterialEntryInput {
  const pbr = extractThreePbrScalars(mat);
  const emI = pbr.emissiveIntensity;
  return {
    baseColor: pbr.baseColor,
    roughness: pbr.roughness,
    metalness: pbr.metallic,
    emissive: [
      pbr.emissive[0] * emI,
      pbr.emissive[1] * emI,
      pbr.emissive[2] * emI,
    ],
    ior: pbr.ior,
    transmission: pbr.transmission,
    attenuationColor: pbr.attenuationColor,
    attenuationDistance: pbr.attenuationDistance,
    // RC's per-tri Beer-Lambert expects a non-zero default thickness; match
    // the pre-W2-C5 legacy packer's 0.1 fallback.
    thickness: pbr.thickness > 0 ? pbr.thickness : 0.1,
  };
}

/**
 * Pack a list of THREE materials into the canonical MaterialEntry SSBO layout
 * (16 × f32 = 64 bytes per entry) consumed by `probeRayCast.wgsl.ts`.
 *
 * Pre-W2-C5 this packer produced a different 16-float order (colorR/G/B/A,
 * then transmission/ior, then attenuationColor/Distance, then
 * roughness/metalness, then emissiveR/G/B, then thickness). The canonical
 * layout (see `@vitrum/shared-bvh/materialEntry.ts`) is shared with DDGI and
 * uses `vec3f` for color triples. Every byte rotates; the shader's field-
 * access sites (e.g. `mat.baseColor`, `mat.attenuationDistance`,
 * `mat.thickness`) updated together to match.
 *
 * Empty material list → emits a single zeroed-out entry so the SSBO has at
 * least 16 floats (every WGSL `array<T>` storage binding needs ≥1 element).
 */
/** Pack THREE materials for RC / ReSTIR-shared TLAS probe rays. */
export function packCascadeMaterials(materials: THREE.Material[]): Float32Array {
  if (materials.length === 0) {
    // packMaterials() already returns a 1-entry zero-pad for empty input,
    // but the legacy RC contract returned exactly 16 floats. Keep that
    // explicit so callers asserting on `.byteLength === 64` keep passing.
    return new Float32Array(MATERIAL_ENTRY_FLOATS);
  }
  return packMaterials(materials.map(threeToMaterialEntryInput));
}

/**
 * Build a SceneBVH from the current scene graph for the cascade pipeline.
 * Cost: ~50 ms for ~30K triangle scenes. Caller debounces.
 */
export function buildRCSceneBVH(
  scene: THREE.Scene,
  opts: BvhBuildOpts = {},
): SceneBVH {
  // Delegate single-root BVH build + per-vertex matId snapshot to shared module.
  // Stride 4 = 16-byte vec3f-aligned layout — required because the WGSL spec
  // defines `array<vec3f>` storage stride as roundUp(16, 12) = 16, NOT 12.
  // The library kernel `bvhIntersectFirstHit` reads positions as `array<vec3f>`,
  // so the CPU-side buffer MUST be packed at 16 bytes/vertex (xyz + 0-pad).
  const result = buildSharedBVH(scene, {
    positionStride: 4,
    ...(opts.filter ? { filter: opts.filter } : {}),
  });

  // Pack the deduped material list into RC's flat-struct SSBO layout.
  const materialFloats = packCascadeMaterials(result.materials);

  return {
    bvh:           result.bvh,
    bvhNodes:      new StorageBufferAttribute(result.bvhNodes,        8),
    positions:     new StorageBufferAttribute(result.positions,        4),
    indices:       new StorageBufferAttribute(result.indices,          3),
    materials:     new StorageBufferAttribute(materialFloats,         16),
    triMaterialId: new StorageBufferAttribute(result.triMaterialId,    1),
    bounds:        result.boundingBox,
  };
}
