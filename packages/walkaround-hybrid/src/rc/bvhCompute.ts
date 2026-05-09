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
 * - Material parameter edits → patch materials SSBO only (planned;
 *   today everything rebuilds).
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
import { buildSceneBVH as buildSharedBVH } from '@vitrum/shared-bvh';

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
 * Pack THREE.Material array into RC's cascade-compute MaterialEntry SSBO
 * layout (16 × f32 = 64 bytes per entry). Matches the flat-struct decode
 * in `probeRayCast.wgsl.ts`.
 *
 * Layout per entry (offsets in floats):
 *   [0]  colorR     [1]  colorG     [2]  colorB     [3]  colorA (= 1.0)
 *   [4]  transmission [5] ior       [6]  attenColorR [7] attenColorG
 *   [8]  attenColorB  [9] attenDist [10] roughness  [11] metalness
 *   [12] emissiveR  [13] emissiveG  [14] emissiveB  [15] thickness
 *
 * Empty material list → emits a single zeroed-out entry so the SSBO has at
 * least 16 floats.
 */
function packCascadeMaterials(materials: THREE.Material[]): Float32Array {
  if (materials.length === 0) {
    return new Float32Array(16);
  }
  const out = new Float32Array(materials.length * 16);
  for (let i = 0; i < materials.length; i++) {
    const mat = materials[i] as
      THREE.MeshPhysicalMaterial & THREE.MeshStandardMaterial;
    const phys = mat as THREE.MeshPhysicalMaterial;
    const o = i * 16;
    out[o + 0]  = mat.color?.r ?? 0.8;
    out[o + 1]  = mat.color?.g ?? 0.8;
    out[o + 2]  = mat.color?.b ?? 0.8;
    out[o + 3]  = 1.0;
    out[o + 4]  = phys.transmission ?? 0;
    out[o + 5]  = phys.ior ?? 1.5;
    out[o + 6]  = phys.attenuationColor?.r ?? 1;
    out[o + 7]  = phys.attenuationColor?.g ?? 1;
    out[o + 8]  = phys.attenuationColor?.b ?? 1;
    out[o + 9]  = phys.attenuationDistance ?? 1e9;
    out[o + 10] = mat.roughness ?? 1;
    out[o + 11] = mat.metalness ?? 0;
    out[o + 12] = (mat.emissive?.r ?? 0) * (mat.emissiveIntensity ?? 1);
    out[o + 13] = (mat.emissive?.g ?? 0) * (mat.emissiveIntensity ?? 1);
    out[o + 14] = (mat.emissive?.b ?? 0) * (mat.emissiveIntensity ?? 1);
    // Beer-Lambert input: glass slab thickness in scene units (inches).
    out[o + 15] = phys.thickness ?? 0.1;
  }
  return out;
}

/**
 * Build a SceneBVH from the current scene graph for the cascade pipeline.
 * Cost: ~50 ms for ~30K triangle scenes. Caller debounces.
 */
export function buildSceneBVH(
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
