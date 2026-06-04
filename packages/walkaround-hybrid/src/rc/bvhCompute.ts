/**
 * BVH build + GPU buffer packing — RC cascade-pipeline edition.
 *
 * Two ingestion paths to the SAME {@link SceneBVH} (raw cascade-pipeline buffers):
 *   - {@link buildRCSceneBVH} (THREE): thin adapter over `@vitrum/shared-bvh`'s
 *     `buildSceneBVH` — single-root three-mesh-bvh `MeshBVH` + raw typed arrays
 *     + a deduped `THREE.Material[]` list, packed via {@link packCascadeMaterials}.
 *   - {@link buildRCSceneBVHFromCore} (THREE-free, items_to_fix F-RC2): builds the
 *     merged BVH DIRECTLY from a `@vitrum/core` `Scene` via `mergeWorldSpaceFromCore`
 *     + packs RC's MaterialEntry from `MaterialSpec[]` via
 *     {@link packCascadeMaterialsFromCore}. NO `vitrumSceneToThree` round-trip.
 *
 * Both:
 *   1. Build merged world-space geometry at `positionStride: 4` (the 16-byte
 *      vec3f-aligned layout the cascade probe kernel reads).
 *   2. Wrap the typed arrays in `StorageBufferAttribute` so the cascade compute
 *      pipeline (`cascadeDispatch.ts`) can access them via the Three.js WebGPU
 *      renderer backend.
 *   3. Pack RC's MaterialEntry flat-struct (16 floats per material) for the
 *      cascade compute SSBO layout in `probeRayCast.wgsl.ts`.
 *
 * The flat-struct packing is RC-specific (DDGI uses different per-material
 * binding; ReSTIR packs per-tri colour into bvhIndex.w) and stays in this
 * file rather than in `@vitrum/shared-bvh`.
 *
 * Re-build policy (the merged-mode entry — `buildRCSceneBVH` is the full SAH
 * rebuild; the host owns when to call it vs. the cheaper in-place paths):
 * - Moving-instance transforms (same vertex count / topology) → NO full rebuild.
 *   `RCSubsystem.refitMergedInstance` re-uploads the re-derived world positions
 *   and refits node AABBs in place (`refitBvhBounds`), preserving tree topology
 *   and the dispatcher. This is the common walkaround case and is wired through
 *   `propagateBvhToGiSubsystems`' merged branch.
 * - Geometry topology / vertex-count changes (scene mount swap, room swap) →
 *   full rebuild via this function (the in-place refit declines on a vertex-
 *   count mismatch, and the caller falls back to `RCSubsystem.setScene`).
 * - Material parameter edits → full BVH rebuild (RC has no fast SSBO-only
 *   material-patch path; the material SSBO is only repacked inside a rebuild).
 *
 * Caller debounces the rebuild call to avoid thrashing on rapid edits.
 *
 * Note: `StorageBufferAttribute` from `three/webgpu` is retained here because
 * the BVH buffers are consumed by the Three.js WebGPU renderer backend in
 * `cascadeDispatch.ts`.  See TSL_TO_RAW_MAPPING.md for rationale.
 *
 * Moved from `@vitrum/walkaround-rc/src/bvhCompute.ts` back into
 * `@vitrum/walkaround-hybrid` (R1.6 complexity-sweep 2026-06-02) because the
 * file's THREE coupling belongs with the already-THREE-coupled hybrid engine,
 * not in the algorithm package.
 */

import { StorageBufferAttribute } from 'three/webgpu';
import * as THREE from 'three';
import type { MeshBVH } from 'three-mesh-bvh';
import type { Scene, MaterialSpec, ScenePrimitive } from '@vitrum/core';
import {
  buildSceneBVH as buildSharedBVH,
  mergeWorldSpaceFromCore,
  coreMaterialToMaterialEntry,
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
 *     NOTE: `ddgi/probeUpdateMaterials.ts` uses the same 0.1 fallback delta
 *     for the DDGI material packer — the two adapters are intentionally
 *     parallel; this cross-ref keeps them in sync.
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

// ──────────────────────────────────────────────────────────────────────────
// THREE-free core ingestion (the RC merged-BVH THREE-decouple, items_to_fix
// F-RC2). Mirrors the ReSTIR-DI emitter decouple (`46a0078`) + the standalone
// DDGI decouple (`15070cd` / `SceneBvh.updateFromCore`): build the merged
// world-space BVH from a `@vitrum/core` `Scene` via `mergeWorldSpaceFromCore`
// (instead of `buildSceneBVH(threeRoot)`) and pack RC's MaterialEntry from
// `MaterialSpec[]` (instead of reading THREE materials). NO `vitrumSceneToThree`
// round-trip, NO `THREE.Material` reads.
//
// The merged tri SET + world geometry are equivalent to the THREE path's; only
// the BVH TOPOLOGY differs (`buildArrayBvh` vs three-mesh-bvh `MeshBVH` — the
// shared-bvh R1 SAH-builder divergence). The GPU traversal returns identical
// closest-hits on either valid tree over the same geometry (proven by the RC
// brute-force oracle once F-RC1's stride bug was fixed), so the converged render
// matches. The result still routes through `RCSubsystem._uploadBVH`, so the
// F-RC1 stride-4 index pad is inherited unchanged.
// ──────────────────────────────────────────────────────────────────────────

/**
 * Core-`Scene` counterpart of {@link buildRCSceneBVH}'s `allMeshesFilter`
 * override (set in `RCSubsystem.setScene`). The THREE path filters
 * `obj.isMesh === true` (so it matches ReSTIR's merged build, which includes
 * non-PBR meshes); the faithful core equivalent accepts every mesh-like
 * primitive — `mesh` / `skinned-mesh` / `instanced-mesh` — which is also the
 * default {@link mergeWorldSpaceFromCore} filter. Analytic primitives have no
 * triangle stream and are skipped (the THREE path never sees them either —
 * they aren't `THREE.Mesh` once round-tripped through `vitrumSceneToThree`).
 * Identical to DDGI's `DDGI_CORE_MESH_FILTER` by construction.
 */
const RC_CORE_MESH_FILTER = (p: ScenePrimitive): boolean =>
  p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh';

/**
 * Force `emissiveIntensity = 1` whenever `emissive` is PRESENT, reproducing the
 * `vitrumSceneToThree` convention (it forces THREE `emissiveIntensity = 1` for
 * EVERY converted core material so a THREE round-trip never double-scales the
 * already-final radiance — `vitrumSceneToThree.ts:211`). RC's probe kernel reads
 * `mat.emissive` directly (`probeRayCast.wgsl.ts:265,271,303`), so the packed
 * `MaterialEntry.emissive` MUST match the THREE path's `emissive · 1` — NOT the
 * raw `emissive · ei`. This ei-collapse fix is LOAD-BEARING for RC (unlike DDGI,
 * whose probe pass never reads `mat.emissive`).
 *
 * A material with NO `emissive` is returned unchanged (not an emitter either
 * way). When the core `emissiveIntensity` is already 1, it is returned unchanged.
 * The same helper exists module-locally in the sibling decouples
 * (`restir/sceneBvhFromCore.ts`, `restir/packingHelpers.ts`,
 * `ddgi/probeUpdateMaterials.ts`) — kept parallel so each consumer owns its own
 * convention reproduction.
 */
function toProductionEmissiveRadiance(m: MaterialSpec): MaterialSpec {
  if (m.emissive === undefined) return m;
  if (m.emissiveIntensity === 1) return m;
  return { ...m, emissiveIntensity: 1 };
}

/**
 * Adapt a core {@link MaterialSpec} to RC's {@link MaterialEntryInput} bag — the
 * THREE-free counterpart of {@link threeToMaterialEntryInput}. Uses the canonical
 * {@link coreMaterialToMaterialEntry} field map (which already pre-multiplies
 * `emissive · emissiveIntensity`), then layers RC's two deliberate policies on top:
 *
 *   1. {@link toProductionEmissiveRadiance} forces `ei = 1` when emissive is
 *      present (the `vitrumSceneToThree` convention — load-bearing for RC since
 *      its probe kernel reads `mat.emissive`). Applied BEFORE the canonical map so
 *      its `emissive · ei` becomes `emissive · 1`.
 *   2. RC's `thickness → 0.1` floor (RC's per-tri Beer-Lambert uses
 *      `thickness / attenuationDistance` and needs a non-zero numerator). The
 *      canonical {@link coreMaterialToMaterialEntry} deliberately does NOT apply
 *      this floor (it's an RC-side policy, not a material property — see its
 *      docstring's "Deliberate non-policy" note), so RC applies it here, exactly
 *      as {@link threeToMaterialEntryInput} does on the `extractThreePbrScalars`
 *      output (this file ~107).
 */
function coreToCascadeMaterialEntryInput(mat: MaterialSpec): MaterialEntryInput {
  const entry = coreMaterialToMaterialEntry(toProductionEmissiveRadiance(mat));
  // RC's per-tri Beer-Lambert expects a non-zero default thickness; match the
  // THREE path's `thickness > 0 ? thickness : 0.1` floor (the canonical core
  // adapter leaves thickness undefined when the spec omits it).
  return {
    ...entry,
    thickness: entry.thickness !== undefined && entry.thickness > 0 ? entry.thickness : 0.1,
  };
}

/**
 * Pack a list of core {@link MaterialSpec} into RC's MaterialEntry SSBO layout
 * (16 × f32 = 64 bytes per entry) consumed by `probeRayCast.wgsl.ts`. THREE-free
 * counterpart of {@link packCascadeMaterials}. Applies the ei-collapse fix + RC's
 * thickness floor per {@link coreToCascadeMaterialEntryInput}.
 *
 * Empty list → a single zeroed-out 16-float entry (matching
 * {@link packCascadeMaterials}, so `.byteLength === 64` callers keep passing).
 */
export function packCascadeMaterialsFromCore(materials: readonly MaterialSpec[]): Float32Array {
  if (materials.length === 0) {
    return new Float32Array(MATERIAL_ENTRY_FLOATS);
  }
  return packMaterials(materials.map(coreToCascadeMaterialEntryInput));
}

/**
 * THREE-free counterpart of {@link buildRCSceneBVH}: build a {@link SceneBVH} for
 * the cascade pipeline DIRECTLY from a `@vitrum/core` `Scene` via
 * {@link mergeWorldSpaceFromCore} (the analogue of `buildSceneBVH`'s
 * `StaticGeometryGenerator` world-bake) + {@link packCascadeMaterialsFromCore}.
 *
 * Stride 4 (16-byte vec3f-aligned positions) matches the THREE path + the WGSL
 * `array<vec3f>` storage stride the RC probe kernel reads (see
 * {@link buildRCSceneBVH}). The returned `indices` are stride-3 (same as the
 * THREE path's `result.indices`); `RCSubsystem._uploadBVH` pads them to stride-4
 * before upload (the F-RC1 fix), so the core path inherits the pad automatically.
 *
 * `opts.filter` defaults to {@link RC_CORE_MESH_FILTER} (all mesh-like primitives
 * — the faithful equivalent of `setScene`'s `allMeshesFilter`).
 *
 * The `bvh` field is left as a thin placeholder `MeshBVH` (the cascade pipeline
 * consumes only the raw storage buffers — `bvhNodes`/`positions`/`indices`/
 * `materials`/`triMaterialId`/`bounds` — never the three-mesh-bvh `MeshBVH`
 * object; verified: `_uploadBVH` + `setScene` read only those fields). The core
 * merge has no `MeshBVH` to surface, so this avoids fabricating one.
 */
export function buildRCSceneBVHFromCore(
  scene: Scene,
  opts: { filter?: (p: ScenePrimitive) => boolean } = {},
): SceneBVH {
  const merged = mergeWorldSpaceFromCore(scene, {
    positionStride: 4,
    filter: opts.filter ?? RC_CORE_MESH_FILTER,
  });

  const materialFloats = packCascadeMaterialsFromCore(merged.materials);

  const bounds = new THREE.Box3(
    new THREE.Vector3(merged.boundingBox.min[0], merged.boundingBox.min[1], merged.boundingBox.min[2]),
    new THREE.Vector3(merged.boundingBox.max[0], merged.boundingBox.max[1], merged.boundingBox.max[2]),
  );

  return {
    // The cascade pipeline never reads `.bvh` (only the raw storage buffers
    // below); the core merge produces no three-mesh-bvh MeshBVH. Surface a
    // null-cast placeholder rather than building a throwaway MeshBVH.
    bvh:           null as unknown as MeshBVH,
    bvhNodes:      new StorageBufferAttribute(merged.bvhNodes,        8),
    positions:     new StorageBufferAttribute(merged.positions,        4),
    indices:       new StorageBufferAttribute(merged.indices,          3),
    materials:     new StorageBufferAttribute(materialFloats,         16),
    triMaterialId: new StorageBufferAttribute(merged.triMaterialId,    1),
    bounds,
  };
}
