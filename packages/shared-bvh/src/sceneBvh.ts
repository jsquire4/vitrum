/**
 * SceneBvh — unified BVH over the raster scene for DDGI probe ray tracing.
 *
 * Thin DDGI wrapper around the Tier 2 shared `buildSceneBVH` core
 * (`./bvhCommon.ts`). The class shell remains in shared-bvh for one reason
 * the per-frame DDGI loop depends on:
 *
 *   - Geometry-version dirty tracking — `update(scene)` walks the same
 *     **visible** mesh set as `buildSceneBVH` and only rebuilds the BVH when
 *     `BufferAttribute.version + mesh.id` sum changes. The shared core is
 *     a pure builder (no dirty cache) by design — see
 *     `useSceneBVH` in the host app for the canonical React-side debounce;
 *     DDGI uses this lower-level cache because it runs inside `useFrame`.
 *
 * GPU buffer layout matches what the inline WGSL in probeUpdateRays.wgsl.ts
 * expects (same layout as three-mesh-bvh's bvhIntersectFirstHit WGSL).
 * Field names match the shared lib core 1:1 — no rename adaptor.
 */

import * as THREE from 'three';
import type { MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import { buildSceneBVH } from './bvhCommon.js';
import { mergeWorldSpaceFromCore } from './worldSpaceMerge.js';
import { fingerprintBuffers } from './bufferFingerprint.js';

export interface SceneBvhBuffers {
  /** Flat BVHNode array — bounds (6 f32) + rightChild/triOffset (u32) +
   *  splitAxis/triCount (u32) = 8 × u32 = 32 bytes per node. */
  bvhNodes: Float32Array;
  positions: Float32Array;
  /** Triangle index triplets (as u32 triples). */
  indices: Uint32Array;
  normals: Float32Array;
  /** One u32 per triangle, material index (matches lib's `triMaterialId`). */
  triMaterialId: Uint32Array;
  /**
   * Array of unique THREE.Material objects in scene order. Populated by the
   * THREE path ({@link SceneBvh.update}); EMPTY on the core-first path
   * ({@link SceneBvh.updateFromCore}), which fills {@link coreMaterials} instead.
   */
  materials: THREE.Material[];
  /**
   * Deduped core `MaterialSpec[]` in scene order — the THREE-free counterpart to
   * {@link materials}, populated ONLY by {@link SceneBvh.updateFromCore} (the
   * THREE path leaves it `undefined`). Consumers that can read core materials
   * directly (the DDGI probe-material packer via `coreMaterialToMaterialEntry`)
   * prefer this when present, avoiding the `THREE.Material` round-trip. Indexed
   * by the same {@link triMaterialId}.
   */
  coreMaterials?: readonly MaterialSpec[];
  /** BVH bounding box (world space). */
  boundingBox: THREE.Box3;
}

/**
 * DDGI's mesh filter — preserves the pre-migration "any visible mesh with
 * geometry + position attribute" semantics. The shared core's default
 * filter would only pick up MeshStandardMaterial + MeshPhysicalMaterial;
 * keeping the broader filter here avoids a silent regression if the DDGI
 * scene ever picks up MeshBasicMaterial / MeshLambertMaterial / etc.
 *
 * (`traverseVisible` inside `buildSceneBVH` already skips `obj.visible
 * === false`, so the original `!obj.visible` early-return is redundant.)
 */
const DDGI_MESH_FILTER = (obj: THREE.Object3D): boolean => {
  if (!(obj instanceof THREE.Mesh)) return false;
  if (!obj.geometry) return false;
  if (!obj.geometry.attributes['position']) return false;
  return true;
};

/**
 * Core-`Scene` counterpart of {@link DDGI_MESH_FILTER} for
 * {@link SceneBvh.updateFromCore}. The THREE filter accepts "any visible mesh
 * with a position attribute" (`InstancedMesh` extends `THREE.Mesh`, so it is
 * INCLUDED). The faithful core equivalent therefore accepts every mesh-like
 * primitive — `mesh` / `skinned-mesh` / `instanced-mesh` — which is exactly the
 * default {@link mergeWorldSpaceFromCore} filter. Analytic primitives have no
 * triangle stream and are skipped (the THREE path never sees them either; they
 * are not `THREE.Mesh` once round-tripped through `vitrumSceneToThree`). Keeping
 * the broader-than-PBR semantics here mirrors the THREE filter's intent ("don't
 * silently drop a mesh because its material isn't Standard/Physical").
 */
const DDGI_CORE_MESH_FILTER = (p: ScenePrimitive): boolean =>
  p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh';

export interface SceneBvhOptions {
  /**
   * Invoked when a BVH rebuild takes longer than 50ms (same threshold as before).
   * When omitted, a console.warn is emitted (legacy behavior).
   */
  readonly onSlowRebuild?: (elapsedMs: number) => void;
}

export class SceneBvh {
  private _buffers: SceneBvhBuffers | null = null;
  /** Cached geometry version for dirty-checking (THREE `update` path). */
  private _lastGeometryVersion = -1;
  /** Cached content fingerprint for the core-first `updateFromCore` path —
   *  bumps when the merged world-space stream or material set changes. Kept
   *  separate from `_lastGeometryVersion` (which hashes THREE attribute
   *  versions) so the two entry points never alias each other's dirty state. */
  private _lastCoreFingerprint = -1;

  private readonly opts: SceneBvhOptions;

  constructor(opts: SceneBvhOptions = {}) {
    this.opts = opts;
  }

  get buffers(): SceneBvhBuffers | null {
    return this._buffers;
  }

  /**
   * Walk `scene`, collect **visible** meshes (same as `buildSceneBVH`'s
   * `traverseVisible`), rebuild BVH if dirty.
   *
   * IMPORTANT: both the version-hash traversal and the actual build pass
   * MUST use the same filter predicate (`DDGI_MESH_FILTER`). Divergence
   * would cause the dirty check to track a different mesh set than was
   * built, producing false-negatives (missed rebuilds) when meshes leave
   * one filter set but stay in the other.
   */
  update(scene: THREE.Scene): void {
    // Collect filtered meshes once for the dirty check. The shared core
    // re-walks scene roots itself, so this list is only used to compute
    // the geometry-version hash — keeping the dirty cache cheap.
    const meshes: THREE.Mesh[] = [];
    scene.traverseVisible((obj) => {
      if (DDGI_MESH_FILTER(obj)) meshes.push(obj as THREE.Mesh);
    });

    if (meshes.length === 0) {
      this._buffers = null;
      this._lastGeometryVersion = -1;
      return;
    }

    // Geometry-version dirty check — sum of every mesh's geometry version
    // bumps plus the mesh count. Cheap to compute, stable across frames
    // where no geometry actually changed.
    //
    // Critical: do NOT include `mesh.id` (Three.js's monotonic global Mesh
    // counter). React reconciliation in the host can construct fresh
    // THREE.Mesh objects per render (e.g. JSX-declared <mesh> inside a
    // .map()), bumping mesh.id each frame even when the structural scene
    // is unchanged. Hashing mesh.id makes the dirty check permanently dirty,
    // triggering a full BVH rebuild every frame (~79ms on a livingRoom-scale
    // scene). posAttr.version is sufficient — it bumps when
    // `position.needsUpdate = true` or the attribute is replaced, which is
    // the actual signal we want. Mesh count detects add/remove without
    // depending on identity. — fix 2026-05-12 (walkaround lockup).
    let version = meshes.length * 1000003;
    for (const m of meshes) {
      const posAttr = m.geometry.attributes['position'] as THREE.BufferAttribute;
      version += posAttr.version ?? 0;
    }
    if (version === this._lastGeometryVersion && this._buffers !== null) return;
    this._lastGeometryVersion = version;

    const t0 = performance.now();

    // Delegate the heavy lifting to the shared Tier 2 builder. Pass our
    // custom filter (broader than the lib default) and stride 4
    // (16-byte-aligned vec3f layout — required because DDGI's WGSL probe
    // pass reads `bvh_position: array<vec3f>` and `bvh_normal:
    // array<vec3f>`, and per the WGSL spec `array<vec3f>` has stride
    // roundUp(16, 12) = 16, NOT 12.  Pre-fix the CPU packed positions at
    // 12 bytes/vertex while the GPU read at 16 bytes/vertex — every
    // vertex past index 0 was garbled (mirrors ReSTIR commit d88f6c8).
    // Stride-4 leaves the .w slot zero-padded; the shader reads only
    // .xyz so the padding is invisible. — see probeUpdateRays.wgsl.ts).
    const result = buildSceneBVH(scene, {
      filter: DDGI_MESH_FILTER,
      positionStride: 4,
    });

    const elapsed = performance.now() - t0;
    if (elapsed > 50) {
      if (this.opts.onSlowRebuild) {
        this.opts.onSlowRebuild(elapsed);
      } else {
        console.warn(`[DDGI SceneBvh] BVH rebuild took ${elapsed.toFixed(0)}ms (>50ms threshold)`);
      }
    }

    // Forward the shared-core result fields 1:1 — no rename adaptor.
    this._buffers = {
      bvhNodes:      result.bvhNodes,
      positions:     result.positions,
      indices:       result.indices,
      normals:       result.normals,
      triMaterialId: result.triMaterialId,
      materials:     result.materials,
      boundingBox:   result.boundingBox,
    };
  }

  /**
   * Core-first counterpart to {@link update}: rebuild the DDGI merged BVH from a
   * `@vitrum/core` `Scene` DIRECTLY — NO `vitrumSceneToThree` round-trip and NO
   * THREE materials. The THREE-decouple of the DDGI merged-BVH ingestion path
   * (`plan/three-decouple-analysis-2026-06-03.md` §4 step 5 / §7 increment 4),
   * mirroring the ReSTIR-DI emitter decouple (`46a0078`).
   *
   * Geometry comes from {@link mergeWorldSpaceFromCore} (the THREE-free analogue
   * of `buildSceneBVH`'s `StaticGeometryGenerator` world-bake) at the same stride
   * 4 the DDGI WGSL reads (`array<vec3f>` = 16-byte stride; see the `update`
   * docstring for why stride 3 garbles every vertex past index 0). Its mesh
   * filter is {@link DDGI_CORE_MESH_FILTER}, the faithful core equivalent of
   * {@link DDGI_MESH_FILTER} (instanced meshes INCLUDED — `InstancedMesh extends
   * THREE.Mesh`).
   *
   * The result carries deduped {@link WorldSpaceMergeResult.materials}
   * (`MaterialSpec[]`) in {@link SceneBvhBuffers.coreMaterials}; {@link
   * SceneBvhBuffers.materials} (THREE) is left EMPTY. The probe-material packer
   * reads `coreMaterials` via `coreMaterialToMaterialEntry` when present
   * (applying the `vitrumSceneToThree` `emissiveIntensity = 1` convention so the
   * packed bytes match the THREE path — see `ProbeUpdatePass`).
   *
   * Dirty tracking: a content fingerprint over the merged stream + materials,
   * kept in {@link _lastCoreFingerprint} (separate from the THREE path's
   * attribute-version hash). When the fingerprint is unchanged the cached buffers
   * are kept (the per-frame DDGI loop only rebuilds on a real change).
   *
   * NOTE: the BVH TOPOLOGY this produces differs from `update`'s
   * (three-mesh-bvh `MeshBVH` vs `buildArrayBvh` — the analysis's R1 SAH-builder
   * divergence), so a low-spp A/B against the THREE path differs on noise; the
   * CONVERGED render matches (the merged tri SET + world geometry are equivalent
   * — pinned by `worldSpaceMerge.test.ts`).
   */
  updateFromCore(scene: Scene): void {
    const merged = mergeWorldSpaceFromCore(scene, {
      positionStride: 4,
      filter: DDGI_CORE_MESH_FILTER,
    });

    if (merged.triangleCount === 0) {
      this._buffers = null;
      this._lastCoreFingerprint = -1;
      return;
    }

    // Content dirty check over the merge-order stream (BVH-order would also work
    // but the merge stream is the order-stable signal) + the per-tri material
    // ids + a material-set hash (so a material edit that doesn't move a vertex
    // still triggers a rebuild — the packed MaterialEntry bytes change).
    const fingerprint = fingerprintBuffers(
      merged.positions.buffer as ArrayBuffer,
      merged.mergedIndices.buffer as ArrayBuffer,
      merged.mergedTriMaterialId.buffer as ArrayBuffer,
      new Float32Array(materialSetHashFloats(merged.materials)).buffer,
    );
    if (fingerprint === this._lastCoreFingerprint && this._buffers !== null) return;
    this._lastCoreFingerprint = fingerprint;

    const t0 = performance.now();

    const boundingBox = new THREE.Box3(
      new THREE.Vector3(
        merged.boundingBox.min[0],
        merged.boundingBox.min[1],
        merged.boundingBox.min[2],
      ),
      new THREE.Vector3(
        merged.boundingBox.max[0],
        merged.boundingBox.max[1],
        merged.boundingBox.max[2],
      ),
    );

    this._buffers = {
      bvhNodes:      merged.bvhNodes,
      positions:     merged.positions,
      indices:       merged.indices,
      normals:       merged.normals,
      triMaterialId: merged.triMaterialId,
      materials:     [], // core path uses `coreMaterials`; THREE list stays empty
      coreMaterials: merged.materials,
      boundingBox,
    };

    const elapsed = performance.now() - t0;
    if (elapsed > 50) {
      if (this.opts.onSlowRebuild) {
        this.opts.onSlowRebuild(elapsed);
      } else {
        console.warn(`[DDGI SceneBvh] core BVH rebuild took ${elapsed.toFixed(0)}ms (>50ms threshold)`);
      }
    }
  }

  dispose(): void {
    this._buffers = null;
    this._lastGeometryVersion = -1;
    this._lastCoreFingerprint = -1;
  }
}

/**
 * Compact per-material float signature for the {@link SceneBvh.updateFromCore}
 * dirty fingerprint — hashes only the fields the DDGI probe pass consumes via
 * the packed `MaterialEntry` (baseColor, emissive·ei, roughness, metallic,
 * transmission, ior, attenuationColor). A material edit that changes any of
 * these bumps the fingerprint even when geometry is unchanged, so the probe
 * BVH/material rebuild fires. (This is a coarse change-detector, not a GPU-byte
 * mirror; the actual upload bytes come from `coreMaterialToMaterialEntry` +
 * `packMaterials`.)
 */
function materialSetHashFloats(materials: readonly MaterialSpec[]): number[] {
  const out: number[] = [];
  for (const m of materials) {
    const ei = m.emissiveIntensity ?? 1;
    const em = m.emissive ?? [0, 0, 0];
    const bc = m.baseColor ?? [1, 1, 1];
    const ac = m.attenuationColor ?? [1, 1, 1];
    out.push(
      bc[0], bc[1], bc[2],
      em[0] * ei, em[1] * ei, em[2] * ei,
      m.roughness ?? 1,
      m.metallic ?? 0,
      m.transmission ?? 0,
      m.ior ?? 1.5,
      ac[0], ac[1], ac[2],
    );
  }
  return out;
}
