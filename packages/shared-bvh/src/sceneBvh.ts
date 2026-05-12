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
import { buildSceneBVH } from './bvhCommon.js';

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
  /** Array of unique THREE.Material objects in scene order. */
  materials: THREE.Material[];
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

export interface SceneBvhOptions {
  /**
   * Invoked when a BVH rebuild takes longer than 50ms (same threshold as before).
   * When omitted, a console.warn is emitted (legacy behavior).
   */
  readonly onSlowRebuild?: (elapsedMs: number) => void;
}

export class SceneBvh {
  private _buffers: SceneBvhBuffers | null = null;
  /** Cached geometry version for dirty-checking. */
  private _lastGeometryVersion = -1;

  constructor(private readonly opts: SceneBvhOptions = {}) {}

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

    if (meshes.length === 0) return;

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

  dispose(): void {
    this._buffers = null;
    this._lastGeometryVersion = -1;
  }
}
