/**
 * HybridEnginePrimitiveUpdates — geometry-change handlers for `HybridEngine.updatePrimitive`.
 *
 * Extracted from `HybridEngine.ts` (refactor sweep 2026-05-18). Both
 * functions here are pure with respect to engine state: they take the
 * owned resources they need as explicit arguments (BVH buffers, pipeline,
 * DDGI, root scene, light params) and return the new BVH buffers handle
 * the caller should swap into `engine._bvhBuffers`. The engine owns the
 * resource lifetimes; this module owns the algorithm.
 *
 * Routing rules (mirrors the pre-extraction dispatcher comment in
 * `HybridEngine.updatePrimitive`):
 *
 *  - `patch.transform` present AND no topology fields → call
 *    {@link transformRefit}: refit BVH bounds in place (no SAH rebuild,
 *    no pipeline recompile, no DDGI atlas invalidation), rewrite the
 *    affected primitive's vertex slice in `bvhPositions`, reset the
 *    accumulator.
 *  - any topology field present (`positions` / `normals` / `uvs` /
 *    `tangents` / `indices` / `instances` / `params` / `shape` /
 *    `fallbackMesh` / `kind`) → call {@link topologyRebuild}: re-run
 *    `buildReSTIRSceneBVH`, destroy + re-upload the four BVH GPU buffers,
 *    reset the accumulator.
 *  - material-only patches → throw with a pointer to the material fast
 *    path on the sibling branch (the dispatcher in HybridEngine handles
 *    the throw; this module only owns geometry / transform).
 *
 * The hot-path branch design is preserved from the pre-extraction code
 * verbatim — no behaviour change.
 */

import * as THREE from 'three';
import type { ScenePrimitive } from '@vitrum/core';
import { refitBvhBounds } from '@vitrum/shared-bvh';
import { buildReSTIRSceneBVH, disposeSceneBVH } from './restir/bvhCompute.js';
import type { SceneBVHBuffers } from './restir/bvhCompute.js';
import type { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import type { DDGI } from './ddgi/DDGI.js';

/** Aggregated resources the primitive-update paths need from the engine. */
export interface PrimitiveUpdateContext {
  /** The engine's currently-owned BVH GPU buffers. May be null if the
   *  pipeline init has not yet published — callers (transformRefit) MUST
   *  fall back to a topology rebuild in that case. */
  readonly bvhBuffers: SceneBVHBuffers | null;
  /** The synthesized-or-host THREE.Scene root that owns the meshes we
   *  patch in place. Null when the engine has no BVH source — caller
   *  treats as an error. */
  readonly threeRoot: THREE.Object3D | null;
  /** Live GPU pipeline; may be null during init. The fast paths fall
   *  through to a rebuild when null. */
  readonly pipeline: WalkaroundGPUPipeline | null;
  /** DDGI subsystem; receives probe-cache invalidation calls. */
  readonly ddgi: DDGI;
  /** Primary directional light dir, threaded into a rebuild's BVH-builder. */
  readonly primaryLightDir: readonly [number, number, number];
  /** Primary directional light intensity, threaded into a rebuild's
   *  BVH-builder. */
  readonly primaryLightIntensity: number;
}

/** Result of a primitive-update call. */
export interface PrimitiveUpdateResult {
  /** The BVH buffers that should be the engine's new `_bvhBuffers` value.
   *  - For {@link transformRefit}, unchanged from the input (the refit
   *    mutates the buffer in place).
   *  - For {@link topologyRebuild}, the freshly-built replacement; the old
   *    buffer has already been disposed inside the function. */
  readonly bvhBuffers: SceneBVHBuffers;
}

/**
 * Transform-only fast path (Option (c) per items_to_fix.md A3).
 *
 * The BVH topology is preserved — only AABB bounds are refit. Cost is
 * O(affectedVertices + totalBvhNodes), no pipeline recompile, no DDGI
 * atlas invalidation. For a single primitive on a 30k-tri scene this
 * runs in well under 1 ms vs. ~50 ms for a full SAH rebuild + pipeline
 * recompile.
 *
 * Steps:
 *  1. Look up the affected mesh by `name === id` in the synthesized
 *     THREE scene (or `_threeScene` for the host-Three-scene path).
 *  2. Apply the new transform to the THREE.Mesh (`matrix` + `matrixWorld`).
 *  3. Compute the matrix delta `D = matrixWorldNew · matrixWorldAtBuild⁻¹`.
 *  4. For each vertex `v` in `[vertexStart, vertexStart + vertexCount)`,
 *     read the old world-space position from `bvhPositions.cpuData`,
 *     apply `D`, write the new world-space position back. (UV in `.w`
 *     is preserved.)
 *  5. Update `matrixWorldAtBuild` snapshot to the new matrix world.
 *  6. Run `refitBvhBounds` on the BVH node buffer.
 *  7. Upload the refit nodes + position slice via the pipeline.
 *  8. Reset the accumulator (history is invalid — the primitive moved).
 *
 * Falls through to {@link topologyRebuild} when the BVH hasn't been
 * published yet or when no vertex range matches the primitive id.
 */
export function transformRefit(
  id: string,
  patch: Partial<ScenePrimitive>,
  ctx: PrimitiveUpdateContext,
): PrimitiveUpdateResult {
  const bvh = ctx.bvhBuffers;
  if (bvh == null) {
    // Pipeline still initialising — nothing to refit. Fall through to a
    // full rebuild so the next setScene picks up the new transform.
    return topologyRebuild(id, patch, ctx);
  }

  const range = bvh.meshVertexRanges.find((r) => r.name === id);
  if (range == null || range.vertexCount === 0) {
    // No vertices for this primitive in the merged buffer (e.g. an
    // emitter-only primitive, or a name mismatch). Fall back to a
    // topology rebuild so the user's intent isn't silently dropped.
    return topologyRebuild(id, patch, ctx);
  }

  const root = ctx.threeRoot;
  if (root == null) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): no THREE scene available for refit.`,
    );
  }
  let mesh: THREE.Mesh | null = null;
  root.traverseVisible((obj) => {
    if (mesh == null && obj.name === id && (obj as THREE.Mesh).isMesh) {
      mesh = obj as THREE.Mesh;
    }
  });
  if (mesh == null) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): primitive has no THREE.Mesh in the synthesized scene.`,
    );
  }
  const meshRef = mesh as THREE.Mesh;

  // Apply the new transform. The Scene contract says transform is a
  // 16-element column-major Mat4 (see core/src/scene.ts:MeshPrimitive).
  const newMat = new THREE.Matrix4();
  const transform = (patch as { transform?: ArrayLike<number> }).transform;
  if (transform && transform.length >= 16) {
    newMat.fromArray(Array.from(transform));
  } else {
    newMat.identity();
  }
  meshRef.matrix.copy(newMat);
  meshRef.matrixWorld.copy(newMat);
  meshRef.matrixAutoUpdate = false;

  // Compute matrix delta D = newMat · oldMat⁻¹. We transform each
  // already-baked world-space vertex through D to get the new
  // world-space vertex; equivalent to local⁻¹ → new-world round-trip
  // but without storing local-space positions.
  const oldMatWorld = new THREE.Matrix4().fromArray(Array.from(range.matrixWorldAtBuild));
  const oldMatWorldInv = new THREE.Matrix4().copy(oldMatWorld).invert();
  const delta = new THREE.Matrix4().multiplyMatrices(newMat, oldMatWorldInv);

  // Rewrite the affected vertex slice of bvhPositions.cpuData. The
  // stride-4 layout packs world-space xyz into [0..2] and UV-as-u32
  // into [3] (preserved here). Use a single typed-array view over the
  // shared ArrayBuffer so the changes land in cpuData.
  const positionsF32 = new Float32Array(bvh.bvhPositions.cpuData);
  const STRIDE = 4;
  const baseVertex = range.vertexStart;
  const sliceVerts = range.vertexCount;
  const tmp = new THREE.Vector3();
  for (let v = 0; v < sliceVerts; v++) {
    const off = (baseVertex + v) * STRIDE;
    tmp.x = positionsF32[off + 0]!;
    tmp.y = positionsF32[off + 1]!;
    tmp.z = positionsF32[off + 2]!;
    tmp.applyMatrix4(delta);
    positionsF32[off + 0] = tmp.x;
    positionsF32[off + 1] = tmp.y;
    positionsF32[off + 2] = tmp.z;
    // .w (UV pack) preserved.
  }

  // Update the matrix snapshot in-place so subsequent transform
  // patches compute their delta against the latest matrix, not the
  // original build-time matrix.
  range.matrixWorldAtBuild.set(newMat.elements);

  // Refit BVH bounds in place against the freshly-updated positions.
  // Use the cached stride-3 index buffer (refit reads 3 u32 per
  // triangle, no padding).
  const bvhNodesF32 = new Float32Array(bvh.bvhNodes.cpuData);
  refitBvhBounds(bvhNodesF32, bvh.bvhIndicesStride3, positionsF32, 4);

  // Upload the refit nodes + the affected position slice to GPU.
  // bvhNodes is small (~32 KB / 1k tris) — upload whole.
  // Positions: write only the affected byte range to honour the
  // "fast path" goal.
  const positionsByteOffset = baseVertex * STRIDE * 4; // f32 = 4 bytes
  const positionsByteLength = sliceVerts * STRIDE * 4;
  const positionsSlice = bvh.bvhPositions.cpuData.slice(
    positionsByteOffset,
    positionsByteOffset + positionsByteLength,
  );
  ctx.pipeline?.refreshBvhRefit(
    bvh.bvhNodes.cpuData.slice(0),
    { byteOffset: positionsByteOffset, data: positionsSlice },
  );

  // Reset the accumulator — temporal history is invalid because the
  // primitive moved (history pixels reference the old world position).
  ctx.pipeline?.requestAccumReset();
  // DDGI probes baked their irradiance against the old position;
  // invalidate so probes re-converge over the next STRIDE frames.
  ctx.ddgi.invalidateProbeCache();

  return { bvhBuffers: bvh };
}

/**
 * Topology-change full-rebuild path (Option (a) per items_to_fix.md A3).
 *
 * Picked over Option (b) ("`rebuildBvhLeaf(bvh, leafIndex, newTriangles)`
 * in shared-bvh") because:
 *  - three-mesh-bvh's MeshBVH constructor builds the whole tree
 *    monolithically; surgical leaf-replacement would require
 *    re-implementing SAH partitioning (Option (b) is genuinely
 *    invasive).
 *  - Topology changes are rarer than transform / material edits — the
 *    fast paths (this branch's (c) + the material branch's bytes-only
 *    re-upload) handle the common case. When topology DOES change,
 *    paying ~50 ms for a clean rebuild is the right trade vs. multi-
 *    sprint engineering on a custom partial-rebuilder.
 *
 * The CPU-side BVH builder runs; the pipeline shaders + bind-group
 * layouts + DDGI atlas + per-frame textures are preserved (no
 * `_initPipeline()` re-run). Cost: BVH build (~50 ms / 30k tris) +
 * 4 buffer destroy/recreate. No pipeline recompile.
 */
export function topologyRebuild(
  id: string,
  patch: Partial<ScenePrimitive>,
  ctx: PrimitiveUpdateContext,
): PrimitiveUpdateResult {
  const root = ctx.threeRoot;
  if (root == null) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): no THREE scene available for rebuild.`,
    );
  }

  // Apply the patch to the affected THREE.Mesh in the synthesized
  // scene so the BVH build picks up the new geometry / transform.
  // For now we support the most common topology patches:
  //   - transform (16-element Mat4)
  //   - positions, normals, uvs, tangents, indices (typed arrays from
  //     core/src/scene.ts MeshPrimitive)
  // Other fields (`instances`, `params`, `shape`, `fallbackMesh`,
  // `kind`) require a wholesale primitive replacement; throw with a
  // clear pointer so the host knows to use setScene().
  let mesh: THREE.Mesh | null = null;
  root.traverseVisible((obj) => {
    if (mesh == null && obj.name === id && (obj as THREE.Mesh).isMesh) {
      mesh = obj as THREE.Mesh;
    }
  });
  if (mesh == null) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): primitive has no THREE.Mesh in the synthesized scene.`,
    );
  }
  const meshRef = mesh as THREE.Mesh;

  const p = patch as {
    transform?: ArrayLike<number>;
    positions?: ArrayLike<number>;
    normals?: ArrayLike<number>;
    uvs?: ArrayLike<number>;
    tangents?: ArrayLike<number>;
    indices?: ArrayLike<number>;
    instances?: unknown;
    params?: unknown;
    shape?: unknown;
    fallbackMesh?: unknown;
    kind?: unknown;
  };
  for (const f of ['instances', 'params', 'shape', 'fallbackMesh', 'kind'] as const) {
    if (p[f] !== undefined) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): patching '${f}' requires a primitive ` +
        `replacement, not just an attribute update. Call setScene() with the ` +
        `modified scene instead.`,
      );
    }
  }

  if (p.transform && p.transform.length >= 16) {
    const m = new THREE.Matrix4().fromArray(Array.from(p.transform));
    meshRef.matrix.copy(m);
    meshRef.matrixWorld.copy(m);
    meshRef.matrixAutoUpdate = false;
  }
  if (p.positions) {
    meshRef.geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(Array.from(p.positions)), 3),
    );
  }
  if (p.normals) {
    meshRef.geometry.setAttribute(
      'normal',
      new THREE.BufferAttribute(new Float32Array(Array.from(p.normals)), 3),
    );
  }
  if (p.uvs) {
    meshRef.geometry.setAttribute(
      'uv',
      new THREE.BufferAttribute(new Float32Array(Array.from(p.uvs)), 2),
    );
  }
  if (p.tangents) {
    meshRef.geometry.setAttribute(
      'tangent',
      new THREE.BufferAttribute(new Float32Array(Array.from(p.tangents)), 4),
    );
  }
  if (p.indices) {
    meshRef.geometry.setIndex(
      new THREE.BufferAttribute(new Uint32Array(Array.from(p.indices)), 1),
    );
  }

  // Rebuild the BVH from the patched THREE scene. The old buffers are
  // released after the new ones are uploaded.
  const oldBuffers = ctx.bvhBuffers;
  const newBuffers = buildReSTIRSceneBVH([root], {
    primaryLightDir:       new THREE.Vector3(...ctx.primaryLightDir),
    primaryLightIntensity: ctx.primaryLightIntensity,
  });
  if (oldBuffers) disposeSceneBVH(oldBuffers);

  // Refresh the four BVH GPU buffers + (in case emissive geometry
  // changed) the emitter buffers. Pipeline shaders + bind-group
  // layouts are NOT touched.
  ctx.pipeline?.refreshBvhFullRebuild(newBuffers);
  ctx.pipeline?.updateEmitters(newBuffers);

  // Reset the accumulator + invalidate DDGI — geometry topology
  // changed, history is meaningless.
  ctx.pipeline?.requestAccumReset();
  ctx.ddgi.invalidateProbeCache();

  return { bvhBuffers: newBuffers };
}
