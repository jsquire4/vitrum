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
 *  - material-only patches → {@link materialPatch}: re-pack affected
 *    `bvhIndex.w` / `bvh_beer` slices and partial GPU upload (no `setScene`).
 *
 * The hot-path branch design is preserved from the pre-extraction code
 * verbatim — no behaviour change.
 */

import * as THREE from 'three';
import type { MaterialSpec, MeshPrimitive, Scene, ScenePrimitive } from '@vitrum/core';
import {
  computeLocalAabb,
  computeWorldAabbForBindings,
  refitBvhBounds,
  refitTlasTransforms,
  type PrimitiveTlasBinding,
  type TlasGpuSnapshot,
} from '@vitrum/shared-bvh';
import { applyVitrumMaterialToMesh } from '@vitrum/three-bindings';
import { applyPrimitivePatchToScene } from './scenePatch.js';
import {
  buildReSTIRSceneBVHForScene,
  rebuildReSTIRSceneBVHPrimitive,
  disposeSceneBVH,
  rebuildEmitterBuffersFromSceneRoots,
} from './restir/bvhCompute.js';
import type { ReSTIRBvhMode } from './restir/bvhCompute.js';
import type { SceneBVHBuffers } from './restir/bvhCompute.js';

/** Union world AABB from merged `bvhPositions` (RC bounds after transform refit). */
function computeWorldAabbFromBvhPositions(
  bvh: SceneBVHBuffers,
): { min: readonly [number, number, number]; max: readonly [number, number, number] } | null {
  const f32 = new Float32Array(bvh.bvhPositions.cpuData);
  const vertCount = Math.floor(f32.length / 4);
  if (vertCount === 0) return null;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let v = 0; v < vertCount; v += 1) {
    const o = v * 4;
    const x = f32[o]!;
    const y = f32[o + 1]!;
    const z = f32[o + 2]!;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  if (!Number.isFinite(minX)) return null;
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

/** `sceneFromThreeJS` keys primitives by `Object3D.uuid`; tests may use `name`. */
function findMeshByPrimitiveId(root: THREE.Object3D, id: string): THREE.Mesh | null {
  let mesh: THREE.Mesh | null = null;
  root.traverseVisible((obj) => {
    if (mesh != null) return;
    if (!(obj as THREE.Mesh).isMesh) return;
    if (obj.uuid === id || obj.name === id) {
      mesh = obj as THREE.Mesh;
    }
  });
  return mesh;
}
import { repackBVHMaterialRange } from './restir/packingHelpers.js';
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
  /** Current vitrum scene snapshot — kept in sync on successful fast paths. */
  readonly lastScene: Scene;
  /** Optional pack-mode override from engine extensions. */
  readonly restirBvhModeOverride?: ReSTIRBvhMode;
}

/** Result of a primitive-update call. */
export interface PrimitiveUpdateResult {
  /** The BVH buffers that should be the engine's new `_bvhBuffers` value.
   *  - For {@link transformRefit}, unchanged from the input (the refit
   *    mutates the buffer in place).
   *  - For {@link topologyRebuild}, the freshly-built replacement; the old
   *    buffer has already been disposed inside the function. */
  readonly bvhBuffers: SceneBVHBuffers;
  /** Patched vitrum scene after a successful geometry update. */
  readonly updatedScene: Scene;
  /**
   * PR-5.5 — when set, HybridEngine should call `RCSubsystem.refitCascadeBounds`
   * instead of a full `setScene` rebuild (TLAS transform / positions refit).
   */
  readonly rcRefitBounds?: {
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
  };
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
  if (bvh.bvhMode === 'tlas') {
    const meshPatch = patch as Partial<MeshPrimitive>;
    const transformOnly =
      meshPatch.transform !== undefined &&
      meshPatch.positions === undefined &&
      meshPatch.normals === undefined &&
      meshPatch.uvs === undefined &&
      meshPatch.tangents === undefined &&
      meshPatch.indices === undefined;
    if (transformOnly && bvh.tlas != null && bvh.primitiveTlasBindings.length > 0) {
      const root = ctx.threeRoot;
      if (root == null) {
        throw new Error(
          `HybridEngine.updatePrimitive("${id}"): no THREE scene available for TLAS refit.`,
        );
      }
      const meshRef = findMeshByPrimitiveId(root, id);
      if (meshRef == null) {
        throw new Error(
          `HybridEngine.updatePrimitive("${id}"): primitive has no THREE.Mesh in the synthesized scene.`,
        );
      }
      if (meshPatch.transform && meshPatch.transform.length >= 16) {
        const m = new THREE.Matrix4().fromArray(Array.from(meshPatch.transform));
        meshRef.matrix.copy(m);
        meshRef.matrixWorld.copy(m);
        meshRef.matrixAutoUpdate = false;
      }
      const updatedScene = applyPrimitivePatchToScene(ctx.lastScene, id, {
        transform: meshPatch.transform,
      });
      const prev: TlasGpuSnapshot = {
        tlasNodes: new Uint32Array(bvh.tlas.nodes.cpuData),
        tlasInstanceIndices: new Uint32Array(bvh.tlas.instanceIndices.cpuData),
        tlasBlasRoots: new Uint32Array(bvh.tlas.blasRoots.cpuData),
        tlasInstanceWorldToLocal: new Float32Array(bvh.tlas.worldToLocal.cpuData),
      };
      const refit = refitTlasTransforms(updatedScene, bvh.primitiveTlasBindings, prev);
      if (refit.ok) {
        bvh.tlas.nodes.cpuData = refit.tlasNodes.buffer.slice(0) as ArrayBuffer;
        bvh.tlas.worldToLocal.cpuData = refit.tlasInstanceWorldToLocal.buffer.slice(0) as ArrayBuffer;
        bvh.tlas.localToWorld.cpuData = refit.tlasInstanceLocalToWorld.buffer.slice(0) as ArrayBuffer;
        ctx.pipeline?.refreshTlasRefit(
          bvh.tlas.nodes.cpuData,
          bvh.tlas.worldToLocal.cpuData,
          bvh.tlas.localToWorld.cpuData,
        );
        const range = bvh.meshVertexRanges.find((r) => r.name === id);
        if (range != null && meshPatch.transform && meshPatch.transform.length >= 16) {
          range.matrixWorldAtBuild.set(new Float32Array(meshPatch.transform));
        }
        ctx.pipeline?.requestAccumReset();
        ctx.ddgi.markInstancesDirty();
        const rcBounds = computeWorldAabbForBindings(
          updatedScene,
          bvh.primitiveTlasBindings,
        );
        return {
          bvhBuffers: bvh,
          updatedScene,
          ...(rcBounds != null ? { rcRefitBounds: rcBounds } : {}),
        };
      }
    }
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
  const meshRef = findMeshByPrimitiveId(root, id);
  if (meshRef == null) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): primitive has no THREE.Mesh in the synthesized scene.`,
    );
  }

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

  const meshPatch = patch as Partial<MeshPrimitive>;
  const updatedScene =
    meshPatch.transform !== undefined
      ? applyPrimitivePatchToScene(ctx.lastScene, id, { transform: meshPatch.transform })
      : ctx.lastScene;

  const rcBounds = computeWorldAabbFromBvhPositions(bvh);
  return {
    bvhBuffers: bvh,
    updatedScene,
    ...(rcBounds != null ? { rcRefitBounds: rcBounds } : {}),
  };
}

/**
 * Positions-only refit fast path (A3 — 2026-05-18).
 *
 * When `patch.positions` is the ONLY geometry field touched (no
 * `normals` / `uvs` / `tangents` / `indices` / `instances` / `params` /
 * `shape` / `fallbackMesh` / `kind`) AND the new positions match the
 * cached vertex count, BVH topology is preserved — only the AABB bounds
 * need to refit against the new vertex positions.
 *
 * Cost: O(triangles) refit walk (~1 ms / 30k tris on the same machine
 * that takes ~50 ms for a full SAH rebuild) + one stride-4 vertex slice
 * upload. Same fast-path shape as {@link transformRefit}.
 *
 * Falls through to {@link topologyRebuild} when:
 *  - the BVH hasn't been published yet (pipeline init in flight)
 *  - no vertex range matches the primitive id (emitter-only primitive)
 *  - the new positions length doesn't match the cached vertex count
 *    (true topology change disguised as a positions patch).
 */
export function positionsRefit(
  id: string,
  patch: Partial<ScenePrimitive>,
  ctx: PrimitiveUpdateContext,
): PrimitiveUpdateResult {
  const bvh = ctx.bvhBuffers;
  if (bvh == null) return topologyRebuild(id, patch, ctx);

  const newLocalPositions = (patch as { positions?: ArrayLike<number> }).positions;
  if (newLocalPositions == null) {
    return topologyRebuild(id, patch, ctx);
  }

  if (bvh.bvhMode === 'tlas' && bvh.tlas != null) {
    const binding = bvh.primitiveTlasBindings.find((b) => b.primitiveId === id);
    if (binding == null || binding.vertexCount === 0) {
      return topologyRebuild(id, patch, ctx);
    }
    if (newLocalPositions.length !== binding.vertexCount * 3) {
      return topologyRebuild(id, patch, ctx);
    }

    const root = ctx.threeRoot;
    if (root == null) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): no THREE scene available for TLAS positions refit.`,
      );
    }
    const meshRef = findMeshByPrimitiveId(root, id);
    if (meshRef == null) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): primitive has no THREE.Mesh in the synthesized scene.`,
      );
    }
    meshRef.geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(Array.from(newLocalPositions)), 3),
    );

    const positionsF32 = new Float32Array(bvh.bvhPositions.cpuData);
    const STRIDE = 4;
    const baseVertex = binding.vertexStart;
    const sliceVerts = binding.vertexCount;
    for (let v = 0; v < sliceVerts; v += 1) {
      const off = (baseVertex + v) * STRIDE;
      positionsF32[off + 0] = newLocalPositions[v * 3] ?? 0;
      positionsF32[off + 1] = newLocalPositions[v * 3 + 1] ?? 0;
      positionsF32[off + 2] = newLocalPositions[v * 3 + 2] ?? 0;
    }

    const localAabb = computeLocalAabb(new Float32Array(Array.from(newLocalPositions)));
    if (localAabb == null) {
      return topologyRebuild(id, patch, ctx);
    }
    const bindings: PrimitiveTlasBinding[] = bvh.primitiveTlasBindings.map((b) =>
      b.primitiveId === id
        ? {
            ...b,
            localAabbMin: localAabb.min,
            localAabbMax: localAabb.max,
          }
        : b,
    );

    const bvhNodesF32 = new Float32Array(bvh.bvhNodes.cpuData);
    refitBvhBounds(bvhNodesF32, bvh.bvhIndicesStride3, positionsF32, 4);

    const positionsByteOffset = baseVertex * STRIDE * 4;
    const positionsByteLength = sliceVerts * STRIDE * 4;
    const positionsSlice = bvh.bvhPositions.cpuData.slice(
      positionsByteOffset,
      positionsByteOffset + positionsByteLength,
    );
    ctx.pipeline?.refreshBvhRefit(
      bvh.bvhNodes.cpuData.slice(0),
      { byteOffset: positionsByteOffset, data: positionsSlice },
    );

    const meshPosPatch = patch as Partial<MeshPrimitive>;
    const posPatch: Partial<MeshPrimitive> = meshPosPatch.normals !== undefined
      ? {
          positions: new Float32Array(Array.from(newLocalPositions)),
          normals: new Float32Array(Array.from(meshPosPatch.normals)),
        }
      : { positions: new Float32Array(Array.from(newLocalPositions)) };
    const updatedScene = applyPrimitivePatchToScene(ctx.lastScene, id, posPatch);

    const prev: TlasGpuSnapshot = {
      tlasNodes: new Uint32Array(bvh.tlas.nodes.cpuData),
      tlasInstanceIndices: new Uint32Array(bvh.tlas.instanceIndices.cpuData),
      tlasBlasRoots: new Uint32Array(bvh.tlas.blasRoots.cpuData),
      tlasInstanceWorldToLocal: new Float32Array(bvh.tlas.worldToLocal.cpuData),
    };
    const refit = refitTlasTransforms(updatedScene, bindings, prev);
    if (refit.ok) {
      bvh.tlas.nodes.cpuData = refit.tlasNodes.buffer.slice(0) as ArrayBuffer;
      bvh.tlas.worldToLocal.cpuData = refit.tlasInstanceWorldToLocal.buffer.slice(0) as ArrayBuffer;
      bvh.tlas.localToWorld.cpuData = refit.tlasInstanceLocalToWorld.buffer.slice(0) as ArrayBuffer;
      ctx.pipeline?.refreshTlasRefit(
        bvh.tlas.nodes.cpuData,
        bvh.tlas.worldToLocal.cpuData,
        bvh.tlas.localToWorld.cpuData,
      );
    } else {
      return topologyRebuild(id, patch, ctx);
    }

    ctx.pipeline?.requestAccumReset();
    ctx.ddgi.invalidateProbeCache();
    const rcBounds = computeWorldAabbForBindings(updatedScene, bindings);
    const outBvh: SceneBVHBuffers = { ...bvh, primitiveTlasBindings: bindings };
    return {
      bvhBuffers: outBvh,
      updatedScene,
      ...(rcBounds != null ? { rcRefitBounds: rcBounds } : {}),
    };
  }

  const range = bvh.meshVertexRanges.find((r) => r.name === id);
  if (range == null || range.vertexCount === 0) {
    return topologyRebuild(id, patch, ctx);
  }

  // 3 floats per vertex. Vertex count mismatch ⇒ true topology change.
  if (newLocalPositions.length !== range.vertexCount * 3) {
    return topologyRebuild(id, patch, ctx);
  }

  const root = ctx.threeRoot;
  if (root == null) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): no THREE scene available for positions refit.`,
    );
  }
  const meshRef = findMeshByPrimitiveId(root, id);
  if (meshRef == null) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): primitive has no THREE.Mesh in the synthesized scene.`,
    );
  }

  // Update the THREE.Mesh's geometry so a later transformRefit picks up
  // the latest local positions. The BufferAttribute itself owns its
  // backing Float32Array, so we construct a fresh one from the patch
  // (no aliasing surprises).
  meshRef.geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(Array.from(newLocalPositions)), 3),
  );

  // The BVH stores WORLD-space positions in a stride-4 layout
  // ([x, y, z, uvPacked] per vertex). Apply the cached matrixWorldAtBuild
  // to lift the new local positions into world space, preserving the .w
  // (UV pack) lane from the existing slice.
  const matWorld = new THREE.Matrix4().fromArray(Array.from(range.matrixWorldAtBuild));
  const positionsF32 = new Float32Array(bvh.bvhPositions.cpuData);
  const STRIDE = 4;
  const baseVertex = range.vertexStart;
  const sliceVerts = range.vertexCount;
  const tmp = new THREE.Vector3();
  for (let v = 0; v < sliceVerts; v++) {
    const off = (baseVertex + v) * STRIDE;
    tmp.x = newLocalPositions[v * 3] ?? 0;
    tmp.y = newLocalPositions[v * 3 + 1] ?? 0;
    tmp.z = newLocalPositions[v * 3 + 2] ?? 0;
    tmp.applyMatrix4(matWorld);
    positionsF32[off + 0] = tmp.x;
    positionsF32[off + 1] = tmp.y;
    positionsF32[off + 2] = tmp.z;
    // .w (UV pack) preserved.
  }

  // Refit BVH bounds in place against the freshly-updated positions.
  const bvhNodesF32 = new Float32Array(bvh.bvhNodes.cpuData);
  refitBvhBounds(bvhNodesF32, bvh.bvhIndicesStride3, positionsF32, 4);

  // Upload the refit nodes + the affected position slice to GPU.
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

  // Reset the accumulator + invalidate DDGI — vertex positions changed,
  // history pixels reference the old geometry. Same invalidation cost as
  // transformRefit.
  ctx.pipeline?.requestAccumReset();
  ctx.ddgi.invalidateProbeCache();

  const meshPosPatch = patch as Partial<MeshPrimitive>;
  const posPatch: Partial<MeshPrimitive> = meshPosPatch.normals !== undefined
    ? {
        positions: new Float32Array(Array.from(newLocalPositions)),
        normals: new Float32Array(Array.from(meshPosPatch.normals)),
      }
    : { positions: new Float32Array(Array.from(newLocalPositions)) };
  const updatedScene = applyPrimitivePatchToScene(ctx.lastScene, id, posPatch);

  return { bvhBuffers: bvh, updatedScene };
}

/**
 * PR-7 — GPU LBS already wrote world positions into `bvhPositions`; sync
 * CPU refit + scene without re-uploading the position slice.
 */
export function refitSkinnedMeshAfterGpuWrite(
  id: string,
  localPositions: Float32Array,
  localNormals: Float32Array | undefined,
  ctx: PrimitiveUpdateContext,
): PrimitiveUpdateResult {
  const bvh = ctx.bvhBuffers;
  if (bvh == null) {
    throw new Error(`refitSkinnedMeshAfterGpuWrite("${id}"): BVH not ready — call setScene first.`);
  }

  const root = ctx.threeRoot;
  if (root == null) {
    throw new Error(`refitSkinnedMeshAfterGpuWrite("${id}"): no THREE scene for skin refit.`);
  }
  const meshRef = findMeshByPrimitiveId(root, id);
  if (meshRef == null) {
    throw new Error(`refitSkinnedMeshAfterGpuWrite("${id}"): mesh not found in THREE scene.`);
  }

  meshRef.geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(localPositions), 3),
  );
  if (localNormals != null) {
    meshRef.geometry.setAttribute(
      'normal',
      new THREE.BufferAttribute(new Float32Array(localNormals), 3),
    );
  }

  const posPatch: Partial<MeshPrimitive> =
    localNormals != null
      ? { positions: new Float32Array(localPositions), normals: new Float32Array(localNormals) }
      : { positions: new Float32Array(localPositions) };
  const updatedScene = applyPrimitivePatchToScene(ctx.lastScene, id, posPatch);

  if (bvh.bvhMode === 'tlas' && bvh.tlas != null) {
    const binding = bvh.primitiveTlasBindings.find((b) => b.primitiveId === id);
    if (binding == null || binding.vertexCount === 0) {
      throw new Error(`refitSkinnedMeshAfterGpuWrite("${id}"): no TLAS binding for primitive.`);
    }
    if (localPositions.length !== binding.vertexCount * 3) {
      throw new Error(
        `refitSkinnedMeshAfterGpuWrite("${id}"): expected ${binding.vertexCount * 3} floats, got ${localPositions.length}.`,
      );
    }

    const positionsF32 = new Float32Array(bvh.bvhPositions.cpuData);
    const STRIDE = 4;
    const baseVertex = binding.vertexStart;
    const sliceVerts = binding.vertexCount;
    for (let v = 0; v < sliceVerts; v += 1) {
      const off = (baseVertex + v) * STRIDE;
      positionsF32[off + 0] = localPositions[v * 3] ?? 0;
      positionsF32[off + 1] = localPositions[v * 3 + 1] ?? 0;
      positionsF32[off + 2] = localPositions[v * 3 + 2] ?? 0;
    }

    const localAabb = computeLocalAabb(localPositions);
    if (localAabb == null) {
      throw new Error(`refitSkinnedMeshAfterGpuWrite("${id}"): degenerate skinned positions.`);
    }
    const bindings: PrimitiveTlasBinding[] = bvh.primitiveTlasBindings.map((b) =>
      b.primitiveId === id
        ? { ...b, localAabbMin: localAabb.min, localAabbMax: localAabb.max }
        : b,
    );

    const bvhNodesF32 = new Float32Array(bvh.bvhNodes.cpuData);
    refitBvhBounds(bvhNodesF32, bvh.bvhIndicesStride3, positionsF32, 4);
    ctx.pipeline?.refreshBvhNodesOnly(bvh.bvhNodes.cpuData.slice(0));

    const prev: TlasGpuSnapshot = {
      tlasNodes: new Uint32Array(bvh.tlas.nodes.cpuData),
      tlasInstanceIndices: new Uint32Array(bvh.tlas.instanceIndices.cpuData),
      tlasBlasRoots: new Uint32Array(bvh.tlas.blasRoots.cpuData),
      tlasInstanceWorldToLocal: new Float32Array(bvh.tlas.worldToLocal.cpuData),
    };
    const refit = refitTlasTransforms(updatedScene, bindings, prev);
    if (!refit.ok) {
      throw new Error(`refitSkinnedMeshAfterGpuWrite("${id}"): TLAS transform refit failed.`);
    }
    bvh.tlas.nodes.cpuData = refit.tlasNodes.buffer.slice(0) as ArrayBuffer;
    bvh.tlas.worldToLocal.cpuData = refit.tlasInstanceWorldToLocal.buffer.slice(0) as ArrayBuffer;
    bvh.tlas.localToWorld.cpuData = refit.tlasInstanceLocalToWorld.buffer.slice(0) as ArrayBuffer;
    ctx.pipeline?.refreshTlasRefit(
      bvh.tlas.nodes.cpuData,
      bvh.tlas.worldToLocal.cpuData,
      bvh.tlas.localToWorld.cpuData,
    );

    ctx.pipeline?.requestAccumReset();
    ctx.ddgi.invalidateProbeCache();
    const rcBounds = computeWorldAabbForBindings(updatedScene, bindings);
    const outBvh: SceneBVHBuffers = { ...bvh, primitiveTlasBindings: bindings };
    return {
      bvhBuffers: outBvh,
      updatedScene,
      ...(rcBounds != null ? { rcRefitBounds: rcBounds } : {}),
    };
  }

  const range = bvh.meshVertexRanges.find((r) => r.name === id);
  if (range == null || range.vertexCount === 0) {
    throw new Error(`refitSkinnedMeshAfterGpuWrite("${id}"): no mesh vertex range in BVH.`);
  }
  if (localPositions.length !== range.vertexCount * 3) {
    throw new Error(
      `refitSkinnedMeshAfterGpuWrite("${id}"): expected ${range.vertexCount * 3} floats, got ${localPositions.length}.`,
    );
  }

  const matWorld = new THREE.Matrix4().fromArray(Array.from(range.matrixWorldAtBuild));
  const positionsF32 = new Float32Array(bvh.bvhPositions.cpuData);
  const STRIDE = 4;
  const baseVertex = range.vertexStart;
  const sliceVerts = range.vertexCount;
  const tmp = new THREE.Vector3();
  for (let v = 0; v < sliceVerts; v += 1) {
    const off = (baseVertex + v) * STRIDE;
    tmp.x = localPositions[v * 3] ?? 0;
    tmp.y = localPositions[v * 3 + 1] ?? 0;
    tmp.z = localPositions[v * 3 + 2] ?? 0;
    tmp.applyMatrix4(matWorld);
    positionsF32[off + 0] = tmp.x;
    positionsF32[off + 1] = tmp.y;
    positionsF32[off + 2] = tmp.z;
  }

  const bvhNodesF32 = new Float32Array(bvh.bvhNodes.cpuData);
  refitBvhBounds(bvhNodesF32, bvh.bvhIndicesStride3, positionsF32, 4);
  ctx.pipeline?.refreshBvhNodesOnly(bvh.bvhNodes.cpuData.slice(0));
  ctx.pipeline?.requestAccumReset();
  ctx.ddgi.invalidateProbeCache();

  return { bvhBuffers: bvh, updatedScene };
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
 *    fast paths handle the common case. When topology DOES change,
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
  const meshRef = findMeshByPrimitiveId(root, id);
  if (meshRef == null) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): primitive has no THREE.Mesh in the synthesized scene.`,
    );
  }

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
  const updatedScene = applyPrimitivePatchToScene(ctx.lastScene, id, patch);
  const oldBuffers = ctx.bvhBuffers;
  const bvhOpts = {
    primaryLightDir: new THREE.Vector3(...ctx.primaryLightDir),
    primaryLightIntensity: ctx.primaryLightIntensity,
    ...(ctx.restirBvhModeOverride !== undefined
      ? { bvhMode: ctx.restirBvhModeOverride }
      : {}),
  };
  let newBuffers: SceneBVHBuffers;
  if (
    oldBuffers != null &&
    oldBuffers.bvhMode === 'tlas' &&
    oldBuffers.scenePack != null
  ) {
    const rebuilt = rebuildReSTIRSceneBVHPrimitive(
      updatedScene,
      id,
      [root],
      oldBuffers,
      bvhOpts,
    );
    newBuffers =
      'ok' in rebuilt && rebuilt.ok === false
        ? buildReSTIRSceneBVHForScene(updatedScene, [root], bvhOpts)
        : (rebuilt as SceneBVHBuffers);
  } else {
    newBuffers = buildReSTIRSceneBVHForScene(updatedScene, [root], bvhOpts);
  }
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

  return { bvhBuffers: newBuffers, updatedScene };
}

const TRANSMISSION_GLASS_THRESHOLD = 0.01;

function vitrumMaterialTransmission(material: MaterialSpec | undefined): number {
  return material?.transmission ?? 0;
}

/**
 * Material-only fast path — re-pack affected triangle slices in
 * `bvhIndex` / `bvhBeerColors` and partial GPU upload (no SAH rebuild,
 * no `setScene`, no pipeline recompile).
 */
export function materialPatch(
  id: string,
  patch: Partial<ScenePrimitive>,
  ctx: PrimitiveUpdateContext,
): PrimitiveUpdateResult {
  const bvh = ctx.bvhBuffers;
  if (bvh == null || ctx.pipeline == null) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): BVH or pipeline not ready for material patch.`,
    );
  }
  if (patch.material === undefined) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): materialPatch requires patch.material.`,
    );
  }

  const range = bvh.meshVertexRanges.find((r) => r.name === id);
  if (range == null || range.triCount === 0) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): no triangle range for material patch.`,
    );
  }

  const root = ctx.threeRoot;
  if (root == null) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): no THREE scene available for material patch.`,
    );
  }

  const meshRef = findMeshByPrimitiveId(root, id);
  if (meshRef == null) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): primitive has no THREE.Mesh in the synthesized scene.`,
    );
  }

  const primIndex = ctx.lastScene.primitives.findIndex((p) => String(p.id) === id);
  const prevPrim = primIndex >= 0 ? ctx.lastScene.primitives[primIndex] : undefined;
  const prevTransmission = vitrumMaterialTransmission(
    prevPrim && 'material' in prevPrim ? prevPrim.material : undefined,
  );
  const nextTransmission = vitrumMaterialTransmission(patch.material);

  applyVitrumMaterialToMesh(meshRef, patch.material);

  const triMaterialIds = new Uint32Array(bvh.triangleMaterialIds.cpuData);
  const matIds = new Set<number>();
  for (let t = range.triStart; t < range.triStart + range.triCount; t++) {
    matIds.add(triMaterialIds[t]!);
  }
  const threeMat = meshRef.material as THREE.Material;
  for (const matId of matIds) {
    (bvh.buildMaterials as THREE.Material[])[matId] = threeMat;
  }

  const indexView = new Uint32Array(bvh.bvhIndex.cpuData);
  const beerView = new Uint32Array(bvh.bvhBeerColors.cpuData);
  repackBVHMaterialRange(
    indexView,
    beerView,
    bvh.bvhIndicesStride3,
    triMaterialIds,
    bvh.buildMaterials,
    range.triStart,
    range.triCount,
  );

  const indexByteOffset = range.triStart * 16;
  const beerByteOffset = range.triStart * 4;
  ctx.pipeline.refreshBvhMaterialSlice(
    {
      byteOffset: indexByteOffset,
      data: bvh.bvhIndex.cpuData.slice(indexByteOffset, indexByteOffset + range.triCount * 16),
    },
    {
      byteOffset: beerByteOffset,
      data: bvh.bvhBeerColors.cpuData.slice(beerByteOffset, beerByteOffset + range.triCount * 4),
    },
  );

  const crossedGlassThreshold =
    (prevTransmission <= TRANSMISSION_GLASS_THRESHOLD && nextTransmission > TRANSMISSION_GLASS_THRESHOLD)
    || (prevTransmission > TRANSMISSION_GLASS_THRESHOLD && nextTransmission <= TRANSMISSION_GLASS_THRESHOLD);

  let outBvh: SceneBVHBuffers = bvh;
  if (crossedGlassThreshold) {
    ctx.ddgi.invalidateProbeCache();
    const emitterSlice = rebuildEmitterBuffersFromSceneRoots([root], bvh, {
      primaryLightDir: new THREE.Vector3(...ctx.primaryLightDir),
      primaryLightIntensity: ctx.primaryLightIntensity,
    });
    outBvh = {
      ...bvh,
      emitters: emitterSlice.emitters,
      emitterCdf: emitterSlice.emitterCdf,
      emitterCount: emitterSlice.emitterCount,
      totalEmissivePower: emitterSlice.totalEmissivePower,
    };
    ctx.pipeline.updateEmitters(outBvh);
  }

  ctx.pipeline.requestAccumReset();

  const updatedScene = applyPrimitivePatchToScene(ctx.lastScene, id, patch);
  return { bvhBuffers: outBvh, updatedScene };
}
