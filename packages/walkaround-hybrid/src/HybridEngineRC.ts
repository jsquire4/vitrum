/**
 * HybridEngineRC — Radiance Cascades subsystem for the HybridEngine path
 * (W8 sprint, started 2026-05-18).
 *
 * Owns:
 *   - One {@link RCDispatcher} per `GPUDevice`.
 *   - The 5 raw `GPUBuffer`s for the RC BVH (built from a `THREE.Scene` via
 *     `buildRCSceneBVH`, or shared from ReSTIR when `bvhMode === 'tlas'`).
 *   - The N raw cascade output `GPUBuffer`s (sized per `CASCADE_DIMS`).
 *
 * Lifecycle:
 *   1. `new RCSubsystem(device)` once per engine.
 *   2. `setScene(threeScene)` for merged single-root BVH, or
 *      `syncRestirBvhBuffers(sceneBVH)` when ReSTIR uses TLAS (C2).
 *   3. On a moving-instance / scene edit (driven by `propagateBvhToGiSubsystems`):
 *        - TLAS mode → `syncRestirBvhBuffers` re-shares ReSTIR's buffers (with
 *          a TLAS-only GPU refit when only transforms changed).
 *        - merged mode → `refitMergedInstance` re-uploads world positions + node
 *          AABBs in place WITHOUT a rebuild/teardown; it declines (→ caller
 *          rebuilds via `setScene`) only on a vertex-count / topology change.
 *      `refitCascadeBounds` cheaply re-aims the cascade probe grid either way.
 *   4. `dispatchFrame({ sunDirection, sunColor, frameSeed, triIntersectEpsilon })`
 *      per frame.
 *   5. `getCascadeC0Buffer()` returns the cascade-0 `GPUBuffer` for the
 *      shade pass to sample (W8 Phase 3 wiring).
 *   6. `dispose()` releases all GPU resources.
 *
 * Plan: `plan/w8-rc-mis-composition.md` (Phase 2 section).
 */

import type * as THREE from 'three';
import type { Scene } from '@vitrum/core';
import type { StorageBufferAttribute } from 'three/webgpu';
import { RCDispatcher, CASCADE_DIMS, type CascadeDim } from '@vitrum/walkaround-rc';
import {
  buildRCSceneBVH,
  buildRCSceneBVHFromCore,
  packCascadeMaterials,
  packCascadeMaterialsFromCore,
  type SceneBVH,
} from './rc/bvhCompute.js';
import { padTriangleIndicesToVec4 } from './ddgi/probeUpdateMaterials.js';
import { refitBvhBounds } from '@vitrum/shared-bvh';
import type { SceneBVHBuffers } from './restir/bvhCompute.js';
import {
  isRestirTlasOnlyRefit,
  makeRestirBvhSnapshot,
  type RestirBvhSnapshot,
} from './restir/restirBvhSnapshot.js';
import type { PipelineSubsystem } from './pipeline/PipelineSubsystem.js';

interface RCBVHBuffers {
  readonly bvhNodesBuf:      GPUBuffer;
  readonly bvhIndicesBuf:    GPUBuffer;
  readonly bvhPositionsBuf:  GPUBuffer;
  readonly materialsBuf:     GPUBuffer;
  readonly triMaterialIdBuf: GPUBuffer;
  readonly tlasNodesBuf?:     GPUBuffer;
  readonly tlasInstanceIndicesBuf?: GPUBuffer;
  readonly tlasBlasRootsBuf?: GPUBuffer;
  readonly tlasInstanceWorldToLocalBuf?: GPUBuffer;
  readonly tlasInstanceLocalToWorldBuf?: GPUBuffer;
}

interface RCFrameInputs {
  readonly sunDirection:        readonly [number, number, number];
  readonly sunColor:            readonly [number, number, number];
  readonly frameSeed:           number;
  readonly triIntersectEpsilon: number;
  /** Rect-area emitter NEE (2026-06-07): the main pipeline's packed
   *  `array<EmitterTri>` buffer + its triangle count, shared into RC so its
   *  probe cast can NEE-sample the emitter list. Omit/0 ⇒ RC's prior light
   *  model (sun + emissive geometry + env). */
  readonly emittersBuf?:        GPUBuffer;
  readonly emitterCount?:       number;
}

/**
 * Pack the WGSL {@link sampleCascadeC0.wgsl} `RCParams` struct (64 bytes).
 * Layout — must match the WGSL struct declaration exactly:
 *   probeOriginWorld: vec3f  (offset 0..11)
 *   rcWeight:         f32    (offset 12..15)
 *   roomSize:         vec3f  (offset 16..27)
 *   enabled:          u32    (offset 28..31)
 *   probeCount:       vec3u  (offset 32..43)
 *   raysPerProbe:     u32    (offset 44..47)
 *   rayGridSize:      u32    (offset 48..51)
 *   _pad0/1/2:        3 × u32 (offset 52..63)
 */
export function packRCParams(
  probeOriginWorld: readonly [number, number, number],
  roomSize:         readonly [number, number, number],
  probeCount:       readonly [number, number, number],
  raysPerProbe:     number,
  rcWeight:         number,
  enabled:          boolean,
): ArrayBuffer {
  const buf = new ArrayBuffer(64);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  f[0] = probeOriginWorld[0]; f[1] = probeOriginWorld[1]; f[2] = probeOriginWorld[2];
  f[3] = rcWeight;
  f[4] = roomSize[0]; f[5] = roomSize[1]; f[6] = roomSize[2];
  u[7] = enabled ? 1 : 0;
  u[8] = probeCount[0]; u[9] = probeCount[1]; u[10] = probeCount[2];
  u[11] = raysPerProbe;
  u[12] = Math.max(1, Math.round(Math.sqrt(raysPerProbe))); // rayGridSize
  // u[13..15] pad (already zero from ArrayBuffer init).
  return buf;
}

export class RCSubsystem implements PipelineSubsystem {
  private readonly _device: GPUDevice;
  private readonly _cascadeDims: readonly CascadeDim[];
  private _dispatcher: RCDispatcher | null = null;
  private _bvhBuffers: RCBVHBuffers | null = null;
  private _cascadeBufs: GPUBuffer[] | null = null;
  private _probeOriginWorld: readonly [number, number, number] | null = null;
  private _roomSize:         readonly [number, number, number] | null = null;
  private _restirSnapshot: RestirBvhSnapshot | null = null;
  private _bvhMode: 'merged' | 'tlas' = 'merged';
  private _tlasNodeCount = 0;
  /**
   * Merged-mode CPU mirrors retained from the last `setScene` build, so a
   * moving-instance refit can recompute node AABBs (`refitBvhBounds`) and
   * re-upload the updated positions + nodes WITHOUT a full SAH rebuild +
   * pipeline teardown — the merged-mode analogue of the TLAS
   * `_refitTlasGpuBuffers` fast path. Null in TLAS mode (the RC BVH buffers
   * are shared from the ReSTIR snapshot there).
   *
   * `_mergedNodesCpu` is the 8-float-per-node packed buffer (refit overwrites
   * its bounds in place); `_mergedIndicesStride3` is the stride-3 triangle
   * index buffer refit reads; `_mergedPositionsStride4` mirrors the live GPU
   * `bvhPositionsBuf` so a partial-range update can be applied + refit run
   * against the full vertex set.
   */
  private _mergedNodesCpu: Float32Array | null = null;
  private _mergedIndicesStride3: Uint32Array | null = null;
  private _mergedPositionsStride4: Float32Array | null = null;
  private _lastBvhVersion = 0;
  private _lastBlasVersion = -1;
  private _lastTlasVersion = -1;

  constructor(device: GPUDevice, cascadeDims: readonly CascadeDim[] = CASCADE_DIMS) {
    this._device = device;
    this._cascadeDims = cascadeDims;
  }

  buildRCInputs(rcWeight: number): { cascade0Buffer: GPUBuffer; paramsBytes: ArrayBuffer } | null {
    if (!this._cascadeBufs || !this._probeOriginWorld || !this._roomSize) return null;
    const c0 = this._cascadeDims[0]!;
    return {
      cascade0Buffer: this._cascadeBufs[0]!,
      paramsBytes: packRCParams(
        this._probeOriginWorld,
        this._roomSize,
        c0.probes,
        c0.rays,
        rcWeight,
        true,
      ),
    };
  }

  refitCascadeBounds(
    boundsMin: readonly [number, number, number],
    boundsMax: readonly [number, number, number],
  ): void {
    if (!this._cascadeBufs) return;
    this._probeOriginWorld = [boundsMin[0], boundsMin[1], boundsMin[2]];
    this._roomSize = [
      Math.max(boundsMax[0] - boundsMin[0], 1e-6),
      Math.max(boundsMax[1] - boundsMin[1], 1e-6),
      Math.max(boundsMax[2] - boundsMin[2], 1e-6),
    ];
  }

  /**
   * C2 — share ReSTIR BLAS/TLAS buffers for RC probe rays (multi-mesh / instanced).
   */
  syncRestirBvhBuffers(buffers: SceneBVHBuffers | null): void {
    if (buffers == null || buffers.bvhMode !== 'tlas') {
      this._restirSnapshot = null;
      this._bvhMode = 'merged';
      this._tlasNodeCount = 0;
      return;
    }
    const snap = makeRestirBvhSnapshot(buffers);
    this._restirSnapshot = snap;
    this._bvhMode = 'tlas';
    this._tlasNodeCount = snap.tlasNodeCount;
    // TLAS mode shares the ReSTIR snapshot buffers — drop any merged-mode CPU
    // mirrors so a stale merged refit can't run against TLAS-shared geometry.
    this._mergedNodesCpu = null;
    this._mergedIndicesStride3 = null;
    this._mergedPositionsStride4 = null;

    const { min, max } = snap.boundingBox;
    this._probeOriginWorld = [min.x, min.y, min.z];
    this._roomSize = [
      Math.max(max.x - min.x, 1e-6),
      Math.max(max.y - min.y, 1e-6),
      Math.max(max.z - min.z, 1e-6),
    ];

    if (snap.contentVersion !== this._lastBvhVersion) {
      const tlasOnly =
        this._bvhBuffers != null &&
        isRestirTlasOnlyRefit(snap, {
          blasContentVersion: this._lastBlasVersion,
          tlasContentVersion: this._lastTlasVersion,
        });
      if (tlasOnly && snap.tlas != null) {
        this._refitTlasGpuBuffers(snap.tlas);
      } else {
        this._disposeBvhBuffersOnly();
        this._bvhBuffers = this._uploadFromRestirSnapshot(snap);
        this._dispatcher?.dispose();
        this._dispatcher = null;
      }
      this._lastBvhVersion = snap.contentVersion;
      this._lastBlasVersion = snap.blasContentVersion;
      this._lastTlasVersion = snap.tlasContentVersion;
    }

    if (this._cascadeBufs == null) {
      this._cascadeBufs = this._allocateCascadeBuffers();
    }
    if (this._dispatcher == null) {
      this._dispatcher = new RCDispatcher(this._cascadeDims);
    }
  }

  setScene(threeScene: THREE.Scene): void {
    // Filter parity with ReSTIR's merged build (`buildReSTIRSceneBVH`, which
    // passes `filter: obj instanceof THREE.Mesh`). `buildRCSceneBVH`'s default
    // filter accepts ONLY MeshStandard/MeshPhysical materials, so absent this
    // override RC's merged vertex set would diverge from ReSTIR's whenever the
    // scene carries a non-PBR mesh (came/solder beads, MeshBasic backdrops,
    // …). That matters because `refitMergedInstance` adopts ReSTIR's
    // `bvhPositions.cpuData` directly into RC's position mirror and refits
    // RC's own nodes/indices against it — correct ONLY when both built the
    // same vertex layout (same meshes, same traverseVisible order). The
    // build-time length guard in `refitMergedInstance` would *catch* a count
    // mismatch and decline, but matching the filter makes the layout parity
    // hold by construction rather than relying on a coincidence-of-counts
    // never occurring. `isMesh` is THREE's official duck-type marker (used
    // by `buildSceneBVH`'s own sky-hide walk) — avoids a THREE value import.
    const allMeshesFilter = (obj: THREE.Object3D): boolean =>
      (obj as THREE.Mesh).isMesh === true;
    this._setSceneFromBVH(buildRCSceneBVH(threeScene, { filter: allMeshesFilter }));
  }

  /**
   * THREE-free counterpart of {@link setScene} (items_to_fix F-RC2): build the
   * merged RC scene BVH DIRECTLY from a `@vitrum/core` `Scene` via
   * {@link buildRCSceneBVHFromCore} (`mergeWorldSpaceFromCore` + the THREE-free
   * `packCascadeMaterialsFromCore` packer) — NO `vitrumSceneToThree` round-trip,
   * NO THREE material reads. Mirrors the ReSTIR-DI emitter decouple (`46a0078`)
   * + the standalone DDGI decouple (`15070cd`).
   *
   * Shares the exact same {@link _setSceneFromBVH} tail as {@link setScene}, so
   * the upload (including the F-RC1 stride-4 index pad in {@link _uploadBVH}),
   * CPU-mirror retention, cascade-bounds derivation, and dispatcher setup are
   * identical — only the BVH SOURCE differs (core merge vs THREE build). The
   * merged tri SET + world geometry are equivalent; only the BVH topology differs
   * (`buildArrayBvh` vs three-mesh-bvh), which a correct GPU traversal renders
   * identically (proven by the RC brute-force oracle post-F-RC1).
   */
  setSceneFromCore(scene: Scene): void {
    this._setSceneFromBVH(buildRCSceneBVHFromCore(scene));
  }

  /**
   * Shared merged-mode scene-set tail for {@link setScene} (THREE) and
   * {@link setSceneFromCore} (core). Given a freshly-built {@link SceneBVH},
   * reset merged/TLAS state, upload the BVH buffers (the F-RC1 stride-4 index
   * pad lives in {@link _uploadBVH}, so BOTH sources inherit it), retain the
   * stride-3/-4 CPU mirrors for the moving-instance refit fast path, derive the
   * cascade probe bounds, and (re)create the cascade buffers + dispatcher.
   */
  private _setSceneFromBVH(bvh: SceneBVH): void {
    this._restirSnapshot = null;
    this._bvhMode = 'merged';
    this._tlasNodeCount = 0;
    this._lastBvhVersion = 0;
    this._lastBlasVersion = -1;
    this._lastTlasVersion = -1;
    this._disposeSceneBuffers();
    if (this._dispatcher) {
      this._dispatcher.dispose();
      this._dispatcher = null;
    }

    this._bvhBuffers = this._uploadBVH(bvh);

    // Retain CPU mirrors for the merged-mode moving-instance refit fast path.
    // `.array` is the typed view backing each StorageBufferAttribute; we copy
    // so a later in-place refit never aliases the attribute the BVH still owns.
    this._mergedNodesCpu = new Float32Array(bvh.bvhNodes.array as Float32Array);
    this._mergedIndicesStride3 = new Uint32Array(bvh.indices.array as Uint32Array);
    this._mergedPositionsStride4 = new Float32Array(bvh.positions.array as Float32Array);

    const { min, max } = bvh.bounds;
    this._probeOriginWorld = [min.x, min.y, min.z];
    this._roomSize = [
      Math.max(max.x - min.x, 1e-6),
      Math.max(max.y - min.y, 1e-6),
      Math.max(max.z - min.z, 1e-6),
    ];

    this._cascadeBufs = this._allocateCascadeBuffers();
    this._dispatcher = new RCDispatcher(this._cascadeDims);
  }

  /**
   * PR-5.3 — merged-mode moving-instance refit WITHOUT pipeline teardown.
   *
   * The merged RC BVH bakes each mesh's world transform into its vertex
   * positions (unlike TLAS mode, where instance transforms live in separate
   * matrices that `_refitTlasGpuBuffers` can re-upload alone). So a moved
   * instance invalidates both the affected vertex positions AND every BVH
   * node AABB that bounds them. This path mirrors the TLAS refit by avoiding
   * the expensive SAH rebuild + buffer realloc + dispatcher recreation:
   *
   *   1. Apply the host-supplied updated world positions into the cached
   *      stride-4 CPU mirror (and the live `bvhPositionsBuf` via writeBuffer).
   *   2. `refitBvhBounds` recomputes every node's AABB in place from the
   *      updated positions — O(nodes + tris), ~sub-ms vs ~50 ms for a rebuild.
   *      Tree topology (split planes, child links, triangle order) is
   *      preserved, so the GPU traversal + cascade dispatch keep working.
   *   3. Re-upload the refit nodes; refit the cascade probe bounds.
   *
   * The dispatcher, cascade buffers, materials, and index buffers are all
   * untouched — only positions + nodes change, exactly like the TLAS path
   * only changes the TLAS payload.
   *
   * @param updatedPositionsStride4 the FULL merged stride-4 world-position
   *   buffer after the move (same length/layout as the build-time positions).
   *   The host re-derives it the same way ReSTIR's merged `updatePrimitive`
   *   does (old-world → local via inverse(matrixWorldAtBuild) → new-world).
   * @param boundsMin / boundsMax updated scene AABB for the cascade probe grid.
   * @returns `true` if the refit ran; `false` if RC is not in merged mode or
   *   has no retained CPU mirrors (caller should fall back to `setScene`).
   */
  refitMergedInstance(
    updatedPositionsStride4: Float32Array,
    boundsMin: readonly [number, number, number],
    boundsMax: readonly [number, number, number],
  ): boolean {
    if (this._bvhMode !== 'merged') return false;
    const bvh = this._bvhBuffers;
    if (
      bvh == null ||
      this._mergedNodesCpu == null ||
      this._mergedIndicesStride3 == null ||
      this._mergedPositionsStride4 == null
    ) {
      return false;
    }
    if (updatedPositionsStride4.length !== this._mergedPositionsStride4.length) {
      // Vertex count changed → topology change; the fast path can't apply.
      return false;
    }

    // 1. Adopt the new positions into the CPU mirror + live GPU buffer.
    const posMirror = this._mergedPositionsStride4;
    posMirror.set(updatedPositionsStride4);
    this._device.queue.writeBuffer(
      bvh.bvhPositionsBuf, 0, posMirror.buffer, posMirror.byteOffset, posMirror.byteLength,
    );

    // 2. Refit node AABBs in place from the updated positions (stride-4).
    const nodesMirror = this._mergedNodesCpu;
    refitBvhBounds(nodesMirror, this._mergedIndicesStride3, posMirror, 4);

    // 3. Re-upload the refit nodes; refit cascade probe bounds. Dispatcher +
    //    cascade buffers stay alive (no teardown).
    this._device.queue.writeBuffer(
      bvh.bvhNodesBuf, 0, nodesMirror.buffer, nodesMirror.byteOffset, nodesMirror.byteLength,
    );
    this.refitCascadeBounds(boundsMin, boundsMax);
    return true;
  }

  dispatchFrame(inputs: RCFrameInputs): void {
    if (!this._dispatcher || !this._bvhBuffers || !this._cascadeBufs ||
        !this._probeOriginWorld || !this._roomSize) {
      return;
    }
    const bvh = this._bvhBuffers;
    this._dispatcher.dispatchFrameRaw({
      device: this._device,
      bvhNodesBuf:      bvh.bvhNodesBuf,
      bvhIndicesBuf:    bvh.bvhIndicesBuf,
      bvhPositionsBuf:  bvh.bvhPositionsBuf,
      materialsBuf:     bvh.materialsBuf,
      triMaterialIdBuf: bvh.triMaterialIdBuf,
      cascadeBufs:      this._cascadeBufs,
      probeOriginWorld: this._probeOriginWorld,
      roomSize:         this._roomSize,
      sunDirection:     inputs.sunDirection,
      sunColor:         inputs.sunColor,
      frameSeed:        inputs.frameSeed,
      triIntersectEpsilon: inputs.triIntersectEpsilon,
      bvhMode:          this._bvhMode,
      tlasNodeCount:    this._tlasNodeCount,
      ...(inputs.emittersBuf != null
        ? { emittersBuf: inputs.emittersBuf, emitterCount: inputs.emitterCount ?? 0 }
        : {}),
      ...(bvh.tlasNodesBuf != null
        ? {
            tlasNodesBuf: bvh.tlasNodesBuf,
            tlasInstanceIndicesBuf: bvh.tlasInstanceIndicesBuf!,
            tlasBlasRootsBuf: bvh.tlasBlasRootsBuf!,
            tlasInstanceWorldToLocalBuf: bvh.tlasInstanceWorldToLocalBuf!,
            tlasInstanceLocalToWorldBuf: bvh.tlasInstanceLocalToWorldBuf!,
          }
        : {}),
    });
  }

  getCascadeC0Buffer(): GPUBuffer | null {
    return this._cascadeBufs?.[0] ?? null;
  }

  getCascadeC0Dims(): { probes: readonly [number, number, number]; rays: number } | null {
    if (!this._cascadeBufs) return null;
    const c0 = this._cascadeDims[0]!;
    return { probes: c0.probes, rays: c0.rays };
  }

  getCascadeGeometry(): {
    probeOriginWorld: readonly [number, number, number];
    roomSize: readonly [number, number, number];
  } | null {
    if (!this._probeOriginWorld || !this._roomSize) return null;
    return { probeOriginWorld: this._probeOriginWorld, roomSize: this._roomSize };
  }

  dispose(): void {
    this._disposeSceneBuffers();
    if (this._dispatcher) {
      this._dispatcher.dispose();
      this._dispatcher = null;
    }
  }

  private _refitTlasGpuBuffers(
    tlas: NonNullable<RestirBvhSnapshot['tlas']>,
  ): void {
    const bvh = this._bvhBuffers;
    if (bvh?.tlasNodesBuf == null) return;
    const q = this._device.queue;
    q.writeBuffer(bvh.tlasNodesBuf, 0, tlas.nodes);
    q.writeBuffer(bvh.tlasInstanceWorldToLocalBuf!, 0, tlas.worldToLocal);
    q.writeBuffer(bvh.tlasInstanceLocalToWorldBuf!, 0, tlas.localToWorld);
  }

  private _uploadFromRestirSnapshot(snap: RestirBvhSnapshot): RCBVHBuffers {
    const device = this._device;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const upload = (label: string, data: ArrayBuffer): GPUBuffer => {
      const buf = device.createBuffer({
        label,
        size: Math.max(data.byteLength, 16),
        usage,
        mappedAtCreation: true,
      });
      new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data));
      buf.unmap();
      return buf;
    };
    // T1 (THREE-decouple): pack RC's RESTIR-shared cascade materials from the
    // core MaterialSpec list (byte-identical to the THREE packer — the standalone
    // RC path already uses FromCore). Falls back to the THREE `materials` only on
    // the legacy path where coreMaterials is empty.
    const matFloats = snap.coreMaterials.length > 0
      ? packCascadeMaterialsFromCore([...snap.coreMaterials])
      : packCascadeMaterials([...snap.materials]);
    const tlas = snap.tlas;
    return {
      bvhNodesBuf: upload('rc-restir-bvh-nodes', snap.bvhNodes),
      bvhIndicesBuf: upload('rc-restir-bvh-index', snap.bvhIndex),
      bvhPositionsBuf: upload('rc-restir-bvh-positions', snap.positions),
      materialsBuf: upload('rc-restir-bvh-materials', matFloats.buffer as ArrayBuffer),
      triMaterialIdBuf: upload('rc-restir-bvh-tri-mat', snap.triMaterialIds),
      ...(tlas != null
        ? {
            tlasNodesBuf: upload('rc-restir-tlas-nodes', tlas.nodes),
            tlasInstanceIndicesBuf: upload('rc-restir-tlas-inst', tlas.instanceIndices),
            tlasBlasRootsBuf: upload('rc-restir-tlas-blas', tlas.blasRoots),
            tlasInstanceWorldToLocalBuf: upload('rc-restir-tlas-w2l', tlas.worldToLocal),
            tlasInstanceLocalToWorldBuf: upload('rc-restir-tlas-l2w', tlas.localToWorld),
          }
        : {}),
    };
  }

  private _uploadBVH(bvh: SceneBVH): RCBVHBuffers {
    // F-RC1 FIX: the merged index buffer from `buildRCSceneBVH` (→ `buildSceneBVH`)
    // is STRIDE-3 (3 u32/tri, 12 bytes, no padding — `bvhCommon.ts` always emits
    // stride-3 indices). But the RC probe shader declares
    // `rc_geom_index: array<vec4u>` and `rcTraceFirstHit` calls the vec4-storage
    // `bvhIntersectFirstHit`, which reads each entry at a 16-byte stride. Uploading
    // the raw stride-3 bytes makes the GPU read `bvhIndex.xyz[t]` from byte 16·t
    // while the data lives at 12·t — correct ONLY for tri 0 (16·0==12·0); for tri≥1
    // it reads ACROSS triangle boundaries → wrong vertices → tree-shape-dependent
    // garbage (the F-RC1 17.75 dB / 3× lit-slot divergence). Pad to stride-4 (vec4u,
    // .w=0) before upload, EXACTLY as the sibling DDGI merged path does
    // (`probeUpdateBvhBuffers.rebuildProbeBvhFromScene` → `padTriangleIndicesToVec4`,
    // which is why DDGI over the same builder renders correctly at 82 dB). RC's TLAS
    // path already feeds stride-4 (`snap.bvhIndex`), so only the merged upload needed
    // this. The stride-3 `bvh.indices` is retained unchanged for the refit fast path
    // (`refitBvhBounds` reads `indices[t*3+k]`), so the CPU mirror at `setScene` stays
    // stride-3. The index buffer is never re-uploaded on a transform refit, so padding
    // once here is sufficient.
    const idxStride4 = padTriangleIndicesToVec4(bvh.indices.array as Uint32Array);
    return {
      bvhNodesBuf:      this._uploadAttribute(bvh.bvhNodes,      'rc-bvh-nodes'),
      bvhIndicesBuf:    this._uploadTypedArray(idxStride4,        'rc-bvh-indices'),
      bvhPositionsBuf:  this._uploadAttribute(bvh.positions,      'rc-bvh-positions'),
      materialsBuf:     this._uploadAttribute(bvh.materials,      'rc-bvh-materials'),
      triMaterialIdBuf: this._uploadAttribute(bvh.triMaterialId,  'rc-bvh-tri-mat-id'),
    };
  }

  /** Upload a raw typed array as a STORAGE buffer (used for the F-RC1 stride-4
   *  index pad, which produces a fresh Uint32Array not backed by a
   *  StorageBufferAttribute). */
  private _uploadTypedArray(arr: Float32Array | Uint32Array, label: string): GPUBuffer {
    const buf = this._device.createBuffer({
      label,
      size:  Math.max(arr.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
    buf.unmap();
    return buf;
  }

  private _uploadAttribute(attr: StorageBufferAttribute, label: string): GPUBuffer {
    const arr = attr.array as Float32Array | Uint32Array;
    const buf = this._device.createBuffer({
      label,
      size:  arr.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
    buf.unmap();
    return buf;
  }

  private _allocateCascadeBuffers(): GPUBuffer[] {
    return this._cascadeDims.map((c, k) => {
      const totalRays = c.probes[0] * c.probes[1] * c.probes[2] * c.rays;
      return this._device.createBuffer({
        label: `rc-cascade-C${k}`,
        size:  totalRays * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    });
  }

  private _disposeBvhBuffersOnly(): void {
    if (!this._bvhBuffers) return;
    this._bvhBuffers.bvhNodesBuf.destroy();
    this._bvhBuffers.bvhIndicesBuf.destroy();
    this._bvhBuffers.bvhPositionsBuf.destroy();
    this._bvhBuffers.materialsBuf.destroy();
    this._bvhBuffers.triMaterialIdBuf.destroy();
    this._bvhBuffers.tlasNodesBuf?.destroy();
    this._bvhBuffers.tlasInstanceIndicesBuf?.destroy();
    this._bvhBuffers.tlasBlasRootsBuf?.destroy();
    this._bvhBuffers.tlasInstanceWorldToLocalBuf?.destroy();
    this._bvhBuffers.tlasInstanceLocalToWorldBuf?.destroy();
    this._bvhBuffers = null;
  }

  private _disposeSceneBuffers(): void {
    this._disposeBvhBuffersOnly();
    if (this._cascadeBufs) {
      for (const b of this._cascadeBufs) b.destroy();
      this._cascadeBufs = null;
    }
    this._probeOriginWorld = null;
    this._roomSize         = null;
    this._restirSnapshot = null;
    this._mergedNodesCpu = null;
    this._mergedIndicesStride3 = null;
    this._mergedPositionsStride4 = null;
    this._lastBvhVersion = 0;
    this._lastBlasVersion = -1;
    this._lastTlasVersion = -1;
  }
}
