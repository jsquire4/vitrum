/**
 * HybridEngineRC — Radiance Cascades subsystem for the HybridEngine path
 * (W8 sprint, started 2026-05-18).
 *
 * Owns:
 *   - One {@link RCDispatcher} per `GPUDevice`.
 *   - The 5 raw `GPUBuffer`s for the RC BVH (built from a core `Scene`, or
 *     shared from ReSTIR when `bvhMode === 'tlas'`).
 *   - The N raw cascade output `GPUBuffer`s (sized per `CASCADE_DIMS`).
 *
 * Lifecycle:
 *   1. `new RCSubsystem(device)` once per engine.
 *   2. `setSceneFromCore(scene)` for merged single-root BVH, or
 *      `syncRestirBvhBuffers(sceneBVH)` when ReSTIR uses TLAS (C2).
 *   3. On a moving-instance / scene edit (driven by `propagateBvhToGiSubsystems`):
 *        - TLAS mode → `syncRestirBvhBuffers` re-shares ReSTIR's buffers (with
 *          a TLAS-only GPU refit when only transforms changed).
 *        - merged mode → `refitMergedInstance` re-uploads world positions + node
 *          AABBs in place WITHOUT a rebuild/teardown; it declines (→ caller
 *          rebuilds via `setSceneFromCore`) only on a vertex-count / topology change.
 *      `refitCascadeBounds` cheaply re-aims the cascade probe grid either way.
 *   4. `dispatchFrame({ sunDirection, sunColor, frameSeed, triIntersectEpsilon })`
 *      per frame.
 *   5. `getCascadeC0Buffer()` returns the cascade-0 `GPUBuffer` for the
 *      shade pass to sample (W8 Phase 3 wiring).
 *   6. `dispose()` releases all GPU resources.
 *
 * Plan: `plan/w8-rc-mis-composition.md` (Phase 2 section).
 */

import type { MaterialSpec, Scene } from '@vitrum/core';
import {
  RCDispatcher,
  CASCADE_DIMS,
  validateCascadeDims,
  type CascadeDim,
} from '@vitrum/walkaround-rc';
import type { DDGILight } from './ddgi/types.js';
import {
  buildRCSceneBVHFromCore,
  packCascadeMaterialsFromCore,
  type StorageAttributeLike,
  type SceneBVH,
} from './rc/bvhCore.js';
import { refitBvhBounds } from '@vitrum/shared-bvh';
import type { SceneBVHBuffers } from './restir/bvhCore.js';
import { packBVHIndexWFromCore } from './restir/packingHelpers.js';
import {
  isRestirTlasOnlyRefit,
  makeRestirBvhSnapshot,
  type RestirBvhSnapshot,
} from './restir/restirBvhSnapshot.js';
import type { PipelineSubsystem } from './pipeline/PipelineSubsystem.js';
// D2.6: packing helpers live in rc/ — import + re-export for back-compat.
export {
  packRCParams,
  packRCLights,
  RC_LIGHT_CAST_SHADOW_DISABLED,
  RC_LIGHT_KIND_MASK,
  RC_LIGHTS_BUFFER_BYTES,
  RC_LIGHTS_HEADER_BYTES,
  RC_LIGHT_ENTRY_BYTES,
  RCLightBufferHeaderOffset,
  RCLightEntryOffset,
} from './rc/packingHelpers.js';
import {
  packRCParams,
  packRCLights,
  RC_LIGHTS_BUFFER_BYTES,
} from './rc/packingHelpers.js';

interface RCBVHBuffers {
  readonly bvhNodesBuf:      GPUBuffer;
  readonly bvhIndicesBuf:    GPUBuffer;
  readonly bvhPositionsBuf:  GPUBuffer;
  readonly bvhNormalsBuf:    GPUBuffer;
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
  /** Scene directional emitter castShadow:false disables RC sun visibility rays. */
  readonly sunCastShadowDisabled?: boolean;
  readonly frameSeed:           number;
  readonly triIntersectEpsilon: number;
  /** Rect-area emitter NEE (2026-06-07): the main pipeline's packed
   *  `array<EmitterTri>` buffer + its triangle count, shared into RC so its
   *  probe cast can NEE-sample the emitter list. Omit/0 ⇒ RC's prior light
   *  model (sun + emissive geometry + env). */
  readonly emittersBuf?:        GPUBuffer;
  readonly emitterCount?:       number;
  /** A7 (2026-06-10): environment equirectangular texture + sampler for the
   *  last-cascade env sample and glass transContrib env branch. When absent
   *  the dispatcher uses a 1×1 black placeholder (byte-identical env-less
   *  behaviour). The pipeline's `BvhBufferHost.envMapTextureView` /
   *  `envSampler` should be forwarded here when an HDRI is active. */
  readonly envTextureView?:     GPUTextureView | null;
  readonly envSampler?:         GPUSampler | null;
  /** Material atlas views for UV-varying material-backed emitter NEE. */
  readonly materialTextureAtlasView?: GPUTextureView | null;
  readonly materialMapMetaTextureView?: GPUTextureView | null;
  readonly bvhTangentTextureView?: GPUTextureView | null;
  /** A7 (2026-06-10): packed point/spot analytic lights buffer and count.
   *  Use `packRCLights()` to build. Omit ⇒ fixtures produce no RC radiance. */
  readonly lightsBuf?:          GPUBuffer | null;
  readonly lightCount?:         number;
}

// packRCParams, packRCLights, and all layout constants are now in rc/packingHelpers.ts
// and re-exported above (D2.6). Internal usages below continue via the local imports.

function sameVec3(
  a: readonly [number, number, number] | null,
  b: readonly [number, number, number],
): boolean {
  return a != null && a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
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
   * Merged-mode CPU mirrors retained from the last `setSceneFromCore` build, so a
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
  /** A7 (2026-06-10): GPU buffer for the packed RCLightBuffer (point/spot lights).
   *  Re-created whenever `updateLights()` is called with a different set.
   *  Null until the first `updateLights()` call; the dispatcher falls back to
   *  its internal 1040-byte placeholder when null (lightCount = 0). */
  private _lightsGpuBuf: GPUBuffer | null = null;
  private _lightsCount = 0;
  /** A7: fingerprint of the last `updateLights` payload to skip redundant GPU
   *  buffer re-creation on frames where the light set hasn't changed. Computed
   *  as a cheap JSON.stringify of the filtered (on && fixture/teaLight) entries. */
  private _lightsFingerprint = '';

  constructor(device: GPUDevice, cascadeDims: readonly CascadeDim[] = CASCADE_DIMS) {
    this._device = device;
    this._cascadeDims = validateCascadeDims(cascadeDims, 'RCSubsystem cascadeDims');
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
    const nextOrigin: [number, number, number] = [boundsMin[0], boundsMin[1], boundsMin[2]];
    const nextSize: [number, number, number] = [
      Math.max(boundsMax[0] - boundsMin[0], 1e-6),
      Math.max(boundsMax[1] - boundsMin[1], 1e-6),
      Math.max(boundsMax[2] - boundsMin[2], 1e-6),
    ];
    const changed =
      !sameVec3(this._probeOriginWorld, nextOrigin) ||
      !sameVec3(this._roomSize, nextSize);
    this._probeOriginWorld = nextOrigin;
    this._roomSize = nextSize;
    if (changed) {
      this._dispatcher?.invalidateBindings();
    }
  }

  invalidateBindings(): void {
    this._dispatcher?.invalidateBindings();
  }

  refreshMaterialsFromCore(materials: readonly MaterialSpec[]): void {
    const bvh = this._bvhBuffers;
    if (bvh == null || materials.length === 0) return;
    const matFloats = packCascadeMaterialsFromCore([...materials]);
    const next = this._uploadTypedArray(matFloats, 'rc-bvh-materials-refresh');
    bvh.materialsBuf.destroy();
    this._bvhBuffers = { ...bvh, materialsBuf: next };
    this.invalidateBindings();
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

  /**
   * Core-scene BVH path (items_to_fix F-RC2): build the
   * merged RC scene BVH DIRECTLY from a `@vitrum/core` `Scene` via
   * {@link buildRCSceneBVHFromCore} (`mergeWorldSpaceFromCore` +
   * `packCascadeMaterialsFromCore`).
   *
   * Shares the exact same {@link _setSceneFromBVH} tail so
   * the upload (including the F-RC1 stride-4 index pad in {@link _uploadBVH}),
   * CPU-mirror retention, cascade-bounds derivation, and dispatcher setup are
   * identical to the former merged path's upload and dispatch tail.
   */
  setSceneFromCore(scene: Scene): void {
    this._setSceneFromBVH(buildRCSceneBVHFromCore(scene));
  }

  /**
   * Shared merged-mode scene-set tail for {@link setSceneFromCore}. Given a freshly-built {@link SceneBVH},
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
    this._mergedNodesCpu = new Float32Array(bvh.bvhNodes.array);
    this._mergedIndicesStride3 = new Uint32Array(bvh.indices.array);
    this._mergedPositionsStride4 = new Float32Array(bvh.positions.array);

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
   *   has no retained CPU mirrors (caller should fall back to `setSceneFromCore`).
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

  /**
   * A7 (2026-06-10): upload/replace the packed point/spot analytic lights
   * buffer that RC probe rays use for direct-lighting NEE.
   *
   * Call once after `setSceneFromCore` / `syncRestirBvhBuffers` and again
   * whenever the scene's fixture list changes (matches the DDGI path:
   * `ProbeUpdatePass.setLights` → `packDDGIProbeLights`).
   *
   * The host should forward the same `DDGILight[]` that DDGI uses so the two
   * GI subsystems see the same analytic lights.  Sun-kind entries are ignored;
   * only 'fixture' and 'teaLight' entries become RC point/spot lights.
   */
  updateLights(lights: readonly DDGILight[]): void {
    // Fingerprint the active fixture/teaLight subset to avoid redundant GPU
    // buffer re-creation on frames where the light set hasn't changed.
    const active = lights.filter((l) => l.on && (l.kind === 'fixture' || l.kind === 'teaLight'));
    const fp = JSON.stringify(active);
    if (fp === this._lightsFingerprint) return;  // no change → skip GPU work
    this._lightsFingerprint = fp;

    const packed = packRCLights(lights);
    if (this._lightsGpuBuf != null) {
      this._lightsGpuBuf.destroy();
      this._lightsGpuBuf = null;
    }
    // Count only fixture / teaLight entries (same filter as packRCLights).
    this._lightsCount = Math.min(active.length, 16);
    this.invalidateBindings();
    if (this._lightsCount === 0) return;  // keep _lightsGpuBuf null; dispatcher uses placeholder
    const buf = this._device.createBuffer({
      label: 'rc-lights',
      size:  RC_LIGHTS_BUFFER_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(packed));
    buf.unmap();
    this._lightsGpuBuf = buf;
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
      bvhNormalsBuf:    bvh.bvhNormalsBuf,
      materialsBuf:     bvh.materialsBuf,
      triMaterialIdBuf: bvh.triMaterialIdBuf,
      cascadeBufs:      this._cascadeBufs,
      probeOriginWorld: this._probeOriginWorld,
      roomSize:         this._roomSize,
      sunDirection:     inputs.sunDirection,
      sunColor:         inputs.sunColor,
      sunCastShadowDisabled: inputs.sunCastShadowDisabled === true,
      frameSeed:        inputs.frameSeed,
      triIntersectEpsilon: inputs.triIntersectEpsilon,
      bvhMode:          this._bvhMode,
      tlasNodeCount:    this._tlasNodeCount,
      // A7: env texture forwarded from the main pipeline (placeholder if absent).
      ...(inputs.envTextureView != null && inputs.envSampler != null
        ? { envTextureView: inputs.envTextureView, envSampler: inputs.envSampler }
        : {}),
      ...(inputs.materialTextureAtlasView != null && inputs.materialMapMetaTextureView != null
        ? {
            materialTextureAtlasView: inputs.materialTextureAtlasView,
            materialMapMetaTextureView: inputs.materialMapMetaTextureView,
            ...(inputs.bvhTangentTextureView != null
              ? { bvhTangentTextureView: inputs.bvhTangentTextureView }
              : {}),
          }
        : {}),
      ...(inputs.emittersBuf != null
        ? { emittersBuf: inputs.emittersBuf, emitterCount: inputs.emitterCount ?? 0 }
        : {}),
      // A7: analytic lights (fixtures). Null lightsBuf falls back to dispatcher placeholder.
      ...(this._lightsGpuBuf != null
        ? { lightsBuf: this._lightsGpuBuf, lightCount: this._lightsCount }
        : (inputs.lightsBuf != null
          ? { lightsBuf: inputs.lightsBuf, lightCount: inputs.lightCount ?? 0 }
          : {})),
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
    if (this._lightsGpuBuf != null) {
      this._lightsGpuBuf.destroy();
      this._lightsGpuBuf = null;
    }
    this._lightsCount = 0;
    this._lightsFingerprint = '';  // reset so next init re-uploads correctly
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
    const matFloats = packCascadeMaterialsFromCore([...snap.coreMaterials]);
    const tlas = snap.tlas;
    return {
      bvhNodesBuf: upload('rc-restir-bvh-nodes', snap.bvhNodes),
      bvhIndicesBuf: upload('rc-restir-bvh-index', snap.bvhIndex),
      bvhPositionsBuf: upload('rc-restir-bvh-positions', snap.positions),
      bvhNormalsBuf: upload('rc-restir-bvh-normals', snap.normals),
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
    // garbage (the F-RC1 17.75 dB / 3× lit-slot divergence). Pack to stride-4
    // before upload. H37: the .w lane must NOT be zero-filled, because
    // rcTraceAny(..., skipGlass=true) reads its transmission nibble (`trans4`) to
    // skip glass. Use the same core-material payload packer as the ReSTIR/TLAS
    // path so merged and TLAS RC visibility agree. The stride-3 `bvh.indices` is
    // retained unchanged for the refit fast path (`refitBvhBounds` reads
    // `indices[t*3+k]`), so the CPU mirror at `setSceneFromCore` stays stride-3.
    // The index buffer is never re-uploaded on a transform refit, so packing once
    // here is sufficient.
    const triCount = Math.floor(bvh.indices.array.length / 3);
    const idxStride4 = packBVHIndexWFromCore(
      bvh.indices.array,
      bvh.triMaterialId.array,
      bvh.coreMaterials,
      triCount,
    );
    return {
      bvhNodesBuf:      this._uploadAttribute(bvh.bvhNodes,      'rc-bvh-nodes'),
      bvhIndicesBuf:    this._uploadTypedArray(idxStride4,        'rc-bvh-indices'),
      bvhPositionsBuf:  this._uploadAttribute(bvh.positions,      'rc-bvh-positions'),
      bvhNormalsBuf:    this._uploadAttribute(bvh.normals,        'rc-bvh-normals'),
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

  private _uploadAttribute(attr: StorageAttributeLike, label: string): GPUBuffer {
    const arr = attr.array;
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
    this._bvhBuffers.bvhNormalsBuf.destroy();
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
