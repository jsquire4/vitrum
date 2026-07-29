/**
 * HybridEngineRC — Radiance Cascades subsystem for the HybridEngine path
 * (W8 sprint, started 2026-05-18).
 *
 * Owns:
 *   - One {@link RCDispatcher} per `GPUDevice`.
 *   - RC-specific materials/TLAS adapter buffers. In the integrated engine,
 *     the large BLAS geometry windows are borrowed directly from the main
 *     scene arena; standalone use retains an owned merged-BVH fallback.
 *   - The N raw cascade output `GPUBuffer`s (sized per `CASCADE_DIMS`).
 *
 * Lifecycle:
 *   1. `new RCSubsystem(device)` once per engine.
 *   2. `setSceneFromCore(scene)` for merged single-root BVH, or
 *      `syncRestirBvhBuffers(sceneBVH)` when ReSTIR uses TLAS (C2).
 *   3. On a moving-instance / scene edit (driven by `propagateBvhToGiSubsystems`):
 *        - TLAS mode → `syncRestirBvhBuffers` transactionally refreshes the
 *          compact RC adapter while retaining borrowed main-arena geometry.
 *        - merged mode → `refitMergedInstance` uploads candidate world-position
 *          and node-AABB buffers without a rebuild/teardown; it declines (→ caller
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

import type { EngineWarning, Scene } from '@vitrum/core';
import {
  RCDispatcher,
  CASCADE_DIMS,
  RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET,
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
  makeRestirBvhSnapshot,
  isRestirTlasOnlyRefit,
  type RestirBvhSnapshot,
} from './restir/restirBvhSnapshot.js';
import type { PipelineSubsystem } from './pipeline/PipelineSubsystem.js';
import type { SceneGeometryBufferBindings } from './pipeline/BvhBufferHost.js';
import {
  rethrowWithSceneMutationCleanup,
  runSceneMutationCleanups,
  type PreparedSceneMutation,
  type SceneMutationCleanup,
} from './SceneMutationTransaction.js';
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
  rcLightsBufferByteLength,
} from './rc/packingHelpers.js';

interface RCBVHBuffers {
  readonly bvhNodesBuf?:      GPUBuffer;
  readonly bvhIndicesBuf?:    GPUBuffer;
  readonly bvhPositionsBuf?:  GPUBuffer;
  readonly bvhNormalsBuf?:    GPUBuffer;
  readonly materialsBuf:     GPUBuffer;
  readonly triMaterialIdBuf: GPUBuffer;
  readonly tlasNodesBuf?:     GPUBuffer;
  readonly tlasInstanceIndicesBuf?: GPUBuffer;
  readonly tlasBlasRootsBuf?: GPUBuffer;
  readonly tlasInstanceWorldToLocalBuf?: GPUBuffer;
  readonly tlasInstanceLocalToWorldBuf?: GPUBuffer;
}

type RCBvhReplacementKind =
  | 'none'
  | 'materials'
  | 'merged-refit'
  | 'tlas-refit'
  | 'full';

function rcReplacementCleanups(
  buffers: RCBVHBuffers | null,
  kind: RCBvhReplacementKind,
): SceneMutationCleanup[] {
  if (buffers == null || kind === 'none') return [];
  let owned: Array<GPUBuffer | undefined>;
  if (kind === 'materials') {
    owned = [buffers.materialsBuf];
  } else if (kind === 'merged-refit') {
    owned = [buffers.bvhPositionsBuf, buffers.bvhNodesBuf];
  } else if (kind === 'tlas-refit') {
    owned = [
      buffers.tlasNodesBuf,
      buffers.tlasInstanceIndicesBuf,
      buffers.tlasBlasRootsBuf,
      buffers.tlasInstanceWorldToLocalBuf,
      buffers.tlasInstanceLocalToWorldBuf,
    ];
  } else {
    owned = [
      buffers.bvhNodesBuf,
      buffers.bvhIndicesBuf,
      buffers.bvhPositionsBuf,
      buffers.bvhNormalsBuf,
      buffers.materialsBuf,
      buffers.triMaterialIdBuf,
      buffers.tlasNodesBuf,
      buffers.tlasInstanceIndicesBuf,
      buffers.tlasBlasRootsBuf,
      buffers.tlasInstanceWorldToLocalBuf,
      buffers.tlasInstanceLocalToWorldBuf,
    ];
  }
  return owned.flatMap((buffer) => buffer == null ? [] : [() => buffer.destroy()]);
}

function destroyGpuBuffer(buffer: GPUBuffer | undefined): void {
  try { buffer?.destroy(); } catch { /* release remaining independently-owned buffers */ }
}

function destroyRCBVHBuffers(buffers: RCBVHBuffers | null): void {
  if (buffers == null) return;
  destroyGpuBuffer(buffers.bvhNodesBuf);
  destroyGpuBuffer(buffers.bvhIndicesBuf);
  destroyGpuBuffer(buffers.bvhPositionsBuf);
  destroyGpuBuffer(buffers.bvhNormalsBuf);
  destroyGpuBuffer(buffers.materialsBuf);
  destroyGpuBuffer(buffers.triMaterialIdBuf);
  destroyRCTLASBuffers(buffers);
}

function destroyRCTLASBuffers(buffers: RCBVHBuffers | null): void {
  if (buffers == null) return;
  destroyGpuBuffer(buffers.tlasNodesBuf);
  destroyGpuBuffer(buffers.tlasInstanceIndicesBuf);
  destroyGpuBuffer(buffers.tlasBlasRootsBuf);
  destroyGpuBuffer(buffers.tlasInstanceWorldToLocalBuf);
  destroyGpuBuffer(buffers.tlasInstanceLocalToWorldBuf);
}

function destroyGpuBuffers(buffers: readonly GPUBuffer[] | null): void {
  if (buffers == null) return;
  for (const buffer of buffers) destroyGpuBuffer(buffer);
}

interface RCFrameInputs {
  readonly sunDirection:        readonly [number, number, number];
  readonly sunColor:            readonly [number, number, number];
  /** Scene directional emitter castShadow:false disables RC sun visibility rays. */
  readonly sunCastShadowDisabled?: boolean;
  /** Finite directional emitter cone radius in radians. */
  readonly sunAngularRadius?: number;
  readonly frameSeed:           number;
  readonly triIntersectEpsilon: number;
  /** Rect-area emitter NEE (2026-06-07): the main pipeline's packed
   *  `array<EmitterTri>` buffer + its triangle count, shared into RC so its
   *  probe cast can NEE-sample the emitter list. Omit/0 ⇒ RC's prior light
   *  model (sun + emissive geometry + env). */
  readonly emittersBuf?:        GPUBuffer;
  readonly emittersOffset?:     number;
  readonly emittersSize?:       number;
  readonly emitterDataOffset?: number;
  readonly emitterAliasOffset?: number;
  readonly emitterCount?:       number;
  /** A7 (2026-06-10): environment equirectangular texture + sampler for the
   *  last-cascade env sample and glass transContrib env branch. When absent
   *  the dispatcher binds a 1×1 black placeholder; the explicit directional
   *  flag below decides whether misses sample it or use scalar sky. */
  readonly envTextureView?:     GPUTextureView | null;
  readonly envSampler?:         GPUSampler | null;
  /** H6 world-to-unrotated-map Y rotation, in radians. */
  readonly envRotationY?:       number;
  /** Linear radiance multiplier; the bound map remains unit-intensity. */
  readonly envIntensity?:       number;
  /** Main/ReSTIR-equivalent scalar sky radiance used when no directional map is
   *  active. This is `skyTint * skyIrradiance`, already in radiance units. */
  readonly scalarSkyRadiance?: readonly [number, number, number];
  /** Explicit directional-payload state. A bindable black placeholder does
   *  not count as a directional environment. */
  readonly hasDirectionalEnvironment?: boolean;
  /** Material atlas views for UV-varying material-backed emitter NEE. */
  readonly materialTextureAtlasView?: GPUTextureView | null;
  readonly materialMapMetaTextureView?: GPUTextureView | null;
  readonly bvhTangentTextureView?: GPUTextureView | null;
  readonly bvhVertexColorTextureView?: GPUTextureView | null;
  /** Runtime punctual/directional lights buffer and count. */
  readonly lightsBuf?:          GPUBuffer | null;
  readonly lightCount?:         number;
  /** Main pipeline scene-arena ranges; preferred over RC-owned geometry copies. */
  readonly sceneGeometryBindings?: SceneGeometryBufferBindings | null;
}

// packRCParams, packRCLights, and all layout constants are now in rc/packingHelpers.ts
// and re-exported above (D2.6). Internal usages below continue via the local imports.

function sameVec3(
  a: readonly [number, number, number] | null,
  b: readonly [number, number, number],
): boolean {
  return a != null && a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

function sameBufferBinding(
  a: GPUBufferBinding,
  b: GPUBufferBinding,
): boolean {
  return a.buffer === b.buffer &&
    Number(a.offset ?? 0) === Number(b.offset ?? 0) &&
    Number(a.size ?? (a.buffer.size - Number(a.offset ?? 0))) ===
      Number(b.size ?? (b.buffer.size - Number(b.offset ?? 0)));
}

function sameSceneGeometryBindings(
  a: SceneGeometryBufferBindings | null,
  b: SceneGeometryBufferBindings,
): boolean {
  return a != null &&
    sameBufferBinding(a.bvhNodes, b.bvhNodes) &&
    sameBufferBinding(a.bvhIndices, b.bvhIndices) &&
    sameBufferBinding(a.bvhPositions, b.bvhPositions) &&
    sameBufferBinding(a.bvhNormals, b.bvhNormals);
}

function stride4TriangleIndicesToStride3(bytes: ArrayBuffer): Uint32Array {
  const source = new Uint32Array(bytes);
  if (source.length % 4 !== 0) {
    throw new Error('[RCSubsystem] shared BVH index stream is not vec4u-strided.');
  }
  const out = new Uint32Array((source.length / 4) * 3);
  for (let tri = 0; tri < source.length / 4; tri += 1) {
    out[tri * 3] = source[tri * 4]!;
    out[tri * 3 + 1] = source[tri * 4 + 1]!;
    out[tri * 3 + 2] = source[tri * 4 + 2]!;
  }
  return out;
}

/** @internal Directionals have exactly one RC owner. */
export function resolveRCLegacySunColor(
  aliasedDirectionalCount: number,
  fallback: readonly [number, number, number],
): readonly [number, number, number] {
  return aliasedDirectionalCount > 0 ? [0, 0, 0] : fallback;
}

export class RCSubsystem implements PipelineSubsystem {
  private readonly _device: GPUDevice;
  private readonly _cascadeDims: readonly CascadeDim[];
  private readonly _onWarning: ((warning: EngineWarning) => void) | null;
  private readonly _warningKeys = new Set<string>();
  private readonly _transmittedInterfaceBudget: number;
  private _dispatcher: RCDispatcher | null = null;
  private _bvhBuffers: RCBVHBuffers | null = null;
  /** Borrowed main-pipeline arena windows; never destroyed by RC. */
  private _sharedGeometryBindings: SceneGeometryBufferBindings | null = null;
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
  private _lastMaterialVersion = -1;
  /** GPU buffer for runtime RCLight records plus their alias table.
   *  Re-created whenever `updateLights()` is called with a different set.
   *  Null until the first `updateLights()` call; the dispatcher falls back to
   *  its internal header-only placeholder when null (lightCount = 0). */
  private _lightsGpuBuf: GPUBuffer | null = null;
  private _lightsCount = 0;
  private _lightsDirectionalCount = 0;
  /** A7: fingerprint of the last `updateLights` payload to skip redundant GPU
   *  buffer re-creation on frames where the light set hasn't changed. Computed
   *  as a cheap JSON.stringify of the filtered (on && fixture/teaLight) entries. */
  private _lightsFingerprint = '';

  /** True once RC has adopted the main pipeline's canonical geometry arena. */
  get sharesSceneGeometry(): boolean {
    return this._sharedGeometryBindings != null;
  }

  constructor(
    device: GPUDevice,
    cascadeDims: readonly CascadeDim[] | undefined = CASCADE_DIMS,
    diagnostics: {
      readonly onWarning?: (warning: EngineWarning) => void;
      readonly transmittedInterfaceBudget?: number;
    } = {},
  ) {
    this._device = device;
    this._cascadeDims = validateCascadeDims(cascadeDims ?? CASCADE_DIMS, 'RCSubsystem cascadeDims');
    this._onWarning = diagnostics.onWarning ?? null;
    this._transmittedInterfaceBudget = diagnostics.transmittedInterfaceBudget
      ?? RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET;
  }

  private _warnOnce(key: string, warning: EngineWarning): void {
    if (this._warningKeys.has(key)) return;
    this._warningKeys.add(key);
    try {
      this._onWarning?.(warning);
    } catch {
      // Diagnostic callbacks never interrupt rendering or resource cleanup.
    }
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

  /**
   * Stage invalidation of dispatcher-cached bind groups for a resource
   * generation swap. Commit and rollback intentionally retain the currently
   * renderable cache; only successful transaction retirement invalidates it.
   */
  prepareBindingInvalidation(): PreparedSceneMutation {
    let committed = false;
    let closed = false;
    return {
      commit: () => {
        if (closed) return;
        committed = true;
      },
      rollback: () => {
        closed = true;
      },
      finalize: () => {
        if (closed) return;
        closed = true;
        if (committed) {
          // Retirement follows the irreversible publication point. A stale
          // cache entry is already generation-keyed and cannot be reused, so
          // cache cleanup is best-effort and must not make a successful scene
          // mutation appear to have failed.
          try {
            this._dispatcher?.invalidateBindings();
          } catch {
            // Intentionally ignored after commit.
          }
        }
      },
    };
  }

  /**
   * C2 — share ReSTIR BLAS/TLAS buffers for RC probe rays (multi-mesh / instanced).
   */
  prepareSceneMutation(
    buffers: SceneBVHBuffers,
    scene: Scene | undefined,
    options: {
      readonly geometryChanged: boolean;
      readonly refreshMaterials: boolean;
      readonly allowMergedRefit: boolean;
      readonly rcRefitBounds?: {
        readonly min: readonly [number, number, number];
        readonly max: readonly [number, number, number];
      };
    },
  ): PreparedSceneMutation {
    const previous = {
      bvhBuffers: this._bvhBuffers,
      restirSnapshot: this._restirSnapshot,
      bvhMode: this._bvhMode,
      tlasNodeCount: this._tlasNodeCount,
      mergedNodesCpu: this._mergedNodesCpu,
      mergedIndicesStride3: this._mergedIndicesStride3,
      mergedPositionsStride4: this._mergedPositionsStride4,
      probeOriginWorld: this._probeOriginWorld,
      roomSize: this._roomSize,
      lastBvhVersion: this._lastBvhVersion,
      lastBlasVersion: this._lastBlasVersion,
      lastTlasVersion: this._lastTlasVersion,
      lastMaterialVersion: this._lastMaterialVersion,
    };
    let nextBvh = previous.bvhBuffers;
    let nextSnapshot = previous.restirSnapshot;
    let nextMode = previous.bvhMode;
    let nextTlasNodeCount = previous.tlasNodeCount;
    let nextNodesCpu = previous.mergedNodesCpu;
    let nextIndicesCpu = previous.mergedIndicesStride3;
    let nextPositionsCpu = previous.mergedPositionsStride4;
    let nextOrigin = previous.probeOriginWorld;
    let nextRoomSize = previous.roomSize;
    let nextBvhVersion = previous.lastBvhVersion;
    let nextBlasVersion = previous.lastBlasVersion;
    let nextTlasVersion = previous.lastTlasVersion;
    let nextMaterialVersion = previous.lastMaterialVersion;
    let replacementKind: RCBvhReplacementKind = 'none';

    try {
      if (options.geometryChanged && this._sharedGeometryBindings != null) {
        // The integrated engine owns geometry in BvhBufferHost. Keep RC's
        // transactional material/TLAS adapter in sync, but never manufacture a
        // second BLAS just because an incremental mutation changed its bytes.
        // BvhBufferHost publishes any replacement arena later in the same
        // transaction; dispatchFrame adopts those fresh borrowed ranges before
        // the next probe dispatch.
        const snap = makeRestirBvhSnapshot(buffers, scene);
        const needsReplacement =
          previous.bvhBuffers == null ||
          snap.contentVersion !== previous.lastBvhVersion;
        const tlasOnly =
          needsReplacement &&
          previous.bvhBuffers != null &&
          snap.tlas != null &&
          isRestirTlasOnlyRefit(snap, {
            blasContentVersion: previous.lastBlasVersion,
            tlasContentVersion: previous.lastTlasVersion,
            materialContentVersion: previous.lastMaterialVersion,
          });
        if (needsReplacement) {
          nextBvh = tlasOnly
            ? { ...previous.bvhBuffers!, ...this._uploadRestirTlasBuffers(snap.tlas) }
            : this._uploadFromRestirSnapshot(snap, false);
          replacementKind = tlasOnly ? 'tlas-refit' : 'full';
        }
        nextSnapshot = snap;
        nextMode = snap.bvhMode;
        nextTlasNodeCount = snap.tlasNodeCount;
        if (snap.bvhMode === 'merged') {
          nextNodesCpu = new Float32Array(snap.bvhNodes);
          nextIndicesCpu = stride4TriangleIndicesToStride3(snap.bvhIndex);
          nextPositionsCpu = new Float32Array(snap.positions);
        } else {
          nextNodesCpu = null;
          nextIndicesCpu = null;
          nextPositionsCpu = null;
        }
        const { min, max } = snap.boundingBox;
        nextOrigin = [min.x, min.y, min.z];
        nextRoomSize = [
          Math.max(max.x - min.x, 1e-6),
          Math.max(max.y - min.y, 1e-6),
          Math.max(max.z - min.z, 1e-6),
        ];
        nextBvhVersion = snap.contentVersion;
        nextBlasVersion = snap.blasContentVersion;
        nextTlasVersion = snap.tlasContentVersion;
        nextMaterialVersion = snap.materialContentVersion;
      } else if (options.geometryChanged && buffers.bvhMode === 'tlas') {
        const snap = makeRestirBvhSnapshot(buffers);
        const needsReplacement =
          previous.bvhBuffers == null || snap.contentVersion !== previous.lastBvhVersion;
        const tlasOnly =
          needsReplacement &&
          previous.bvhBuffers != null &&
          snap.tlas != null &&
          isRestirTlasOnlyRefit(snap, {
            blasContentVersion: previous.lastBlasVersion,
            tlasContentVersion: previous.lastTlasVersion,
            materialContentVersion: previous.lastMaterialVersion,
          });
          if (needsReplacement) {
            nextBvh = tlasOnly
              ? { ...previous.bvhBuffers!, ...this._uploadRestirTlasBuffers(snap.tlas) }
            : this._uploadFromRestirSnapshot(snap);
          replacementKind = tlasOnly ? 'tlas-refit' : 'full';
        }
        nextSnapshot = snap;
        nextMode = 'tlas';
        nextTlasNodeCount = snap.tlasNodeCount;
        nextNodesCpu = null;
        nextIndicesCpu = null;
        nextPositionsCpu = null;
        const { min, max } = snap.boundingBox;
        nextOrigin = [min.x, min.y, min.z];
        nextRoomSize = [
          Math.max(max.x - min.x, 1e-6),
          Math.max(max.y - min.y, 1e-6),
          Math.max(max.z - min.z, 1e-6),
        ];
        nextBvhVersion = snap.contentVersion;
        nextBlasVersion = snap.blasContentVersion;
        nextTlasVersion = snap.tlasContentVersion;
        nextMaterialVersion = snap.materialContentVersion;
      } else if (options.geometryChanged && buffers.bvhMode === 'merged') {
        const positions = new Float32Array(buffers.bvhPositions.cpuData);
        const canRefit =
          options.allowMergedRefit &&
          previous.bvhMode === 'merged' &&
          previous.bvhBuffers != null &&
          previous.mergedNodesCpu != null &&
          previous.mergedIndicesStride3 != null &&
          previous.mergedPositionsStride4 != null &&
          positions.length === previous.mergedPositionsStride4.length;
        if (canRefit) {
          const nodes = new Float32Array(previous.mergedNodesCpu!);
          refitBvhBounds(nodes, previous.mergedIndicesStride3!, positions, 4);
          const posGpu = this._uploadTypedArray(positions, 'rc-bvh-positions-refit');
          let nodesGpu: GPUBuffer | null = null;
          try {
            nodesGpu = this._uploadTypedArray(nodes, 'rc-bvh-nodes-refit');
          } catch (error) {
            rethrowWithSceneMutationCleanup(
              error,
              [() => posGpu.destroy()],
              'RC merged-refit preparation failed and cleanup also failed',
            );
          }
          nextBvh = {
            ...previous.bvhBuffers!,
            bvhPositionsBuf: posGpu,
            bvhNodesBuf: nodesGpu,
          };
          nextPositionsCpu = positions;
          nextNodesCpu = nodes;
          replacementKind = 'merged-refit';
        } else if (scene != null) {
          const built = buildRCSceneBVHFromCore(scene);
          nextBvh = this._uploadBVH(built);
          nextNodesCpu = new Float32Array(built.bvhNodes.array);
          nextIndicesCpu = new Uint32Array(built.indices.array);
          nextPositionsCpu = new Float32Array(built.positions.array);
          replacementKind = 'full';
        }
        nextSnapshot = null;
        nextMode = 'merged';
        nextTlasNodeCount = 0;
        nextBvhVersion = 0;
        nextBlasVersion = -1;
        nextTlasVersion = -1;
        nextMaterialVersion = -1;
      } else if (options.refreshMaterials && previous.bvhBuffers != null) {
        const matFloats = packCascadeMaterialsFromCore([...buffers.coreMaterials]);
        const materialsBuf = this._uploadTypedArray(matFloats, 'rc-bvh-materials-refresh');
        nextBvh = { ...previous.bvhBuffers, materialsBuf };
        replacementKind = 'materials';
        if (buffers.bvhMode === 'tlas' || this._sharedGeometryBindings != null) {
          const snap = makeRestirBvhSnapshot(buffers, scene);
          nextSnapshot = snap;
          nextBvhVersion = snap.contentVersion;
          nextBlasVersion = snap.blasContentVersion;
          nextTlasVersion = snap.tlasContentVersion;
          nextMaterialVersion = snap.materialContentVersion;
        }
      }
    } catch (error) {
      rethrowWithSceneMutationCleanup(
        error,
        rcReplacementCleanups(nextBvh, replacementKind),
        'RC scene-mutation preparation failed and cleanup also failed',
      );
    }

    if (options.rcRefitBounds) {
      const { min, max } = options.rcRefitBounds;
      nextOrigin = [min[0], min[1], min[2]];
      nextRoomSize = [
        Math.max(max[0] - min[0], 1e-6),
        Math.max(max[1] - min[1], 1e-6),
        Math.max(max[2] - min[2], 1e-6),
      ];
    }

    const releaseCandidate = (): void => {
      runSceneMutationCleanups(
        rcReplacementCleanups(nextBvh, replacementKind),
        'RC scene-mutation candidate cleanup failed',
      );
    };
    const releasePrevious = (): void => {
      runSceneMutationCleanups(
        rcReplacementCleanups(previous.bvhBuffers, replacementKind),
        'RC scene-mutation retirement failed',
      );
    };

    let committed = false;
    let closed = false;
    return {
      commit: () => {
        if (closed || committed) return;
        this._bvhBuffers = nextBvh;
        this._restirSnapshot = nextSnapshot;
        this._bvhMode = nextMode;
        this._tlasNodeCount = nextTlasNodeCount;
        this._mergedNodesCpu = nextNodesCpu;
        this._mergedIndicesStride3 = nextIndicesCpu;
        this._mergedPositionsStride4 = nextPositionsCpu;
        this._probeOriginWorld = nextOrigin;
        this._roomSize = nextRoomSize;
        this._lastBvhVersion = nextBvhVersion;
        this._lastBlasVersion = nextBlasVersion;
        this._lastTlasVersion = nextTlasVersion;
        this._lastMaterialVersion = nextMaterialVersion;
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        closed = true;
        if (committed) {
          this._bvhBuffers = previous.bvhBuffers;
          this._restirSnapshot = previous.restirSnapshot;
          this._bvhMode = previous.bvhMode;
          this._tlasNodeCount = previous.tlasNodeCount;
          this._mergedNodesCpu = previous.mergedNodesCpu;
          this._mergedIndicesStride3 = previous.mergedIndicesStride3;
          this._mergedPositionsStride4 = previous.mergedPositionsStride4;
          this._probeOriginWorld = previous.probeOriginWorld;
          this._roomSize = previous.roomSize;
          this._lastBvhVersion = previous.lastBvhVersion;
          this._lastBlasVersion = previous.lastBlasVersion;
          this._lastTlasVersion = previous.lastTlasVersion;
          this._lastMaterialVersion = previous.lastMaterialVersion;
        }
        releaseCandidate();
      },
      finalize: () => {
        if (closed) return;
        closed = true;
        if (committed) {
          runSceneMutationCleanups(
            [
              () => {
                if (
                  replacementKind !== 'none' &&
                  replacementKind !== 'materials' &&
                  replacementKind !== 'tlas-refit'
                ) this.invalidateBindings();
              },
              releasePrevious,
            ],
            'RC scene-mutation retirement failed',
          );
        } else {
          releaseCandidate();
        }
      },
    };
  }
  syncRestirBvhBuffers(
    buffers: SceneBVHBuffers | null,
    sharedGeometryBindings?: SceneGeometryBufferBindings | null,
  ): void {
    if (buffers == null) {
      this._disposeBvhBuffersOnly();
      this._restirSnapshot = null;
      this._sharedGeometryBindings = null;
      this._bvhMode = 'merged';
      this._tlasNodeCount = 0;
      this._probeOriginWorld = null;
      this._roomSize = null;
      this._mergedNodesCpu = null;
      this._mergedIndicesStride3 = null;
      this._mergedPositionsStride4 = null;
      this._lastBvhVersion = 0;
      this._lastBlasVersion = -1;
      this._lastTlasVersion = -1;
      this._lastMaterialVersion = -1;
      this.invalidateBindings();
      return;
    }
    const effectiveSharedGeometry = sharedGeometryBindings === undefined
      ? this._sharedGeometryBindings
      : sharedGeometryBindings;
    // Historical standalone merged mode still uses setSceneFromCore(). The
    // integrated engine supplies its canonical arena ranges and uses this
    // unified snapshot path for both merged and TLAS scenes.
    if (buffers.bvhMode !== 'tlas' && effectiveSharedGeometry == null) {
      this._restirSnapshot = null;
      this._bvhMode = 'merged';
      this._tlasNodeCount = 0;
      return;
    }
    const snap = makeRestirBvhSnapshot(buffers);
    const sharedGeometryChanged =
      (this._sharedGeometryBindings == null) !== (effectiveSharedGeometry == null) ||
      (
        effectiveSharedGeometry != null &&
        !sameSceneGeometryBindings(this._sharedGeometryBindings, effectiveSharedGeometry)
      );
    const needsReplacement =
      this._bvhBuffers == null || sharedGeometryChanged ||
      snap.contentVersion !== this._lastBvhVersion;
    const tlasOnly =
      needsReplacement &&
      !sharedGeometryChanged &&
      this._bvhBuffers != null &&
      snap.tlas != null &&
      isRestirTlasOnlyRefit(snap, {
        blasContentVersion: this._lastBlasVersion,
        tlasContentVersion: this._lastTlasVersion,
        materialContentVersion: this._lastMaterialVersion,
      });
    let replacement: RCBVHBuffers | null = null;
    let cascadeCandidate: GPUBuffer[] | null = null;
    let dispatcherCandidate = this._dispatcher;
    try {
      if (needsReplacement) {
        replacement = tlasOnly
          ? { ...this._bvhBuffers!, ...this._uploadRestirTlasBuffers(snap.tlas) }
          : this._uploadFromRestirSnapshot(
              snap,
              effectiveSharedGeometry == null,
            );
      }
      if (this._cascadeBufs == null) cascadeCandidate = this._allocateCascadeBuffers();
      if (dispatcherCandidate == null) {
        dispatcherCandidate = new RCDispatcher(this._cascadeDims);
      }
    } catch (error) {
      if (tlasOnly) destroyRCTLASBuffers(replacement);
      else destroyRCBVHBuffers(replacement);
      destroyGpuBuffers(cascadeCandidate);
      if (dispatcherCandidate !== this._dispatcher) dispatcherCandidate?.dispose();
      throw error;
    }

    const previousBvh = this._bvhBuffers;
    if (needsReplacement) this._bvhBuffers = replacement;
    if (cascadeCandidate != null) this._cascadeBufs = cascadeCandidate;

    this._restirSnapshot = snap;
    this._sharedGeometryBindings = effectiveSharedGeometry;
    this._dispatcher = dispatcherCandidate;
    this._bvhMode = snap.bvhMode;
    this._tlasNodeCount = snap.tlasNodeCount;
    if (snap.bvhMode === 'merged') {
      this._mergedNodesCpu = new Float32Array(snap.bvhNodes);
      this._mergedIndicesStride3 = stride4TriangleIndicesToStride3(snap.bvhIndex);
      this._mergedPositionsStride4 = new Float32Array(snap.positions);
    } else {
      this._mergedNodesCpu = null;
      this._mergedIndicesStride3 = null;
      this._mergedPositionsStride4 = null;
    }

    const { min, max } = snap.boundingBox;
    this._probeOriginWorld = [min.x, min.y, min.z];
    this._roomSize = [
      Math.max(max.x - min.x, 1e-6),
      Math.max(max.y - min.y, 1e-6),
      Math.max(max.z - min.z, 1e-6),
    ];
    this._lastBvhVersion = snap.contentVersion;
    this._lastBlasVersion = snap.blasContentVersion;
    this._lastTlasVersion = snap.tlasContentVersion;
    this._lastMaterialVersion = snap.materialContentVersion;

    if (needsReplacement) {
      if (!tlasOnly) dispatcherCandidate?.invalidateBindings();
      if (tlasOnly) destroyRCTLASBuffers(previousBvh);
      else destroyRCBVHBuffers(previousBvh);
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
    let nextBvh: RCBVHBuffers | null = null;
    let nextCascades: GPUBuffer[] | null = null;
    let nextDispatcher: RCDispatcher | null = null;
    let nextNodes!: Float32Array;
    let nextIndices!: Uint32Array;
    let nextPositions!: Float32Array;
    try {
      nextBvh = this._uploadBVH(bvh);
      nextCascades = this._allocateCascadeBuffers();
      nextNodes = new Float32Array(bvh.bvhNodes.array);
      nextIndices = new Uint32Array(bvh.indices.array);
      nextPositions = new Float32Array(bvh.positions.array);
      nextDispatcher = new RCDispatcher(this._cascadeDims);
    } catch (error) {
      destroyRCBVHBuffers(nextBvh);
      destroyGpuBuffers(nextCascades);
      nextDispatcher?.dispose();
      throw error;
    }

    const previousBvh = this._bvhBuffers;
    const previousCascades = this._cascadeBufs;
    const previousDispatcher = this._dispatcher;
    this._bvhBuffers = nextBvh;
    this._cascadeBufs = nextCascades;
    this._dispatcher = nextDispatcher;
    this._restirSnapshot = null;
    this._sharedGeometryBindings = null;
    this._bvhMode = 'merged';
    this._tlasNodeCount = 0;
    this._lastBvhVersion = 0;
    this._lastBlasVersion = -1;
    this._lastTlasVersion = -1;
    this._lastMaterialVersion = -1;
    this._mergedNodesCpu = nextNodes;
    this._mergedIndicesStride3 = nextIndices;
    this._mergedPositionsStride4 = nextPositions;

    const { min, max } = bvh.bounds;
    this._probeOriginWorld = [min.x, min.y, min.z];
    this._roomSize = [
      Math.max(max.x - min.x, 1e-6),
      Math.max(max.y - min.y, 1e-6),
      Math.max(max.z - min.z, 1e-6),
    ];

    previousDispatcher?.dispose();
    destroyRCBVHBuffers(previousBvh);
    destroyGpuBuffers(previousCascades);
  }

  /**
   * PR-5.3 — merged-mode moving-instance refit WITHOUT pipeline teardown.
   *
   * The merged RC BVH bakes each mesh's world transform into its vertex
   * positions (unlike TLAS mode, where instance transforms live in separate
   * matrices). So a moved instance invalidates both the affected vertex
   * positions and every BVH node AABB that bounds them. This path avoids the
   * expensive SAH rebuild and dispatcher/cascade recreation:
   *
   *   1. Copy the host-supplied world positions and build a candidate node mirror.
   *   2. `refitBvhBounds` recomputes every candidate node AABB in O(nodes + tris)
   *      while preserving split planes, child links, and triangle order.
   *   3. Upload candidate position + node buffers, then publish both identities
   *      together and release the previous pair.
   *
   * The dispatcher, cascade buffers, materials, and index buffers remain alive.
   * A failed candidate allocation leaves the live GPU buffers and CPU mirrors
   * untouched.
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

    const posMirror = new Float32Array(updatedPositionsStride4);
    const nodesMirror = new Float32Array(this._mergedNodesCpu);
    refitBvhBounds(nodesMirror, this._mergedIndicesStride3, posMirror, 4);

    let nextPositions: GPUBuffer | null = null;
    let nextNodes: GPUBuffer | null = null;
    if (this._sharedGeometryBindings == null) {
      try {
        nextPositions = this._uploadTypedArray(posMirror, 'rc-bvh-positions-refit');
        nextNodes = this._uploadTypedArray(nodesMirror, 'rc-bvh-nodes-refit');
      } catch (error) {
        nextPositions?.destroy();
        nextNodes?.destroy();
        throw error;
      }
      this._bvhBuffers = {
        ...bvh,
        bvhPositionsBuf: nextPositions,
        bvhNodesBuf: nextNodes,
      };
    }
    this._mergedPositionsStride4 = posMirror;
    this._mergedNodesCpu = nodesMirror;
    this._probeOriginWorld = [boundsMin[0], boundsMin[1], boundsMin[2]];
    this._roomSize = [
      Math.max(boundsMax[0] - boundsMin[0], 1e-6),
      Math.max(boundsMax[1] - boundsMin[1], 1e-6),
      Math.max(boundsMax[2] - boundsMin[2], 1e-6),
    ];
    if (this._sharedGeometryBindings == null) {
      this.invalidateBindings();
      bvh.bvhPositionsBuf?.destroy();
      bvh.bvhNodesBuf?.destroy();
    }
    return true;
  }

  /**
   * Upload/replace the runtime analytic/directional light alias buffer.
   *
   * Call once after `setSceneFromCore` / `syncRestirBvhBuffers` and again
   * whenever the scene's fixture list changes (matches the DDGI path:
   * `ProbeUpdatePass.setLights` → `packDDGIProbeLights`).
   *
   * The host forwards the same oriented `DDGILight[]` used by DDGI.
   */
  updateLights(lights: readonly DDGILight[]): void {
    const active = lights.filter((light) => light.on);
    const nextCount = active.length;
    const requiredBytes = rcLightsBufferByteLength(nextCount);
    const maxBufferSize = this._device.limits?.maxBufferSize;
    if (typeof maxBufferSize === 'number' && requiredBytes > maxBufferSize) {
      throw new RangeError(
        `[RCSubsystem] ${nextCount} lights require ${requiredBytes} bytes, ` +
        `exceeding device.limits.maxBufferSize=${maxBufferSize}.`,
      );
    }
    const maxStorageBinding = this._device.limits?.maxStorageBufferBindingSize;
    if (typeof maxStorageBinding === 'number' && requiredBytes > maxStorageBinding) {
      throw new RangeError(
        `[RCSubsystem] ${nextCount} lights require a ${requiredBytes}-byte storage binding, ` +
        `exceeding device.limits.maxStorageBufferBindingSize=${maxStorageBinding}.`,
      );
    }
    const fp = JSON.stringify(active);
    if (fp === this._lightsFingerprint) return;  // no change → skip GPU work
    const nextDirectionalCount = active.filter((light) => light.kind === 'sun').length;
    let next: GPUBuffer | null = null;
    if (nextCount > 0) {
      const packed = packRCLights(active);
      if (packed.byteLength !== requiredBytes) {
        throw new Error(
          `[RCSubsystem] packRCLights ABI mismatch: preflight=${requiredBytes}, packed=${packed.byteLength}.`,
        );
      }
      next = this._device.createBuffer({
        label: 'rc-lights',
        size: packed.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      try {
        new Uint8Array(next.getMappedRange()).set(new Uint8Array(packed));
        next.unmap();
      } catch (error) {
        next.destroy();
        throw error;
      }
    }

    const previous = this._lightsGpuBuf;
    this._lightsGpuBuf = next;
    this._lightsCount = nextCount;
    this._lightsDirectionalCount = nextDirectionalCount;
    this._lightsFingerprint = fp;
    this.invalidateBindings();
    previous?.destroy();
  }

  dispatchFrame(inputs: RCFrameInputs): void {
    if ((inputs.envTextureView != null) !== (inputs.envSampler != null)) {
      throw new Error(
        '[RCSubsystem] envTextureView and envSampler must be supplied together.',
      );
    }
    if (!this._dispatcher || !this._bvhBuffers || !this._cascadeBufs ||
        !this._probeOriginWorld || !this._roomSize) {
      this._warnOnce('dispatch-unavailable', {
        code: 'walkaround-hybrid.rc-dispatch-unavailable',
        backend: 'walkaround-hybrid',
        phase: 'renderFrame',
        method: 'RCSubsystem.dispatchFrame',
        message:
          '[RCSubsystem] RC frame dispatch was skipped because its scene, ' +
          'cascade, or dispatcher state has not been published.',
        details: {
          dispatcherReady: this._dispatcher != null,
          bvhReady: this._bvhBuffers != null,
          cascadesReady: this._cascadeBufs != null,
          boundsReady:
            this._probeOriginWorld != null && this._roomSize != null,
          fallback: 'skip-rc-frame',
        },
      });
      return;
    }
    const sharedGeometry = inputs.sceneGeometryBindings ?? this._sharedGeometryBindings;
    if (
      sharedGeometry != null &&
      !sameSceneGeometryBindings(this._sharedGeometryBindings, sharedGeometry)
    ) {
      const current = this._bvhBuffers;
      const {
        bvhNodesBuf,
        bvhIndicesBuf,
        bvhPositionsBuf,
        bvhNormalsBuf,
        ...nonGeometry
      } = current;
      this._bvhBuffers = nonGeometry;
      this._sharedGeometryBindings = sharedGeometry;
      this.invalidateBindings();
      destroyGpuBuffer(bvhNodesBuf);
      destroyGpuBuffer(bvhIndicesBuf);
      destroyGpuBuffer(bvhPositionsBuf);
      destroyGpuBuffer(bvhNormalsBuf);
    }
    const bvh = this._bvhBuffers;
    const nodesBinding = sharedGeometry?.bvhNodes;
    const indicesBinding = sharedGeometry?.bvhIndices;
    const positionsBinding = sharedGeometry?.bvhPositions;
    const normalsBinding = sharedGeometry?.bvhNormals;
    this._dispatcher.dispatchFrameRaw({
      device: this._device,
      bvhNodesBuf:      nodesBinding?.buffer ?? bvh.bvhNodesBuf!,
      ...(nodesBinding?.offset == null ? {} : { bvhNodesOffset: Number(nodesBinding.offset) }),
      ...(nodesBinding?.size == null ? {} : { bvhNodesSize: Number(nodesBinding.size) }),
      bvhIndicesBuf:    indicesBinding?.buffer ?? bvh.bvhIndicesBuf!,
      ...(indicesBinding?.offset == null ? {} : { bvhIndicesOffset: Number(indicesBinding.offset) }),
      ...(indicesBinding?.size == null ? {} : { bvhIndicesSize: Number(indicesBinding.size) }),
      bvhPositionsBuf:  positionsBinding?.buffer ?? bvh.bvhPositionsBuf!,
      ...(positionsBinding?.offset == null ? {} : { bvhPositionsOffset: Number(positionsBinding.offset) }),
      ...(positionsBinding?.size == null ? {} : { bvhPositionsSize: Number(positionsBinding.size) }),
      bvhNormalsBuf:    normalsBinding?.buffer ?? bvh.bvhNormalsBuf!,
      ...(normalsBinding?.offset == null ? {} : { bvhNormalsOffset: Number(normalsBinding.offset) }),
      ...(normalsBinding?.size == null ? {} : { bvhNormalsSize: Number(normalsBinding.size) }),
      materialsBuf:     bvh.materialsBuf,
      triMaterialIdBuf: bvh.triMaterialIdBuf,
      cascadeBufs:      this._cascadeBufs,
      probeOriginWorld: this._probeOriginWorld,
      roomSize:         this._roomSize,
      sunDirection:     inputs.sunDirection,
      // Directional lights have one owner: when updateLights contains any sun,
      // the alias estimator owns all directionals and the legacy sun lane is zero.
      sunColor:         resolveRCLegacySunColor(this._lightsDirectionalCount, inputs.sunColor),
      sunCastShadowDisabled: inputs.sunCastShadowDisabled === true,
      sunAngularRadius: inputs.sunAngularRadius ?? 0,
      frameSeed:        inputs.frameSeed,
      triIntersectEpsilon: inputs.triIntersectEpsilon,
      scalarSkyRadiance: inputs.scalarSkyRadiance ?? [0, 0, 0],
      hasDirectionalEnvironment:
        inputs.hasDirectionalEnvironment ??
        (inputs.envTextureView != null && inputs.envSampler != null),
      transmittedInterfaceBudget: this._transmittedInterfaceBudget,
      bvhMode:          this._bvhMode,
      tlasNodeCount:    this._tlasNodeCount,
      tlasArenaVersion: this._lastTlasVersion,
      // A7: env texture forwarded from the main pipeline (placeholder if absent).
      ...(inputs.envTextureView != null && inputs.envSampler != null
        ? {
            envTextureView: inputs.envTextureView,
            envSampler: inputs.envSampler,
            envRotationY: inputs.envRotationY ?? 0,
            envIntensity: inputs.envIntensity ?? 1,
          }
        : {}),
      ...(inputs.materialTextureAtlasView != null && inputs.materialMapMetaTextureView != null
        ? {
            materialTextureAtlasView: inputs.materialTextureAtlasView,
            materialMapMetaTextureView: inputs.materialMapMetaTextureView,
            ...(inputs.bvhTangentTextureView != null
              ? { bvhTangentTextureView: inputs.bvhTangentTextureView }
              : {}),
            ...(inputs.bvhVertexColorTextureView != null
              ? { bvhVertexColorTextureView: inputs.bvhVertexColorTextureView }
              : {}),
          }
        : {}),
      ...(inputs.emittersBuf != null
        ? {
            emittersBuf: inputs.emittersBuf,
            emittersOffset: inputs.emittersOffset ?? 0,
            ...(inputs.emittersSize == null ? {} : { emittersSize: inputs.emittersSize }),
            emitterDataOffset: inputs.emitterDataOffset ?? 0,
            emitterAliasOffset: inputs.emitterAliasOffset ?? 0,
            emitterCount: inputs.emitterCount ?? 0,
          }
        : {}),
      // Runtime analytic/directional lights; null uses the header-only placeholder.
      ...(this._lightsGpuBuf != null
        ? { lightsBuf: this._lightsGpuBuf, lightsSize: this._lightsGpuBuf.size, lightCount: this._lightsCount }
        : (inputs.lightsBuf != null
          ? { lightsBuf: inputs.lightsBuf, lightsSize: inputs.lightsBuf.size, lightCount: inputs.lightCount ?? 0 }
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

  /** Flatten RC-owned persistent GPU buffers for the engine memory budget.
   * Borrowed main-pipeline geometry is absent from `_bvhBuffers` after arena
   * adoption and therefore is not double-counted. */
  gpuMemorySection(): Readonly<Record<string, GPUBuffer>> {
    const section: Record<string, GPUBuffer> = {};
    const seen = new Set<GPUBuffer>();
    const add = (label: string, value: unknown): void => {
      if (
        value == null ||
        typeof value !== 'object' ||
        typeof (value as GPUBuffer).size !== 'number' ||
        typeof (value as GPUBuffer).usage !== 'number' ||
        seen.has(value as GPUBuffer)
      ) {
        return;
      }
      seen.add(value as GPUBuffer);
      section[label] = value as GPUBuffer;
    };
    for (const [label, buffer] of Object.entries(this._bvhBuffers ?? {})) {
      add(label, buffer);
    }
    for (let index = 0; index < (this._cascadeBufs?.length ?? 0); index++) {
      add(`cascade${index}`, this._cascadeBufs?.[index]);
    }
    add('lights', this._lightsGpuBuf);
    return section;
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
    const dispatcher = this._dispatcher;
    this._dispatcher = null;
    try { dispatcher?.dispose(); } catch { /* continue releasing owned resources */ }
    const lightsBuffer = this._lightsGpuBuf;
    this._lightsGpuBuf = null;
    destroyGpuBuffer(lightsBuffer ?? undefined);
    this._lightsCount = 0;
    this._lightsDirectionalCount = 0;
    this._lightsFingerprint = '';  // reset so next init re-uploads correctly
  }



  private _uploadRestirTlasBuffers(tlas: NonNullable<RestirBvhSnapshot['tlas']>) {
    const created: GPUBuffer[] = [];
    const upload = (label: string, data: ArrayBuffer): GPUBuffer => {
      const buffer = this._device.createBuffer({
        label,
        size: Math.max(data.byteLength, 16),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      created.push(buffer);
      new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
      buffer.unmap();
      return buffer;
    };
    try {
      return {
        tlasNodesBuf: upload('rc-restir-tlas-nodes', tlas.nodes),
        tlasInstanceIndicesBuf: upload('rc-restir-tlas-inst', tlas.instanceIndices),
        tlasBlasRootsBuf: upload('rc-restir-tlas-blas', tlas.blasRoots),
        tlasInstanceWorldToLocalBuf: upload('rc-restir-tlas-w2l', tlas.worldToLocal),
        tlasInstanceLocalToWorldBuf: upload('rc-restir-tlas-l2w', tlas.localToWorld),
      };
    } catch (error) {
      destroyGpuBuffers(created);
      throw error;
    }
  }
  private _uploadFromRestirSnapshot(
    snap: RestirBvhSnapshot,
    includeGeometry = true,
  ): RCBVHBuffers {
    const device = this._device;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const created: GPUBuffer[] = [];
    const upload = (label: string, data: ArrayBuffer): GPUBuffer => {
      const buf = device.createBuffer({
        label,
        size: Math.max(data.byteLength, 16),
        usage,
        mappedAtCreation: true,
      });
      created.push(buf);
      new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data));
      buf.unmap();
      return buf;
    };
    try {
      const matFloats = packCascadeMaterialsFromCore([...snap.coreMaterials]);
      const tlas = snap.tlas;
      return {
        ...(includeGeometry
          ? {
              bvhNodesBuf: upload('rc-restir-bvh-nodes', snap.bvhNodes),
              bvhIndicesBuf: upload('rc-restir-bvh-index', snap.bvhIndex),
              bvhPositionsBuf: upload('rc-restir-bvh-positions', snap.positions),
              bvhNormalsBuf: upload('rc-restir-bvh-normals', snap.normals),
            }
          : {}),
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
    } catch (error) {
      destroyGpuBuffers(created);
      throw error;
    }
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
    const created: GPUBuffer[] = [];
    const keep = (buffer: GPUBuffer): GPUBuffer => {
      created.push(buffer);
      return buffer;
    };
    try {
      return {
        bvhNodesBuf: keep(this._uploadAttribute(bvh.bvhNodes, 'rc-bvh-nodes')),
        bvhIndicesBuf: keep(this._uploadTypedArray(idxStride4, 'rc-bvh-indices')),
        bvhPositionsBuf: keep(this._uploadAttribute(bvh.positions, 'rc-bvh-positions')),
        bvhNormalsBuf: keep(this._uploadAttribute(bvh.normals, 'rc-bvh-normals')),
        materialsBuf: keep(this._uploadAttribute(bvh.materials, 'rc-bvh-materials')),
        triMaterialIdBuf: keep(this._uploadAttribute(bvh.triMaterialId, 'rc-bvh-tri-mat-id')),
      };
    } catch (error) {
      destroyGpuBuffers(created);
      throw error;
    }
  }

  /** Upload a raw typed array as a STORAGE buffer (used for the F-RC1 stride-4
   *  index pad, which produces a fresh Uint32Array not backed by a
   *  StorageBufferAttribute). */
  private _uploadTypedArray(arr: Float32Array | Uint32Array, label: string): GPUBuffer {
    const buf = this._device.createBuffer({
      label,
      size: Math.max(arr.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    try {
      new Uint8Array(buf.getMappedRange()).set(
        new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength),
      );
      buf.unmap();
      return buf;
    } catch (error) {
      buf.destroy();
      throw error;
    }
  }

  private _uploadAttribute(attr: StorageAttributeLike, label: string): GPUBuffer {
    const arr = attr.array;
    const buf = this._device.createBuffer({
      label,
      size: arr.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    try {
      new Uint8Array(buf.getMappedRange()).set(
        new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength),
      );
      buf.unmap();
      return buf;
    } catch (error) {
      buf.destroy();
      throw error;
    }
  }

  private _allocateCascadeBuffers(): GPUBuffer[] {
    const created: GPUBuffer[] = [];
    try {
      for (const [k, c] of this._cascadeDims.entries()) {
        const totalRays = c.probes[0] * c.probes[1] * c.probes[2] * c.rays;
        created.push(this._device.createBuffer({
          label: `rc-cascade-C${k}`,
          size: totalRays * 16,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        }));
      }
      return created;
    } catch (error) {
      destroyGpuBuffers(created);
      throw error;
    }
  }

  private _disposeBvhBuffersOnly(): void {
    const previous = this._bvhBuffers;
    this._bvhBuffers = null;
    destroyRCBVHBuffers(previous);
  }

  private _disposeSceneBuffers(): void {
    this._disposeBvhBuffersOnly();
    const cascades = this._cascadeBufs;
    this._cascadeBufs = null;
    destroyGpuBuffers(cascades);
    this._probeOriginWorld = null;
    this._roomSize         = null;
    this._restirSnapshot = null;
    this._sharedGeometryBindings = null;
    this._mergedNodesCpu = null;
    this._mergedIndicesStride3 = null;
    this._mergedPositionsStride4 = null;
    this._lastBvhVersion = 0;
    this._lastBlasVersion = -1;
    this._lastTlasVersion = -1;
    this._lastMaterialVersion = -1;
  }
}
