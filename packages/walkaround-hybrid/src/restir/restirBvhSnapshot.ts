/**
 * C2 — CPU mirror of ReSTIR `SceneBVHBuffers` for subsystems that ray-cast
 * the same BLAS/TLAS (DDGI probe update, RC cascades).
 */

import type { MaterialSpec, Scene } from '@vitrum/core';
import {
  computeWorldAabbForBindings,
  fingerprintBuffersExact,
  fingerprintBuffersExactAndEqual,
  isTlasOnlyVersionBump,
} from '@vitrum/shared-bvh';
import type { SceneBVHBuffers } from './bvhTypes.js';
import {
  materialTextureAtlasFingerprintParts,
  type MaterialTextureAtlasPayload,
} from '../pipeline/materialTextureAtlas.js';
import {
  packDDGIMaterialsFromCoreN,
  packDDGIMaterialsN,
} from '../ddgi/probeUpdateMaterials.js';
import type { PbrScalarSource } from '../pbrScalars.js';
import {
  makeEmptyAabb,
  setAabb,
  copyBoxLike,
  isAabbEmpty,
} from './aabbHelpers.js';

interface RestirBvhVector3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

interface RestirBvhAabb {
  readonly min: RestirBvhVector3;
  readonly max: RestirBvhVector3;
}

export interface RestirBvhSnapshot {
  readonly bvhMode: 'merged' | 'tlas';
  readonly tlasNodeCount: number;
  readonly bvhNodes: ArrayBuffer;
  readonly positions: ArrayBuffer;
  /** Stride-4 triangle index buffer (ReSTIR `bvhIndex`). */
  readonly bvhIndex: ArrayBuffer;
  /** Exact optical component/range identity, vec2u per BLAS triangle. */
  readonly opticalTriangleIdentity: ArrayBuffer;
  /** Encoded boundary base plus one, one u32 per TLAS instance or one merged entry. */
  readonly opticalInstanceBoundaryIdBasePlusOne: ArrayBuffer;
  readonly normals: ArrayBuffer;
  /** Per-vertex authored/generated tangent.xyzw stream. Zero means derive TBN from UVs. */
  readonly tangents: ArrayBuffer;
  /** Per-vertex COLOR_0 rgba stream. Missing authored colors are white-filled upstream. */
  readonly vertexColors: ArrayBuffer;
  readonly triMaterialIds: ArrayBuffer;
  readonly materials: readonly unknown[];
  /**
   * THREE-DECOUPLE of the production ReSTIR MATERIAL path. The deduped core
   * `MaterialSpec[]`, slot-aligned with {@link materials} (THREE) and
   * `triMaterialIds`. Mirrors `SceneBVHBuffers.coreMaterials`. Production DDGI
   * (`probeUpdatePass.ts`) prefers this — packing its per-material struct from
   * core `MaterialSpec`s via `coreMaterialToMaterialEntry`, NO THREE round-trip —
   * and falls back to {@link materials} (THREE) when it is empty (the legacy
   * THREE-only merged build). RC's cascade-material packer is unaffected: it
   * keeps reading {@link materials} (THREE).
   */
  readonly coreMaterials: readonly MaterialSpec[];
  /** Readable material-map atlas payload, slot-aligned with the active BVH triangles. */
  readonly materialTextureAtlas: MaterialTextureAtlasPayload;
  readonly boundingBox: RestirBvhAabb;
  readonly tlas?: {
    readonly nodes: ArrayBuffer;
    readonly instanceIndices: ArrayBuffer;
    readonly blasRoots: ArrayBuffer;
    readonly worldToLocal: ArrayBuffer;
    readonly localToWorld: ArrayBuffer;
  };
  /** Bumps when any mirrored buffer payload changes (not just lengths). */
  readonly contentVersion: number;
  /** BLAS concat buffers only — stable across TLAS transform-only refit. */
  readonly blasContentVersion: number;
  /** TLAS nodes + instance transforms — bumps on transform refit. */
  readonly tlasContentVersion: number;
  /** Scalar material bytes + texture-atlas bytes consumed by DDGI/RC. */
  readonly materialContentVersion: number;
}

type SnapshotBufferPart = ArrayBuffer | ArrayBufferView;

interface RetainedRestirBvhSnapshotState {
  readonly bvhMode: RestirBvhSnapshot['bvhMode'];
  readonly tlasNodeCount: number;
  readonly bounds: readonly [number, number, number, number, number, number];
  readonly blasParts: readonly Uint8Array[];
  readonly tlasParts: readonly Uint8Array[];
  readonly materialParts: readonly Uint8Array[];
}

const RETAINED_SNAPSHOT_STATE: unique symbol = Symbol(
  'vitrum.restirBvhSnapshot.retainedState',
);
const BUFFER_SNAPSHOT_CACHE: unique symbol = Symbol(
  'vitrum.restirBvhSnapshot.bufferCache',
);

type RetainedRestirBvhSnapshot = RestirBvhSnapshot & {
  readonly [RETAINED_SNAPSHOT_STATE]: RetainedRestirBvhSnapshotState;
};

type SnapshotCachedBuffers = SceneBVHBuffers & {
  [BUFFER_SNAPSHOT_CACHE]?: RetainedRestirBvhSnapshot;
};

interface SnapshotParts {
  readonly bvhMode: RestirBvhSnapshot['bvhMode'];
  readonly tlasNodeCount: number;
  readonly bounds: readonly [number, number, number, number, number, number];
  readonly blasParts: readonly SnapshotBufferPart[];
  readonly tlasParts: readonly SnapshotBufferPart[];
  readonly materialParts: readonly SnapshotBufferPart[];
}

interface SnapshotCandidate {
  readonly snapshot: RestirBvhSnapshot;
  readonly parts: SnapshotParts;
  readonly equalsRetainedState: boolean;
}

function materialAtlasMetadata(atlas: MaterialTextureAtlasPayload): Uint32Array {
  return new Uint32Array([
    atlas.atlasLayers.length,
    atlas.atlasLayers.reduce(
      (maximum, layer) => Math.max(maximum, layer.mipLevelCount),
      0,
    ),
    atlas.atlasLayers.reduce(
      (sum, layer) => sum + layer.width * layer.height,
      0,
    ),
    atlas.gpuSourceLayers.length,
    atlas.baseColorMetaWidth,
    atlas.baseColorMetaHeight,
    atlas.readableBaseColorLayerCount,
    atlas.readableNormalLayerCount,
    atlas.readableRoughnessLayerCount,
    atlas.readableMetallicLayerCount,
    atlas.readableAoLayerCount,
    atlas.readableAlphaLayerCount,
    atlas.readableEmissiveLayerCount,
    atlas.readableTransmissionLayerCount,
    atlas.readableLightLayerCount,
    atlas.readableSpecularColorLayerCount,
    atlas.readableSpecularIntensityLayerCount,
    atlas.readableClearcoatLayerCount,
    atlas.readableClearcoatRoughnessLayerCount,
    atlas.readableClearcoatNormalLayerCount,
    atlas.readableSheenColorLayerCount,
    atlas.readableSheenRoughnessLayerCount,
    atlas.readableAnisotropyLayerCount,
    atlas.readableIridescenceLayerCount,
    atlas.readableIridescenceThicknessLayerCount,
    atlas.readableThicknessLayerCount,
    atlas.readableBumpLayerCount,
  ]);
}

function byteView(part: SnapshotBufferPart): Uint8Array {
  const isShared =
    typeof SharedArrayBuffer !== 'undefined' && part instanceof SharedArrayBuffer;
  return part instanceof ArrayBuffer || isShared
    ? new Uint8Array(part)
    : new Uint8Array(part.buffer, part.byteOffset, part.byteLength);
}

function cloneParts(parts: readonly SnapshotBufferPart[]): readonly Uint8Array[] {
  return parts.map((part) => byteView(part).slice());
}

function boundsTuple(
  bounds: RestirBvhAabb,
): readonly [number, number, number, number, number, number] {
  return [
    bounds.min.x,
    bounds.min.y,
    bounds.min.z,
    bounds.max.x,
    bounds.max.y,
    bounds.max.z,
  ];
}

function sameBounds(
  a: readonly [number, number, number, number, number, number],
  b: readonly [number, number, number, number, number, number],
): boolean {
  for (let index = 0; index < a.length; index += 1) {
    if (!Object.is(a[index], b[index])) return false;
  }
  return true;
}

function exactPartsEqual(
  a: readonly Uint8Array[],
  b: readonly Uint8Array[],
): boolean {
  if (a.length !== b.length) return false;
  for (let partIndex = 0; partIndex < a.length; partIndex += 1) {
    const left = a[partIndex]!;
    const right = b[partIndex]!;
    if (left.byteLength !== right.byteLength) return false;
    for (let byteIndex = 0; byteIndex < left.byteLength; byteIndex += 1) {
      if (left[byteIndex] !== right[byteIndex]) return false;
    }
  }
  return true;
}

function retainedState(
  snapshot: RestirBvhSnapshot | null,
): RetainedRestirBvhSnapshotState | null {
  return snapshot == null
    ? null
    : (snapshot as Partial<RetainedRestirBvhSnapshot>)[RETAINED_SNAPSHOT_STATE] ?? null;
}

function retainCandidate(candidate: SnapshotCandidate): RetainedRestirBvhSnapshot {
  const retained: RetainedRestirBvhSnapshotState = {
    bvhMode: candidate.parts.bvhMode,
    tlasNodeCount: candidate.parts.tlasNodeCount,
    bounds: candidate.parts.bounds,
    blasParts: cloneParts(candidate.parts.blasParts),
    tlasParts: cloneParts(candidate.parts.tlasParts),
    materialParts: cloneParts(candidate.parts.materialParts),
  };
  return Object.assign(candidate.snapshot, {
    [RETAINED_SNAPSHOT_STATE]: retained,
  });
}

function cacheSnapshot(
  buffers: SceneBVHBuffers,
  snapshot: RetainedRestirBvhSnapshot,
): void {
  try {
    Object.defineProperty(buffers, BUFFER_SNAPSHOT_CACHE, {
      configurable: true,
      enumerable: false,
      writable: true,
      value: snapshot,
    });
  } catch {
    // A frozen structural test double can still use the caller-retained state;
    // the cache only shares one retained copy between DDGI and RC.
  }
}

function buildRestirBvhSnapshotCandidate(
  buffers: SceneBVHBuffers,
  scene?: Scene,
  previousState: RetainedRestirBvhSnapshotState | null = null,
): SnapshotCandidate {
  const bbox = makeEmptyAabb();
  if (
    buffers.bvhMode === 'tlas' &&
    scene != null &&
    buffers.primitiveTlasBindings.length > 0
  ) {
    const world = computeWorldAabbForBindings(scene, buffers.primitiveTlasBindings);
    if (world != null) {
      setAabb(
        bbox,
        world.min[0], world.min[1], world.min[2],
        world.max[0], world.max[1], world.max[2],
      );
    }
  }
  if (isAabbEmpty(bbox)) {
    if (buffers.mergedGeometry.boundingBox != null) {
      copyBoxLike(bbox, buffers.mergedGeometry.boundingBox);
    } else {
      buffers.mergedGeometry.computeBoundingBox();
      if (buffers.mergedGeometry.boundingBox != null) {
        copyBoxLike(bbox, buffers.mergedGeometry.boundingBox);
      }
    }
  }

  const tlas = buffers.tlas;
  const blasParts: readonly SnapshotBufferPart[] = [
    buffers.bvhNodes.cpuData,
    buffers.bvhPositions.cpuData,
    buffers.bvhIndex.cpuData,
    buffers.opticalTriangleIdentity.cpuData,
    buffers.opticalInstanceBoundaryIdBasePlusOne.cpuData,
    buffers.bvhNormals.cpuData,
    buffers.bvhTangents.cpuData,
    buffers.bvhColors.cpuData,
    buffers.triangleMaterialIds.cpuData,
  ];
  const tlasParts: readonly SnapshotBufferPart[] = tlas != null
    ? [
        tlas.nodes.cpuData,
        tlas.instanceIndices.cpuData,
        tlas.blasRoots.cpuData,
        tlas.worldToLocal.cpuData,
        tlas.localToWorld.cpuData,
      ]
    : [];
  const materialSlots = Math.max(
    1,
    buffers.coreMaterials.length,
    buffers.buildMaterials.length,
  );
  const materialBytes = buffers.coreMaterials.length > 0
    ? packDDGIMaterialsFromCoreN(buffers.coreMaterials, materialSlots)
    : packDDGIMaterialsN(
        buffers.buildMaterials as readonly PbrScalarSource[],
        materialSlots,
      );
  const atlas = buffers.materialTextureAtlas;
  const materialParts: readonly SnapshotBufferPart[] = [
    materialBytes,
    atlas.baseColorMetaData,
    materialAtlasMetadata(atlas),
    ...materialTextureAtlasFingerprintParts(atlas),
  ];
  const blasComparison = previousState == null
    ? {
        fingerprint: fingerprintBuffersExact(...blasParts),
        equal: false,
      }
    : fingerprintBuffersExactAndEqual(blasParts, previousState.blasParts);
  const tlasComparison = tlasParts.length === 0
    ? {
        fingerprint: 0,
        equal: previousState != null && previousState.tlasParts.length === 0,
      }
    : previousState == null
      ? {
          fingerprint: fingerprintBuffersExact(...tlasParts),
          equal: false,
        }
      : fingerprintBuffersExactAndEqual(tlasParts, previousState.tlasParts);
  const materialComparison = previousState == null
    ? {
        fingerprint: fingerprintBuffersExact(...materialParts),
        equal: false,
      }
    : fingerprintBuffersExactAndEqual(materialParts, previousState.materialParts);
  const blasContentVersion = blasComparison.fingerprint;
  const tlasContentVersion = tlasComparison.fingerprint;
  const materialContentVersion = materialComparison.fingerprint;
  const bounds = boundsTuple(bbox);
  const parts: SnapshotParts = {
    bvhMode: buffers.bvhMode,
    tlasNodeCount: tlas?.nodeCount ?? 0,
    bounds,
    blasParts,
    tlasParts,
    materialParts,
  };

  const snapshot: RestirBvhSnapshot = {
    bvhMode: buffers.bvhMode,
    tlasNodeCount: tlas?.nodeCount ?? 0,
    bvhNodes: buffers.bvhNodes.cpuData,
    positions: buffers.bvhPositions.cpuData,
    bvhIndex: buffers.bvhIndex.cpuData,
    opticalTriangleIdentity: buffers.opticalTriangleIdentity.cpuData,
    opticalInstanceBoundaryIdBasePlusOne:
      buffers.opticalInstanceBoundaryIdBasePlusOne.cpuData,
    normals: buffers.bvhNormals.cpuData,
    tangents: buffers.bvhTangents.cpuData,
    vertexColors: buffers.bvhColors.cpuData,
    triMaterialIds: buffers.triangleMaterialIds.cpuData,
    materials: buffers.buildMaterials,
    coreMaterials: buffers.coreMaterials,
    materialTextureAtlas: buffers.materialTextureAtlas,
    boundingBox: bbox,
    ...(tlas != null
      ? {
          tlas: {
            nodes: tlas.nodes.cpuData,
            instanceIndices: tlas.instanceIndices.cpuData,
            blasRoots: tlas.blasRoots.cpuData,
            worldToLocal: tlas.worldToLocal.cpuData,
            localToWorld: tlas.localToWorld.cpuData,
          },
        }
      : {}),
    contentVersion: fingerprintBuffersExact(
      new Uint32Array([
        blasContentVersion,
        tlasContentVersion,
        materialContentVersion,
      ]),
    ),
    blasContentVersion,
    tlasContentVersion,
    materialContentVersion,
  };
  return {
    snapshot,
    parts,
    equalsRetainedState:
      previousState != null &&
      previousState.bvhMode === parts.bvhMode &&
      previousState.tlasNodeCount === parts.tlasNodeCount &&
      sameBounds(previousState.bounds, parts.bounds) &&
      blasComparison.equal &&
      tlasComparison.equal &&
      materialComparison.equal,
  };
}

/**
 * Build a snapshot, sharing a single immutable exact-comparison state between
 * DDGI and RC for the same mutable `SceneBVHBuffers` object. The retained bytes
 * make equality collision-safe even when a producer mutates CPU mirrors in
 * place; unchanged frames hash and compare in one pass and allocate nothing.
 */
export function refreshRestirBvhSnapshot(
  previous: RestirBvhSnapshot | null,
  buffers: SceneBVHBuffers,
  scene?: Scene,
): RestirBvhSnapshot {
  const cached = (buffers as SnapshotCachedBuffers)[BUFFER_SNAPSHOT_CACHE] ?? null;
  const comparisonSnapshot = cached ?? previous;
  const candidate = buildRestirBvhSnapshotCandidate(
    buffers,
    scene,
    retainedState(comparisonSnapshot),
  );
  if (comparisonSnapshot != null && candidate.equalsRetainedState) {
    if (cached == null) {
      cacheSnapshot(buffers, comparisonSnapshot as RetainedRestirBvhSnapshot);
    }
    return comparisonSnapshot;
  }
  const retained = retainCandidate(candidate);
  cacheSnapshot(buffers, retained);
  return retained;
}

export function makeRestirBvhSnapshot(
  buffers: SceneBVHBuffers,
  scene?: Scene,
): RestirBvhSnapshot {
  return refreshRestirBvhSnapshot(null, buffers, scene);
}

/** Collision-safe equality over every runtime-affecting retained byte. */
export function restirBvhSnapshotStateEqual(
  a: RestirBvhSnapshot | null,
  b: RestirBvhSnapshot | null,
): boolean {
  if (a === b) return true;
  const aState = retainedState(a);
  const bState = retainedState(b);
  return aState != null && bState != null &&
    aState.bvhMode === bState.bvhMode &&
    aState.tlasNodeCount === bState.tlasNodeCount &&
    sameBounds(aState.bounds, bState.bounds) &&
    exactPartsEqual(aState.blasParts, bState.blasParts) &&
    exactPartsEqual(aState.tlasParts, bState.tlasParts) &&
    exactPartsEqual(aState.materialParts, bState.materialParts);
}

/** True only when exact retained BLAS/material state is unchanged and the TLAS
 * payload alone changed. Compact hashes are not used as equality proofs. */
export function isRestirTlasOnlySnapshotChange(
  previous: RestirBvhSnapshot | null,
  next: RestirBvhSnapshot,
): boolean {
  const previousState = retainedState(previous);
  const nextState = retainedState(next);
  return previousState != null && nextState != null &&
    previousState.bvhMode === 'tlas' &&
    nextState.bvhMode === 'tlas' &&
    next.tlas != null &&
    previousState.tlasNodeCount !== 0 &&
    nextState.tlasNodeCount !== 0 &&
    exactPartsEqual(previousState.blasParts, nextState.blasParts) &&
    exactPartsEqual(previousState.materialParts, nextState.materialParts) &&
    !exactPartsEqual(previousState.tlasParts, nextState.tlasParts);
}

/** True when only TLAS nodes / instance transforms changed (transform-only refit). */
export function isRestirTlasOnlyRefit(
  snap: RestirBvhSnapshot,
  prev: {
    readonly blasContentVersion: number;
    readonly tlasContentVersion: number;
    readonly materialContentVersion: number;
  },
): boolean {
  return (
    snap.tlas != null &&
    snap.materialContentVersion === prev.materialContentVersion &&
    isTlasOnlyVersionBump(snap.blasContentVersion, snap.tlasContentVersion, prev)
  );
}
