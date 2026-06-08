import type { MaterialSpec } from '@vitrum/core';
import type { PrimitiveTlasBinding, ScenePackResult } from '@vitrum/shared-bvh';

export type ReSTIRBvhMode = 'merged' | 'tlas';

/** A CPU-side byte payload that will be uploaded to a WebGPU storage binding. */
export interface StorageBufferHandle {
  cpuData: ArrayBuffer;
  byteLength: number;
  count: number;
}

export interface RestirBoundsLike {
  readonly min: { readonly x: number; readonly y: number; readonly z: number };
  readonly max: { readonly x: number; readonly y: number; readonly z: number };
}

/** Minimal geometry handle retained for bounds fallback and deterministic teardown. */
export interface RestirMergedGeometryLike {
  boundingBox: RestirBoundsLike | null;
  computeBoundingBox(): void;
  dispose(): void;
}

export interface SceneBVHBuffers {
  bvhMode: ReSTIRBvhMode;
  bvhNodes: StorageBufferHandle;
  bvhIndex: StorageBufferHandle;
  bvhPositions: StorageBufferHandle;
  triangleMaterialIds: StorageBufferHandle;
  bvhBeerColors: StorageBufferHandle;
  bvhEmissiveLe: StorageBufferHandle;
  bvhNormals: StorageBufferHandle;
  emitters: StorageBufferHandle;
  emitterCdf: StorageBufferHandle;
  emitterCount: number;
  totalEmissivePower: number;
  lightTree: StorageBufferHandle;
  lightTreeNodeCount: number;
  lightTreeEnabled: boolean;
  mergedGeometry: RestirMergedGeometryLike;
  meshVertexRanges: ReadonlyArray<{
    name: string;
    vertexStart: number;
    vertexCount: number;
    triStart: number;
    triCount: number;
    matrixWorldAtBuild: Float32Array;
  }>;
  bvhIndicesStride3: Uint32Array;
  /** Legacy adapters may retain source material handles; core callers use coreMaterials. */
  buildMaterials: readonly unknown[];
  coreMaterials: readonly MaterialSpec[];
  emitterNormals: Float32Array;
  tlas?: {
    nodes: StorageBufferHandle;
    instanceIndices: StorageBufferHandle;
    blasRoots: StorageBufferHandle;
    worldToLocal: StorageBufferHandle;
    localToWorld: StorageBufferHandle;
    nodeCount: number;
  };
  primitiveTlasBindings: readonly PrimitiveTlasBinding[];
  scenePack?: ScenePackResult;
  warnings?: readonly string[];
}

export type RebuiltEmitterBuffers = Pick<
  SceneBVHBuffers,
  | 'emitters'
  | 'emitterCdf'
  | 'emitterCount'
  | 'totalEmissivePower'
  | 'lightTree'
  | 'lightTreeNodeCount'
  | 'lightTreeEnabled'
>;
