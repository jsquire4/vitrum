import type { MaterialSpec } from '@vitrum/core';
import type { PrimitiveTlasBinding, ScenePackResult } from '@vitrum/shared-bvh';
import type { MaterialTextureAtlasPayload } from '../pipeline/materialTextureAtlas.js';

export type ReSTIRBvhMode = 'merged' | 'tlas';

/** A CPU-side byte payload that will be uploaded to a WebGPU storage binding. */
interface StorageBufferHandle {
  cpuData: ArrayBuffer;
  byteLength: number;
  count: number;
}

interface RestirBoundsLike {
  readonly min: { readonly x: number; readonly y: number; readonly z: number };
  readonly max: { readonly x: number; readonly y: number; readonly z: number };
}

/** Minimal geometry handle retained for bounds fallback and deterministic teardown. */
export interface RestirMergedGeometryLike {
  boundingBox: RestirBoundsLike | null;
  computeBoundingBox(): void;
}

export interface SceneBVHBuffers {
  bvhMode: ReSTIRBvhMode;
  bvhNodes: StorageBufferHandle;
  bvhIndex: StorageBufferHandle;
  bvhPositions: StorageBufferHandle;
  triangleMaterialIds: StorageBufferHandle;
  bvhBeerColors: StorageBufferHandle;
  bvhEmissiveLe: StorageBufferHandle;
  /** First material-texture atlas slice: baseColorMap metadata + RGBA32F array layers. */
  materialTextureAtlas: MaterialTextureAtlasPayload;
  /** B1 (road-to-100) — per-triangle roughness+metalness lane
   *  (bits[31:24]=rough×255, bits[23:16]=metal×255), uploaded as an r32uint
   *  texture (BvhBufferHost) and read by the ReSTIR/shade GGX BRDF +
   *  glossy/metal GI target via decodeRoughMetal(triIndex). See
   *  packingHelpers.packBVHRoughMetal for the diffuse-default invariant. */
  bvhRoughMetal: StorageBufferHandle;
  bvhNormals: StorageBufferHandle;
  /** Per-vertex authored/generated tangents, vec4f stride. Uploaded as a
   *  texture binding (not storage) so the scene bind group stays within the
   *  WebGPU storage-buffer floor. xyz = tangent, w = bitangent handedness;
   *  0,0,0,0 means derive a UV-gradient frame in WGSL. */
  bvhTangents: StorageBufferHandle;
  /** Per-vertex COLOR_0 colors, vec4f stride (rgba). Uploaded as a texture
   *  binding and multiplied into visible baseColor/alpha; missing colors are
   *  already white-filled by shared-bvh. */
  bvhColors: StorageBufferHandle;
  /** Compact GPU-visible membership domains for manifold/SMS facet proposals.
   * Each 32-byte record stores a disjoint triangle range, instance range, and
   * Walker/Vose alias entry weighted by their Cartesian-product cardinality.
   * This represents every concrete facet identity without O(triangles*instances)
   * storage and remains valid across material-only mutations. */
  mneeFacetDomains?: StorageBufferHandle;
  emitters: StorageBufferHandle;
  emitterCdf: StorageBufferHandle;
  /** Runtime-sized alias sampler for bounded-work RC emitter selection. */
  emitterAlias: StorageBufferHandle;
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
  /**
   * Merged-BVH postorder node indices affected by each primitive. Built once
   * with the BVH so live refits can update/journal only leaves containing that
   * primitive plus their ancestors, without rescanning unrelated topology.
   */
  primitiveRefitNodeIndices?: ReadonlyMap<string, Uint32Array>;
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
  | 'emitterAlias'
  | 'emitterCount'
  | 'totalEmissivePower'
  | 'lightTree'
  | 'lightTreeNodeCount'
  | 'lightTreeEnabled'
>;
