import type { SceneBVHBuffers } from '../restir/bvhTypes.js';
import type { MaterialTextureAtlasPayload } from './materialTextureAtlas.js';
import type {
  BvhUpdateSink,
  TlasRefitMutation,
} from './BvhUpdateSink.js';

export interface CollectedBvhMutation {
  /** Every exact BLAS/merged-BVH node slice accumulated by the transaction. */
  readonly nodes?: ReadonlyArray<{
    readonly byteOffset: number;
    readonly data: ArrayBuffer;
  }>;
  /** Every disjoint CPU-authored position slice accumulated by the batch. */
  readonly positions?: ReadonlyArray<{ readonly byteOffset: number; readonly data: ArrayBuffer }>;
  readonly learningPositions?: ReadonlyArray<{ readonly byteOffset: number; readonly data: ArrayBuffer }>;
  /** Every disjoint CPU-authored normal slice accumulated by the batch. */
  readonly normals?: ReadonlyArray<{ readonly byteOffset: number; readonly data: ArrayBuffer }>;
  readonly tlas?: TlasRefitMutation;
  readonly replacement?: SceneBVHBuffers;
  readonly material?: {
    readonly index: { readonly byteOffset: number; readonly data: ArrayBuffer };
    readonly beer: { readonly data: ArrayBuffer; readonly triCount: number };
    readonly emissive: { readonly data: ArrayBuffer; readonly triCount: number };
    readonly roughMetal?: { readonly data: ArrayBuffer; readonly triCount: number };
  };
  readonly atlas?: MaterialTextureAtlasPayload;
  readonly emitters?: Pick<
    SceneBVHBuffers,
    'emitters' | 'emitterCdf' | 'emitterAlias' | 'lightTree' | 'lightTreeNodeCount' | 'lightTreeEnabled'
  >;
  readonly resetAccumulator: boolean;
}

function bytes(value: ArrayBuffer): ArrayBuffer {
  return value.slice(0);
}

/**
 * Records the existing incremental helper calls without publishing any GPU or
 * learned-state mutation. HybridEngine records against the live CPU BVH while
 * a bounded undo journal protects the touched ranges, then gives the resulting
 * immutable record to the pipeline transaction.
 */
export class CollectingBvhUpdateSink implements BvhUpdateSink {
  private readonly _nodes: Array<{ byteOffset: number; data: ArrayBuffer }> = [];
  private readonly _positions: Array<{ byteOffset: number; data: ArrayBuffer }> = [];
  private readonly _learningPositions: Array<{ byteOffset: number; data: ArrayBuffer }> = [];
  private readonly _normals: Array<{ byteOffset: number; data: ArrayBuffer }> = [];
  private _tlas: CollectedBvhMutation['tlas'];
  private _replacement: SceneBVHBuffers | undefined;
  private _material: CollectedBvhMutation['material'];
  private _atlas: MaterialTextureAtlasPayload | undefined;
  private _emitters: CollectedBvhMutation['emitters'];
  private _resetAccumulator = false;

  refreshBvhRefit(
    bvhNodesBytes: ArrayBuffer,
    positionsSlice: { byteOffset: number; data: ArrayBuffer },
    bvhNodesByteOffset = 0,
  ): void {
    this._nodes.push({
      byteOffset: bvhNodesByteOffset,
      data: bytes(bvhNodesBytes),
    });
    this._positions.push({
      byteOffset: positionsSlice.byteOffset,
      data: bytes(positionsSlice.data),
    });
  }

  refreshBvhNodesOnly(
    bvhNodesBytes: ArrayBuffer,
    bvhNodesByteOffset = 0,
  ): void {
    this._nodes.push({
      byteOffset: bvhNodesByteOffset,
      data: bytes(bvhNodesBytes),
    });
  }

  refreshBvhNormalsSlice(
    normalsSlice: { byteOffset: number; data: ArrayBuffer },
  ): void {
    this._normals.push({
      byteOffset: normalsSlice.byteOffset,
      data: bytes(normalsSlice.data),
    });
  }

  recordLearningBvhPositionsSlice(
    positionsSlice: { byteOffset: number; data: ArrayBuffer },
  ): void {
    this._learningPositions.push({
      byteOffset: positionsSlice.byteOffset,
      data: bytes(positionsSlice.data),
    });
  }

  refreshTlasRefit(mutation: TlasRefitMutation): void {
    this._tlas = {
      nodes: mutation.nodes.map((slice) => ({
        byteOffset: slice.byteOffset,
        data: bytes(slice.data),
      })),
      worldToLocal: mutation.worldToLocal.map((slice) => ({
        byteOffset: slice.byteOffset,
        data: bytes(slice.data),
      })),
      localToWorld: mutation.localToWorld.map((slice) => ({
        byteOffset: slice.byteOffset,
        data: bytes(slice.data),
      })),
    };
  }

  replaceBvhAndEmitters(bvhBuffers: SceneBVHBuffers): void {
    this._replacement = bvhBuffers;
  }

  updateEmitters(
    bvhBuffers: Pick<
      SceneBVHBuffers,
      'emitters' | 'emitterCdf' | 'emitterAlias' | 'lightTree' | 'lightTreeNodeCount' | 'lightTreeEnabled'
    >,
  ): void {
    this._emitters = bvhBuffers;
  }

  refreshBvhMaterialSlice(
    indexSlice: { byteOffset: number; data: ArrayBuffer },
    beerFull: { data: ArrayBuffer; triCount: number },
    emissiveFull: { data: ArrayBuffer; triCount: number },
    roughMetalFull?: { data: ArrayBuffer; triCount: number },
  ): void {
    this._material = {
      index: { byteOffset: indexSlice.byteOffset, data: bytes(indexSlice.data) },
      beer: { data: bytes(beerFull.data), triCount: beerFull.triCount },
      emissive: { data: bytes(emissiveFull.data), triCount: emissiveFull.triCount },
      ...(roughMetalFull
        ? { roughMetal: { data: bytes(roughMetalFull.data), triCount: roughMetalFull.triCount } }
        : {}),
    };
  }

  refreshMaterialTextureAtlas(materialTextureAtlas: MaterialTextureAtlasPayload): void {
    this._atlas = materialTextureAtlas;
  }

  requestAccumReset(): void {
    this._resetAccumulator = true;
  }

  snapshot(): CollectedBvhMutation {
    return {
      ...(this._nodes.length > 0 ? { nodes: [...this._nodes] } : {}),
      ...(this._positions.length > 0 ? { positions: [...this._positions] } : {}),
      ...(this._normals.length > 0 ? { normals: [...this._normals] } : {}),
      ...(this._learningPositions.length > 0 ? { learningPositions: [...this._learningPositions] } : {}),
      ...(this._tlas ? { tlas: this._tlas } : {}),
      ...(this._replacement ? { replacement: this._replacement } : {}),
      ...(this._material ? { material: this._material } : {}),
      ...(this._atlas ? { atlas: this._atlas } : {}),
      ...(this._emitters ? { emitters: this._emitters } : {}),
      resetAccumulator: this._resetAccumulator,
    };
  }
}
