/**
 * SceneBvh - core-native merged BVH wrapper for DDGI probe ray tracing.
 *
 * This root module is intentionally THREE-free. Legacy THREE scene ingestion
 * lives behind `@vitrum/shared-bvh/legacy/three`.
 */

import type { MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import {
  fingerprintPackedSceneBvhState,
  packedSceneBvhStateEqual,
  type PackedSceneBvhFingerprintState,
} from './bufferFingerprint.js';
import { clonePlainAabb, type PlainAabb } from './aabb.js';
import { materialSig } from './materialSignature.js';
import { mergeWorldSpaceFromCore } from './worldSpaceMerge.js';
import { coreMaterialToMaterialEntry, packMaterials } from './materialEntry.js';
import {
  analyzeOpticalMediumTopology,
  lowerTransmissiveAnalyticPrimitives,
} from './opticalMediumTopology.js';
import { packMergedOpticalMediumBoundaryIds } from './opticalMediumBoundaryPacking.js';

export interface SceneBvhBuffers {
  /** Flat BVHNode array: bounds (6 f32) + rightChild/triOffset + splitAxis/triCount. */
  bvhNodes: Float32Array;
  positions: Float32Array;
  /** Triangle index triplets (as u32 triples). */
  indices: Uint32Array;
  normals: Float32Array;
  /** One u32 per triangle, material index. */
  triMaterialId: Uint32Array;
  /** Interleaved vec2u optical component/range identity per BVH triangle. */
  opticalTriangleIdentity: Uint32Array;
  /** Encoded optical boundary base plus one for merged instance zero. */
  opticalInstanceBoundaryIdBasePlusOne: Uint32Array;
  /** Deduped core materials in scene order, indexed by {@link triMaterialId}. */
  materials: readonly MaterialSpec[];
  /**
   * Back-compat alias for older DDGI material packers. New consumers should use
   * {@link materials}; both fields point at the same core-native list.
   */
  coreMaterials?: readonly MaterialSpec[];
  /** BVH bounding box (world space). */
  boundingBox: PlainAabb;
}

export interface SceneBvhOptions {
  /**
   * Invoked when a BVH rebuild takes longer than 50 ms. When omitted, a
   * console.warn is emitted.
   */
  readonly onSlowRebuild?: (elapsedMs: number) => void;
  /**
   * Receives geometry/material warnings produced while merging the core scene.
   * When omitted, warnings are written to `console.warn` instead of being
   * silently discarded by this wrapper.
   */
  readonly onWarning?: (warning: string) => void;
}

export interface UpdateFromCoreOptions {
  /**
   * H34-h (D12) — monotonic scene-mutation tag supplied by the caller. When
   * provided AND unchanged since the last {@link SceneBvh.updateFromCore}
   * call, the expensive {@link mergeWorldSpaceFromCore} is skipped entirely
   * (the existing {@link SceneBvh.buffers} are kept as-is). Callers that do
   * not have a cheap tag available may omit this field; the content-fingerprint
   * fallback path (compare hashes of geometry + material bytes) is used
   * instead. The tag must change whenever the scene geometry or materials
   * change — monotonic integers (e.g. a mutation counter) are a safe choice.
   */
  readonly sceneVersionTag?: number | string;
}

const DDGI_CORE_MESH_FILTER = (p: ScenePrimitive): boolean =>
  p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh';

export class SceneBvh {
  protected _buffers: SceneBvhBuffers | null = null;
  private _lastCoreFingerprint = -1;
  private _lastMaterialEntries: Float32Array | null = null;
  private _lastMaterialSignatures: readonly string[] | null = null;
  /** H34-h — last sceneVersionTag seen; `undefined` means "no tag was supplied". */
  private _lastSceneVersionTag: number | string | undefined = undefined;

  protected readonly opts: SceneBvhOptions;

  constructor(opts: SceneBvhOptions = {}) {
    this.opts = opts;
  }

  get buffers(): SceneBvhBuffers | null {
    return this._buffers;
  }

  /**
   * Rebuild the DDGI merged BVH from a `@vitrum/core` Scene directly.
   * Geometry comes from `mergeWorldSpaceFromCore` at the same stride-4 layout
   * the DDGI WGSL probe pass reads (`array<vec3f>` = 16-byte stride).
   *
   * The slow-rebuild timer covers the entire scope of real work (merge + BVH
   * build + fingerprint) so that `onSlowRebuild` can actually fire. (Previously
   * the timer only measured an object-literal assignment — an ~0 µs no-op.)
   *
   * @param opts.sceneVersionTag — H34-h (D12): optional monotonic scene-mutation
   *   tag. When provided and UNCHANGED since the last call, the expensive merge
   *   is skipped and the existing buffers are returned as-is. Callers without a
   *   cheap tag may omit this; the fingerprint fallback is used instead.
   */
  updateFromCore(scene: Scene, opts: UpdateFromCoreOptions = {}): void {
    // H34-h — fast path: if the caller supplies a sceneVersionTag and it
    // matches the last seen value, skip the merge entirely. The buffers field
    // is already current (or null, which is the correct answer for an empty
    // scene that has not changed either).
    const { sceneVersionTag } = opts;
    if (
      sceneVersionTag !== undefined &&
      sceneVersionTag === this._lastSceneVersionTag
    ) {
      return;
    }

    // H34-g: start timing BEFORE the expensive merge so onSlowRebuild can fire.
    const t0 = performance.now();

    // Every authored-transmissive analytic is first lowered to the canonical
    // triangle representation used by exact source-feature exclusion. Analyze
    // and merge that same scene so topology IDs cannot drift from geometry.
    const transportScene = lowerTransmissiveAnalyticPrimitives(scene);
    const opticalAnalysis = analyzeOpticalMediumTopology(transportScene, {
      analyticGeometry: 'generated-triangle',
      transformArithmetic: 'merged-world-f64-to-f32',
    });
    const merged = mergeWorldSpaceFromCore(transportScene, {
      positionStride: 4,
      filter: DDGI_CORE_MESH_FILTER,
      splitMaterialsByCastShadow: true,
      onWarning: (warning) => {
        if (this.opts.onWarning) {
          try {
            this.opts.onWarning(warning);
          } catch {
            // Diagnostics must not interrupt BVH construction.
          }
        } else {
          console.warn(`[SceneBvh] ${warning}`);
        }
      },
    });

    if (merged.triangleCount === 0) {
      this._buffers = null;
      this._lastCoreFingerprint = -1;
      this._lastMaterialEntries = null;
      this._lastMaterialSignatures = null;
      // Record the tag for an empty scene too — next call with the same tag
      // still correctly returns null buffers via the fast path above.
      this._lastSceneVersionTag = sceneVersionTag;
      return;
    }

    // Fingerprint the exact buffers SceneBvh would publish and the exact
    // canonical MaterialEntry bytes the DDGI consumer uploads. In particular,
    // the material byte stream includes its true-u32 flags lane, so additions
    // to the shared GPU ABI cannot silently drift out of a parallel field list.
    const opticalIds = packMergedOpticalMediumBoundaryIds(
      transportScene,
      merged,
      opticalAnalysis,
    );
    const opticalTriangleIdentity = new Uint32Array(
      merged.triangleCount * 2,
    );
    for (let triangle = 0; triangle < merged.triangleCount; triangle += 1) {
      opticalTriangleIdentity[triangle * 2] =
        opticalIds.triangleComponentIndexPlusOne[triangle]!;
      opticalTriangleIdentity[triangle * 2 + 1] =
        opticalIds.triangleRepresentedPrimitiveInstanceIds[triangle]!;
    }
    const materialEntries = packMaterials(
      merged.materials.map(coreMaterialToMaterialEntry),
    );
    const materialSignatures = merged.materials.map(materialSig);
    const candidateState: PackedSceneBvhFingerprintState = {
      bvhNodes: merged.bvhNodes,
      positions: merged.positions,
      indices: merged.indices,
      normals: merged.normals,
      triMaterialId: merged.triMaterialId,
      opticalTriangleIdentity,
      opticalInstanceBoundaryIdBasePlusOne:
        opticalIds.instanceBoundaryIdBasePlusOne,
      materialEntries,
      materialSignatures,
    };
    const fingerprint = fingerprintPackedSceneBvhState(candidateState);
    const retainedState =
      this._buffers !== null &&
      this._lastMaterialEntries !== null &&
      this._lastMaterialSignatures !== null
        ? {
            bvhNodes: this._buffers.bvhNodes,
            positions: this._buffers.positions,
            indices: this._buffers.indices,
            normals: this._buffers.normals,
            triMaterialId: this._buffers.triMaterialId,
            opticalTriangleIdentity:
              this._buffers.opticalTriangleIdentity,
            opticalInstanceBoundaryIdBasePlusOne:
              this._buffers.opticalInstanceBoundaryIdBasePlusOne,
            materialEntries: this._lastMaterialEntries,
            materialSignatures: this._lastMaterialSignatures,
          } satisfies PackedSceneBvhFingerprintState
        : null;
    if (
      fingerprint === this._lastCoreFingerprint &&
      retainedState !== null &&
      packedSceneBvhStateEqual(candidateState, retainedState)
    ) {
      // Content fingerprint matched — update tag so the fast path fires on the
      // next call even when the caller switches from no-tag to tag mode.
      this._lastSceneVersionTag = sceneVersionTag;
      return;
    }
    this._lastCoreFingerprint = fingerprint;
    this._lastMaterialEntries = materialEntries;
    this._lastMaterialSignatures = materialSignatures;
    this._lastSceneVersionTag = sceneVersionTag;

    this._buffers = {
      bvhNodes: merged.bvhNodes,
      positions: merged.positions,
      indices: merged.indices,
      normals: merged.normals,
      triMaterialId: merged.triMaterialId,
      opticalTriangleIdentity,
      opticalInstanceBoundaryIdBasePlusOne:
        opticalIds.instanceBoundaryIdBasePlusOne,
      materials: merged.materials,
      coreMaterials: merged.materials,
      boundingBox: clonePlainAabb(merged.boundingBox),
    };

    const elapsed = performance.now() - t0;
    if (elapsed > 50) {
      if (this.opts.onSlowRebuild) {
        this.opts.onSlowRebuild(elapsed);
      } else {
        console.warn(`[DDGI SceneBvh] core BVH rebuild took ${elapsed.toFixed(0)}ms (>50ms threshold)`);
      }
    }
  }

  dispose(): void {
    this._buffers = null;
    this._lastCoreFingerprint = -1;
    this._lastMaterialEntries = null;
    this._lastMaterialSignatures = null;
    this._lastSceneVersionTag = undefined;
  }
}
