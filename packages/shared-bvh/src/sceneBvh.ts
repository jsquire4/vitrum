/**
 * SceneBvh - core-native merged BVH wrapper for DDGI probe ray tracing.
 *
 * This root module is intentionally THREE-free. Legacy THREE scene ingestion
 * lives behind `@vitrum/shared-bvh/legacy/three`.
 */

import type { MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import { fingerprintBuffersExact } from './bufferFingerprint.js';
import { clonePlainAabb, type PlainAabb } from './aabb.js';
import { mergeWorldSpaceFromCore } from './worldSpaceMerge.js';
import { MATERIAL_ATTEN_DIST_INFINITE } from './materialEntry.js';

export interface SceneBvhBuffers {
  /** Flat BVHNode array: bounds (6 f32) + rightChild/triOffset + splitAxis/triCount. */
  bvhNodes: Float32Array;
  positions: Float32Array;
  /** Triangle index triplets (as u32 triples). */
  indices: Uint32Array;
  normals: Float32Array;
  /** One u32 per triangle, material index. */
  triMaterialId: Uint32Array;
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

    const merged = mergeWorldSpaceFromCore(scene, {
      positionStride: 4,
      filter: DDGI_CORE_MESH_FILTER,
      splitMaterialsByCastShadow: true,
    });

    if (merged.triangleCount === 0) {
      this._buffers = null;
      this._lastCoreFingerprint = -1;
      // Record the tag for an empty scene too — next call with the same tag
      // still correctly returns null buffers via the fast path above.
      this._lastSceneVersionTag = sceneVersionTag;
      return;
    }

    const fingerprint = fingerprintBuffersExact(
      merged.positions.buffer as ArrayBuffer,
      merged.mergedIndices.buffer as ArrayBuffer,
      merged.mergedTriMaterialId.buffer as ArrayBuffer,
      new Float32Array(materialSetHashFloats(merged.materials)).buffer,
    );
    if (fingerprint === this._lastCoreFingerprint && this._buffers !== null) {
      // Content fingerprint matched — update tag so the fast path fires on the
      // next call even when the caller switches from no-tag to tag mode.
      this._lastSceneVersionTag = sceneVersionTag;
      return;
    }
    this._lastCoreFingerprint = fingerprint;
    this._lastSceneVersionTag = sceneVersionTag;

    this._buffers = {
      bvhNodes: merged.bvhNodes,
      positions: merged.positions,
      indices: merged.indices,
      normals: merged.normals,
      triMaterialId: merged.triMaterialId,
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
    this._lastSceneVersionTag = undefined;
  }
}

/**
 * Compact per-material float signature for the `updateFromCore` dirty
 * fingerprint. It hashes only the fields the DDGI probe pass consumes via the
 * packed `MaterialEntry`.
 */
function materialSetHashFloats(materials: readonly MaterialSpec[]): number[] {
  const out: number[] = [];
  for (const m of materials) {
    const ei = m.emissiveIntensity ?? 1;
    const em = m.emissive ?? [0, 0, 0];
    const bc = m.baseColor ?? [1, 1, 1];
    const ac = m.attenuationColor ?? [1, 1, 1];
    const attenuationDistance = m.attenuationDistance;
    const attenDistF =
      attenuationDistance === undefined ||
      !Number.isFinite(attenuationDistance) ||
      attenuationDistance <= 0
        ? MATERIAL_ATTEN_DIST_INFINITE
        : attenuationDistance;
    out.push(
      bc[0], bc[1], bc[2],
      em[0] * ei, em[1] * ei, em[2] * ei,
      m.roughness ?? 1,
      m.metallic ?? 0,
      m.transmission ?? 0,
      m.ior ?? 1.5,
      ac[0], ac[1], ac[2],
      attenDistF,
      m.thickness ?? 0,
    );
  }
  return out;
}
