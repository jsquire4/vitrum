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
 *    no pipeline recompile), rewrite the affected primitive's vertex slice
 *    in `bvhPositions` when using merged BVH, reset the accumulator, and
 *    invalidate DDGI probes because cached irradiance is anchored to the old
 *    object placement.
 *  - packed structural topology fields present (`uvs` / `tangents` /
 *    vertex-color streams or selection / `indices` / `castShadow`) → call
 *    {@link topologyRebuild}: re-run `buildReSTIRSceneBVH`, destroy +
 *    re-upload the four BVH GPU buffers, reset the accumulator.
 *  - wholesale primitive fields (`instances` / `params` / `shape` /
 *    `fallbackMesh`) are intercepted by `HybridEngine.updatePrimitive` and
 *    routed through `setScene`; changed `id` / `kind` discriminants are
 *    rejected canonically, while equal values are neutral.
 *  - `patch.positions` present, with optional same-count `normals` → call
 *    {@link positionsRefit}: update packed vertex data, refit bounds, and
 *    upload normals when provided.
 *  - skinned pose patches (`bones`, `boneInverses`, `morphWeights`) → call
 *    {@link skinnedPosePatch}: solve the pose, then reuse the positions/normals
 *    refit path when only geometry moved, or the topology rebuild path when
 *    morph-animated tangents/UVs need attribute-texture refresh, while preserving
 *    authored rest/pose fields in scene state.
 *  - material-only patches → {@link materialPatch}: re-pack bvhIndex,
 *    bvhBeerColors/material textures, and partial GPU upload (no setScene).
 *
 * The hot-path branch design is preserved from the pre-extraction code.
 */

import {
  type EngineWarning,
  solveSkin,
  type Mat4,
  type MaterialSpec,
  type MeshPrimitive,
  type Scene,
  type ScenePrimitive,
  type SkinnedMeshPrimitive,
} from '@vitrum/core';
import {
  computeLocalAabb,
  computeWorldAabbForBindings,
  invertMat4,
  isLeafSplit,
  refitBvhBounds,
  refitTlasTransforms,
  tlasRefitNodeIndices,
  type PrimitiveTlasBinding,
  type RefitTlasResult,
  type TlasGpuSnapshot,
} from '@vitrum/shared-bvh';
import { applyPrimitivePatchToScene } from './scenePatch.js';
import {
  buildReSTIRSceneBVHForCoreScene,
  rebuildReSTIRSceneBVHPrimitiveCore,
  rebuildEmitterBuffersFromCoreScene,
} from './restir/bvhCore.js';
import type { ReSTIRBvhMode, SceneBVHBuffers } from './restir/bvhCore.js';
import { packMaterialTextureAtlas } from './bvh/materialTextureAtlasPack.js';
import {
  needsAuthoredMorphStreamRestore,
  solvedSkinRenderPatch,
} from './skin/solvedSkinPatch.js';

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

import {
  packBVHIndexWFromCore,
  packBVHBeerColorsFromCore,
  packBVHEmissiveLeFromCore,
  packBVHRoughMetalFromCore,
} from './restir/packingHelpers.js';
import {
  collectUnconsumedMaterialFieldsForMaterial,
  type UnconsumedMaterialPrimitiveFields,
} from './restir/consumedMaterialFields.js';
import type { BvhUpdateSink } from './pipeline/BvhUpdateSink.js';
import type { DDGI } from './ddgi/DDGI.js';

const LAYERED_MATERIAL_KEYS = ['frontLayer', 'backLayer'] as const;

// ── Shared refit helpers (behaviour-preserving extraction, WD sweep) ─────────
//
// `transformRefit`, `positionsRefit`, and `refitSkinnedMeshAfterGpuWrite` all
// repeated three byte-for-byte blocks: (1) snapshot the live TLAS GPU buffers,
// (2) write a successful `refitTlasTransforms` result back into `bvh.tlas.*` +
// push it to the pipeline, and (3) refit BVH node bounds against an updated
// world-position buffer and upload the affected stride-4 vertex slice. These
// helpers fold each block into one call — no behaviour change (the call
// sequence + GPU side effects are identical to the inlined code).

const REFIT_STRIDE = 4; // bvhPositions packs world xyz into [0..2] + uv-as-u32 in [3]

function f32Copy(values: ArrayLike<number>): Float32Array {
  return new Float32Array(values);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value) && !ArrayBuffer.isView(value);
}

function hasOwn(value: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function mergeMaterialPatch(
  base: MaterialSpec,
  patch: Partial<MaterialSpec>,
): MaterialSpec {
  const baseRecord = base as unknown as Record<string, unknown>;
  const patchRecord = patch as unknown as Record<string, unknown>;
  const merged: Record<string, unknown> = { ...baseRecord, ...patchRecord };
  for (const key of LAYERED_MATERIAL_KEYS) {
    if (!hasOwn(patchRecord, key)) continue;
    const baseLayer = baseRecord[key];
    const patchLayer = patchRecord[key];
    merged[key] = isRecord(baseLayer) && isRecord(patchLayer)
      ? { ...baseLayer, ...patchLayer }
      : patchLayer;
  }
  return merged as unknown as MaterialSpec;
}

/**
 * Compute the inverse-transpose of the upper-3×3 of a column-major 4×4
 * matrix. Returns the 9 elements in column-major order as a flat array
 * [c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2].
 *
 * Required for correct normal transformation under non-uniform scale:
 *   n_world = normalize( (M^{-T}) * n_local )
 * where M is the upper-3×3. Under uniform scale or pure rotation the
 * inverse-transpose equals the rotation part — this path is always correct.
 *
 * Singular guard: if det ≈ 0 the matrix cannot be inverted (degenerate
 * scale), so we fall back to the raw rotation submatrix (upper-3×3 of M)
 * and let the normalization step handle any length change.
 */
function mat3InverseTransposeFromMat4(m: ArrayLike<number>): Float32Array {
  // Column-major extraction of upper-3×3.
  const m00 = m[0]!; const m10 = m[1]!; const m20 = m[2]!;
  const m01 = m[4]!; const m11 = m[5]!; const m21 = m[6]!;
  const m02 = m[8]!; const m12 = m[9]!; const m22 = m[10]!;

  // Cofactor matrix (transposed inverse numerator).
  const c00 = m11 * m22 - m21 * m12;
  const c01 = m20 * m12 - m10 * m22;
  const c02 = m10 * m21 - m20 * m11;
  const c10 = m21 * m02 - m01 * m22;
  const c11 = m00 * m22 - m20 * m02;
  const c12 = m20 * m01 - m00 * m21;
  const c20 = m01 * m12 - m11 * m02;
  const c21 = m10 * m02 - m00 * m12;
  const c22 = m00 * m11 - m10 * m01;

  const det = m00 * c00 + m01 * c01 + m02 * c02;

  if (Math.abs(det) < 1e-12) {
    // Degenerate scale — fall back to the raw upper-3×3 (rotation part).
    // normalize() in the caller will unit-length the result.
    return new Float32Array([m00, m10, m20, m01, m11, m21, m02, m12, m22]);
  }

  const invDet = 1 / det;
  // The inverse-transpose of M3 is the cofactor matrix / det (no extra
  // transpose needed because cofactor rows are already the transposed-inverse
  // columns).
  return new Float32Array([
    c00 * invDet, c10 * invDet, c20 * invDet,
    c01 * invDet, c11 * invDet, c21 * invDet,
    c02 * invDet, c12 * invDet, c22 * invDet,
  ]);
}

const IDENTITY_MAT4 = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

function matrixFromArrayLike(values: ArrayLike<number> | undefined): Float32Array {
  if (values == null || values.length < 16) return new Float32Array(IDENTITY_MAT4);
  const out = new Float32Array(16);
  for (let i = 0; i < 16; i += 1) out[i] = values[i] ?? 0;
  return out;
}

function mat4Multiply(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const ae = a, be = b;
  const a11 = ae[0]!, a12 = ae[4]!, a13 = ae[8]!, a14 = ae[12]!;
  const a21 = ae[1]!, a22 = ae[5]!, a23 = ae[9]!, a24 = ae[13]!;
  const a31 = ae[2]!, a32 = ae[6]!, a33 = ae[10]!, a34 = ae[14]!;
  const a41 = ae[3]!, a42 = ae[7]!, a43 = ae[11]!, a44 = ae[15]!;
  const b11 = be[0]!, b12 = be[4]!, b13 = be[8]!, b14 = be[12]!;
  const b21 = be[1]!, b22 = be[5]!, b23 = be[9]!, b24 = be[13]!;
  const b31 = be[2]!, b32 = be[6]!, b33 = be[10]!, b34 = be[14]!;
  const b41 = be[3]!, b42 = be[7]!, b43 = be[11]!, b44 = be[15]!;
  return new Float32Array([
    a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41,
    a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41,
    a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41,
    a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41,
    a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42,
    a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42,
    a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42,
    a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42,
    a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43,
    a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43,
    a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43,
    a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43,
    a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44,
    a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44,
    a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44,
    a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44,
  ]);
}

function transformPoint(
  m: ArrayLike<number>,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  const w = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!;
  const invW = w !== 0 && Number.isFinite(w) ? 1 / w : 1;
  return [
    (m[0]! * x + m[4]! * y + m[8]! * z + m[12]!) * invW,
    (m[1]! * x + m[5]! * y + m[9]! * z + m[13]!) * invW,
    (m[2]! * x + m[6]! * y + m[10]! * z + m[14]!) * invW,
  ];
}

/**
 * Patch fields whose presence forces the full SAH `topologyRebuild` path
 * (Option (a)) rather than a transform/positions fast path. `positions` is
 * deliberately excluded — it has its own count-preserving refit fast path
 * (which falls through to a rebuild internally on a vertex-count mismatch).
 *
 * Single source of truth: the `updatePrimitive` dispatcher in
 * `HybridEngine.ts` imports this to decide routing; the routing-rules comments
 * in both files reference the same list.
 */
export const TOPOLOGY_PATCH_FIELDS = [
  'normals', 'uvs', 'uv1', 'uvSets', 'tangents',
  'colors', 'colorSets', 'vertexColorSet', 'indices',
  'instances', 'params', 'shape', 'fallbackMesh', 'castShadow',
] as const;

/** Topology fields that require wholesale scene replacement rather than a
 *  count-preserving packed-buffer update. `HybridEngine.updatePrimitive`
 *  intercepts these and routes them through `setScene` (instead of
 *  `topologyRebuild` throwing). */
export const TOPOLOGY_PATCH_WHOLESALE_FIELDS = [
  'instances', 'params', 'shape', 'fallbackMesh',
] as const;

export const SKIN_POSE_PATCH_FIELDS = [
  'skinIndices', 'skinWeights', 'skinInfluencesPerVertex',
  'bones', 'boneInverses', 'bindMatrix', 'bindMatrixInverse',
  'morphTargets', 'morphTargetNormals', 'morphTargetTangents',
  'morphTargetUvs', 'morphTargetUv1s', 'morphTargetUvSets',
  'morphTargetColors', 'morphTargetColorSets', 'morphWeights',
] as const;

/** Authored rest streams on a skinned primitive. Public patches to any of
 * these fields must be solved against the primitive's current pose before the
 * render scene/BVH is updated. */
export const SKIN_REST_STREAM_PATCH_FIELDS = [
  'positions', 'normals', 'tangents', 'uvs', 'uv1', 'uvSets',
  'colors', 'colorSets',
] as const;

/** Snapshot the live TLAS GPU buffers as the `prev` input to `refitTlasTransforms`. */
function captureTlasSnapshot(tlas: NonNullable<SceneBVHBuffers['tlas']>): TlasGpuSnapshot {
  return {
    tlasNodes: new Uint32Array(tlas.nodes.cpuData),
    tlasInstanceIndices: new Uint32Array(tlas.instanceIndices.cpuData),
    tlasBlasRoots: new Uint32Array(tlas.blasRoots.cpuData),
    tlasInstanceWorldToLocal: new Float32Array(tlas.worldToLocal.cpuData),
    tlasInstanceLocalToWorld: new Float32Array(tlas.localToWorld.cpuData),
  };
}

type TlasRefitView = Uint32Array | Float32Array;

function copyTlasView(view: TlasRefitView): ArrayBuffer {
  const copy = new Uint8Array(view.byteLength);
  copy.set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
  return copy.buffer;
}

function publishTlasRefitSlices(
  target: ArrayBuffer,
  source: TlasRefitView,
  ranges: readonly BvhNodeByteRange[],
): ReadonlyArray<{ readonly byteOffset: number; readonly data: ArrayBuffer }> {
  if (target.byteLength !== source.byteLength) {
    throw new RangeError(
      '[walkaround-hybrid] partial TLAS refit changed a live buffer extent.',
    );
  }
  for (const range of ranges) {
    if (
      range.byteOffset < 0 ||
      range.byteLength < 0 ||
      range.byteOffset + range.byteLength > target.byteLength
    ) {
      throw new RangeError(
        '[walkaround-hybrid] partial TLAS refit produced an out-of-range slice.',
      );
    }
  }
  return ranges.map((range) => {
    const data = new Uint8Array(range.byteLength);
    data.set(new Uint8Array(
      source.buffer,
      source.byteOffset + range.byteOffset,
      range.byteLength,
    ));
    new Uint8Array(target, range.byteOffset, range.byteLength).set(data);
    return {
      byteOffset: range.byteOffset,
      data: data.buffer,
    };
  });
}

/** Write a successful TLAS-transform refit back into `bvh.tlas.*` and push the
 *  three refreshed buffers to the pipeline. Mutates `bvh.tlas` in place
 *  (matching the pre-extraction call sites). */
function applyTlasRefitResult(
  tlas: NonNullable<SceneBVHBuffers['tlas']>,
  refit: Extract<RefitTlasResult, { readonly ok: true }>,
  pipeline: BvhUpdateSink | null | undefined,
): void {
  const dirtyNodes = refit.dirtyTlasNodeIndices;
  const dirtyTransforms = refit.dirtyTlasTransformInstanceIndices;
  if (dirtyNodes != null && dirtyTransforms != null) {
    const nodeRanges = coalesceFixedByteRanges(dirtyNodes, 32);
    const matrixRanges = coalesceFixedByteRanges(dirtyTransforms, 64);
    // Validate every extent before publishing the first byte so a malformed
    // result cannot compromise the caller's undo baseline mid-apply.
    if (
      tlas.nodes.cpuData.byteLength !== refit.tlasNodes.byteLength ||
      tlas.worldToLocal.cpuData.byteLength !==
        refit.tlasInstanceWorldToLocal.byteLength ||
      tlas.localToWorld.cpuData.byteLength !==
        refit.tlasInstanceLocalToWorld.byteLength
    ) {
      throw new RangeError(
        '[walkaround-hybrid] partial TLAS refit changed a live buffer extent.',
      );
    }
    const nodes = publishTlasRefitSlices(
      tlas.nodes.cpuData,
      refit.tlasNodes,
      nodeRanges,
    );
    const worldToLocal = publishTlasRefitSlices(
      tlas.worldToLocal.cpuData,
      refit.tlasInstanceWorldToLocal,
      matrixRanges,
    );
    const localToWorld = publishTlasRefitSlices(
      tlas.localToWorld.cpuData,
      refit.tlasInstanceLocalToWorld,
      matrixRanges,
    );
    pipeline?.refreshTlasRefit({
      nodes,
      worldToLocal,
      localToWorld,
    });
    return;
  }
  tlas.nodes.cpuData = copyTlasView(refit.tlasNodes);
  tlas.worldToLocal.cpuData = copyTlasView(refit.tlasInstanceWorldToLocal);
  tlas.localToWorld.cpuData = copyTlasView(refit.tlasInstanceLocalToWorld);
  pipeline?.refreshTlasRefit({
    nodes: [{ byteOffset: 0, data: tlas.nodes.cpuData }],
    worldToLocal: [{ byteOffset: 0, data: tlas.worldToLocal.cpuData }],
    localToWorld: [{ byteOffset: 0, data: tlas.localToWorld.cpuData }],
  });
}

function blasNodeByteRange(
  bvh: SceneBVHBuffers,
  blasRoot: number,
): { byteOffset: number; byteLength: number } {
  const totalNodes = bvh.bvhNodes.cpuData.byteLength / 32;
  let endNode = totalNodes;
  for (const candidate of bvh.primitiveTlasBindings) {
    if (candidate.blasRoot > blasRoot && candidate.blasRoot < endNode) {
      endNode = candidate.blasRoot;
    }
  }
  if (
    !Number.isSafeInteger(blasRoot) ||
    blasRoot < 0 ||
    blasRoot >= endNode ||
    endNode > totalNodes
  ) {
    throw new Error(
      `[walkaround-hybrid] invalid BLAS node slice [${blasRoot}, ${endNode}).`,
    );
  }
  return {
    byteOffset: blasRoot * 32,
    byteLength: (endNode - blasRoot) * 32,
  };
}

interface BvhNodeByteRange {
  readonly byteOffset: number;
  readonly byteLength: number;
}

function coalesceFixedByteRanges(
  indices: ArrayLike<number>,
  stride: number,
): readonly BvhNodeByteRange[] {
  const ascending = [...new Set(Array.from(indices))].sort((a, b) => a - b);
  const ranges: BvhNodeByteRange[] = [];
  for (const index of ascending) {
    const byteOffset = index * stride;
    const previous = ranges[ranges.length - 1];
    if (
      previous != null &&
      previous.byteOffset + previous.byteLength === byteOffset
    ) {
      ranges[ranges.length - 1] = {
        byteOffset: previous.byteOffset,
        byteLength: previous.byteLength + stride,
      };
    } else {
      ranges.push({ byteOffset, byteLength: stride });
    }
  }
  return ranges;
}

function coalesceNodeByteRanges(
  nodeIndices: ArrayLike<number>,
): readonly BvhNodeByteRange[] {
  return coalesceFixedByteRanges(nodeIndices, 32);
}

function refitMergedNodeSubset(
  bvh: SceneBVHBuffers,
  positions: Float32Array,
  nodesPostorder: Uint32Array,
): readonly BvhNodeByteRange[] {
  const f32 = new Float32Array(bvh.bvhNodes.cpuData);
  const u32 = new Uint32Array(bvh.bvhNodes.cpuData);
  const indices = bvh.bvhIndicesStride3;
  const totalNodes = f32.length / 8;
  const triangleCount = indices.length / 3;
  const vertexCount = positions.length / 4;
  let previous = Number.POSITIVE_INFINITY;
  for (const node of nodesPostorder) {
    if (node >= totalNodes || node >= previous) {
      throw new Error(
        '[walkaround-hybrid] primitive refit nodes must be unique descending indices.',
      );
    }
    previous = node;
    const base = node * 8;
    const splitOrCount = u32[base + 7]!;
    let mnX = Number.POSITIVE_INFINITY;
    let mnY = Number.POSITIVE_INFINITY;
    let mnZ = Number.POSITIVE_INFINITY;
    let mxX = Number.NEGATIVE_INFINITY;
    let mxY = Number.NEGATIVE_INFINITY;
    let mxZ = Number.NEGATIVE_INFINITY;
    if (isLeafSplit(splitOrCount)) {
      const count = splitOrCount & 0xffff;
      const triangleOffset = u32[base + 6]!;
      if (triangleOffset > triangleCount || count > triangleCount - triangleOffset) {
        throw new Error('[walkaround-hybrid] primitive refit leaf range is corrupt.');
      }
      if (count === 0) {
        mnX = 0; mnY = 0; mnZ = 0;
        mxX = 0; mxY = 0; mxZ = 0;
      }
      for (let tri = triangleOffset; tri < triangleOffset + count; tri += 1) {
        for (let lane = 0; lane < 3; lane += 1) {
          const vertex = indices[tri * 3 + lane]!;
          if (vertex >= vertexCount) {
            throw new Error('[walkaround-hybrid] primitive refit vertex is out of bounds.');
          }
          const offset = vertex * 4;
          const x = positions[offset]!;
          const y = positions[offset + 1]!;
          const z = positions[offset + 2]!;
          if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
            throw new Error('[walkaround-hybrid] primitive refit vertex is non-finite.');
          }
          mnX = Math.min(mnX, x); mnY = Math.min(mnY, y); mnZ = Math.min(mnZ, z);
          mxX = Math.max(mxX, x); mxY = Math.max(mxY, y); mxZ = Math.max(mxZ, z);
        }
      }
    } else {
      const left = node + 1;
      const right = node + u32[base + 6]!;
      if (left >= totalNodes || right <= node || right >= totalNodes) {
        throw new Error('[walkaround-hybrid] primitive refit interior node is corrupt.');
      }
      for (const child of [left, right]) {
        const childBase = child * 8;
        mnX = Math.min(mnX, f32[childBase]!);
        mnY = Math.min(mnY, f32[childBase + 1]!);
        mnZ = Math.min(mnZ, f32[childBase + 2]!);
        mxX = Math.max(mxX, f32[childBase + 3]!);
        mxY = Math.max(mxY, f32[childBase + 4]!);
        mxZ = Math.max(mxZ, f32[childBase + 5]!);
      }
    }
    f32[base] = mnX; f32[base + 1] = mnY; f32[base + 2] = mnZ;
    f32[base + 3] = mxX; f32[base + 4] = mxY; f32[base + 5] = mxZ;
  }
  return coalesceNodeByteRanges(nodesPostorder);
}

/** Refit BVH node bounds against the updated stride-4 positions, then upload
 *  only the affected BLAS node bytes plus the affected vertex slice. Merged-BVH
 *  callers have one root and therefore intentionally pass the complete tree. */
function refitBvhNodesAndUploadSlice(
  bvh: SceneBVHBuffers,
  positionsF32: Float32Array,
  baseVertex: number,
  sliceVerts: number,
  pipeline: BvhUpdateSink | null | undefined,
  blasRoot?: number,
  primitiveId?: string,
): void {
  let nodeRanges: readonly BvhNodeByteRange[];
  const selectiveNodes = primitiveId == null
    ? undefined
    : bvh.primitiveRefitNodeIndices?.get(primitiveId);
  if (blasRoot !== undefined) {
    const range = blasNodeByteRange(bvh, blasRoot);
    refitBvhBounds(
      new Float32Array(
        bvh.bvhNodes.cpuData,
        range.byteOffset,
        range.byteLength / 4,
      ),
      bvh.bvhIndicesStride3,
      positionsF32,
      4,
    );
    nodeRanges = [range];
  } else if (selectiveNodes != null && selectiveNodes.length > 0) {
    nodeRanges = refitMergedNodeSubset(bvh, positionsF32, selectiveNodes);
  } else {
    const range = {
      byteOffset: 0,
      byteLength: bvh.bvhNodes.cpuData.byteLength,
    };
    refitBvhBounds(
      new Float32Array(bvh.bvhNodes.cpuData),
      bvh.bvhIndicesStride3,
      positionsF32,
      4,
    );
    nodeRanges = [range];
  }
  const positionsByteOffset = baseVertex * REFIT_STRIDE * 4; // f32 = 4 bytes
  const positionsByteLength = sliceVerts * REFIT_STRIDE * 4;
  const positionsSlice = bvh.bvhPositions.cpuData.slice(
    positionsByteOffset,
    positionsByteOffset + positionsByteLength,
  );
  const [firstNodeRange, ...remainingNodeRanges] = nodeRanges;
  if (firstNodeRange != null) {
    pipeline?.refreshBvhRefit(
      bvh.bvhNodes.cpuData.slice(
        firstNodeRange.byteOffset,
        firstNodeRange.byteOffset + firstNodeRange.byteLength,
      ),
      { byteOffset: positionsByteOffset, data: positionsSlice },
      firstNodeRange.byteOffset,
    );
  }
  for (const range of remainingNodeRanges) {
    pipeline?.refreshBvhNodesOnly(
      bvh.bvhNodes.cpuData.slice(
        range.byteOffset,
        range.byteOffset + range.byteLength,
      ),
      range.byteOffset,
    );
  }
}

/**
 * H19 — apply the inverse-transpose of the upper-3×3 of a 4×4 column-major
 * matrix to per-vertex normals, then upload the affected slice to the GPU.
 *
 * Two modes controlled by the optional `inputNormals` argument:
 *  - **Rotate in-place** (`inputNormals` absent): reads existing world-space
 *    normals from `bvhNormals.cpuData` and applies the transform delta (used
 *    by `transformRefit` when the mesh has been rigidly moved).
 *  - **Write from local-space** (`inputNormals` provided, stride-3): writes
 *    caller-supplied local-space normals after transforming to world space
 *    (used by the positions-patch path that supplies new normals alongside
 *    new positions).
 *
 * The inverse-transpose is required for correctness under non-uniform scale
 * (plain upper-3×3 would mis-orient normals perpendicular to scaled axes).
 * Under uniform scale or pure rotation the paths are equivalent.
 *
 * Output: stride-4 (vec4f, .w=0). Skipped when `pipeline` is null.
 */
function applyNormalTransformAndUpload(
  bvh: SceneBVHBuffers,
  mat4: ArrayLike<number>,
  baseVertex: number,
  sliceVerts: number,
  pipeline: BvhUpdateSink | null | undefined,
  inputNormals?: ArrayLike<number>,
): void {
  const NORM_STRIDE = 4; // vec4f per vertex (.w unused)
  const normalsF32 = new Float32Array(bvh.bvhNormals.cpuData);
  // Inverse-transpose of the upper-3×3 for correct normal transform under
  // non-uniform scale. Falls back to the raw submatrix when degenerate.
  const it = mat3InverseTransposeFromMat4(mat4);
  const r00 = it[0]!; const r10 = it[1]!; const r20 = it[2]!;
  const r01 = it[3]!; const r11 = it[4]!; const r21 = it[5]!;
  const r02 = it[6]!; const r12 = it[7]!; const r22 = it[8]!;
  for (let v = 0; v < sliceVerts; v++) {
    const off = (baseVertex + v) * NORM_STRIDE;
    // Source: caller-supplied local-space (stride-3) or existing world-space buffer.
    const nx = inputNormals != null ? inputNormals[v * 3]!     : normalsF32[off]!;
    const ny = inputNormals != null ? inputNormals[v * 3 + 1]! : normalsF32[off + 1]!;
    const nz = inputNormals != null ? inputNormals[v * 3 + 2]! : normalsF32[off + 2]!;
    let wx = r00 * nx + r01 * ny + r02 * nz;
    let wy = r10 * nx + r11 * ny + r12 * nz;
    let wz = r20 * nx + r21 * ny + r22 * nz;
    const len = Math.sqrt(wx * wx + wy * wy + wz * wz);
    if (len > 1e-12) { wx /= len; wy /= len; wz /= len; }
    normalsF32[off]     = wx;
    normalsF32[off + 1] = wy;
    normalsF32[off + 2] = wz;
    if (inputNormals != null) normalsF32[off + 3] = 0; // .w=0 when writing fresh
    // When rotating in-place: .w already=0 from build time, left unchanged.
  }
  const byteOffset = baseVertex * NORM_STRIDE * 4;
  const byteLength = sliceVerts * NORM_STRIDE * 4;
  const sliceData = bvh.bvhNormals.cpuData.slice(byteOffset, byteOffset + byteLength);
  pipeline?.refreshBvhNormalsSlice({ byteOffset, data: sliceData });
}


/** Aggregated resources the primitive-update paths need from the engine. */
export interface PrimitiveUpdateContext {
  /** The engine's currently-owned BVH GPU buffers. May be null if the
   *  pipeline init has not yet published — callers (transformRefit) MUST
   *  fall back to a topology rebuild in that case. */
  readonly bvhBuffers: SceneBVHBuffers | null;
  /** Live GPU pipeline; may be null during init. The fast paths fall
   *  through to a rebuild when null. Typed as `BvhUpdateSink` to decouple
   *  the update helpers from the full pipeline class (complexity sweep
   *  2026-06-02). `WalkaroundGPUPipeline implements BvhUpdateSink`. */
  readonly pipeline: BvhUpdateSink | null;
  /** DDGI subsystem; receives probe-cache invalidation calls. */
  readonly ddgi: DDGI;
  /** Primary directional light dir, threaded into a rebuild's BVH-builder. */
  readonly primaryLightDir: readonly [number, number, number];
  /** Primary directional light intensity, threaded into a rebuild's
   *  BVH-builder. */
  readonly primaryLightIntensity: number;
  /** Current vitrum scene snapshot — kept in sync on successful fast paths. */
  readonly lastScene: Scene;
  /** Mesh-like render-ingestion scene matching the current BVH pack. Backends
   *  that accept authored analytic primitives convert them to generated mesh
   *  fallbacks here, so TLAS refit / BVH rebuild helpers always see the same
   *  primitive kinds that were packed into {@link PrimitiveTlasBinding}. */
  readonly renderScene: Scene;
  /** Whether the engine's render scene supplies core mesh/skinned/instanced
   *  primitive payloads, allowing incremental emitter rebuilds. */
  readonly coreSceneSuppliesMeshes?: boolean;
  /** Emits backend-honesty warnings for material-only patches that carry fields
   *  the walkaround material packer does not consume. */
  readonly warnUnconsumedMaterialFields?: (
    fields: readonly string[],
    primitiveFields?: readonly UnconsumedMaterialPrimitiveFields[],
  ) => void;
  /** Structured warning sink for BVH/emitter rebuild compatibility issues. */
  readonly onWarning?: (warning: EngineWarning) => void;
  /** Suppress publication/invalidation while HybridEngine prepares a transaction. */
  readonly deferSubsystemSideEffects?: boolean;
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
  /** Deformed/derived scene used by render-facing subsystems when it differs. */
  readonly updatedRenderScene?: Scene;

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
  /**
   * Whether the engine should apply the GI-subsystem propagation epilogue
   * (`_applyPrimitiveUpdateSubsystems`) after swapping in this result.
   *
   * Geometry paths (transform / positions / topology / skinned refit) DO —
   * they changed the BVH that DDGI + RC index off. The material-only fast path
   * sets this `false`: it re-packs material/emissive slices in place without
   * moving geometry. Material-only paths may still request a DDGI material
   * snapshot refresh so probe rays see the new BRDF/emissive payload. Absent
   * ⇒ apply (the geometry-path default).
   */
  readonly applySubsystems?: boolean;
  /** Material-only paths changed RC-visible material/emitter data without moving geometry. */
  readonly refreshRcMaterials?: boolean;
  /** Material-only paths changed DDGI-visible material data without moving geometry. */
  readonly refreshDdgiMaterialSnapshot?: boolean;
}
export interface PrimitiveMutationUndo {
  /** Bytes retained solely for rollback; exposed so complexity tests can pin the fast path. */
  readonly retainedByteLength: number;
  restore(): void;
  accept(): void;
}

function restoreBytes(target: ArrayBuffer, offset: number, source: Uint8Array): void {
  new Uint8Array(target, offset, source.byteLength).set(source);
}

/**
 * Capture only CPU bytes an incremental primitive helper may overwrite.
 *
 * Geometry refits retain the affected BLAS/TLAS node records plus the affected
 * position/normal slice. Transform refits additionally retain only the affected
 * 64-byte TLAS matrix slots. Material edits retain the already-full small
 * per-triangle payloads they repack. No vertex/index/atlas/scene-wide payload is
 * copied merely to establish the transaction boundary.
 */
export function capturePrimitiveMutationUndo(
  bvh: SceneBVHBuffers | null,
  id: string,
  patch: Partial<ScenePrimitive>,
  additionalIds: readonly string[] = [],
): PrimitiveMutationUndo {
  if (bvh == null) {
    return { retainedByteLength: 0, restore: () => undefined, accept: () => undefined };
  }
  const targetIds = [id, ...additionalIds];

  const defined = (key: string): boolean =>
    (patch as unknown as Record<string, unknown>)[key] !== undefined;
  const hasMaterial = defined('material');
  const hasGeometry =
    SKIN_POSE_PATCH_FIELDS.some(defined) ||
    defined('transform') ||
    defined('positions') ||
    defined('normals') ||
    TOPOLOGY_PATCH_FIELDS.some((field) => field !== 'normals' && defined(field));
  const materialOnly = hasMaterial && !hasGeometry;
  const hasSkinPose = SKIN_POSE_PATCH_FIELDS.some(defined);
  const refitsBlas =
    hasSkinPose ||
    defined('positions') ||
    defined('normals') ||
    (defined('transform') && bvh.bvhMode !== 'tlas');
  const mutatesPositions =
    hasSkinPose ||
    defined('positions') ||
    (defined('transform') && bvh.bvhMode !== 'tlas');
  const mutatesNormals =
    hasSkinPose ||
    defined('normals') ||
    (defined('transform') && bvh.bvhMode !== 'tlas');
  const refitsTlas =
    bvh.bvhMode === 'tlas' &&
    (hasSkinPose ||
      defined('transform') ||
      defined('positions') ||
      defined('normals'));

  const snapshots: Array<{
    readonly target: () => ArrayBuffer;
    readonly offset: number;
    readonly bytes: Uint8Array;
  }> = [];
  const save = (
    target: () => ArrayBuffer,
    offset = 0,
    requestedByteLength?: number,
  ): void => {
    const buffer = target();
    const byteLength = requestedByteLength ?? buffer.byteLength - offset;
    snapshots.push({
      target,
      offset,
      bytes: new Uint8Array(buffer.slice(offset, offset + byteLength)),
    });
  };

  const rangeMatrices: Array<{ readonly target: Float32Array; readonly bytes: Float32Array }> = [];

  if (!materialOnly && hasGeometry) {
    if (refitsBlas) {
      if (bvh.bvhMode === 'tlas') {
        const roots = new Set<number>();
        for (const targetId of targetIds) {
          const binding = bvh.primitiveTlasBindings.find(
            (candidate) => candidate.primitiveId === targetId,
          );
          if (binding != null) roots.add(binding.blasRoot);
        }
        for (const root of [...roots].sort((a, b) => a - b)) {
          const range = blasNodeByteRange(bvh, root);
          save(
            () => bvh.bvhNodes.cpuData,
            range.byteOffset,
            range.byteLength,
          );
        }
      } else {
        const affectedNodes = new Set<number>();
        let hasCompleteMetadata = true;
        for (const targetId of targetIds) {
          const nodes = bvh.primitiveRefitNodeIndices?.get(targetId);
          if (nodes == null || nodes.length === 0) {
            hasCompleteMetadata = false;
            break;
          }
          for (const node of nodes) affectedNodes.add(node);
        }
        if (hasCompleteMetadata) {
          for (const range of coalesceNodeByteRanges([...affectedNodes])) {
            save(
              () => bvh.bvhNodes.cpuData,
              range.byteOffset,
              range.byteLength,
            );
          }
        } else {
          // Compatibility fallback for externally-constructed test/adapter
          // buffers that predate the build-time primitive refit index.
          save(() => bvh.bvhNodes.cpuData);
        }
      }
    }
    for (const targetId of targetIds) {
      const range = bvh.meshVertexRanges.find((candidate) => candidate.name === targetId);
      const binding = bvh.primitiveTlasBindings.find(
        (candidate) => candidate.primitiveId === targetId,
      );
      const vertexStart = binding?.vertexStart ?? range?.vertexStart;
      const vertexCount = binding?.vertexCount ?? range?.vertexCount;
      if (vertexStart != null && vertexCount != null && vertexCount > 0) {
        const byteOffset = vertexStart * REFIT_STRIDE * 4;
        const byteLength = vertexCount * REFIT_STRIDE * 4;
        if (mutatesPositions) {
          save(() => bvh.bvhPositions.cpuData, byteOffset, byteLength);
        }
        if (mutatesNormals) {
          save(() => bvh.bvhNormals.cpuData, byteOffset, byteLength);
        }
      }
      if (range != null && defined('transform')) {
        rangeMatrices.push({
          target: range.matrixWorldAtBuild,
          bytes: new Float32Array(range.matrixWorldAtBuild),
        });
      }
    }
    if (refitsTlas && bvh.tlas != null) {
      const targetIdSet = new Set(targetIds);
      const targetInstanceIndices: number[] = [];
      let instanceStart = 0;
      for (const binding of bvh.primitiveTlasBindings) {
        if (targetIdSet.has(binding.primitiveId)) {
          for (let i = 0; i < binding.instanceCount; i += 1) {
            targetInstanceIndices.push(instanceStart + i);
          }
        }
        instanceStart += binding.instanceCount;
      }
      if (targetInstanceIndices.length > 0) {
        const affectedNodes = tlasRefitNodeIndices(
          {
            nodes: new Uint32Array(bvh.tlas.nodes.cpuData),
            nodeCount: bvh.tlas.nodeCount,
            instanceIndices: new Uint32Array(bvh.tlas.instanceIndices.cpuData),
            blasRoots: new Uint32Array(bvh.tlas.blasRoots.cpuData),
            instanceTransforms: new Float32Array(
              bvh.tlas.worldToLocal.cpuData,
            ),
          },
          targetInstanceIndices,
        );
        for (const range of coalesceNodeByteRanges(affectedNodes)) {
          save(
            () => bvh.tlas!.nodes.cpuData,
            range.byteOffset,
            range.byteLength,
          );
        }
        if (defined('transform')) {
          for (const range of coalesceFixedByteRanges(targetInstanceIndices, 64)) {
            save(
              () => bvh.tlas!.worldToLocal.cpuData,
              range.byteOffset,
              range.byteLength,
            );
            save(
              () => bvh.tlas!.localToWorld.cpuData,
              range.byteOffset,
              range.byteLength,
            );
          }
        }
      }
    }
  }

  let active = true;
  const retainedByteLength =
    snapshots.reduce((sum, snapshot) => sum + snapshot.bytes.byteLength, 0) +
    rangeMatrices.reduce((sum, matrix) => sum + matrix.bytes.byteLength, 0);
  return {
    retainedByteLength,
    restore: () => {
      if (!active) return;
      active = false;
      for (const snapshot of snapshots) {
        restoreBytes(snapshot.target(), snapshot.offset, snapshot.bytes);
      }
      for (const matrix of rangeMatrices) matrix.target.set(matrix.bytes);
    },
    accept: () => {
      active = false;
    },
  };
}


function findSkinnedPrimitive(scene: Scene, id: string): SkinnedMeshPrimitive | null {
  const primitive = scene.primitives.find((p) => String(p.id) === id);
  return primitive?.kind === 'skinned-mesh' ? primitive : null;
}

/**
 * Skinned-pose mutation path for host patches that update skeleton state,
 * skinning inputs, bind matrices, or morph-target definitions/weights.
 *
 * The renderer buffers still need deformed positions/normals, so this resolves
 * the next skinned pose through the canonical CPU solver and then delegates to
 * the existing geometry refit path. The authored scene keeps submitted rest/pose
 * fields, while the separate render scene carries the solved geometry. Mixed
 * transform+pose patches rebuild conservatively so the solved vertices and the
 * BVH transform snapshot are published atomically.
 */
export function skinnedPosePatch(
  id: string,
  patch: Partial<ScenePrimitive>,
  ctx: PrimitiveUpdateContext,
): PrimitiveUpdateResult {
  const current = findSkinnedPrimitive(ctx.lastScene, id);
  if (current == null) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): skin/morph patches ` +
      `require a skinned-mesh primitive.`,
    );
  }

  const nextPrimitive = { ...current, ...patch } as SkinnedMeshPrimitive;
  const solved = solveSkin(nextPrimitive);
  const rendered = findSkinnedPrimitive(ctx.renderScene, id);
  const renderPatch = solvedSkinRenderPatch(
    nextPrimitive,
    solved,
    needsAuthoredMorphStreamRestore(nextPrimitive, rendered),
  );
  const resolvedPatch = {
    ...patch,
    ...renderPatch,
  } as Partial<ScenePrimitive>;

  const patchRecord = patch as unknown as Record<string, unknown>;
  const currentRecord = current as unknown as Record<string, unknown>;
  // A defined topology value replaces a stream as before. An OWN undefined
  // value is also structural when it removes a currently-authored optional
  // stream (UV/color/tangent/index/etc.); treating that as omission would leave
  // the preceding GPU attribute payload alive after the scene field is cleared.
  const hasStructuralPatch = TOPOLOGY_PATCH_FIELDS.some(
    (field) =>
      field !== 'normals' &&
      (
        patchRecord[field] !== undefined ||
        (
          Object.prototype.hasOwnProperty.call(patchRecord, field) &&
          currentRecord[field] !== undefined
        )
      ),
  );
  if (
    patch.material !== undefined ||
    (patch as { transform?: unknown }).transform !== undefined ||
    hasStructuralPatch ||
    renderPatch.tangents != null ||
    renderPatch.uvs != null ||
    renderPatch.uv1 != null ||
    renderPatch.uvSets != null ||
    renderPatch.colors != null ||
    renderPatch.colorSets != null
  ) {
    const result = topologyRebuild(id, resolvedPatch, ctx);
    return {
      ...result,
      // SkinnedMeshPrimitive geometry fields are authored rest-pose data.
      // Publish the submitted authored patch to the host scene; solved streams
      // remain private to the render scene.
      updatedScene: applyPrimitivePatchToScene(ctx.lastScene, id, patch),
      updatedRenderScene: applyPrimitivePatchToScene(
        ctx.renderScene,
        id,
        resolvedPatch,
      ),
    };
  }

  const result = positionsRefit(id, resolvedPatch, ctx);
  return {
    ...result,
    updatedScene: applyPrimitivePatchToScene(ctx.lastScene, id, patch),
    updatedRenderScene: applyPrimitivePatchToScene(
      ctx.renderScene,
      id,
      resolvedPatch,
    ),
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
 *  1. Read the affected primitive's cached matrix snapshot from the packed BVH.
 *  2. Compute the matrix delta `D = matrixWorldNew · matrixWorldAtBuild⁻¹`.
 *  3. For each vertex `v` in `[vertexStart, vertexStart + vertexCount)`,
 *     read the old world-space position from `bvhPositions.cpuData`,
 *     apply `D`, write the new world-space position back. (UV in `.w`
 *     is preserved.)
 *  4. Update `matrixWorldAtBuild` snapshot to the new matrix world.
 *  5. Run `refitBvhBounds` on the BVH node buffer.
 *  6. Upload the refit nodes + position slice via the pipeline.
 *  7. Reset the accumulator (history is invalid — the primitive moved).
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
      meshPatch.uv1 === undefined &&
      meshPatch.uvSets === undefined &&
      meshPatch.tangents === undefined &&
      meshPatch.indices === undefined;
    if (transformOnly && bvh.tlas != null && bvh.primitiveTlasBindings.length > 0) {
      const updatedScene = applyPrimitivePatchToScene(ctx.lastScene, id, {
        transform: meshPatch.transform,
      });
      const updatedRenderScene = applyPrimitivePatchToScene(ctx.renderScene, id, {
        transform: meshPatch.transform,
      });
      const prev = captureTlasSnapshot(bvh.tlas);
      const refit = refitTlasTransforms(
        updatedRenderScene,
        bvh.primitiveTlasBindings,
        prev,
        { primitiveId: id },
      );
      if (refit.ok) {
        applyTlasRefitResult(bvh.tlas, refit, ctx.pipeline);
        const range = bvh.meshVertexRanges.find((r) => r.name === id);
        if (range != null && meshPatch.transform && meshPatch.transform.length >= 16) {
          range.matrixWorldAtBuild.set(matrixFromArrayLike(meshPatch.transform));
        }
        ctx.pipeline?.requestAccumReset();
        if (!ctx.deferSubsystemSideEffects) {
          ctx.ddgi.markInstancesDirty();
          ctx.ddgi.invalidateProbeCache();
        }
        const rcBounds = computeWorldAabbForBindings(
          updatedRenderScene,
          bvh.primitiveTlasBindings,
        );
        return {
          bvhBuffers: bvh,
          updatedScene,
          updatedRenderScene,
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

  // Apply the new transform. The Scene contract says transform is a
  // 16-element column-major Mat4 (see core/src/scene.ts:MeshPrimitive).
  const transform = (patch as { transform?: ArrayLike<number> }).transform;
  const newMat = matrixFromArrayLike(transform);

  // Compute matrix delta D = newMat · oldMat⁻¹. We transform each
  // already-baked world-space vertex through D to get the new
  // world-space vertex; equivalent to local⁻¹ → new-world round-trip
  // but without storing local-space positions.
  const oldMatWorld = matrixFromArrayLike(range.matrixWorldAtBuild);
  const oldMatWorldInv = invertMat4(oldMatWorld as Mat4);
  if (oldMatWorldInv == null) {
    return topologyRebuild(id, patch, ctx);
  }
  const delta = mat4Multiply(newMat, oldMatWorldInv);

  // Rewrite the affected vertex slice of bvhPositions.cpuData. The
  // stride-4 layout packs world-space xyz into [0..2] and UV-as-u32
  // into [3] (preserved here). Use a single typed-array view over the
  // shared ArrayBuffer so the changes land in cpuData.
  const positionsF32 = new Float32Array(bvh.bvhPositions.cpuData);
  const STRIDE = 4;
  const baseVertex = range.vertexStart;
  const sliceVerts = range.vertexCount;
  for (let v = 0; v < sliceVerts; v++) {
    const off = (baseVertex + v) * STRIDE;
    const [x, y, z] = transformPoint(
      delta,
      positionsF32[off + 0]!,
      positionsF32[off + 1]!,
      positionsF32[off + 2]!,
    );
    positionsF32[off + 0] = x;
    positionsF32[off + 1] = y;
    positionsF32[off + 2] = z;
    // .w (UV pack) preserved.
  }

  // Update the matrix snapshot in-place so subsequent transform
  // patches compute their delta against the latest matrix, not the
  // original build-time matrix.
  range.matrixWorldAtBuild.set(newMat);

  // Refit BVH bounds in place against the freshly-updated positions (using
  // the cached stride-3 index buffer), then upload only affected node ranges
  // plus the affected stride-4 vertex slice.
  refitBvhNodesAndUploadSlice(
    bvh,
    positionsF32,
    baseVertex,
    sliceVerts,
    ctx.pipeline,
    undefined,
    id,
  );

  // H19 — apply the same rotation delta to bvhNormals so smooth-shading
  // normals stay correct after a transform refit (skin path exempt — the GPU
  // skin kernel writes normals directly every frame).
  applyNormalTransformAndUpload(bvh, delta, baseVertex, sliceVerts, ctx.pipeline);

  // Reset the accumulator — temporal history is invalid because the
  // primitive moved (history pixels reference the old world position).
  ctx.pipeline?.requestAccumReset();
  // DDGI probes baked their irradiance against the old position;
  // invalidate so probes re-converge over the next STRIDE frames.
  if (!ctx.deferSubsystemSideEffects) ctx.ddgi.invalidateProbeCache();

  const meshPatch = patch as Partial<MeshPrimitive>;
  const updatedScene =
    meshPatch.transform !== undefined
      ? applyPrimitivePatchToScene(ctx.lastScene, id, { transform: meshPatch.transform })
      : ctx.lastScene;
  const updatedRenderScene =
    meshPatch.transform !== undefined
      ? applyPrimitivePatchToScene(ctx.renderScene, id, { transform: meshPatch.transform })
      : ctx.renderScene;

  const rcBounds = computeWorldAabbFromBvhPositions(bvh);
  return {
    bvhBuffers: bvh,
    updatedScene,
    updatedRenderScene,
    ...(rcBounds != null ? { rcRefitBounds: rcBounds } : {}),
  };
}

/**
 * Positions-only refit fast path (A3 — 2026-05-18).
 *
 * When `patch.positions` is the ONLY geometry field touched (no
 * `normals` / `uvs` / `uv1` / `tangents` / `indices` / `instances` / `params` /
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

  // A merged BVH flattens every instance into a separate same-id vertex range.
  // The single-range refit below cannot update all of them; rebuild so one
  // shared authored geometry patch is expanded through every instance matrix.
  if (
    bvh.bvhMode === 'merged' &&
    ctx.renderScene.primitives.find((primitive) => String(primitive.id) === id)
      ?.kind === 'instanced-mesh'
  ) {
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

    const localAabb = computeLocalAabb(f32Copy(newLocalPositions));
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

    refitBvhNodesAndUploadSlice(
      bvh,
      positionsF32,
      baseVertex,
      sliceVerts,
      ctx.pipeline,
      binding.blasRoot,
    );

    // H19 — TLAS BLAS slices store local-space normals. When a count-preserving
    // positions patch also supplies replacement normals, upload the matching
    // bvhNormals slice so smooth-shading reads the new deformed normals instead
    // of the build-time data.
    const meshPosPatch0 = patch as Partial<MeshPrimitive>;
    if (meshPosPatch0.normals !== undefined && meshPosPatch0.normals.length === sliceVerts * 3) {
      applyNormalTransformAndUpload(
        bvh,
        IDENTITY_MAT4,
        baseVertex,
        sliceVerts,
        ctx.pipeline,
        meshPosPatch0.normals,
      );
    }

    const meshPosPatch = patch as Partial<MeshPrimitive>;
    const posPatch: Partial<MeshPrimitive> = meshPosPatch.normals !== undefined
      ? {
          positions: f32Copy(newLocalPositions),
          normals: f32Copy(meshPosPatch.normals),
        }
      : { positions: f32Copy(newLocalPositions) };
    const updatedScene = applyPrimitivePatchToScene(ctx.lastScene, id, posPatch);
    const updatedRenderScene = applyPrimitivePatchToScene(ctx.renderScene, id, posPatch);

    const prev = captureTlasSnapshot(bvh.tlas);
    const refit = refitTlasTransforms(
      updatedRenderScene,
      bindings,
      prev,
      { primitiveId: id, localBoundsChanged: true },
    );
    if (refit.ok) {
      applyTlasRefitResult(bvh.tlas, refit, ctx.pipeline);
    } else {
      return topologyRebuild(id, patch, ctx);
    }

    ctx.pipeline?.requestAccumReset();
    if (!ctx.deferSubsystemSideEffects) ctx.ddgi.invalidateProbeCache();
    const rcBounds = computeWorldAabbForBindings(updatedRenderScene, bindings);
    const outBvh: SceneBVHBuffers = { ...bvh, primitiveTlasBindings: bindings };
    return {
      bvhBuffers: outBvh,
      updatedScene,
      updatedRenderScene,
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

  // The BVH stores WORLD-space positions in a stride-4 layout
  // ([x, y, z, uvPacked] per vertex). Apply the cached matrixWorldAtBuild
  // to lift the new local positions into world space, preserving the .w
  // (UV pack) lane from the existing slice.
  const matWorld = matrixFromArrayLike(range.matrixWorldAtBuild);
  const positionsF32 = new Float32Array(bvh.bvhPositions.cpuData);
  const STRIDE = 4;
  const baseVertex = range.vertexStart;
  const sliceVerts = range.vertexCount;
  for (let v = 0; v < sliceVerts; v++) {
    const off = (baseVertex + v) * STRIDE;
    const [x, y, z] = transformPoint(
      matWorld,
      newLocalPositions[v * 3] ?? 0,
      newLocalPositions[v * 3 + 1] ?? 0,
      newLocalPositions[v * 3 + 2] ?? 0,
    );
    positionsF32[off + 0] = x;
    positionsF32[off + 1] = y;
    positionsF32[off + 2] = z;
    // .w (UV pack) preserved.
  }

  // Refit BVH bounds in place against the freshly-updated positions, then
  // upload only affected node ranges + the affected vertex slice to GPU.
  refitBvhNodesAndUploadSlice(
    bvh,
    positionsF32,
    baseVertex,
    sliceVerts,
    ctx.pipeline,
    undefined,
    id,
  );

  // H19 — if the positions patch also carries new local normals, transform them
  // to world space and upload the affected normals slice. Without this the GPU
  // bvhNormals buffer keeps the build-time normals even after vertex positions
  // move (smooth-shading references stale normals until a topology rebuild).
  const meshPosPatch0 = patch as Partial<MeshPrimitive>;
  if (meshPosPatch0.normals !== undefined && meshPosPatch0.normals.length === sliceVerts * 3) {
    applyNormalTransformAndUpload(
      bvh, matWorld, baseVertex, sliceVerts, ctx.pipeline, meshPosPatch0.normals);
  }

  // Reset the accumulator + invalidate DDGI — vertex positions changed,
  // history pixels reference the old geometry. Same invalidation cost as
  // transformRefit.
  ctx.pipeline?.requestAccumReset();
  if (!ctx.deferSubsystemSideEffects) ctx.ddgi.invalidateProbeCache();

  const meshPosPatch = patch as Partial<MeshPrimitive>;
  const posPatch: Partial<MeshPrimitive> = meshPosPatch.normals !== undefined
      ? {
        positions: f32Copy(newLocalPositions),
        normals: f32Copy(meshPosPatch.normals),
      }
    : { positions: f32Copy(newLocalPositions) };
  const updatedScene = applyPrimitivePatchToScene(ctx.lastScene, id, posPatch);
  const updatedRenderScene = applyPrimitivePatchToScene(ctx.renderScene, id, posPatch);

  return { bvhBuffers: bvh, updatedScene, updatedRenderScene };
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

  const posPatch: Partial<MeshPrimitive> =
    localNormals != null
      ? { positions: new Float32Array(localPositions), normals: new Float32Array(localNormals) }
      : { positions: new Float32Array(localPositions) };
  const updatedRenderScene = applyPrimitivePatchToScene(ctx.renderScene, id, posPatch);
  const updatedScene = ctx.lastScene;

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
    const positionByteOffset = baseVertex * STRIDE * 4;
    const positionByteLength = sliceVerts * STRIDE * 4;
    ctx.pipeline?.recordLearningBvhPositionsSlice?.({
      byteOffset: positionByteOffset,
      data: bvh.bvhPositions.cpuData.slice(
        positionByteOffset, positionByteOffset + positionByteLength,
      ),
    });

    if (localNormals != null && localNormals.length === sliceVerts * 3) {
      // TLAS skinning writes local-space normals. Mirror the exact stride-4
      // bytes into the CPU BVH shadow without staging a second GPU upload.
      applyNormalTransformAndUpload(
        bvh, IDENTITY_MAT4, baseVertex, sliceVerts, null, localNormals,
      );
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

    const nodeRange = blasNodeByteRange(bvh, binding.blasRoot);
    const bvhNodesF32 = new Float32Array(
      bvh.bvhNodes.cpuData,
      nodeRange.byteOffset,
      nodeRange.byteLength / 4,
    );
    refitBvhBounds(bvhNodesF32, bvh.bvhIndicesStride3, positionsF32, 4);
    ctx.pipeline?.refreshBvhNodesOnly(
      bvh.bvhNodes.cpuData.slice(
        nodeRange.byteOffset,
        nodeRange.byteOffset + nodeRange.byteLength,
      ),
      nodeRange.byteOffset,
    );

    const prev = captureTlasSnapshot(bvh.tlas);
    const refit = refitTlasTransforms(
      updatedRenderScene,
      bindings,
      prev,
      { primitiveId: id, localBoundsChanged: true },
    );
    if (!refit.ok) {
      throw new Error(`refitSkinnedMeshAfterGpuWrite("${id}"): TLAS transform refit failed.`);
    }
    applyTlasRefitResult(bvh.tlas, refit, ctx.pipeline);

    ctx.pipeline?.requestAccumReset();
    if (!ctx.deferSubsystemSideEffects) {
      ctx.ddgi.markInstancesDirty();
      ctx.ddgi.invalidateProbeCache();
    }
    const rcBounds = computeWorldAabbForBindings(updatedRenderScene, bindings);
    const outBvh: SceneBVHBuffers = { ...bvh, primitiveTlasBindings: bindings };
    return {
      bvhBuffers: outBvh,
      updatedScene,
      ...(rcBounds != null ? { rcRefitBounds: rcBounds } : {}),
      updatedRenderScene,
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

  const matWorld = matrixFromArrayLike(range.matrixWorldAtBuild);
  const positionsF32 = new Float32Array(bvh.bvhPositions.cpuData);
  const STRIDE = 4;
  const baseVertex = range.vertexStart;
  const sliceVerts = range.vertexCount;
  for (let v = 0; v < sliceVerts; v += 1) {
    const off = (baseVertex + v) * STRIDE;
    const [x, y, z] = transformPoint(
      matWorld,
      localPositions[v * 3] ?? 0,
      localPositions[v * 3 + 1] ?? 0,
      localPositions[v * 3 + 2] ?? 0,
    );
    positionsF32[off + 0] = x;
    positionsF32[off + 1] = y;
    positionsF32[off + 2] = z;
  }

  const positionByteOffset = baseVertex * STRIDE * 4;
  const positionByteLength = sliceVerts * STRIDE * 4;
  ctx.pipeline?.recordLearningBvhPositionsSlice?.({
    byteOffset: positionByteOffset,
    data: bvh.bvhPositions.cpuData.slice(
      positionByteOffset, positionByteOffset + positionByteLength,
    ),
  });
  const selectiveNodes = bvh.primitiveRefitNodeIndices?.get(id);
  let nodeRanges: readonly BvhNodeByteRange[];
  if (selectiveNodes != null && selectiveNodes.length > 0) {
    nodeRanges = refitMergedNodeSubset(bvh, positionsF32, selectiveNodes);
  } else {
    refitBvhBounds(
      new Float32Array(bvh.bvhNodes.cpuData),
      bvh.bvhIndicesStride3,
      positionsF32,
      4,
    );
    nodeRanges = [{
      byteOffset: 0,
      byteLength: bvh.bvhNodes.cpuData.byteLength,
    }];
  }
  if (localNormals != null && localNormals.length === sliceVerts * 3) {
    // The merged kernel applies matrixWorld's inverse-transpose after skinning.
    // Keep the CPU normal mirror byte-identical, but let the prefix compute
    // command remain the sole GPU writer for this slice.
    applyNormalTransformAndUpload(
      bvh, matWorld, baseVertex, sliceVerts, null, localNormals,
    );
  }
  for (const nodeRange of nodeRanges) {
    ctx.pipeline?.refreshBvhNodesOnly(
      bvh.bvhNodes.cpuData.slice(
        nodeRange.byteOffset,
        nodeRange.byteOffset + nodeRange.byteLength,
      ),
      nodeRange.byteOffset,
    );
  }
  ctx.pipeline?.requestAccumReset();
  if (!ctx.deferSubsystemSideEffects) ctx.ddgi.invalidateProbeCache();

  return { bvhBuffers: bvh, updatedScene, updatedRenderScene };
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
  // For now we support the most common topology patches:
  //   - transform (16-element Mat4)
  //   - positions, normals, uvs, uv1, uvSets, tangents, colors, colorSets,
  //     vertexColorSet, indices (typed arrays / semantic selectors from
  //     core/src/scene/primitives.ts MeshPrimitive)
  // Other fields (`instances`, `params`, `shape`, `fallbackMesh`) require a
  // wholesale primitive replacement. `HybridEngine.
  // updatePrimitive` intercepts these and routes them through setScene
  // BEFORE reaching here, so the throw below is now a defensive backstop
  // for any direct caller — not the host-facing path.
  const p = patch as {
    transform?: ArrayLike<number>;
    positions?: ArrayLike<number>;
    normals?: ArrayLike<number>;
    uvs?: ArrayLike<number>;
    uv1?: ArrayLike<number>;
    uvSets?: ReadonlyArray<ArrayLike<number> | undefined>;
    tangents?: ArrayLike<number>;
    colors?: ArrayLike<number>;
    colorSets?: ReadonlyArray<ArrayLike<number> | undefined>;
    vertexColorSet?: number | null;
    indices?: ArrayLike<number>;
    instances?: unknown;
    params?: unknown;
    shape?: unknown;
    fallbackMesh?: unknown;
    kind?: unknown;
    id?: unknown;
  };
  for (const f of TOPOLOGY_PATCH_WHOLESALE_FIELDS) {
    if (p[f] !== undefined) {
      throw new Error(
        `topologyRebuild("${id}"): '${f}' is a wholesale-replacement field and must ` +
        `be routed through setScene by HybridEngine.updatePrimitive before reaching ` +
        `topologyRebuild (internal invariant).`,
      );
    }
  }

  // Rebuild the BVH from the patched core scene. The old buffers are
  // released after the new ones are uploaded.
  const updatedScene = applyPrimitivePatchToScene(ctx.lastScene, id, patch);
  const updatedRenderScene = applyPrimitivePatchToScene(ctx.renderScene, id, patch);
  const oldBuffers = ctx.bvhBuffers;
  const bvhOpts = {
    primaryLightDir: {
      x: ctx.primaryLightDir[0],
      y: ctx.primaryLightDir[1],
      z: ctx.primaryLightDir[2],
    },
    primaryLightIntensity: ctx.primaryLightIntensity,
    ...(ctx.restirBvhModeOverride !== undefined
      ? { bvhMode: ctx.restirBvhModeOverride }
      : {}),
    ...(ctx.onWarning !== undefined
      ? { onWarning: ctx.onWarning, warningPhase: 'mutation' as const, warningMethod: 'updatePrimitive' }
      : {}),
  };
  let newBuffers: SceneBVHBuffers;
  if (
    oldBuffers != null &&
    oldBuffers.bvhMode === 'tlas' &&
    oldBuffers.scenePack != null
  ) {
    const rebuilt = rebuildReSTIRSceneBVHPrimitiveCore(
      updatedRenderScene,
      id,
      oldBuffers,
      bvhOpts,
    );
    newBuffers =
      'ok' in rebuilt && rebuilt.ok === false
        ? buildReSTIRSceneBVHForCoreScene(updatedRenderScene, bvhOpts)
        : (rebuilt as SceneBVHBuffers);
  } else {
    newBuffers = buildReSTIRSceneBVHForCoreScene(updatedRenderScene, bvhOpts);
  }
  // Publish the complete GPU replacement only after every allocation succeeds.
  ctx.pipeline?.replaceBvhAndEmitters(newBuffers);



  // Reset the accumulator + invalidate DDGI — geometry topology
  // changed, history is meaningless.
  ctx.pipeline?.requestAccumReset();
  if (!ctx.deferSubsystemSideEffects) ctx.ddgi.invalidateProbeCache();

  const rcBounds =
    newBuffers.bvhMode === 'tlas' && newBuffers.primitiveTlasBindings.length > 0
      ? computeWorldAabbForBindings(updatedRenderScene, newBuffers.primitiveTlasBindings)
      : null;

  return {
    bvhBuffers: newBuffers,
    updatedScene,
    updatedRenderScene,
    ...(rcBounds != null ? { rcRefitBounds: rcBounds } : {}),
  };
}

const TRANSMISSION_GLASS_THRESHOLD = 0.01;

function vitrumMaterialTransmission(material: MaterialSpec | undefined): number {
  return material?.transmission ?? 0;
}

function productionEmissiveRadiance(material: MaterialSpec | undefined): readonly [number, number, number] {
  const emissive = material?.emissive;
  if (emissive == null) return [0, 0, 0];
  const intensity = material?.emissiveIntensity ?? 1;
  return [emissive[0] * intensity, emissive[1] * intensity, emissive[2] * intensity];
}

function productionEmissiveRadianceChanged(
  prev: MaterialSpec | undefined,
  next: MaterialSpec | undefined,
): boolean {
  const a = productionEmissiveRadiance(prev);
  const b = productionEmissiveRadiance(next);
  return a[0] !== b[0] || a[1] !== b[1] || a[2] !== b[2];
}

function color3Changed(
  prev: readonly [number, number, number] | undefined,
  next: readonly [number, number, number] | undefined,
  fallback: readonly [number, number, number],
): boolean {
  const a = prev ?? fallback;
  const b = next ?? fallback;
  return a[0] !== b[0] || a[1] !== b[1] || a[2] !== b[2];
}

function materialRadianceOrVisibilityChanged(
  prev: MaterialSpec | undefined,
  next: MaterialSpec | undefined,
): boolean {
  const prevTransmission = vitrumMaterialTransmission(prev);
  const nextTransmission = vitrumMaterialTransmission(next);
  return prevTransmission > TRANSMISSION_GLASS_THRESHOLD
    || nextTransmission > TRANSMISSION_GLASS_THRESHOLD
    || productionEmissiveRadianceChanged(prev, next)
    || color3Changed(prev?.attenuationColor, next?.attenuationColor, [1, 1, 1])
    || (prev?.attenuationDistance ?? Infinity) !== (next?.attenuationDistance ?? Infinity)
    || (prev?.thickness ?? 0) !== (next?.thickness ?? 0)
    || color3Changed(prev?.baseColor, next?.baseColor, [1, 1, 1]);
}

function materialAffectsDdgiProbeCache(
  prev: MaterialSpec | undefined,
  next: MaterialSpec | undefined,
): boolean {
  return materialRadianceOrVisibilityChanged(prev, next)
    || (prev?.roughness ?? 0.5) !== (next?.roughness ?? 0.5)
    || (prev?.metallic ?? 0) !== (next?.metallic ?? 0);
}

function textureRefLike(value: unknown): {
  readonly handle: unknown;
  readonly texCoord?: number;
  readonly transform?: {
    readonly offset?: readonly [number, number];
    readonly scale?: readonly [number, number];
    readonly rotation?: number;
  };
  readonly wrapS?: string;
  readonly wrapT?: string;
  readonly magFilter?: string;
  readonly minFilter?: string;
  readonly mipFilter?: string;
} | null {
  if (value == null || typeof value !== 'object') return null;
  if ('handle' in value) {
    return value;
  }
  return { handle: value };
}

function uv2Component(
  value: readonly [number, number] | undefined,
  index: 0 | 1,
  fallback: number,
): number {
  return value?.[index] ?? fallback;
}

const ATLAS_MATERIAL_MAP_FIELDS = [
  'baseColorMap',
  'normalMap',
  'roughnessMap',
  'metallicMap',
  'aoMap',
  'alphaMap',
  'emissiveMap',
  'transmissionMap',
  'lightMap',
  'specularColorMap',
  'specularIntensityMap',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'anisotropyMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'thicknessMap',
  'bumpMap',
] as const;

function textureMapPatchRequiresFullRebuild(
  prev: MaterialSpec | undefined,
  next: MaterialSpec | undefined,
  field: (typeof ATLAS_MATERIAL_MAP_FIELDS)[number],
): boolean {
  return textureRefPatchChanged(prev?.[field], next?.[field]);
}

function textureRefPatchChanged(
  prevValue: unknown,
  nextValue: unknown,
): boolean {
  const a = textureRefLike(prevValue);
  const b = textureRefLike(nextValue);
  if (a == null || b == null) return a !== b;
  if (a.handle !== b.handle) return true;
  if ((a.texCoord ?? 0) !== (b.texCoord ?? 0)) return true;
  if ((a.wrapS ?? 'repeat') !== (b.wrapS ?? 'repeat')) return true;
  if ((a.wrapT ?? 'repeat') !== (b.wrapT ?? 'repeat')) return true;
  if ((a.magFilter ?? 'unspecified') !== (b.magFilter ?? 'unspecified')) return true;
  if ((a.minFilter ?? 'unspecified') !== (b.minFilter ?? 'unspecified')) return true;
  if ((a.mipFilter ?? 'unspecified') !== (b.mipFilter ?? 'unspecified')) return true;
  const at = a.transform;
  const bt = b.transform;
  if (uv2Component(at?.offset, 0, 0) !== uv2Component(bt?.offset, 0, 0)) return true;
  if (uv2Component(at?.offset, 1, 0) !== uv2Component(bt?.offset, 1, 0)) return true;
  if (uv2Component(at?.scale, 0, 1) !== uv2Component(bt?.scale, 0, 1)) return true;
  if (uv2Component(at?.scale, 1, 1) !== uv2Component(bt?.scale, 1, 1)) return true;
  return (at?.rotation ?? 0) !== (bt?.rotation ?? 0);
}

export function materialPatchAffectsDisplacementGeometry(
  prev: MaterialSpec | undefined,
  patch: Partial<MaterialSpec>,
): boolean {
  const next: MaterialSpec =
    prev != null
      ? mergeMaterialPatch(prev, patch)
      : patch as MaterialSpec;
  if (textureRefPatchChanged(prev?.displacementMap, next.displacementMap)) return true;
  const prevHasMap = textureRefLike(prev?.displacementMap) != null;
  const nextHasMap = textureRefLike(next.displacementMap) != null;
  if (!prevHasMap && !nextHasMap) return false;
  return (prev?.displacementScale ?? 1) !== (next.displacementScale ?? 1) ||
    (prev?.displacementBias ?? 0) !== (next.displacementBias ?? 0) ||
    (prev?.displacementSubdivisions ?? 0) !==
      (next.displacementSubdivisions ?? 0);
}

function alphaModeAtlasIndex(mode: MaterialSpec['alphaMode'] | undefined): number {
  switch (mode ?? 'opaque') {
    case 'mask':
      return 1;
    case 'blend':
      return 2;
    case 'opaque':
      return 0;
  }
}

function alphaAtlasUnit(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value ?? fallback)) : fallback;
}

function minClampedUnit(value: number | undefined, min: number, fallback: number): number {
  return Number.isFinite(value) ? Math.max(min, value ?? fallback) : fallback;
}

function nonNegativeScalar(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? Math.max(0, value ?? fallback) : fallback;
}

function iridescenceThicknessBound(
  material: MaterialSpec | undefined,
  index: 0 | 1,
  fallback: number,
): number {
  const value = material?.iridescenceThicknessRange?.[index];
  return Number.isFinite(value) ? Math.max(0, value ?? fallback) : fallback;
}

function colorUnit(
  material: MaterialSpec | undefined,
  field: 'specularColor' | 'sheenColor',
  index: 0 | 1 | 2,
  fallback: number,
): number {
  const value = material?.[field]?.[index];
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value ?? fallback)) : fallback;
}

function materialAtlasPatchRequiresFullRebuild(
  prev: MaterialSpec | undefined,
  next: MaterialSpec | undefined,
): boolean {
  const mapChanged = ATLAS_MATERIAL_MAP_FIELDS.some((field) =>
    textureMapPatchRequiresFullRebuild(prev, next, field),
  );
  if (mapChanged) return true;
  const normalScaleChanged =
    (prev?.normalMap != null || next?.normalMap != null) &&
    (prev?.normalScale ?? 1) !== (next?.normalScale ?? 1);
  const clearcoatNormalScaleChanged =
    (prev?.clearcoatNormalMap != null || next?.clearcoatNormalMap != null) &&
    (prev?.clearcoatNormalScale ?? 1) !== (next?.clearcoatNormalScale ?? 1);
  const bumpScaleChanged =
    (prev?.bumpMap != null || next?.bumpMap != null) &&
    (prev?.bumpScale ?? 1) !== (next?.bumpScale ?? 1);
  const lightMapIntensityChanged =
    (prev?.lightMap != null || next?.lightMap != null) &&
    (prev?.lightMapIntensity ?? 1) !== (next?.lightMapIntensity ?? 1);
  const alphaCoverageChanged =
    alphaModeAtlasIndex(prev?.alphaMode) !== alphaModeAtlasIndex(next?.alphaMode) ||
    alphaAtlasUnit(prev?.opacity, 1) !== alphaAtlasUnit(next?.opacity, 1) ||
    alphaAtlasUnit(prev?.alphaCutoff, 0.5) !== alphaAtlasUnit(next?.alphaCutoff, 0.5);
  const specularChanged =
    colorUnit(prev, 'specularColor', 0, 1) !== colorUnit(next, 'specularColor', 0, 1) ||
    colorUnit(prev, 'specularColor', 1, 1) !== colorUnit(next, 'specularColor', 1, 1) ||
    colorUnit(prev, 'specularColor', 2, 1) !== colorUnit(next, 'specularColor', 2, 1) ||
    alphaAtlasUnit(prev?.specularIntensity, 1) !== alphaAtlasUnit(next?.specularIntensity, 1);
  const clearcoatChanged =
    alphaAtlasUnit(prev?.clearcoat, 0) !== alphaAtlasUnit(next?.clearcoat, 0) ||
    alphaAtlasUnit(prev?.clearcoatRoughness, 0) !== alphaAtlasUnit(next?.clearcoatRoughness, 0);
  const sheenChanged =
    alphaAtlasUnit(prev?.sheen, 0) !== alphaAtlasUnit(next?.sheen, 0) ||
    alphaAtlasUnit(prev?.sheenRoughness, 0) !== alphaAtlasUnit(next?.sheenRoughness, 0) ||
    colorUnit(prev, 'sheenColor', 0, 0) !== colorUnit(next, 'sheenColor', 0, 0) ||
    colorUnit(prev, 'sheenColor', 1, 0) !== colorUnit(next, 'sheenColor', 1, 0) ||
    colorUnit(prev, 'sheenColor', 2, 0) !== colorUnit(next, 'sheenColor', 2, 0);
  const anisotropyChanged =
    alphaAtlasUnit(prev?.anisotropy, 0) !== alphaAtlasUnit(next?.anisotropy, 0) ||
    (Number.isFinite(prev?.anisotropyRotation) ? prev?.anisotropyRotation ?? 0 : 0) !==
      (Number.isFinite(next?.anisotropyRotation) ? next?.anisotropyRotation ?? 0 : 0);
  const iridescenceChanged =
    alphaAtlasUnit(prev?.iridescence, 0) !== alphaAtlasUnit(next?.iridescence, 0) ||
    minClampedUnit(prev?.iridescenceIor, 1, 1.3) !== minClampedUnit(next?.iridescenceIor, 1, 1.3) ||
    iridescenceThicknessBound(prev, 0, 100) !== iridescenceThicknessBound(next, 0, 100) ||
    iridescenceThicknessBound(prev, 1, 400) !== iridescenceThicknessBound(next, 1, 400);
  const envMapIntensityChanged =
    nonNegativeScalar(prev?.envMapIntensity, 1) !== nonNegativeScalar(next?.envMapIntensity, 1);
  return normalScaleChanged ||
    clearcoatNormalScaleChanged ||
    bumpScaleChanged ||
    lightMapIntensityChanged ||
    alphaCoverageChanged ||
    specularChanged ||
    clearcoatChanged ||
    sheenChanged ||
    anisotropyChanged ||
    iridescenceChanged ||
    envMapIntensityChanged;
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
  const materialPatchValue = patch.material;
  const primIndex = ctx.lastScene.primitives.findIndex((p) => String(p.id) === id);
  const prevPrim = primIndex >= 0 ? ctx.lastScene.primitives[primIndex] : undefined;
  const prevMaterial =
    prevPrim && 'material' in prevPrim ? prevPrim.material : undefined;
  const nextMaterial: MaterialSpec =
    prevMaterial != null
      ? mergeMaterialPatch(prevMaterial, materialPatchValue)
      : materialPatchValue;
  const unconsumedMaterialFields = collectUnconsumedMaterialFieldsForMaterial(
    nextMaterial as unknown as Record<string, unknown>,
  );
  ctx.warnUnconsumedMaterialFields?.(
    unconsumedMaterialFields,
    unconsumedMaterialFields.length > 0 ? [{ primitiveId: id, fields: unconsumedMaterialFields }] : [],
  );
  const range = bvh.meshVertexRanges.find((r) => r.name === id);
  if (range == null || range.triCount === 0) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): no triangle range for material patch.`,
    );
  }

  if (ctx.coreSceneSuppliesMeshes !== true || bvh.coreMaterials.length === 0) {
    throw new Error(
      `HybridEngine.updatePrimitive("${id}"): material patch requires core scene material slots.`,
    );
  }

  const atlasNeedsRefresh = materialAtlasPatchRequiresFullRebuild(prevMaterial, nextMaterial);
  const updatedScene = applyPrimitivePatchToScene(ctx.lastScene, id, patch);
  const updatedRenderScene = applyPrimitivePatchToScene(ctx.renderScene, id, patch);

  const triMaterialIds = new Uint32Array(bvh.triangleMaterialIds.cpuData);
  const matIds = new Set<number>();
  for (let t = range.triStart; t < range.triStart + range.triCount; t++) {
    matIds.add(triMaterialIds[t]!);
  }

  const triStart = range.triStart;
  const triEnd = range.triStart + range.triCount;
  const totalTris = triMaterialIds.length;
  let slotIsShared = false;
  for (let t = 0; t < totalTris && !slotIsShared; t++) {
    if (t >= triStart && t < triEnd) continue;
    if (matIds.has(triMaterialIds[t]!)) slotIsShared = true;
  }

  let materialIdsForPacking = triMaterialIds;
  let updatedTriangleMaterialIds = bvh.triangleMaterialIds;
  const updatedCoreMaterials = [...bvh.coreMaterials] as MaterialSpec[];
  if (slotIsShared) {
    const splitSlotByOriginalSlot = new Map<number, number>();
    for (const matId of matIds) {
      const splitSlot = updatedCoreMaterials.length;
      updatedCoreMaterials.push(nextMaterial);
      splitSlotByOriginalSlot.set(matId, splitSlot);
    }
    const splitTriMaterialIds = new Uint32Array(triMaterialIds);
    for (let t = triStart; t < triEnd; t += 1) {
      const splitSlot = splitSlotByOriginalSlot.get(triMaterialIds[t]!);
      if (splitSlot != null) splitTriMaterialIds[t] = splitSlot;
    }
    materialIdsForPacking = splitTriMaterialIds;
    const splitBytes = splitTriMaterialIds.buffer.slice(
      splitTriMaterialIds.byteOffset,
      splitTriMaterialIds.byteOffset + splitTriMaterialIds.byteLength,
    );
    updatedTriangleMaterialIds = {
      cpuData: splitBytes,
      byteLength: splitBytes.byteLength,
      count: splitTriMaterialIds.length,
    };
  } else {
    for (const matId of matIds) {
      if (matId < updatedCoreMaterials.length) updatedCoreMaterials[matId] = nextMaterial;
    }
  }

  const fullIndex = packBVHIndexWFromCore(
    bvh.bvhIndicesStride3,
    materialIdsForPacking,
    updatedCoreMaterials,
    bvh.bvhBeerColors.count,
  );
  const fullBeer = packBVHBeerColorsFromCore(
    materialIdsForPacking,
    updatedCoreMaterials,
    bvh.bvhBeerColors.count,
  );
  // B1 — repack the per-tri roughness+metalness lane for the edited materials.
  const fullRoughMetal = packBVHRoughMetalFromCore(
    materialIdsForPacking,
    updatedCoreMaterials,
    bvh.bvhBeerColors.count,
  );
  const materialTextureAtlas = atlasNeedsRefresh
    ? packMaterialTextureAtlas(
        updatedCoreMaterials,
        materialIdsForPacking,
        bvh.bvhBeerColors.count,
        bvh.materialTextureAtlas.triangleUvs,
      )
    : bvh.materialTextureAtlas;
  const indexBytes = fullIndex.buffer.slice(
    fullIndex.byteOffset,
    fullIndex.byteOffset + fullIndex.byteLength,
  );
  const beerBytes = fullBeer.buffer.slice(
    fullBeer.byteOffset,
    fullBeer.byteOffset + fullBeer.byteLength,
  );
  const roughMetalBytes = fullRoughMetal.buffer.slice(
    fullRoughMetal.byteOffset,
    fullRoughMetal.byteOffset + fullRoughMetal.byteLength,
  );
  const updatedBvhIndex = {
    cpuData: indexBytes,
    byteLength: indexBytes.byteLength,
    count: bvh.bvhIndex.count,
  };
  const updatedBeerColors = {
    cpuData: beerBytes,
    byteLength: beerBytes.byteLength,
    count: bvh.bvhBeerColors.count,
  };
  const updatedRoughMetal = {
    cpuData: roughMetalBytes,
    byteLength: roughMetalBytes.byteLength,
    count: bvh.bvhRoughMetal.count,
  };
  const fullEmissive = packBVHEmissiveLeFromCore(
    materialIdsForPacking,
    updatedCoreMaterials,
    bvh.bvhBeerColors.count,
  );

  // Camera-visible emitters — repack the FULL per-tri emissive Le from the
  // now-updated materials (buildMaterials was patched above) so an emissive edit
  // is reflected; the slice path re-uploads the whole emissive texture wholesale.
  const updatedEmissiveLe = {
    cpuData: fullEmissive.buffer.slice(
      fullEmissive.byteOffset,
      fullEmissive.byteOffset + fullEmissive.byteLength,
    ),
    byteLength: fullEmissive.byteLength,
    count: bvh.bvhBeerColors.count,
  };

  // Decide whether to upload a slice or the full bvhIndex buffer.
  //
  // `packBVHIndexWFromCore` already re-packed all triangles sharing the
  // edited material slot(s) into the full CPU `indexView`. The GPU upload
  // must cover ALL triangles whose bvhIndex.w changed — not just this
  // primitive's range. If the edited material slot is shared by triangles
  // outside [triStart, triStart+triCount), uploading only the primitive's
  // slice would leave the GPU with a stale bvhIndex.w for those triangles
  // until the next full rebuild.
  //
  // Strategy:
  //  - Fast path (exclusive slot): the edited slot(s) are used only by this
  //    primitive → slice upload stays cheap.
  //  - Slow path (shared slot): any edited slot is used by at least one
  //    triangle outside the primitive's range → upload the whole bvhIndex.
  //
  // Detection: scan triMaterialIds once; if any triangle outside the
  // primitive's range carries one of `matIds`, the slot is shared.
  const indexSlice = slotIsShared
    ? // Shared slot: upload the entire bvhIndex so all affected triangles
      // (including a merged-mode slot split) get the updated bvhIndex.w.
      { byteOffset: 0, data: updatedBvhIndex.cpuData.slice(0) }
    : // Exclusive slot: only this primitive's triangles were affected; the
      // slice upload is correct and avoids transferring the whole buffer.
      (() => {
        const indexByteOffset = triStart * 16;
        return {
          byteOffset: indexByteOffset,
          data: updatedBvhIndex.cpuData.slice(indexByteOffset, indexByteOffset + range.triCount * 16),
        };
      })();

  ctx.pipeline.refreshBvhMaterialSlice(
    indexSlice,
    // WS1 — beer is a texture: re-upload the full beer data (a contiguous tri
    // slice is not a rectangular texture region unless it spans full rows).
    { data: updatedBeerColors.cpuData, triCount: updatedBeerColors.count },
    {
      data: fullEmissive.buffer.slice(
        fullEmissive.byteOffset,
        fullEmissive.byteOffset + fullEmissive.byteLength,
      ),
      triCount: bvh.bvhBeerColors.count,
    },
    // B1 — re-upload the whole roughness+metalness texture wholesale (same
    // wholesale rationale as beer/emissive).
    { data: updatedRoughMetal.cpuData, triCount: updatedRoughMetal.count },
  );
  if (atlasNeedsRefresh) {
    ctx.pipeline.refreshMaterialTextureAtlas(materialTextureAtlas);
  }

  let outBvh: SceneBVHBuffers = {
    ...bvh,
    bvhEmissiveLe: updatedEmissiveLe,
    bvhIndex: updatedBvhIndex,
    bvhBeerColors: updatedBeerColors,
    bvhRoughMetal: updatedRoughMetal,
    triangleMaterialIds: updatedTriangleMaterialIds,
    materialTextureAtlas,
    coreMaterials: updatedCoreMaterials,
  };
  const emitterAffectingMaterialChanged =
    atlasNeedsRefresh || materialRadianceOrVisibilityChanged(prevMaterial, nextMaterial);
  if (atlasNeedsRefresh || materialAffectsDdgiProbeCache(prevMaterial, nextMaterial)) {
    if (!ctx.deferSubsystemSideEffects) ctx.ddgi.invalidateProbeCache();
  }
  if (emitterAffectingMaterialChanged) {
    const emitterOptions = {
      primaryLightDir: {
        x: ctx.primaryLightDir[0],
        y: ctx.primaryLightDir[1],
        z: ctx.primaryLightDir[2],
      },
      primaryLightIntensity: ctx.primaryLightIntensity,
      packSourceTriIndex: true,
      ...(bvh.bvhMode === 'tlas'
        ? { tlasPrimitiveBindings: bvh.primitiveTlasBindings }
        : {}),
      ...(ctx.onWarning !== undefined
        ? { onWarning: ctx.onWarning, warningPhase: 'mutation' as const, warningMethod: 'updatePrimitive' }
        : {}),
    };
    const emitterSlice = rebuildEmitterBuffersFromCoreScene(updatedRenderScene, emitterOptions);
    outBvh = {
      ...outBvh,
      emitters: emitterSlice.emitters,
      emitterCdf: emitterSlice.emitterCdf,
      emitterAlias: emitterSlice.emitterAlias,
      emitterCount: emitterSlice.emitterCount,
      totalEmissivePower: emitterSlice.totalEmissivePower,
      lightTree: emitterSlice.lightTree,
      lightTreeNodeCount: emitterSlice.lightTreeNodeCount,
      lightTreeEnabled: emitterSlice.lightTreeEnabled,
    };
    ctx.pipeline.updateEmitters(outBvh);
  }

  ctx.pipeline.requestAccumReset();

  // Material-only edits re-pack material/emissive slices in place — the BVH
  // geometry that DDGI + RC index off is unchanged, so geometry propagation is
  // skipped. DDGI still needs a fresh RestirBvhSnapshot because probe rays read
  // coreMaterials/materialTextureAtlas from that snapshot.
  return {
    bvhBuffers: outBvh,
    updatedScene,
    updatedRenderScene,
    applySubsystems: false,
    refreshRcMaterials: true,
    refreshDdgiMaterialSnapshot: true,
  };
}
